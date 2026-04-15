# =============================================================================
# v2_mil_gated/networks.py
# 多模态 Attention-MIL 联合门控融合网络
#
# 架构流程：
#   ① Instance Encoder: 共享 backbone → 每个切面 → h_i ∈ ℝ^D
#   ② MIL Pooling:      Attention (或 Mean) 聚合 → H_img ∈ ℝ^D
#   ③ Clinical Encoder:  MLP → H_clin ∈ ℝ^D
#   ④ Fusion:            Gated (或 Concat) → H_final
#   ⑤ Multi-task Heads:  LNM Head + CentralLNM Head
#
# 通过 model_mode 参数控制使用哪些模块（消融实验）
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False
from typing import Dict, List, Optional, Tuple


# =============================================================================
# ① Instance Encoder（共享权重的 Backbone）
# =============================================================================

class InstanceEncoder(nn.Module):
    """
    将多模态通道拼接后的切面图像 [M*3, H, W] 编码为 D 维隐式向量 h_i。

    处理流程：
        [M*3, H, W] → 修改首层 Conv 的 backbone → [backbone_out_dim]
                    → FC 降维 → [hidden_dim]
    """

    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        n_input_channels: int = 9,  # 3 模态 × 3 RGB
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        def _get_torchvision_weights_arg(name: str, use_pretrained: bool):
            if not use_pretrained:
                return None
            # 兼容常见骨干的默认 ImageNet 权重枚举与字符串接口
            # 在较新 torchvision 中推荐使用 Weights Enum，这里用字符串以保持向后兼容
            return 'IMAGENET1K_V1'

        def _get_out_dim_and_strip_head(net: nn.Module, name: str) -> int:
            # 返回分类头输入维度，并将分类头替换为 Identity，使 forward 返回特征
            if hasattr(net, 'fc') and isinstance(getattr(net, 'fc'), nn.Linear):
                out_dim = net.fc.in_features
                net.fc = nn.Identity()
                return out_dim
            if hasattr(net, 'classifier'):
                cls = getattr(net, 'classifier')
                # classifier 可能是 Sequential，最后一层为 Linear
                if isinstance(cls, nn.Sequential):
                    lin = None
                    for m in reversed(cls):
                        if isinstance(m, nn.Linear):
                            lin = m
                            break
                    if lin is not None:
                        out_dim = lin.in_features
                        setattr(net, 'classifier', nn.Identity())
                        return out_dim
                if isinstance(cls, nn.Linear):
                    out_dim = cls.in_features
                    setattr(net, 'classifier', nn.Identity())
                    return out_dim
            if hasattr(net, 'heads'):
                heads = getattr(net, 'heads')
                if isinstance(heads, nn.Sequential):
                    lin = None
                    for m in reversed(heads):
                        if isinstance(m, nn.Linear):
                            lin = m
                            break
                    if lin is not None:
                        out_dim = lin.in_features
                        setattr(net, 'heads', nn.Identity())
                        return out_dim
                if isinstance(heads, nn.Linear):
                    out_dim = heads.in_features
                    setattr(net, 'heads', nn.Identity())
                    return out_dim
            if hasattr(net, 'head') and isinstance(getattr(net, 'head'), nn.Linear):
                out_dim = net.head.in_features
                net.head = nn.Identity()
                return out_dim
            raise RuntimeError(f'Unsupported backbone head structure for {name}')

        def _replace_first_conv_in_module(mod: nn.Module, n_in: int, use_pretrained: bool) -> bool:
            # 在给定模块下递归寻找第一层输入通道为3的 Conv2d，并替换为 n_in
            for name, m in mod.named_children():
                if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                    new_conv = nn.Conv2d(
                        n_in, m.out_channels,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        dilation=m.dilation,
                        groups=m.groups,
                        bias=(m.bias is not None),
                        padding_mode=m.padding_mode if hasattr(m, 'padding_mode') else 'zeros',
                    )
                    if use_pretrained and m.weight is not None:
                        with torch.no_grad():
                            mean_w = m.weight.data.mean(dim=1, keepdim=True)
                            new_conv.weight.data = mean_w.repeat(1, n_in, 1, 1)
                            if m.bias is not None and new_conv.bias is not None:
                                new_conv.bias.data = m.bias.data.clone()
                    setattr(mod, name, new_conv)
                    return True
                # 递归进入子模块
                if _replace_first_conv_in_module(m, n_in, use_pretrained):
                    return True
            return False

        # ── 构建 backbone ──
        net = None
        # 对部分骨干优先使用 timm，避免 torchvision 权重下载/API 差异
        _prefer_timm = backbone_name in {
            'efficientnet_b0','efficientnet_b2',
            'convnext_tiny','vit_b_16','swin_t',
            'mobilevit_s','mobilevit_xs',
        }
        if not _prefer_timm and hasattr(models, backbone_name):
            model_fn = getattr(models, backbone_name)
            weights = _get_torchvision_weights_arg(backbone_name, pretrained)
            net = model_fn(weights=weights)
            print(f'[Backbone] torchvision::{backbone_name} pretrained={pretrained}')
        # 退回到 timm
        if net is None:
            if not _HAS_TIMM:
                raise RuntimeError(f'Backbone {backbone_name} not found in torchvision and timm is not installed')
            name_map = {
                'vit_b_16': 'vit_base_patch16_224',
                'swin_t': 'swin_tiny_patch4_window7_224',
            }
            timm_name = name_map.get(backbone_name, backbone_name)
            net = timm.create_model(
                timm_name,
                pretrained=pretrained,
                in_chans=3,  # 将在下方替换为 n_input_channels
                num_classes=0,  # 直接输出特征
            )
            print(f'[Backbone] timm::{timm_name} pretrained={pretrained}')

        # 修改第一层卷积以接受 M*3 通道（包括 ResNet/EffNet/ConvNeXt/ViT/Swin 等）
        if n_input_channels != 3:
            replaced = False
            # 常见命名快速路径
            if hasattr(net, 'conv1') and isinstance(net.conv1, nn.Conv2d) and net.conv1.in_channels == 3:
                old_conv = net.conv1
                new_conv = nn.Conv2d(
                    n_input_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=(old_conv.bias is not None),
                    padding_mode=old_conv.padding_mode if hasattr(old_conv, 'padding_mode') else 'zeros',
                )
                if pretrained and old_conv.weight is not None:
                    with torch.no_grad():
                        mean_w = old_conv.weight.data.mean(dim=1, keepdim=True)
                        new_conv.weight.data = mean_w.repeat(1, n_input_channels, 1, 1)
                        if old_conv.bias is not None and new_conv.bias is not None:
                            new_conv.bias.data = old_conv.bias.data.clone()
                net.conv1 = new_conv
                replaced = True
            # 通用搜索（EfficientNet.features[0][0], ConvNeXt stem, ViT conv_proj, Swin patch_embed.proj 等）
            if not replaced:
                replaced = _replace_first_conv_in_module(net, n_input_channels, pretrained)
            if not replaced:
                raise RuntimeError(f'Failed to replace first Conv2d for backbone={backbone_name}')

        # timm 模型已通过 num_classes=0 去除了分类头；torchvision 需手动去头
        def _pool_backbone_output(feat: torch.Tensor) -> torch.Tensor:
            # 统一将 backbone 输出压成 [B, C]
            if feat.dim() == 4:
                # CNN 常见 NCHW；部分 Transformer/Hybrid 可能是 NHWC
                if feat.size(1) <= 8 and feat.size(-1) > 8:
                    # 近似判定为 NHWC
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            elif feat.dim() == 3:
                # Token 序列 [B, T, C]，做 token 平均池化
                feat = feat.mean(dim=1)
            elif feat.dim() == 2:
                pass
            else:
                feat = feat.reshape(feat.size(0), -1)
            return feat

        try:
            out_dim = _get_out_dim_and_strip_head(net, backbone_name)
        except RuntimeError:
            # timm 大多直接输出特征；若为特征图，输出通道数作为 out_dim
            with torch.no_grad():
                dummy = torch.zeros(1, n_input_channels, 224, 224)
                if hasattr(net, 'forward_features'):
                    ff = net.forward_features(dummy)
                    ff = _pool_backbone_output(ff)
                    out_dim = ff.size(1)
                else:
                    fv = net(dummy)
                    fv = _pool_backbone_output(fv)
                    out_dim = fv.size(1)

        self.backbone = net
        self.backbone_out_dim = out_dim

        # ── 降维 FC ──
        self.proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, M*3, H, W] （N 个实例）
        → [N, hidden_dim]
        """
        # 兼容 torchvision 与 timm：timm 常返回特征图，需要 forward_features + GAP
        if hasattr(self.backbone, 'forward_features'):
            feat = self.backbone.forward_features(x)
            if feat.dim() == 4:
                if feat.size(1) <= 8 and feat.size(-1) > 8:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)
            elif feat.dim() > 2:
                feat = feat.reshape(feat.size(0), -1)
        else:
            feat = self.backbone(x)
            if feat.dim() > 2:
                feat = feat.reshape(feat.size(0), -1)
        return self.proj(feat)


# =============================================================================
# ② MIL 聚合模块
# =============================================================================

class AttentionMILPooling(nn.Module):
    """
    Gated Attention MIL (Ilse et al., 2018) 聚合。

    a_i = softmax( w^T tanh(V h_i) ⊙ σ(U h_i) )
    H_img = Σ a_i h_i
    """

    def __init__(self, input_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn_V = nn.Linear(input_dim, attn_dim)
        self.attn_U = nn.Linear(input_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(
        self,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H:    [B, N, D] instance features
        mask: [B, N]    True = 有效实例

        Returns:
            bag_embed: [B, D]
            attn_weights: [B, N]
        """
        V = torch.tanh(self.attn_V(H))        # [B, N, attn_dim]
        U = torch.sigmoid(self.attn_U(H))     # [B, N, attn_dim]
        A = self.attn_w(V * U).squeeze(-1)     # [B, N]

        # 无效实例 attention 设为 -inf（softmax 后为 0）
        if mask is not None:
            A = A.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(A, dim=1)         # [B, N]

        # 处理全 mask 情况（避免 NaN）
        attn = torch.nan_to_num(attn, nan=0.0)

        bag_embed = torch.bmm(attn.unsqueeze(1), H).squeeze(1)  # [B, D]
        return bag_embed, attn


class MeanMILPooling(nn.Module):
    """简单均值池化 MIL（消融对比用）"""

    def forward(
        self,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H:    [B, N, D]
        mask: [B, N]
        Returns: bag_embed [B, D], dummy_weights [B, N]
        """
        if mask is not None:
            # masked mean
            mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
            n_valid = mask_f.sum(dim=1).clamp(min=1)  # [B, 1]
            bag_embed = (H * mask_f).sum(dim=1) / n_valid  # [B, D]
            dummy_attn = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            bag_embed = H.mean(dim=1)
            N = H.size(1)
            dummy_attn = torch.full((H.size(0), N), 1.0 / N, device=H.device)
        return bag_embed, dummy_attn


# =============================================================================
# ③ Clinical Encoder
# =============================================================================

class ClinicalEncoder(nn.Module):
    """临床特征 MLP → H_clin ∈ ℝ^D"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [B, hidden_dim]


class RadiomicsEncoder(nn.Module):
    """患者级 ROI radiomics → H_rad ∈ ℝ^D"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# =============================================================================
# ④ Fusion 模块
# =============================================================================

class GatedFusion(nn.Module):
    """
    Clinical-First Gated Fusion:
        G = σ(W_gate · H_clin + b_gate)
        H_final = H_clin + G ⊙ H_img
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, H_clin: torch.Tensor, H_img: torch.Tensor) -> torch.Tensor:
        G = torch.sigmoid(self.gate(H_clin))   # [B, D]
        return H_clin + G * H_img               # [B, D]


class ConcatFusion(nn.Module):
    """
    直接拼接融合（消融对比用）:
        H_final = MLP([H_clin; H_img])
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, H_clin: torch.Tensor, H_img: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([H_clin, H_img], dim=1))  # [B, D]


# =============================================================================
# ⑤ Multi-task Classification Heads
# =============================================================================

class DualHead(nn.Module):
    """LNM Head + CentralLNM Head，各输出 1 个 logit"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head_lnm = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.head_central = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: [B, D] → (logit_lnm [B], logit_central [B])"""
        return self.head_lnm(h).squeeze(-1), self.head_central(h).squeeze(-1)


# =============================================================================
# ⑥ 完整模型（支持所有消融模式）
# =============================================================================

class MultiModalMILGatedNet(nn.Module):
    """
    多模态 Attention-MIL + Gated Fusion 网络。

    model_mode 控制行为：
      clinical_only        → 仅 ClinicalEncoder → DualHead
      slice_mean_concat    → InstanceEncoder + 切片均值 → ConcatFusion → DualHead
      mean_mil_gated       → InstanceEncoder + MeanMIL → GatedFusion → DualHead
      attention_mil_concat → InstanceEncoder + AttentionMIL → ConcatFusion → DualHead
      attention_mil_gated  → InstanceEncoder + AttentionMIL → GatedFusion → DualHead
    """

    def __init__(self, opt):
        super().__init__()
        self.model_mode = opt.model_mode
        self.use_radiomics = bool(getattr(opt, 'use_radiomics', False))
        self.radiomics_fusion_mode = getattr(opt, 'radiomics_fusion_mode', 'legacy')
        D = opt.hidden_dim

        # ── 临床编码器（所有模式都需要，除了可以不用时跳过）──
        self.clinical_encoder = ClinicalEncoder(
            input_dim=opt.clinical_input_dim,
            hidden_dim=D,
            dropout=opt.dropout,
        )
        if self.use_radiomics:
            if getattr(opt, 'radiomics_input_dim', 0) <= 0:
                raise ValueError('use_radiomics=True 但 radiomics_input_dim <= 0，请先生成 fold 级 radiomics 特征')
            if self.radiomics_fusion_mode == 'align_proj':
                P = int(getattr(opt, 'rad_proj_dim', 128))
                self.deep_aligner = nn.Sequential(
                    nn.Linear(D, P),
                    nn.BatchNorm1d(P),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=opt.dropout),
                )
                self.rad_aligner = nn.Sequential(
                    nn.Linear(opt.radiomics_input_dim, P),
                    nn.BatchNorm1d(P),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=opt.dropout),
                )
                if getattr(opt, 'rad_align_fusion', 'add') == 'concat':
                    self.aligned_fuser = nn.Sequential(
                        nn.Linear(P * 2, P),
                        nn.BatchNorm1d(P),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=opt.dropout),
                    )
                else:
                    self.aligned_fuser = None
                self.aligned_to_img = nn.Sequential(
                    nn.Linear(P, D),
                    nn.BatchNorm1d(D),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=opt.dropout),
                )
            elif self.radiomics_fusion_mode == 'film':
                fh = int(getattr(opt, 'film_hidden_dim', 128))
                self.film_gamma_scale = float(getattr(opt, 'film_gamma_scale', 0.1))
                self.film_mlp = nn.Sequential(
                    nn.Linear(opt.radiomics_input_dim, fh),
                    nn.BatchNorm1d(fh),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=opt.dropout),
                )
                self.film_gamma = nn.Linear(fh, D)
                self.film_beta = nn.Linear(fh, D)
            else:
                self.radiomics_encoder = RadiomicsEncoder(
                    input_dim=opt.radiomics_input_dim,
                    hidden_dim=D,
                    dropout=opt.dropout,
                )
                self.image_mixer = nn.Sequential(
                    nn.Linear(D * 2, D),
                    nn.BatchNorm1d(D),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=opt.dropout),
                )

        # ── 图像分支（非 clinical_only 时构建）──
        self.need_image = (opt.model_mode != 'clinical_only')
        if self.need_image:
            n_channels = len(opt.modalities) * 3
            self.instance_encoder = InstanceEncoder(
                backbone_name=opt.backbone,
                pretrained=opt.pretrained,
                n_input_channels=n_channels,
                hidden_dim=D,
                dropout=opt.dropout,
            )

            # MIL 聚合
            use_attention = opt.model_mode in (
                'attention_mil_concat', 'attention_mil_gated',
            )
            if use_attention:
                self.mil_pool = AttentionMILPooling(D, opt.attn_dim)
            else:
                self.mil_pool = MeanMILPooling()

            # 融合
            use_gating = opt.model_mode in (
                'mean_mil_gated', 'attention_mil_gated',
            )
            if use_gating:
                self.fusion = GatedFusion(D)
            else:
                self.fusion = ConcatFusion(D, opt.dropout)

        # ── 分类头 ──
        self.dual_head = DualHead(D, hidden_dim=64, dropout=opt.dropout)

        self._print_summary(opt)

    def _print_summary(self, opt):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\n{"─"*60}')
        print(f'  MultiModalMILGatedNet  mode={self.model_mode}')
        print(f'  hidden_dim={opt.hidden_dim}  backbone={opt.backbone}')
        print(f'  use_radiomics={self.use_radiomics}')
        if self.use_radiomics:
            print(f'  radiomics_fusion_mode={self.radiomics_fusion_mode}')
        print(f'  params: total={total:,}  trainable={trainable:,}')
        print(f'{"─"*60}\n')

    def freeze_backbone(self):
        if self.need_image:
            for p in self.instance_encoder.backbone.parameters():
                p.requires_grad = False
            print('[Net] backbone frozen')

    def unfreeze_backbone(self):
        if self.need_image:
            for p in self.instance_encoder.backbone.parameters():
                p.requires_grad = True
            print('[Net] backbone unfrozen')

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        batch keys:
            bag:      [B, N, C, H, W]
            mask:     [B, N]
            clinical: [B, D_clin]

        Returns:
            {
                'logit_lnm': [B],
                'logit_central': [B],
                'attn_weights': [B, N] or None,
            }
        """
        clinical = batch['clinical']           # [B, D_clin]
        H_clin = self.clinical_encoder(clinical)  # [B, D]
        X_rad = batch['radiomics'] if self.use_radiomics else None  # [B, K]

        if not self.need_image:
            # ── clinical_only ──
            logit_lnm, logit_central = self.dual_head(H_clin)
            return {
                'logit_lnm': logit_lnm,
                'logit_central': logit_central,
                'attn_weights': None,
            }

        bag = batch['bag']    # [B, N, C, H, W]
        mask = batch['mask']  # [B, N]
        B, N, C, H, W = bag.shape

        # ── 编码所有实例（reshape → batch encode → reshape back）──
        instances = bag.view(B * N, C, H, W)
        h_all = self.instance_encoder(instances)   # [B*N, D]
        h_all = h_all.view(B, N, -1)              # [B, N, D]

        # ── MIL 聚合 ──
        H_img, attn_weights = self.mil_pool(h_all, mask)  # [B, D], [B, N]
        if X_rad is not None:
            if self.radiomics_fusion_mode == 'align_proj':
                # 方案一：深度/组学先投影到共享空间，再映射回图像语义空间
                Z_deep = self.deep_aligner(H_img)  # [B, P]
                Z_rad = self.rad_aligner(X_rad)    # [B, P]
                if self.aligned_fuser is None:
                    Z_mix = Z_deep + Z_rad
                else:
                    Z_mix = self.aligned_fuser(torch.cat([Z_deep, Z_rad], dim=1))
                H_img = self.aligned_to_img(Z_mix)  # [B, D]
            elif self.radiomics_fusion_mode == 'film':
                # 方案二：FiLM 调制（组学生成 gamma/beta 调制深度特征）
                H_film = self.film_mlp(X_rad)
                gamma = 1.0 + self.film_gamma_scale * torch.tanh(self.film_gamma(H_film))
                beta = self.film_beta(H_film)
                H_img = gamma * H_img + beta
            else:
                H_rad = self.radiomics_encoder(X_rad)  # [B, D]
                H_img = self.image_mixer(torch.cat([H_img, H_rad], dim=1))

        # ── 融合 ──
        H_final = self.fusion(H_clin, H_img)  # [B, D]

        # ── 分类 ──
        logit_lnm, logit_central = self.dual_head(H_final)

        return {
            'logit_lnm': logit_lnm,
            'logit_central': logit_central,
            'attn_weights': attn_weights,
        }

    def get_parameter_groups(self, opt) -> List[Dict]:
        """差分学习率：backbone 用 lr_backbone，其余用 lr"""
        if not self.need_image:
            return [{'params': self.parameters(), 'lr': opt.lr}]

        backbone_ids = set(id(p) for p in self.instance_encoder.backbone.parameters())
        backbone_params = [p for p in self.parameters()
                           if id(p) in backbone_ids and p.requires_grad]
        other_params = [p for p in self.parameters()
                        if id(p) not in backbone_ids and p.requires_grad]
        return [
            {'params': backbone_params, 'lr': opt.lr_backbone, 'name': 'backbone'},
            {'params': other_params, 'lr': opt.lr, 'name': 'head'},
        ]


def build_model(opt) -> MultiModalMILGatedNet:
    return MultiModalMILGatedNet(opt)
