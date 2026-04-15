# =============================================================================
# models/networks.py
# 多模态级联多任务分类网络模块
#
# 核心设计：
#   1. 根据 --modalities 动态构建对应数量的 Backbone（Late Fusion）
#      或单个 Backbone（Early Fusion），网络结构完全由参数驱动
#   2. 支持 --fusion_type: early / late 两种融合策略
#   3. 支持 --task_mode: single / multi_parallel / multi_cascaded 三种预测模式
#   4. 级联预测头（multi_cascaded）体现临床先验：
#      先预测 LNM + CentralLNM，将其 logit 作为额外特征融合后再预测 LateralLNM
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 一、骨干网络构建工厂
# =============================================================================

def build_backbone(backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    构建骨干网络并返回（网络模块, 输出特征维度）。
    所有骨干网络去掉最终分类层，仅保留特征提取部分，输出 [B, D] 的全局特征向量。

    Returns:
        feature_extractor: 特征提取网络（无分类头）
        out_dim:           输出特征维度
    """
    if backbone_name.startswith('resnet'):
        # ── ResNet 系列
        model_fn = getattr(models, backbone_name)
        weights = 'IMAGENET1K_V1' if pretrained else None
        net = model_fn(weights=weights)
        out_dim = net.fc.in_features  # 获取最后一个全连接层的输入维度

        # 替换最终分类层为全局平均池化后的恒等映射（保留特征）
        net.fc = nn.Identity()
        feature_extractor = net

    elif backbone_name.startswith('efficientnet'):
        # ── EfficientNet 系列（torchvision >= 0.13）
        model_fn = getattr(models, backbone_name)
        weights = 'IMAGENET1K_V1' if pretrained else None
        net = model_fn(weights=weights)
        # EfficientNet 的分类头是 net.classifier，最后一层是 Linear
        out_dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        feature_extractor = net

    else:
        raise ValueError(f'不支持的骨干网络类型: {backbone_name}。'
                         f'支持: resnet18/34/50/101, efficientnet_b0/b2')

    return feature_extractor, out_dim


# =============================================================================
# 二、临床特征 MLP
# =============================================================================

class ClinicalMLP(nn.Module):
    """
    临床特征编码器（多层感知机）。
    将离散+连续临床特征映射到固定维度的嵌入向量，参与后续融合。

    架构：Linear → BN → ReLU → Dropout → ... → Linear
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        """
        Args:
            input_dim:   输入特征维度（由 ClinicalFeatureProcessor.output_dim 决定）
            hidden_dims: 各隐藏层维度列表，如 [128, 64]
            dropout:     Dropout 比率
        """
        super().__init__()
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D_clinical] → [B, hidden_dims[-1]]"""
        return self.mlp(x)


# =============================================================================
# 三、多模态特征融合层
# =============================================================================

class LateFusionLayer(nn.Module):
    """
    Late Fusion 特征融合层。
    将多个模态的特征（以及可选的临床特征）在特征维度拼接，
    之后通过 MLP 压缩到统一的融合特征维度。

    输入特征来源：
    - M 个 Backbone 各自输出 [B, D_img]
    - （可选）ClinicalMLP 输出 [B, D_clin]
    
    输出：[B, fusion_hidden_dim]
    """

    def __init__(
        self,
        n_modalities: int,
        img_feat_dim: int,
        clinical_feat_dim: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        total_input_dim = n_modalities * img_feat_dim + clinical_feat_dim
        self.output_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats: List[torch.Tensor], clinical_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            img_feats:     List of [B, D_img]，每个模态的特征
            clinical_feat: [B, D_clin] 或 None

        Returns:
            fused: [B, hidden_dim]
        """
        parts = img_feats.copy()
        if clinical_feat is not None:
            parts.append(clinical_feat)
        concat = torch.cat(parts, dim=1)  # [B, total_input_dim]
        return self.fusion_mlp(concat)


class EarlyFusionLayer(nn.Module):
    """
    Early Fusion 说明：
    Early Fusion 的核心在于 DataLoader 侧将多模态图像在通道维度拼接，
    网络侧使用单个 Backbone 处理拼接后的图像。
    此类主要处理 Backbone 之后的临床特征融合（若有）。
    """

    def __init__(
        self,
        img_feat_dim: int,
        clinical_feat_dim: int = 0,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        total_input_dim = img_feat_dim + clinical_feat_dim
        self.output_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feat: torch.Tensor, clinical_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [img_feat]
        if clinical_feat is not None:
            parts.append(clinical_feat)
        concat = torch.cat(parts, dim=1)
        return self.fusion_mlp(concat)


# =============================================================================
# 四、分类预测头
# =============================================================================

class ClassificationHead(nn.Module):
    """
    单任务二分类预测头。
    输入：融合特征 → 输出：logit（未经 Sigmoid）
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),  # 输出单个 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D] → logit: [B, 1]"""
        return self.head(x).squeeze(1)  # [B]


class CascadedLateralHead(nn.Module):
    """
    级联 LateralLNM 预测头（multi_cascaded 模式专用）。

    临床先验逻辑：
    - LateralLNM（侧方转移）是 LNM（整体转移）的子集
    - 预测 LateralLNM 时，LNM 和 CentralLNM 的预测结果应作为重要先验信息融入
    - 本头将融合特征 + LNM logit + CentralLNM logit 拼接后再预测 LateralLNM

    架构：[fused_feat | lnm_logit | central_logit] → MLP → lateral_logit
    """

    def __init__(self, input_dim: int, cascade_hidden_dim: int = 64, dropout: float = 0.2):
        """
        Args:
            input_dim:          融合特征维度（不含级联 logit）
            cascade_hidden_dim: 级联预测头隐藏层维度
        """
        super().__init__()
        # 输入 = 融合特征 + 2 个来自前序任务的 logit
        total_input_dim = input_dim + 2

        self.cascade_head = nn.Sequential(
            nn.Linear(total_input_dim, cascade_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(cascade_hidden_dim, 1),
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        lnm_logit: torch.Tensor,
        central_logit: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feat:    [B, D] 多模态融合特征
            lnm_logit:     [B]   LNM 预测 logit（stop-gradient，避免梯度反向影响前序任务）
            central_logit: [B]   CentralLNM 预测 logit

        Returns:
            lateral_logit: [B]
        """
        # stop-gradient：级联 logit 仅作为特征使用，不反传梯度到前序任务头
        # 如希望端到端训练（允许梯度反传），去掉 .detach() 即可
        lnm_feat = lnm_logit.detach().unsqueeze(1)      # [B, 1]
        central_feat = central_logit.detach().unsqueeze(1)  # [B, 1]

        # 拼接：[B, D+2]
        cascade_input = torch.cat([fused_feat, lnm_feat, central_feat], dim=1)
        return self.cascade_head(cascade_input).squeeze(1)  # [B]


# =============================================================================
# 五、主网络模型
# =============================================================================

class MultiModalCascadedNet(nn.Module):
    """
    多模态级联多任务医学影像分类网络。

    支持通过 opt 参数完全控制网络结构：
    - opt.modalities:   动态决定 Backbone 数量
    - opt.fusion_type:  early / late
    - opt.use_clinical: 是否引入临床特征 MLP 分支
    - opt.task_mode:    single / multi_parallel / multi_cascaded

    Forward 输出格式（根据 task_mode 不同）：
    - single:           {'lateral': Tensor[B]}
    - multi_parallel:   {'lnm': Tensor[B], 'central': Tensor[B], 'lateral': Tensor[B]}
    - multi_cascaded:   {'lnm': Tensor[B], 'central': Tensor[B], 'lateral': Tensor[B]}
    """

    # 任务 key 与任务名的映射（保持顺序一致性）
    TASK_KEYS = ['lnm', 'central', 'lateral']

    def __init__(self, opt, clinical_input_dim: int = 0):
        """
        Args:
            opt:                实验配置对象
            clinical_input_dim: 临床特征维度（由 ClinicalFeatureProcessor.output_dim 决定）
                                use_clinical=False 时传 0
        """
        super().__init__()
        self.opt = opt
        self.fusion_type = opt.fusion_type
        self.task_mode = opt.task_mode
        self.modalities = opt.modalities
        self.use_clinical = opt.use_clinical and (clinical_input_dim > 0)

        # 任务名 → 预测字典 key 的映射（需与 MultiTaskLoss 中保持一致）
        self.task_name_to_key = {
            'LNM': 'lnm',
            'CentralLNM': 'central',
            'LateralLNM': 'lateral',
        }

        # single 模式下，只预测 tasks 列表中的最后一个任务
        self.single_task_key = None
        if self.task_mode == 'single' and len(opt.tasks) >= 1:
            single_task_name = opt.tasks[-1]
            self.single_task_key = self.task_name_to_key.get(
                single_task_name,
                single_task_name.lower()
            )

        # ── Step 1: 构建图像骨干网络
        self._build_image_backbones(opt)

        # ── Step 2: 构建临床特征 MLP（可选）
        if self.use_clinical:
            self.clinical_mlp = ClinicalMLP(
                input_dim=clinical_input_dim,
                hidden_dims=opt.clinical_mlp_dims,
                dropout=opt.clinical_mlp_dropout,
            )
            clinical_feat_dim = self.clinical_mlp.output_dim
        else:
            self.clinical_mlp = None
            clinical_feat_dim = 0

        # ── Step 3: 构建特征融合层
        self._build_fusion_layer(opt, clinical_feat_dim)

        # ── Step 4: 构建任务预测头
        self._build_task_heads(opt)

        # 打印网络概况
        self._print_architecture_summary()

    # ─────────────────────────────── 网络构建辅助方法 ─────────────────────────

    def _build_image_backbones(self, opt):
        """根据 fusion_type 和 modalities 动态构建骨干网络"""
        if opt.fusion_type == 'late':
            # Late Fusion：每个模态一个独立的 Backbone
            self.backbones = nn.ModuleDict()
            sample_backbone, self.img_feat_dim = build_backbone(opt.backbone, opt.pretrained)

            for i, mod in enumerate(opt.modalities):
                if i == 0:
                    self.backbones[mod] = sample_backbone
                else:
                    # 每个 Backbone 权重独立（不共享），实现模态专属特征提取
                    backbone_i, _ = build_backbone(opt.backbone, opt.pretrained)
                    self.backbones[mod] = backbone_i

        else:  # early fusion
            # Early Fusion：单个 Backbone，处理通道拼接后的输入
            # 若使用 3 模态，输入通道数 = 3 * 3 = 9（RGB × M）
            n_channels = len(opt.modalities) * 3
            backbone, self.img_feat_dim = build_backbone(opt.backbone, opt.pretrained)

            # 修改第一层卷积以接受多通道输入
            if n_channels != 3:
                first_conv = self._get_first_conv(backbone)
                if first_conv is not None:
                    new_conv = nn.Conv2d(
                        n_channels,
                        first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=first_conv.bias is not None,
                    )
                    # 权重初始化：对预训练权重在通道维度做平均后扩展（迁移学习最佳实践）
                    if opt.pretrained:
                        original_weight = first_conv.weight.data  # [out, 3, k, k]
                        mean_weight = original_weight.mean(dim=1, keepdim=True)  # [out, 1, k, k]
                        new_conv.weight.data = mean_weight.repeat(1, n_channels, 1, 1)
                    self._set_first_conv(backbone, new_conv)

            self.single_backbone = backbone

    def _get_first_conv(self, model: nn.Module) -> Optional[nn.Conv2d]:
        """获取骨干网络第一层卷积（支持 ResNet / EfficientNet）"""
        if hasattr(model, 'conv1'):  # ResNet
            return model.conv1
        elif hasattr(model, 'features') and isinstance(model.features[0], nn.Sequential):
            # EfficientNet
            return model.features[0][0]
        return None

    def _set_first_conv(self, model: nn.Module, new_conv: nn.Conv2d):
        """设置骨干网络第一层卷积"""
        if hasattr(model, 'conv1'):
            model.conv1 = new_conv
        elif hasattr(model, 'features') and isinstance(model.features[0], nn.Sequential):
            model.features[0][0] = new_conv

    def _build_fusion_layer(self, opt, clinical_feat_dim: int):
        """构建融合层，维度由模态数和临床特征维度动态决定"""
        if opt.fusion_type == 'late':
            self.fusion_layer = LateFusionLayer(
                n_modalities=len(opt.modalities),
                img_feat_dim=self.img_feat_dim,
                clinical_feat_dim=clinical_feat_dim,
                hidden_dim=opt.fusion_hidden_dim,
                dropout=opt.fusion_dropout,
            )
        else:
            self.fusion_layer = EarlyFusionLayer(
                img_feat_dim=self.img_feat_dim,
                clinical_feat_dim=clinical_feat_dim,
                hidden_dim=opt.fusion_hidden_dim,
                dropout=opt.fusion_dropout,
            )
        self.fused_dim = self.fusion_layer.output_dim

    def _build_task_heads(self, opt):
        """根据 task_mode 构建对应的任务预测头"""
        mode = opt.task_mode
        hidden = opt.cascade_hidden_dim

        if mode == 'single':
            # 仅预测 tasks 列表中的最后一个任务（默认设计为 LateralLNM，
            # 但也兼容只预测 LNM 或 CentralLNM 的情况）
            self.head_lateral = ClassificationHead(self.fused_dim, hidden)

        elif mode == 'multi_parallel':
            # 三任务平行独立预测（共享融合特征，各有独立预测头）
            self.head_lnm = ClassificationHead(self.fused_dim, hidden)
            self.head_central = ClassificationHead(self.fused_dim, hidden)
            self.head_lateral = ClassificationHead(self.fused_dim, hidden)

        elif mode == 'multi_cascaded':
            # 级联：LNM 和 CentralLNM 先预测，Lateral 利用其结果
            self.head_lnm = ClassificationHead(self.fused_dim, hidden)
            self.head_central = ClassificationHead(self.fused_dim, hidden)
            self.head_lateral = CascadedLateralHead(self.fused_dim, hidden)

        else:
            raise ValueError(f'未知的 task_mode: {mode}')

    def _print_architecture_summary(self):
        """打印网络结构关键参数"""
        print('\n' + '─' * 60)
        print(' 网络结构概览')
        print('─' * 60)
        print(f'  Backbone:        {self.opt.backbone} × {len(self.modalities)}')
        print(f'  Image feat dim:  {self.img_feat_dim}')
        print(f'  Fusion type:     {self.fusion_type}')
        print(f'  Fused feat dim:  {self.fused_dim}')
        print(f'  Clinical MLP:    {self.use_clinical}')
        print(f'  Task mode:       {self.task_mode}')
        print(f'  Modalities:      {self.modalities}')

        # 计算可训练参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total params:    {total_params:,}')
        print(f'  Trainable:       {trainable_params:,}')
        print('─' * 60 + '\n')

    # ─────────────────────────────── 前向传播 ────────────────────────────────

    def extract_image_features(self, images) -> List[torch.Tensor]:
        """
        提取图像特征（根据 fusion_type 选择策略）。

        Args:
            images: Late Fusion 时为 Dict[modality_name, Tensor[B, C, H, W]]
                    Early Fusion 时为 Tensor[B, M*C, H, W]

        Returns:
            若 late: List of [B, D_img]，每个模态一个
            若 early: [Tensor[B, D_img]]（单元素列表）
        """
        if self.fusion_type == 'late':
            feats = []
            for mod in self.modalities:
                img_tensor = images[mod]  # [B, C, H, W]
                feat = self.backbones[mod](img_tensor)  # [B, D_img]
                feats.append(feat)
            return feats
        else:
            # Early Fusion：images 已经是拼接好的 Tensor
            feat = self.single_backbone(images)  # [B, D_img]
            return [feat]

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        前向传播（完整版，包含所有分支）。

        Args:
            batch: DataLoader 返回的字典，包含：
                   - 'images':   图像数据（格式由 fusion_type 决定）
                   - 'clinical': [B, D_clinical] 临床特征
                   - 'labels':   [B, N_tasks] 标签（仅用于损失计算，Forward 不使用）

        Returns:
            预测字典，具体 key 由 task_mode 决定：
            - single:        {'lateral': [B]}
            - multi_parallel / multi_cascaded: {'lnm': [B], 'central': [B], 'lateral': [B]}
        """
        images = batch['images']
        clinical = batch['clinical'] if self.use_clinical else None

        # ── Step 1: 提取图像特征
        img_feats = self.extract_image_features(images)

        # ── Step 2: 提取临床特征（可选）
        if self.use_clinical and self.clinical_mlp is not None and clinical is not None:
            clinical_feat = self.clinical_mlp(clinical)  # [B, D_clin]
        else:
            clinical_feat = None

        # ── Step 3: 多模态特征融合
        if self.fusion_type == 'late':
            fused = self.fusion_layer(img_feats, clinical_feat)
        else:
            fused = self.fusion_layer(img_feats[0], clinical_feat)
        # fused: [B, fused_dim]

        # ── Step 4: 根据 task_mode 执行不同的预测逻辑
        return self._predict(fused)

    def forward_with_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        扩展版前向传播：
        在返回任务 logit 的同时，额外返回中间特征，便于可解释性分析（SHAP 等）。

        Returns:
            {
              'logits': {'lnm': [B], 'central': [B], 'lateral': [B]} 或 {'lateral': [B]}（single 模式）
              'probs':  同上，但经过 Sigmoid 的概率
              'fused':  [B, fused_dim] 多模态 + 临床融合后的特征
              'clinical_feat': [B, D_clin] 临床 MLP 输出（use_clinical=False 时为 None）
            }
        """
        images = batch['images']
        clinical = batch['clinical'] if self.use_clinical else None

        # ── Step 1: 提取图像特征
        img_feats = self.extract_image_features(images)

        # ── Step 2: 提取临床特征（可选）
        if self.use_clinical and self.clinical_mlp is not None and clinical is not None:
            clinical_feat = self.clinical_mlp(clinical)  # [B, D_clin]
        else:
            clinical_feat = None

        # ── Step 3: 多模态特征融合
        if self.fusion_type == 'late':
            fused = self.fusion_layer(img_feats, clinical_feat)
        else:
            fused = self.fusion_layer(img_feats[0], clinical_feat)

        logits = self._predict(fused)
        probs = {k: torch.sigmoid(v) for k, v in logits.items()}

        return {
            'logits': logits,
            'probs': probs,
            'fused': fused,
            'clinical_feat': clinical_feat,
        }

    def _predict(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据 task_mode 分发预测逻辑"""
        mode = self.task_mode

        if mode == 'single':
            # ── 仅预测一个任务（由 single_task_key 决定）
            key = self.single_task_key or 'lateral'
            return {key: self.head_lateral(fused)}

        elif mode == 'multi_parallel':
            # ── 三任务平行预测（消融实验：去掉级联依赖，但保留多任务学习）
            lnm_logit = self.head_lnm(fused)
            central_logit = self.head_central(fused)
            lateral_logit = self.head_lateral(fused)
            return {
                'lnm': lnm_logit,
                'central': central_logit,
                'lateral': lateral_logit,
            }

        elif mode == 'multi_cascaded':
            # ── 级联预测（完整模型）
            # 第一阶段：预测 LNM 和 CentralLNM
            lnm_logit = self.head_lnm(fused)         # [B]
            central_logit = self.head_central(fused)   # [B]

            # 第二阶段：利用第一阶段结果级联预测 LateralLNM
            lateral_logit = self.head_lateral(
                fused_feat=fused,
                lnm_logit=lnm_logit,
                central_logit=central_logit,
            )
            return {
                'lnm': lnm_logit,
                'central': central_logit,
                'lateral': lateral_logit,
            }

        else:
            raise ValueError(f'未知的 task_mode: {mode}')

    # ─────────────────────────────── 工具方法 ────────────────────────────────

    def freeze_backbones(self):
        """冻结所有骨干网络参数（用于训练初期热身阶段）"""
        if self.fusion_type == 'late':
            for backbone in self.backbones.values():
                for param in backbone.parameters():
                    param.requires_grad = False
        else:
            for param in self.single_backbone.parameters():
                param.requires_grad = False
        print('[Network] 骨干网络参数已冻结')

    def unfreeze_backbones(self):
        """解冻所有骨干网络参数"""
        if self.fusion_type == 'late':
            for backbone in self.backbones.values():
                for param in backbone.parameters():
                    param.requires_grad = True
        else:
            for param in self.single_backbone.parameters():
                param.requires_grad = True
        print('[Network] 骨干网络参数已解冻')

    def get_parameter_groups(self, opt) -> List[Dict]:
        """
        返回差分学习率参数组列表，用于优化器构建。
        骨干网络使用较小学习率（opt.lr_backbone），其余层使用 opt.lr。

        Returns:
            List[Dict]，可直接传给 optimizer 的 param_groups
        """
        backbone_params = []
        other_params = []

        if self.fusion_type == 'late':
            backbone_modules = list(self.backbones.values())
        else:
            backbone_modules = [self.single_backbone]

        backbone_param_ids = set()
        for m in backbone_modules:
            for p in m.parameters():
                backbone_param_ids.add(id(p))
                if p.requires_grad:
                    backbone_params.append(p)

        for name, p in self.named_parameters():
            if id(p) not in backbone_param_ids and p.requires_grad:
                other_params.append(p)

        return [
            {'params': backbone_params, 'lr': opt.lr_backbone, 'name': 'backbone'},
            {'params': other_params, 'lr': opt.lr, 'name': 'head'},
        ]


# =============================================================================
# 六、网络构建入口函数
# =============================================================================

def build_model(opt, clinical_input_dim: int = 0) -> MultiModalCascadedNet:
    """
    网络构建入口函数，供 train.py 和 test.py 调用。

    Args:
        opt:                Options.parse() 返回的配置对象
        clinical_input_dim: 临床特征维度（use_clinical=False 时传 0）

    Returns:
        完整构建的 MultiModalCascadedNet 模型
    """
    model = MultiModalCascadedNet(opt, clinical_input_dim=clinical_input_dim)
    return model


# =============================================================================
# 模块自测
# =============================================================================
if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from options import Options

    # ── 测试1: 全三模态 + 临床特征 + 级联多任务
    print('=== Test 1: 三模态 + 临床 + 级联 ===')
    opt = Options().parse([
        '--modalities', 'B', 'VUSE', 'VCEUS',
        '--fusion_type', 'late',
        '--task_mode', 'multi_cascaded',
        '--use_clinical', 'true',
        '--backbone', 'resnet50',
        '--exp_name', 'test_run',
    ])
    model = build_model(opt, clinical_input_dim=32)
    # 构造假输入
    B = 4
    dummy_batch = {
        'images': {mod: torch.randn(B, 3, 224, 224) for mod in opt.modalities},
        'clinical': torch.randn(B, 32),
        'labels': torch.zeros(B, 3),
    }
    out = model(dummy_batch)
    for k, v in out.items():
        print(f'  {k}: {v.shape}')  # 期望 [4]

    # ── 测试2: 单模态 + 无临床 + 单任务
    print('\n=== Test 2: 单模态 + 无临床 + 单任务 ===')
    opt2 = Options().parse([
        '--modalities', 'B',
        '--fusion_type', 'late',
        '--task_mode', 'single',
        '--use_clinical', 'false',
        '--backbone', 'resnet50',
        '--tasks', 'LateralLNM',
        '--task_loss_weights', '1.0',
        '--exp_name', 'test_ablation',
    ])
    model2 = build_model(opt2, clinical_input_dim=0)
    dummy_batch2 = {
        'images': {'B': torch.randn(B, 3, 224, 224)},
        'clinical': torch.zeros(B, 1),
        'labels': torch.zeros(B, 1),
    }
    out2 = model2(dummy_batch2)
    for k, v in out2.items():
        print(f'  {k}: {v.shape}')  # 期望 [4]
