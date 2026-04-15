import torch
import torch.nn as nn
from typing import Tuple


class AttentionMIL(nn.Module):
    """
    Minimal Attention MIL (Ilse et al.) with ResNet backbone as instance encoder.
    - Instance encoder: ResNet (removes fc, outputs [B, D_inst])
    - Attention pooling over instances to bag embedding
    - Bag classifier: MLP -> 1 logit
    """
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, attn_dim: int = 128, bag_hidden: int = 128):
        super().__init__()
        self.encoder, self.inst_dim = self._build_backbone(backbone, pretrained)
        self.attn_V = nn.Linear(self.inst_dim, attn_dim)
        self.attn_U = nn.Linear(self.inst_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.bag_mlp = nn.Sequential(
            nn.Linear(self.inst_dim, bag_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(bag_hidden, 1),
        )

    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Build backbone; if torchvision is unavailable/incompatible, use a simple CNN fallback.
        """
        try:
            from torchvision import models  # lazy import
            fn = getattr(models, backbone_name)
            weights = "IMAGENET1K_V1" if pretrained else None
            net = fn(weights=weights)
            out_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, out_dim
        except Exception:
            # Fallback: simple CNN encoder (no torchvision dependency)
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1,1)),
                    )
                    self.out_dim = 256
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.features(x)
                    return x.flatten(1)
            net = SimpleCNN()
            return net, net.out_dim

    def encode_instances(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W] → [N, D]
        return self.encoder(x)

    def attention_pool(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: [N, D] instances of a bag
        V = torch.tanh(self.attn_V(H))
        U = torch.sigmoid(self.attn_U(H))
        A = self.attn_w(V * U)  # [N, 1]
        A = torch.softmax(A, dim=0)  # normalize across instances
        M = torch.sum(A * H, dim=0)  # [D]
        return M, A.squeeze(1)       # bag embedding, attn weights

    def forward(self, bag_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            bag_images: [N, 3, H, W] instances in a bag
        Returns:
            logit (scalar), bag_embed [D], attn_weights [N]
        """
        H = self.encode_instances(bag_images)  # [N, D]
        M, A = self.attention_pool(H)          # [D], [N]
        logit = self.bag_mlp(M.unsqueeze(0)).squeeze(0).squeeze(0)  # []
        return logit, M, A

