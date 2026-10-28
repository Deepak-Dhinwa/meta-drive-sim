import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class VisualBranch(nn.Module):
    """
    Input:  (B, 3, 128, 128) RGB frame in [0, 1]
    Output: (B, 512)

    EfficientNet-B0 features, global average pool, then Linear 1280 → 512.
    The backbone starts frozen. Call unfreeze_backbone() after the warm-up epochs.
    """

    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None).features
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Index 0 has no parameters, so the checkpoint keys stay head.1 / head.2.
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    model = VisualBranch()
    frames = torch.randn(4, 3, 128, 128)
    out = model(frames)
    assert out.shape == (4, 512), out.shape
    print(f"Output: {tuple(out.shape)}")
    print("complete — VisualBranch OK")
