import torch
import torch.nn as nn
from models.visual_branch import VisualBranch
from models.sensor_mlp    import SensorMLP

class FusionModel(nn.Module):
    """
    Inputs:  frame  (B, 3, 128, 128)
             sensor (B, 19)
    Output:  action (B, 2)  →  [steering ∈ (-1,1), acceleration ∈ (-1,1)]

    Architecture:
      VisualBranch  → (B, 512)  ─┐
                                  cat → (B, 640) → FC→FC → (B, 2)
      SensorMLP     → (B, 128)  ─┘
    """
    def __init__(self):
        super().__init__()
        self.visual_branch = VisualBranch()
        self.sensor_branch = SensorMLP(input_dim=19)
        self.fusion_head   = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        # Output activations applied in forward (not in Sequential)
        # to allow loss computation on raw logits if needed

    def forward(self, frame: torch.Tensor,
                sensor: torch.Tensor) -> torch.Tensor:

        v = self.visual_branch(frame)    # (B, 512)
        s = self.sensor_branch(sensor)   # (B, 128)
        x = torch.cat([v, s], dim=1)     # (B, 640)
        x = self.fusion_head(x)          # (B, 2)

        # Bounded outputs
        steering     = torch.tanh(x[:, 0:1])          # (-1, 1)
        acceleration = torch.tanh(x[:, 1:2])          # (-1, 1)
        return torch.cat([steering, acceleration], dim=1)  # (B, 2)

    def unfreeze_backbone(self):
        self.visual_branch.unfreeze_backbone()


if __name__ == "__main__":
    model  = FusionModel()
    frames  = torch.randn(4, 3, 128, 128)
    sensors = torch.randn(4, 19)
    out     = model(frames, sensors)

    assert out.shape == (4, 2),            f"Shape wrong: {out.shape}"
    assert out[:, 0].abs().max() <= 1.0,   "Steering out of (-1,1)"
    assert out[:, 1].abs().max() <= 1.0,   "Acceleration out of (-1,1)"

    print(f"Frame input:   {tuple(frames.shape)}")
    print(f"Sensor input:  {tuple(sensors.shape)}")
    print(f"Action output: {tuple(out.shape)}")
    print(f"Steering range:     [{out[:,0].min():.3f}, {out[:,0].max():.3f}]")
    print(f"Acceleration range: [{out[:,1].min():.3f}, {out[:,1].max():.3f}]")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nParams total:     {total:,}")
    print(f"Params trainable: {trainable:,}")
    print("complete — FusionModel OK")