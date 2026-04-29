import torch
import torch.nn as nn

class SensorMLP(nn.Module):
    """
    Input:  (B, 19)  — normalized sensor state vector
    Output: (B, 128) — sensor feature embedding
    """
    def __init__(self, input_dim: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 128)


if __name__ == "__main__":
    import os; os.makedirs("models", exist_ok=True)
    model = SensorMLP(input_dim=19)
    x     = torch.randn(8, 19)
    out   = model(x)
    assert out.shape == (8, 128), f"Expected (8,128), got {out.shape}"
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    print("complete — SensorMLP OK")