import torch
import numpy as np
import cv2
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel
from inference import load_model, preprocess_frame, preprocess_sensor

CHECKPOINT   = "checkpoints_v2/best_model.pth"
OUT_DIR      = "gradcam_outputs"
N_SAMPLES    = 10    # number of frames to visualize
os.makedirs(OUT_DIR, exist_ok=True)

# ── GradCAM implementation ────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: FusionModel):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Hook onto last conv block of EfficientNet backbone
        # EfficientNet-B0 last block = backbone[-1]
        target_layer = model.visual_branch.backbone[-1]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, frame_tensor: torch.Tensor,
                sensor_tensor: torch.Tensor,
                target_dim: int = 0) -> np.ndarray:
        """
        target_dim: 0=steering, 1=acceleration
        Returns heatmap (H, W) normalized [0,1]
        """
        self.model.zero_grad()
        output = self.model(frame_tensor, sensor_tensor)  # (1, 2)

        # Backprop w.r.t. target output dimension
        output[0, target_dim].backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = torch.relu(cam).squeeze().cpu().numpy()                 # (H, W)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(frame_uint8: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Overlay GradCAM heatmap on original frame."""
    heatmap = cv2.resize(cam, (frame_uint8.shape[1], frame_uint8.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)        # RGB

    frame_rgb = frame_uint8  # already RGB
    overlay   = (alpha * heatmap + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    from glob import glob
    import pandas as pd

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(CHECKPOINT)
    model.train()   # GradCAM needs grads — eval() disables some but hooks still work
                    # We keep BatchNorm in eval mode manually
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or \
           isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    gradcam = GradCAM(model)

    # ── Sample frames from processed dataset ─────────────────────
    ep_dirs  = sorted(glob("data_clean/processed/episode_*"))
    samples  = []   # (frame_np, sensor_np, steering_label)

    for ep_dir in ep_dirs:
        act = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
        sen = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))
        frm = os.path.join(ep_dir, "frames")

        # Pick frames with significant steering (turns) — more interesting for viz
        turn_idx = act.index[act["steering"].abs() > 0.1].tolist()
        if not turn_idx:
            continue

        for idx in turn_idx[:2]:   # max 2 per episode
            timestep   = int(act["timestep"].iloc[idx])
            frame_path = os.path.join(frm, f"{timestep:06d}.npy")
            if not os.path.exists(frame_path):
                continue
            frame_np  = np.load(frame_path)             # (128,128,3) uint8
            sensor_np = sen.iloc[idx][[f"s{j}"
                        for j in range(19)]].values.astype(np.float32)
            steer_lbl = float(act["steering"].iloc[idx])
            samples.append((frame_np, sensor_np, steer_lbl))

        if len(samples) >= N_SAMPLES:
            break

    print(f"Samples collected: {len(samples)}")

    # ── Generate GradCAM for each sample ──────────────────────────
    fig, axes = plt.subplots(len(samples), 4,
                             figsize=(16, 4 * len(samples)))
    fig.suptitle("GradCAM — What the model attends to", fontsize=14)

    for i, (frame_np, sensor_np, steer_lbl) in enumerate(samples):
        # Preprocess
        frame_f32  = frame_np.astype(np.float32) / 255.0   # (128,128,3)
        frame_t    = torch.from_numpy(frame_f32).permute(2,0,1)\
                         .unsqueeze(0).float().to(DEVICE)
        sensor_t   = torch.from_numpy(sensor_np).unsqueeze(0).to(DEVICE)
        frame_t.requires_grad = True

        # Forward + GradCAM for steering
        cam_steer  = gradcam.compute(frame_t, sensor_t, target_dim=0)

        # Forward + GradCAM for acceleration (no grad accumulation — re-run)
        model.zero_grad()
        cam_accel  = gradcam.compute(frame_t, sensor_t, target_dim=1)

        # Predicted action
        with torch.no_grad():
            pred = model(frame_t, sensor_t).squeeze().cpu().numpy()

        # Overlays
        overlay_steer = overlay_heatmap(frame_np, cam_steer)
        overlay_accel = overlay_heatmap(frame_np, cam_accel)

        row = axes[i] if len(samples) > 1 else axes

        row[0].imshow(frame_np)
        row[0].set_title(f"Original | steer_gt={steer_lbl:.3f}")
        row[0].axis("off")

        row[1].imshow(cam_steer, cmap="jet")
        row[1].set_title("Steering CAM (raw)")
        row[1].axis("off")

        row[2].imshow(overlay_steer)
        row[2].set_title(f"Steer overlay | pred={pred[0]:.3f}")
        row[2].axis("off")

        row[3].imshow(overlay_accel)
        row[3].set_title(f"Accel overlay | pred={pred[1]:.3f}")
        row[3].axis("off")

        # Save individual sample
        cv2.imwrite(
            os.path.join(OUT_DIR, f"sample_{i:02d}_steer{steer_lbl:.2f}.png"),
            cv2.cvtColor(overlay_steer, cv2.COLOR_RGB2BGR)
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gradcam_grid.png"), dpi=120)
    plt.close()
    print(f"Saved to {OUT_DIR}/gradcam_grid.png")