import torch
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel

CHECKPOINT = "checkpoints_v2/best_model.pth"
FRAME_SIZE  = (128, 128)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at module level (not on every call) ───────────────────────
def load_model(checkpoint_path: str = CHECKPOINT) -> FusionModel:
    model = FusionModel()
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"   Trained for {ckpt['epoch']} epochs | best val loss: {ckpt['val_loss']:.5f}")
    return model


def preprocess_frame(frame_np: np.ndarray) -> torch.Tensor:
    """
    frame_np: (H, W, 3) float32 [0,1]  — raw from MetaDrive norm_pixel=True
    returns:  (1, 3, 128, 128) float32 tensor on DEVICE
    """
    frame = cv2.resize(frame_np, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame.to(DEVICE)


def preprocess_sensor(sensor_np: np.ndarray) -> torch.Tensor:
    """
    sensor_np: (19,) float32 — already normalized [0,1] by MetaDrive
    returns:   (1, 19) float32 tensor on DEVICE
    """
    sensor = torch.from_numpy(sensor_np.astype(np.float32)).unsqueeze(0)
    return sensor.to(DEVICE)


@torch.no_grad()
def predict(model: FusionModel,
            frame_np: np.ndarray,
            sensor_np: np.ndarray) -> np.ndarray:
    """
    Single-step inference.
    Returns: np.ndarray shape (2,) → [steering, acceleration] both in [-1, 1]
    """
    frame  = preprocess_frame(frame_np)
    sensor = preprocess_sensor(sensor_np)
    action = model(frame, sensor)              # (1, 2)
    return action.squeeze(0).cpu().numpy()     # (2,)


# ── Unit test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_model()

    # Simulate what MetaDrive will send
    dummy_frame  = np.random.rand(900, 1200, 3).astype(np.float32)  # raw env size
    dummy_sensor = np.random.rand(19).astype(np.float32)

    # Warmup pass (first CUDA call is slow due to kernel compilation)
    for _ in range(3):
        _ = predict(model, dummy_frame, dummy_sensor)

    # Timed inference
    import time
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        action = predict(model, dummy_frame, dummy_sensor)
    elapsed = time.perf_counter() - t0

    print(f"\nInference benchmark ({N} calls):")
    print(f"  Avg latency : {elapsed/N*1000:.2f} ms/call")
    print(f"  Throughput  : {N/elapsed:.1f} Hz")
    print(f"  Sample output: steering={action[0]:.4f}  accel={action[1]:.4f}")
    print(f"\ncomplete — inference")