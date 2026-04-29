import torch
import numpy as np
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel
from inference import load_model

CHECKPOINT   = "checkpoints_v2/best_model.pth"
ONNX_PATH    = "checkpoints_v2/fusion_model.onnx"
DEVICE       = torch.device("cpu")   # export on CPU — ONNX runtime uses CPU

def export(model, onnx_path):
    model.eval()

    # Dummy inputs — shapes must match exactly what inference.py sends
    dummy_frame  = torch.randn(1, 3, 128, 128)   # (B, C, H, W)
    dummy_sensor = torch.randn(1, 19)             # (B, state_dim)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_frame, dummy_sensor),
        onnx_path,
        export_params    = True,
        opset_version    = 17,
        do_constant_folding = True,
        input_names      = ["frame", "sensor"],
        output_names     = ["action"],
        dynamic_axes     = {
            "frame":  {0: "batch_size"},
            "sensor": {0: "batch_size"},
            "action": {0: "batch_size"},
        }
    )
    print(f"Saved: {onnx_path}")

def verify(onnx_path, model):
    import onnxruntime as ort

    # ── Load ONNX session ──────────────────────────────────────────
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    print(f"\nONNX inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} {inp.type}")
    print(f"ONNX outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape} {out.type}")

    # ── Compare PyTorch vs ONNX output ────────────────────────────
    np.random.seed(0)
    frame_np  = np.random.randn(1, 3, 128, 128).astype(np.float32)
    sensor_np = np.random.randn(1, 19).astype(np.float32)

    # PyTorch
    model.eval()
    with torch.no_grad():
        pt_out = model(
            torch.from_numpy(frame_np),
            torch.from_numpy(sensor_np)
        ).numpy()

    # ONNX
    onnx_out = sess.run(
        ["action"],
        {"frame": frame_np, "sensor": sensor_np}
    )[0]

    max_diff = np.abs(pt_out - onnx_out).max()
    print(f"\nOutput comparison:")
    print(f"  PyTorch : {pt_out.round(6)}")
    print(f"  ONNX    : {onnx_out.round(6)}")
    print(f"  Max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-4 else 'MISMATCH'}")

    # ── Benchmark: PyTorch vs ONNX latency ────────────────────────
    N = 200

    # PyTorch CPU
    model.cpu()
    frame_t  = torch.from_numpy(frame_np)
    sensor_t = torch.from_numpy(sensor_np)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(frame_t, sensor_t)   # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N): model(frame_t, sensor_t)
    pt_ms = (time.perf_counter() - t0) / N * 1000

    # ONNX CPU
    for _ in range(10):
        sess.run(["action"], {"frame": frame_np, "sensor": sensor_np})
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(["action"], {"frame": frame_np, "sensor": sensor_np})
    onnx_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\nLatency benchmark ({N} calls, CPU only):")
    print(f"  PyTorch CPU : {pt_ms:.2f} ms/call  ({1000/pt_ms:.1f} Hz)")
    print(f"  ONNX CPU    : {onnx_ms:.2f} ms/call  ({1000/onnx_ms:.1f} Hz)")
    print(f"  Speedup     : {pt_ms/onnx_ms:.2f}x")

    # ── File size ──────────────────────────────────────────────────
    pt_size   = os.path.getsize(CHECKPOINT)   / 1024 / 1024
    onnx_size = os.path.getsize(onnx_path)    / 1024 / 1024
    print(f"\nFile sizes:")
    print(f"  PyTorch .pth : {pt_size:.1f} MB")
    print(f"  ONNX .onnx   : {onnx_size:.1f} MB")

    return max_diff < 1e-4

if __name__ == "__main__":
    # Load model on CPU for export
    model = FusionModel()
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"Model loaded | val_loss={ckpt['val_loss']:.5f}")

    export(model, ONNX_PATH)

    try:
        import onnxruntime
        ok = verify(ONNX_PATH, model)
        print(f"\n{'ONNX export verified.' if ok else 'Verification failed.'}")
    except ImportError:
        print("\nonnxruntime not installed. Installing...")
        os.system("pip install onnxruntime")
        import onnxruntime
        ok = verify(ONNX_PATH, model)
        print(f"\n{'ONNX export verified.' if ok else 'Verification failed.'}")

    print(f"\nONNX model ready: {ONNX_PATH}")