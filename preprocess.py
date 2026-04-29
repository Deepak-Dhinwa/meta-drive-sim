import numpy as np
import pandas as pd
import os
import cv2
from glob import glob

RAW_DIR    = "data_clean/raw"
PROC_DIR   = "data_clean/processed"
FRAME_SIZE = (128, 128)

# ── Parameters derived from EDA ───────────────────────────────────────────────
MIN_EP_STEPS        = 100     # drop episodes shorter than this
IDLE_VEL_THRESHOLD  = 1.0     # drop timesteps where velocity < this
STRAIGHT_THRESHOLD  = 0.05    # |steering| < this = straight
STRAIGHT_KEEP_RATIO = 0.56    # keep 56% of straight timesteps → ~1:1 balance

np.random.seed(42)

def process_episode(ep_dir, out_dir):
    actions = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
    sensors = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))
    frames_dir = os.path.join(ep_dir, "frames")

    # ── Drop short episodes ───────────────────────────────────────
    if len(actions) < MIN_EP_STEPS:
        return 0, "short"

    steering  = actions["steering"].values
    accel     = actions["acceleration"].values
    velocity  = sensors["velocity"].values
    n         = len(actions)

    # ── Build keep mask ───────────────────────────────────────────
    keep_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if velocity[i] < IDLE_VEL_THRESHOLD:
            continue
        if abs(steering[i]) < STRAIGHT_THRESHOLD:
            if np.random.rand() > STRAIGHT_KEEP_RATIO:
                continue
        keep_mask[i] = True

    kept = np.where(keep_mask)[0]
    if len(kept) < 20:
        return 0, "too_short_after_filter"

    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    new_actions = []
    new_sensors = []
    out_idx     = 0

    for old_idx in kept:
        frame_path = os.path.join(frames_dir, f"{old_idx:06d}.npy")
        if not os.path.exists(frame_path):
            continue

        frame = np.load(frame_path)   # (128,128,3) uint8

        # ── Augmentation: brightness jitter ──────────────────────
        if np.random.rand() < 0.4:
            factor = np.random.uniform(0.8, 1.2)
            frame  = np.clip(frame.astype(np.float32) * factor,
                             0, 255).astype(np.uint8)

        # ── Save original ─────────────────────────────────────────
        np.save(os.path.join(out_dir, "frames", f"{out_idx:06d}.npy"), frame)
        new_actions.append({
            "timestep":     out_idx,
            "steering":     float(steering[old_idx]),
            "acceleration": float(accel[old_idx]),
        })
        new_sensors.append({
            "timestep": out_idx,
            **{f"s{j}": sensors[f"s{j}"].values[old_idx] for j in range(19)},
            "velocity": float(velocity[old_idx]),
        })
        out_idx += 1

        # ── Augmentation: horizontal flip ─────────────────────────
        # Only flip turning timesteps — flipping straight adds no new info
        if abs(steering[old_idx]) > STRAIGHT_THRESHOLD and np.random.rand() < 0.7:
            flipped = cv2.flip(frame, 1)
            np.save(os.path.join(out_dir, "frames", f"{out_idx:06d}.npy"), flipped)
            new_actions.append({
                "timestep":     out_idx,
                "steering":     float(-steering[old_idx]),   # negate steering
                "acceleration": float(accel[old_idx]),
            })
            new_sensors.append({
                "timestep": out_idx,
                **{f"s{j}": sensors[f"s{j}"].values[old_idx] for j in range(19)},
                "velocity": float(velocity[old_idx]),
            })
            out_idx += 1

    pd.DataFrame(new_actions).to_csv(
        os.path.join(out_dir, "actions.csv"), index=False)
    pd.DataFrame(new_sensors).to_csv(
        os.path.join(out_dir, "sensors.csv"), index=False)

    return out_idx, "ok"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    ep_dirs = sorted(glob(os.path.join(RAW_DIR, "episode_*")))

    total_out = 0
    skipped_short = 0
    skipped_filter = 0
    processed = 0

    for ep_dir in ep_dirs:
        ep_name = os.path.basename(ep_dir)
        out_dir = os.path.join(PROC_DIR, ep_name)
        n, reason = process_episode(ep_dir, out_dir)

        if reason == "short":
            skipped_short += 1
        elif reason == "too_short_after_filter":
            skipped_filter += 1
            print(f"  ⚠️  Filtered to nothing: {ep_name}")
        else:
            total_out += n
            processed += 1

    print(f"\n── Preprocessing Summary ────────────────────────────────")
    print(f"  Raw episodes          : {len(ep_dirs)}")
    print(f"  Processed             : {processed}")
    print(f"  Skipped (short <{MIN_EP_STEPS}) : {skipped_short}")
    print(f"  Skipped (filter)      : {skipped_filter}")
    print(f"  Total timesteps out   : {total_out:,}  (incl. flip augmentation)")
    print(f"  Output dir            : {PROC_DIR}")

    # ── Verify output ──────────────────────────────────────────────
    print(f"\n── Output Verification ──────────────────────────────────")
    out_eps = sorted(glob(os.path.join(PROC_DIR, "episode_*")))
    steer_all = []
    for ep in out_eps:
        act = pd.read_csv(os.path.join(ep, "actions.csv"))
        steer_all.extend(act["steering"].tolist())
    steer_all = np.array(steer_all)
    straight_pct = (np.abs(steer_all) < STRAIGHT_THRESHOLD).mean() * 100
    turning_pct  = (np.abs(steer_all) >= STRAIGHT_THRESHOLD).mean() * 100
    print(f"  Output episodes       : {len(out_eps)}")
    print(f"  Straight %            : {straight_pct:.1f}%")
    print(f"  Turning  %            : {turning_pct:.1f}%  ← target ~50%")
    print(f"  Ratio                 : {straight_pct/max(turning_pct,1):.2f}:1")
    print(f"\nPreprocessing complete.")

if __name__ == "__main__":
    main()