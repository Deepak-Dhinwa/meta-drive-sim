# OOM-fixed
import numpy as np
import cv2
import os
import csv
import time
from metadrive import MetaDriveEnv

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/raw"
FRAME_SIZE      = (128, 128)
NUM_SEEDS       = 50          # more road variety
START_SEED      = 42
ALPHA           = 0.4
FLUSH_EVERY     = 20          # flush CSV writers every N steps

config = dict(
    use_render=True,
    image_observation=True,
    norm_pixel=True,
    num_scenarios=NUM_SEEDS,
    start_seed=START_SEED,
    vehicle_config=dict(image_source="main_camera"),
    sensors=dict(main_camera=("MainCamera", 128, 128)),
    manual_control=True,
)

def get_episode_dir(episode_idx):
    path = os.path.join(DATA_DIR, f"episode_{episode_idx:04d}")
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    return path

def process_frame(raw_image):
    frame = raw_image[:, :, :, -1]                         # latest frame (H,W,3)
    resized = cv2.resize(frame, FRAME_SIZE,
                         interpolation=cv2.INTER_LINEAR)   # (128,128,3) float32
    return (resized * 255).astype(np.uint8)                # ← uint8, 4x smaller

def open_csv_writers(episode_dir):
    sf = open(os.path.join(episode_dir, "sensors.csv"), "w", newline="", buffering=1)
    af = open(os.path.join(episode_dir, "actions.csv"),  "w", newline="", buffering=1)
    sw, aw = csv.writer(sf), csv.writer(af)
    sw.writerow(["timestep"] + [f"s{i}" for i in range(19)] + ["velocity"])
    aw.writerow(["timestep", "steering", "acceleration"])
    return sf, af, sw, aw

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    env = MetaDriveEnv(config)

    episode_idx     = 0
    global_step     = 0
    smoothed_action = np.array([0.0, 0.0])

    print("=" * 60)
    print("DATA COLLECTION — OOM FIXED (uint8 frames)")
    print("  W/S = accel/brake  |  A/D = steer")
    print("  Episode ends on crash / OOR / destination")
    print("  Ctrl+C = quit and save current episode")
    print("=" * 60)

    obs, info    = env.reset()
    episode_dir  = get_episode_dir(episode_idx)
    sf, af, sw, aw = open_csv_writers(episode_dir)
    ep_step      = 0
    ep_start     = time.time()

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step([0, 0])

            raw_action      = np.array(info['action'], dtype=np.float32)
            smoothed_action = ALPHA * raw_action + (1 - ALPHA) * smoothed_action

            # ── Save frame as uint8 ───────────────────────────────
            frame      = process_frame(obs['image'])
            frame_path = os.path.join(episode_dir, "frames", f"{ep_step:06d}.npy")
            np.save(frame_path, frame)

            # ── Save sensors + actions ────────────────────────────
            state = obs['state']
            sw.writerow([ep_step] + state.tolist() + [info['velocity']])
            aw.writerow([ep_step,
                         round(float(smoothed_action[0]), 6),
                         round(float(smoothed_action[1]), 6)])

            # ── Periodic flush to disk ────────────────────────────
            if ep_step % FLUSH_EVERY == 0:
                sf.flush(); af.flush()
                os.fsync(sf.fileno()); os.fsync(af.fileno())

            if ep_step % 50 == 0:
                elapsed = time.time() - ep_start
                print(f"Ep {episode_idx:03d} | Step {ep_step:04d} | "
                      f"vel={info['velocity']:.2f} | "
                      f"smooth={np.round(smoothed_action,2)} | "
                      f"crash={info['crash']} oor={info['out_of_road']} | "
                      f"t={elapsed:.0f}s")

            ep_step     += 1
            global_step += 1

            if terminated or truncated:
                sf.flush(); af.flush()
                sf.close(); af.close()

                status = "CRASH" if info['crash'] else \
                         "OOR"   if info['out_of_road'] else \
                         "DEST"  if info['arrive_dest'] else "MAX"

                print(f"\n[{status}] Ep {episode_idx:03d} → {ep_step} steps | "
                      f"reward={info['episode_reward']:.2f} | "
                      f"total_global={global_step}")

                episode_idx    += 1
                obs, info       = env.reset()
                smoothed_action = np.array([0.0, 0.0])
                episode_dir     = get_episode_dir(episode_idx)
                sf, af, sw, aw  = open_csv_writers(episode_dir)
                ep_step         = 0
                ep_start        = time.time()

    except KeyboardInterrupt:
        sf.flush(); af.flush()
        sf.close(); af.close()
        print(f"\nStopped manually.")
        print(f"Episodes: {episode_idx} | Total steps: {global_step}")
    finally:
        env.close()
        print(f"Data saved to: {DATA_DIR}")

if __name__ == "__main__":
    main()