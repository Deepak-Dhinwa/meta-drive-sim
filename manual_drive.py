import numpy as np
import cv2
from metadrive import MetaDriveEnv

config = dict(
    use_render=True,
    image_observation=True,
    norm_pixel=True,
    num_scenarios=5,
    start_seed=42,
    vehicle_config=dict(image_source="main_camera"),
    sensors=dict(main_camera=("MainCamera", 128, 128)),
    manual_control=True,          # Enable keyboard control
)

env = MetaDriveEnv(config)
obs, info = env.reset()

FRAME_TARGET = (128, 128)        # We'll resize here regardless of env output

print("Controls: W/S = accel/brake | A/D = steer | R = reset | ESC = quit")
print("=" * 60)
print(f"State dim: {obs['state'].shape[0]}")
raw_frame = obs['image']
print(f"Raw image shape from env: {raw_frame.shape}")
print("=" * 60)

step = 0
try:
    while True:
        obs, reward, terminated, truncated, info = env.step([0, 0])  # env handles keyboard internally

        # ── Frame processing ──────────────────────────────────────
        raw = obs['image']                        # (H, W, C, stack)
        frame = raw[:, :, :, -1]                  # take latest frame → (H, W, 3)
        frame_resized = cv2.resize(
            frame,
            FRAME_TARGET,
            interpolation=cv2.INTER_LINEAR
        )                                         # → (128, 128, 3)

        # ── Sensor state ──────────────────────────────────────────
        state = obs['state']                      # (19,)

        # ── Action actually applied ───────────────────────────────
        action = info['action']                   # [steering, acceleration]

        # ── Print every 10 steps ──────────────────────────────────
        if step % 10 == 0:
            print(f"Step {step:04d} | "
                  f"frame={frame_resized.shape} dtype={frame_resized.dtype} "
                  f"min={frame_resized.min():.2f} max={frame_resized.max():.2f} | "
                  f"state={np.round(state, 3)} | "
                  f"action={np.round(action, 3)} | "
                  f"vel={info['velocity']:.2f} | "
                  f"crash={info['crash']} oor={info['out_of_road']}")

        if terminated or truncated:
            print(f"\n--- Episode ended at step {step} | "
                  f"reward={info['episode_reward']:.2f} | "
                  f"crash={info['crash']} ---\n")
            obs, info = env.reset()

        step += 1

except KeyboardInterrupt:
    print(f"\nStopped at step {step}")
finally:
    env.close()