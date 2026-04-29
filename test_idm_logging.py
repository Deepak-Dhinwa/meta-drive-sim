from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
import numpy as np
import cv2

config = dict(
    use_render        = False,
    image_observation = True,
    norm_pixel        = True,
    num_scenarios     = 1,
    start_seed        = 1000,
    vehicle_config    = dict(image_source="main_camera"),
    sensors           = dict(main_camera=("MainCamera", 128, 128)),
    agent_policy      = IDMPolicy,
)

env = MetaDriveEnv(config)
obs, info = env.reset()

print("Step | steering | acceleration | vel | steer_inbounds | accel_inbounds")
print("-" * 75)

for i in range(30):
    obs, reward, terminated, truncated, info = env.step([0, 0])

    steering     = float(info['steering'])
    acceleration = float(info['acceleration'])
    vel          = float(info['velocity'])

    # Frame sanity
    raw   = obs['image']
    frame = raw[:, :, :, -1]
    frame_resized = (cv2.resize(frame, (128,128), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)

    steer_ok = abs(steering)     <= 1.0
    accel_ok = abs(acceleration) <= 1.0
    frame_ok = frame_resized.dtype == np.uint8 and frame_resized.max() <= 255

    print(f"  {i:02d} | {steering:+.4f}   | {acceleration:+.4f}       | "
          f"{vel:.2f} | {'OK' if steer_ok else 'FAIL'} | {'OK' if accel_ok else 'FAIL'} | "
          f"frame={'OK' if frame_ok else 'FAIL'}")

    if terminated or truncated:
        print("Episode ended early.")
        break

env.close()
print("\nTest complete.")