from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
import numpy as np

config = dict(
    use_render=False,
    image_observation=False,
    num_scenarios=1,
    start_seed=1000,
    agent_policy=IDMPolicy,
)

env = MetaDriveEnv(config)
obs, info = env.reset()

for i in range(20):
    obs, reward, terminated, truncated, info = env.step([0, 0])
    print(f"step {i:02d} | "
          f"info['action']={np.round(info['action'],3)} | "
          f"info['raw_action']={np.round(info['raw_action'],3)} | "
          f"info['steering']={info['steering']:.3f} | "
          f"info['acceleration']={info['acceleration']:.3f} | "
          f"vel={info['velocity']:.2f}")
    if terminated or truncated:
        break

env.close()