import numpy as np
from metadrive import MetaDriveEnv

config = dict(
    use_render        = False,
    image_observation = False,   # RL uses sensor-only obs — faster
    num_scenarios     = 100,
    start_seed        = 5000,    # fresh seeds away from all previous data
    traffic_density   = 0.0,     # no traffic for Phase 2 start
)

env = MetaDriveEnv(config)
obs, info = env.reset()

print(f"Obs space : {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Obs shape : {obs.shape}")
print(f"Obs sample: {obs[:5].round(3)}")

# Run 200 random steps
total_reward = 0
for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()

print(f"200 random steps done | total_reward={total_reward:.2f}")
print(f"Info keys: {list(info.keys())}")
env.close()
print("RL env working")