import numpy as np

# --- Test 1: Import ---
from metadrive import MetaDriveEnv
print("MetaDrive imported successfully")

# --- Test 2: Environment creation ---
config = dict(
    use_render=True,          # Enable visual window
    image_observation=False,  # Sensor-only obs first (we'll enable image later)
    num_scenarios=1,
    start_seed=0,
)

env = MetaDriveEnv(config)
print("Environment created")

# --- Test 3: Reset ---
obs, info = env.reset()
print(f"Reset successful")
print(f"Obs type: {type(obs)}")
print(f"Obs shape: {obs.shape}")
print(f"Obs min/max: {obs.min():.3f} / {obs.max():.3f}")
print(f"Action space: {env.action_space}")
print(f"Obs space: {env.observation_space}")

# --- Test 4: 100-step loop with random actions ---
print("\nRunning 100 random steps...")
for i in range(100):
    action = env.action_space.sample()   # random [steering, throttle/brake]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print(f"100 steps complete")
print(f"Final obs shape: {obs.shape}")
print(f"Sample obs values: {obs[:5]}")  # first 5 sensor dims
print(f"Sample info keys: {list(info.keys())}")

env.close()
print("\nMetaDrive is working.")