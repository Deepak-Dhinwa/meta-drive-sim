import os
import sys
import gc
import json
import importlib.util
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Config
# ============================================================
BC_CHECKPOINT = "checkpoints_v2/best_model.pth"

TOTAL_STEPS = 50_000          # short diagnostic run first
SEED = 42

LOG_FREQ = 5_000
ENV_RESTART_EP = 20

RL_LOG_DIR = "rl_logs"
os.makedirs(RL_LOG_DIR, exist_ok=True)


# ============================================================
# Utility
# ============================================================
def tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None


# ============================================================
# Environment
# ============================================================
class MetaDriveGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self._env = None
        self._ep_count = 0

        self._printed_info_keys = False
        self._printed_obs = False

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(259,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def _make_env(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            gc.collect()

        from metadrive import MetaDriveEnv

        config = dict(
            use_render=False,
            image_observation=False,
            num_scenarios=500,
            start_seed=5000,
            traffic_density=0.0,

            # Diagnostic: enforce episode cutoff
            max_step_per_agent=4000,
            truncate_as_terminate=True,
        )

        print("\n[Env Config]")
        for k, v in config.items():
            print(f"  {k}: {v}")

        self._env = MetaDriveEnv(config)

    def reset(self, seed=None, options=None):
        if self._env is None or self._ep_count % ENV_RESTART_EP == 0:
            self._make_env()

        self._ep_count += 1

        obs, info = self._env.reset()

        obs = obs.astype(np.float32)

        if not self._printed_obs:
            print("\n[Observation Debug]")
            print("obs.shape =", obs.shape)
            print("First 40 values:")
            print(np.round(obs[:40], 4))
            print("\nFirst 19 values (candidate BC sensor input):")
            print(np.round(obs[:19], 4))
            print()

            self._printed_obs = True

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self._env.step(action)
        obs = obs.astype(np.float32)

        if not self._printed_info_keys:
            print("\n[Info Keys]")
            print(sorted(info.keys()))
            print()

            interesting = [
                "velocity",
                "lateral_dist",
                "heading_error",
                "lane_offset",
                "dist_to_left_side",
                "dist_to_right_side",
                "out_of_road",
                "arrive_dest",
                "crash",
            ]

            print("[Interesting Info Values]")
            for k in interesting:
                if k in info:
                    print(f"  {k}: {info[k]}")
            print()

            self._printed_info_keys = True

        # ======================================================
        # Reward shaping
        # ======================================================
        reward = 0.0

        if "velocity" in info:
            reward += 0.001 * info["velocity"]

        if "lateral_dist" in info:
            reward -= 0.5 * abs(info["lateral_dist"])

        if "heading_error" in info:
            reward -= 0.2 * abs(info["heading_error"])

        if "lane_offset" in info:
            reward -= 0.5 * abs(info["lane_offset"])

        if "dist_to_left_side" in info:
            reward -= 0.05 / max(info["dist_to_left_side"], 0.1)

        if "dist_to_right_side" in info:
            reward -= 0.05 / max(info["dist_to_right_side"], 0.1)

        if info.get("arrive_dest", False):
            reward += 2000.0

        if info.get("out_of_road", False):
            reward -= 300.0

        if info.get("crash", False):
            reward -= 100.0

        if truncated and not info.get("arrive_dest", False):
            reward -= 300.0

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# ============================================================
# Feature extractor
# ============================================================
class BCWarmStartExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, bc_checkpoint=None):
        super().__init__(observation_space, features_dim=128)

        # IMPORTANT DIAGNOSTIC:
        # assume BC sensor vector might already be obs[:19]
        self.sensor_mlp = nn.Sequential(
            nn.Linear(19, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self._printed_forward = False

        if bc_checkpoint is not None and os.path.exists(bc_checkpoint):
            self._load_bc_weights(bc_checkpoint)
        else:
            print("[BCWarmStart] No BC checkpoint found.")

    def _load_bc_weights(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"]

        filtered = {}

        print("\n[BC Checkpoint Debug]")
        print("Available sensor_branch keys:")

        for k in state_dict.keys():
            if k.startswith("sensor_branch.net."):
                print(" ", k)

        for k, v in state_dict.items():
            if not k.startswith("sensor_branch.net."):
                continue

            new_key = k.replace("sensor_branch.net.", "")

            if any(x in new_key for x in [
                "running_mean",
                "running_var",
                "num_batches_tracked"
            ]):
                continue

            filtered[new_key] = v

        result = self.sensor_mlp.load_state_dict(filtered, strict=False)

        print("\n[BCWarmStart] Weight loading result")
        print("Missing keys   :", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)

    def forward(self, obs):
        sensor = obs[:, :19]

        if not self._printed_forward:
            print("\n[Extractor Debug]")
            print("Extractor input shape:", obs.shape)
            print("Using first 19 obs dims as BC sensor input")
            print("Sensor sample:")
            print(sensor[0].detach().cpu().numpy())
            print()

            self._printed_forward = True

        return self.sensor_mlp(sensor)


# ============================================================
# Logging callback
# ============================================================
class RLLogger(BaseCallback):
    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.last_log = 0

        self.ep_rewards = []
        self.ep_success = []
        self.ep_oor = []
        self.ep_crash = []
        self.ep_len = []

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            if done:
                self.ep_rewards.append(info.get("episode_reward", 0.0))
                self.ep_success.append(float(info.get("arrive_dest", False)))
                self.ep_oor.append(float(info.get("out_of_road", False)))
                self.ep_crash.append(float(info.get("crash", False)))
                self.ep_len.append(info.get("episode_length", 0))

        if self.num_timesteps - self.last_log >= self.log_freq and len(self.ep_rewards) > 0:
            n = min(20, len(self.ep_rewards))

            print(
                f"[{self.num_timesteps:>7,}] "
                f"reward={np.mean(self.ep_rewards[-n:]):7.2f} | "
                f"success={100*np.mean(self.ep_success[-n:]):5.1f}% | "
                f"oor={100*np.mean(self.ep_oor[-n:]):5.1f}% | "
                f"crash={100*np.mean(self.ep_crash[-n:]):5.1f}% | "
                f"ep_len={np.mean(self.ep_len[-n:]):.0f}"
            )

            self.last_log = self.num_timesteps

        return True


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PPO Diagnostic Run")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    vec_env = DummyVecEnv([MetaDriveGymEnv])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=BCWarmStartExtractor,
        features_extractor_kwargs=dict(
            bc_checkpoint=BC_CHECKPOINT
        ),
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        ),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=RL_LOG_DIR if tensorboard_available() else None,
    )

    # Freeze first two BC layers
    sensor_layers = model.policy.features_extractor.sensor_mlp

    for name, p in sensor_layers.named_parameters():
        if (
            "0." in name or "1." in name or
            "3." in name or "4." in name
        ):
            p.requires_grad = False
        else:
            p.requires_grad = True

    print("\n[Trainable Parameters]")
    total = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    print("Total params    :", f"{total:,}")
    print("Trainable params:", f"{trainable:,}")
    print("Device          :", model.device)

    print("\nStarting PPO...\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=RLLogger(LOG_FREQ),
        reset_num_timesteps=True,
    )

    vec_env.close()
    print("\nFinished.")