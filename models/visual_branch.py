import os
import sys
import json
import gc
import importlib.util
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BC_CHECKPOINT = "checkpoints_v2/best_model.pth"

RL_LOG_DIR = "rl_logs"
RL_CKPT_DIR = "rl_checkpoints"

TOTAL_STEPS = 100_000
SEED = 42

ENV_RESTART_EP = 30
LOG_FREQ = 5_000
SAVE_FREQ = 10_000

os.makedirs(RL_LOG_DIR, exist_ok=True)
os.makedirs(RL_CKPT_DIR, exist_ok=True)


def tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None


# ─────────────────────────────────────────────────────────────
# MetaDrive wrapper
# ─────────────────────────────────────────────────────────────
class MetaDriveGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self._env = None
        self._ep_count = 0
        self._printed_info_keys = False

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(259,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
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

        self._env = MetaDriveEnv(dict(
            use_render=False,
            image_observation=False,
            num_scenarios=500,
            start_seed=5000,
            traffic_density=0.0,
        ))

    def reset(self, seed=None, options=None):
        if self._env is None or self._ep_count % ENV_RESTART_EP == 0:
            self._make_env()

        self._ep_count += 1

        obs, info = self._env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Print info keys once so you can inspect available reward terms
        if not self._printed_info_keys:
            print("\n[DEBUG] Available info keys:")
            print(sorted(info.keys()))
            print()
            self._printed_info_keys = True

        # ---------------------------------------------------------
        # Reward shaping
        # ---------------------------------------------------------
        shaped = info.get("velocity", 0.0) * 0.01

        # Small reward for surviving while staying on-road
        if not info.get("out_of_road", False):
            shaped += 0.05

        # Strong penalty for crashes / leaving road
        if info.get("crash", False):
            shaped -= 20.0

        if info.get("out_of_road", False):
            shaped -= 100.0

        # Large success reward
        if info.get("arrive_dest", False):
            shaped += 150.0

        # ---------------------------------------------------------
        # Continuous lane-centering penalties
        # These only apply if MetaDrive exposes the values
        # ---------------------------------------------------------
        if "lateral_dist" in info:
            shaped -= 0.30 * abs(info["lateral_dist"])

        if "heading_error" in info:
            shaped -= 0.10 * abs(info["heading_error"])

        if "lane_offset" in info:
            shaped -= 0.30 * abs(info["lane_offset"])

        if "dist_to_left_side" in info:
            shaped -= 0.02 / max(info["dist_to_left_side"], 0.1)

        if "dist_to_right_side" in info:
            shaped -= 0.02 / max(info["dist_to_right_side"], 0.1)

        return (
            obs.astype(np.float32),
            shaped,
            terminated,
            truncated,
            info,
        )

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# ─────────────────────────────────────────────────────────────
# BC warm-start feature extractor
# ─────────────────────────────────────────────────────────────
class BCWarmStartExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, bc_checkpoint=None):
        super().__init__(observation_space, features_dim=128)

        # Convert MetaDrive obs -> BC sensor vector
        self.obs_to_sensor = nn.Sequential(
            nn.Linear(259, 64),
            nn.ReLU(),
            nn.Linear(64, 19),
        )

        # BC sensor architecture with LayerNorm instead of BatchNorm
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

        if bc_checkpoint is not None and os.path.exists(bc_checkpoint):
            self._load_bc_weights(bc_checkpoint)
        else:
            print("[BCWarmStart] No BC checkpoint found.")

    def _load_bc_weights(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"]

        filtered = {}

        for k, v in state_dict.items():
            if not k.startswith("sensor_branch.net."):
                continue

            new_key = k.replace("sensor_branch.net.", "")

            # Skip BatchNorm stats
            if any(x in new_key for x in [
                "running_mean",
                "running_var",
                "num_batches_tracked"
            ]):
                continue

            filtered[new_key] = v

        result = self.sensor_mlp.load_state_dict(filtered, strict=False)

        print("[BCWarmStart] Sensor branch loaded")
        print("  Missing keys   :", result.missing_keys)
        print("  Unexpected keys:", result.unexpected_keys)

    def forward(self, obs):
        x = self.obs_to_sensor(obs)
        x = self.sensor_mlp(x)
        return x


# ─────────────────────────────────────────────────────────────
# Logger callback
# ─────────────────────────────────────────────────────────────
class RLLogger(BaseCallback):
    def __init__(self, log_dir, log_freq):
        super().__init__()

        self.log_dir = log_dir
        self.log_freq = log_freq
        self.last_log = 0

        self.log_path = os.path.join(log_dir, "rl_log.json")

        self.ep_rewards = []
        self.ep_successes = []
        self.ep_oors = []
        self.ep_crashes = []
        self.ep_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            if not done:
                continue

            self.ep_rewards.append(info.get("episode_reward", 0.0))
            self.ep_successes.append(float(info.get("arrive_dest", False)))
            self.ep_oors.append(float(info.get("out_of_road", False)))
            self.ep_crashes.append(float(info.get("crash", False)))
            self.ep_lengths.append(info.get("episode_length", 0))

        if self.num_timesteps - self.last_log >= self.log_freq and len(self.ep_rewards) > 0:
            n = min(20, len(self.ep_rewards))

            mean_reward = float(np.mean(self.ep_rewards[-n:]))
            success = float(np.mean(self.ep_successes[-n:]) * 100)
            oor = float(np.mean(self.ep_oors[-n:]) * 100)
            crash = float(np.mean(self.ep_crashes[-n:]) * 100)
            ep_len = float(np.mean(self.ep_lengths[-n:]))

            print(
                f"[{self.num_timesteps:>7,}] "
                f"reward={mean_reward:7.2f} | "
                f"success={success:5.1f}% | "
                f"oor={oor:5.1f}% | "
                f"crash={crash:5.1f}% | "
                f"ep_len={ep_len:.0f}"
            )

            with open(self.log_path, "w") as f:
                json.dump({
                    "timesteps": self.num_timesteps,
                    "reward": mean_reward,
                    "success": success,
                    "oor": oor,
                    "crash": crash,
                    "ep_len": ep_len
                }, f, indent=2)

            self.last_log = self.num_timesteps

        return True


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 — PPO Fine-tuning (IL→RL Transfer)")
    print(f"Target: {TOTAL_STEPS:,} steps")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    vec_env = DummyVecEnv([MetaDriveGymEnv])
    vec_env = VecMonitor(
        vec_env,
        filename=os.path.join(RL_LOG_DIR, "monitor")
    )

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

    tb_dir = RL_LOG_DIR if tensorboard_available() else None

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
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
        tensorboard_log=tb_dir,
    )

    # ---------------------------------------------------------
    # Freeze early BC layers but leave final representation trainable
    # ---------------------------------------------------------
    sensor_layers = model.policy.features_extractor.sensor_mlp

    for name, p in sensor_layers.named_parameters():
        # Freeze first Linear + LayerNorm block
        if "0." in name or "1." in name:
            p.requires_grad = False

        # Freeze second Linear + LayerNorm block
        elif "3." in name or "4." in name:
            p.requires_grad = False

        # Leave final block trainable
        else:
            p.requires_grad = True

    print("[Freeze] Early BC layers frozen, final layer trainable.")
    print(f"Device: {model.device}")

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters()
        if p.requires_grad
    )

    print(f"Policy params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    logger_cb = RLLogger(RL_LOG_DIR, LOG_FREQ)

    ckpt_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=RL_CKPT_DIR,
        name_prefix="ppo_av",
        verbose=1,
    )

    print(f"\n{'Timestep':>10} | {'Reward':>8} | {'Success':>8} | "
          f"{'OOR':>7} | {'Crash':>7} | {'EpLen':>6}")
    print("-" * 60)

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[logger_cb, ckpt_cb],
        reset_num_timesteps=True,
    )

    final_path = os.path.join(RL_CKPT_DIR, "ppo_av_final")
    model.save(final_path)

    vec_env.close()

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Final model: {final_path}.zip")
    print(f"Logs: {os.path.join(RL_LOG_DIR, 'rl_log.json')}")
    print("=" * 60)