import os, sys, json, gc
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Config ────────────────────────────────────────────────────────────────────
BC_CHECKPOINT  = "checkpoints_v2/best_model.pth"
RL_LOG_DIR     = "rl_logs"
RL_CKPT_DIR    = "rl_checkpoints"
TOTAL_STEPS    = 500_000
SEED           = 42
LOG_FREQ       = 5_000
SAVE_FREQ      = 50_000
ENV_RESTART_EP = 20      # restart env every N episodes → OOM prevention

os.makedirs(RL_LOG_DIR,  exist_ok=True)
os.makedirs(RL_CKPT_DIR, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
class MetaDriveGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self._env      = None
        self._ep_count = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _make_env(self):
        if self._env is not None:
            try:   self._env.close()
            except: pass
            self._env = None
            gc.collect()

        from metadrive import MetaDriveEnv
        self._env = MetaDriveEnv(dict(
            use_render          = False,
            image_observation   = False,
            num_scenarios       = 500,
            start_seed          = 5000,
            traffic_density     = 0.0,
            horizon             = 4000,
            truncate_as_terminate = True,
        ))

    def reset(self, seed=None, options=None):
        if self._env is None or self._ep_count % ENV_RESTART_EP == 0:
            self._make_env()
        self._ep_count += 1
        obs, info = self._env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, _, terminated, truncated, info = self._env.step(action)
        obs = obs.astype(np.float32)

        vel     = float(info.get("velocity",    0.0))
        steer   = float(info.get("steering",    0.0))
        dest    = bool(info.get("arrive_dest",  False))
        oor     = bool(info.get("out_of_road",  False))
        crash   = bool(info.get("crash",        False))

        # ── Reward function (v3 — from diagnostic findings) ───────
        reward = 0.0

        # Forward progress — primary signal
        reward += 0.05 * vel

        # Penalize lateral deviation from road center
        # obs[0] ≈ 0.29 when centered — use empirical baseline not 0.5
        center_signal = float(obs[0])
        reward -= 0.5 * abs(center_signal - 0.30)

        # Small steering smoothness penalty — discourages oscillation
        reward -= 0.05 * abs(steer)

        # Terminal rewards
        if dest:   reward += 3000.0
        if oor:    reward -=  500.0
        if crash:  reward -=  200.0

        # Soft timeout penalty — reduced from -200 to -50
        if truncated and not dest:
            reward -= 50.0

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            try:   self._env.close()
            except: pass
            self._env = None


# ── Feature extractor — BC warm start via obs[:19] ────────────────────────────
class BCWarmStartExtractor(BaseFeaturesExtractor):
    """
    Uses first 19 dims of MetaDrive obs directly as BC sensor vector.
    Confirmed: obs[:19] matches BC training state vector exactly.
    LayerNorm instead of BatchNorm1d — more stable for RL (variable batch sizes).
    """
    def __init__(self, observation_space, bc_checkpoint=None):
        super().__init__(observation_space, features_dim=128)

        self.sensor_mlp = nn.Sequential(
            nn.Linear(19, 64),   nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 128),  nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        )

        if bc_checkpoint and os.path.exists(bc_checkpoint):
            self._load_bc_weights(bc_checkpoint)
        else:
            print("[BCWarmStart] No checkpoint — random init.")

    def _load_bc_weights(self, path):
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state"]

        # Extract sensor_branch weights, skip BN running stats
        filtered = {}
        for k, v in state_dict.items():
            if not k.startswith("sensor_branch.net."):
                continue
            new_key = k.replace("sensor_branch.net.", "")
            if any(x in new_key for x in
                   ["running_mean","running_var","num_batches_tracked"]):
                continue
            filtered[new_key] = v

        result = self.sensor_mlp.load_state_dict(filtered, strict=False)
        print(f"[BCWarmStart] Weights loaded. "
              f"Missing={result.missing_keys} "
              f"Unexpected={result.unexpected_keys}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sensor_mlp(obs[:, :19])   # use first 19 dims directly


# ── Resume helper ─────────────────────────────────────────────────────────────
def find_latest_checkpoint():
    ckpts = [f for f in os.listdir(RL_CKPT_DIR)
             if f.startswith("ppo_av_") and f.endswith(".zip")
             and "final" not in f]
    if not ckpts:
        return None, 0
    def get_step(fname):
        try:   return int(fname.replace("ppo_av_","").replace("_steps.zip",""))
        except: return 0
    latest = max(ckpts, key=get_step)
    return os.path.join(RL_CKPT_DIR, latest), get_step(latest)


# ── Logging callback ──────────────────────────────────────────────────────────
class RLLogger(BaseCallback):
    def __init__(self, log_dir, log_freq):
        super().__init__()
        self.log_freq  = log_freq
        self.log_path  = os.path.join(log_dir, "rl_log.json")
        self._last_log = 0

        # Load existing log on resume
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.log = json.load(f)
            print(f"[Logger] Resuming — {len(self.log['timesteps'])} entries loaded.")
        else:
            self.log = {
                "timesteps":    [],
                "mean_reward":  [],
                "success_rate": [],
                "oor_rate":     [],
                "crash_rate":   [],
                "mean_ep_len":  [],
            }

        self.ep_rewards   = []
        self.ep_successes = []
        self.ep_oors      = []
        self.ep_crashes   = []
        self.ep_lens      = []

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", [{}]),
            self.locals.get("dones", [False])
        ):
            if done:
                self.ep_rewards.append(  info.get("episode_reward", 0))
                self.ep_successes.append(float(info.get("arrive_dest", False)))
                self.ep_oors.append(     float(info.get("out_of_road", False)))
                self.ep_crashes.append(  float(info.get("crash",       False)))
                self.ep_lens.append(     info.get("episode_length", 0))

        if (self.num_timesteps - self._last_log) >= self.log_freq \
                and len(self.ep_rewards) > 0:

            n       = min(20, len(self.ep_rewards))
            mean_r  = np.mean(self.ep_rewards[-n:])
            succ    = np.mean(self.ep_successes[-n:]) * 100
            oor     = np.mean(self.ep_oors[-n:])      * 100
            crash   = np.mean(self.ep_crashes[-n:])   * 100
            ep_len  = np.mean(self.ep_lens[-n:])

            self.log["timesteps"].append(   self.num_timesteps)
            self.log["mean_reward"].append( float(mean_r))
            self.log["success_rate"].append(float(succ))
            self.log["oor_rate"].append(    float(oor))
            self.log["crash_rate"].append(  float(crash))
            self.log["mean_ep_len"].append( float(ep_len))
            self._last_log = self.num_timesteps

            print(f"  [{self.num_timesteps:>7,}] "
                  f"reward={mean_r:8.2f} | "
                  f"success={succ:5.1f}% | "
                  f"oor={oor:5.1f}% | "
                  f"crash={crash:4.1f}% | "
                  f"ep_len={ep_len:.0f}")

            # Crash-safe: write after every log entry
            with open(self.log_path, "w") as f:
                json.dump(self.log, f, indent=2)

        return True


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("Phase 2 — PPO Fine-tuning (IL→RL Transfer)")
    print(f"Target   : {TOTAL_STEPS:,} steps")
    print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Resume check ──────────────────────────────────────────────
    latest_ckpt, resumed_steps = find_latest_checkpoint()
    if latest_ckpt:
        print(f"[Resume] Found checkpoint at step {resumed_steps:,}")
        remaining = TOTAL_STEPS - resumed_steps
        if remaining <= 0:
            print("Already complete."); sys.exit(0)
    else:
        print("[Fresh] Starting from scratch.")
        remaining = TOTAL_STEPS

    # ── Env ───────────────────────────────────────────────────────
    vec_env = DummyVecEnv([MetaDriveGymEnv])
    vec_env = VecMonitor(vec_env,
                         filename=os.path.join(RL_LOG_DIR, "monitor"))

    # ── Policy kwargs ─────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = BCWarmStartExtractor,
        features_extractor_kwargs = dict(bc_checkpoint=BC_CHECKPOINT),
        net_arch                  = dict(pi=[64,64], vf=[64,64]),
        activation_fn             = nn.ReLU,
    )

    # ── Build or load model ───────────────────────────────────────
    if latest_ckpt:
        print(f"[Resume] Loading {latest_ckpt}")
        model = PPO.load(
            latest_ckpt, env=vec_env,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        reset_num = False
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = 1e-5,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 5,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.1,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            verbose         = 0,
            seed            = SEED,
            device          = "cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log = RL_LOG_DIR,
        )
        reset_num = True

    print(f"Device  : {model.device}")
    print(f"Params  : {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"Steps   : {remaining:,} remaining\n")
    print(f"{'Timestep':>10} | {'Reward':>10} | {'Success':>8} | "
          f"{'OOR':>7} | {'Crash':>7} | {'EpLen':>7}")
    print("-"*60)

    # ── Train ─────────────────────────────────────────────────────
    logger_cb = RLLogger(log_dir=RL_LOG_DIR, log_freq=LOG_FREQ)
    ckpt_cb   = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=RL_CKPT_DIR,
        name_prefix="ppo_av", verbose=1,
    )

    model.learn(
        total_timesteps     = remaining,
        callback            = [logger_cb, ckpt_cb],
        reset_num_timesteps = reset_num,
    )

    model.save(os.path.join(RL_CKPT_DIR, "ppo_av_final"))
    vec_env.close()

    print(f"\n{'='*60}")
    print(f"Done: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Final model : {RL_CKPT_DIR}/ppo_av_final.zip")
    print(f"Logs        : {RL_LOG_DIR}/rl_log.json")