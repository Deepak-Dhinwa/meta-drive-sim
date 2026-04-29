import numpy as np
import json
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_model, predict

N_EPISODES    = 20
START_SEED    = 3000
TARGET_SPEED  = 5.0
RESTART_EVERY = 8     # restart env every N episodes

config = dict(
    use_render        = False,
    image_observation = True,
    norm_pixel        = True,
    num_scenarios     = N_EPISODES,
    start_seed        = START_SEED,
    vehicle_config    = dict(image_source="main_camera"),
    sensors           = dict(main_camera=("MainCamera", 128, 128)),
    manual_control    = False,
)

class PIDController:
    def __init__(self, kp=0.15, ki=0.001, kd=0.02, target=5.0):
        self.kp=kp; self.ki=ki; self.kd=kd; self.target=target
        self.integral=0.0; self.prev_err=0.0

    def reset(self):
        self.integral=0.0; self.prev_err=0.0

    def step(self, current_speed, target_override=None):
        t             = target_override if target_override is not None else self.target
        err           = t - current_speed
        self.integral = np.clip(self.integral + err, -3.0, 3.0)
        derivative    = err - self.prev_err
        self.prev_err = err
        return float(np.clip(
            self.kp*err + self.ki*self.integral + self.kd*derivative,
            -1.0, 1.0))

def extract_obs(obs):
    return obs['image'][:, :, :, -1], obs['state']

def make_env():
    from metadrive import MetaDriveEnv
    return MetaDriveEnv(config)

def run_episode(model, env, pid, ep_num):
    obs, info = env.reset()
    pid.reset()
    done = False; ep_step = 0; ep_reward = 0.0
    velocities = []; steerings = []

    while not done:
        frame, sensor = extract_obs(obs)
        raw           = predict(model, frame, sensor)
        steering      = float(raw[0])
        ramp_target   = min(TARGET_SPEED, ep_step * 0.03)
        acceleration  = pid.step(float(info['velocity']), ramp_target)

        obs, reward, terminated, truncated, info = env.step(
            np.array([steering, acceleration], dtype=np.float32))
        ep_reward += reward
        ep_step   += 1
        done        = terminated or truncated

        velocities.append(float(info['velocity']))
        steerings.append(steering)

    status = ("CRASH" if info['crash']      else
              "OOR"   if info['out_of_road'] else
              "DEST"  if info['arrive_dest'] else "MAX")

    avg_vel  = float(np.mean(velocities))
    dist_m   = ep_step * avg_vel * 0.1

    result = {
        "episode":      ep_num,
        "seed":         int(info.get('env_seed', 0)),
        "status":       status,
        "steps":        ep_step,
        "reward":       float(ep_reward),
        "crash":        bool(info['crash']),
        "oor":          bool(info['out_of_road']),
        "arrive_dest":  bool(info['arrive_dest']),
        "avg_velocity": avg_vel,
        "max_velocity": float(np.max(velocities)),
        "steer_std":    float(np.std(steerings)),
        "distance_m":   dist_m,
    }

    print(f"Ep {ep_num:02d}/{N_EPISODES} [{status:5s}] "
          f"steps={ep_step:4d} | reward={ep_reward:7.2f} | "
          f"avg_vel={avg_vel:.2f} | dist={dist_m:.0f}m")

    return result

if __name__ == "__main__":
    model   = load_model()
    pid     = PIDController(target=TARGET_SPEED)
    results = []
    ep      = 1

    while ep <= N_EPISODES:
        env = make_env()
        try:
            for _ in range(RESTART_EVERY):
                if ep > N_EPISODES:
                    break
                result = run_episode(model, env, pid, ep)
                results.append(result)
                ep += 1
        finally:
            env.close()

    # ── Aggregate ─────────────────────────────────────────────────
    n_success  = sum(1 for r in results if r['arrive_dest'])
    n_oor      = sum(1 for r in results if r['oor'])
    n_crash    = sum(1 for r in results if r['crash'])
    avg_steps  = np.mean([r['steps']        for r in results])
    avg_reward = np.mean([r['reward']       for r in results])
    avg_vel    = np.mean([r['avg_velocity'] for r in results])
    avg_dist   = np.mean([r['distance_m']   for r in results])
    total_km   = sum(r['distance_m'] for r in results) / 1000.0
    inc_rate   = (n_crash + n_oor) / max(total_km, 0.001)

    summary = {
        "model":               "FusionModel v2 (IDM data, hybrid PID)",
        "checkpoint":          "checkpoints_v2/best_model.pth",
        "val_loss":            0.00043,
        "n_episodes":          N_EPISODES,
        "success_rate":        f"{n_success}/{N_EPISODES} ({n_success/N_EPISODES*100:.1f}%)",
        "oor_count":           n_oor,
        "crash_count":         n_crash,
        "avg_steps":           float(avg_steps),
        "avg_reward":          float(avg_reward),
        "avg_velocity_ms":     float(avg_vel),
        "avg_distance_m":      float(avg_dist),
        "incident_rate_per_km": float(inc_rate),
        "episode_results":     results,
    }

    os.makedirs("checkpoints_v2", exist_ok=True)
    with open("checkpoints_v2/eval_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY — {N_EPISODES} episodes")
    print(f"{'='*60}")
    print(f"  Success rate       : {n_success}/{N_EPISODES} ({n_success/N_EPISODES*100:.1f}%)")
    print(f"  OOR                : {n_oor}")
    print(f"  Crashes            : {n_crash}")
    print(f"  Avg steps/episode  : {avg_steps:.1f}")
    print(f"  Avg reward         : {avg_reward:.2f}")
    print(f"  Avg velocity (m/s) : {avg_vel:.2f}")
    print(f"  Avg distance (m)   : {avg_dist:.1f}")
    print(f"  Incident rate/km   : {inc_rate:.2f}")
    print(f"\nSaved: checkpoints_v2/eval_metrics.json")