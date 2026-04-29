import numpy as np
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadrive import MetaDriveEnv
from inference import load_model, predict

TARGET_SPEED = 5.0

config = dict(
    use_render        = True,
    image_observation = True,
    norm_pixel        = True,
    num_scenarios     = 10,
    start_seed        = 2000,
    vehicle_config    = dict(image_source="main_camera"),
    sensors           = dict(main_camera=("MainCamera", 128, 128)),
    manual_control    = False,
)

class PIDController:
    def __init__(self, kp=0.15, ki=0.001, kd=0.02, target=5.0):
        self.kp       = kp
        self.ki       = ki
        self.kd       = kd
        self.target   = target
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0

    def step(self, current_speed, target_override=None):
        t              = target_override if target_override is not None else self.target
        err            = t - current_speed
        self.integral  = np.clip(self.integral + err, -3.0, 3.0)  # anti-windup
        derivative     = err - self.prev_err
        self.prev_err  = err
        output = self.kp*err + self.ki*self.integral + self.kd*derivative
        return float(np.clip(output, -1.0, 1.0))

def extract_obs(obs):
    return obs['image'][:, :, :, -1], obs['state']

def run_autonomous(model, env, n_episodes):
    results = []
    pid     = PIDController()

    for ep in range(n_episodes):
        obs, info = env.reset()
        pid.reset()
        done = False; ep_step = 0; ep_reward = 0.0

        print(f"\n── Episode {ep+1}/{n_episodes} "
              f"(seed={info.get('env_seed','?')}) ──────────────")

        while not done:
            frame, sensor = extract_obs(obs)
            steering      = float(predict(model, frame, sensor)[0])

            # Gradual speed ramp: target increases 0.03 m/s per step up to TARGET_SPEED
            ramp_target  = min(TARGET_SPEED, ep_step * 0.03)
            acceleration = pid.step(info['velocity'], ramp_target)

            obs, reward, terminated, truncated, info = env.step(
                np.array([steering, acceleration], dtype=np.float32)
            )
            ep_reward += reward
            ep_step   += 1
            done        = terminated or truncated

            if ep_step % 100 == 0:
                print(f"  step={ep_step:04d} | vel={info['velocity']:.2f} | "
                      f"steer={steering:.3f} accel={acceleration:.3f} | "
                      f"oor={info['out_of_road']}")

        status = ("CRASH" if info['crash']      else
                  "OOR"   if info['out_of_road'] else
                  "DEST"  if info['arrive_dest'] else "MAX")

        print(f"  [{status}] steps={ep_step} | reward={ep_reward:.2f}")
        results.append({
            "episode": ep+1, "status": status, "steps": ep_step,
            "reward": ep_reward, "crash": info['crash'],
            "oor": info['out_of_road'], "arrive_dest": info['arrive_dest'],
        })

    return results

if __name__ == "__main__":
    model = load_model()
    env   = MetaDriveEnv(config)
    try:
        results = run_autonomous(model, env, n_episodes=10)
    finally:
        env.close()

    print("\n" + "="*55)
    print("SUMMARY — Hybrid (model steers, PID speeds)")
    print("="*55)
    n_dest  = sum(1 for r in results if r['arrive_dest'])
    n_oor   = sum(1 for r in results if r['oor'])
    n_crash = sum(1 for r in results if r['crash'])
    print(f"Success : {n_dest}/{len(results)}")
    print(f"OOR     : {n_oor}")
    print(f"Crash   : {n_crash}")
    print(f"Avg steps: {np.mean([r['steps'] for r in results]):.1f}")
    for r in results:
        print(f"  Ep{r['episode']:02d} [{r['status']:5s}] "
              f"steps={r['steps']:4d} reward={r['reward']:.1f}")

    with open("checkpoints_v2/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)