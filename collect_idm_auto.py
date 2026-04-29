"""
Auto-restarts on OOM or any error.
Stores data in data_idm_auto/batch_001/raw, batch_002/raw, ...
Each batch = RESTART_EVERY episodes → new env instance → no memory leak.
Runs until TARGET_STEPS reached. No manual intervention needed.
"""
import numpy as np
import cv2
import os
import csv
import gc
import traceback

TARGET_STEPS  = 90000
RESTART_EVERY = 5          # new env every 8 episodes (safe margin before OOM)
BASE_DIR      = "data_idm_auto"
FRAME_SIZE    = (128, 128)
FLUSH_EVERY   = 10
START_SEED    = 2000       # fresh seeds, away from manual + v1 data

os.makedirs(BASE_DIR, exist_ok=True)

# ── Track global state in a simple text file ──────────────────────────────────
STATE_FILE = os.path.join(BASE_DIR, "progress.txt")

def read_progress():
    if not os.path.exists(STATE_FILE):
        return 0, 0, 1    # global_ep, global_step, batch_num
    with open(STATE_FILE) as f:
        parts = f.read().strip().split(",")
        return int(parts[0]), int(parts[1]), int(parts[2])

def write_progress(global_ep, global_step, batch_num):
    with open(STATE_FILE, "w") as f:
        f.write(f"{global_ep},{global_step},{batch_num}")

def get_ep_dir(batch_num, ep_idx):
    path = os.path.join(BASE_DIR, f"batch_{batch_num:03d}", "raw",
                        f"episode_{ep_idx:04d}")
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    return path

def process_frame(raw_image):
    frame = raw_image[:, :, :, -1]
    return (cv2.resize(frame, FRAME_SIZE,
                       interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)

def open_writers(ep_dir):
    sf = open(os.path.join(ep_dir, "sensors.csv"), "w", newline="", buffering=1)
    af = open(os.path.join(ep_dir, "actions.csv"),  "w", newline="", buffering=1)
    sw, aw = csv.writer(sf), csv.writer(af)
    sw.writerow(["timestep"] + [f"s{i}" for i in range(19)] + ["velocity"])
    aw.writerow(["timestep", "steering", "acceleration"])
    return sf, af, sw, aw

def make_env(seed_offset):
    from metadrive import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    return MetaDriveEnv(dict(
        use_render        = False,
        image_observation = True,
        norm_pixel        = True,
        num_scenarios     = 500,
        start_seed        = START_SEED + seed_offset,
        vehicle_config    = dict(image_source="main_camera"),
        sensors           = dict(main_camera=("MainCamera", 128, 128)),
        agent_policy      = IDMPolicy,
    ))

def run_batch(batch_num, global_ep, global_step):
    """Run RESTART_EVERY episodes in one env instance. Returns updated counters."""
    seed_offset = global_ep * 3   # vary seeds across batches
    env = make_env(seed_offset)
    print(f"\n[Batch {batch_num:03d}] env created | "
          f"global_ep={global_ep} | global_step={global_step}")

    try:
        for _ in range(RESTART_EVERY):
            if global_step >= TARGET_STEPS:
                break

            obs, info = env.reset()
            ep_dir    = get_ep_dir(batch_num, global_ep)
            sf, af, sw, aw = open_writers(ep_dir)
            ep_step   = 0
            done      = False

            while not done:
                obs, reward, terminated, truncated, info = env.step([0, 0])
                done = terminated or truncated

                steering     = float(info['steering'])
                acceleration = float(info['acceleration'])

                # Hard guard — skip corrupted steps
                if abs(steering) > 1.5 or abs(acceleration) > 1.5:
                    continue

                np.save(os.path.join(ep_dir, "frames", f"{ep_step:06d}.npy"),
                        process_frame(obs['image']))

                sw.writerow([ep_step] + obs['state'].tolist() + [info['velocity']])
                aw.writerow([ep_step, round(steering, 6), round(acceleration, 6)])

                if ep_step % FLUSH_EVERY == 0:
                    sf.flush(); af.flush()

                ep_step     += 1
                global_step += 1

            sf.flush(); af.flush()
            sf.close(); af.close()

            status = ("DEST"  if info['arrive_dest'] else
                      "CRASH" if info['crash']        else
                      "OOR"   if info['out_of_road']  else "MAX")

            print(f"  Ep {global_ep:04d} [{status}] "
                  f"steps={ep_step:4d} | total={global_step:6d}")
            global_ep += 1
            write_progress(global_ep, global_step, batch_num)

    except Exception as e:
        print(f"\n⚠️  Error in batch {batch_num}: {e}")
        traceback.print_exc()
        try:
            sf.close(); af.close()
        except:
            pass

    finally:
        try:
            env.close()
        except:
            pass
        gc.collect()
        print(f"  [Batch {batch_num:03d}] env closed. "
              f"global_step={global_step}")

    return global_ep, global_step

# ── Main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    global_ep, global_step, batch_num = read_progress()

    print("=" * 60)
    print(f"IDM Auto-Collector | Target: {TARGET_STEPS} steps")
    print(f"Resuming from: ep={global_ep} step={global_step} batch={batch_num}")
    print("=" * 60)

    while global_step < TARGET_STEPS:
        try:
            global_ep, global_step = run_batch(batch_num, global_ep, global_step)
            batch_num += 1
            write_progress(global_ep, global_step, batch_num)
        except Exception as e:
            print(f"\nOuter error: {e} — retrying batch {batch_num}")
            traceback.print_exc()
            batch_num += 1   # skip to next batch on hard failure

    print(f"\n{'='*60}")
    print(f"   Collection complete!")
    print(f"   Total episodes : {global_ep}")
    print(f"   Total steps    : {global_step}")
    print(f"   Data location  : {BASE_DIR}/batch_XXX/raw/")