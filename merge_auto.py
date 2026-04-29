import os
import shutil
from glob import glob
import pandas as pd

BASE_DIR = "data_idm_auto"
OUT_DIR  = "data_clean/raw"
os.makedirs(OUT_DIR, exist_ok=True)

batches = sorted(glob(os.path.join(BASE_DIR, "batch_*", "raw")))
print(f"Batches to merge: {len(batches)}")

global_ep = 0
skipped   = 0

for batch_dir in batches:
    episodes = sorted(glob(os.path.join(batch_dir, "episode_*")))
    for ep_dir in episodes:
        act_path = os.path.join(ep_dir, "actions.csv")
        sen_path = os.path.join(ep_dir, "sensors.csv")
        frm_dir  = os.path.join(ep_dir, "frames")

        if not all(os.path.exists(p) for p in [act_path, sen_path, frm_dir]):
            skipped += 1
            continue

        n_frames = len(os.listdir(frm_dir))
        if n_frames == 0:
            skipped += 1
            continue

        dst = os.path.join(OUT_DIR, f"episode_{global_ep:04d}")
        shutil.copytree(ep_dir, dst)
        global_ep += 1

print(f"\n── Merge Summary ────────────────────────────────────")
print(f"  Episodes merged : {global_ep}")
print(f"  Skipped         : {skipped}")
print(f"  Output          : {OUT_DIR}")

# ── Quick sanity check on merged output ──────────────────────────────────────
print(f"\n── Spot Check (first 3 + last 3 episodes) ───────────")
all_eps = sorted(glob(os.path.join(OUT_DIR, "episode_*")))
for ep in all_eps[:3] + all_eps[-3:]:
    act      = pd.read_csv(os.path.join(ep, "actions.csv"))
    n_frames = len(os.listdir(os.path.join(ep, "frames")))
    steer_max = act["steering"].abs().max()
    accel_max = act["acceleration"].abs().max()
    print(f"  {os.path.basename(ep)} | steps={len(act):4d} | "
          f"frames={n_frames:4d} | "
          f"steer_max={steer_max:.3f} | accel_max={accel_max:.3f} | "
          f"match={'OK' if len(act)==n_frames else 'FAIL'}")

print(f"\nMerge complete.")