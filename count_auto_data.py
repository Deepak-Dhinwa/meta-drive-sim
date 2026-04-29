import os
import pandas as pd
from glob import glob

BASE_DIR = "data_idm_auto"
batches  = sorted(glob(os.path.join(BASE_DIR, "batch_*", "raw")))
print(f"Batches found: {len(batches)}")

total_eps   = 0
total_steps = 0
corrupt_eps = 0

for batch_dir in batches:
    eps = sorted(glob(os.path.join(batch_dir, "episode_*")))
    for ep in eps:
        act_path = os.path.join(ep, "actions.csv")
        sen_path = os.path.join(ep, "sensors.csv")
        frm_dir  = os.path.join(ep, "frames")

        # Check all three exist
        if not all(os.path.exists(p) for p in [act_path, sen_path, frm_dir]):
            print(f"Missing files: {ep}")
            corrupt_eps += 1
            continue

        act      = pd.read_csv(act_path)
        sen      = pd.read_csv(sen_path)
        n_frames = len(os.listdir(frm_dir))
        n_act    = len(act)
        n_sen    = len(sen)

        # Check frame/csv alignment
        if not (n_frames == n_act == n_sen):
            print(f"Mismatch ep={ep} "
                  f"frames={n_frames} act={n_act} sen={n_sen}")
            corrupt_eps += 1
            continue

        # Check action bounds
        steer_ok = act["steering"].abs().max() <= 1.5
        accel_ok = act["acceleration"].abs().max() <= 1.5
        if not (steer_ok and accel_ok):
            print(f"Out-of-bounds actions: {ep} "
                  f"steer_max={act['steering'].abs().max():.3f} "
                  f"accel_max={act['acceleration'].abs().max():.3f}")
            corrupt_eps += 1
            continue

        total_eps   += 1
        total_steps += n_act

print(f"\n── Summary ──────────────────────────────────────────")
print(f"  Clean episodes  : {total_eps}")
print(f"  Corrupt/skipped : {corrupt_eps}")
print(f"  Total steps     : {total_steps}")
print(f"  Avg steps/ep    : {total_steps/max(total_eps,1):.1f}")