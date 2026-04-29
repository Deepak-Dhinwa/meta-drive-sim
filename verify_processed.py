import numpy as np
import pandas as pd
from glob import glob
import os

PROC_DIR = "data_clean/processed"
eps = sorted(glob(os.path.join(PROC_DIR, "episode_*")))

total_steps  = 0
mismatches   = 0
steer_all    = []
accel_all    = []
corrupt_frames = 0

for ep in eps:
    act      = pd.read_csv(os.path.join(ep, "actions.csv"))
    sen      = pd.read_csv(os.path.join(ep, "sensors.csv"))
    frm_dir  = os.path.join(ep, "frames")
    n_frames = len(os.listdir(frm_dir))

    if len(act) != len(sen) or len(act) != n_frames:
        mismatches += 1

    steer_all.extend(act["steering"].tolist())
    accel_all.extend(act["acceleration"].tolist())
    total_steps += len(act)

    # Sample frame check
    sample = np.load(os.path.join(frm_dir,
                     sorted(os.listdir(frm_dir))[0]))
    if sample.dtype != np.uint8 or sample.shape != (128, 128, 3):
        corrupt_frames += 1

steer_all = np.array(steer_all)
accel_all = np.array(accel_all)
straight  = (np.abs(steer_all) < 0.05).mean() * 100
turning   = 100 - straight

print(f"Episodes         : {len(eps)}")
print(f"Total timesteps  : {total_steps:,}")
print(f"Mismatches       : {mismatches}")
print(f"Corrupt frames   : {corrupt_frames}")
print(f"Straight %       : {straight:.1f}%")
print(f"Turning  %       : {turning:.1f}%")
print(f"Accel mean       : {accel_all.mean():.4f}  (expect >0)")
print(f"Accel % positive : {(accel_all>0).mean()*100:.1f}%")
print(f"Steer bounds ok  : {'OK' if np.abs(steer_all).max() <= 1.0 else 'FAIL'}")
print(f"Accel bounds ok  : {'OK' if np.abs(accel_all).max() <= 1.0 else 'FAIL'}")