import os
import json
import pandas as pd
from glob import glob

PROC_DIR = "data_clean/processed"
OUT_DIR  = "data_clean"
os.makedirs(OUT_DIR, exist_ok=True)

all_eps = sorted(glob(os.path.join(PROC_DIR, "episode_*")))
print(f"Total episodes: {len(all_eps)}")

# Count steps per episode
ep_info = []
for ep in all_eps:
    n = len(pd.read_csv(os.path.join(ep, "actions.csv")))
    ep_info.append((ep, n))

total = sum(n for _, n in ep_info)

# 80/20 split at episode boundary
N_VAL     = int(len(ep_info) * 0.20)
train_eps = [ep for ep, _ in ep_info[:-N_VAL]]
val_eps   = [ep for ep, _ in ep_info[-N_VAL:]]

train_steps = sum(n for ep, n in ep_info[:-N_VAL])
val_steps   = sum(n for ep, n in ep_info[-N_VAL:])

print(f"Train: {len(train_eps)} episodes | {train_steps:,} timesteps")
print(f"Val:   {len(val_eps)}  episodes | {val_steps:,}  timesteps")
print(f"Split: {train_steps/total*100:.1f}% / {val_steps/total*100:.1f}%")

print(f"\nVal episodes (last {N_VAL}):")
for ep, n in ep_info[-N_VAL:]:
    print(f"  {os.path.basename(ep)} — {n} steps")

with open(os.path.join(OUT_DIR, "split.json"), "w") as f:
    json.dump({"train": train_eps, "val": val_eps}, f, indent=2)

print(f"\nSplit saved: data_clean/split.json")