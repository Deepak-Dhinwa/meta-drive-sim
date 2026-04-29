import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

DATA_DIR     = "data_clean/raw"
episode_dirs = sorted(glob(os.path.join(DATA_DIR, "episode_*")))
print(f"Episodes: {len(episode_dirs)}")

# ── Per-episode collection ────────────────────────────────────────────────────
records          = []
all_steering     = []
all_acceleration = []
all_velocity     = []

for ep_dir in episode_dirs:
    act      = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
    sen      = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))
    frm_dir  = os.path.join(ep_dir, "frames")
    n_frames = len(os.listdir(frm_dir))

    steer = act["steering"].values
    accel = act["acceleration"].values
    vel   = sen["velocity"].values

    all_steering.extend(steer.tolist())
    all_acceleration.extend(accel.tolist())
    all_velocity.extend(vel.tolist())

    # Sample one frame for dtype check
    sample_frame = np.load(os.path.join(frm_dir,
                           sorted(os.listdir(frm_dir))[len(os.listdir(frm_dir))//2]))

    records.append({
        "episode":        os.path.basename(ep_dir),
        "steps":          len(act),
        "frames":         n_frames,
        "match":          n_frames == len(act),
        "steer_mean":     steer.mean(),
        "steer_std":      steer.std(),
        "steer_max":      steer.max(),
        "steer_min":      steer.min(),
        "accel_mean":     accel.mean(),
        "accel_std":      accel.std(),
        "vel_mean":       vel.mean(),
        "vel_min":        vel.min(),
        "vel_max":        vel.max(),
        "frame_dtype":    str(sample_frame.dtype),
        "frame_min":      int(sample_frame.min()),
        "frame_max":      int(sample_frame.max()),
        "pct_turning":    (np.abs(steer) > 0.05).mean() * 100,
        "pct_braking":    (accel < 0).mean() * 100,
        "pct_idle":       (vel < 0.5).mean() * 100,
    })

df = pd.DataFrame(records)

all_steering     = np.array(all_steering)
all_acceleration = np.array(all_acceleration)
all_velocity     = np.array(all_velocity)
total_steps      = len(all_steering)

# ── 1. Dataset-level summary ──────────────────────────────────────────────────
print("\n── 1. Dataset Summary ───────────────────────────────────────────")
print(f"  Episodes            : {len(df)}")
print(f"  Total timesteps     : {total_steps:,}")
print(f"  Avg steps/episode   : {df['steps'].mean():.1f}")
print(f"  Min/Max steps/ep    : {df['steps'].min()} / {df['steps'].max()}")
print(f"  Frame/CSV mismatches: {(~df['match']).sum()}")
print(f"  Frame dtypes        : {df['frame_dtype'].value_counts().to_dict()}")
print(f"  Frame pixel range   : min={df['frame_min'].min()} max={df['frame_max'].max()}")

# ── 2. Action statistics ──────────────────────────────────────────────────────
print("\n── 2. Action Statistics ─────────────────────────────────────────")
print(f"  Steering")
print(f"    min={all_steering.min():.4f}  max={all_steering.max():.4f}")
print(f"    mean={all_steering.mean():.4f}  std={all_steering.std():.4f}")
print(f"    % near-zero (|s|<0.05)  : {(np.abs(all_steering)<0.05).mean()*100:.1f}%")
print(f"    % turning   (|s|>0.05)  : {(np.abs(all_steering)>0.05).mean()*100:.1f}%")
print(f"    % hard turn (|s|>0.3)   : {(np.abs(all_steering)>0.3).mean()*100:.1f}%")

print(f"  Acceleration")
print(f"    min={all_acceleration.min():.4f}  max={all_acceleration.max():.4f}")
print(f"    mean={all_acceleration.mean():.4f}  std={all_acceleration.std():.4f}")
print(f"    % positive (throttle)   : {(all_acceleration>0).mean()*100:.1f}%")
print(f"    % negative (brake)      : {(all_acceleration<0).mean()*100:.1f}%")
print(f"    % near-zero (<|0.05|)   : {(np.abs(all_acceleration)<0.05).mean()*100:.1f}%")

# ── 3. Velocity statistics ────────────────────────────────────────────────────
print("\n── 3. Velocity Statistics ───────────────────────────────────────")
print(f"  min={all_velocity.min():.3f}  max={all_velocity.max():.3f}")
print(f"  mean={all_velocity.mean():.3f}  std={all_velocity.std():.3f}")
print(f"  % idle (vel<0.5)    : {(all_velocity<0.5).mean()*100:.1f}%")
print(f"  % slow (0.5-3.0)    : {((all_velocity>=0.5)&(all_velocity<3)).mean()*100:.1f}%")
print(f"  % cruise (3.0-7.0)  : {((all_velocity>=3)&(all_velocity<7)).mean()*100:.1f}%")
print(f"  % fast (>7.0)       : {(all_velocity>=7).mean()*100:.1f}%")

# ── 4. Imbalance analysis ─────────────────────────────────────────────────────
print("\n── 4. Imbalance Analysis ────────────────────────────────────────")
straight = (np.abs(all_steering) < 0.05).sum()
turning  = (np.abs(all_steering) >= 0.05).sum()
print(f"  Straight : {straight:6,} ({straight/total_steps*100:.1f}%)")
print(f"  Turning  : {turning:6,}  ({turning/total_steps*100:.1f}%)")
print(f"  Ratio    : {straight/max(turning,1):.1f}:1 (straight:turning)")
print(f"  Recommended STRAIGHT_KEEP_RATIO for ~1:1 balance: "
      f"{turning/max(straight,1):.2f}")

# ── 5. Episode quality tiers ──────────────────────────────────────────────────
print("\n── 5. Episode Quality Tiers ─────────────────────────────────────")
short  = (df['steps'] <  100).sum()
medium = ((df['steps'] >= 100) & (df['steps'] < 400)).sum()
long_  = (df['steps'] >= 400).sum()
print(f"  Short  (<100 steps)  : {short}  episodes — likely crashes, low quality")
print(f"  Medium (100-400)     : {medium} episodes")
print(f"  Long   (>400 steps)  : {long_}  episodes — high quality")
print(f"  Recommendation: drop short episodes in preprocessing")

# ── 6. Per-episode turning coverage ──────────────────────────────────────────
print("\n── 6. Turning Coverage per Episode ──────────────────────────────")
print(f"  Avg % turning per episode : {df['pct_turning'].mean():.1f}%")
print(f"  Episodes with >20% turning: {(df['pct_turning']>20).sum()}")
print(f"  Episodes with <5% turning : {(df['pct_turning']<5).sum()}  ← pure straight roads")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("IDM Clean Dataset — Full EDA", fontsize=14)

# Steering dist
axes[0,0].hist(all_steering, bins=80, color="steelblue", edgecolor="none")
axes[0,0].axvline(0, color="red", linestyle="--", alpha=0.6)
axes[0,0].set_title("Steering Distribution"); axes[0,0].set_xlabel("steering")

# Acceleration dist
axes[0,1].hist(all_acceleration, bins=80, color="darkorange", edgecolor="none")
axes[0,1].axvline(0, color="red", linestyle="--", alpha=0.6)
axes[0,1].set_title("Acceleration Distribution"); axes[0,1].set_xlabel("acceleration")

# Velocity dist
axes[0,2].hist(all_velocity, bins=80, color="green", edgecolor="none")
axes[0,2].set_title("Velocity Distribution"); axes[0,2].set_xlabel("velocity (m/s)")

# Steps per episode
axes[1,0].bar(range(len(df)), df["steps"], color="purple", alpha=0.6)
axes[1,0].axhline(100, color="red", linestyle="--", alpha=0.6, label="100-step threshold")
axes[1,0].set_title("Steps per Episode"); axes[1,0].legend()

# Steering trace — 3 long episodes
long_eps = df[df['steps'] > 400].head(3).index.tolist()
for i, idx in enumerate(long_eps):
    act = pd.read_csv(os.path.join(episode_dirs[idx], "actions.csv"))
    axes[1,1].plot(act["steering"].values, alpha=0.7, label=f"ep{idx}")
axes[1,1].set_title("Steering Trace (3 long eps)"); axes[1,1].legend()

# Acceleration trace — same episodes
for i, idx in enumerate(long_eps):
    act = pd.read_csv(os.path.join(episode_dirs[idx], "actions.csv"))
    axes[1,2].plot(act["acceleration"].values, alpha=0.7, label=f"ep{idx}")
axes[1,2].set_title("Acceleration Trace (3 long eps)"); axes[1,2].legend()

# Turning % per episode histogram
axes[2,0].hist(df["pct_turning"], bins=40, color="teal", edgecolor="none")
axes[2,0].set_title("% Turning per Episode"); axes[2,0].set_xlabel("% timesteps turning")

# Velocity mean per episode
axes[2,1].hist(df["vel_mean"], bins=40, color="crimson", edgecolor="none")
axes[2,1].set_title("Mean Velocity per Episode"); axes[2,1].set_xlabel("mean vel (m/s)")

# Steering vs acceleration scatter (sample 3000 pts)
idx_sample = np.random.choice(total_steps, min(3000, total_steps), replace=False)
axes[2,2].scatter(all_steering[idx_sample], all_acceleration[idx_sample],
                  alpha=0.2, s=5, color="navy")
axes[2,2].set_title("Steering vs Acceleration"); axes[2,2].set_xlabel("steering")
axes[2,2].set_ylabel("acceleration")

plt.tight_layout()
plt.savefig("eda_clean_report.png", dpi=120)
plt.show()
print("\nEDA complete. Saved: eda_clean_report.png")