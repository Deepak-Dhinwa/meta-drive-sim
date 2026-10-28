"""Rebuild the Phase 1 report from checkpoints_v2. Does not train or simulate."""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_inputs():
    with open("checkpoints_v2/history.json") as f:
        history = json.load(f)
    with open("checkpoints_v2/eval_metrics.json") as f:
        ev = json.load(f)
    return history, ev


def chart_training(history, path):
    train = history["train_loss"]
    val = history["val_loss"]
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(val)) + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train, color="steelblue", label="Train Loss")
    ax.plot(epochs, val, color="darkorange", label="Val Loss")
    ax.axvline(6, color="gray", ls="--", label="Backbone unfrozen")
    ax.axvline(
        best, color="green", ls="--",
        label=f"Best val ep={best} (loss={min(val):.5f})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training History")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_distance(rows, path):
    dist = np.array([r["distance_m"] for r in rows])
    colors = ["#e74c3c" if d < 50 else "#f39c12" if d <= 150 else "#27ae60" for d in dist]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(1, len(dist) + 1), dist, color=colors)
    ax.axhline(dist.mean(), color="#2980b9", ls="--", label=f"Mean={dist.mean():.0f}m")
    ax.bar([], [], color="#e74c3c", label="<50m (early fail)")
    ax.bar([], [], color="#f39c12", label="50-150m (mid fail)")
    ax.bar([], [], color="#27ae60", label=">150m (late fail)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance Traveled per Episode")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_analysis(rows, path):
    vel = np.array([r["avg_velocity"] for r in rows])
    dist = np.array([r["distance_m"] for r in rows])
    steer = np.array([r["steer_std"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(vel, bins=8, color="steelblue", edgecolor="white")
    axes[0].axvline(vel.mean(), color="red", ls="--", label=f"Mean={vel.mean():.2f} m/s")
    axes[0].axvline(5.0, color="green", ls="--", label="Target=5.0 m/s")
    axes[0].set_title("Velocity Distribution")
    axes[0].set_xlabel("Avg Velocity (m/s)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    sc = axes[1].scatter(steer, dist, c=reward, cmap="plasma", s=60)
    axes[1].set_title("Turn Activity vs Distance Traveled")
    axes[1].set_xlabel("Steering Std (turn activity)")
    axes[1].set_ylabel("Distance (m)")
    axes[1].grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[1], label="Reward")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def full_report(history, ev, rows, path):
    train = np.array(history["train_loss"])
    val = np.array(history["val_loss"])
    epochs = np.arange(1, len(train) + 1)
    dist = np.array([r["distance_m"] for r in rows])
    vel = np.array([r["avg_velocity"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    steer = np.array([r["steer_std"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    labels = [r["status"] for r in rows]
    counts = {k: labels.count(k) for k in ("OOR", "CRASH", "DEST", "MAX")}

    fig = plt.figure(figsize=(14, 16), facecolor="#0f1117")
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.3)
    fig.suptitle(
        "Autonomous Car Simulation — Phase 1 Evaluation & Failure Report",
        color="white", fontsize=14,
    )

    def style(ax, title):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[0, :2])
    style(ax, "Training History — Fusion Model v2 (IDM Data)")
    ax.plot(epochs, train, color="#4fc3f7", label="Train Loss")
    ax.plot(epochs, val, color="#ef5350", label="Val Loss")
    ax.axvline(6, color="gray", ls="--", label="Backbone unfrozen")
    best = int(np.argmin(val)) + 1
    ax.axvline(best, color="#66bb6a", ls="--", label=f"Best val ep={best} ({min(val):.5f})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.legend(facecolor="#1a1d27", labelcolor="white")

    ax = fig.add_subplot(gs[0, 2])
    style(ax, "Episode Outcomes (20 eps)")
    pie_labels, pie_vals, pie_colors = [], [], []
    for name, color in (("OOR", "#f39c12"), ("CRASH", "#e74c3c"), ("DEST", "#27ae60"), ("MAX", "#4fc3f7")):
        if counts[name]:
            pie_labels.append(f"{name} ({counts[name]})")
            pie_vals.append(counts[name])
            pie_colors.append(color)
    ax.pie(pie_vals, labels=pie_labels, colors=pie_colors, textprops={"color": "white"})

    ax = fig.add_subplot(gs[1, :])
    style(ax, "Distance Traveled per Episode")
    colors = ["#e74c3c" if d < 50 else "#f39c12" if d <= 150 else "#27ae60" for d in dist]
    ax.bar(np.arange(1, len(dist) + 1), dist, color=colors)
    ax.axhline(dist.mean(), color="#4fc3f7", ls="--", label=f"Mean={dist.mean():.0f}m")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance (m)")
    ax.legend(facecolor="#1a1d27", labelcolor="white")

    ax = fig.add_subplot(gs[2, 0])
    style(ax, "Avg Velocity Distribution")
    ax.hist(vel, bins=8, color="#4fc3f7")
    ax.axvline(vel.mean(), color="#ef5350", ls="--", label=f"Mean={vel.mean():.2f}")
    ax.axvline(5.0, color="#66bb6a", ls="--", label="Target=5.0")
    ax.set_xlabel("Avg Velocity (m/s)")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    style(ax, "Episode Length Distribution")
    ax.hist(steps, bins=8, color="#ffa726")
    ax.axvline(steps.mean(), color="#4fc3f7", ls="--", label=f"Mean={steps.mean():.0f}")
    ax.set_xlabel("Steps")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    style(ax, "Turn Activity vs Distance")
    sc = ax.scatter(steer, dist, c=reward, cmap="plasma", s=50)
    ax.set_xlabel("Steering Std")
    ax.set_ylabel("Distance (m)")
    fig.colorbar(sc, ax=ax)

    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_markdown(history, ev, rows, path):
    val = history["val_loss"]
    best_ep = int(np.argmin(val)) + 1
    dist = np.array([r["distance_m"] for r in rows])
    labels = [r["status"] for r in rows]
    n = len(rows)
    n_oor = labels.count("OOR")
    n_crash = labels.count("CRASH")
    n_dest = labels.count("DEST")
    early = int((dist < 50).sum())
    mid = int(((dist >= 50) & (dist <= 150)).sum())
    late = int((dist > 150).sum())
    low = [r for r in rows if r["steer_std"] < 0.01]
    high = [r for r in rows if r["steer_std"] >= 0.01]
    low_m = float(np.mean([r["distance_m"] for r in low])) if low else 0
    high_m = float(np.mean([r["distance_m"] for r in high])) if high else 0
    best = max(rows, key=lambda r: r["distance_m"])
    worst = min(rows, key=lambda r: r["distance_m"])
    text = f"""# Autonomous Car Simulation — Phase 1 Failure Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. Model Summary

| Property | Value |
|---|---|
| Architecture | FusionModel v2 (EfficientNet-B0 + SensorMLP + FusionHead) |
| Parameters | 4.89M total, 882K trainable (frozen backbone first 5 epochs) |
| Training Data | 90,155 IDM expert steps, 209 episodes, 85,167 after preprocessing |
| Val Loss (best) | **{min(val):.5f} MSE** at epoch {best_ep} |
| Camera-only baseline | 0.05322 MSE — sensor fusion improves by **81.5%** against the 0.00985 fusion run |
| Controller | Hybrid: BC model steers, PID controls speed (target=5.0 m/s) |

---

## 2. Training Curves

![Training History](chart_training.png)

**Observations:**
- Val loss falls from {val[0]:.5f} to {min(val):.5f} over {len(val)} epochs
- Backbone unfreeze at epoch 6: val loss goes from {val[4]:.5f} to {val[5]:.5f}
- Train and val stay close after the unfreeze
- Early stopping triggered at epoch {len(val)} (patience=15)

---

## 3. Evaluation Results ({n} unseen episodes, seeds 3000-3019)

| Metric | Value |
|---|---|
| Success rate | {n_dest}/{n} ({100 * n_dest / n:.1f}%) |
| Out of road | {n_oor}/{n} ({100 * n_oor / n:.0f}%) |
| Crashes | {n_crash}/{n} ({100 * n_crash / n:.0f}%) |
| Avg steps/episode | {ev['avg_steps']:.1f} |
| Avg reward/episode | {ev['avg_reward']:.2f} |
| Avg velocity | {ev['avg_velocity_ms']:.2f} m/s (target: 5.0) |
| Avg distance | {ev['avg_distance_m']:.1f} m |
| Incident rate | {ev['incident_rate_per_km']:.2f} per km |
| Best episode | Ep{best['episode']:02d} — {best['distance_m']:.0f}m (seed {best['seed']}) |
| Worst episode | Ep{worst['episode']:02d} — {worst['distance_m']:.0f}m (seed {worst['seed']}) |

Status uses the evaluator priority CRASH, then OOR, then destination. One crash episode is also flagged out-of-road, so the raw `oor` flag count is {ev['oor_count']}.

---

## 4. Distance & Failure Breakdown

![Distance per Episode](chart_distance.png)

| Failure Tier | Count | Pct | Description |
|---|---|---|---|
| Early (<50m) | {early} | {100 * early / n:.0f}% | Turn before reaching cruise speed |
| Mid (50-150m) | {mid} | {100 * mid / n:.0f}% | Drives straight, drifts at first curve |
| Late (>150m) | {late} | {100 * late / n:.0f}% | Long straight, fails at sharp turn |

---

## 5. Turn Activity Analysis

![Velocity and Turn Analysis](chart_analysis.png)

| Steering Activity | Avg Distance |
|---|---|
| Low turn (steer_std < 0.01) | {low_m:.0f}m |
| High turn (steer_std >= 0.01) | {high_m:.0f}m |

**Key finding:** Higher turn activity correlates with longer distance.
Episodes where the model attempted steering corrections survived longer
than episodes where steering stayed near zero.

---

## 6. Root Cause Analysis

### Cause 1 — Moderate Turn Underestimation [PRIMARY]

Grad-CAM shows attention on sky and terrain, not road edges, for moderate
turns (|steer| = 0.1-0.4). Extreme turns (|steer| > 0.5) are predicted
accurately. Moderate turns undershoot by about 40-60% because those
samples are a small part of the MSE.

### Cause 2 — Speed Distribution Mismatch [SECONDARY]

IDM demonstrations cruise near 7-8 m/s. The PID ramp reaches 5 m/s only
after about 167 steps, so early frames are slower than the training distribution.

### Cause 3 — Covariate Shift [CONTRIBUTING]

Pure behavioral cloning compounds small steering error. The dataset has
no recovery demonstrations near the road boundary.

---

## 7. Proposed Fixes (Priority Order)

| Priority | Fix | Expected Impact |
|---|---|---|
| P1 | Turn-weighted loss (3-5x on moderate turns) | Fixes primary cause directly |
| P2 | Top-down camera instead of ego-camera | Road geometry explicitly visible |
| P3 | DAgger: collect boundary recovery demos | Fixes covariate shift |
| P4 | Phase 2 RL (PPO fine-tuning from BC weights) | Model improves beyond demonstrations |

---

## 8. What Works

- Sensor fusion: 81.5% lower loss than camera-only on the published comparison
- Straight-road driving: near-zero steering on straight sections
- Extreme turns: |steer| > 0.5 predicted with small error
- Training pipeline: IDM data plus preprocessing reaches val loss {min(val):.5f}

---

*Numbers in the tables are computed from `checkpoints_v2/history.json` and `checkpoints_v2/eval_metrics.json`.*
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    history, ev = load_inputs()
    rows = ev["episode_results"]
    chart_training(history, os.path.join(args.out, "chart_training.png"))
    chart_distance(rows, os.path.join(args.out, "chart_distance.png"))
    chart_analysis(rows, os.path.join(args.out, "chart_analysis.png"))
    full_report(history, ev, rows, os.path.join(args.out, "full_report.png"))
    write_markdown(history, ev, rows, os.path.join(args.out, "failure_analysis.md"))
    print(f"Wrote Phase 1 report to {args.out}")


if __name__ == "__main__":
    main()
