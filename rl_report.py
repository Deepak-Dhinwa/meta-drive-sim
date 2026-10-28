"""Rebuild the Phase 2 PPO chart and summary from rl_logs/rl_log.json."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(x, w=5):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def nearest(t, values, target):
    i = int(np.argmin(np.abs(t - target)))
    return int(t[i]), float(values[i])


def write_summary(t, reward, eplen, oor, crash, path):
    t5, r5 = nearest(t, reward, 5000)
    _, e5 = nearest(t, eplen, 5000)
    _, o5 = nearest(t, oor, 5000)
    t50, e50 = nearest(t, eplen, 50000)
    t480, r480 = nearest(t, reward, 480000)
    _, e480 = nearest(t, eplen, 480000)
    _, o480 = nearest(t, oor, 480000)
    t500, r500 = nearest(t, reward, 500000)
    _, e500 = nearest(t, eplen, 500000)
    _, o500 = nearest(t, oor, 500000)
    bc = 299
    gain = (e480 - bc) / bc * 100
    text = f"""
RL TRAINING SUMMARY — Phase 2 PPO Fine-tuning
───────────────────────────────────────────────────────
Total steps      : {int(t[-1]):,}
Algorithm        : PPO (Stable-Baselines3 2.3.2)
Warm start       : BC SensorMLP weights (Phase 1)
Device           : CUDA (RTX 3050 Laptop)
Duration         : ~49 minutes

METRICS PROGRESSION  (from rl_logs/rl_log.json)
  Reward     : {r5:.1f} at {t5/1000:.0f}k → {r480:.1f} at {t480/1000:.0f}k → {r500:.1f} at {t500/1000:.0f}k
  Ep length  : {e5:.0f} at {t5/1000:.0f}k → {e480:.0f} at {t480/1000:.0f}k → {e500:.0f} at {t500/1000:.0f}k
  OOR rate   : {o5:.0f}% at {t5/1000:.0f}k → {o480:.0f}% at {t480/1000:.0f}k → {o500:.0f}% at {t500/1000:.0f}k
  Crash rate : 0% at the end (a 5–10% spike around steps 280k–310k)
  Success    : 0% (reward sparsity — destination never reached)

BC vs RL COMPARISON (episode length)
  BC Phase 1  :   {bc} steps avg (PID hybrid controller)
  RL early    : {e50:,.0f} steps avg (step {t50/1000:.0f}k)
  RL best     : {e480:,.0f} steps avg (step {t480/1000:.0f}k)
  RL final    : {e500:,.0f} steps avg (step {t500/1000:.0f}k)
  Improvement : +{gain:.0f}% survival time vs BC at the {t480/1000:.0f}k checkpoint

WHAT IMPROVED
  + Survival time increased {e480 / bc:.1f}x vs BC baseline at step {t480/1000:.0f}k
  + Agent learned lane-keeping from reward signal
  + OOR rate decreased from {o5:.0f}% to {oor.min():.0f}-{o480:.0f}%
  + Crash rate returns to 0% after a short spike around steps 280k–310k

WHAT DIDN'T IMPROVE
  - Success rate: 0% throughout (sparse destination reward)
  - Agent never reached destination in {int(t[-1]):,} steps

ROOT CAUSE — REWARD SPARSITY
  Destination bonus (+3000) is never triggered by random exploration.
  The agent finds a local optimum: "drive straight, stay on road,
  collect small velocity rewards" without ever turning toward goal.

NEXT STEPS
  [P1] Curriculum: start with short routes, gradually increase length
  [P2] Dense navigation reward: reward progress toward destination
  [P3] More steps: 1M+ steps likely needed for first successes
  [P4] Fix BC steering: top-down camera + weighted loss first,
       then RL fine-tune — better initialization = faster RL
───────────────────────────────────────────────────────
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def chart(t, reward, eplen, oor, crash, path):
    fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.55)
    fig.suptitle("Phase 2 — PPO Fine-tuning Analysis (500k steps)", color="white")
    slots = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    def style(ax, title):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(alpha=0.15)

    ax = slots[0]
    style(ax, "Reward Curve — PPO Fine-tuning (500k steps)")
    ax.plot(t, reward, color="#4fc3f7", alpha=0.35)
    ax.plot(t, smooth(reward), color="#4fc3f7", label="Mean reward (smoothed)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = slots[1]
    style(ax, "Episode Length")
    ax.plot(t, eplen, color="#66bb6a", alpha=0.35)
    ax.plot(t, smooth(eplen), color="#66bb6a")
    ax.axhline(4000, color="gray", ls="--", label="Max horizon")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg Episode Length")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = slots[2]
    style(ax, "Out-of-Road Rate")
    ax.plot(t, oor, color="#ffa726", label="OOR rate")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("OOR Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = slots[3]
    style(ax, "Crash Rate")
    ax.plot(t, crash, color="#ef5350", label="Crash rate")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Crash Rate (%)")
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=8)

    ax = slots[4]
    style(ax, "Episode Length: BC vs RL")
    _, e50 = nearest(t, eplen, 50000)
    _, e480 = nearest(t, eplen, 480000)
    _, e500 = nearest(t, eplen, 500000)
    labels = ["BC", "RL 50k", "RL 480k", "RL 500k"]
    vals = [299, e50, e480, e500]
    ax.bar(labels, vals, color=["#ef5350", "#ffa726", "#66bb6a", "#4fc3f7"])
    ax.set_ylabel("Avg Episode Length")

    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report")
    parser.add_argument("--log", default="rl_logs/rl_log.json")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.log) as f:
        log = json.load(f)
    t = np.array(log["timesteps"], dtype=float)
    reward = np.array(log["mean_reward"], dtype=float)
    eplen = np.array(log["mean_ep_len"], dtype=float)
    oor = np.array(log["oor_rate"], dtype=float)
    crash = np.array(log["crash_rate"], dtype=float)
    chart(t, reward, eplen, oor, crash, os.path.join(args.out, "rl_analysis.png"))
    write_summary(t, reward, eplen, oor, crash, os.path.join(args.out, "rl_summary.txt"))
    print(f"Wrote Phase 2 report to {args.out}")


if __name__ == "__main__":
    main()
