# launcher that restarts collect_idm_auto.py on any crash
import subprocess
import sys
import time
import os

TARGET_STEPS = 90000
STATE_FILE   = "data_idm_auto/progress.txt"

def get_current_steps():
    if not os.path.exists(STATE_FILE):
        return 0
    with open(STATE_FILE) as f:
        parts = f.read().strip().split(",")
        return int(parts[1])

attempt = 0
while True:
    steps = get_current_steps()
    if steps >= TARGET_STEPS:
        print(f"Target reached: {steps} steps. Done.")
        break

    attempt += 1
    print(f"\n{'='*55}")
    print(f"Attempt {attempt} | Steps so far: {steps}/{TARGET_STEPS}")
    print(f"{'='*55}")

    result = subprocess.run(
        [sys.executable, "collect_idm_auto.py"],
        check=False
    )

    if result.returncode == 0:
        print("Collector exited cleanly.")
        break
    else:
        print(f"Collector crashed (code {result.returncode}). Restarting in 3s...")
        time.sleep(3)