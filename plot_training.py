import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY = "checkpoints_v2/history.json"
OUT = "checkpoints_v2/training_curve.png"

with open(HISTORY) as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss = history["val_loss"]
epochs = range(1, len(train_loss) + 1)
best_ep = val_loss.index(min(val_loss)) + 1
best_val = min(val_loss)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", color="steelblue")
plt.plot(epochs, val_loss, label="Val Loss", color="darkorange")
# train.py unfreezes when epoch == UNFREEZE_EPOCH + 1 (epoch 6).
plt.axvline(x=6, color="gray", linestyle="--", alpha=0.7, label="Backbone unfrozen")
plt.axvline(
    x=best_ep,
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"Best val ep={best_ep} (loss={best_val:.5f})",
)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log scale)")
plt.title("Training History")
plt.legend()
plt.grid(alpha=0.3)
plt.yscale("log")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=120)
plt.close()
print(f"Saved: {OUT}")