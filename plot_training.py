import json
import matplotlib.pyplot as plt

with open("checkpoints/history.json") as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss   = history["val_loss"]
epochs     = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", color="steelblue")
plt.plot(epochs, val_loss,   label="Val Loss",   color="darkorange")
plt.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="Backbone unfrozen")
plt.axvline(x=train_loss.index(min(train_loss))+1,
            color="green", linestyle="--", alpha=0.5, label="Best train")
plt.axvline(x=val_loss.index(min(val_loss))+1,
            color="red", linestyle="--", alpha=0.5, label="Best val (ep 39)")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Training History — Fusion Model")
plt.legend(); plt.grid(alpha=0.3)
plt.yscale("log")   # log scale shows early convergence better
plt.tight_layout()
plt.savefig("checkpoints/training_curve.png", dpi=120)
plt.show()
print("Saved: checkpoints/training_curve.png")