import torch
import torch.nn as nn
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel
from dataset import get_dataloaders

if __name__ == '__main__':

    CHECKPOINT_DIR = "checkpoints"
    EPOCHS         = 60
    BATCH_SIZE     = 32
    PATIENCE       = 12
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    with open("data_clean/split.json") as f:
        split = json.load(f)

    train_loader, val_loader = get_dataloaders(
        proc_dir   = "data_clean/processed",
        train_eps  = split["train"],
        val_eps    = split["val"],
        batch_size = BATCH_SIZE,
        num_workers= 0,
    )

    model     = FusionModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_loss": []}

    print(f"Device: {DEVICE} | Camera-only baseline (sensor zeroed)")

    for epoch in range(1, EPOCHS + 1):
        if epoch == 6:
            model.unfreeze_backbone()

        model.train()
        train_loss = 0.0
        for frames, sensors, actions in train_loader:
            frames  = frames.to(DEVICE)
            sensors = torch.zeros_like(sensors).to(DEVICE)  # ← zeroed sensors
            actions = actions.to(DEVICE)

            optimizer.zero_grad()
            preds = model(frames, sensors)
            loss  = criterion(preds, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, sensors, actions in val_loader:
                frames  = frames.to(DEVICE)
                sensors = torch.zeros_like(sensors).to(DEVICE)  # ← zeroed
                actions = actions.to(DEVICE)
                val_loss += criterion(model(frames, sensors), actions).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}/{EPOCHS} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "camera_only_model.pth"))
            print(f"New best (val={val_loss:.5f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    with open(os.path.join(CHECKPOINT_DIR, "camera_only_history.json"), "w") as f:
        json.dump(history, f)

    # 0.00985 is the fusion loss from the run that produced the published
    # 81.5% figure. The saved checkpoints_v2 model later reached 0.00043.
    published_fusion = 0.00985
    current_fusion = None
    ckpt_path = "checkpoints_v2/best_model.pth"
    if os.path.exists(ckpt_path):
        current_fusion = torch.load(ckpt_path, map_location="cpu", weights_only=True)["val_loss"]

    print(f"\nCamera-only best val loss: {best_val_loss:.5f}")
    print(f"Fusion loss in the published comparison: {published_fusion:.5f}")
    print(f"Sensor fusion improvement vs that run: "
          f"{(best_val_loss - published_fusion)/best_val_loss*100:.1f}%")
    if current_fusion is not None:
        print(f"Current checkpoints_v2 val loss: {current_fusion:.5f}")