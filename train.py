import torch
import torch.nn as nn
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import FusionModel
from dataset import DrivingDataset, get_dataloaders

if __name__ == '__main__':

    SPLIT_JSON     = "data_clean/split.json"
    CHECKPOINT_DIR = "checkpoints_v2"
    EPOCHS         = 80
    BATCH_SIZE     = 32
    LR_HEAD        = 1e-4
    LR_BACKBONE    = 1e-5
    UNFREEZE_EPOCH = 5
    PATIENCE       = 15
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    with open(SPLIT_JSON) as f:
        split = json.load(f)

    train_loader, val_loader = get_dataloaders(
        proc_dir   = "data_clean/processed",
        train_eps  = split["train"],
        val_eps    = split["val"],
        batch_size = BATCH_SIZE,
        num_workers= 0,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = FusionModel().to(DEVICE)

    backbone_params = list(model.visual_branch.backbone.parameters())
    head_params     = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    optimizer = torch.optim.Adam([
        {"params": head_params,     "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    criterion = nn.MSELoss()

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):

        if epoch == UNFREEZE_EPOCH + 1:
            model.unfreeze_backbone()

        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for frames, sensors, actions in train_loader:
            frames  = frames.to(DEVICE)
            sensors = sensors.to(DEVICE)
            actions = actions.to(DEVICE)

            optimizer.zero_grad()
            preds = model(frames, sensors)
            loss  = criterion(preds, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, sensors, actions in val_loader:
                frames  = frames.to(DEVICE)
                sensors = sensors.to(DEVICE)
                actions = actions.to(DEVICE)
                preds   = model(frames, sensors)
                val_loss += criterion(preds, actions).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        unfreeze_marker = " [BACKBONE UNFROZEN]" if epoch == UNFREEZE_EPOCH + 1 else ""
        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"train={train_loss:.5f} | val={val_loss:.5f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}"
              f"{unfreeze_marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best saved (val={val_loss:.5f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n── Training complete ────────────────────────────────────")
    print(f"   Best val loss : {best_val_loss:.5f}")
    print(f"   Checkpoint    : {CHECKPOINT_DIR}/best_model.pth")