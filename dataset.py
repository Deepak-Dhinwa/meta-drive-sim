import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob

class DrivingDataset(Dataset):
    def __init__(self, episode_dirs: list[str]):
        """
        episode_dirs: list of paths to processed episode folders
        Loads all timesteps from all episodes into an index.
        Does NOT load frames into RAM — loads on __getitem__ (disk-backed).
        """
        self.samples = []   # list of (frame_path, sensor_row, action_row)

        for ep_dir in episode_dirs:
            actions = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
            sensors = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))
            frames_dir = os.path.join(ep_dir, "frames")

            assert len(actions) == len(sensors), \
                f"Mismatch in {ep_dir}: {len(actions)} actions vs {len(sensors)} sensors"

            for i in range(len(actions)):
                timestep   = int(actions["timestep"].iloc[i])
                frame_path = os.path.join(frames_dir, f"{timestep:06d}.npy")

                if not os.path.exists(frame_path):
                    continue   # skip missing frames silently

                sensor_vec = sensors.iloc[i][[f"s{j}" for j in range(19)]].values.astype(np.float32)
                action_vec = np.array([
                    actions["steering"].iloc[i],
                    actions["acceleration"].iloc[i],
                ], dtype=np.float32)

                self.samples.append((frame_path, sensor_vec, action_vec))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, sensor_vec, action_vec = self.samples[idx]

        # ── Frame: uint8 → float32 [0,1], HWC → CHW ──────────────
        frame = np.load(frame_path).astype(np.float32) / 255.0  # (128,128,3)
        frame = torch.from_numpy(frame).permute(2, 0, 1)         # (3,128,128)

        sensor = torch.from_numpy(sensor_vec)                     # (19,)
        action = torch.from_numpy(action_vec)                     # (2,)

        return frame, sensor, action


def get_dataloaders(proc_dir: str, train_eps: list[str], val_eps: list[str],
                    batch_size: int = 32, num_workers: int = 2):

    train_dataset = DrivingDataset(train_eps)
    val_dataset   = DrivingDataset(val_eps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle at sample level — OK since we broke temporal
                                # continuity intentionally via flip augmentation
        num_workers=num_workers,
        pin_memory=True,        # faster CPU→GPU transfer
        drop_last=True,         # avoid incomplete final batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ── Unit test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from glob import glob

    with open("data_clean/split.json") as f:
        split = json.load(f)

    train_eps = split["train"]
    val_eps   = split["val"]

    print(f"Train episodes: {len(train_eps)}")
    print(f"Val episodes:   {len(val_eps)}")

    # Test on first 3 train episodes
    ds = DrivingDataset(train_eps[:3])
    print(f"Dataset size (3 eps): {len(ds)}")

    frame, sensor, action = ds[0]
    print(f"\nSample [0]:")
    print(f"  frame  shape={tuple(frame.shape)} dtype={frame.dtype} "
          f"min={frame.min():.3f} max={frame.max():.3f}")
    print(f"  sensor shape={tuple(sensor.shape)} dtype={sensor.dtype}")
    print(f"  action shape={tuple(action.shape)} values={action.numpy().round(3)}")

    train_loader, val_loader = get_dataloaders(
        proc_dir   = "data_clean/processed",
        train_eps  = train_eps,
        val_eps    = val_eps,
        batch_size = 32,
        num_workers= 0,
    )

    frames_b, sensors_b, actions_b = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  frames  {tuple(frames_b.shape)}")
    print(f"  sensors {tuple(sensors_b.shape)}")
    print(f"  actions {tuple(actions_b.shape)}")
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"\nDataset check complete.")