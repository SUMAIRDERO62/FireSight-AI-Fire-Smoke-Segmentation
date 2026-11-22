"""Checkpoint saving and restoration for segmentation training."""

from pathlib import Path

import torch


class CheckpointManager:
    """Manage best and latest model checkpoints."""

    def __init__(self, directory="models/checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.best_path = self.directory / "unet_best.pth"
        self.latest_path = self.directory / "unet_latest.pth"

    def save(
        self,
        model,
        optimizer,
        epoch,
        best_val_loss,
        scheduler=None,
        is_best=False,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        self._atomic_save(checkpoint, self.latest_path)

        if is_best:
            self._atomic_save(checkpoint, self.best_path)

    @staticmethod
    def _atomic_save(checkpoint, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".tmp")

        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Only load checkpoints from trusted sources.
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
        }
