"""Training and validation engine for semantic segmentation."""

import csv
import json
from pathlib import Path

import torch

from firesight.training.checkpoint import CheckpointManager


class SegmentationTrainer:
    """Train, validate, checkpoint and monitor segmentation models."""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        checkpoint_dir="models/checkpoints",
        gradient_clip=None,
        scheduler=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

        self.checkpoints = CheckpointManager(checkpoint_dir)
        self.best_val_loss = float("inf")

    def _run_epoch(self, loader, training):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for images, masks in loader:
                images = images.to(
                    self.device,
                    dtype=torch.float32,
                )
                masks = masks.to(
                    self.device,
                    dtype=torch.long,
                )

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("Training or validation loss is invalid.")

                if training:
                    loss.backward()

                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip,
                            error_if_nonfinite=True,
                        )

                    self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Data loader is empty.")

        return total_loss / total_samples

    def train_one_epoch(self, train_loader):
        return self._run_epoch(train_loader, training=True)

    def validate(self, val_loader):
        return self._run_epoch(val_loader, training=False)

    def resume(self, checkpoint_path=None):
        """Restore model, optimizer and scheduler from a checkpoint."""
        path = checkpoint_path or self.checkpoints.latest_path

        state = self.checkpoints.load(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        self.best_val_loss = state["best_val_loss"]
        return state["epoch"] + 1

    @staticmethod
    def _save_history(history, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "training_history.json"
        csv_path = output_dir / "training_history.csv"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"],
            )
            writer.writeheader()

            for row in zip(
                history["epoch"],
                history["train_loss"],
                history["val_loss"],
                history["learning_rate"],
            ):
                writer.writerow(
                    dict(
                        zip(
                            ["epoch", "train_loss", "val_loss", "learning_rate"],
                            row,
                        )
                    )
                )

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        start_epoch=1,
        early_stopping_patience=10,
        min_delta=0.0,
        history_dir="reports/training",
    ):
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if start_epoch < 1 or start_epoch > epochs:
            raise ValueError("start_epoch must be between 1 and epochs.")

        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")

        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        epochs_without_improvement = 0

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            improved = val_loss < self.best_val_loss - min_delta

            if improved:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            learning_rate = self.optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(learning_rate)

            self.checkpoints.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_val_loss=self.best_val_loss,
                scheduler=self.scheduler,
                is_best=improved,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Validation: {val_loss:.4f} | "
                f"LR: {learning_rate:.6g}"
            )

            if improved:
                print("Best checkpoint updated.")

            self._save_history(history, history_dir)

            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping: validation loss did not improve.")
                break

        return history
