from pathlib import Path

import torch


class SegmentationTrainer:
    """Training and validation engine for segmentation models."""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        checkpoint_path="models/checkpoints/unet_best.pth",
        gradient_clip=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_path = Path(checkpoint_path)
        self.gradient_clip = gradient_clip

        self.best_val_loss = float("inf")

    def train_one_epoch(self, train_loader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss detected.")

            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Training loader is empty.")

        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for images, masks in val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device, dtype=torch.long)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss detected.")

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Validation loader is empty.")

        return total_loss / total_samples

    def save_checkpoint(self, epoch, val_loss):
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "validation_loss": val_loss,
        }

        temporary_path = self.checkpoint_path.with_suffix(".tmp")

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(self.checkpoint_path)

    def fit(self, train_loader, val_loader, epochs):
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print("Best checkpoint saved.")

        return history
