import torch
import pytest

from torch.utils.data import DataLoader, TensorDataset

from firesight.training.trainer import SegmentationTrainer


class TinySegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, images):
        return self.conv(images)


@pytest.fixture
def loaders():
    images = torch.rand(4, 3, 16, 16)
    masks = torch.randint(0, 2, (4, 16, 16))

    dataset = TensorDataset(images, masks)

    return (
        DataLoader(dataset, batch_size=2),
        DataLoader(dataset, batch_size=2),
    )


@pytest.fixture
def trainer(tmp_path):
    model = TinySegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        gradient_clip=1.0,
    )


def test_training_epoch_returns_finite_loss(trainer, loaders):
    train_loader, _ = loaders

    loss = trainer.train_one_epoch(train_loader)

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_validation_returns_finite_loss(trainer, loaders):
    _, val_loader = loaders

    loss = trainer.validate(val_loader)

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_empty_loader_is_rejected(trainer):
    empty_loader = DataLoader(
        TensorDataset(
            torch.empty(0, 3, 16, 16),
            torch.empty(0, 16, 16, dtype=torch.long),
        ),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="empty"):
        trainer.validate(empty_loader)


def test_fit_creates_best_and_latest_checkpoints(
    trainer,
    loaders,
    tmp_path,
):
    train_loader, val_loader = loaders

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
        early_stopping_patience=2,
        history_dir=tmp_path / "history",
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    assert trainer.checkpoints.best_path.is_file()
    assert trainer.checkpoints.latest_path.is_file()

    assert (tmp_path / "history" / "training_history.json").is_file()
    assert (tmp_path / "history" / "training_history.csv").is_file()


def test_resume_restores_checkpoint(trainer, loaders):
    train_loader, val_loader = loaders

    trainer.fit(
        train_loader,
        val_loader,
        epochs=1,
    )

    next_epoch = trainer.resume()

    assert next_epoch == 2
    assert trainer.best_val_loss < float("inf")
