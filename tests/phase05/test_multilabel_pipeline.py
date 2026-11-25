import numpy as np
import torch
from PIL import Image

from firesight.data.multilabel_dataset import MultiLabelSegmentationDataset
from firesight.data.transforms import SegmentationTransform
from firesight.training.losses import BCEWithLogitsDiceLoss, MultiLabelDiceLoss


def test_multilabel_dataset_preserves_overlap(tmp_path):
    images = tmp_path / "images"
    fire = tmp_path / "fire"
    smoke = tmp_path / "smoke"
    for p in (images, fire, smoke):
        p.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images / "x.jpg")
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.zeros((32, 32), dtype=np.uint8)
    a[4:20, 4:20] = 255
    b[12:28, 12:28] = 255
    Image.fromarray(a).save(fire / "x.png")
    Image.fromarray(b).save(smoke / "x.png")
    ds = MultiLabelSegmentationDataset(images, fire, smoke)
    _, target = ds[0]
    assert target.shape == (2, 32, 32)
    assert target[0, 15, 15] == 1
    assert target[1, 15, 15] == 1


def test_multilabel_transform_and_loss_backward():
    image = torch.rand(3, 32, 32)
    mask = torch.zeros(2, 32, 32)
    mask[:, 8:20, 8:20] = 1
    transform = SegmentationTransform(image_size=(24, 24))
    image, mask = transform(image, mask)
    assert image.shape == (3, 24, 24)
    assert mask.shape == (2, 24, 24)
    logits = torch.randn(1, 2, 24, 24, requires_grad=True)
    loss = BCEWithLogitsDiceLoss()(logits, mask.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
