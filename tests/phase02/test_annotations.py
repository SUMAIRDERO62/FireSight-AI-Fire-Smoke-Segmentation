import pytest
from firesight.data.annotations import parse_yolo_detection_line

def test_valid_detection_row():
    row=parse_yolo_detection_line("1 0.5 0.5 0.2 0.3")
    assert row["class_id"] == 1
    assert row["width"] == 0.2

def test_blank_row():
    assert parse_yolo_detection_line("  ") is None

@pytest.mark.parametrize("line", [
    "1 0.5 0.5 0.2",
    "1 0.5 0.5 0.2 0.3 0.4",
    "1 1.2 0.5 0.2 0.3",
    "-1 0.5 0.5 0.2 0.3",
    "0 0.5 0.5 0 0.2",
])
def test_invalid_rows(line):
    with pytest.raises((ValueError,)):
        parse_yolo_detection_line(line)
