"""Strict parser for normalized YOLO detection labels (five fields)."""

def parse_yolo_detection_line(line: str):
    parts=line.strip().split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields for YOLO detection, received {len(parts)}")
    class_id=int(parts[0])
    values=[float(v) for v in parts[1:]]
    if class_id < 0:
        raise ValueError("class_id must be non-negative")
    if any(v < 0 or v > 1 for v in values):
        raise ValueError("normalized coordinates must be within [0, 1]")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("box width and height must be positive")
    return {"class_id":class_id,"x_center":values[0],"y_center":values[1],
            "width":values[2],"height":values[3]}
