"""Deterministic hash-based assignment helpers."""
import hashlib

def stable_bucket(identifier: str, seed: int = 42, buckets: int = 10000) -> int:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    token = f"{seed}:{identifier}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % buckets

def assign_train_val(identifier: str, val_ratio: float = 0.15, seed: int = 42) -> str:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    return "val" if stable_bucket(identifier, seed) < int(val_ratio * 10000) else "train"
