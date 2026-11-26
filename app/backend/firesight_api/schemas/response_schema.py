
"""Standard response schemas for FireSight AI API."""

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Common API response structure."""

    success: bool = True
    message: str
    data: dict = Field(default_factory=dict)


class APIErrorResponse(BaseModel):
    """Standard API error structure."""

    success: bool = False
    error: str
    detail: str | None = None


class ModelInformation(BaseModel):
    """Information and availability of a segmentation model."""

    id: str
    name: str
    task: str
    classes: list[str]
    input_size: str
    loaded: bool
    checkpoint_available: bool
    error: str | None = None
