# Architecture

FireSight AI is an image-only segmentation application with four separable
layers:

1. Data and preprocessing utilities under `src/firesight/data` and
   `src/firesight/preprocessing`.
2. Segmentation model adapters under `src/firesight/models`.
3. A shared predictor service under `src/firesight/inference`.
4. A FastAPI backend and static browser frontend under `app`.

Both U-Net and YOLO11-seg implement the same `SegmentationModel` interface.
The backend does not expose video endpoints or decode video streams.

Until a user supplies a validated checkpoint, model inference is unavailable.
The application must return an explicit error and must not fabricate masks,
confidence values, or research results.

