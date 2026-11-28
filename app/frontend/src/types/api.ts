export type ModelName = 'unet' | 'yolo11-seg'

export interface ClassPrediction {
  detected: boolean
  pixel_count: number
  coverage_percent: number
}

export interface PredictionResponse {
  success: boolean
  message: string
  result_id: string
  model: ModelName
  image: {
    width: number
    height: number
  }
  processing_time_seconds: number
  classes: {
    fire: ClassPrediction
    smoke: ClassPrediction
  }
  results: {
    overlay: string
    fire_mask: string
    smoke_mask: string
  }
}

export interface PredictionRecord {
  response: PredictionResponse
  originalImage: string
  fileName: string
}

export interface HealthResponse {
  success: boolean
  application: string
  version: string
  status: string
  models: Record<string, {
    loaded: boolean
    checkpoint_available: boolean
    error: string | null
  }>
}
