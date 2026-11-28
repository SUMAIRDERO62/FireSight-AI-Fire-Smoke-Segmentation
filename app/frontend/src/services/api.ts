import type { ModelName, PredictionResponse } from '../types/api'

export async function predictImage(
  file: File,
  modelName: ModelName,
): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model_name', modelName)

  const response = await fetch('/api/v1/predict', {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(
      typeof data.detail === 'string'
        ? data.detail
        : 'Prediction request failed. Please try again.',
    )
  }

  return data as PredictionResponse
}

export async function getHealth() {
  const response = await fetch('/api/v1/health')
  if (!response.ok) throw new Error('API health check failed')
  return response.json()
}
