import type { Inputs } from './schema'

// Backend base URL. Set VITE_API_BASE_URL at build time (Vercel project settings).
// Local development falls back to the local FastAPI server.
const RAW_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000'
export const API_BASE = RAW_BASE.replace(/\/+$/, '')

export interface PredictionResult {
  estimated_probability: number
  decision_threshold: number
  above_threshold: boolean
  result_label: string
  interpretation: string
  target: string
  model: { model_name: string; calibration: string; training_mode: string; data_source: string }
  disclaimer: string
  request_id: string
}

export class ApiError extends Error {
  status: number | null
  constructor(message: string, status: number | null) {
    super(message)
    this.status = status
  }
}

const TIMEOUT_MS = 60_000 // Render free instances can take a while to wake up

async function request(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS)
  try {
    return await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal })
  } catch {
    throw new ApiError('The prediction service could not be reached. Check your connection and try again.', null)
  } finally {
    window.clearTimeout(timer)
  }
}

export async function predict(inputs: Inputs): Promise<PredictionResult> {
  const res = await request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(inputs),
  })
  if (!res.ok) {
    let message = 'The prediction service is unavailable right now. Please try again later.'
    try {
      const body = await res.json()
      if (res.status === 422 && body?.errors?.length) {
        message = 'Some answers were not accepted. Please review your answers and try again.'
      } else if (res.status === 422 || res.status === 400) {
        message = 'Some answers were not accepted. Please review your answers and try again.'
      }
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(message, res.status)
  }
  return (await res.json()) as PredictionResult
}
