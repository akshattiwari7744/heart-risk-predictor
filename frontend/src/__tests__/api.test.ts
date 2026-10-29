import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { predict, ApiError } from '../api'
import { toInputs, ALL_FIELDS, type FormAnswers } from '../schema'

const inputs = (() => {
  const a: FormAnswers = {}
  for (const f of ALL_FIELDS) a[f.key] = f.kind === 'select' ? String(f.options[0].value) : String(f.min)
  return toInputs(a)
})()

const okBody = {
  estimated_probability: 0.02, decision_threshold: 0.0999, above_threshold: false,
  result_label: 'below_decision_threshold', interpretation: 'x', target: 't',
  model: { model_name: 'logistic_regression', calibration: 'isotonic', training_mode: 'full', data_source: 'd' },
  disclaimer: 'x', request_id: 'abc12345',
}

beforeEach(() => {
  // The client uses window timers; provide them for the node test environment.
  ;(globalThis as unknown as { window: unknown }).window = { setTimeout, clearTimeout }
})
afterEach(() => {
  vi.unstubAllGlobals()
  delete (globalThis as unknown as { window?: unknown }).window
})

describe('predict()', () => {
  it('posts JSON to /predict and returns the parsed result', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(okBody), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const r = await predict(inputs)
    expect(r.estimated_probability).toBe(0.02)
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url.endsWith('/predict')).toBe(true)
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual(inputs)
  })

  it('turns a 422 into a friendly ApiError without exposing server detail', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: 'validation_error', message: 'x', errors: [{ field: 'bmi', problem: 'p' }] }), { status: 422 })))
    const err = await predict(inputs).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(422)
    expect(err.message).toMatch(/review your answers/i)
    expect(err.message).not.toMatch(/bmi/)
  })

  it('reports a generic message on 503', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 503 })))
    const err = await predict(inputs).catch((e) => e)
    expect(err.status).toBe(503)
    expect(err.message).toMatch(/unavailable/i)
  })

  it('reports a network failure as an ApiError with no status', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch') }))
    const err = await predict(inputs).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBeNull()
    expect(err.message).toMatch(/could not be reached/i)
  })
})
