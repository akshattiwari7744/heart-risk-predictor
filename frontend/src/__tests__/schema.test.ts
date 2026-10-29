import { describe, expect, it } from 'vitest'
import { ALL_FIELDS, FEATURE_KEYS, STEPS, fieldError, toInputs, type FormAnswers } from '../schema'

const validAnswers = (): FormAnswers => {
  const a: FormAnswers = {}
  for (const f of ALL_FIELDS) {
    if (f.kind === 'select') a[f.key] = String(f.options[0].value)
    else a[f.key] = String(f.min)
  }
  return a
}

describe('schema matches the backend contract', () => {
  it('has exactly the 19 model inputs (the form order may differ)', () => {
    expect(FEATURE_KEYS).toHaveLength(19)
    expect(ALL_FIELDS.map((f) => f.key).sort()).toEqual([...FEATURE_KEYS].sort())
  })

  it('covers every input exactly once across the three steps', () => {
    const keys = STEPS.flatMap((s) => s.fields.map((f) => f.key))
    expect(new Set(keys).size).toBe(19)
  })
})

describe('fieldError', () => {
  const bmi = ALL_FIELDS.find((f) => f.key === 'bmi')!
  const days = ALL_FIELDS.find((f) => f.key === 'phys_unhealthy_days')!
  const smoker = ALL_FIELDS.find((f) => f.key === 'smoker_status')!

  it('requires an answer', () => {
    expect(fieldError(smoker, undefined)).not.toBeNull()
    expect(fieldError(smoker, '   ')).not.toBeNull()
  })

  it('accepts only listed select values', () => {
    expect(fieldError(smoker, '4')).toBeNull()
    expect(fieldError(smoker, '9')).not.toBeNull()
  })

  it('enforces the BMI range (12 to 99.99)', () => {
    expect(fieldError(bmi, '12')).toBeNull()
    expect(fieldError(bmi, '99.99')).toBeNull()
    expect(fieldError(bmi, '11.9')).not.toBeNull()
    expect(fieldError(bmi, '100')).not.toBeNull()
    expect(fieldError(bmi, 'abc')).not.toBeNull()
  })

  it('requires whole numbers from 0 to 30 for day counts', () => {
    expect(fieldError(days, '0')).toBeNull()
    expect(fieldError(days, '30')).toBeNull()
    expect(fieldError(days, '2.5')).not.toBeNull()
    expect(fieldError(days, '31')).not.toBeNull()
    expect(fieldError(days, '-1')).not.toBeNull()
  })
})

describe('toInputs', () => {
  it('produces numbers for all 19 keys when every answer is valid', () => {
    const inputs = toInputs(validAnswers())
    expect(Object.keys(inputs).sort()).toEqual([...FEATURE_KEYS].sort())
    for (const k of FEATURE_KEYS) expect(typeof inputs[k]).toBe('number')
  })

  it('returns keys in the backend order', () => {
    expect(Object.keys(toInputs(validAnswers()))).toEqual([...FEATURE_KEYS])
  })

  it('passes validated answers through with correct numeric values', () => {
    const a = { ...validAnswers(), bmi: '27.5', smoker_status: '3' }
    const inputs = toInputs(a)
    expect(inputs.bmi).toBe(27.5)
    expect(inputs.smoker_status).toBe(3)
  })
})
