// Single source of truth for the 19 model inputs shown in the questionnaire.
// Keys, order, codes and ranges MUST match api/schemas.py and src/config.py in the backend.
// (The backend validates everything again; this file only makes the form friendly.)

export const FEATURE_KEYS = [
  'age_group_5yr',
  'sex_female',
  'bmi',
  'gen_health',
  'phys_unhealthy_days',
  'ment_unhealthy_days',
  'any_physical_activity',
  'hypertension_told',
  'high_cholesterol_told',
  'diabetes_status',
  'asthma_ever',
  'copd_ever',
  'kidney_disease_ever',
  'depression_ever',
  'other_cancer_ever',
  'smoker_status',
  'heavy_drinker',
  'education_level',
  'income_level',
] as const

export type FeatureKey = (typeof FEATURE_KEYS)[number]

/** Answers as numbers, exactly what POST /predict expects. */
export type Inputs = Record<FeatureKey, number>

/** Answers as the form holds them (select values are strings, '' = not answered). */
export type FormAnswers = Partial<Record<FeatureKey, string>>

interface Choice {
  value: number
  label: string
}

export type FieldSpec =
  | { key: FeatureKey; label: string; help: string; kind: 'select'; options: Choice[] }
  | { key: FeatureKey; label: string; help: string; kind: 'number'; min: number; max: number; step: number; unit?: string }

export interface Step {
  title: string
  hint: string
  fields: FieldSpec[]
}

const YES_NO: Choice[] = [
  { value: 0, label: 'No' },
  { value: 1, label: 'Yes' },
]

const AGE: Choice[] = [
  { value: 1, label: '18–24' },
  { value: 2, label: '25–29' },
  { value: 3, label: '30–34' },
  { value: 4, label: '35–39' },
  { value: 5, label: '40–44' },
  { value: 6, label: '45–49' },
  { value: 7, label: '50–54' },
  { value: 8, label: '55–59' },
  { value: 9, label: '60–64' },
  { value: 10, label: '65–69' },
  { value: 11, label: '70–74' },
  { value: 12, label: '75–79' },
  { value: 13, label: '80 or older' },
]

const INCOME: Choice[] = [
  { value: 1, label: 'Less than $15,000' },
  { value: 2, label: '$15,000 to under $25,000' },
  { value: 3, label: '$25,000 to under $35,000' },
  { value: 4, label: '$35,000 to under $50,000' },
  { value: 5, label: '$50,000 to under $100,000' },
  { value: 6, label: '$100,000 to under $200,000' },
  { value: 7, label: '$200,000 or more' },
]

export const STEPS: Step[] = [
  {
    title: 'About you',
    hint: 'Basic demographic information, as it would be recorded in the survey.',
    fields: [
      { key: 'age_group_5yr', label: 'Age range', help: 'Select the band that includes your age', kind: 'select', options: AGE },
      { key: 'sex_female', label: 'Sex', help: 'Sex as recorded in the survey', kind: 'select', options: [
        { value: 0, label: 'Male' },
        { value: 1, label: 'Female' },
      ] },
      { key: 'bmi', label: 'Body mass index (BMI)', help: 'BMI = weight (kg) ÷ height (m)². Enter a number between 12 and 99.99', kind: 'number', min: 12, max: 99.99, step: 0.1, unit: 'kg/m²' },
      { key: 'education_level', label: 'Education', help: 'Highest level of education completed', kind: 'select', options: [
        { value: 1, label: 'Did not graduate high school' },
        { value: 2, label: 'High school graduate' },
        { value: 3, label: 'Attended college or technical school' },
        { value: 4, label: 'Graduated college or technical school' },
      ] },
      { key: 'income_level', label: 'Household income', help: 'Yearly household income, before taxes', kind: 'select', options: INCOME },
    ],
  },
  {
    title: 'Health conditions',
    hint: 'Conditions you have been told about by a health professional.',
    fields: [
      { key: 'hypertension_told', label: 'High blood pressure', help: 'Have you been told you have high blood pressure?', kind: 'select', options: YES_NO },
      { key: 'high_cholesterol_told', label: 'High cholesterol', help: 'Have you been told your cholesterol is high?', kind: 'select', options: YES_NO },
      { key: 'diabetes_status', label: 'Diabetes', help: 'Select the most accurate response', kind: 'select', options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'Yes, but only during pregnancy' },
        { value: 3, label: 'No' },
        { value: 4, label: 'No, pre-diabetes or borderline diabetes' },
      ] },
      { key: 'asthma_ever', label: 'Asthma', help: 'Have you ever been told you have asthma?', kind: 'select', options: YES_NO },
      { key: 'copd_ever', label: 'COPD or chronic bronchitis', help: 'Ever told of COPD, emphysema or chronic bronchitis?', kind: 'select', options: YES_NO },
      { key: 'kidney_disease_ever', label: 'Kidney disease', help: 'Ever told you have kidney disease? (Not counting kidney stones, bladder infection or incontinence)', kind: 'select', options: YES_NO },
      { key: 'depression_ever', label: 'Depression', help: 'Ever told you have a depressive disorder?', kind: 'select', options: YES_NO },
      { key: 'other_cancer_ever', label: 'Cancer (other than skin cancer)', help: 'Ever told you have melanoma or any other cancer? (Non-melanoma skin cancer excluded)', kind: 'select', options: YES_NO },
    ],
  },
  {
    title: 'Lifestyle and wellbeing',
    hint: 'Your habits, activity and how you have felt recently.',
    fields: [
      { key: 'smoker_status', label: 'Smoking', help: 'Which best describes you?', kind: 'select', options: [
        { value: 1, label: 'Smoke every day' },
        { value: 2, label: 'Smoke some days' },
        { value: 3, label: 'Former smoker' },
        { value: 4, label: 'Never smoked' },
      ] },
      { key: 'heavy_drinker', label: 'Heavy drinking', help: 'Men: more than 14 drinks a week. Women: more than 7 drinks a week', kind: 'select', options: YES_NO },
      { key: 'any_physical_activity', label: 'Physical activity', help: 'Any exercise or physical activity outside of work in the past 30 days?', kind: 'select', options: YES_NO },
      { key: 'gen_health', label: 'General health', help: 'How would you rate your general health?', kind: 'select', options: [
        { value: 1, label: 'Excellent' },
        { value: 2, label: 'Very good' },
        { value: 3, label: 'Good' },
        { value: 4, label: 'Fair' },
        { value: 5, label: 'Poor' },
      ] },
      { key: 'phys_unhealthy_days', label: 'Days of poor physical health', help: 'Past 30 days, out of 30 (whole number)', kind: 'number', min: 0, max: 30, step: 1, unit: 'days' },
      { key: 'ment_unhealthy_days', label: 'Days of poor mental health', help: 'Past 30 days, out of 30 (whole number)', kind: 'number', min: 0, max: 30, step: 1, unit: 'days' },
    ],
  },
]

export const ALL_FIELDS: FieldSpec[] = STEPS.flatMap((s) => s.fields)

/** Returns an error message for one field, or null if the answer is valid. */
export function fieldError(spec: FieldSpec, raw: string | undefined): string | null {
  if (raw === undefined || raw.trim() === '') return 'Please answer this question.'
  if (spec.kind === 'select') {
    return spec.options.some((o) => String(o.value) === raw) ? null : 'Please choose an option.'
  }
  const n = Number(raw)
  if (!Number.isFinite(n)) return 'Please enter a number.'
  if (n < spec.min || n > spec.max) return `Enter a value from ${spec.min} to ${spec.max}.`
  if (spec.step === 1 && !Number.isInteger(n)) return 'Enter a whole number.'
  return null
}

/** Converts validated form answers into the exact JSON body for POST /predict. */
export function toInputs(answers: FormAnswers): Inputs {
  const out = {} as Inputs
  // Built in the backend's canonical FEATURE_KEYS order.
  for (const key of FEATURE_KEYS) {
    const raw = answers[key] ?? ''
    out[key] = Number(raw)
  }
  return out
}

/** Find a human-readable label for a stored numeric answer (used on the review screen). */
export function displayValue(spec: FieldSpec, value: number | string | undefined): string {
  if (value === undefined || value === '') return '—'
  if (spec.kind === 'number') return spec.unit ? `${value} ${spec.unit}` : String(value)
  const hit = spec.options.find((o) => String(o.value) === String(value))
  return hit ? hit.label : String(value)
}
