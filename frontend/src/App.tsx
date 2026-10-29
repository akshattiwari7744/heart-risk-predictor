import { predict, type PredictionResult } from './api'
import { ALL_FIELDS, STEPS, displayValue, fieldError, toInputs, type FeatureKey, type FieldSpec, type FormAnswers, type Inputs } from './schema'
import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'

type Page = 'home' | 'assessment' | 'results' | 'simulator' | 'about'

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    heart: <><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21l7.8-7.5 1.1-1.1a5.5 5.5 0 0 0-.1-7.8Z"/><path d="M7 12h2l1.2-3 2.2 6 1.4-3H17"/></>,
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></>,
    spark: <><path d="m12 3-1.4 4.2a2 2 0 0 1-1.3 1.3L5 10l4.3 1.5a2 2 0 0 1 1.3 1.3L12 17l1.4-4.2a2 2 0 0 1 1.3-1.3L19 10l-4.3-1.5a2 2 0 0 1-1.3-1.3L12 3Z"/></>,
    lock: <><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    chart: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function Logo({ onClick }: { onClick: () => void }) {
  return <button className="logo" onClick={onClick} aria-label="Heart Risk Predictor home"><span className="logo-mark"><Icon name="heart" size={21}/></span><span>Heart Risk <b>Predictor</b></span></button>
}

function Button({ children, secondary, onClick, type = 'button' }: { children: ReactNode; secondary?: boolean; onClick?: () => void; type?: 'button' | 'submit' }) {
  return <button type={type} className={secondary ? 'btn btn-secondary' : 'btn btn-primary'} onClick={onClick}>{children}</button>
}

function Header({ navigate, page, hasResult }: { navigate: (p: Page) => void; page: Page; hasResult: boolean }) {
  const [open, setOpen] = useState(false)
  const [homeSection, setHomeSection] = useState<'home'|'how'|'faq'>('home')
  const go = (p: Page) => { navigate(p); setOpen(false) }
  return <header className="header"><div className="nav-shell"><Logo onClick={() => go('home')}/><nav className={open ? 'nav-links open' : 'nav-links'}>
    <button className={page === 'home' && homeSection === 'home' ? 'active' : ''} onClick={() => { setHomeSection('home'); go('home') }}>Home</button>
    <button className={page === 'home' && homeSection === 'how' ? 'active' : ''} onClick={() => { setHomeSection('how'); go('home'); setTimeout(() => document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' }), 10) }}>How It Works</button>
    <button className={page === 'about' ? 'active' : ''} onClick={() => go('about')}>About</button>
    <button className={page === 'home' && homeSection === 'faq' ? 'active' : ''} onClick={() => { setHomeSection('faq'); go('home'); setTimeout(() => document.getElementById('faq')?.scrollIntoView({ behavior: 'smooth' }), 10) }}>FAQ</button>
  </nav><div className="nav-action">{hasResult && page !== 'results' ? <button className="result-return" onClick={() => go('results')}><Icon name="chart" size={16}/> View last result</button> : <Button onClick={() => go('assessment')}>Start Assessment <Icon name="arrow" size={16}/></Button>}</div>
  <button className="menu-button" onClick={() => setOpen(!open)} aria-label="Toggle menu"><Icon name="menu"/></button></div></header>
}

function Footer({ navigate }: { navigate: (p: Page) => void }) {
  return <footer><div className="footer-grid"><div><Logo onClick={() => navigate('home')}/><p className="footer-copy">An educational research prototype helping people explore heart-health patterns with greater clarity.</p></div><div><h4>Explore</h4><button onClick={() => navigate('assessment')}>Assessment</button><button onClick={() => navigate('simulator')}>What-If Simulator</button><button onClick={() => navigate('about')}>About the model</button></div><div><h4>Information</h4><button onClick={() => navigate('about')}>Privacy</button><button onClick={() => navigate('home')}>FAQs</button></div><div className="footer-note"><Icon name="info"/><p><b>Medical disclaimer</b><br/>This tool is for education and research only. It does not provide a diagnosis or predict a future heart attack.</p></div></div><div className="footer-bottom"><span>© 2025 Heart Risk Predictor</span><span>Educational use only</span></div></footer>
}

function HeartVisual() {
  return <div className="heart-visual" aria-label="Abstract cardiovascular illustration"><div className="visual-grid"></div><div className="orbit orbit-one"></div><div className="orbit orbit-two"></div><svg className="heart-art" viewBox="0 0 520 520" role="img" aria-label="Stylized anatomical heart with flowing cardiovascular lines"><defs><linearGradient id="heartFill" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#f2a08f"/><stop offset="1" stopColor="#d95f59"/></linearGradient><filter id="glow"><feGaussianBlur stdDeviation="12" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><path className="flow-line" d="M-20 300C105 244 127 348 239 294s143-146 301-91"/><path className="flow-line soft" d="M-10 350c130-45 180 44 285-14s119-105 260-87"/><path className="vessel" d="M274 195c-2-48 2-99 35-129 26-23 63-17 74 7 12 27-10 48-32 62-29 18-31 42-31 71"/><path className="vessel" d="M249 198c-22-44-55-65-51-102 4-31 35-52 61-38 24 13 18 43 8 65-11 24-3 45 6 69"/><path fill="url(#heartFill)" d="M269 189c-41-30-103-11-125 40-30 70 25 147 124 221 8 6 19 6 27-1 94-81 147-162 111-229-25-47-85-59-122-25l-8 8-7-14Z" filter="url(#glow)"/><path className="heart-detail" d="M279 215c-20 64-9 137 12 199M244 235c-31 21-50 55-51 91M312 232c39 26 56 55 58 91"/><circle cx="282" cy="310" r="7" fill="#fff"/><circle cx="282" cy="310" r="15" fill="none" stroke="#fff" opacity=".35"/></svg><div className="visual-caption"><span><i></i> Cardiovascular patterns</span><b>Health, seen with greater clarity</b></div></div>
}

function AsyncState({ type, title, message }: { type: 'loading'|'error'|'unavailable'; title: string; message: string }) {
  return <div className={`async-state ${type}`} role={type === 'loading' ? 'status' : 'alert'}>{type === 'loading' ? <span className="loading-ring"></span> : <span className="async-icon"><Icon name="info" size={18}/></span>}<div><b>{title}</b><p>{message}</p></div></div>
}

function Home({ navigate }: { navigate: (p: Page) => void }) {
  const steps = [
    ['01', 'Answer a few questions', 'Share basic health and lifestyle information based on common survey questions.'],
    ['02', 'Review your results', 'See a clear visual summary of the model’s estimated probability.'],
    ['03', 'Explore what-if scenarios', 'Compare supported inputs without treating model changes as causal effects.'],
  ]
  return <main>
    <section className="hero"><div className="hero-copy"><div className="eyebrow"><span></span> Heart-health awareness, made clearer</div><h1>Understand Your Heart.<br/><em>Take Charge of Your Health.</em></h1><p className="hero-lede">Explore how a machine-learning model can present an educational estimate of future heart-disease likelihood based on health questionnaire patterns. This prototype is not a diagnosis or clinical risk calculator.</p><div className="hero-actions"><Button onClick={() => navigate('assessment')}>Start Assessment <Icon name="arrow" size={18}/></Button><Button secondary onClick={() => document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' })}>How It Works</Button></div><div className="trust-row"><span><Icon name="shield" size={17}/> Educational prototype</span><span><Icon name="spark" size={17}/> Model-informed</span><span><Icon name="info" size={17}/> Not medical advice</span></div></div>
      <div className="hero-visual"><HeartVisual/><div className="floating-card fc-top"><span className="pulse"><i></i></span><div><small>Designed for clarity</small><b>Simple, guided experience</b></div></div><div className="floating-card fc-bottom"><span className="mini-icon"><Icon name="shield" size={20}/></span><div><small>Thoughtful experience</small><b>Clear and transparent</b></div></div><div className="coral-dot"></div></div>
    </section>
    <section className="product-preview"><div className="preview-copy"><div className="section-label">One connected experience</div><h2>From thoughtful questions<br/>to clearer understanding.</h2><p>Move through a focused assessment, understand the model output, then compare supported inputs in the What-If Simulator.</p><Button secondary onClick={() => navigate('assessment')}>See the experience <Icon name="arrow" size={17}/></Button></div><div className="preview-window"><div className="preview-nav"><span></span><i></i><i></i><b>Assessment overview</b></div><div className="preview-body"><div className="preview-progress"><i></i></div><div className="preview-content"><small>PERSONAL PROFILE</small><h3>Tell us a little about you</h3><div className="preview-field"></div><div className="preview-field short"></div><div className="preview-result"><span>Model output</span><b>Clearly explained</b></div></div></div><div className="preview-tabs"><span>01 Assessment</span><span>02 Results</span><span>03 What-If</span></div></div></section>
    <section className="intro" id="how"><div className="section-label">How it works</div><div className="section-heading"><h2>A clearer picture in<br/>three thoughtful steps.</h2><p>No jargon. No overwhelming dashboards. Just a guided experience designed to help you explore health patterns with confidence.</p></div><div className="steps">{steps.map(([n,t,d]) => <article className="step-card" key={n}><span className="step-num">{n}</span><div className="step-line"></div><h3>{t}</h3><p>{d}</p><span className="round-arrow"><Icon name="arrow" size={17}/></span></article>)}</div></section>
    <section className="values"><div className="values-copy"><div className="section-label light">Thoughtful by design</div><h2>Built around what matters: clarity, care, and context.</h2><p>Health information can feel complex. We make each step calm, approachable, and transparent—without overstating what a research model can tell you.</p><Button secondary onClick={() => navigate('about')}>Our approach <Icon name="arrow" size={17}/></Button></div><div className="value-list">
      <article><span><Icon name="spark"/></span><div><h3>Refreshingly simple</h3><p>Plain-language questions and results designed for everyday understanding.</p></div></article>
      <article><span><Icon name="lock"/></span><div><h3>Privacy-conscious</h3><p>Answers go to the prediction service once to produce your result and are not stored or logged. See the About page.</p></div></article>
      <article><span><Icon name="chart"/></span><div><h3>Awareness, not diagnosis</h3><p>A research-informed view of patterns—not medical advice or a clinical conclusion.</p></div></article>
    </div></section>
    <section className="cta"><div><div className="section-label light">Ready when you are</div><h2>Take the first step toward<br/>better health awareness.</h2></div><Button onClick={() => navigate('assessment')}>Start your assessment <Icon name="arrow"/></Button></section>
    <section className="faq" id="faq"><div><div className="section-label">Common questions</div><h2>Good questions deserve clear answers.</h2></div><div className="faq-list"><details><summary>Is this a medical diagnosis?<span>+</span></summary><p>No. This is an educational research prototype and cannot diagnose any condition.</p></details><details><summary>Does it predict a future heart attack?<span>+</span></summary><p>No. It estimates likelihood of reporting existing heart disease based on survey patterns.</p></details><details><summary>What does the simulator prove?<span>+</span></summary><p>It only compares model outputs. Changes do not establish causation or guarantee a health outcome.</p></details></div></section>
  </main>
}


function ProbabilityGauge({ probability, label }: { probability: number | null; label: string }) {
  const pct = probability === null ? 0 : Math.max(0, Math.min(100, probability * 100))
  return <div className={`score-ring ${probability === null ? 'pending' : ''}`} style={{'--gauge-value': `${pct * 3.6}deg`} as CSSProperties}><div><strong>{probability === null ? '—' : (probability * 100).toFixed(1)}{probability !== null && <small>%</small>}</strong><span>{label}</span></div></div>
}

const pct = (p: number) => `${(p * 100).toFixed(1)}%`

function FieldControl({ spec, value, onChange }: { spec: FieldSpec; value: string | undefined; onChange: (v: string) => void }) {
  if (spec.kind === 'select') {
    return <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
      <option value="">Choose an option</option>
      {spec.options.map((o) => <option key={o.value} value={String(o.value)}>{o.label}</option>)}
    </select>
  }
  return <input type="number" inputMode="decimal" min={spec.min} max={spec.max} step={spec.step} value={value ?? ''} placeholder={`${spec.min}–${spec.max}`} onChange={(e) => onChange(e.target.value)} />
}

function Assessment({ navigate, initialAnswers, onSubmit, submitting, submitError }: {
  navigate: (p: Page) => void
  initialAnswers: FormAnswers
  onSubmit: (a: FormAnswers) => void
  submitting: boolean
  submitError: string | null
}) {
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState<FormAnswers>(initialAnswers)
  const [showErrors, setShowErrors] = useState(false)
  const totalSteps = STEPS.length + 1
  const isReview = step === STEPS.length
  const current = STEPS[step]
  const stepComplete = isReview || current.fields.every((f) => fieldError(f, answers[f.key]) === null)
  const allValid = ALL_FIELDS.every((f) => fieldError(f, answers[f.key]) === null)
  const next = () => {
    if (!stepComplete) return setShowErrors(true)
    setShowErrors(false)
    setStep(step + 1)
  }
  const back = () => {
    setShowErrors(false)
    if (step === 0) navigate('home')
    else setStep(step - 1)
  }
  const submit = () => {
    if (!allValid) return setShowErrors(true)
    onSubmit(answers)
  }
  return <main className="assessment-page"><div className="assessment-shell"><button className="back-link" onClick={() => navigate('home')}>← Back to home</button><div className="assessment-top"><div><div className="eyebrow"><span></span> Educational assessment</div><h1>Let’s get to know your health.</h1><p>Answer each question as accurately as you can. The answers are sent once to the prediction service to produce your result.</p></div><div className="progress-copy"><b>Step {step + 1} of {totalSteps}</b><span>{Math.round(((step + 1) / totalSteps) * 100)}% complete</span></div></div><div className="progress"><i style={{width: `${((step + 1) / totalSteps) * 100}%`}}></i></div>
    <div className="form-card">{!isReview ? <>
      <div className="form-heading"><span className="form-icon"><Icon name={step === 0 ? 'heart' : step === 1 ? 'shield' : 'spark'}/></span><div><h2>{current.title}</h2><p>{current.hint}</p></div></div>
      <div className="fields">{current.fields.map((spec) => {
        const err = fieldError(spec, answers[spec.key])
        return <label key={spec.key}><span>{spec.label}</span><small>{spec.help}</small>
          <FieldControl spec={spec} value={answers[spec.key]} onChange={(v) => { setAnswers({ ...answers, [spec.key]: v }); setShowErrors(false) }}/>
          {showErrors && err && <em className="field-error">{err}</em>}
        </label>
      })}</div>
      {showErrors && !stepComplete && <p className="validation">Please answer each question before continuing.</p>}
    </> : <>
      <div className="form-heading"><span className="form-icon"><Icon name="check"/></span><div><h2>Review your answers</h2><p>Check these before submitting. Use Previous to change any step.</p></div></div>
      <div className="review-grid">{ALL_FIELDS.map((spec) => <div key={spec.key}><span>{spec.label}</span><b>{displayValue(spec, answers[spec.key])}</b></div>)}</div>
      {showErrors && !allValid && <p className="validation">Some answers are missing or invalid. Use Previous to find them.</p>}
    </>}
    {submitError && <AsyncState type="error" title="We couldn’t generate your result." message={submitError}/>}
    <div className="form-actions"><Button secondary onClick={back}>Previous</Button><Button onClick={() => isReview ? submit() : next()}>{isReview ? 'Submit Assessment' : 'Continue'} <Icon name="arrow" size={17}/></Button></div></div>
    {submitting && <AsyncState type="loading" title="Preparing your results" message="Sending your answers to the prediction service. If it has been idle, the first request can take up to a minute."/>}
    <p className="secure-note"><Icon name="info" size={15}/> Your answers are not stored in your browser. See the About page for what the service logs.</p></div></main>
}

function Results({ result, navigate, onStartNew }: { result: PredictionResult; navigate: (p: Page) => void; onStartNew: () => void }) {
  const guidance = [['Physical activity','Explore regular movement appropriate for your abilities.'],['Tobacco avoidance','Avoid tobacco and seek professional support if you want help stopping.'],['Balanced nutrition','Choose a varied pattern with vegetables, fruit, whole grains, and appropriate portions.'],['Routine checks','Ask a healthcare professional when blood pressure and cholesterol checks are appropriate.'],['Professional follow-up','Discuss personal questions or concerns with a qualified healthcare professional.']]
  const above = result.above_threshold
  return <main className="results-page"><div className="results-shell">
    <button className="result-return" onClick={() => navigate('home')}>← Back to home</button>
    <div className="results-heading"><div><div className="eyebrow"><span></span> Your results</div><h1>Your heart-health snapshot</h1><p className="result-intro">A simple explanation of your model estimate, with practical information to explore next.</p></div><span className="result-status"><Icon name="check" size={15}/> Assessment complete</span></div>
    <div className="urgent-note"><Icon name="heart" size={18}/><p><b>Having urgent symptoms?</b> Seek immediate medical care for chest pain, severe shortness of breath, fainting, or other potentially serious symptoms. Do not rely on this tool.</p></div>
    <section className="score-card result-score-card"><ProbabilityGauge probability={result.estimated_probability} label="Model estimate"/><div className="score-copy"><span className="result-pill">{above ? 'At or above the model threshold' : 'Below the model threshold'}</span><h2>What this estimate means</h2><p>{result.interpretation}</p><p className="result-context">This estimates the likelihood of reporting existing heart disease in the survey data. It is not a diagnosis or a prediction of a future heart attack.</p></div></section>
    <section className="guidance"><div className="guidance-heading"><div><div className="section-label">Small steps, useful awareness</div><h2>Everyday heart-health foundations</h2></div><p>General educational information—not personal treatment advice.</p></div><div className="guidance-list">{guidance.map(([title,text]) => <article key={title}><span><Icon name="check" size={16}/></span><div><b>{title}</b><p>{text}</p></div></article>)}</div></section>
    <section className="simulator-callout"><div><div className="section-label light">Explore the model</div><h2>Curious how a different scenario compares?</h2><p>Try changing selected inputs to compare model estimates.</p></div><Button onClick={() => navigate('simulator')}>Explore scenarios <Icon name="arrow"/></Button></section>
    <div className="disclaimer"><Icon name="info" size={24}/><p><b>Educational prototype only.</b> {result.disclaimer}</p></div>
    <div className="result-actions"><Button onClick={() => navigate('simulator')}>Explore scenarios</Button><Button secondary onClick={() => navigate('assessment')}>Edit answers</Button><button className="text-action" onClick={onStartNew}>Start new assessment</button></div>
  </div></main>
}

const SIM_CONTROLS: { key: FeatureKey; label: string; options: { value: string; label: string }[] }[] = [
  { key: 'any_physical_activity', label: 'Physical activity (past 30 days)', options: [{ value: '1', label: 'Yes' }, { value: '0', label: 'No' }] },
  { key: 'smoker_status', label: 'Smoking', options: [{ value: '1', label: 'Smoke every day' }, { value: '2', label: 'Smoke some days' }, { value: '3', label: 'Former smoker' }, { value: '4', label: 'Never smoked' }] },
  { key: 'heavy_drinker', label: 'Heavy drinking', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }] },
]

function Simulator({ navigate, baseline }: {
  navigate: (p: Page) => void
  baseline: { inputs: Inputs; result: PredictionResult } | null
}) {
  const [scenario, setScenario] = useState<Partial<Record<FeatureKey, string>>>({})
  const [scenarioResult, setScenarioResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  if (!baseline) return <main className="simulator-page"><div className="simulator-empty"><span className="empty-visual"><Icon name="chart" size={30}/></span><div className="eyebrow"><span></span> What-If Simulator</div><h1>Complete an assessment first.</h1><p>The simulator compares a changed scenario against your submitted answers. Complete an assessment to use it.</p><Button onClick={() => navigate('assessment')}>Start Assessment <Icon name="arrow" size={17}/></Button><button className="text-action" onClick={() => navigate('home')}>Back to home</button></div></main>
  const changedKeys = Object.keys(scenario).filter((k) => scenario[k as FeatureKey])
  const compare = async () => {
    if (changedKeys.length === 0) return setNote('Change at least one input to compare.')
    const overrides: Partial<Inputs> = {}
    for (const k of changedKeys) overrides[k as FeatureKey] = Number(scenario[k as FeatureKey])
    setLoading(true); setError(null); setNote(null)
    try {
      const r = await predict({ ...baseline.inputs, ...overrides })
      setScenarioResult(r)
    } catch (e) {
      setScenarioResult(null)
      setError(e instanceof Error ? e.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }
  const reset = () => { setScenario({}); setScenarioResult(null); setError(null); setNote(null) }
  const diff = scenarioResult ? (scenarioResult.estimated_probability - baseline.result.estimated_probability) * 100 : null
  return <main className="simulator-page"><div className="simulator-shell"><button className="back-link" onClick={() => navigate('results')}>← Back to your result</button><div className="simulator-hero"><div><div className="eyebrow"><span></span> What-If Simulator</div><h1>Compare a different scenario.</h1><p>Change a habit below to get a second model estimate. Your submitted answers stay unchanged.</p></div><div className="scenario-status"><span></span><div><small>Current session</small><b>{changedKeys.length ? 'Unsaved changes' : 'Baseline loaded'}</b></div></div></div>
    <div className="simulator-layout"><section className="scenario-controls"><div className="panel-heading"><div><span>02</span><h2>Alternative scenario</h2></div><button onClick={reset}>Reset scenario</button></div><p className="panel-help">Leave a control on “Keep my answer” to use your submitted value.</p>
      <div className="sim-fields">{SIM_CONTROLS.map((c) => <label key={c.key}><span>{c.label}</span><small>Scenario input</small><select value={scenario[c.key] ?? ''} onChange={(e) => { setScenario({ ...scenario, [c.key]: e.target.value }); setScenarioResult(null) }}><option value="">Keep my answer</option>{c.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select></label>)}</div>
      <Button onClick={compare}>{loading ? 'Calculating…' : 'Compare Scenarios'} <Icon name="arrow" size={17}/></Button>
      {note && <p className="service-message"><Icon name="info" size={16}/> {note}</p>}
      {error && <AsyncState type="error" title="Scenario could not be calculated." message={error}/>}</section>
      <section className="comparison-panel"><div className="panel-heading"><div><span>01</span><h2>Scenario comparison</h2></div><small>Model estimates</small></div>
        <div className="scenario-gauge"><div className="empty-ring"><span>{pct(baseline.result.estimated_probability)}</span></div><div><small>Submitted baseline</small><b>{pct(baseline.result.estimated_probability)}</b><p>{baseline.result.above_threshold ? 'At or above' : 'Below'} the decision threshold ({pct(baseline.result.decision_threshold)}).</p></div></div>
        <div className="scenario-gauge"><div className="empty-ring"><span>{scenarioResult ? pct(scenarioResult.estimated_probability) : '—'}</span></div><div><small>Alternative scenario</small><b>{scenarioResult ? pct(scenarioResult.estimated_probability) : 'Awaiting comparison'}</b><p>{scenarioResult ? `${scenarioResult.above_threshold ? 'At or above' : 'Below'} the decision threshold.` : 'Press Compare Scenarios to calculate.'}</p></div></div>
        <div className="difference"><span>Difference</span><b>{diff === null ? '— percentage points' : `${diff >= 0 ? '+' : ''}${diff.toFixed(1)} percentage points`}</b><small>Shown only when both estimates are available.</small></div></section></div>
    <section className="change-summary"><div><h2>What changed</h2><p>{changedKeys.length ? `${changedKeys.length} input${changedKeys.length > 1 ? 's' : ''} changed from your answers.` : 'Change an input to create an alternative scenario.'}</p></div></section>
    <div className="disclaimer"><Icon name="info" size={24}/><p><b>Association is not causation.</b> A change in model output does not prove that changing an input would change a real-world health outcome or improve anyone’s health. This simulator is educational and is not medical advice.</p></div></div></main>
}

function About() {
  const sections: [string, string][] = [
    ['Dataset', 'CDC Behavioral Risk Factor Surveillance System (BRFSS) 2025 (LLCP2025): a telephone survey of US adults, with self-reported answers. 352,145 respondents had a valid outcome. About 9% of them reported existing coronary heart disease or a heart attack.'],
    ['Methodology', 'The 19 survey answers go into a logistic regression, calibrated with isotonic regression so the output behaves like a probability. The decision threshold (about 10%) was chosen on validation data to flag at least 80% of reported cases. Data were split 70/15/15 into training, validation and test sets (246,501 / 52,822 / 52,822 people).'],
    ['Model evaluation', 'On the held-out test set: PR-AUC 0.328 against a 0.090 baseline (the rate of reported disease), ROC-AUC 0.836. At the threshold: recall 0.789, precision 0.226, specificity 0.733. Most people flagged do not have reported disease, so precision is low by design. See the project report for confidence intervals and subgroup results.'],
    ['Limitations', 'The outcome and most inputs are self-reported, so the data contain recall and misreporting bias. Some inputs, such as general health and blood pressure, can be consequences of heart disease rather than causes. The model covers US adults and may not transfer to other populations, including people in India. It does not predict a future heart attack and does not establish causation.'],
  ]
  return <main className="about-page"><section className="about-hero"><div className="section-label light">About the model</div><h1>Better context for<br/>health awareness.</h1><p>Heart Risk Predictor is an educational prototype designed to make machine-learning outputs easier to understand—without presenting them as medical truth.</p></section><section className="about-content"><div><div className="section-label">Transparency</div><h2>Designed for learning, not diagnosis.</h2></div><div className="about-prose"><p>The model estimates how likely a person with these answers is to <b>report</b> existing coronary heart disease or a past heart attack. It does not predict a future heart attack.</p><div className="model-sections">{sections.map(([title,text]) => <article key={title}><h3>{title}</h3><p>{text}</p></article>)}</div><div className="limitation"><Icon name="info"/><p><b>Disclaimer</b><br/>This prototype does not provide medical advice, diagnosis, treatment, or emergency support. A model output cannot replace evaluation by a qualified healthcare professional.</p></div><h3>Privacy and logging</h3><p>Your answers are sent to the prediction service once, to produce your result. The service does not store answers and does not log their values; it logs a request ID, the response status, timing, and whether the result was above or below the threshold. This website stores nothing in browser storage. Hosting providers may keep their own standard server access logs.</p></div></section></main>
}

export default function App() {
  const [page, setPage] = useState<Page>('home')
  const [answers, setAnswers] = useState<FormAnswers>({})
  const [baseline, setBaseline] = useState<{ inputs: Inputs; result: PredictionResult } | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const hasResult = baseline !== null
  const navigate = (p: Page) => { setPage(p); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const submitAssessment = async (a: FormAnswers) => {
    setSubmitting(true)
    setSubmitError(null)
    try {
      const inputs = toInputs(a)
      const result = await predict(inputs)
      setAnswers(a)
      setBaseline({ inputs, result })
      setSubmitting(false)
      navigate('results')
    } catch (e) {
      setSubmitting(false)
      setSubmitError(e instanceof Error ? e.message : 'Unexpected error')
    }
  }
  const startNew = () => { setAnswers({}); setBaseline(null); setSubmitError(null); navigate('assessment') }
  useEffect(() => { document.title = 'Heart Risk Predictor' }, [])
  return <><Header navigate={navigate} page={page} hasResult={hasResult}/>
    {page === 'home' && <Home navigate={navigate}/>}
    {page === 'assessment' && <Assessment key={hasResult ? 'edit' : 'new'} navigate={navigate} initialAnswers={answers} onSubmit={submitAssessment} submitting={submitting} submitError={submitError}/>}
    {page === 'results' && baseline && <Results result={baseline.result} navigate={navigate} onStartNew={startNew}/>}
    {page === 'results' && !baseline && <Assessment key="new" navigate={navigate} initialAnswers={{}} onSubmit={submitAssessment} submitting={submitting} submitError={submitError}/>}
    {page === 'simulator' && <Simulator navigate={navigate} baseline={baseline}/>}
    {page === 'about' && <About/>}
    <Footer navigate={navigate}/></>
}
