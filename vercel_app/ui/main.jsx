import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const repo = 'https://github.com/trism-dev/heart-disease-prediction';
const fields = [
  ['age', 'Age', 'years'],
  ['sex', 'Sex', '', [[1, 'Male'], [0, 'Female']]],
  ['cp', 'Chest pain type', '', [[1, 'Typical angina'], [2, 'Atypical angina'], [3, 'Non-anginal pain'], [4, 'No symptoms']]],
  ['trestbps', 'Resting blood pressure', 'mm Hg'],
  ['chol', 'Cholesterol', 'mg/dL'],
  ['fbs', 'Fasting blood sugar > 120 mg/dL', '', [[0, 'No'], [1, 'Yes']]],
  ['restecg', 'Resting ECG', '', [[0, 'Normal'], [1, 'ST-T abnormality'], [2, 'LV hypertrophy']]],
  ['thalach', 'Maximum heart rate', 'bpm'],
  ['exang', 'Exercise-induced angina', '', [[0, 'No'], [1, 'Yes']]],
  ['oldpeak', 'ST depression', 'mm'],
  ['slope', 'ST slope', '', [[1, 'Upsloping'], [2, 'Flat'], [3, 'Downsloping']]],
  ['ca', 'Vessels colored', '', [[0, '0'], [1, '1'], [2, '2'], [3, '3'], [null, 'Unknown']]],
  ['thal', 'Thallium test', '', [[3, 'Normal'], [6, 'Fixed defect'], [7, 'Reversible defect'], [null, 'Unknown']]],
];
const names = Object.fromEntries(fields.map(([key, name]) => [key, name]));
const examples = [
  { name: 'Example 1 · 63-year-old man', values: { age: 63, sex: 1, cp: 1, trestbps: 145, chol: 233, fbs: 1, restecg: 2, thalach: 150, exang: 0, oldpeak: 2.3, slope: 3, ca: 0, thal: 6 } },
  { name: 'Example 2 · 67-year-old man', values: { age: 67, sex: 1, cp: 4, trestbps: 160, chol: 286, fbs: 0, restecg: 2, thalach: 108, exang: 1, oldpeak: 1.5, slope: 2, ca: 3, thal: 3 } },
  { name: 'Example 3 · 41-year-old woman', values: { age: 41, sex: 0, cp: 2, trestbps: 130, chol: 204, fbs: 0, restecg: 2, thalach: 172, exang: 0, oldpeak: 1.4, slope: 1, ca: 0, thal: 3 } },
];
const pct = p => p >= .995 ? '>99%' : p < .005 ? '<1%' : `${Math.round(p * 100)}%`;

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(url, { ...options, signal: options.signal || controller.signal });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error('Unable to complete the request.'), { errors: data.errors });
    return data;
  } finally { clearTimeout(timeout); }
}

function App() {
  const [model, setModel] = useState(null);
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({});
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setError('');
    fetchJson('/api/model').then(data => {
      if (active) { setModel(data); setInputs(data.reference_values); }
    }).catch(() => { if (active) setError('The model could not load. Please try again.'); });
    return () => { active = false; };
  }, [retry]);
  const update = values => { setInputs(values); setResult(null); setFieldErrors({}); setError(''); };
  async function submit(event) {
    event.preventDefault();
    setBusy(true); setResult(null); setError(''); setFieldErrors({});
    try { setResult(await fetchJson('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(inputs) })); }
    catch (e) {
      setFieldErrors(e.errors || {});
      setError(e.errors ? 'Please check the highlighted fields.' : 'Prediction unavailable. Please try again; the model may be starting up.');
    } finally { setBusy(false); }
  }
  return <>
    <a className="skip" href="#main">Skip to content</a>
    <header className="topbar"><a className="brand" href="/" aria-label="Heart Disease Prediction home"><svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="currentColor" /><path d="M5 17h6l3-8 4 15 3-7h6" fill="none" stroke="white" strokeWidth="2" /></svg><span>Heart Disease Prediction<span className="brand-sub">An educational model demo</span></span></a><a className="source" href={repo} target="_blank" rel="noreferrer">View source <span aria-hidden="true">↗</span></a></header>
    <main id="main" className="container">
      <section className="intro"><p className="eyebrow">Heart disease classification</p><h1>Heart disease prediction</h1><p className="lead">Logistic-regression model trained on UCI Cleveland data. Predictions use a fixed pipeline, a recall-focused threshold, and feature contributions.</p></section>
      <aside className="notice"><strong>Educational demo, not a diagnostic tool.</strong><span>Trained on 303 patients from one clinic using 1980s data. Do not use predictions for clinical decisions. Use the example profiles to explore the model.</span></aside>
      <nav className="page-nav" aria-label="Page sections"><a href="#explore">Prediction</a><a href="#evaluation">Evaluation & limitations</a></nav>
      <section id="explore" className="workspace" aria-label="Prediction form and explanation">
        <div className="panel form-panel"><div className="panel-heading"><p className="eyebrow">01 · Input profile</p><h2>Patient characteristics</h2><p>Use an example profile or enter values within the training ranges.</p></div>
          {!model ? <div role="status">{error ? <><p role="alert">{error}</p><button className="primary" onClick={() => setRetry(r => r + 1)}>Retry loading model</button></> : 'Loading the frozen model…'}</div> : <>
            <div className="examples" aria-label="Example profiles">{examples.map(example => <button key={example.name} disabled={busy} onClick={() => update(example.values)}>{example.name}</button>)}<button disabled={busy} onClick={() => update(model.reference_values)}>Reset to typical values</button></div>
            <form onSubmit={submit}>
              <fieldset disabled={busy}><legend className="sr-only">Model inputs</legend><div className="fields">{fields.map(([key, label, unit, options]) => <div className="field" key={key}>
                <label htmlFor={key}>{label}{unit && <span className="unit">{unit}</span>}</label>
                {options ? <select id={key} value={inputs[key] === null ? 'unknown' : inputs[key]} onChange={e => update({ ...inputs, [key]: e.target.value === 'unknown' ? null : Number(e.target.value) })} aria-invalid={!!fieldErrors[key]} aria-describedby={key + '-help'}>{options.map(([value, text]) => <option key={value ?? 'unknown'} value={value ?? 'unknown'}>{text}</option>)}</select> : <input id={key} type="number" required min={model.input_ranges[key][0]} max={model.input_ranges[key][1]} step={key === 'oldpeak' ? '.1' : '1'} value={inputs[key] ?? ''} onChange={e => update({ ...inputs, [key]: e.target.value === '' ? '' : Number(e.target.value) })} aria-invalid={!!fieldErrors[key]} aria-describedby={key + '-help'} />}
                <p id={key + '-help'} className={fieldErrors[key] ? 'field-error' : 'hint'}>{fieldErrors[key] || (options ? key === 'ca' || key === 'thal' ? 'Unknown values use the fitted imputer.' : '\u00a0' : `Training range: ${model.input_ranges[key][0]}–${model.input_ranges[key][1]}`)}</p>
              </div>)}</div></fieldset>
              {error && <p className="field-error" role="alert">{error}</p>}
              <button className="primary calculate" disabled={busy} type="submit">{busy ? 'Calculating…' : 'Calculate model score'}<span aria-hidden="true">→</span></button>
            </form>
          </>}
        </div>
        <div className="panel result-panel" aria-busy={busy}><div className="panel-heading"><p className="eyebrow">02 · Model response</p><h2>Score & explanation</h2><p>Recalibrated score, classification threshold, and feature contributions.</p></div>
          <div aria-live="polite" aria-atomic="true">{!result ? <div className="empty"><div className="empty-icon" aria-hidden="true">＋</div><h3>{busy ? 'Running the saved pipeline' : 'No prediction calculated'}</h3><p>{busy ? 'Applying the same preprocessing and threshold used in the original app.' : 'Submit the input profile to calculate a score.'}</p></div> : <>
            <div className={'score-card ' + (result.elevated ? 'elevated' : 'lower')}><span className="score-label">Recalibrated model score</span><div className="score-value">{pct(result.probability)}</div><h3>{result.label}</h3><p>Decision threshold: {Math.round(result.threshold * 100)}%</p></div>
            <div className="score-track" aria-hidden="true"><div style={{ width: result.probability * 100 + '%' }} className={result.elevated ? 'track-elevated' : 'track-lower'} /><span style={{ left: result.threshold * 100 + '%' }} /></div><div className="track-labels"><span>0%</span><span>Threshold {Math.round(result.threshold * 100)}%</span><span>100%</span></div>
            <p className="explanation">The low threshold prioritizes recall and increases false alarms. This score is recalibrated to Cleveland’s 46% disease prevalence; it is not a general estimate of someone’s chance of disease.</p>
            {result.edge_fields.length > 0 && <p className="edge-note">At the edge of the training range: {result.edge_fields.map(f => names[f]).join(', ')}.</p>}
            <h3 className="drivers-title">Feature contributions</h3><p className="hint">Effect on log-odds versus a typical training patient. Blue raises the score; teal lowers it.</p>
            <div className="drivers">{Object.entries(result.contributions).sort((a, b) => b[1] - a[1]).map(([key, value]) => { const maximum = Math.max(...Object.values(result.contributions).map(Math.abs), .001); return <div className="driver" key={key}><span>{names[key]}</span><div className="driver-track" aria-hidden="true"><i className={value >= 0 ? 'positive' : 'negative'} style={{ width: Math.abs(value) / maximum * 48 + '%', left: value >= 0 ? '50%' : (50 - Math.abs(value) / maximum * 48) + '%' }} /></div><span className="driver-number">{value > 0 ? '+' : ''}{value.toFixed(2)}</span></div>; })}</div>
            <p className="explanation"><strong>Largest contributions:</strong> {result.top.map(([key, value]) => `${names[key]} (${value > 0 ? 'raises' : 'lowers'})`).join(', ')}.</p>
          </>}</div>
        </div>
      </section>
      <section id="evaluation" className="evaluation"><div><p className="eyebrow">03 · Model evaluation</p><h2>Evaluation and limitations</h2><p className="lead">Results from 61 held-out Cleveland patients and validation on other UCI sites.</p></div>
        {model && <><div className="metrics"><div><span>Held-out test recall</span><strong>{model.test_metrics.recall.toFixed(3)}</strong><small>95% CI {model.test_metrics.recall_ci.map(v => v.toFixed(2)).join('–')} · 27 of 28 disease cases identified</small></div><div><span>Held-out precision</span><strong>{model.test_metrics.precision.toFixed(3)}</strong><small>95% CI {model.test_metrics.precision_ci.map(v => v.toFixed(2)).join('–')}</small></div><div><span>Cross-validated precision</span><strong>{model.cv_metrics.repeated.precision_at_recall.mean.toFixed(3)}</strong><small>± {model.cv_metrics.repeated.precision_at_recall.std.toFixed(3)} · 5×5 repeated CV</small></div></div><div className="model-notes"><article><h3>Model selection</h3><p>Logistic regression achieved higher cross-validated precision at the recall target than random forest and XGBoost. Model selection and the recall-focused threshold use training data only; the test set was evaluated once.</p></article><article><h3>Known limitations</h3><p>The Cleveland test set has only 61 patients. Pooled recall fell to 0.70 on 617 patients from other UCI sites. Cross-validated recall was lower for women, with only 18 women with disease in training.</p></article><article><h3>Inference implementation</h3><p>Runs the committed pipeline without retraining, a database, or prediction storage. Feature effects explain the model’s score; they do not establish medical causation.</p></article></div></>}
        <a className="report-link" href={repo + '/blob/main/reports/model_comparison.md'} target="_blank" rel="noreferrer">Full evaluation report <span aria-hidden="true">↗</span></a>
      </section>
      <footer><p>Built by <a href="https://portfolio-site-trism2.vercel.app/">Tristan Morton</a> · React & Python · Educational use only</p><p>Data: <a href="https://archive.ics.uci.edu/dataset/45/heart-disease">UCI Heart Disease</a>, CC BY 4.0.</p></footer>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App />);
