/* Used-car price with a neural network — the whole page.
 *
 *   1 Data       a dataset (synthetic Spain, real US listings, or your own CSV)
 *   2 Variables  the price column and the features
 *   3 Network    layers, activations, regularisation, optimiser
 *   4 Train      live, in the browser, with early stopping
 *   5 Results    against a mean baseline and a linear regression on the same features
 *   6 My car     a price for a car you describe, with an honest range and similar cars
 *
 * The configuration lives in the URL (#cfg=…), so a lecturer can send a link
 * that opens exactly the network the class should train. */
import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import { parseCSV, profile, clean, split, fitEncoder, encodeRow, yToModel, yFromModel, toNum } from "./lib/data.js";
import { fitLinear, metrics, errorBand, fitKNN, tuneK } from "./lib/models.js";
import { buildModel, train, predict, permutationImportance, paramCount, PRESETS, ACTIVATIONS, OPTIMIZERS } from "./lib/nn.js";
import { LossChart, ScatterChart, Bars } from "./charts.jsx";

const BASE = import.meta.env.BASE_URL;
const DATASETS = {
  spain: {
    file: "data/spain_synthetic.csv", name: "Spain · synthetic market (6,000 cars)", target: "price_eur", unit: "€",
    note: "Generated with a documented, non-linear pricing rule (scraper/synthetic_spain.py): exponential depreciation that is faster for premium brands and electric cars, a km penalty that saturates, diesel B-label cars losing value in Madrid and Barcelona. No open dataset of Spanish listings exists — the portals' terms forbid scraping them — and a known rule lets you check what the network learned.",
  },
  us: {
    file: "data/us_craigslist_sample.csv", name: "USA · real Craigslist listings (8,000)", target: "price_usd", unit: "$",
    note: "A random sample of real listings: “Craigslist Used Cars and Trucks: EDA” (Hugging Face, Yoad22), CC-BY-4.0, from Austin Reese's Craigslist data on Kaggle. Asking prices in US dollars, odometer in miles. Real data is noisier than any rule: expect a lower R².",
  },
};

const DEFAULT_CFG = {
  dataset: "spain", logTarget: false, testShare: 0.2, seed: 7, minCount: 10,
  layers: PRESETS.medium.layers, dropout: 0.1, l2: 0.0001, optimizer: "adam", lr: 0.005,
  epochs: 150, batchSize: 64, valShare: 0.15, patience: 15, features: null,
  knnK: 10, knnWeighted: true, knnAuto: true,
};

/* One line under each setting: what it is, and which way to move it. */
const Help = ({ children }) => <small className="help">{children}</small>;

function readHashCfg() {
  try {
    const m = window.location.hash.match(/cfg=([^&]+)/);
    return m ? { ...DEFAULT_CFG, ...JSON.parse(decodeURIComponent(atob(m[1]))) } : DEFAULT_CFG;
  } catch { return DEFAULT_CFG; }
}

const money = (v, unit) => `${unit}${Math.round(v).toLocaleString("en-GB")}`;
const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function App() {
  const [cfg, setCfg] = useState(readHashCfg);
  const [data, setData] = useState(null);           // { headers, rows, name, unit, note, target }
  const [err, setErr] = useState("");
  const [status, setStatus] = useState("idle");      // idle | training | done
  const [history, setHistory] = useState([]);
  const [result, setResult] = useState(null);
  const [copied, setCopied] = useState(false);
  const stopRef = useRef(false);
  const modelRef = useRef(null);
  const fileRef = useRef(null);
  const set = (k, v) => setCfg((c) => ({ ...c, [k]: v }));

  /* ── 1 · Data ── */
  const loadDataset = useCallback(async (key) => {
    setErr(""); setResult(null); setHistory([]);
    const d = DATASETS[key];
    try {
      const text = await (await fetch(BASE + d.file)).text();
      const p = parseCSV(text);
      setData({ ...p, name: d.name, unit: d.unit, note: d.note, target: d.target });
    } catch (e) { setErr(`Could not load the dataset: ${e.message}`); }
  }, []);

  useEffect(() => { if (cfg.dataset !== "upload") loadDataset(cfg.dataset); }, [cfg.dataset, loadDataset]);

  const onUpload = async (file) => {
    setErr(""); setResult(null); setHistory([]);
    try {
      const p = parseCSV(await file.text());
      const prof = profile(p.rows, p.headers);
      const guess = prof.find((c) => c.numeric && /price|precio|pvp/i.test(c.key)) || prof.find((c) => c.numeric);
      setData({ ...p, name: file.name, unit: /usd|\$/i.test(guess?.key || "") ? "$" : "€", note: "Your file. Everything stays in this browser.", target: guess?.key || "" });
      setCfg((c) => ({ ...c, dataset: "upload", features: null }));
    } catch (e) { setErr(`Could not read that file: ${e.message}`); }
  };

  const prof = useMemo(() => (data ? profile(data.rows, data.headers) : []), [data]);
  const target = data?.target || "";
  const usable = prof.filter((c) => c.key !== target && (c.numeric || c.distinct <= 60));
  const features = cfg.features ?? usable.map((c) => c.key);
  const numeric = features.filter((k) => prof.find((c) => c.key === k)?.numeric);
  const categorical = features.filter((k) => !prof.find((c) => c.key === k)?.numeric);

  /* ── The design matrix ── */
  const prepared = useMemo(() => {
    if (!data || !target || !features.length) return null;
    const spec = { target, numeric, categorical };
    const { rows, dropped } = clean(data.rows, spec);
    const { train: tr, test: te } = split(rows, cfg.testShare, cfg.seed);
    const enc = fitEncoder(tr, { ...spec, logTarget: cfg.logTarget, minCount: cfg.minCount });
    return { spec, rows, dropped, tr, te, enc };
  }, [data, target, features.join("|"), cfg.testShare, cfg.seed, cfg.logTarget, cfg.minCount]);

  const nParams = prepared ? paramCount(prepared.enc.names.length, cfg.layers) : 0;

  /* ── 4 · Train ── */
  const run = async () => {
    if (!prepared) return;
    setStatus("training"); setHistory([]); setResult(null); stopRef.current = false;
    // The CPU backend, on purpose: for a few thousand cars and a small network it
    // trains in seconds, gives the same numbers on every laptop, and does not
    // stall when the GPU is busy or the tab is in the background.
    if (tf.getBackend() !== "cpu") await tf.setBackend("cpu");
    await tf.ready();
    const { tr, te, enc } = prepared;
    const X = tr.map((r) => encodeRow(enc, r)), y = tr.map((r) => yToModel(enc, toNum(r[target])));
    const Xt = te.map((r) => encodeRow(enc, r)), actual = te.map((r) => toNum(r[target]));

    modelRef.current?.dispose();
    const model = buildModel(X[0].length, cfg);
    modelRef.current = model;
    const t0 = performance.now();
    const fit = await train(model, X, y, cfg, { onEpoch: (h) => setHistory([...h]), shouldStop: () => stopRef.current });
    const secs = (performance.now() - t0) / 1000;

    const toPrice = (z) => yFromModel(enc, z);
    const nnPred = predict(model, Xt).map(toPrice);
    const lin = fitLinear(X, y);
    const linPred = Xt.map((x) => toPrice(lin.predict(x)));
    // The comparables model. k is chosen on the TRAINING cars (each priced from
    // the others), never on the test cars, which stay for the final comparison.
    let knnK = cfg.knnK, kCurve = null;
    if (cfg.knnAuto) {
      kCurve = tuneK(X, y, { weighted: cfg.knnWeighted, errFn: (a, p) => metrics(a.map(toPrice), p.map(toPrice)).mae });
      knnK = kCurve.reduce((b, r) => (r.err < b.err ? r : b), kCurve[0]).k;
    }
    const knn = fitKNN(X, y, { k: knnK, weighted: cfg.knnWeighted });
    const knnPred = Xt.map((x) => toPrice(knn.predict(x)));
    const meanPrice = tr.reduce((s, r) => s + toNum(r[target]), 0) / tr.length;
    const predictRows = (rows) => predict(model, rows.map((r) => encodeRow(enc, r))).map(toPrice);
    const importance = permutationImportance(predictRows, te, actual, features, (a, p) => metrics(a, p).mae, cfg.seed);
    setResult({
      actual, nnPred, linPred, knnPred, lin, knn, knnK, kCurve, trRows: tr, fit, secs, importance,
      mNN: metrics(actual, nnPred), mLin: metrics(actual, linPred), mKNN: metrics(actual, knnPred),
      mMean: metrics(actual, actual.map(() => meanPrice)),
      bandNN: errorBand(actual, nnPred), bandLin: errorBand(actual, linPred), bandKNN: errorBand(actual, knnPred),
    });
    setStatus("done");
  };

  const shareLink = () => {
    const payload = btoa(encodeURIComponent(JSON.stringify({ ...cfg, dataset: cfg.dataset === "upload" ? "spain" : cfg.dataset })));
    const url = `${window.location.origin}${window.location.pathname}#cfg=${payload}`;
    navigator.clipboard?.writeText(url);
    window.history.replaceState(null, "", `#cfg=${payload}`);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  const setLayer = (i, k, v) => set("layers", cfg.layers.map((l, j) => (j === i ? { ...l, [k]: v } : l)));
  const unit = data?.unit || "€";

  return (
    <div className="wrap">
      <header>
        <div className="kicker">MARKETING ANALYTICS · UFV</div>
        <h1>What is my used car worth?</h1>
        <p className="lead">
          A neural network learns the price of second-hand cars from listings, and then prices yours. Everything is
          configurable — the data, the variables, the layers of the network — and everything runs in this browser.
          It is always compared with a linear regression on the same variables: a network is only worth its
          complexity if it beats that.
        </p>
      </header>

      {/* 1 · Data */}
      <section>
        <h2>1 · Data</h2>
        <div className="chips">
          {Object.entries(DATASETS).map(([k, d]) => (
            <button key={k} className={cfg.dataset === k ? "chip on" : "chip"} onClick={() => setCfg((c) => ({ ...c, dataset: k, features: null }))}>{d.name}</button>
          ))}
          <button className={cfg.dataset === "upload" ? "chip on" : "chip"} onClick={() => fileRef.current?.click()}>Upload a CSV…</button>
          <input ref={fileRef} type="file" accept=".csv,.txt" hidden onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        </div>
        {err && <p className="bad">{err}</p>}
        {data && (
          <>
            <p className="hint">{data.note}</p>
            <p className="hint">{data.rows.length.toLocaleString("en-GB")} rows · {data.headers.length} columns. Build your own with the scraper in the repository (scraper/scrape.py), then upload it here.</p>
            <div className="scroll"><table>
              <thead><tr>{data.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
              <tbody>{data.rows.slice(0, 5).map((r, i) => <tr key={i}>{data.headers.map((h) => <td key={h}>{r[h]}</td>)}</tr>)}</tbody>
            </table></div>
          </>
        )}
      </section>

      {/* 2 · Variables */}
      {data && (
        <section>
          <h2>2 · Variables</h2>
          <div className="grid">
            <label>Price column
              <select value={target} onChange={(e) => setData((d) => ({ ...d, target: e.target.value }))}>
                {prof.filter((c) => c.numeric).map((c) => <option key={c.key}>{c.key}</option>)}
              </select>
            </label>
            <label>Target
              <select value={cfg.logTarget ? "log" : "raw"} onChange={(e) => set("logTarget", e.target.value === "log")}>
                <option value="raw">the price itself</option>
                <option value="log">log of the price</option>
              </select>
              <Help>Log: the model learns % differences (a year of age costs about the same share of the price for any car). Usually helps a linear model a lot.</Help>
            </label>
            <label>Test cars (held out)
              <select value={cfg.testShare} onChange={(e) => set("testShare", Number(e.target.value))}>
                {[0.1, 0.2, 0.3].map((v) => <option key={v} value={v}>{v * 100}%</option>)}
              </select>
              <Help>Cars kept aside and used only at the end, to measure every model on cars it never saw.</Help>
            </label>
            <label>Seed <input type="number" value={cfg.seed} onChange={(e) => set("seed", Number(e.target.value) || 1)} />
              <Help>Fixes which cars are test cars and the network’s starting weights. The order in which cars are shuffled during training is still random, so the network’s error moves a little between runs — train twice before believing a small difference.</Help></label>
            <label>Rare categories <input type="number" min="1" value={cfg.minCount} onChange={(e) => set("minCount", Math.max(1, Number(e.target.value) || 1))} />
              <Help>A level seen in fewer training cars than this is pooled as “other”: a handful of cars cannot teach a weight.</Help></label>
          </div>
          <p className="hint">Features — numeric ones are standardised, categorical ones become one dummy per level (levels with fewer than {cfg.minCount} training cars are pooled as “other”). Click to include or leave out.</p>
          <div className="chips">
            {usable.map((c) => (
              <button key={c.key} className={features.includes(c.key) ? "chip on" : "chip"} title={c.numeric ? `numeric ${c.min}–${c.max}` : `${c.distinct} categories`}
                onClick={() => set("features", features.includes(c.key) ? features.filter((k) => k !== c.key) : [...features, c.key])}>
                {c.key} <small>{c.numeric ? "num" : `${c.distinct} cat`}</small>
              </button>
            ))}
          </div>
          {prepared && <p className="hint">{prepared.tr.length.toLocaleString("en-GB")} training cars · {prepared.te.length.toLocaleString("en-GB")} test cars · {prepared.enc.names.length} network inputs{prepared.dropped ? ` · ${prepared.dropped} rows left out for missing values` : ""}.</p>}
        </section>
      )}

      {/* 3 · Network */}
      {prepared && (
        <section>
          <h2>3 · Network</h2>
          <div className="chips">
            {Object.entries(PRESETS).map(([k, p]) => <button key={k} className="chip" onClick={() => set("layers", p.layers)}>{p.label}</button>)}
          </div>
          <div className="arch">
            <span className="node">{prepared.enc.names.length} inputs</span>
            {cfg.layers.map((l, i) => (
              <span key={i} className="layer">
                →
                <input type="number" min="1" max="512" value={l.units} onChange={(e) => setLayer(i, "units", Math.max(1, Number(e.target.value) || 1))} />
                <select value={l.activation} onChange={(e) => setLayer(i, "activation", e.target.value)}>{ACTIVATIONS.map((a) => <option key={a}>{a}</option>)}</select>
                <button className="x" title="remove layer" onClick={() => set("layers", cfg.layers.filter((_, j) => j !== i))}>×</button>
              </span>
            ))}
            <button className="chip" onClick={() => set("layers", [...cfg.layers, { units: 16, activation: "relu" }])}>+ layer</button>
            <span className="node">→ price</span>
          </div>
          <p className="hint"><strong>Layers and units.</strong> Each hidden layer combines the inputs into new features; the units (neurons) are how many it builds. More layers and units = more capacity to capture interactions (a premium brand that loses value faster) — and more risk of memorising. Start with 64 → 32; add capacity only if both curves stay high. <strong>Activation</strong> is what makes it non-linear: <em>relu</em> is the default; <em>tanh</em>/<em>sigmoid</em> saturate and learn slower; <em>linear</em> removes the non-linearity altogether.</p>
          <p className={nParams > prepared.tr.length ? "warn" : "hint"}>
            {nParams.toLocaleString("en-GB")} weights to learn from {prepared.tr.length.toLocaleString("en-GB")} cars.
            {nParams > prepared.tr.length && " More weights than cars: the network can memorise the training set. Watch the validation curve, and use dropout or L2."}
            {!cfg.layers.length && " With no hidden layer the network IS a linear regression, fitted by gradient descent — it should land near the linear baseline."}
          </p>
          <div className="grid">
            <label>Dropout <input type="number" step="0.05" min="0" max="0.8" value={cfg.dropout} onChange={(e) => set("dropout", Number(e.target.value))} />
              <Help>Share of neurons switched off at random at each step, so no single one can memorise a car. 0–0.3; raise it when validation is much worse than training.</Help></label>
            <label>L2 penalty <input type="number" step="0.0001" min="0" value={cfg.l2} onChange={(e) => set("l2", Number(e.target.value))} />
              <Help>Cost on large weights; keeps the network smooth. 0.00001–0.001; raise it against memorising, lower it if both curves stay high.</Help></label>
            <label>Optimiser <select value={cfg.optimizer} onChange={(e) => set("optimizer", e.target.value)}>{OPTIMIZERS.map((o) => <option key={o}>{o}</option>)}</select>
              <Help>How weights are updated. <em>adam</em> adapts its steps and works almost always; <em>sgd</em> needs a higher learning rate and more epochs.</Help></label>
            <label>Learning rate <input type="number" step="0.001" min="0.0001" value={cfg.lr} onChange={(e) => set("lr", Number(e.target.value))} />
              <Help>Size of each step. Too high: the loss jumps or explodes. Too low: it creeps down for ever. Adam: 0.001–0.01.</Help></label>
            <label>Epochs (max) <input type="number" min="1" max="2000" value={cfg.epochs} onChange={(e) => set("epochs", Number(e.target.value))} />
              <Help>Passes over the training cars. A ceiling: early stopping usually ends sooner.</Help></label>
            <label>Batch size <input type="number" min="8" max="1024" value={cfg.batchSize} onChange={(e) => set("batchSize", Number(e.target.value))} />
              <Help>Cars per weight update. Small: noisier, slower, sometimes generalises better. Large: smoother and faster. 32–256.</Help></label>
            <label>Validation share <input type="number" step="0.05" min="0.05" max="0.4" value={cfg.valShare} onChange={(e) => set("valShare", Number(e.target.value))} />
              <Help>Part of the training cars the network does not learn from; the orange curve. It decides early stopping — and your tuning.</Help></label>
            <label>Early stop after <input type="number" min="0" value={cfg.patience} onChange={(e) => set("patience", Number(e.target.value))} />
              <Help>Stop when validation has not improved for this many epochs, and keep the best epoch’s weights. 0 = off.</Help></label>
          </div>

          <h3>The comparables model: k-nearest neighbours (KNN)</h3>
          <p className="hint">No training at all: a car is priced as the average of the <em>k</em> most similar cars in the training data — what an appraiser or an estate agent does with “comparables”. Similarity is measured on the same standardised variables the network sees, so the variables you include decide what “similar” means.</p>
          <div className="grid">
            <label>Choose k automatically
              <select value={cfg.knnAuto ? "yes" : "no"} onChange={(e) => set("knnAuto", e.target.value === "yes")}>
                <option value="yes">yes — try 1, 3, 5, 10, 20, 40</option><option value="no">no — use the k below</option>
              </select>
              <Help>Each training car is priced from the others with every k; the best k is kept. The test cars are not used.</Help></label>
            <label>k (neighbours) <input type="number" min="1" max="100" value={cfg.knnK} disabled={cfg.knnAuto} onChange={(e) => set("knnK", Math.max(1, Number(e.target.value) || 1))} />
              <Help>Small k follows the data closely but is noisy (one odd car moves the price); large k is smoother but blurs real differences.</Help></label>
            <label>Weighting
              <select value={cfg.knnWeighted ? "distance" : "uniform"} onChange={(e) => set("knnWeighted", e.target.value === "distance")}>
                <option value="distance">closer cars count more</option><option value="uniform">all k count the same</option>
              </select>
              <Help>With distance weighting, a nearly identical car dominates the average.</Help></label>
          </div>
        </section>
      )}

      {/* 4 · Train */}
      {prepared && (
        <section>
          <h2>4 · Train</h2>
          <div className="chips">
            <button className="primary" disabled={status === "training"} onClick={run}>{status === "training" ? "Training…" : "Train the network"}</button>
            {status === "training" && <button className="chip" onClick={() => { stopRef.current = true; }}>Stop</button>}
            <button className="chip" onClick={shareLink}>{copied ? "Link copied" : "Copy a link to this configuration"}</button>
          </div>
          <details className="guide">
            <summary>How to tune the network — read the two curves</summary>
            <table>
              <thead><tr><th>What you see</th><th>What it means</th><th>What to change</th></tr></thead>
              <tbody>
                <tr><td>Both curves stay high</td><td>Underfitting: the network is too simple or learns too slowly</td><td>More units or a layer; less dropout / L2; a higher learning rate; log target</td></tr>
                <tr><td>Training keeps falling, validation turns up</td><td>Overfitting: it is memorising the training cars</td><td>More dropout or L2; fewer units; keep early stopping on; more data</td></tr>
                <tr><td>The curves jump up and down</td><td>Steps too large</td><td>Lower the learning rate, or a larger batch</td></tr>
                <tr><td>The loss creeps down very slowly</td><td>Steps too small</td><td>Raise the learning rate; use adam</td></tr>
                <tr><td>Loss becomes NaN / explodes</td><td>Learning rate far too high</td><td>Divide it by 10</td></tr>
                <tr><td>Network ≈ linear regression</td><td>No non-linearity worth learning — or it cannot find it</td><td>Check the activation is not <em>linear</em>; otherwise prefer the simpler model</td></tr>
              </tbody>
            </table>
            <p className="hint">Rules: change one thing at a time and keep the seed, so the difference is yours and not luck. Tune by looking at the <strong>validation</strong> curve; the test cars in step 5 are for the final comparison only — tuning until the test result looks good makes it a second training set, and the reported error stops being honest.</p>
          </details>
          {history.length > 0 && (
            <>
              <LossChart history={history} />
              <p className="hint">
                Epoch {history.length}. Training loss (blue) always falls; what matters is validation (orange), measured on cars the network
                never learns from. When validation stops improving while training keeps falling, the network has started memorising.
                {result && ` Kept the weights of epoch ${result.fit.history.reduce((b, d) => (d.val < b.val ? d : b), result.fit.history[0]).epoch} (best validation) · ${result.secs.toFixed(1)} s.`}
              </p>
            </>
          )}
        </section>
      )}

      {/* 5 · Results */}
      {result && (
        <section>
          <h2>5 · How good is it? On {result.actual.length.toLocaleString("en-GB")} test cars it never saw</h2>
          <table className="metrics">
            <thead><tr><th>Model</th><th>Typical error (MAE)</th><th>RMSE</th><th>Average % error (MAPE)</th><th>R²</th></tr></thead>
            <tbody>
              {[["Average price for every car", result.mMean], ["Linear regression · same variables", result.mLin], [`k-nearest neighbours · k = ${result.knnK}`, result.mKNN], ["Neural network", result.mNN]].map(([name, m]) => (
                <tr key={name} className={name === "Neural network" ? "hl" : ""}><td>{name}</td><td>{money(m.mae, unit)}</td><td>{money(m.rmse, unit)}</td><td>{pct(m.mape)}</td><td>{m.r2.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            {result.mNN.mape < result.mLin.mape * 0.9
              ? `The network’s errors are ${pct(1 - result.mNN.mape / result.mLin.mape)} smaller than the linear regression’s: it found structure a straight line cannot draw.`
              : result.mNN.mape <= result.mLin.mape * 1.02
                ? "The network is not clearly better than a linear regression. Then the simpler model is the one to use — or the variables, not the model, are what is missing."
                : "The network is worse than a linear regression. Try fewer weights, more regularisation, a smaller learning rate — or the log of the price."}
            {" "}{result.mKNN.mape < result.mNN.mape
              ? `The comparables model (KNN) does better than the network here: with these variables, averaging the ${result.knnK} most similar cars is enough.`
              : `KNN, averaging the ${result.knnK} most similar cars, lands at ${pct(result.mKNN.mape)}: comparables work, but a model that learns how each variable moves the price does better.`}
            {!cfg.logTarget && result.mLin.mape > 0.2 && " Tip: switch the target to the log of the price and train again — see what it does to the linear regression."}
          </p>
          <div className="two">
            <div>
              <h3>Predicted against actual</h3>
              <ScatterChart actual={result.actual} unit={unit} series={[{ label: "linear regression", color: "#9ca3af", pred: result.linPred }, { label: "KNN", color: "#16a34a", pred: result.knnPred }, { label: "neural network", color: "#2563eb", pred: result.nnPred }]} />
              {result.kCurve && (
                <>
                  <h3>Choosing k (on the training cars)</h3>
                  <Bars items={result.kCurve.map((r) => ({ label: `k = ${r.k}${r.k === result.knnK ? " ✓" : ""}`, value: r.err, text: money(r.err, unit) }))} />
                  <p className="hint">Typical error with each k when every training car is priced from the others. Too few neighbours copy noise; too many average unlike cars.</p>
                </>
              )}
            </div>
            <div>
              <h3>What the network relies on</h3>
              <p className="hint">How much the network’s error grows when one variable is shuffled across the test cars — a variable it ignores costs nothing to scramble.</p>
              <Bars items={result.importance.map((d) => ({ label: d.col, value: d.increase, text: `+${pct(d.increase)}` }))} />
            </div>
          </div>
        </section>
      )}

      {/* 6 · My car */}
      {result && prepared && <MyCar prepared={prepared} result={result} model={modelRef.current} unit={unit} features={features} numeric={numeric} target={target} />}

      <footer>
        César Moreno Pascual, PhD · Marketing Analytics, UFV. Source, data licences and the scraper:{" "}
        <a href="https://github.com/cmoreno34/used-car-price-nn">github.com/cmoreno34/used-car-price-nn</a>
      </footer>
    </div>
  );
}

function MyCar({ prepared, result, model, unit, features, numeric, target }) {
  const { rows, enc } = prepared;
  const defaults = useMemo(() => Object.fromEntries(features.map((k) => {
    if (numeric.includes(k)) {
      const v = rows.map((r) => toNum(r[k])).sort((a, b) => a - b);
      return [k, v[Math.floor(v.length / 2)]];
    }
    const counts = new Map(); rows.forEach((r) => counts.set(r[k], (counts.get(r[k]) || 0) + 1));
    return [k, [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0]];
  })), [rows, features, numeric]);
  const [car, setCar] = useState(defaults);
  useEffect(() => setCar(defaults), [defaults]);

  const x = encodeRow(enc, car);
  const nn = yFromModel(enc, predict(model, [x])[0]);
  const lin = yFromModel(enc, result.lin.predict(x));
  // The comparables the KNN price averages: the k nearest training cars.
  const nb = result.knn.neighbours(x, Math.min(result.knnK, 12));
  const knnPrice = yFromModel(enc, result.knn.predict(x));
  const similar = nb.idx.map((i, j) => ({ r: result.trRows[i], d: nb.dist[j] }));

  return (
    <section>
      <h2>6 · Price my car</h2>
      <div className="grid">
        {features.map((k) => {
          const levels = enc.cat.find((c) => c.key === k)?.all;
          return (
            <label key={k}>{k}
              {levels ? (
                <select value={car[k] ?? ""} onChange={(e) => setCar((c) => ({ ...c, [k]: e.target.value }))}>{levels.map((l) => <option key={l}>{l}</option>)}</select>
              ) : (
                <input type="number" value={car[k] ?? ""} onChange={(e) => setCar((c) => ({ ...c, [k]: e.target.value }))} />
              )}
            </label>
          );
        })}
      </div>
      <div className="price-cards">
        <div className="price-card main">
          <div className="lbl">Neural network</div>
          <div className="big">{money(nn, unit)}</div>
          <div className="hint">Likely range {money(nn * result.bandNN.lo, unit)} – {money(nn * result.bandNN.hi, unit)}: for 80% of the test cars, the real price fell within this band around the prediction.</div>
        </div>
        <div className="price-card">
          <div className="lbl">Linear regression</div>
          <div className="big muted">{money(lin, unit)}</div>
          <div className="hint">range {money(lin * result.bandLin.lo, unit)} – {money(lin * result.bandLin.hi, unit)}</div>
        </div>
        <div className="price-card">
          <div className="lbl">KNN · average of {result.knnK} comparables</div>
          <div className="big muted">{money(knnPrice, unit)}</div>
          <div className="hint">range {money(knnPrice * result.bandKNN.lo, unit)} – {money(knnPrice * result.bandKNN.hi, unit)}</div>
        </div>
      </div>
      <h3>The comparable cars the KNN price is built from{result.knnK > 12 ? " (first 12)" : ""}</h3>
      <div className="scroll"><table>
        <thead><tr>{features.map((k) => <th key={k}>{k}</th>)}<th>{target}</th><th>distance</th></tr></thead>
        <tbody>{similar.map(({ r, d }, i) => <tr key={i}>{features.map((k) => <td key={k}>{r[k]}</td>)}<td><strong>{money(toNum(r[target]), unit)}</strong></td><td>{d.toFixed(2)}</td></tr>)}</tbody>
      </table></div>
      <p className="hint">A price is a starting point for a negotiation, not a verdict. The model knows only the variables above: not the state of the paint, the service record, or how quickly you need to sell.</p>
    </section>
  );
}
