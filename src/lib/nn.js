/* The neural network, in TensorFlow.js — trained in the browser.
 *
 * A multilayer perceptron for regression: the encoded features in, one number
 * out (the standardised price, or log price). Everything a student can change
 * is in `cfg`:
 *   layers      [{ units, activation }]  — no layers = linear regression by SGD
 *   dropout     share of units switched off at random while training
 *   l2          weight penalty
 *   optimizer   adam | sgd | rmsprop, with its learning rate
 *   epochs, batchSize, valShare, patience (early stopping), seed
 */
import * as tf from "@tensorflow/tfjs";
import { mulberry32 } from "./data.js";

export const ACTIVATIONS = ["relu", "elu", "tanh", "sigmoid", "linear"];
export const OPTIMIZERS = ["adam", "sgd", "rmsprop"];

export const PRESETS = {
  linear: { label: "No hidden layer (a linear model)", layers: [] },
  small: { label: "Small · 16", layers: [{ units: 16, activation: "relu" }] },
  medium: { label: "Medium · 64 → 32", layers: [{ units: 64, activation: "relu" }, { units: 32, activation: "relu" }] },
  deep: { label: "Deep · 128 → 64 → 32", layers: [{ units: 128, activation: "relu" }, { units: 64, activation: "relu" }, { units: 32, activation: "relu" }] },
};

export function paramCount(nIn, layers) {
  let n = 0, prev = nIn;
  for (const l of layers) { n += prev * l.units + l.units; prev = l.units; }
  return n + prev + 1;
}

export function buildModel(nIn, cfg) {
  const seedInit = (i) => tf.initializers.glorotUniform({ seed: (cfg.seed ?? 1) * 97 + i });
  const reg = cfg.l2 > 0 ? tf.regularizers.l2({ l2: cfg.l2 }) : undefined;
  const m = tf.sequential();
  cfg.layers.forEach((l, i) => {
    m.add(tf.layers.dense({ units: l.units, activation: l.activation, inputShape: i === 0 ? [nIn] : undefined,
      kernelInitializer: seedInit(i), kernelRegularizer: reg }));
    if (cfg.dropout > 0) m.add(tf.layers.dropout({ rate: cfg.dropout, seed: (cfg.seed ?? 1) + i }));
  });
  m.add(tf.layers.dense({ units: 1, inputShape: cfg.layers.length ? undefined : [nIn], kernelInitializer: seedInit(99) }));
  const opt = cfg.optimizer === "sgd" ? tf.train.sgd(cfg.lr) : cfg.optimizer === "rmsprop" ? tf.train.rmsprop(cfg.lr) : tf.train.adam(cfg.lr);
  m.compile({ optimizer: opt, loss: "meanSquaredError" });
  return m;
}

/* Trains, reporting every epoch. Keeps the weights of the best validation epoch
 * (early stopping restores them), so an over-trained network does not win.
 *
 * Reproducible: the validation cars and the order of the training cars in
 * every epoch come from the seed (TF.js's own shuffle uses Math.random), so
 * the same settings and seed give the same network on every computer — a
 * class can compare numbers, and a change of setting is the only difference. */
export async function train(model, X, y, cfg, { onEpoch, shouldStop } = {}) {
  const rng = mulberry32((cfg.seed ?? 1) * 7919 + 13);
  const shuffle = (a) => { const b = [...a]; for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; };
  const order = shuffle(X.map((_, i) => i));
  const nVal = Math.round(X.length * cfg.valShare);
  const valIdx = order.slice(0, nVal), trIdx = order.slice(nVal);
  const xv = tf.tensor2d(valIdx.map((i) => X[i])), yv = tf.tensor2d(valIdx.map((i) => [y[i]]));
  let best = Infinity, bestWeights = null, wait = 0;
  const history = [];
  for (let epoch = 0; epoch < cfg.epochs; epoch++) {
    const perm = shuffle(trIdx);
    const xs = tf.tensor2d(perm.map((i) => X[i])), ys = tf.tensor2d(perm.map((i) => [y[i]]));
    // TF.js otherwise yields every few batches through requestAnimationFrame, which
    // stops in a background tab; we yield once per epoch instead (below).
    const h = await model.fit(xs, ys, { epochs: 1, batchSize: cfg.batchSize, shuffle: false, validationData: [xv, yv], verbose: 0, yieldEvery: "never" });
    xs.dispose(); ys.dispose();
    const loss = h.history.loss[0], val = h.history.val_loss[0];
    history.push({ epoch: epoch + 1, loss, val });
    if (val < best - 1e-5) {
      best = val; wait = 0;
      bestWeights?.forEach((t) => t.dispose());
      bestWeights = model.getWeights().map((w) => w.clone());
    } else wait++;
    onEpoch?.(history);
    if ((cfg.patience > 0 && wait >= cfg.patience) || shouldStop?.()) break;
    // Yield to the page so the charts redraw. setTimeout, not requestAnimationFrame:
    // rAF stops in a background tab and training would freeze with it.
    await new Promise((r) => setTimeout(r, 0));
  }
  if (bestWeights) { model.setWeights(bestWeights); bestWeights.forEach((t) => t.dispose()); }
  xv.dispose(); yv.dispose();
  return { history, bestVal: best, stoppedAt: history.length };
}

export function predict(model, X) {
  return tf.tidy(() => Array.from(model.predict(tf.tensor2d(X)).dataSync()));
}

/* Permutation importance: shuffle one ORIGINAL column (all its dummies move
 * together) across the test cars and measure how much worse the error gets.
 * A column the model ignores costs nothing to scramble. */
export function permutationImportance(predictPrices, testRows, actual, columns, errFn, seed = 1) {
  const base = errFn(actual, predictPrices(testRows));
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  return columns.map((col) => {
    const vals = testRows.map((r) => r[col]);
    for (let i = vals.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [vals[i], vals[j]] = [vals[j], vals[i]]; }
    const shuffled = testRows.map((r, i) => ({ ...r, [col]: vals[i] }));
    return { col, increase: errFn(actual, predictPrices(shuffled)) / base - 1 };
  }).sort((a, b) => b.increase - a.increase);
}
