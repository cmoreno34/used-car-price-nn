/* The yardsticks a neural network has to beat, and the numbers that say by how much.
 *
 *   mean baseline      predict the average training price for every car
 *   linear regression  ordinary least squares on the same encoded features
 *
 * Both are fitted on exactly the design matrix the network sees, so the
 * comparison is about the model, not about the features. Checked against
 * NumPy in test/models.test.js. */

function solve(A, b) {
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    if (Math.abs(d) < 1e-12) continue;
    for (let j = c; j <= n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f) for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((r) => r[n]);
}

/* OLS with an intercept. A tiny ridge term keeps the system solvable when two
 * dummies are collinear (a full set of one-hot levels plus an intercept is). */
export function fitLinear(X, y, ridge = 1e-6) {
  const k = X[0].length + 1;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < X.length; i++) {
    const xi = [1, ...X[i]];
    for (let a = 0; a < k; a++) {
      Xty[a] += xi[a] * y[i];
      for (let b = a; b < k; b++) XtX[a][b] += xi[a] * xi[b];
    }
  }
  for (let a = 0; a < k; a++) { for (let b = 0; b < a; b++) XtX[a][b] = XtX[b][a]; if (a) XtX[a][a] += ridge * X.length; }
  const w = solve(XtX, Xty);
  return { w, predict: (x) => w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0) };
}

export function metrics(actual, pred) {
  const n = actual.length;
  const mean = actual.reduce((s, v) => s + v, 0) / n;
  let ae = 0, se = 0, ape = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const e = pred[i] - actual[i];
    ae += Math.abs(e); se += e * e; ape += Math.abs(e) / actual[i]; ss += (actual[i] - mean) ** 2;
  }
  return { mae: ae / n, rmse: Math.sqrt(se / n), mape: ape / n, r2: 1 - se / ss, n };
}

/* The range that contained 80 % of the test cars' errors, as a ratio of the
 * prediction: an honest "give or take" for a new car, taken from how the model
 * actually did rather than from an assumption about its errors. */
export function errorBand(actual, pred, lo = 0.1, hi = 0.9) {
  const r = actual.map((a, i) => a / pred[i]).sort((a, b) => a - b);
  const q = (p) => r[Math.min(r.length - 1, Math.max(0, Math.floor(p * (r.length - 1))))];
  return { lo: q(lo), hi: q(hi) };
}

/* k-nearest neighbours: the price of a car is the average price of the k
 * training cars most similar to it — the "comparables" an appraiser or an
 * estate agent uses, made systematic. Nothing is learned in advance; all the
 * work happens at prediction time, measuring distance in the same encoded
 * space the network sees (standardised numbers, 0/1 dummies). So the scaling
 * of the variables decides what "similar" means.
 *   k          how many neighbours: small k follows the data closely (noisy),
 *              large k averages more cars (smoother, but blurs real differences)
 *   weighted   closer neighbours count more (weight 1 / distance)
 * Checked against scikit-learn's KNeighborsRegressor in test/models.test.js. */
export function fitKNN(X, y, { k = 10, weighted = false } = {}) {
  const n = X.length, d = X[0].length;
  const flat = new Float64Array(n * d);
  X.forEach((r, i) => flat.set(r, i * d));
  const ys = Float64Array.from(y);
  const neighbours = (x, kk = k, skip = -1) => {
    // Keep the kk smallest distances with a simple bounded insertion list.
    const bd = new Float64Array(kk).fill(Infinity), bi = new Int32Array(kk).fill(-1);
    for (let i = 0; i < n; i++) {
      if (i === skip) continue;
      let s = 0;
      const o = i * d;
      for (let j = 0; j < d; j++) { const t = flat[o + j] - x[j]; s += t * t; if (s >= bd[kk - 1]) break; }
      if (s < bd[kk - 1]) {
        let p = kk - 1;
        while (p > 0 && bd[p - 1] > s) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
        bd[p] = s; bi[p] = i;
      }
    }
    return { idx: Array.from(bi).filter((i) => i >= 0), dist: Array.from(bd).map(Math.sqrt) };
  };
  const average = ({ idx, dist }) => {
    if (!weighted) return idx.reduce((s, i) => s + ys[i], 0) / idx.length;
    if (dist[0] === 0) { const exact = idx.filter((_, j) => dist[j] === 0); return exact.reduce((s, i) => s + ys[i], 0) / exact.length; }
    let w = 0, s = 0;
    idx.forEach((i, j) => { const wi = 1 / dist[j]; w += wi; s += wi * ys[i]; });
    return s / w;
  };
  return { k, weighted, neighbours, predict: (x) => average(neighbours(x)), predictSkip: (x, i) => average(neighbours(x, k, i)) };
}

/* Chooses k on the training cars only, by leave-one-out on a sample: each car
 * is priced from the others. The test cars are never used to tune — they are
 * kept for the final comparison, as the network's are. */
export function tuneK(X, y, { ks = [1, 3, 5, 10, 20, 40], weighted = false, sample = 600, errFn }) {
  const idx = X.map((_, i) => i).filter((_, i) => i % Math.max(1, Math.floor(X.length / sample)) === 0);
  return ks.map((k) => {
    const m = fitKNN(X, y, { k, weighted });
    const pred = idx.map((i) => m.predictSkip(X[i], i));
    return { k, err: errFn(idx.map((i) => y[i]), pred) };
  });
}
