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
