/* Data: CSV in, a numeric design matrix out.
 *
 * A neural network only sees numbers, so every choice here is a modelling
 * choice the page makes visible: numeric columns are standardised (z-scores,
 * with the TRAINING set's mean and sd), categorical columns become one-hot
 * dummies (levels too rare to learn from are pooled into "other"), and the
 * price can be modelled in logs. The encoder is fitted on the training rows
 * only and then applied to test rows and to "my car", so nothing about the
 * test set leaks into training. */

export function parseCSV(text) {
  const first = text.split(/\r?\n/, 1)[0];
  const delim = [",", ";", "\t"].map((d) => [d, first.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === delim) { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); if (row.some((v) => v !== "")) rows.push(row); }
  const headers = rows.shift().map((h) => h.trim());
  return { headers, rows: rows.map((r) => Object.fromEntries(headers.map((h, j) => [h, (r[j] ?? "").trim()]))) };
}

export const toNum = (v) => {
  if (typeof v === "number") return v;
  const s = String(v ?? "").trim();
  if (s === "") return NaN;
  return Number(/^-?\d{1,3}(\.\d{3})+(,\d+)?$/.test(s) ? s.replace(/\./g, "").replace(",", ".") : s.replace(",", "."));
};

/* Numeric when (almost) every non-empty value parses and there are more than a
 * handful of distinct values; otherwise categorical. */
export function profile(rows, headers) {
  return headers.map((h) => {
    const vals = rows.map((r) => r[h]).filter((v) => v !== "" && v != null);
    const nums = vals.map(toNum).filter(Number.isFinite);
    const distinct = new Set(vals).size;
    const numeric = vals.length > 0 && nums.length / vals.length > 0.97 && distinct > 6;
    return { key: h, numeric, distinct, missing: rows.length - vals.length,
      min: numeric ? Math.min(...nums) : null, max: numeric ? Math.max(...nums) : null };
  });
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function split(rows, testShare, seed) {
  const rng = mulberry32(seed);
  const idx = rows.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTest = Math.round(rows.length * testShare);
  return { test: idx.slice(0, nTest).map((i) => rows[i]), train: idx.slice(nTest).map((i) => rows[i]) };
}

/* Fits on training rows. `minCount`: a category level seen fewer times than this
 * in training is pooled into "(other)" — one car cannot teach a weight. */
export function fitEncoder(train, { target, numeric, categorical, logTarget, minCount = 10 }) {
  const num = numeric.map((k) => {
    const v = train.map((r) => toNum(r[k]));
    const mu = v.reduce((s, x) => s + x, 0) / v.length;
    const sd = Math.sqrt(v.reduce((s, x) => s + (x - mu) ** 2, 0) / v.length) || 1;
    return { key: k, mu, sd };
  });
  const cat = categorical.map((k) => {
    const counts = new Map();
    for (const r of train) counts.set(r[k], (counts.get(r[k]) || 0) + 1);
    const levels = [...counts.entries()].filter(([, n]) => n >= minCount).sort((a, b) => b[1] - a[1]).map(([l]) => l);
    const pooled = counts.size > levels.length;
    return { key: k, levels, pooled, all: [...counts.keys()].sort() };
  });
  const names = [
    ...num.map((n) => n.key),
    ...cat.flatMap((c) => [...c.levels.map((l) => `${c.key}=${l}`), ...(c.pooled ? [`${c.key}=(other)`] : [])]),
  ];
  const ys = train.map((r) => toNum(r[target]));
  const yT = ys.map((y) => (logTarget ? Math.log(y) : y));
  const yMu = yT.reduce((s, x) => s + x, 0) / yT.length;
  const ySd = Math.sqrt(yT.reduce((s, x) => s + (x - yMu) ** 2, 0) / yT.length) || 1;
  return { target, logTarget, num, cat, names, yMu, ySd };
}

export function encodeRow(enc, r) {
  const x = enc.num.map((n) => (toNum(r[n.key]) - n.mu) / n.sd);
  for (const c of enc.cat) {
    const v = r[c.key];
    const known = c.levels.includes(v);
    for (const l of c.levels) x.push(v === l ? 1 : 0);
    if (c.pooled) x.push(known ? 0 : 1);
  }
  return x;
}

/* The network is trained on a standardised target; these go back and forth. */
export const yToModel = (enc, price) => ((enc.logTarget ? Math.log(price) : price) - enc.yMu) / enc.ySd;
export const yFromModel = (enc, z) => { const t = z * enc.ySd + enc.yMu; return enc.logTarget ? Math.exp(t) : t; };

/* Rows with a usable target and every selected feature present. */
export function clean(rows, { target, numeric, categorical }) {
  let dropped = 0;
  const out = rows.filter((r) => {
    const ok = toNum(r[target]) > 0 && numeric.every((k) => Number.isFinite(toNum(r[k]))) &&
      categorical.every((k) => r[k] !== "" && r[k] != null);
    if (!ok) dropped++;
    return ok;
  });
  return { rows: out, dropped };
}
