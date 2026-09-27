/* Linear baseline and metrics against NumPy / scikit-learn (test/ref_models.py),
 * the encoder's leakage rules, and — the point of the repository — that a small
 * network trained here learns the non-linear Spanish price rule better than a
 * linear regression on the raw price. */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { fitLinear, metrics, fitKNN, tuneK } from "../src/lib/models.js";
import { parseCSV, fitEncoder, encodeRow, clean, split, yToModel, yFromModel, toNum } from "../src/lib/data.js";

const ref = JSON.parse(fs.readFileSync(new URL("./models_reference.json", import.meta.url)));
const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

test("OLS coefficients match NumPy lstsq", () => {
  const { w } = fitLinear(ref.X, ref.y, 0);
  w.forEach((v, i) => assert.ok(close(v, ref.w[i], 1e-9), `w${i}: ${v} vs ${ref.w[i]}`));
});

test("metrics match scikit-learn", () => {
  const m = metrics(ref.act, ref.prd);
  assert.ok(close(m.mae, ref.mae, 1e-9)); assert.ok(close(m.rmse, ref.rmse, 1e-9));
  assert.ok(close(m.r2, ref.r2, 1e-9)); assert.ok(close(m.mape, ref.mape, 1e-9));
});

test("KNN matches scikit-learn KNeighborsRegressor (uniform and distance weights)", () => {
  for (const w of ["uniform", "distance"]) for (const k of [1, 5, 12]) {
    const m = fitKNN(ref.X, ref.y, { k, weighted: w === "distance" });
    ref.Xq.forEach((x, i) => assert.ok(close(m.predict(x), ref.knn[`${w}_${k}`][i], 1e-9), `${w} k=${k} row ${i}`));
  }
});

test("tuneK prices each training car from the others and returns an error per k", () => {
  const res = tuneK(ref.X, ref.y, { ks: [1, 5, 20], sample: 100, errFn: (a, p) => metrics(a.map(Math.exp), p.map(Math.exp)).mae });
  assert.equal(res.length, 3);
  assert.ok(res.every((r) => Number.isFinite(r.err) && r.err > 0));
});

test("Spanish number formats and CSV quoting", () => {
  assert.equal(toNum("14.990"), 14990); assert.equal(toNum("1,6"), 1.6); assert.equal(toNum("2019"), 2019);
  const d = parseCSV('a;b\n"x;y";1\n');
  assert.deepEqual(d.rows[0], { a: "x;y", b: "1" });
});

const spain = parseCSV(fs.readFileSync(new URL("../public/data/spain_synthetic.csv", import.meta.url), "utf8"));
const spec = { target: "price_eur", numeric: ["year", "km", "power_cv"],
  categorical: ["brand", "segment", "fuel", "gearbox", "dgt_label", "province", "seller"] };

test("the encoder is fitted on training rows only and pools rare levels", () => {
  const { rows } = clean(spain.rows, spec);
  const { train } = split(rows, 0.2, 1);
  const enc = fitEncoder(train, { ...spec, logTarget: true, minCount: 10 });
  const mu = train.reduce((s, r) => s + toNum(r.km), 0) / train.length;
  assert.ok(close(enc.num[1].mu, mu, 1e-12));
  const x = encodeRow(enc, { ...train[0], brand: "NotABrand" });
  const other = enc.names.indexOf("brand=(other)");
  if (other >= 0) assert.equal(x[other], 1);
  assert.equal(x.length, enc.names.length);
  assert.ok(close(yFromModel(enc, yToModel(enc, 12345)), 12345, 1e-9));
});

test("a small network beats linear regression on the raw price (non-linear rule)", { timeout: 240000 }, async () => {
  const tf = await import("@tensorflow/tfjs");
  const { buildModel, train, predict } = await import("../src/lib/nn.js");
  await tf.setBackend("cpu");
  const { rows } = clean(spain.rows, spec);
  const { train: tr, test: te } = split(rows.slice(0, 3000), 0.2, 1);
  const enc = fitEncoder(tr, { ...spec, logTarget: false });
  const X = tr.map((r) => encodeRow(enc, r)), Xt = te.map((r) => encodeRow(enc, r));
  const y = tr.map((r) => yToModel(enc, toNum(r.price_eur))), act = te.map((r) => toNum(r.price_eur));
  const lin = fitLinear(X, y);
  const mLin = metrics(act, Xt.map((x) => yFromModel(enc, lin.predict(x))));
  const cfg = { layers: [{ units: 32, activation: "relu" }, { units: 16, activation: "relu" }], dropout: 0, l2: 0,
    optimizer: "adam", lr: 0.01, epochs: 60, batchSize: 64, valShare: 0.15, patience: 10, seed: 1 };
  const model = buildModel(X[0].length, cfg);
  await train(model, X, y, cfg);
  const mNN = metrics(act, predict(model, Xt).map((z) => yFromModel(enc, z)));
  assert.ok(mNN.mape < mLin.mape * 0.6, `network MAPE ${mNN.mape.toFixed(3)} vs linear ${mLin.mape.toFixed(3)}`);
});
