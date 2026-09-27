/* Small canvas charts: one y-axis each, recessive grid, 2px lines. */
import { useRef, useEffect, useState } from "react";

const INK = "#1f2937", MUTED = "#6b7280", GRID = "#e5e7eb";
const fmt = (v) => {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return Math.round(v / 1e3) + "k";
  if (a >= 100) return Math.round(v).toString();
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(3);
};
function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(v);
  return out;
}

/* Draws on a canvas sized to its container, and redraws when the container
 * resizes — a chart first rendered while its section was still laying out
 * (width 0) would otherwise stay blank. */
function useCanvas(draw, deps, height) {
  const ref = useRef(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const box = ref.current?.parentElement;
    if (!box) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(box);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || w < 50) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = height * dpr; cv.style.width = w + "px"; cv.style.height = height + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, height);
    ctx.font = "11px system-ui, sans-serif";
    draw(ctx, w, height);
  }, [...deps, w]);
  return ref;
}

function frame(ctx, w, h, pad, x0, x1, y0, y1, xLabel, yLabel, xf = fmt, yf = fmt) {
  const tx = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r);
  const ty = (v) => h - pad.b - ((v - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b);
  ctx.strokeStyle = GRID; ctx.fillStyle = MUTED; ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of ticks(y0, y1)) { ctx.beginPath(); ctx.moveTo(pad.l, ty(v)); ctx.lineTo(w - pad.r, ty(v)); ctx.stroke(); ctx.fillText(yf(v), pad.l - 6, ty(v)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of ticks(x0, x1)) ctx.fillText(xf(v), tx(v), h - pad.b + 6);
  ctx.fillText(xLabel, (pad.l + w - pad.r) / 2, h - 16);
  ctx.save(); ctx.translate(12, (pad.t + h - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
  return { tx, ty };
}

/* Training and validation loss by epoch. */
export function LossChart({ history, height = 240 }) {
  const ref = useCanvas((ctx, w, h) => {
    if (!history.length) return;
    const pad = { l: 56, r: 16, t: 14, b: 40 };
    const vals = history.flatMap((d) => [d.loss, d.val]).filter(Number.isFinite);
    const y1 = Math.max(...vals), y0 = Math.min(0, Math.min(...vals));
    const { tx, ty } = frame(ctx, w, h, pad, 1, Math.max(2, history.length), y0, y1 * 1.05, "epoch", "loss (MSE, standardised)");
    const line = (key, col, dash) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash);
      ctx.beginPath();
      history.forEach((d, i) => (i ? ctx.lineTo(tx(d.epoch), ty(d[key])) : ctx.moveTo(tx(d.epoch), ty(d[key]))));
      ctx.stroke(); ctx.setLineDash([]);
    };
    line("loss", "#2563eb", []); line("val", "#ea580c", [5, 4]);
    const best = history.reduce((b, d) => (d.val < b.val ? d : b), history[0]);
    ctx.fillStyle = "#ea580c"; ctx.beginPath(); ctx.arc(tx(best.epoch), ty(best.val), 4, 0, 7); ctx.fill();
    ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText(`best validation · epoch ${best.epoch}`, Math.min(tx(best.epoch) + 6, w - 170), ty(best.val) - 6);
  }, [history], height);
  return (
    <div>
      <canvas ref={ref} />
      <div className="legend"><span><i style={{ background: "#2563eb" }} />training</span><span><i style={{ background: "#ea580c" }} />validation (cars the network does not learn from)</span></div>
    </div>
  );
}

/* Predicted against actual price, both models, with the 45° line. */
export function ScatterChart({ actual, series, unit, height = 320 }) {
  const ref = useCanvas((ctx, w, h) => {
    if (!actual.length) return;
    const pad = { l: 60, r: 16, t: 14, b: 42 };
    const all = [...actual, ...series.flatMap((s) => s.pred)];
    const hi = Math.max(...all) * 1.02, lo = Math.max(0, Math.min(...all) * 0.98);
    const { tx, ty } = frame(ctx, w, h, pad, lo, hi, lo, hi, `actual price (${unit})`, `predicted price (${unit})`);
    ctx.strokeStyle = MUTED; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(tx(lo), ty(lo)); ctx.lineTo(tx(hi), ty(hi)); ctx.stroke(); ctx.setLineDash([]);
    for (const s of series) {
      ctx.fillStyle = s.color + "66";
      s.pred.forEach((p, i) => { ctx.beginPath(); ctx.arc(tx(actual[i]), ty(p), 2.2, 0, 7); ctx.fill(); });
    }
  }, [actual, series], height);
  return (
    <div>
      <canvas ref={ref} />
      <div className="legend">{series.map((s) => <span key={s.label}><i style={{ background: s.color }} />{s.label}</span>)}<span>dashed: perfect prediction</span></div>
    </div>
  );
}

export function Bars({ items, height }) {
  const max = Math.max(...items.map((d) => d.value), 1e-9);
  return (
    <div className="bars" style={{ minHeight: height }}>
      {items.map((d) => (
        <div key={d.label} className="bar-row">
          <span className="bar-label">{d.label}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(0, d.value / max) * 100}%` }} /></span>
          <span className="bar-value">{d.text}</span>
        </div>
      ))}
    </div>
  );
}
