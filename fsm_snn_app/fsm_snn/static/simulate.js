/* Simulate tab: simulation settings, results (metrics, raster, per-event
 * table) and the stand-alone script. */
"use strict";

let lastResult = null;
let sim = { backend: "ideal", mode: "random" };

// ------------------------------------------------------------------ simulation settings
function renderBackends() {
  const saved = load("fsm_snn_backend");
  const ok = (id) => info.backends.find((b) => b.id === id && b.available);
  sim.backend = saved && ok(saved) ? saved : "ideal";
  $("#backendList").innerHTML = info.backends.map((b) => `<label class="${b.available ? "" : "disabled"}">
    <input type="radio" name="backend" value="${b.id}" ${b.available ? "" : "disabled"} ${b.id === sim.backend ? "checked" : ""}>
    <span>${esc(b.label)}${b.available ? "" : `<small>${esc(b.reason)}</small>`}</span></label>`).join("");
  $$("input[name=backend]").forEach((r) => r.addEventListener("change", () => { sim.backend = r.value; store("fsm_snn_backend", r.value); }));
}
function renderManualFields() {
  const old = {};
  $$("#manualList textarea").forEach((t) => (old[t.dataset.input] = t.value));
  const saved = load("fsm_snn_manual") || {};
  $("#manualList").innerHTML = spec.inputs.map((i) => `<label class="field">${esc(i)} <span class="unit">ms</span>
    <textarea rows="2" data-input="${esc(i)}" placeholder="e.g. 10, 25, 40">${esc(old[i] ?? saved[i] ?? "")}</textarea></label>`).join("");
  checkTyped();
}
// ------------------------------------------------------------------ timing hints (in ms)
const curDt = () => (parseFloat($("#dt").value) > 0 ? parseFloat($("#dt").value) : 1);
const msText = (x) => String(+x.toFixed(3));
// same rounding as Python's round() on the server (half to even)
const roundSteps = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };

// "5 to 15 steps" is also shown in ms for the current time step
function updateGapMs() {
  const dt = curDt(), lo = Math.max(5, +$("#minGap").value), hi = Math.max(lo, +$("#maxGap").value);
  $("#gapMs").textContent = ` With a ${msText(dt)} ms step: ${msText(lo * dt)} to ${msText(hi * dt)} ms between events.`;
}

// Typed spikes are checked as they are written: the network needs the initial
// states to be set before the first event and a minimum gap between events.
function checkTyped() {
  const box = $("#manualCheck");
  if (sim.mode === "random") { box.innerHTML = ""; return; }
  const dt = curDt(), gap = info.timing.min_event_gap, ready = info.timing.ready_step;
  const byStep = new Map();
  $$("#manualList textarea").forEach((t) => (t.value.match(/-?\d+(?:\.\d+)?/g) || []).forEach((v) => {
    const ms = parseFloat(v), step = roundSteps(ms / dt);
    if (step < 1) return;
    if (!byStep.has(step)) byStep.set(step, []);
    byStep.get(step).push(`${msText(ms)} ms (${t.dataset.input})`);
  }));
  const steps = [...byStep.keys()].sort((a, b) => a - b);
  const what = (st) => byStep.get(st).join(" + ");
  const issues = [];
  for (const st of steps) if (st <= ready) issues.push(`${what(st)} comes before the initial states are set: use ${msText((ready + 1) * dt)} ms or later.`);
  for (let k = 1; k < steps.length; k++) {
    const a = steps[k - 1], b = steps[k];
    if (b - a < gap) issues.push(`${what(b)} is only ${msText((b - a) * dt)} ms after ${what(a)}: they must be at least ${msText(gap * dt)} ms apart.`);
  }
  const shown = issues.slice(0, 6).map((m) => `<div class="msg warn">${esc(m)}</div>`).join("")
    + (issues.length > 6 ? `<div class="msg warn">…and ${issues.length - 6} more.</div>` : "");
  box.innerHTML = `<p class="note">With a ${msText(dt)} ms time step, typed spikes must come at ${msText((ready + 1) * dt)} ms or later and be at least ${msText(gap * dt)} ms (${gap} steps) apart. Spikes of different inputs in the same step form one event.</p>${shown}`;
}

function collectParams() {
  const manual = {};
  $$("#manualList textarea").forEach((t) => (manual[t.dataset.input] = t.value));
  store("fsm_snn_manual", manual);
  const p = {
    backend: sim.backend, dt: parseFloat($("#dt").value) > 0 ? parseFloat($("#dt").value) : 1, mode: sim.mode,
    n_events: +$("#nEvents").value, min_gap: Math.max(5, +$("#minGap").value), max_gap: Math.max(5, +$("#maxGap").value),
    p_input: +$("#pInput").value, manual,
  };
  store("fsm_snn_sim", { n_events: p.n_events, min_gap: p.min_gap, max_gap: p.max_gap, p_input: p.p_input, mode: p.mode, dt: p.dt });
  return p;
}
function restoreSimFields() {
  const s = load("fsm_snn_sim"); if (!s) return;
  $("#nEvents").value = s.n_events; $("#minGap").value = s.min_gap; $("#maxGap").value = s.max_gap;
  $("#pInput").value = s.p_input;
  if (s.dt > 0) $("#dt").value = s.dt;
  setMode(["manual", "mixed"].includes(s.mode) ? s.mode : "random");
}
function setMode(m) {
  sim.mode = m;
  $$("#modeSeg button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  $("#randomFields").hidden = m === "manual"; $("#manualFields").hidden = m === "random";
  $("#manualNote").innerHTML = m === "mixed"
    ? `Spikes inserted on top of the random sequence, in milliseconds, separated by commas. They are always kept: random events closer than ${info.timing.min_event_gap} steps to one of them are left out.`
    : "Type the spike times of each input in milliseconds, separated by commas.";
  checkTyped();
  $("#pInputOut").textContent = fmt(+$("#pInput").value, 2);
}

// ------------------------------------------------------------------ run simulation
async function simulate() {
  const p = collectParams();
  busy(true, sim.backend === "ideal" ? "Simulating…" : "Simulating with PyNN (" + sim.backend + "). This may take a while…");
  try {
    const r = await api("/api/simulate", { spec, params: p });
    if (!r.ok) { toast((r.errors || ["Simulation error"]).join(" · "), 6000); if (r.invalid_spec) showTab("design"); return; }
    lastResult = r; renderResults();
  } catch (e) { toast("Could not reach the server: " + e.message, 6000); }
  finally { busy(false); }
}
// Spikes of the states (and Output) that differ from the reference automaton.
function spikeDiffs(r) {
  return r.rows.filter((row) => row.expected).map((row) => {
    const exp = new Set(row.expected), got = new Set(row.spikes);
    return { label: row.label, missing: row.expected.filter((t) => !got.has(t)), extra: row.spikes.filter((t) => !exp.has(t)) };
  }).filter((d) => d.missing.length || d.extra.length);
}
// consecutive steps -> [first, last] ranges (a state that stays active spikes every step)
function stepRanges(steps) {
  const out = [];
  for (const t of steps) {
    const last = out[out.length - 1];
    if (last && t === last[1] + 1) last[1] = t; else out.push([t, t]);
  }
  return out;
}
function rangesText(steps, dt) {
  if (!steps.length) return "—";
  const rs = stepRanges(steps).map(([a, b]) => (a === b ? `${msText(a * dt)}` : `${msText(a * dt)}–${msText(b * dt)}`) + " ms");
  return esc(rs.slice(0, 4).join(", ") + (rs.length > 4 ? ` and ${rs.length - 4} more` : "")) + ` <span class="muted">(${steps.length})</span>`;
}
function renderDiffs(r, diffs) {
  if (!diffs.length) { $("#simDiffs").innerHTML = ""; return; }
  const nMiss = diffs.reduce((a, d) => a + d.missing.length, 0), nExtra = diffs.reduce((a, d) => a + d.extra.length, 0);
  const chip = (n, cls, what) => `<span class="diff-chip ${n ? cls : "none"}">${n} ${what} spike${n === 1 ? "" : "s"}</span>`;
  $("#simDiffs").innerHTML = `<div class="card"><h3>Differences with the reference</h3>
    <div class="diff-totals">${chip(nMiss, "miss", "missing")}${chip(nExtra, "extra", "extra")}</div>
    <div class="tbl-wrap"><table class="tbl diff-tbl"><thead><tr><th></th><th>Missing spikes</th><th>Extra spikes</th></tr></thead><tbody>
    ${diffs.map((d) => `<tr><td><b>${esc(d.label)}</b></td><td class="miss">${rangesText(d.missing, r.dt)}</td><td class="extra">${rangesText(d.extra, r.dt)}</td></tr>`).join("")}
    </tbody></table></div></div>`;
}

function renderResults() {
  const r = lastResult, m = r.metrics;
  $("#simEmpty").hidden = true; $("#simOut").hidden = false;
  const diffs = spikeDiffs(r);
  const good = !diffs.length && m.state_agreement === 1 && m.final_state_ok !== false;
  renderDiffs(r, diffs);
  $$(".lg-diff").forEach((e) => (e.hidden = !diffs.length));
  const msgs = (r.warnings || []).map((w) => `<div class="msg warn">${esc(w)}</div>`)
    .concat((r.notes || []).map((n) => `<div class="msg info">${esc(n)}</div>`));
  // with the ideal simulator a match is the expected outcome: only report problems
  if (!good) msgs.unshift(`<div class="msg err">The network does not match the reference automaton: the differences are listed below and highlighted in the plot (extra spikes in red, missing spikes in orange).</div>`);
  else if (r.backend !== "ideal") msgs.unshift(`<div class="msg ok">The network behaves exactly like the reference automaton.</div>`);
  $("#simMessages").innerHTML = msgs.join("");
  const typed = new Set(r.mode === "mixed" ? r.manual_steps : []);    // spikes typed in "Random + manual"
  const rows = r.per_event.map((e) => `<tr class="${e.ok ? "" : "bad"}"><td class="num">${msText(e.step * r.dt)}</td><td>${esc(e.inputs.join(", "))}${typed.has(e.step) ? ' <span class="tag">manual</span>' : ""}</td><td>${esc(e.expected.join(", ") || "—")}</td><td>${esc(e.obtained.join(", ") || "—")}</td><td>${e.ok ? "✓" : "✗"}</td></tr>`).join("");
  $("#eventTable").innerHTML = `<div class="tbl-wrap"><table class="tbl"><thead><tr><th class="num">Time (ms)</th><th>Inputs that spike</th><th>Expected states</th><th>Obtained states</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  $("#savedTo").textContent = [r.mode !== "manual" ? `Random seed ${r.seed}` : "", r.saved_to ? "results saved in " + r.saved_to : ""]
    .filter(Boolean).join(" · ").replace(/^r/, "R");
  // the ideal simulator is the reference behaviour itself: no expected overlay
  // (mismatches, e.g. events closer than the minimum gap, are still marked in red)
  const ideal = r.backend === "ideal";
  $("#showExpectedField").hidden = ideal; $("#legendExpected").hidden = ideal;
  drawRaster();
}

// Raster in the style of the paper's figures: inputs at the bottom, then OP, a
// separator line, the states (all three latch neurons on one row) and Output on
// top; dotted row guides, framed axes and time in ms. Same draw.io palette.
const RASTER_COLOR = {
  input: "#6c8ebf", init: "#666666", op: "#d79b00", state: "#82b366", latch: "#82b366",
  output: "#9673a6", term: "#9673a6", not_input: "#b85450", not_state: "#b85450", delay: "#d6b656",
};
const RASTER_ORDER = ["input", "init", "op", "delay", "not_input", "not_state", "term", "latch", "state", "output"];
const isSourceKind = (k) => k === "input" || k === "init" || k === "op";

function drawRaster() {
  const r = lastResult; if (!r) return;
  const showAll = $("#showAll").checked, showExp = r.backend !== "ideal" && $("#showExpected").checked;
  // bottom-to-top order (row 0 is drawn at the bottom)
  const rows = showAll
    ? r.all_rows.map((x, i) => [x, i]).sort((a, b) => RASTER_ORDER.indexOf(a[0].kind) - RASTER_ORDER.indexOf(b[0].kind) || a[1] - b[1]).map((x) => x[0])
    : r.rows;
  const n = rows.length, zoom = +$("#zoom").value;
  const avail = Math.max(600, $("#rasterScroll").clientWidth - 2);
  const left = showAll ? 130 : 70, right = 16, top = 14, rowH = showAll ? 14 : 26;
  const plotH = (n + 1) * rowH;                       // half a row of margin above and below
  const T = r.duration, dt = r.dt;
  const W = Math.max(avail, (avail - left - right) * zoom + left + right);
  const sx = (W - left - right) / T;
  const X = (t) => left + t * sx;
  const Y = (k) => top + plotH - (k + 1) * rowH;      // centre of row k
  const tick = Math.max(0.5, rowH * 0.32);           // half height of a spike mark
  const bw = Math.max(1.6, Math.min(sx * 0.6, 3));
  const parts = [];
  // dotted row guides and the separator between the inputs/OP and the network state
  rows.forEach((row, k) => parts.push(`<line x1="${left}" x2="${W - right}" y1="${Y(k)}" y2="${Y(k)}" stroke="#c3b5d6" stroke-width=".8" stroke-dasharray="1.5 2.5"/>`));
  for (let k = 1; k < n; k++) {
    const split = showAll ? rows[k].kind !== rows[k - 1].kind && !(isSourceKind(rows[k].kind) && isSourceKind(rows[k - 1].kind))
                          : isSourceKind(rows[k - 1].kind) && !isSourceKind(rows[k].kind);
    if (split) parts.push(`<line x1="${left}" x2="${W - right}" y1="${Y(k) + rowH / 2}" y2="${Y(k) + rowH / 2}" stroke="#4b3f72" stroke-width=".6"/>`);
  }
  rows.forEach((row, k) => {
    const y = Y(k), col = RASTER_COLOR[row.kind] || "#334155";
    parts.push(`<line x1="${left - 6}" x2="${left}" y1="${y}" y2="${y}" stroke="#111" stroke-width="1.4"/>`);
    parts.push(`<text x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(row.label.length > 20 ? row.label.slice(0, 19) + "…" : row.label)}</text>`);
    const exp = new Set(row.expected || []), got = new Set(row.spikes);
    const extra = row.expected ? row.spikes.filter((t) => !exp.has(t)) : [];
    const missing = row.expected ? row.expected.filter((t) => !got.has(t)) : [];
    // tinted band behind every stretch of differences, so they stand out when zoomed out
    const band = (steps, fill) => stepRanges(steps).forEach(([a, b]) => parts.push(
      `<rect x="${X(a) - Math.max(3, bw)}" y="${y - rowH / 2 + 1}" width="${X(b) - X(a) + 2 * Math.max(3, bw)}" height="${rowH - 2}" fill="${fill}"/>`));
    band(missing, "var(--miss-bg)"); band(extra, "var(--extra-bg)");
    if (showExp && row.expected) for (const t of row.expected) parts.push(`<rect x="${X(t) - bw * 1.4}" y="${y - tick - 2}" width="${bw * 2.8}" height="${2 * tick + 4}" fill="#e2e8f0"/>`);
    for (const t of row.spikes) {
      const bad = row.expected && !exp.has(t);
      parts.push(`<rect x="${X(t) - bw / 2}" y="${y - tick}" width="${bw}" height="${2 * tick}" fill="${bad ? "var(--extra)" : col}"><title>${esc(row.label)}: ${fmt(t * dt, 1)} ms${bad ? " (extra spike)" : ""}</title></rect>`);
    }
    for (const t of missing) parts.push(`<rect x="${X(t) - bw / 2 - .75}" y="${y - tick}" width="${bw + 1.5}" height="${2 * tick}" fill="none" stroke="var(--miss)" stroke-width="1.5"><title>${esc(row.label)}: ${fmt(t * dt, 1)} ms (missing spike)</title></rect>`);
  });
  // frame and time axis (ms)
  const yA = top + plotH;
  parts.push(`<rect x="${left}" y="${top}" width="${W - left - right}" height="${plotH}" fill="none" stroke="#111" stroke-width="1.4"/>`);
  const msTick = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].find((s) => (s / dt) * sx >= 70) || 20000;
  for (let ms = 0; ms <= T * dt + 1e-9; ms += msTick) {
    const x = X(ms / dt);
    parts.push(`<line x1="${x}" x2="${x}" y1="${yA}" y2="${yA + 5}" stroke="#111" stroke-width="1.4"/><text x="${x}" y="${yA + 18}" text-anchor="middle">${+ms.toFixed(2)}</text>`);
  }
  parts.push(`<text x="${(left + W - right) / 2}" y="${yA + 36}" text-anchor="middle">Time (ms)</text>`);
  const s = $("#raster"); s.setAttribute("width", W); s.setAttribute("height", yA + 46); s.innerHTML = parts.join("");
}

async function downloadScript() {
  const r = await api("/api/script", { spec, params: collectParams() });
  if (!r.ok) return toast((r.errors || ["Error"]).join(" · "), 5000);
  download(r.filename, r.text, "text/x-python");
  toast("Script downloaded: run it with \"python " + r.filename + " --backend spinnaker\"", 5000);
}
