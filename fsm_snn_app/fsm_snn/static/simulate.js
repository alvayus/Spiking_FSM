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
    <span>${esc(b.label)}${b.desc ? `<small>${esc(b.desc)}</small>` : ""}${b.available ? "" : `<small class="why">${esc(b.reason)}</small>`}</span></label>`).join("");
  $$("input[name=backend]").forEach((r) => r.addEventListener("change", () => { sim.backend = r.value; store("fsm_snn_backend", r.value); updateTsfNote(); }));
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
// the arrows of the time step field move along round values (1-2-5) instead of a fixed increment;
// any other value can still be typed
const DT_LADDER = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
let dtBefore = null;                       // value just before an arrow click or key
function dtArrow(e) {
  const el = $("#dt"), v = parseFloat(el.value);
  if (!e.inputType && dtBefore > 0 && v > 0 && v !== dtBefore) {   // a step, not typing
    const next = v > dtBefore ? DT_LADDER.find((x) => x > dtBefore + 1e-9)
                              : [...DT_LADDER].reverse().find((x) => x < dtBefore - 1e-9);
    el.value = next ?? dtBefore;
  }
  dtBefore = parseFloat(el.value);
}
// same rounding as Python's round() on the server (half to even)
const roundSteps = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };

// "5 to 15 steps" is also shown in ms for the current time step
// smallest time scale factor SpiNNaker supports for the time step (pynn_runner.time_scale_factor)
const minTsf = (dt) => Math.max(1, Math.ceil(1 / dt - 1e-9));
// the time scale factor only applies to SpiNNaker: the field is disabled for the other simulators
function updateTsfNote() {
  const spinn = sim.backend !== "ideal";
  $("#tsf").disabled = !spinn;
  $("#tsfField").classList.toggle("disabled", !spinn);
  if (!spinn) { $("#tsfNote").innerHTML = " Only for SpiNNaker."; return; }
  const need = minTsf(curDt()), v = parseInt($("#tsf").value, 10);
  $("#tsfNote").innerHTML = v >= 1 && v < need
    ? ` <b class="warn-text">With a ${msText(curDt())} ms step SpiNNaker needs a factor of at least ${need}.</b>`
    : ` With a ${msText(curDt())} ms step: ${v >= 1 ? v : need}.`;
}
function updateGapMs() {
  const dt = curDt(), lo = Math.max(1, +$("#minGap").value), hi = Math.max(lo, +$("#maxGap").value);
  $("#gapMs").textContent = ` With a ${msText(dt)} ms step: ${msText(lo * dt)} to ${msText(hi * dt)} ms between input spikes.`;
  const g = archGap();
  $("#archNote").innerHTML = g > 1
    ? `The <b>pipelined</b> network (Network tab) guarantees correct behaviour for input spikes at least <b>${g}</b> steps apart, the time a transition needs to be written to the latches. The parallel one accepts any spacing.`
    : `The <b>parallel</b> network (Network tab) accepts any spacing, even input spikes on consecutive steps.`;
}
const archGap = () => info.architectures[archOf()].min_event_gap;

// Typed spikes are checked as they are written: the network needs the initial
// states to be set before the first event and a minimum gap between events.
function checkTyped() {
  const box = $("#manualCheck");
  if (sim.mode === "random") { box.innerHTML = ""; return; }
  const dt = curDt(), gap = archGap(), ready = info.timing.ready_step;
  const byStep = new Map(), offGrid = [];
  $$("#manualList textarea").forEach((t) => (t.value.match(/-?\d+(?:\.\d+)?/g) || []).forEach((v) => {
    const ms = parseFloat(v), step = roundSteps(ms / dt);
    if (step < 1) return;
    // input spikes happen on time steps: a time between two steps is moved to the nearest one
    if (Math.abs(ms / dt - step) > 1e-6) offGrid.push(`${msText(ms)} ms (${t.dataset.input}) → ${msText(step * dt)} ms`);
    if (!byStep.has(step)) byStep.set(step, []);
    byStep.get(step).push(`${msText(ms)} ms (${t.dataset.input})`);
  }));
  const steps = [...byStep.keys()].sort((a, b) => a - b);
  const what = (st) => byStep.get(st).join(" + ");
  const issues = [];
  if (offGrid.length) issues.push(`Not a multiple of the ${msText(dt)} ms time step, moved to the nearest step: ${offGrid.slice(0, 4).join(", ")}${offGrid.length > 4 ? ` and ${offGrid.length - 4} more` : ""}.`);
  for (const st of steps) if (st <= ready) issues.push(`${what(st)} comes before the initial states are set: use ${msText((ready + 1) * dt)} ms or later.`);
  for (let k = 1; gap > 1 && k < steps.length; k++) {
    const a = steps[k - 1], b = steps[k];
    if (b - a < gap) issues.push(`${what(b)} is only ${msText((b - a) * dt)} ms after ${what(a)}: correct behaviour is guaranteed from ${msText(gap * dt)} ms apart.`);
  }
  const shown = issues.slice(0, 6).map((m) => `<div class="msg warn">${esc(m)}</div>`).join("")
    + (issues.length > 6 ? `<div class="msg warn">…and ${issues.length - 6} more.</div>` : "");
  box.innerHTML = `<p class="note">With a ${msText(dt)} ms time step, typed spikes must come at ${msText((ready + 1) * dt)} ms or later.${gap > 1 ? ` The pipelined network guarantees correct behaviour for spikes at least ${msText(gap * dt)} ms (${gap} steps) apart.` : ""} Spikes of different inputs in the same step are evaluated together.</p>${shown}`;
}

function collectParams() {
  const manual = {};
  $$("#manualList textarea").forEach((t) => (manual[t.dataset.input] = t.value));
  store("fsm_snn_manual", manual);
  const p = {
    backend: sim.backend, dt: parseFloat($("#dt").value) > 0 ? parseFloat($("#dt").value) : 1, mode: sim.mode,
    n_events: +$("#nEvents").value, min_gap: Math.max(1, Math.round(+$("#minGap").value)), max_gap: Math.max(1, Math.round(+$("#minGap").value), Math.round(+$("#maxGap").value)),
    p_input: +$("#pInput").value, manual,
    time_scale_factor: sim.backend !== "ideal" && parseInt($("#tsf").value, 10) >= 1 ? parseInt($("#tsf").value, 10) : null,
  };
  store("fsm_snn_sim", { n_events: p.n_events, min_gap: p.min_gap, max_gap: p.max_gap, p_input: p.p_input, mode: p.mode, dt: p.dt,
    tsf: parseInt($("#tsf").value, 10) || null });   // the field, also while it is disabled
  return p;
}
function restoreSimFields() {
  const s = load("fsm_snn_sim"); if (!s) return;
  $("#nEvents").value = s.n_events; $("#minGap").value = s.min_gap; $("#maxGap").value = s.max_gap;
  $("#pInput").value = s.p_input;
  if (s.dt > 0) $("#dt").value = s.dt;
  if (s.tsf >= 1) $("#tsf").value = s.tsf;   // kept while another simulator is chosen
  setMode(["manual", "mixed"].includes(s.mode) ? s.mode : "random");
}
function setMode(m) {
  sim.mode = m;
  $$("#modeSeg button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  $("#randomFields").hidden = m === "manual"; $("#manualFields").hidden = m === "random";
  $("#manualNote").innerHTML = m === "mixed"
    ? `Spikes inserted on top of the random sequence, in milliseconds, separated by commas. They are always kept: random spikes closer than the minimum spacing to one of them are left out.`
    : "Type the spike times of each input in milliseconds, separated by commas.";
  checkTyped();
  $("#pInputOut").textContent = fmt(+$("#pInput").value, 2);
}

// ------------------------------------------------------------------ run simulation
async function simulate() {
  const p = collectParams();
  busy(true, sim.backend === "ideal" ? "Simulating…" : "Simulating with PyNN (" + sim.backend + "). This may take a while…");
  // Any simulation that lasts more than a moment can be cancelled. It stops at its next
  // safe point; SpiNNaker is then closed cleanly (it is never killed, which could leave
  // the board unusable).
  const cancel = $("#busyCancel");
  p.job = Math.random().toString(36).slice(2);
  Object.assign(cancel, { hidden: true, disabled: false, textContent: "Cancel" });
  const showCancel = setTimeout(() => (cancel.hidden = false), sim.backend === "ideal" ? 1500 : 0);
  cancel.onclick = async () => {
    Object.assign(cancel, { disabled: true, textContent: "Cancelling…" });
    $("#busyText").textContent = sim.backend === "ideal" ? "Cancelling…"
      : "Cancelling: SpiNNaker stops at its next safe point and closes cleanly. It cannot be interrupted while it loads the network on the board, so this may take a few seconds.";
    try { await api("/api/cancel", { job: p.job }); } catch (e) { /* the run reports it */ }
  };
  try {
    const r = await api("/api/simulate", { spec, params: p });
    if (r.cancelled) { toast("Simulation cancelled"); return; }
    if (!r.ok) { toast((r.errors || ["Simulation error"]).join(" · "), 6000); if (r.invalid_spec) showTab("design"); return; }
    // name the neurons as in the Network tab (with any names given there)
    try { if (await fetchNetwork()) r.netKey = netKey; } catch (e) { /* keep the netlist labels */ }
    lastResult = r; renderResults();
  } catch (e) { toast("Could not reach the server: " + e.message, 6000); }
  finally { clearTimeout(showCancel); busy(false); cancel.hidden = true; cancel.onclick = null; }
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
  const sm = r.summary || {};
  $("#simDiffs").innerHTML = `<div class="card"><h3>Differences with the reference</h3>
    <p class="note">The network is compared with the reference automaton shifted by its processing time: a state becomes active ${sm.state_latency} steps after the input spike and Output spikes ${sm.output_latency} steps after it.</p>
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
  drawRaster();
}

// Raster in the style of the paper's figures: inputs at the bottom, then OP, a
// separator line, the states (all three latch neurons on one row) and Output on
// top; dotted row guides, framed axes and time in ms. Same draw.io palette.
const RASTER_COLOR = {
  input: "#6c8ebf", init: "#666666", op: "#d79b00", state: "#82b366", latch: "#82b366",
  output: "#9673a6", term: "#9673a6", not_input: "#b85450", not_state: "#b85450", delay: "#d6b656",
  relay: "#6c8ebf", no_event: "#b85450", enter_or: "#d79b00", leave_or: "#d79b00",
};
const RASTER_ORDER = ["input", "init", "op", "relay", "delay", "not_input", "not_state", "no_event", "term",
  "enter_or", "leave_or", "latch", "state", "output"];
const isSourceKind = (k) => k === "input" || k === "init" || k === "op";
// Name of a raster row: with "All neurons", the name the neuron has in the Network tab.
function rowName(row, r) {
  if (!netData || netKey !== r.netKey) return row.label;
  const n = row.idx !== undefined ? netData.nodes["n" + row.idx]
    : row.kind === "input" ? Object.values(netData.nodes).find((x) => x.kind === "input" && x.label === row.label) : null;
  return n ? netName(n) : row.label;
}

function drawRaster() {
  const r = lastResult; if (!r) return;
  const showAll = $("#showAll").checked, showExp = $("#showExpected").checked;
  // bottom-to-top order (row 0 is drawn at the bottom)
  // all neurons: the inputs (spike sources) and every neuron, grouped by kind as in the Network tab
  const rows = showAll
    ? r.rows.filter((x) => x.kind === "input").concat(r.all_rows.map((x, i) => [x, i])
        .sort((a, b) => RASTER_ORDER.indexOf(a[0].kind) - RASTER_ORDER.indexOf(b[0].kind) || a[1] - b[1]).map((x) => x[0]))
    : r.rows;
  // both plots share the time axis: same left margin and width, scrolled together
  const avail = Math.max(600, $("#rasterScroll").clientWidth - 2), zoom = +$("#zoom").value;
  const left = showAll ? 170 : 70, right = 16;
  const W = Math.max(avail, (avail - left - right) * zoom + left + right);
  const geo = { W, left, right, T: r.duration, dt: r.dt };
  plotRaster($("#raster"), rows.map((row) => {
    const exp = new Set(row.expected || []), got = new Set(row.spikes);
    return { row, name: rowName(row, r), spikes: row.spikes,
             extra: row.expected ? row.spikes.filter((t) => !exp.has(t)) : [],
             missing: row.expected ? row.expected.filter((t) => !got.has(t)) : [] };
  }), { ...geo, rowH: showAll ? 14 : 26,
        split: (a, b) => showAll ? a.kind !== b.kind && !(isSourceKind(a.kind) && isSourceKind(b.kind))
                                 : isSourceKind(a.kind) && !isSourceKind(b.kind) });
  // expected: the FSM itself, without synaptic delays: input signals and states only
  $("#expectedBox").hidden = !showExp;
  if (showExp) {
    const fsmRows = r.rows.filter((x) => x.kind === "input" || x.kind === "state");
    plotRaster($("#rasterExp"), fsmRows.map((row) => ({ row, name: row.label, spikes: row.kind === "input" ? row.spikes : [], bars: row.fsm || [] })),
      { ...geo, rowH: 26, split: (a, b) => a.kind !== b.kind });
    $("#rasterExpScroll").scrollLeft = $("#rasterScroll").scrollLeft;
  }
}

// One raster plot in the style of the paper's figures. items: bottom-to-top rows with
// {row, name, spikes, extra, missing, bars ([first, end) step intervals)}.
function plotRaster(svg, items, { W, left, right, T, dt, rowH, split }) {
  const top = 14, n = items.length;
  const plotH = (n + 1) * rowH;                       // half a row of margin above and below
  const sx = (W - left - right) / T;
  const X = (t) => left + t * sx;
  const Y = (k) => top + plotH - (k + 1) * rowH;      // centre of row k
  const tick = Math.max(0.5, rowH * 0.32);           // half height of a spike mark
  const bw = Math.max(1.6, Math.min(sx * 0.6, 3));
  const parts = [];
  // dotted row guides and the separators between groups of rows
  items.forEach((it, k) => parts.push(`<line x1="${left}" x2="${W - right}" y1="${Y(k)}" y2="${Y(k)}" stroke="#c3b5d6" stroke-width=".8" stroke-dasharray="1.5 2.5"/>`));
  for (let k = 1; k < n; k++)
    if (split(items[k - 1].row, items[k].row)) parts.push(`<line x1="${left}" x2="${W - right}" y1="${Y(k) + rowH / 2}" y2="${Y(k) + rowH / 2}" stroke="#4b3f72" stroke-width=".6"/>`);
  items.forEach((it, k) => {
    const { row, name } = it, y = Y(k), col = RASTER_COLOR[row.kind] || "#334155";
    const extra = it.extra || [], missing = it.missing || [], bad = new Set(extra);
    parts.push(`<line x1="${left - 6}" x2="${left}" y1="${y}" y2="${y}" stroke="#111" stroke-width="1.4"/>`);
    parts.push(`<text x="${left - 10}" y="${y + 4}" text-anchor="end">${esc(name.length > 26 ? name.slice(0, 25) + "…" : name)}<title>${esc(name)}</title></text>`);
    // tinted band behind every stretch of differences, so they stand out when zoomed out
    const band = (steps, fill) => stepRanges(steps).forEach(([a, b]) => parts.push(
      `<rect x="${X(a) - Math.max(3, bw)}" y="${y - rowH / 2 + 1}" width="${X(b) - X(a) + 2 * Math.max(3, bw)}" height="${rowH - 2}" fill="${fill}"/>`));
    band(missing, "var(--miss-bg)"); band(extra, "var(--extra-bg)");
    for (const [a, b] of it.bars || []) parts.push(
      `<rect x="${X(a)}" y="${y - tick}" width="${Math.max(X(b) - X(a), bw)}" height="${2 * tick}" fill="${col}"><title>${esc(name)}: active ${fmt(a * dt, 1)}–${fmt(b * dt, 1)} ms</title></rect>`);
    for (const t of it.spikes) {
      const wrong = bad.has(t);
      parts.push(`<rect x="${X(t) - bw / 2}" y="${y - tick}" width="${bw}" height="${2 * tick}" fill="${wrong ? "var(--extra)" : col}"><title>${esc(name)}: ${fmt(t * dt, 1)} ms${wrong ? " (extra spike)" : ""}</title></rect>`);
    }
    for (const t of missing) parts.push(`<rect x="${X(t) - bw / 2 - .75}" y="${y - tick}" width="${bw + 1.5}" height="${2 * tick}" fill="none" stroke="var(--miss)" stroke-width="1.5"><title>${esc(name)}: ${fmt(t * dt, 1)} ms (missing spike)</title></rect>`);
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
  svg.setAttribute("width", W); svg.setAttribute("height", yA + 46); svg.innerHTML = parts.join("");
}

// ------------------------------------------------------------------ export the plots
// One self-contained SVG with the network activity and, when shown, the expected plot
// below it, each with its title. Computed styles are written into the elements.
function simSvgText() {
  const plots = [["Network activity", $("#raster")]];
  if (!$("#expectedBox").hidden) plots.push(["Expected FSM (ideal, without synaptic delays)", $("#rasterExp")]);
  const NS = "http://www.w3.org/2000/svg", out = document.createElementNS(NS, "svg");
  const W = Math.max(...plots.map(([, s]) => +s.getAttribute("width")));
  let y = 0;
  const bg = document.createElementNS(NS, "rect");
  out.appendChild(bg);
  for (const [title, live] of plots) {
    const t = document.createElementNS(NS, "text");
    Object.entries({ x: 12, y: y + 20, "font-family": "Segoe UI, Arial, sans-serif", "font-size": 13, "font-weight": 700, fill: "#1e293b" })
      .forEach(([k, v]) => t.setAttribute(k, v));
    t.textContent = title; out.appendChild(t);
    const g = document.createElementNS(NS, "g"), clone = live.cloneNode(true);
    const L = $$("*", live), C = $$("*", clone);
    L.forEach((el, i) => C[i].setAttribute("style", SVG_PROPS.map((k) => `${k}:${getComputedStyle(el).getPropertyValue(k)}`).join(";")));
    clone.querySelectorAll("title").forEach((x) => x.remove());
    g.setAttribute("transform", `translate(0 ${y + 28})`);
    [...clone.childNodes].forEach((c) => g.appendChild(c));
    out.appendChild(g);
    y += 28 + +live.getAttribute("height") + 10;
  }
  const H = Math.round(y);
  Object.entries({ x: 0, y: 0, width: W, height: H, fill: "#ffffff" }).forEach(([k, v]) => bg.setAttribute(k, v));
  out.setAttribute("xmlns", NS); out.setAttribute("width", W); out.setAttribute("height", H);
  out.setAttribute("viewBox", `0 0 ${W} ${H}`);
  return { text: '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(out), W, H };
}
async function saveSimImage(kind) {
  $("#simSaveMenu").hidden = true;
  if (!lastResult) return toast("Run a simulation first");
  const { text, W, H } = simSvgText();
  let body;
  try { body = kind === "svg" ? { text } : { base64: await netPngBase64(text, W, H) }; }
  catch (e) { return toast(e.message); }
  writeImage(kind, text, body, "_simulation", "the simulation plots");
}

