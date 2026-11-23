/* Simulate and Scaling tabs: simulation settings, results (metrics, raster,
 * per-event table), stand-alone script and the scaling study. */
"use strict";

let lastResult = null;
let scaleRows = null;
let sim = { backend: "ideal", mode: "planted" };

// ------------------------------------------------------------------ simulation settings
function renderBackends() {
  const saved = load("fsm_snn_backend");
  const ok = (id) => info.backends.find((b) => b.id === id && b.available);
  sim.backend = saved && ok(saved) ? saved : "ideal";
  $("#backendList").innerHTML = info.backends.map((b) => `<label class="${b.available ? "" : "disabled"}">
    <input type="radio" name="backend" value="${b.id}" ${b.available ? "" : "disabled"} ${b.id === sim.backend ? "checked" : ""}>
    <span>${esc(b.label)}${b.available ? "" : `<small>${esc(b.reason)}</small>`}</span></label>`).join("");
  $$("input[name=backend]").forEach((r) => r.addEventListener("change", () => { sim.backend = r.value; store("fsm_snn_backend", r.value); updateDt(); }));
  updateDt();
}
function updateDt() { const pynn = sim.backend !== "ideal"; $("#dt").disabled = !pynn; $("#dtField").style.opacity = pynn ? 1 : .5; }
function renderManualFields() {
  const old = {};
  $$("#manualList textarea").forEach((t) => (old[t.dataset.input] = t.value));
  const saved = load("fsm_snn_manual") || {};
  $("#manualList").innerHTML = spec.inputs.map((i) => `<label class="field">${esc(i)} <span class="unit">ms</span>
    <textarea rows="2" data-input="${esc(i)}" placeholder="e.g. 10, 25, 40">${esc(old[i] ?? saved[i] ?? "")}</textarea></label>`).join("");
}
function renderParams() {
  $("#wExc").value = info.weights.exc; $("#wInh").value = info.weights.inh;
  const labels = { cm: "cm (nF)", tau_m: "tau_m (ms)", tau_refrac: "tau_refrac (ms)", tau_syn_E: "tau_syn_E (ms)", tau_syn_I: "tau_syn_I (ms)", v_rest: "v_rest (mV)", v_reset: "v_reset (mV)", v_thresh: "v_thresh (mV)" };
  $("#neuronParams").innerHTML = Object.entries(info.neuron_params).map(([k, v]) =>
    `<label class="field">${labels[k] || k}<input type="number" step="any" data-param="${k}" value="${v}"></label>`).join("");
}
function collectParams() {
  const manual = {};
  $$("#manualList textarea").forEach((t) => (manual[t.dataset.input] = t.value));
  store("fsm_snn_manual", manual);
  const np = {};
  $$("[data-param]").forEach((i) => (np[i.dataset.param] = parseFloat(i.value)));
  const p = {
    backend: sim.backend, dt: parseFloat($("#dt").value), mode: sim.mode,
    n_events: +$("#nEvents").value, min_gap: Math.max(5, +$("#minGap").value), max_gap: Math.max(5, +$("#maxGap").value),
    p_input: +$("#pInput").value, planted: +$("#planted").value, seed: +$("#seed").value, manual,
    weights: { exc: parseFloat($("#wExc").value), inh: parseFloat($("#wInh").value) }, neuron_params: np,
  };
  if (sim.backend === "ideal") p.dt = 1.0;
  store("fsm_snn_sim", { n_events: p.n_events, min_gap: p.min_gap, max_gap: p.max_gap, p_input: p.p_input, planted: p.planted, seed: p.seed, mode: p.mode });
  return p;
}
function restoreSimFields() {
  const s = load("fsm_snn_sim"); if (!s) return;
  $("#nEvents").value = s.n_events; $("#minGap").value = s.min_gap; $("#maxGap").value = s.max_gap;
  $("#pInput").value = s.p_input; $("#planted").value = s.planted; $("#seed").value = s.seed;
  setMode(s.mode || "planted");
}
function setMode(m) {
  sim.mode = m;
  $$("#modeSeg button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  $("#randomFields").hidden = m === "manual"; $("#manualFields").hidden = m !== "manual";
  $("#plantedField").hidden = m !== "planted";
  $("#pInputOut").textContent = fmt(+$("#pInput").value, 2);
}

// ------------------------------------------------------------------ run simulation
async function simulate() {
  const p = collectParams();
  busy(true, sim.backend === "ideal" ? "Simulating…" : "Simulating with PyNN (" + sim.backend + "). This may take a while…");
  try {
    const r = await api("/api/simulate", { spec, params: p });
    if (!r.ok) { toast((r.errors || ["Simulation error"]).join(" · "), 6000); if (r.errors) showTab("design"); return; }
    lastResult = r; renderResults();
  } catch (e) { toast("Could not reach the server: " + e.message, 6000); }
  finally { busy(false); }
}
function renderResults() {
  const r = lastResult, m = r.metrics, c = r.summary;
  $("#simEmpty").hidden = true; $("#simOut").hidden = false;
  const card = (lbl, val, sub, cls = "") => `<div class="metric ${cls}"><div class="lbl">${lbl}</div><div class="val">${val}</div><div class="sub">${sub}</div></div>`;
  const good = m.state_agreement === 1 && m.fp === 0 && m.fn === 0 && m.final_state_ok !== false;
  $("#metrics").innerHTML = [
    card("State agreement", pct(m.state_agreement), `${m.events} events compared with the reference`, m.state_agreement === 1 ? "good" : "bad"),
    card("Network", `${c.neurons} · ${c.synapses}`, "neurons · synapses"),
    card("Synaptic events", fmt(m.synaptic_events_per_event, 1), "per input event (energy proxy)"),
    card("Simulator", esc(r.backend), `${fmt(m.wall_time_s, 2)} s · offset ${m.offset} steps`),
  ].join("");
  const msgs = (r.warnings || []).map((w) => `<div class="msg warn">${esc(w)}</div>`);
  msgs.unshift(good ? `<div class="msg ok">The network behaves exactly like the reference automaton.</div>`
    : `<div class="msg err">There are mismatches with the reference automaton: check the red marks in the plot and the per-event detail.</div>`);
  $("#simMessages").innerHTML = msgs.join("");
  const rows = r.per_event.map((e) => `<tr class="${e.ok ? "" : "bad"}"><td class="num">${e.step}</td><td>${esc(e.inputs.join(", "))}</td><td>${esc(e.expected.join(", ") || "—")}</td><td>${esc(e.obtained.join(", ") || "—")}</td><td>${e.ok ? "✓" : "✗"}</td></tr>`).join("");
  $("#eventTable").innerHTML = `<div class="tbl-wrap"><table class="tbl"><thead><tr><th class="num">Step</th><th>Inputs that spike</th><th>Expected states</th><th>Obtained states</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  $("#savedTo").textContent = r.saved_to ? "Results saved in " + r.saved_to : "";
  drawRaster();
}

function drawRaster() {
  const r = lastResult; if (!r) return;
  const showAll = $("#showAll").checked, showExp = $("#showExpected").checked;
  const rows = showAll ? r.all_rows : r.rows;
  const zoom = +$("#zoom").value;
  const avail = Math.max(600, $("#rasterScroll").clientWidth - 2);
  const left = 120, rowH = showAll ? 16 : 24, top = 10;
  const T = r.duration;
  const W = Math.max(avail, (avail - left) * zoom + left);
  const sx = (W - left - 14) / T;
  const H = top + rows.length * rowH + 30;
  const color = { input: "var(--inp)", op: "var(--op)", state: "var(--st)", output: "var(--out)", latch: "var(--st)", term: "#9673a6", not_input: "var(--inh)", not_state: "var(--inh)", delay: "#d6b656" };   // paper colours
  const bw = Math.max(1.5, Math.min(sx * 0.8, 4));
  const parts = [];
  // event guide lines
  for (const [e] of r.events) parts.push(`<line x1="${left + e * sx}" x2="${left + e * sx}" y1="${top}" y2="${top + rows.length * rowH}" stroke="#e5ebf2"/>`);
  rows.forEach((row, k) => {
    const y = top + k * rowH;
    parts.push(`<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="${k % 2 ? "#fafbfc" : "#fff"}" opacity=".6"/>`);
    parts.push(`<text x="${left - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(row.label.length > 18 ? row.label.slice(0, 17) + "…" : row.label)}</text>`);
    const exp = new Set(row.expected || []), got = new Set(row.spikes);
    if (showExp && row.expected) for (const t of row.expected) parts.push(`<rect x="${left + t * sx - bw}" y="${y + 3}" width="${2 * bw}" height="${rowH - 6}" fill="#e2e8f0"/>`);
    for (const t of row.spikes) {
      const bad = row.expected && !exp.has(t);
      parts.push(`<rect x="${left + t * sx - bw / 2}" y="${y + 5}" width="${bw}" height="${rowH - 10}" fill="${bad ? "var(--err)" : color[row.kind] || "#334155"}"/>`);
    }
    if (row.expected) for (const t of row.expected) if (!got.has(t)) parts.push(`<rect x="${left + t * sx - bw / 2}" y="${y + rowH - 6}" width="${bw}" height="4" fill="var(--err)"/>`);
  });
  // time axis
  const yA = top + rows.length * rowH;
  const stepTick = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000].find((s) => s * sx >= 60) || 10000;
  parts.push(`<line x1="${left}" x2="${W - 10}" y1="${yA}" y2="${yA}" stroke="#94a3b8"/>`);
  for (let t = 0; t <= T; t += stepTick) parts.push(`<line x1="${left + t * sx}" x2="${left + t * sx}" y1="${yA}" y2="${yA + 4}" stroke="#94a3b8"/><text x="${left + t * sx}" y="${yA + 16}" text-anchor="middle">${t}</text>`);
  parts.push(`<text x="${W - 12}" y="${yA + 28}" text-anchor="end">time (steps of ${fmt(r.dt, 1)} ms)</text>`);
  const s = $("#raster"); s.setAttribute("width", W); s.setAttribute("height", H + 12); s.innerHTML = parts.join("");
}

async function downloadScript() {
  const r = await api("/api/script", { spec, params: collectParams() });
  if (!r.ok) return toast((r.errors || ["Error"]).join(" · "), 5000);
  download(r.filename, r.text, "text/x-python");
  toast("Script downloaded: run it with \"python " + r.filename + " --backend spinnaker\"", 5000);
}

// ------------------------------------------------------------------ scaling
async function runScaling() {
  const body = { family: $("#family").value, n_min: +$("#nMin").value, n_max: +$("#nMax").value, simulate: $("#scaleSim").checked, params: collectParams() };
  if (body.n_max < body.n_min) return toast("\"n to\" must be greater than or equal to \"n from\"");
  busy(true, "Running the scaling study…");
  try {
    const r = await api("/api/scaling", body);
    if (!r.ok) return toast((r.errors || ["Error"]).join(" · "), 6000);
    scaleRows = r.rows; renderScaling(body.simulate);
  } finally { busy(false); }
}
function renderScaling(simulated) {
  $("#scaleEmpty").hidden = true; $("#scaleOut").hidden = false; $("#btnScalingCsv").disabled = false;
  const cols = [["n", "n"], ["states", "States"], ["transitions", "Transitions"], ["neurons", "Neurons"], ["synapses", "Synapses"], ["output_latency", "Output latency (steps)"], ["min_event_gap", "Min. interval (steps)"]];
  if (simulated) cols.push(["state_agreement", "Agreement"], ["tp", "TP"], ["fp", "FP"], ["fn", "FN"], ["synaptic_events_per_event", "Syn. events/event"]);
  const head = cols.map(([, l]) => `<th class="num">${l}</th>`).join("");
  const body = scaleRows.map((r) => `<tr class="${simulated && (r.state_agreement !== 1 || r.fp || r.fn) ? "bad" : ""}">${cols.map(([k]) => `<td class="num">${k === "state_agreement" ? pct(r[k]) : k === "synaptic_events_per_event" ? fmt(r[k], 1) : r[k]}</td>`).join("")}</tr>`).join("");
  $("#scaleTable").innerHTML = `<div class="tbl-wrap"><table class="tbl"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
    <p class="note">${slopeText()}</p>`;
  drawScaleChart();
}
function slopeText() {
  if (scaleRows.length < 2) return "";
  const a = scaleRows[0], b = scaleRows[scaleRows.length - 1], dn = b.n - a.n;
  return `Average growth: ${fmt((b.neurons - a.neurons) / dn, 1)} neurons and ${fmt((b.synapses - a.synapses) / dn, 1)} synapses per added state; latency does not depend on n.`;
}
function drawScaleChart() {
  const s = $("#scaleChart"); const W = s.clientWidth || 700, H = 300, L = 54, B = 40, Tp = 16, Rt = 120;
  const xs = scaleRows.map((r) => r.n), ymax = Math.max(...scaleRows.map((r) => r.synapses)) * 1.08;
  const x = (n) => L + (n - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0]) * (W - L - Rt);
  const y = (v) => H - B - v / ymax * (H - B - Tp);
  const parts = [];
  const tick = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000].find((t) => ymax / t <= 6) || 2000;
  for (let v = 0; v <= ymax; v += tick) parts.push(`<line x1="${L}" x2="${W - Rt}" y1="${y(v)}" y2="${y(v)}" stroke="#eef1f4"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`);
  for (const n of xs) parts.push(`<text x="${x(n)}" y="${H - B + 16}" text-anchor="middle">${n}</text>`);
  parts.push(`<text x="${(L + W - Rt) / 2}" y="${H - 6}" text-anchor="middle">n (machine size)</text>`);
  const series = [["synapses", "Synapses", "#2b6cb0"], ["neurons", "Neurons", "#3c9a5f"]];
  for (const [k, lbl, col] of series) {
    const pts = scaleRows.map((r) => `${x(r.n)},${y(r[k])}`).join(" ");
    parts.push(`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2.5"/>`);
    for (const r of scaleRows) parts.push(`<circle cx="${x(r.n)}" cy="${y(r[k])}" r="3.5" fill="${col}"><title>${lbl}: ${r[k]} (n = ${r.n})</title></circle>`);
    const lr = scaleRows[scaleRows.length - 1];
    parts.push(`<text x="${x(lr.n) + 10}" y="${y(lr[k]) + 4}" fill="${col}" style="font-weight:600;fill:${col}">${lbl} (${lr[k]})</text>`);
  }
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.innerHTML = parts.join("");
}
function scalingCsv() {
  const keys = Object.keys(scaleRows[0]);
  const lines = [keys.join(",")].concat(scaleRows.map((r) => keys.map((k) => JSON.stringify(r[k] ?? "")).join(",")));
  download("scaling_" + $("#family").value + ".csv", lines.join("\n"), "text/csv");
}
