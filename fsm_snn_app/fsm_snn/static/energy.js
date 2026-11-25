/* Power tab: theoretical dynamic energy of the network, estimated from the spikes of the
 * last simulation with the energy model of the spiking ALU work (fsm_snn/energy.py). It
 * depends only on the spikes, not on the simulator that produced them. */
"use strict";

let energyStatic = null;           // /api/energy for the current network (energy per spike of each neuron)
let energyKey = null;
// the three kinds of events of the formula (fixed order: generation, communication, pulses)
const ENERGY_PARTS = [
  { key: "generation", name: "Spike generation and encoding", terms: "E<sub>spike</sub> + E<sub>en</sub>", color: "#2a78d6" },
  { key: "communication", name: "Broadcast and routing", terms: "E<sub>br</sub> + E<sub>rt</sub>", color: "#eb6834" },
  { key: "pulses", name: "Pulse extension", terms: "N<sub>cam-match</sub> E<sub>pulse</sub>", color: "#1baf7a" },
];

// pJ -> readable energy; W -> readable power
function fmtEnergy(pj) {
  const units = [[1e12, "J"], [1e9, "mJ"], [1e6, "µJ"], [1e3, "nJ"], [1, "pJ"]];
  const [f, u] = units.find(([f]) => Math.abs(pj) >= f) || units[units.length - 1];
  return `${+(pj / f).toPrecision(4)} ${u}`;
}
function fmtPower(w) {
  if (w == null) return "—";
  const units = [[1, "W"], [1e-3, "mW"], [1e-6, "µW"], [1e-9, "nW"], [1e-12, "pW"]];
  const [f, u] = units.find(([f]) => Math.abs(w) >= f) || units[units.length - 1];
  return `${+(w / f).toPrecision(4)} ${u}`;
}
const fmtInt = (n) => n.toLocaleString("en-US");
const pct = (a, b) => (b > 0 ? `${+(100 * a / b).toFixed(1)} %` : "—");

async function loadEnergy() {
  const key = netKeyOf(spec);
  if (key !== energyKey || !energyStatic) {
    const r = await api("/api/energy", { spec });
    energyStatic = r.ok ? r : null; energyKey = r.ok ? key : null;
    if (!r.ok) { $("#energyOut").innerHTML = `<div class="msg err">${esc((r.errors || ["Unknown error"]).join(" · "))}</div>`; return; }
  }
  try { await fetchNetwork(); } catch (e) { /* netlist labels */ }
  renderEnergy();
}

// name of a neuron as in the Network tab (with any renaming done there)
function energyName(row) {
  const n = netData && netKey === energyKey ? netData.nodes["n" + row.idx] : null;
  return n ? netName(n) : row.label;
}
const kindName = (k) => (NET_STYLE[k] || NET_STYLE.output).name;
const kindDot = (k) => { const s = NET_STYLE[k] || NET_STYLE.output; return `<i class="dot" style="background:${s.fill};border-color:${s.stroke}"></i>`; };

function renderEnergy() {
  const st = energyStatic; if (!st) return;
  const r = lastResult;
  // the energy of a run applies only while the network is the one that was simulated
  const fresh = r && r.energy && r.netKey === energyKey;
  const run = fresh ? r.energy.run : null;
  const rows = (fresh ? r.energy.neurons : st.neurons).slice();
  rows.sort(fresh ? (a, b) => b.energy - a.energy || a.idx - b.idx : (a, b) => b.per_spike - a.per_spike || a.idx - b.idx);
  $("#energyOut").innerHTML = (run ? energyRunCard(r, run) + energyPartsCard(run, rows) : energyNoRun(r))
    + energyNeuronsCard(rows, !!run);
}

// Energy of the neurons of each type, most first.
function energyByKind(rows) {
  const byKind = new Map();
  for (const x of rows) {
    const k = byKind.get(x.kind) || { kind: x.kind, neurons: 0, spikes: 0, energy: 0 };
    k.neurons++; k.spikes += x.spikes; k.energy += x.energy; byKind.set(x.kind, k);
  }
  return [...byKind.values()].sort((a, b) => b.energy - a.energy);
}

// Title of the first card, with the menu to save the summary.
function energyHead(title) {
  return `<div class="raster-head"><h3>${title}</h3>
    <div class="split" id="energySaveWrap">
      <button id="btnEnergySave" class="btn small" title="Save the summary of the power analysis" aria-haspopup="menu">Save summary ▾</button>
      <div class="menu" id="energySaveMenu" role="menu" hidden>
        <button class="menu-item" data-fmt="md"><b>Markdown</b><span>Report with the model, the results and every neuron</span></button>
        <button class="menu-item" data-fmt="csv"><b>CSV</b><span>Table of the neurons, for a spreadsheet</span></button>
      </div>
    </div></div>`;
}

function energyNoRun(r) {
  const why = r ? "The machine or the implementation has changed since the last simulation." : "There is no simulation yet.";
  return `<div class="card">${energyHead("Power consumption of a simulation")}
    <div class="msg info">${why} Simulate it to estimate the energy and power consumed by its spikes. Below, the energy that each neuron takes per spike.</div>
    <button class="btn primary energy-go" onclick="showTab('simulate')">Go to Simulate</button></div>`;
}

function energyRunCard(r, run) {
  const tile = (value, label, sub) => `<div class="kpi"><div class="kpi-value">${value}</div><div class="kpi-label">${label}</div>${sub ? `<div class="kpi-sub">${sub}</div>` : ""}</div>`;
  return `<div class="card">${energyHead("Last simulation")}
    <div class="kpis">
      ${tile(fmtEnergy(run.energy_pj), "Dynamic energy", "of all the neuron spikes")}
      ${tile(fmtPower(run.power_w), "Mean power", "energy over the simulated time")}
      ${tile(fmtInt(run.spikes), "Neuron spikes", `${run.spikes && run.time_ms ? +(run.spikes / energyStatic.neurons.length / (run.time_ms / 1000)).toPrecision(3) : 0} Hz per neuron on average`)}
      ${tile(`${msText(run.time_ms)} ms`, "Simulated time", `${fmtInt(r.duration)} steps of ${msText(run.dt)} ms`)}
    </div>
    <p class="note">${r.events.length} time steps with input spikes. The estimate depends only on the spikes of the neurons, not on the simulator. The spike generators (inputs and initialisation) and the static power are not included.</p></div>`;
}

function energyPartsCard(run, rows) {
  const total = run.energy_pj;
  const segs = ENERGY_PARTS.map((p) => ({ ...p, v: run.parts[p.key] })).filter((p) => p.v > 0);
  const bar = `<div class="ebar" role="img" aria-label="Energy by kind of event">${segs.map((p) =>
    `<div class="ebar-seg" style="flex:${p.v};background:${p.color}" title="${p.name}: ${fmtEnergy(p.v)} (${pct(p.v, total)})"></div>`).join("")}</div>`;
  const legend = `<div class="elegend">${ENERGY_PARTS.map((p) => `<div class="elegend-item"><i class="sw" style="background:${p.color}"></i>
    <div><b>${p.name}</b> <span class="muted">${p.terms}</span><div class="elegend-val">${fmtEnergy(run.parts[p.key])} · ${pct(run.parts[p.key], total)}</div></div></div>`).join("")}</div>`;
  const kinds = energyByKind(rows);
  const max = Math.max(...kinds.map((k) => k.energy), 1);
  const kindRows = kinds.map((k) => `<tr title="${esc(kindName(k.kind))}: ${fmtEnergy(k.energy)} (${pct(k.energy, total)})">
    <td>${kindDot(k.kind)}${esc(kindName(k.kind))}</td><td class="num">${k.neurons}</td><td class="num">${fmtInt(k.spikes)}</td>
    <td class="num">${fmtEnergy(k.energy)}</td>
    <td class="share"><div class="meter"><div style="width:${100 * k.energy / max}%"></div></div><span>${pct(k.energy, total)}</span></td></tr>`).join("");
  return `<div class="card"><h3>Where the energy goes</h3>
    <div class="sub-title">By kind of event</div>${bar}${legend}
    <div class="sub-title">By type of neuron</div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th class="num">Neurons</th><th class="num">Spikes</th><th class="num">Energy</th><th>Share</th></tr></thead>
    <tbody>${kindRows}</tbody></table></div></div>`;
}

function energyNeuronsCard(rows, withRun) {
  const body = rows.map((x) => `<tr>
    <td>${kindDot(x.kind)}${esc(energyName(x))}</td>
    <td class="num">${x.fan_in}</td><td class="num">${x.fan_out}</td>
    <td class="num">${fmtEnergy(x.per_spike)}</td>
    ${withRun ? `<td class="num">${fmtInt(x.spikes)}</td><td class="num">${fmtEnergy(x.energy)}</td>` : ""}</tr>`).join("");
  return `<details class="card"${withRun ? "" : " open"}><summary>Per neuron</summary>
    <p class="note">Input and output synapses (N<sub>cam-match</sub>) of each neuron, and the energy of each of its spikes.${withRun ? " Sorted by energy." : ""}</p>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Neuron</th><th class="num">Synapses in</th><th class="num">Synapses out</th><th class="num">Per spike</th>
    ${withRun ? '<th class="num">Spikes</th><th class="num">Energy</th>' : ""}</tr></thead><tbody>${body}</tbody></table></div></details>`;
}


// ------------------------------------------------------------------ saving the summary
// What the tab shows: the last simulation when it is of this network, else the energy per spike.
function energyData() {
  const st = energyStatic, r = lastResult;
  const fresh = !!(st && r && r.energy && r.netKey === energyKey);
  const rows = (fresh ? r.energy.neurons : st.neurons).slice();
  rows.sort(fresh ? (a, b) => b.energy - a.energy || a.idx - b.idx : (a, b) => b.per_spike - a.per_spike || a.idx - b.idx);
  return { r, run: fresh ? r.energy.run : null, rows };
}
const plainName = (row) => energyName(row).replace(/\|/g, "/");
const pj = (v) => +v.toFixed(3);

function energyMarkdown() {
  const { r, run, rows } = energyData();
  const L = [];
  const table = (head, body, right) => {
    L.push(`| ${head.join(" | ")} |`, `|${head.map((_, i) => (right.includes(i) ? " ---: " : " --- ")).join("|")}|`);
    body.forEach((c) => L.push(`| ${c.join(" | ")} |`)); L.push("");
  };
  L.push(`# Power analysis: ${spec.name || "FSM"}`, "",
    `Estimated dynamic energy and power of a possible implementation of the design on Dynap-SE. Generated on ${new Date().toLocaleString("en-GB")}.`, "",
    "## Model", "",
    "P = Σ_n r_n [E_spike + E_en + N_cores (E_br + E_rt) + N_cam-match E_pulse]", "",
    "Energy model of Risi (2022), used for the spiking ALU (Ayuso-Martinez et al., 2025). r_n: firing rate of neuron n; " +
    "the term in brackets, E_n, is its energy per spike; N_cam-match: its output synapses; N_cores: 1 if it has output synapses, 0 otherwise. " +
    "Over a simulation with k_n spikes of neuron n, E = Σ_n k_n E_n, and the mean power is E over the simulated time.", "");
  table(["Energy", "Event", "pJ"], [
    ["E_spike", "Generation of a spike", 883], ["E_en", "Encoding of the spike and its destinations", 883],
    ["E_br", "Broadcast of the spike", 6840], ["E_rt", "Routing of the spike", 360],
    ["E_pulse", "Pulse extension at each synapse", 324]], [2]);
  L.push("Measured on Dynap-SE by Moradi et al. (2018). Only the spikes of the neurons are counted: the spike generators and the static power are left out.", "");
  if (run) {
    const hz = run.spikes && run.time_ms ? +(run.spikes / energyStatic.neurons.length / (run.time_ms / 1000)).toPrecision(3) : 0;
    L.push("## Last simulation", "");
    table(["Quantity", "Value"], [
      ["Simulator", ((info && info.backends || []).find((b) => b.id === r.backend) || { label: r.backend }).label],["Dynamic energy", fmtEnergy(run.energy_pj)], ["Mean power", fmtPower(run.power_w)],
      ["Neuron spikes", `${fmtInt(run.spikes)} (${hz} Hz per neuron on average)`],
      ["Simulated time", `${msText(run.time_ms)} ms (${fmtInt(r.duration)} steps of ${msText(run.dt)} ms)`],
      ["Time steps with input spikes", r.events.length]], [1]);
    L.push("## Energy by kind of event", "");
    table(["Kind", "Energy", "Share"], ENERGY_PARTS.map((p) =>
      [p.name, fmtEnergy(run.parts[p.key]), pct(run.parts[p.key], run.energy_pj)]), [1, 2]);
    L.push("## Energy by type of neuron", "");
    table(["Type", "Neurons", "Spikes", "Energy", "Share"], energyByKind(rows).map((k) =>
      [kindName(k.kind), k.neurons, fmtInt(k.spikes), fmtEnergy(k.energy), pct(k.energy, run.energy_pj)]), [1, 2, 3, 4]);
  } else {
    L.push("No simulation of this network yet: only the energy per spike of each neuron is given.", "");
  }
  L.push("## Per neuron", "");
  table(["Neuron", "Type", "Synapses in", "Synapses out", "Per spike"].concat(run ? ["Spikes", "Energy"] : []),
    rows.map((x) => [plainName(x), kindName(x.kind), x.fan_in, x.fan_out, fmtEnergy(x.per_spike)]
      .concat(run ? [fmtInt(x.spikes), fmtEnergy(x.energy)] : [])), run ? [2, 3, 4, 5, 6] : [2, 3, 4]);
  return L.join("\n");
}

function energyCsv() {
  const { run, rows } = energyData();
  const q = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
  const head = ["neuron", "type", "synapses_in", "synapses_out", "energy_per_spike_pJ"].concat(run ? ["spikes", "energy_pJ"] : []);
  return [head.join(",")].concat(rows.map((x) => [energyName(x), kindName(x.kind), x.fan_in, x.fan_out, pj(x.per_spike)]
    .concat(run ? [x.spikes, pj(x.energy)] : []).map(q).join(","))).join("\n") + "\n";
}

function saveEnergySummary(fmt) {
  $("#energySaveMenu").hidden = true;
  if (!energyStatic) return toast("There is no power analysis to save yet");
  const text = fmt === "md" ? energyMarkdown() : energyCsv();
  writeImage(fmt, text, { text }, "_power", "the power summary", "Summary");
}
document.addEventListener("click", (e) => {
  if (e.target.closest("#btnEnergySave")) { $("#energySaveMenu").hidden = !$("#energySaveMenu").hidden; return; }
  const b = e.target.closest("#energySaveMenu [data-fmt]"); if (b) saveEnergySummary(b.dataset.fmt);
});
document.addEventListener("pointerdown", (e) => { const m = $("#energySaveMenu"); if (m && !e.target.closest("#energySaveWrap")) m.hidden = true; });
