/* Energy tab: dynamic energy of the network on DYNAP-SE (Dynap-SE1), estimated from the
 * spikes of the last simulation as in the spiking ALU work (fsm_snn/energy.py). */
"use strict";

let energyStatic = null;           // /api/energy for the current network (energy per spike, resources)
let energyKey = null;
// the three kinds of events of the formula (fixed order: generation, communication, pulses)
const ENERGY_PARTS = [
  { key: "generation", name: "Spike generation and encoding", terms: "E<sub>spike</sub> + E<sub>en</sub>", color: "#2a78d6" },
  { key: "communication", name: "Broadcast and routing", terms: "N<sub>cores</sub> (E<sub>br</sub> + E<sub>rt</sub>)", color: "#eb6834" },
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
    + energyNeuronsCard(rows, !!run) + energyResourcesCard(st);
}

function energyNoRun(r) {
  const why = r ? "The machine or the implementation has changed since the last simulation." : "There is no simulation yet.";
  return `<div class="card"><h3>Energy of a simulation</h3>
    <div class="msg info">${why} Simulate it to estimate the energy that its spikes would take on Dynap-SE1. Below, the energy that each neuron takes per spike.</div>
    <button class="btn primary energy-go" onclick="showTab('simulate')">Go to Simulate</button></div>`;
}

function energyRunCard(r, run) {
  const sim = (info.backends.find((b) => b.id === r.backend) || {}).label || r.backend;
  const tile = (value, label, sub) => `<div class="kpi"><div class="kpi-value">${value}</div><div class="kpi-label">${label}</div>${sub ? `<div class="kpi-sub">${sub}</div>` : ""}</div>`;
  return `<div class="card"><h3>Last simulation</h3>
    <div class="kpis">
      ${tile(fmtEnergy(run.energy_pj), "Dynamic energy", "of all the neuron spikes")}
      ${tile(fmtPower(run.power_w), "Mean power", "in real time")}
      ${tile(fmtInt(run.spikes), "Neuron spikes", `${run.spikes && run.time_ms ? +(run.spikes / energyStatic.resources.neurons / (run.time_ms / 1000)).toPrecision(3) : 0} Hz per neuron on average`)}
      ${tile(`${msText(run.time_ms)} ms`, "Simulated time", `${fmtInt(r.duration)} steps of ${msText(run.dt)} ms`)}
    </div>
    <p class="note">${esc(sim)} simulator, ${r.events.length} time steps with input spikes. Dynap-SE1 runs in real time, so the mean power is the energy over the simulated time. The spike generators (inputs and initialisation) and the static power of the chip are not included.</p></div>`;
}

function energyPartsCard(run, rows) {
  const total = run.energy_pj;
  const segs = ENERGY_PARTS.map((p) => ({ ...p, v: run.parts[p.key] })).filter((p) => p.v > 0);
  const bar = `<div class="ebar" role="img" aria-label="Energy by kind of event">${segs.map((p) =>
    `<div class="ebar-seg" style="flex:${p.v};background:${p.color}" title="${p.name}: ${fmtEnergy(p.v)} (${pct(p.v, total)})"></div>`).join("")}</div>`;
  const legend = `<div class="elegend">${ENERGY_PARTS.map((p) => `<div class="elegend-item"><i class="sw" style="background:${p.color}"></i>
    <div><b>${p.name}</b> <span class="muted">${p.terms}</span><div class="elegend-val">${fmtEnergy(run.parts[p.key])} · ${pct(run.parts[p.key], total)}</div></div></div>`).join("")}</div>`;
  // by neuron type
  const byKind = new Map();
  for (const x of rows) {
    const k = byKind.get(x.kind) || { kind: x.kind, neurons: 0, spikes: 0, energy: 0 };
    k.neurons++; k.spikes += x.spikes; k.energy += x.energy; byKind.set(x.kind, k);
  }
  const kinds = [...byKind.values()].sort((a, b) => b.energy - a.energy);
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
  const lim = energyStatic.limits;
  const body = rows.map((x) => `<tr${x.fan_in > lim.cam_per_neuron ? ' class="bad"' : ""}>
    <td>${kindDot(x.kind)}${esc(energyName(x))}</td>
    <td class="num">${x.fan_in}</td><td class="num">${x.fan_out}</td><td class="num">${x.cores}</td>
    <td class="num">${fmtEnergy(x.per_spike)}</td>
    ${withRun ? `<td class="num">${fmtInt(x.spikes)}</td><td class="num">${fmtEnergy(x.energy)}</td>` : ""}</tr>`).join("");
  return `<details class="card"${withRun ? "" : " open"}><summary>Per neuron</summary>
    <p class="note">Input synapses (CAM entries, at most ${lim.cam_per_neuron} on Dynap-SE1), output synapses (N<sub>cam-match</sub>) and target cores (N<sub>cores</sub>) of each neuron, and the energy of each of its spikes.${withRun ? " Sorted by energy." : ""}</p>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Neuron</th><th class="num">Synapses in</th><th class="num">Synapses out</th><th class="num">Target cores</th><th class="num">Per spike</th>
    ${withRun ? '<th class="num">Spikes</th><th class="num">Energy</th>' : ""}</tr></thead><tbody>${body}</tbody></table></div></details>`;
}

function energyResourcesCard(st) {
  const res = st.resources, lim = st.limits;
  const row = (k, v, sub) => `<tr><td>${k}</td><td class="num">${v}</td><td class="muted">${sub}</td></tr>`;
  const cam = res.over_cam.length
    ? `<div class="msg warn">${res.over_cam.length} neuron${res.over_cam.length === 1 ? " has" : "s have"} more than ${lim.cam_per_neuron} input synapses: ${esc(res.over_cam.slice(0, 6).map((i) => energyName(st.neurons[i])).join(", "))}${res.over_cam.length > 6 ? "…" : ""}.</div>` : "";
  const board = res.fits_board ? "" : `<div class="msg warn">The network needs more than the ${lim.chips_per_board * lim.cores_per_chip} cores of one board.</div>`;
  return `<div class="card"><h3>Resources on Dynap-SE1</h3>
    <table class="tbl"><tbody>
      ${row("Neurons", fmtInt(res.neurons), "spike sources not included")}
      ${row("Cores", res.cores, `${lim.neurons_per_core} neurons per core; neurons placed in order`)}
      ${row("Chips", res.chips, `${lim.cores_per_chip} cores per chip, ${lim.chips_per_board} chips per board`)}
      ${row("Synapses", fmtInt(res.synapses), "including those from the spike sources")}
      ${row("Largest fan-in", res.max_fan_in, `at most ${lim.cam_per_neuron} input synapses (CAM entries) per neuron`)}
    </tbody></table>${cam}${board}</div>`;
}
