/* Network tab: the spiking network neuron by neuron and synapse by synapse.
 *
 * Colours follow the figures of the paper (draw.io palette): inputs blue, OP orange,
 * NOT gates red, delay neuron yellow, AND gates (transitions) purple, latches green.
 * Excitatory synapses end in an arrow, inhibitory synapses in an open circle.
 *
 * Layout. Columns follow the processing pipeline after an input event at time step t:
 *   inputs | OP | delay and NOT of inputs | NOT of states | AND gates | latches   (t, t+1, t+2, t+2, t+3, t+5)
 * and every state gets its own horizontal band with its NOT gate, the AND gates of the
 * transitions that leave it and its latch.
 *
 * Routing. Synapses are drawn as orthogonal wires (with rounded corners): they leave a
 * neuron horizontally, turn in the free channel between two columns and enter the
 * target horizontally. Every wire gets its own port on each neuron and its own vertical
 * lane in each channel, so wires with similar routes run side by side instead of on
 * top of each other. Wires that would cross a neuron are rerouted along a free street.
 */
"use strict";

const NET_STYLE = {
  input:     { fill: "#dae8fc", stroke: "#6c8ebf", name: "Input (spike source)" },
  init:      { fill: "#f5f5f5", stroke: "#666666", name: "Initialisation source" },
  op:        { fill: "#ffe6cc", stroke: "#d79b00", name: "OP: input-driven OR" },
  relay:     { fill: "#dae8fc", stroke: "#6c8ebf", name: "Copy of an input" },
  delay:     { fill: "#fff2cc", stroke: "#d6b656", name: "Delay neuron of the AND gates (OP + 1)" },
  delay_line: { fill: "#fff2cc", stroke: "#d6b656", name: "Delay line: repeats its source one step later" },
  not_input: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of an input" },
  not_state: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of a state" },
  term:      { fill: "#e1d5e7", stroke: "#9673a6", name: "AND gate (transition term)" },
  no_event:  { fill: "#f8cecc", stroke: "#b85450", name: "Bypass: no input spikes now" },
  enter_or:  { fill: "#ffe6cc", stroke: "#d79b00", name: "OR gate: the state was entered" },
  leave_or:  { fill: "#ffe6cc", stroke: "#d79b00", name: "OR gate: the state was left" },
  latch:     { fill: "#d5e8d4", stroke: "#82b366", name: "Latch neuron" },
  output:    { fill: "#f5f5f5", stroke: "#666666", name: "Output neuron" },
};
const NET_DESC = {
  input: "Spike source of an input signal. It drives OP and inhibits the NOT gate of the input and the AND gates in which the input appears negated.",
  init: "Spike source with a single spike at the start: it sets the latches of the states that are active at start.",
  op: "Input-driven OR: spikes one step after any input spikes. It triggers every evaluation of the gates.",
  relay: "Parallel implementation: repeats an input one step later. The AND gates reached by the 1-step bypass receive one copy per bypass spike they may get, so they are never excited more than inhibited.",
  delay: "Yellow neuron of the AND gates: repeats OP one step later and provides the excitation of every AND gate.",
  delay_line: "Every synapse has a delay of one step. Where a signal must arrive later, it goes through a delay line: each of its neurons repeats the spikes of its source one step later (+1, +2, ...), and every synapse that needs that delay leaves from it.",
  not_input: "Excited by OP and inhibited by the input: spikes when some input spikes but this one does not.",
  not_state: "Excited by OP and inhibited by the latch (N1, N2, N3): spikes when some input spikes and the state is not active.",
  term: "AND gate of a transition term. Excited by the delay neuron and inhibited by the NOT gates of its positive literals, by the inputs of its negated literals and by the NOT gate of its source state; it then sets the destination latch and resets the source latch. Parallel implementation: each term has up to four of them, one per case, and at most one fires: via 1 step (excited by the AND gates that entered the source state one step earlier), via 2 or 3 steps (excited by the \"entered\" OR gate of the source state) and via the latch (the case of the pipelined network); any later change of the source state inhibits the older cases.",
  no_event: "Parallel implementation: spikes when there were input spikes 1, 2 or 3 steps earlier and there are none now. It cancels the bypasses of those spikes, so a bypass never fires a transition by itself.",
  enter_or: "Parallel implementation: OR of the AND gates that enter the state. It sets the latch one step later and carries the bypass of input spikes 2 and 3 steps earlier.",
  leave_or: "Parallel implementation: OR of the AND gates that leave the state. It resets the latch a second time (one step later) and carries the bypass of input spikes 2 and 3 steps earlier.",
  set: "Set neuron (N1) of the latch: starts the ring. Pipelined: setting a state that is already active does no harm. Parallel: it is set one step after the reset, and the ring inhibits it, so a state that is already active is not set again.",
  ring: "Ring neuron (N2 / N3) of the latch: the two ring neurons excite each other, so the state is active while they keep spiking. A transition that leaves the state inhibits them (reset).",
  output: "OR of the transitions that enter states marked as accepting in the JSON specification.",
};
// Neuron types that can be shown or hidden (the coloured checkboxes double as legend),
// with when they spike after input spikes at step t, for each implementation.
const NET_GROUPS = {
  pipelined: [
    ["source", "Spike sources (inputs, init)", ["input", "init"], "t"],
    ["op", "OP (OR of the inputs)", ["op"], "t + 1"],
    ["delay", "Delay neurons (delay lines)", ["delay", "delay_line"], "+1 step each"],
    ["not", "NOT gates", ["not_input", "not_state"], "t + 2"],
    ["term", "AND gates (transitions)", ["term"], "t + 3"],
    ["latch", "Latches (states)", ["latch"], "set t + 4, active t + 5"],
    ["output", "Output", ["output"], "t + 4"],
  ],
  parallel: [
    ["source", "Spike sources (inputs, init)", ["input", "init"], "t"],
    ["op", "OP (OR of the inputs)", ["op"], "t + 1"],
    ["relay", "Copies of the inputs", ["relay"], "t + 1"],
    ["delay", "Delay neurons (delay lines)", ["delay", "delay_line"], "+1 step each"],
    ["not", "NOT gates", ["not_input", "not_state"], "t + 2"],
    ["noevent", "No input spikes now (bypass)", ["no_event"], "t + 2"],
    ["term", "AND gates (up to 3 per term)", ["term"], "t + 3"],
    ["or", "OR: state entered / left", ["enter_or", "leave_or"], "t + 4"],
    ["latch", "Latches (states)", ["latch"], "set t + 5, active t + 6"],
    ["output", "Output", ["output"], "t + 4"],
  ],
};
const netGroups = () => NET_GROUPS[archOf()] || NET_GROUPS.pipelined;
const netHidden = new Set();       // hidden neuron kinds
const NR = 15;                     // neuron radius
const GAP = 1.5;                   // space between the end of a synapse and the neuron it reaches
const COLX = { src: 0, op: 160, gate: 330, notst: 500, term: 690, latch: 920, out: 1150 };
const DL_DX = 110;                 // columns of the delay lines of the inputs (input -> +1 -> +2)
// parallel: as pipelined, plus the OR gates of the states and a corridor for the bypasses
const COLP = { src: 0, op: 160, gate: 330, notst: 500, term: 690, or: 900, latch: 1060, out: 1290 };
const BYPASS_LANE = 8;             // spacing of the bypass lanes (two per state)
const ROW_P = 84;                  // parallel: vertical spacing of the AND gates (labels above, bypasses below)
const LATCH_DX = 84;               // ring neurons at COLX.latch, set neuron N1 to their right
const TERM_DY = 64;                // vertical spacing of stacked neurons
const CLEAR = NR + 6;              // clearance a wire keeps from a neuron

let netVp = null;
let netData = null;                // {nodes, edges, groups, headers, box}
let netKey = null;
let netPinned = null;
let netHover = null;
let netSynPinned = null;          // synapse (edge index) clicked on the diagram
let netSynHover = null;

function netKeyOf(s) {
  return JSON.stringify([archOf(), s.inputs, s.states.map((x) => [x.id, x.name, !!x.initial, !!x.accepting]),
    s.transitions.map((t) => [t.id, t.from, t.to, t.options])]);
}

// ------------------------------------------------------------------ layout
function netLayout(net) {
  const parallel = net.architecture === "parallel";
  const nodes = {};
  const add = (key, o) => (nodes[key] = { key, ...o });
  net.sources.forEach((s, i) => add("s" + i, { src: true, kind: s.kind, label: s.label }));
  const latchOf = {};
  for (const [sid, ns] of Object.entries(net.latches || {})) ns.forEach((n, j) => (latchOf[n] = { sid, j }));
  const termOf = Object.fromEntries(net.terms.map(([tid, k, n]) => [n, { tid, k }]));
  net.neurons.forEach((n, i) => {
    const o = { kind: n.kind, label: n.label, idx: i };
    if (n.sid) o.sid = n.sid;
    if (n.via) o.via = n.via;
    if (n.cond) o.cond = n.cond;
    if (n.g) o.g = n.g;
    if (n.base) Object.assign(o, { base: n.base, k: n.k });
    if (n.tid) Object.assign(o, { tid: n.tid, from: (transById(n.tid) || {}).from, to: (transById(n.tid) || {}).to });
    if (latchOf[i]) Object.assign(o, latchOf[i]);
    if (termOf[i]) Object.assign(o, termOf[i], { from: (transById(termOf[i].tid) || {}).from });
    add("n" + i, o);
  });
  const all = Object.values(nodes);
  const place = parallel ? netPlaceParallel : netPlacePipelined;
  const dlK = Math.max(0, ...all.filter((n) => n.kind === "delay_line" && n.base && n.base[0] === "src").map((n) => n.k));
  const shift = dlK * DL_DX;
  const { X, bands, groups, H, headerCols } = place(net, nodes, all, shift);
  const mid = H / 2;
  const stack = (list, x, dy) => list.forEach((n, k) => Object.assign(n, { x, y: mid + (k - (list.length - 1) / 2) * dy }));
  stack(all.filter((n) => n.src), X.src, 70);
  // delay lines of the inputs on the row of their input (input -> +1 -> +2), one column per
  // step; the others (OP + 2, ...) under the delay neuron
  for (const n of all) if (n.kind === "delay_line" && n.base && n.base[0] === "src") {
    const b = nodes["s" + n.base[1]];
    Object.assign(n, { x: X.src + DL_DX * n.k, y: b.y, major: X.src + DL_DX * n.k });
  }
  stack(all.filter((n) => n.kind === "op"), X.op, 70);
  stack(all.filter((n) => n.kind === "relay"), X.op, 56);
  stack(all.filter((n) => n.kind === "delay").concat(all.filter((n) => n.kind === "delay_line" && !(n.base && n.base[0] === "src")),
    all.filter((n) => n.kind === "not_input")), X.gate, 56);
  // "no input spikes now" gates (parallel): in the column of the NOT gates of the states, above
  // the bands, so that the delay neurons reach them with right angles
  all.filter((n) => n.kind === "no_event").forEach((n, k, l) => Object.assign(n, { x: X.notst, y: -70 - 64 * (l.length - 1 - k), major: X.notst }));
  stack(all.filter((n) => n.kind === "output"), X.out, 70);
  for (const n of all) {
    if (n.x === undefined) Object.assign(n, { x: X.out, y: mid });
    if (n.major === undefined) n.major = n.x;
  }

  // ---- synapses: kind of route
  const edges = net.conns.map(([pre, post, rec, delay], i) => {
    const e = { i, from: (pre[0] === "src" ? "s" : "n") + pre[1], to: "n" + post, rec, delay };
    const P = nodes[e.from], Q = nodes[e.to];
    // bypasses of the parallel implementation (towards AND gates)
    e.bypass = Q.kind === "term" && ["term", "enter_or", "leave_or", "no_event"].includes(P.kind);
    if (e.bypass && P.kind !== "no_event") e.type = "bypassBus";
    else if (e.from === e.to) e.type = "self";
    else if (P.kind === "latch" && Q.kind === "not_state") { e.type = P.j === 1 ? "arcTop" : "arcBottom"; e.band = bands[P.sid]; }
    else if (P.kind === "latch" && Q.kind === "latch" && P.sid === Q.sid) e.type = "local";
    else if (Q.major > P.major) e.type = "wire";
    else e.type = "local";
    return e;
  });
  routeWires(edges.filter((e) => e.type === "wire"), nodes, all);
  if (parallel) routeBypassBus(edges.filter((e) => e.type === "bypassBus"), nodes, X);

  const xs = all.map((n) => n.x), ys = all.map((n) => n.y);
  const box = { x0: Math.min(...xs) - 50, y0: Math.min(0, ...ys) - 50, x1: Math.max(...xs) + 90, y1: Math.max(H, ...ys) + 40 };
  const used = (k) => all.some(k);
  const headers = headerCols.filter((h) => used(h[2])).map(([c, text, , dx = 0]) => ({ c, x: X[c] + dx, y: box.y0 + 14, text }));
  for (let k = 1; k <= dlK; k++) headers.push({ c: "dl" + k, x: X.src + DL_DX * k, y: box.y0 + 14, text: `Inputs +${k}` });
  return { nodes, edges, groups, headers, box };
}

// every column but the inputs moves right by `shift` (room for the delay lines of the inputs)
const shifted = (cols, shift) => Object.fromEntries(Object.entries(cols).map(([c, x]) => [c, c === "src" ? x : x + shift]));

// Pipelined: one band per state with its NOT gate, the AND gates of its outgoing
// transitions and its latch.
function netPlacePipelined(net, nodes, all, shift = 0) {
  const X = shifted(COLX, shift);
  const notStateOf = {};
  for (const n of all) if (n.kind === "not_state") {
    const st = spec.states.find((s) => "NOT " + s.name === n.label);
    if (st) { notStateOf[st.id] = n; n.sid = st.id; }
  }
  const bands = {}, groups = [];
  let y = 0;
  for (const st of spec.states.filter((s) => net.latches[s.id])) {
    const terms = all.filter((n) => n.kind === "term" && n.from === st.id);
    const h = Math.max(156, terms.length * TERM_DY) + 64, cy = y + h / 2;
    bands[st.id] = { top: y, bottom: y + h, cy };
    const [n1, n2, n3] = net.latches[st.id].map((i) => nodes["n" + i]);
    Object.assign(n2, { x: X.latch, y: cy - 38, major: X.latch });
    Object.assign(n3, { x: X.latch, y: cy + 38, major: X.latch });
    Object.assign(n1, { x: X.latch + LATCH_DX, y: cy, major: X.latch });
    groups.push({ x0: X.latch - 38, y0: cy - 88, x1: X.latch + LATCH_DX + 62, y1: cy + 72, label: st.name, initial: st.initial, sid: st.id });
    if (notStateOf[st.id]) Object.assign(notStateOf[st.id], { x: X.notst, y: cy });
    terms.forEach((n, k) => Object.assign(n, { x: X.term, y: cy + (k - (terms.length - 1) / 2) * TERM_DY }));
    y += h;
  }
  const headerCols = [
    ["src", "Inputs", (n) => n.src], ["op", "OR", (n) => n.kind === "op"],
    ["gate", "Delay · NOT inputs", (n) => n.kind === "delay" || n.kind === "not_input"],
    ["notst", "NOT states", (n) => n.kind === "not_state"], ["term", "AND (terms)", (n) => n.kind === "term"],
    ["latch", "Latches", (n) => n.kind === "latch", LATCH_DX / 2], ["out", "Output", (n) => n.kind === "output"],
  ];
  return { X, bands, groups, H: Math.max(y, 200), headerCols };
}

// Parallel: as pipelined (NOT gate, AND gates and latch per state), plus the OR gates
// "entered" / "left" of each state and, between the AND gates and the OR gates, a
// corridor with two bypass lanes per state.
function netPlaceParallel(net, nodes, all, shift = 0) {
  const X = shifted(COLP, shift);
  const withLanes = spec.states.filter((s) => all.some((n) => n.kind === "term" && n.from === s.id)).length;
  const corridor = 40 + BYPASS_LANE * 2 * withLanes;
  X.or += corridor; X.latch += corridor; X.out += corridor;
  const notStateOf = {};
  for (const n of all) if (n.kind === "not_state") {
    const st = spec.states.find((s) => "NOT " + s.name === n.label);
    if (st) { notStateOf[st.id] = n; n.sid = st.id; }
  }
  const bands = {}, groups = [];
  let y = 0;
  for (const st of spec.states.filter((s) => net.latches[s.id])) {
    const terms = all.filter((n) => n.kind === "term" && n.from === st.id);
    const ors = all.filter((n) => (n.kind === "enter_or" || n.kind === "leave_or") && n.sid === st.id);
    const h = Math.max(156, terms.length * ROW_P) + 64, cy = y + h / 2;
    bands[st.id] = { top: y, bottom: y + h, cy };
    const [n1, n2, n3] = net.latches[st.id].map((i) => nodes["n" + i]);
    Object.assign(n2, { x: X.latch, y: cy - 38, major: X.latch });
    Object.assign(n3, { x: X.latch, y: cy + 38, major: X.latch });
    Object.assign(n1, { x: X.latch + LATCH_DX, y: cy, major: X.latch });
    groups.push({ x0: X.latch - 38, y0: cy - 88, x1: X.latch + LATCH_DX + 62, y1: cy + 72, label: st.name, initial: st.initial, sid: st.id });
    if (notStateOf[st.id]) Object.assign(notStateOf[st.id], { x: X.notst, y: cy });
    terms.forEach((n, k) => Object.assign(n, { x: X.term, y: cy + (k - (terms.length - 1) / 2) * ROW_P }));
    ors.forEach((n, k) => Object.assign(n, { x: X.or, y: cy + (k - (ors.length - 1) / 2) * 80 }));
    y += h;
  }
  const headerCols = [
    ["src", "Inputs", (n) => n.src], ["op", "OR · input copies", (n) => n.kind === "op"],
    ["gate", "Delay · NOT inputs", (n) => n.kind === "delay" || n.kind === "not_input"],
    ["notst", "No spikes · NOT states", (n) => n.kind === "not_state"], ["term", "AND (terms)", (n) => n.kind === "term"],
    ["or", "OR (entered · left)", (n) => n.kind === "enter_or" || n.kind === "leave_or"],
    ["latch", "Latches", (n) => n.kind === "latch", LATCH_DX / 2], ["out", "Output", (n) => n.kind === "output"],
  ];
  return { X, bands, groups, H: Math.max(y, 200), headerCols };
}

// Parallel: bypasses towards the AND gates, drawn like a bus with right angles. Each state
// has two lanes in the corridor (excitatory and inhibitory bypasses towards the AND gates
// leaving it). An AND gate sends its bypasses from below (left half), an OR gate from its
// left side; every AND gate gets at most one head of each type: excitatory from the right
// (below its outgoing synapses), inhibitory from below (right half).
function routeBypassBus(edges, nodes, X) {
  const states = spec.states.map((s) => s.id).filter((sid) => edges.some((e) => nodes[e.to].from === sid));
  const lane = {};
  states.forEach((sid, i) => {
    lane[sid + "exc"] = X.term + NR + 26 + BYPASS_LANE * 2 * i;
    lane[sid + "inh"] = X.term + NR + 26 + BYPASS_LANE * (2 * i + 1);
  });
  const dyExc = 11, sideExc = Math.sqrt(NR * NR - dyExc * dyExc), dxInh = 8, botInh = Math.sqrt(NR * NR - dxInh * dxInh);
  for (const e of edges) {
    const P = nodes[e.from], Q = nodes[e.to], x = lane[Q.from + e.rec];
    let start;
    if (P.kind === "term") {
      const dx = e.rec === "exc" ? -7 : -2, y1 = P.y + NR + (e.rec === "exc" ? 13 : 17);
      start = [[P.x + dx, P.y + Math.sqrt(NR * NR - dx * dx)], [P.x + dx, y1], [x, y1]];
    } else {                                              // OR gates, on the right of the lanes
      const dy = e.rec === "exc" ? -6 : 6;
      start = [[P.x - Math.sqrt(NR * NR - dy * dy) - 0.5, P.y + dy], [x, P.y + dy]];
    }
    if (e.rec === "exc") {
      const y2 = Q.y + dyExc;
      e.points = start.concat([[x, y2], [Q.x + sideExc + GAP, y2]]);
    } else {
      const y2 = Q.y + NR + 7;
      e.points = start.concat([[x, y2], [Q.x + dxInh, y2], [Q.x + dxInh, Q.y + botInh + GAP]]);
    }
    e.lane = x; e.joinY = start[start.length - 1][1]; e.branchY = y2Of(e);
  }
  function y2Of(e) { return e.points[e.points.length - (e.rec === "exc" ? 2 : 3)][1]; }
}

// Dots on the bypass lanes where sources join and branches leave (not at the lane ends).
function busDots(edges) {
  const lanes = {};
  for (const e of edges) {
    if (e.lane === undefined) continue;
    const l = (lanes[e.lane] = lanes[e.lane] || { ys: [], rec: e.rec });
    l.ys.push(e.joinY, e.branchY);
  }
  const dots = [];
  for (const [x, l] of Object.entries(lanes)) {
    const lo = Math.min(...l.ys), hi = Math.max(...l.ys);
    for (const y of new Set(l.ys)) if (y > lo + 0.5 && y < hi - 0.5) dots.push({ x: +x, y, rec: l.rec });
  }
  return dots;
}

// ------------------------------------------------------------------ orthogonal routing
function routeWires(wires, nodes, all) {
  const majors = [...new Set(all.map((n) => n.major))].sort((a, b) => a - b);
  const gapAfter = (x) => majors.indexOf(x);           // channel between majors[g] and majors[g + 1]
  const chan = (g) => [majors[g] + NR + 18, majors[g + 1] - NR - 18];

  // ports: each wire gets its own height on the side of the neuron
  const spread = (list, keyY, setKey, step = 4.5) => {
    const n = list.length, half = Math.min(NR - 4, step * (n - 1));
    list.sort((a, b) => keyY(a) - keyY(b));
    list.forEach((e, k) => (e[setKey] = n === 1 ? 0 : -half + 2 * half * k / (n - 1)));
  };
  const group = (end) => { const m = {}; for (const e of wires) (m[e[end]] = m[e[end]] || []).push(e); return m; };
  for (const l of Object.values(group("from"))) spread(l, (e) => nodes[e.to].y + nodes[e.to].x * 1e-4, "dyOut", 3);
  for (const l of Object.values(group("to"))) spread(l, (e) => nodes[e.from].y - nodes[e.from].x * 1e-4, "dyIn");

  const blocked = (xa, xb, y, P, Q) => {
    const lo = Math.min(xa, xb), hi = Math.max(xa, xb);
    return all.some((n) => n !== P && n !== Q && n.x + CLEAR > lo && n.x - CLEAR < hi && Math.abs(n.y - y) < CLEAR);
  };
  const sideX = (n, dy, dir) => n.x + dir * (n.src ? NR : Math.sqrt(Math.max(0, NR * NR - dy * dy)));

  const verticals = {};                                 // gap -> [{e, slot, y1, y2}]
  for (const e of wires) {
    const P = nodes[e.from], Q = nodes[e.to];
    const ys = P.y + e.dyOut, yt = Q.y + e.dyIn;
    e.start = [sideX(P, e.dyOut, 1) + 0.5, ys];
    e.end = [sideX(Q, e.dyIn, -1) - GAP, yt];
    const gP = gapAfter(P.major), gQ = gapAfter(Q.major) - 1;
    let plan;
    if (gP === gQ) plan = [{ g: gP, y1: ys, y2: yt }];
    else if (!blocked(e.start[0], chan(gQ)[0], ys, P, Q)) plan = [{ g: gQ, y1: ys, y2: yt }];      // turn late
    else if (!blocked(chan(gP)[1], e.end[0], yt, P, Q)) plan = [{ g: gP, y1: ys, y2: yt }];      // turn early
    else {                                                                                       // free street
      const xa = chan(gP)[1], xb = chan(gQ)[0];
      const cands = [ys, yt];
      for (const n of all) if (n.x > xa && n.x < xb) cands.push(n.y - CLEAR - 3, n.y + CLEAR + 3);
      const free = cands.filter((y) => !blocked(xa, xb, y, P, Q));
      const yf = free.sort((a, b) => Math.abs(a - ys) + Math.abs(a - yt) - Math.abs(b - ys) - Math.abs(b - yt))[0] ?? (ys + yt) / 2;
      plan = [{ g: gP, y1: ys, y2: yf }, { g: gQ, y1: yf, y2: yt }];
    }
    e.plan = plan;
    for (const v of plan) (verticals[v.g] = verticals[v.g] || []).push(v);
  }
  // lanes: order the vertical runs of each channel so that they cross as little as possible
  for (const [g, list] of Object.entries(verticals)) {
    const down = list.filter((v) => v.y2 >= v.y1).sort((a, b) => (b.y1 + b.y2) - (a.y1 + a.y2));
    const up = list.filter((v) => v.y2 < v.y1).sort((a, b) => (a.y1 + a.y2) - (b.y1 + b.y2));
    const ordered = down.concat(up), [a, b] = chan(+g), n = ordered.length;
    ordered.forEach((v, k) => (v.x = n === 1 ? (a + b) / 2 : a + (b - a) * k / (n - 1)));
  }
  for (const e of wires) {
    const pts = [e.start];
    for (const v of e.plan) pts.push([v.x, v.y1], [v.x, v.y2]);
    pts.push(e.end);
    e.points = pts.filter((p, k) => k === 0 || Math.abs(p[0] - pts[k - 1][0]) + Math.abs(p[1] - pts[k - 1][1]) > 0.01);
  }
}

// polyline with rounded corners
function roundedPath(pts, r = 7) {
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let k = 1; k < pts.length - 1; k++) {
    const [x0, y0] = pts[k - 1], [x1, y1] = pts[k], [x2, y2] = pts[k + 1];
    const l1 = Math.hypot(x1 - x0, y1 - y0), l2 = Math.hypot(x2 - x1, y2 - y1);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    if (rr < 0.5) { d += ` L${x1},${y1}`; continue; }
    const a = [x1 - (x1 - x0) / l1 * rr, y1 - (y1 - y0) / l1 * rr], b = [x1 + (x2 - x1) / l2 * rr, y1 + (y2 - y1) / l2 * rr];
    d += ` L${a[0]},${a[1]} Q${x1},${y1} ${b[0]},${b[1]}`;
  }
  const z = pts[pts.length - 1];
  return d + ` L${z[0]},${z[1]}`;
}

function netEdgePath(e, D) {
  const P = D.nodes[e.from], Q = D.nodes[e.to];
  if (e.type === "wire") return roundedPath(e.points);
  if (e.type === "arcTop" || e.type === "arcBottom") {       // latch ring -> NOT gate of the same state
    const sg = e.type === "arcTop" ? -1 : 1, yy = sg < 0 ? e.band.top + 18 : e.band.bottom - 18;
    return roundedPath([[P.x, P.y + sg * (NR + 0.5)], [P.x, yy], [Q.x, yy], [Q.x, Q.y + sg * (NR + GAP)]], 10);
  }
  if (e.type === "bypassBus") return roundedPath(e.points, 5);
  return roundedPath(latchRoute(e, P, Q), 5);
}

// Orthogonal routes of the synapses inside a latch: N2 (j = 1) top left, N3 (j = 2) bottom left,
// N1 (j = 0) on the right. Each route uses its own side of each neuron, so none overlap.
function latchRoute(e, P, Q) {
  const side = (n, d) => Math.sqrt(Math.max(0, NR * NR - d * d));      // half chord at offset d
  const o = 6;
  if (e.type === "self") {                                              // N1 inhibits itself: loop on its right
    const xr = P.x + NR + 16;
    return [[P.x + side(P, o), P.y - o], [xr, P.y - o], [xr, P.y + o], [P.x + side(P, o) + GAP, P.y + o]];
  }
  const k = `${P.j}${Q.j}`;
  if (k === "12") return [[P.x + o, P.y + side(P, o)], [Q.x + o, Q.y - side(Q, o) - GAP]];        // N2 -> N3 (ring)
  if (k === "21") return [[P.x - o, P.y - side(P, o)], [Q.x - o, Q.y + side(Q, o) + GAP]];        // N3 -> N2 (ring)
  if (k === "01") return [[P.x + o, P.y - side(P, o)], [P.x + o, Q.y - o], [Q.x + side(Q, o) + GAP, Q.y - o]];   // set N1 -> N2
  if (k === "10") return [[P.x + side(P, o), P.y + o], [Q.x - o, P.y + o], [Q.x - o, Q.y - side(Q, o) - GAP]];   // N2 -| N1
  if (k === "20") return [[P.x + NR, P.y], [Q.x, P.y], [Q.x, Q.y + NR + GAP]];                   // N3 -| N1
  if (k === "02") return [[P.x, P.y + NR], [P.x, Q.y], [Q.x + NR + GAP, Q.y]];
  // any other backward synapse: under both neurons
  const yb = Math.max(P.y, Q.y) + NR + 22;
  return [[P.x, P.y + NR], [P.x, yb], [Q.x, yb], [Q.x, Q.y + NR + GAP]];
}

// ------------------------------------------------------------------ latency (time of the first spike after an input event)
function netLatency(key) {
  const D = netData, n0 = D.nodes[key];
  if (!n0) return null;
  // shortest excitatory path from the input sources (t = 0) or, failing that, from the init source
  const run = (starts) => {
    const t = {}, prev = {};
    for (const s of starts) t[s] = 0;
    const exc = D.edges.filter((e) => e.rec === "exc");
    for (let k = 0; k < Object.keys(D.nodes).length + 2; k++) {
      let changed = false;
      for (const e of exc) {
        if (t[e.from] === undefined) continue;
        const v = t[e.from] + e.delay;
        if (t[e.to] === undefined || v < t[e.to]) { t[e.to] = v; prev[e.to] = e; changed = true; }
      }
      if (!changed) break;
    }
    return { t, prev };
  };
  const inputs = Object.values(D.nodes).filter((n) => n.kind === "input").map((n) => n.key);
  let r = run(inputs), from = "input";
  if (r.t[key] === undefined) { r = run(Object.values(D.nodes).filter((n) => n.kind === "init").map((n) => n.key)); from = "init"; }
  if (r.t[key] === undefined) return null;
  const path = [];
  for (let k = key; r.prev[k]; k = r.prev[k].from) path.unshift(r.prev[k]);
  return { steps: r.t[key], from, path };
}

// ------------------------------------------------------------------ custom names
// Every text of the diagram can be renamed (double-click it). The names are kept in the
// machine (spec.net_labels) under keys that survive rebuilding the network: the column,
// the state or transition term a neuron belongs to, or its kind and label.
function netLbl(key, def) {
  const m = spec.net_labels;
  return m && typeof m[key] === "string" ? m[key] : def;
}
function netId(n) {
  if (n.kind === "latch") return `latch:${n.sid}:${n.j}`;
  if (n.kind === "term") return `term:${n.tid}:${n.k}` + (n.via ? `:${n.via}` : "");
  if (n.kind === "not_state" && n.sid) return `not_state:${n.sid}`;
  return `${n.kind}:${n.label}`;
}
const netGlyphOf = (n) => netLbl(netId(n) + "#g", netGlyph(n));
const netLinesOf = (n) => netLabelLines(n).map((l, k) => netLbl(`${netId(n)}#${k}`, l));
const netGroupDefault = (g) => g.label + (g.initial ? " · active at start" : "");
function setNetLabel(key, def, text) {
  const t = text.trim(), cur = netLbl(key, def);
  if (t === cur || (!t && cur === def)) return;
  pushUndo();
  const m = { ...(spec.net_labels || {}) };
  if (!t || t === def) delete m[key]; else m[key] = t;
  if (Object.keys(m).length) spec.net_labels = m; else delete spec.net_labels;
  changed();
}
function resetNetLabels() {
  if (!spec.net_labels) return;
  pushUndo(); delete spec.net_labels; changed(); toast("Original names restored (Ctrl+Z to undo)");
}

// ------------------------------------------------------------------ drawing
function netGlyph(n) {
  if (n.kind === "input") return n.label.length <= 3 ? n.label : n.label.slice(0, 2) + "…";
  if (n.kind === "init") return "init";
  if (n.kind === "op") return "OP";
  if (n.kind === "delay") return "Δ";
  // the name of a delay line fits inside it when short (no label below, which wires would cross)
  if (n.kind === "delay_line") return n.label.replace(" ", "").length <= 6 ? n.label.replace(" ", "") : "+" + n.k;
  if (["not_input", "not_state", "no_event"].includes(n.kind)) return "¬";
  if (n.kind === "enter_or" || n.kind === "leave_or") return "∨";
  if (n.kind === "relay") return "=";
  if (n.kind === "term") return "∧";
  if (n.kind === "latch") return "N" + (n.j + 1);
  return "out";
}
function netLabelLines(n) {
  if (n.kind === "input" && n.label.length > 3) return [n.label];
  if (n.kind === "not_input" || n.kind === "not_state") return [n.label.replace(/^NOT /, "¬")];
  if (n.kind === "delay") return ["delay"];
  if (n.kind === "delay_line") return n.label.replace(" ", "").length <= 6 ? [] : [n.label];
  if (["no_event", "enter_or", "leave_or", "relay"].includes(n.kind)) return [n.label];
  if (n.kind === "term") {
    const t = transById(n.tid);
    const cond = n.cond || (t ? t.options[n.k] : null);
    return [n.label.replace(/ \(\d+\)$/, ""), (cond ? optionText(cond) : "") + (n.via ? ` · via ${n.via}` : "")];
  }
  return [];
}
function netName(n) {
  if (n.kind === "latch") {
    const g = netData && netData.groups.find((x) => x.sid === n.sid);
    const st = g ? netLbl("group:" + g.sid, g.label) : n.label.split(".")[0];
    return `${st} · ${netGlyphOf(n)} (${n.j ? "ring" : "set"})`;
  }
  const l = netLinesOf(n);
  if (n.kind === "term") return `${l[0]} · ${l[1]}`;
  if (l.length) return l[0];
  return netLbl(netId(n) + "#g", n.label);
}

function drawNetwork() {
  const D = netData, show = { exc: $("#netExc").checked, inh: $("#netInh").checked, bypass: $("#netBypass").checked };
  const visible = (n) => !netHidden.has(n.kind);
  const parts = [];
  const ed = (key, def) => `data-lbl="${esc(key)}" data-def="${esc(def)}"`;
  for (const h of D.headers) parts.push(`<text class="col-head" x="${h.x}" y="${h.y}" ${ed("head:" + h.c, h.text)}>${esc(netLbl("head:" + h.c, h.text))}</text>`);
  if (!netHidden.has("latch")) for (const g of D.groups) {
    const def = netGroupDefault(g);
    parts.push(`<rect class="latch-box" x="${g.x0}" y="${g.y0}" width="${g.x1 - g.x0}" height="${g.y1 - g.y0}" rx="10"/>
      <text class="latch-name" x="${g.x0 + 10}" y="${g.y0 + 16}" ${ed("group:" + g.sid, def)}>${esc(netLbl("group:" + g.sid, def))}</text>`);
  }
  const eparts = [];
  for (const e of D.edges) {
    const P = D.nodes[e.from], Q = D.nodes[e.to];
    // bypasses have their own checkbox, independent of the excitatory / inhibitory ones
    if ((e.bypass ? !show.bypass : !show[e.rec]) || !visible(P) || !visible(Q)) continue;
    const d = netEdgePath(e, D);
    const tip = `${netName(P)} → ${netName(Q)} · ${e.rec === "exc" ? "excitatory" : "inhibitory"}${e.bypass ? " bypass" : ""}${e.delay !== 1 ? ` · delay ${e.delay}` : ""}`;
    eparts.push(`<g class="syn ${e.rec}${e.bypass ? " bypass" : ""}" data-e="${e.i}"><title>${esc(tip)}</title>
      <path class="hit" d="${d}"/><path class="line" d="${d}" marker-end="url(#${e.rec === "exc" ? "synExc" : "synInh"}${e.bypass ? "B" : ""})"/></g>`);
  }
  const nparts = Object.values(D.nodes).filter(visible).map((n) => {
    const st = NET_STYLE[n.kind] || NET_STYLE.output;
    const shape = n.src ? `<rect class="nb" x="${-NR}" y="${-NR}" width="${2 * NR}" height="${2 * NR}" rx="6" fill="${st.fill}" stroke="${st.stroke}"/>`
      : `<circle class="nb" r="${NR}" fill="${st.fill}" stroke="${st.stroke}"/>`;
    const id = netId(n), glyph = netGlyphOf(n), defs = netLabelLines(n), lines = netLinesOf(n);
    const at = (k) => ed(`${id}#${k}`, defs[k]);
    const lbl = n.kind === "term"          // synapses arrive from the left and leave to the right: label above
      ? lines.map((l, k) => `<text class="nl${k ? " sub" : ""}" y="${k ? -NR - 5 : -NR - 17}" ${at(k)}>${esc(l.length > 26 ? l.slice(0, 25) + "…" : l)}</text>`).join("")
      : n.kind === "not_state"             // arcs arrive above and below: label on the upper left
      ? `<text class="nl" x="${-NR - 4}" y="${-NR + 2}" style="text-anchor:end" ${at(0)}>${esc(lines[0])}</text>`
      : lines.map((l, k) => `<text class="nl${k ? " sub" : ""}" y="${NR + 13 + 12 * k}" ${at(k)}>${esc(l)}</text>`).join("");
    const gsize = glyph.length <= 2 ? 12 : glyph.length <= 4 ? 9 : 7;
    return `<g class="nn" data-n="${n.key}" transform="translate(${n.x},${n.y})"><title>${esc(netName(n) + " — " + st.name)}</title>
      ${shape}<text class="ng" style="font-size:${gsize}px" ${ed(id + "#g", netGlyph(n))}>${esc(glyph)}</text>${lbl}</g>`;
  });
  // junction dots of the bypass lanes (parallel)
  const shownBus = D.edges.filter((e) => e.lane !== undefined && show.bypass && visible(D.nodes[e.from]) && visible(D.nodes[e.to]));
  const dots = busDots(shownBus).map((p) => `<circle class="byp-dot ${p.rec}" cx="${p.x}" cy="${p.y}" r="2.2"/>`).join("");
  $("#netLayer").innerHTML = parts.join("") + `<g>${eparts.join("")}${dots}</g><g>${nparts.join("")}</g>`;
  $("#btnNetReset").hidden = !spec.net_labels;
  netHighlight();
}

function netNeighbours(key) {
  const edges = new Set(), nodes = new Set([key]);
  for (const e of netData.edges) if (e.from === key || e.to === key) { edges.add(e.i); nodes.add(e.from); nodes.add(e.to); }
  return { edges, nodes };
}
function netHighlight() {
  // a neuron under the pointer first, then a synapse under the pointer, then the selection
  const syn = netHover ? null : netSynHover ?? (netPinned ? null : netSynPinned);
  const key = netHover || (syn !== null ? null : netPinned), layer = $("#netLayer");
  layer.classList.toggle("focus", !!key || syn !== null);
  let nb = key ? netNeighbours(key) : { edges: new Set(), nodes: new Set() };
  if (syn !== null) {             // a synapse: only it (its two neurons stay visible)
    const e = netData.edges[syn];
    nb = { edges: new Set([syn]), nodes: new Set([e.from, e.to]) };
  }
  $$(".syn", layer).forEach((el) => el.classList.toggle("sel", +el.dataset.e === syn));
  // the pinned neuron also shows the path of its latency
  const lat = key && netPinned && !netHover ? netLatency(netPinned) : null;
  const path = new Set(lat ? lat.path.map((e) => e.i) : []);
  if (lat) for (const e of lat.path) { nb.nodes.add(e.from); nb.nodes.add(e.to); }
  $$(".syn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.edges.has(+el.dataset.e) || path.has(+el.dataset.e));
    el.classList.toggle("path", path.has(+el.dataset.e));
  });
  $$(".nn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.nodes.has(el.dataset.n));
    el.classList.toggle("pinned", !!key && el.dataset.n === netPinned);
  });
}

// Card of a clicked synapse: its two neurons, type, weight and delay.
function netSynDetails(el) {
  const e = netData.edges[netSynPinned], P = netData.nodes[e.from], Q = netData.nodes[e.to];
  const dot = (n) => { const s = NET_STYLE[n.kind] || NET_STYLE.output; return `<i class="dot" style="background:${s.fill};border-color:${s.stroke}"></i>`; };
  const type = (e.rec === "exc" ? "Excitatory" : "Inhibitory") + (e.bypass ? " bypass" : "");
  el.hidden = false;
  el.innerHTML = `<div class="inspector-head"><h3>Selected synapse</h3><button class="btn small" id="netUnpin">Close</button></div>
    <ul class="syn-list">
      <li>${dot(P)}<a href="#" data-goto-n="${e.from}">${esc(netName(P))}</a><span class="muted">from</span></li>
      <li>${dot(Q)}<a href="#" data-goto-n="${e.to}">${esc(netName(Q))}</a><span class="muted">to</span></li>
    </ul>
    <dl class="kv syn-kv">
      <dt>Type</dt><dd><span class="syn-tag ${e.rec}${e.bypass ? " bypass" : ""}">${type}</span></dd>
      ${e.delay !== 1 ? `<dt>Delay</dt><dd>${e.delay} time steps</dd>` : ""}
    </dl>
`;
  $("#netUnpin").addEventListener("click", () => { netSynPinned = null; netHighlight(); netDetails(); });
  $$("[data-goto-n]", el).forEach((a) => a.addEventListener("click", (ev) => {
    ev.preventDefault(); netSynPinned = null; netPinned = a.dataset.gotoN; netHighlight(); netDetails();
  }));
}
function netDetails() {
  const el = $("#netDetails");
  if (netSynPinned !== null && netData && netData.edges[netSynPinned]) return netSynDetails(el);
  el.hidden = !netPinned || !netData || !netData.nodes[netPinned];       // shown for a clicked neuron
  if (el.hidden) { el.innerHTML = ""; return; }
  const n = netData.nodes[netPinned], st = NET_STYLE[n.kind] || NET_STYLE.output;
  const desc = n.kind === "latch" ? (n.j ? NET_DESC.ring : NET_DESC.set) : NET_DESC[n.kind] || "";
  const row = (e, other) => {
    const o = netData.nodes[other], os = NET_STYLE[o.kind] || NET_STYLE.output;
    // a row is the synapse: clicking it selects the synapse (its card links to both neurons)
    return `<li class="syn-row" data-goto-e="${e.i}" title="Select this synapse"><i class="dot" style="background:${os.fill};border-color:${os.stroke}"></i><span class="syn-name">${esc(netName(o))}</span>
      <span class="syn-tag ${e.rec}${e.bypass ? " bypass" : ""}">${e.rec === "exc" ? "exc" : "inh"}</span></li>`;
  };
  const lat = netLatency(netPinned);
  let latHtml = "";
  if (n.kind === "input") latHtml = `<p class="lat"><b>t</b><span>An input spikes at time step t.</span></p>`;
  else if (n.kind === "init") latHtml = `<p class="lat"><b>t = 0</b><span>Single spike at the start of the simulation.</span></p>`;
  else if (lat) {
    const chain = [netData.nodes[lat.path[0].from]].concat(lat.path.map((e) => netData.nodes[e.to])).map(netName).map(esc).join(" → ");
    latHtml = lat.from === "input"
      ? `<p class="lat"><b>t + ${lat.steps}</b><span>First spikes ${lat.steps} time step${lat.steps === 1 ? "" : "s"} after the input spike that drives it, along the path highlighted in blue:<br>${chain}</span></p>`
      : `<p class="lat"><b>init + ${lat.steps}</b><span>Driven by the initialisation spike:<br>${chain}</span></p>`;
  }
  const inc = netData.edges.filter((e) => e.to === n.key), out = netData.edges.filter((e) => e.from === n.key);
  el.innerHTML = `<div class="inspector-head"><h3>${n.src ? "Selected spike source" : "Selected neuron"}</h3><button class="btn small" id="netUnpin">Close</button></div>
    <div class="net-title"><i class="dot big" style="background:${st.fill};border-color:${st.stroke}"></i><div><b>${esc(netName(n))}</b><span>${esc(st.name)}</span></div></div>
    ${latHtml}
    <p class="note">${esc(desc)}</p>
    ${n.src ? "" : `<div class="field">Incoming synapses (${inc.length})<ul class="syn-list">${inc.map((e) => row(e, e.from)).join("") || "<li class='muted'>None</li>"}</ul></div>`}
    <div class="field">Outgoing synapses (${out.length})<ul class="syn-list">${out.map((e) => row(e, e.to)).join("") || "<li class='muted'>None</li>"}</ul></div>`;
  $("#netUnpin").addEventListener("click", () => { netPinned = null; netHighlight(); netDetails(); });
  $$("[data-goto-e]", el).forEach((li) => {
    li.addEventListener("click", () => { netSynPinned = +li.dataset.gotoE; netPinned = null; netSynHover = null; netHighlight(); netDetails(); });
    li.addEventListener("pointerenter", () => { netSynHover = +li.dataset.gotoE; netHighlight(); });
    li.addEventListener("pointerleave", () => { netSynHover = null; netHighlight(); });
  });
}

function netFit() { if (netData) netVp.fit(netData.box, 30, 1.4); }

// Build the network of the current machine (also used by the Simulate tab, to name the
// neurons as here). The drawing is refreshed the next time the tab is shown.
let netCountsData = null, netErrors = null, netFresh = false;
async function fetchNetwork() {
  const key = netKeyOf(spec);
  if (key === netKey && netData) return true;
  const r = await api("/api/network", { spec });
  if (!r.ok) { netData = null; netKey = null; netErrors = r.errors || ["Unknown error"]; return false; }
  netData = netLayout(r.net); netKey = key; netCountsData = r.counts; netFresh = true;
  if (netPinned && !netData.nodes[netPinned]) netPinned = null;
  netSynPinned = netSynHover = null;        // edge indices change with the network
  return true;
}
// Implementation of the network (builder.ARCHITECTURES), saved with the machine.
const archOf = () => (info.architectures[spec.architecture] ? spec.architecture : info.default_architecture);
function renderArch() {
  const cur = archOf();
  $("#archList").innerHTML = Object.entries(info.architectures).map(([id, a]) => `<label>
    <input type="radio" name="arch" value="${id}" ${id === cur ? "checked" : ""}>
    <span>${esc(a.label)}<small>${esc(a.desc)}</small></span></label>`).join("");
  $("#netBypassField").hidden = cur !== "parallel";
}
async function loadNetwork() {
  renderArch();
  if (!(await fetchNetwork())) {
    $("#netLayer").innerHTML = "";
    $("#netEmpty").hidden = false;
    $("#netEmpty").innerHTML = `<h2>The network cannot be built yet</h2><p>${netErrors.map(esc).join("<br>")}</p><p>Fix it in the <b>Design</b> tab.</p>`;
    $("#netCounts").innerHTML = "—";
    return;
  }
  $("#netEmpty").hidden = true;
  if (!netFresh) { drawNetwork(); return; }
  netFresh = false;
  const c = netCountsData;
  $("#netCounts").innerHTML = `<dl class="kv counts">
    <dt>Neurons<span>${c.neurons} neurons + ${c.sources} spike source${c.sources === 1 ? "" : "s"} (inputs and initialisation)</span></dt><dd class="big">${c.neurons + c.sources}</dd>
    <dt>Synapses<span>${c.excitatory} excitatory · ${c.inhibitory} inhibitory</span></dt><dd class="big">${c.synapses}</dd></dl>`;
  renderNetKinds(); drawNetwork(); netDetails();
  netFit();                        // the structure changed: show the whole network
}

// ------------------------------------------------------------------ rename in place
function editNetLabel(textEl) {
  const key = textEl.dataset.lbl, def = textEl.dataset.def;
  const wrap = textEl.closest(".canvas-wrap"), r = textEl.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  const size = parseFloat(getComputedStyle(textEl).fontSize) * netVp.k;
  const inp = document.createElement("input");
  inp.className = "net-edit"; inp.value = netLbl(key, def); inp.placeholder = def;
  inp.style.cssText = `left:${r.left - w.left - 6}px;top:${r.top - w.top + r.height / 2}px;width:${Math.max(r.width + 40, 110)}px;font-size:${Math.max(12, size)}px`;
  wrap.appendChild(inp); inp.focus(); inp.select();
  let done = false;
  const finish = (save) => {
    if (done) return; done = true;
    const v = inp.value; inp.remove();
    if (save) setNetLabel(key, def, v);
  };
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
  });
  inp.addEventListener("blur", () => finish(true));
}

// ------------------------------------------------------------------ export the diagram
// Self-contained SVG of the whole network (with the current names and visible neuron
// kinds): the computed styles are written into each element, so it needs no CSS.
const SVG_PROPS = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-dasharray", "stroke-linejoin", "stroke-linecap",
  "font-family", "font-size", "font-weight", "font-style", "letter-spacing", "text-anchor", "dominant-baseline"];
function netSvgText() {
  const pinned = netPinned, hover = netHover, sp = netSynPinned, sh = netSynHover;
  netPinned = netHover = netSynPinned = netSynHover = null; netHighlight();          // no highlighted neuron in the image
  const live = $("#netCanvas"), clone = live.cloneNode(true);
  const L = $$("#netLayer *", live), C = $$("#netLayer *", clone);
  L.forEach((el, i) => {
    const cs = getComputedStyle(el), c = C[i];
    c.setAttribute("style", SVG_PROPS.map((k) => `${k}:${cs.getPropertyValue(k)}`).join(";"));
    if (el.tagName === "text" && cs.textTransform === "uppercase") c.textContent = c.textContent.toUpperCase();
    ["class", "data-n", "data-e", "data-lbl", "data-def"].forEach((a) => c.removeAttribute(a));
  });
  clone.querySelectorAll("title, .hit, path.hit").forEach((x) => x.remove());
  $$("path", clone).filter((x) => x.getAttribute("stroke") === "transparent" || /stroke:\s*transparent/.test(x.getAttribute("style") || "")).forEach((x) => x.remove());
  netPinned = pinned; netHover = hover; netSynPinned = sp; netSynHover = sh; netHighlight();
  const b = netData.box, W = Math.round(b.x1 - b.x0), H = Math.round(b.y1 - b.y0);
  const layer = clone.querySelector("#netLayer");
  layer.removeAttribute("transform"); layer.removeAttribute("style"); layer.removeAttribute("class");
  ["class", "tabindex", "aria-label", "style"].forEach((a) => clone.removeAttribute(a));
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("viewBox", `${b.x0} ${b.y0} ${W} ${H}`);
  clone.setAttribute("width", W); clone.setAttribute("height", H);
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  Object.entries({ x: b.x0, y: b.y0, width: W, height: H, fill: "#ffffff" }).forEach(([k, v]) => bg.setAttribute(k, v));
  clone.insertBefore(bg, layer);
  return { text: '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone), W, H };
}
function netPngBase64(text, W, H, scale = 2) {
  return new Promise((ok, ko) => {
    const img = new Image(), url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    img.onload = () => {
      const cv = document.createElement("canvas");
      cv.width = W * scale; cv.height = H * scale;
      const ctx = cv.getContext("2d"); ctx.scale(scale, scale); ctx.drawImage(img, 0, 0, W, H);
      URL.revokeObjectURL(url);
      ok(cv.toDataURL("image/png").split(",")[1]);
    };
    img.onerror = () => { URL.revokeObjectURL(url); ko(new Error("Could not render the PNG image")); };
    img.src = url;
  });
}
// The image is written by the local server, through the same file dialog as the machines.
async function saveNetImage(kind) {
  $("#netSaveMenu").hidden = true;
  if (!netData) return toast("There is no network to save yet");
  const { text, W, H } = netSvgText();
  let body;
  try { body = kind === "svg" ? { text } : { base64: await netPngBase64(text, W, H) }; }
  catch (e) { return toast(e.message); }
  writeImage(kind, text, body, "_network", "the network");
}
// Save an exported image (SVG text, or PNG as base64 in body) next to the machine file.
async function writeImage(kind, text, body, suffix, what) {
  const ext = "." + kind;
  const base = (filePath ? baseName(filePath).replace(/\.json$/i, "") : slug(spec.name)) + suffix;
  if (WEB) {          // the browser downloads the image
    if (kind === "svg") download(base + ext, text, "image/svg+xml");
    else download(base + ext, await (await fetch("data:image/png;base64," + body.base64)).blob(), "image/png");
    return toast(`Downloaded as ${base + ext}`);
  }
  fileDialog("save", {
    ext, name: base + ext, dirKey: "fsm_snn_img_dir",
    title: `Save ${what} as ${kind.toUpperCase()}`,
    write: async (path, overwrite) => {
      const r = await api("/api/fs/write", { path, overwrite, ...body });
      if (r.ok) toast(`Image saved as ${r.name}`);
      else if (!r.exists) toast("Could not save the image: " + (r.errors || ["unknown error"]).join(" "), 5000);
      return r;
    },
  });
}

// ------------------------------------------------------------------ PyNN code viewer
function highlightPython(src) {
  const re = /("""[\s\S]*?"""|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|#[^\n]*|\b(?:import|as|def|return|for|in|if|else|elif|from|with|not|and|or|None|True|False|lambda)\b|\b(?:print|sorted|int|float|round|dict|list|range)\b(?=\()|\b\d+(?:\.\d+)?\b)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const t = m[0];
    const cls = t.startsWith("#") ? "c" : /^['"]/.test(t) ? "s" : /^\d/.test(t) ? "n" : /^(print|sorted|int|float|round|dict|list|range)$/.test(t) ? "b" : "k";
    out += `<span class="py-${cls}">${esc(t)}</span>`;
    last = m.index + t.length;
  }
  return out + esc(src.slice(last));
}
let codeCache = null;
const CODE_KINDS = {
  network: { url: "/api/code", title: "PyNN code",
    note: "One population per gate or latch and one projection per synapse, generated from the same netlist that the application simulates. Input spike times and parameters come from the <b>Simulate</b> tab." },
  script: { url: "/api/script", title: "PyNN script of the simulation",
    note: "Stand-alone script with everything needed: the network written out as in the <b>Network</b> tab, the input spikes and parameters of this tab, the run, the comparison with the reference automaton and the saved results (spikes and a raster plot). It only needs PyNN with sPyNNaker. Run it on the computer connected to the SpiNNaker board: <code>python SCRIPT</code>." },
};
// kind: "network" (Network tab) or "script" (Simulate tab: the whole simulation)
async function showCode(kind = "network") {
  const k = CODE_KINDS[kind];
  const r = await api(k.url, { spec, params: collectParams() });
  if (!r.ok) return toast((r.errors || ["Error"]).join(" · "), 5000);
  codeCache = r;
  $("#codeTitle").textContent = k.title;
  $("#codeNote").innerHTML = k.note.replace("SCRIPT", esc(r.filename));
  $("#codeName").textContent = r.filename;
  $("#codeView").innerHTML = highlightPython(r.text);
  $("#codeModal").hidden = false;
}

function initNetwork() {
  const svgEl = $("#netCanvas");
  netVp = createViewport(svgEl, $("#netLayer"), {
    panOnDrag: true,
    isBackground: (e) => !e.target.closest(".nn, .syn, [data-lbl]"),
    onBackgroundClick: () => {
      if (netPinned || netSynPinned !== null) { netPinned = netSynPinned = null; netHighlight(); netDetails(); }
    },
  });
  mountZoomBar($("#netZoom"), netVp, netFit);
  netDetails();
  svgEl.addEventListener("pointerover", (e) => {
    const n = e.target.closest(".nn"); const k = n ? n.dataset.n : null;
    const sy = n ? null : e.target.closest(".syn"); const s = sy ? +sy.dataset.e : null;
    if (k !== netHover || s !== netSynHover) { netHover = k; netSynHover = s; netHighlight(); }
  });
  svgEl.addEventListener("pointerleave", () => { netHover = netSynHover = null; netHighlight(); });
  svgEl.addEventListener("click", (e) => {
    const n = e.target.closest(".nn");
    if (n) {
      netPinned = n.dataset.n === netPinned ? null : n.dataset.n; netSynPinned = null;
    } else {
      const sy = e.target.closest(".syn"); if (!sy) return;
      netSynPinned = +sy.dataset.e === netSynPinned ? null : +sy.dataset.e; netPinned = null;
    }
    netHighlight(); netDetails();
  });
  ["#netExc", "#netInh", "#netBypass"].forEach((id) => $(id).addEventListener("change", () => netData && drawNetwork()));
  $("#archList").addEventListener("change", (e) => {
    if (e.target.name !== "arch" || e.target.value === archOf()) return;
    spec.architecture = e.target.value;
    changed();
    if (!$("#tab-network").classList.contains("active")) loadNetwork();
  });
  $("#netKinds").addEventListener("change", (e) => {
    const kinds = netGroups().find((g) => g[0] === e.target.dataset.group)[2];
    kinds.forEach((k) => (e.target.checked ? netHidden.delete(k) : netHidden.add(k)));
    if (netData) drawNetwork();
  });
  $("#btnCode").addEventListener("click", () => showCode("network"));
  svgEl.addEventListener("dblclick", (e) => {
    const nn = e.target.closest(".nn");
    const t = e.target.closest("[data-lbl]") || (nn && nn.querySelector(".ng")); if (!t) return;
    e.preventDefault(); e.stopPropagation(); editNetLabel(t);
  });
  $("#btnNetReset").addEventListener("click", resetNetLabels);
  $("#btnNetSave").addEventListener("click", () => ($("#netSaveMenu").hidden = !$("#netSaveMenu").hidden));
  $("#netSaveMenu").addEventListener("click", (e) => { const b = e.target.closest("[data-img]"); if (b) saveNetImage(b.dataset.img); });
  document.addEventListener("pointerdown", (e) => { if (!e.target.closest("#netSaveWrap")) $("#netSaveMenu").hidden = true; });
  $("#codeClose").addEventListener("click", () => ($("#codeModal").hidden = true));
  $("#codeModal").addEventListener("click", (e) => { if (e.target.id === "codeModal") $("#codeModal").hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#codeModal").hidden = true; });
  $("#codeCopy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(codeCache.text); toast("Code copied to the clipboard"); }
    catch (err) { toast("Could not copy: select the code and copy it by hand"); }
  });
  $("#codeDownload").addEventListener("click", () => download(codeCache.filename, codeCache.text, "text/x-python"));
}

// one checkbox per neuron type present in the network, with its colour
function renderNetKinds() {
  const present = new Set(Object.values(netData.nodes).map((n) => n.kind));
  $("#netKinds").innerHTML = netGroups().filter(([, , kinds]) => kinds.some((k) => present.has(k))).map(([id, label, kinds, when]) => {
    const st = NET_STYLE[kinds[0]];
    return `<label class="chk"><input type="checkbox" data-group="${id}" ${netHidden.has(kinds[0]) ? "" : "checked"}>
      <i class="dot${id === "source" ? " sq" : ""}" style="background:${st.fill};border-color:${st.stroke}"></i>${label}<span class="when">${when}</span></label>`;
  }).join("");
}
