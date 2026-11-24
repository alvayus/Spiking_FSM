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
  delay:     { fill: "#fff2cc", stroke: "#d6b656", name: "Delay neuron of the AND gates" },
  not_input: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of an input" },
  not_state: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of a state" },
  no_event:  { fill: "#f8cecc", stroke: "#b85450", name: "Bypass: no input spike at t+1" },
  term:      { fill: "#e1d5e7", stroke: "#9673a6", name: "AND gate (transition term)" },
  latch:     { fill: "#d5e8d4", stroke: "#82b366", name: "Latch neuron" },
  output:    { fill: "#f5f5f5", stroke: "#666666", name: "Output neuron" },
};
const NET_DESC = {
  input: "Spike source of an input signal. It drives OP and inhibits the NOT gate of the input and the AND gates in which the input appears negated.",
  init: "Spike source with a single spike that sets the latches of the states that are active at start.",
  op: "Input-driven OR: spikes one step after any input spikes. It triggers every evaluation of the gates.",
  delay: "Yellow neuron of the AND gates: repeats OP one step later and provides the excitation of every AND gate.",
  not_input: "Excited by OP and inhibited by the input: spikes when some input spikes but this one does not.",
  not_state: "Excited by OP and inhibited by the latch (N1, N2, N3): spikes when some input spikes and the state is not active. In the parallel implementation, the transitions fired by input spikes two steps earlier override the latch (bypass).",
  no_event: "Parallel implementation: spikes when input spikes are not followed by more input spikes on the next step. It then cancels their bypass towards the AND gates, so a bypass never fires a transition by itself.",
  term: "Excited by the delay neuron and inhibited by the NOT gates of its positive literals, by the inputs of its negated literals and by the NOT gate of its source state. It spikes when the source state is active and the term holds; it then sets the destination latch and resets the source latch. In the parallel implementation it also receives the transitions fired one step earlier (bypass): entering its source state excites it, leaving it inhibits it.",
  set: "Set neuron (N1) of the latch: starts the ring. Setting a state that is already active does no harm (both ring neurons may then spike on every step until the next reset).",
  ring: "Ring neuron (N2 / N3) of the latch: the two ring neurons excite each other, so the state is active while they keep spiking. A transition that leaves the state inhibits them (reset).",
  output: "OR of the transitions that enter states marked as accepting in the JSON specification.",
};
// Neuron types that can be shown or hidden (the coloured checkboxes double as legend).
// The last field is when they spike after input spikes at step t.
const NET_GROUPS = [
  ["source", "Spike sources (inputs, init)", ["input", "init"], "t"],
  ["op", "OP (OR of the inputs)", ["op"], "t + 1"],
  ["delay", "Delay neuron", ["delay"], "t + 2"],
  ["not", "NOT gates", ["not_input", "not_state"], "t + 2"],
  ["noevent", "No spike at t+1 (bypass)", ["no_event"], "t + 3"],
  ["term", "AND gates (transitions)", ["term"], "t + 3"],
  ["latch", "Latches (states)", ["latch"], "set t + 4, active t + 5"],
  ["output", "Output", ["output"], "t + 4"],
];
const netHidden = new Set();       // hidden neuron kinds
const NR = 15;                     // neuron radius
const GAP = 1.5;                   // space between the end of a synapse and the neuron it reaches
const COLX = { src: 0, op: 160, gate: 330, notst: 500, term: 690, latch: 920, out: 1150 };
const BYPASS_LANE = 7;             // spacing of the bypass trunks (one per AND gate and type)
const LATCH_DX = 84;               // ring neurons at COLX.latch, set neuron N1 to their right
const TERM_DY = 64;                // vertical spacing of stacked neurons
const CLEAR = NR + 6;              // clearance a wire keeps from a neuron

let netVp = null;
let netData = null;                // {nodes, edges, groups, headers, box}
let netKey = null;
let netPinned = null;
let netHover = null;

function netKeyOf(s) {
  return JSON.stringify([archOf(), s.inputs, s.states.map((x) => [x.id, x.name, !!x.initial, !!x.accepting]),
    s.transitions.map((t) => [t.id, t.from, t.to, t.options])]);
}

// ------------------------------------------------------------------ layout
function netLayout(net) {
  // columns; with bypasses, a corridor between the AND gates and the latches holds
  // their vertical runs (one lane each)
  const X = { ...COLX };
  const termSet = new Set(net.terms.map((t) => t[2]));
  // one lane per AND gate and synapse type with bypasses towards other AND gates
  const nTrunks = new Set(net.conns.filter((c) => c[0][0] === "n" && termSet.has(c[0][1]) && termSet.has(c[1])).map((c) => c[0][1] + c[2])).size;
  const corridor = nTrunks ? 26 + BYPASS_LANE * nTrunks : 0;
  X.latch += corridor; X.out += corridor;
  const nodes = {};
  const add = (key, o) => (nodes[key] = { key, ...o });
  net.sources.forEach((s, i) => add("s" + i, { src: true, kind: s.kind, label: s.label }));
  const latchOf = {};
  for (const [sid, ns] of Object.entries(net.latches)) ns.forEach((n, j) => (latchOf[n] = { sid, j }));
  const termOf = Object.fromEntries(net.terms.map(([tid, k, n]) => [n, { tid, k }]));
  net.neurons.forEach((n, i) => {
    const o = { kind: n.kind, label: n.label, idx: i };
    if (latchOf[i]) Object.assign(o, latchOf[i]);
    if (termOf[i]) Object.assign(o, termOf[i], { from: (transById(termOf[i].tid) || {}).from });
    add("n" + i, o);
  });
  const all = Object.values(nodes);
  const notStateOf = {};
  for (const n of all) if (n.kind === "not_state") {
    const st = spec.states.find((s) => "NOT " + s.name === n.label);
    if (st) { notStateOf[st.id] = n; n.sid = st.id; }
  }

  // ---- one band per state: NOT gate, AND gates of its outgoing transitions, latch
  const bands = {}, groups = [];
  let y = 0;
  for (const st of spec.states.filter((s) => net.latches[s.id])) {
    const terms = all.filter((n) => n.kind === "term" && n.from === st.id);
    const content = Math.max(156, terms.length * TERM_DY);
    const h = content + 64;
    const cy = y + h / 2;
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
  const H = Math.max(y, 200), mid = H / 2;
  const stack = (list, x, dy) => list.forEach((n, k) => Object.assign(n, { x, y: mid + (k - (list.length - 1) / 2) * dy }));
  stack(all.filter((n) => n.src), X.src, 70);
  stack(all.filter((n) => n.kind === "op"), X.op, 70);
  stack(all.filter((n) => n.kind === "delay").concat(all.filter((n) => n.kind === "not_input")), X.gate, TERM_DY);
  // "no spike at t+1" (bypasses): below everything, its trunk goes up on the left of the AND gates
  for (const n of all) if (n.kind === "no_event") Object.assign(n, { x: X.notst, y: H + 60 });
  stack(all.filter((n) => n.kind === "output"), X.out, 70);
  for (const n of all) {
    if (n.x === undefined) Object.assign(n, { x: X.out, y: mid });
    if (n.major === undefined) n.major = n.x;
  }

  // ---- synapses: kind of route
  const edges = net.conns.map(([pre, post, rec, delay, w = 1], i) => {
    const e = { i, from: (pre[0] === "src" ? "s" : "n") + pre[1], to: "n" + post, rec, delay, w };
    const P = nodes[e.from], Q = nodes[e.to];
    // bypasses of the parallel implementation: between AND gates, towards NOT(state), "no event"
    e.bypass = (P.kind === "term" && (Q.kind === "term" || Q.kind === "not_state")) || P.kind === "no_event";
    if (P.kind === "no_event" && Q.kind === "term") e.type = "noEventBus";
    else if (P.kind === "term" && Q.kind === "term") e.type = "bypassTerm";
    else if (P.kind === "term" && Q.kind === "not_state") e.type = "bypassBack";
    else if (e.from === e.to) e.type = "self";
    else if (P.kind === "latch" && Q.kind === "not_state") { e.type = P.j === 1 ? "arcTop" : "arcBottom"; e.band = bands[P.sid]; }
    else if (P.kind === "latch" && Q.kind === "latch" && P.sid === Q.sid) e.type = "local";
    else if (Q.major > P.major) e.type = "wire";
    else e.type = "local";
    return e;
  });
  routeWires(edges.filter((e) => e.type === "wire"), nodes, all);
  routeBypasses(edges.filter((e) => e.type === "bypassTerm" || e.type === "bypassBack" || e.type === "noEventBus"), nodes, X);

  const xs = all.map((n) => n.x), ys = all.map((n) => n.y);
  const box = { x0: Math.min(...xs) - 50, y0: Math.min(0, ...ys) - 50, x1: Math.max(...xs) + 90, y1: Math.max(H, ...ys) + 40 };
  const used = (k) => all.some(k);
  const headers = [
    ["src", "Inputs", used((n) => n.src)], ["op", "OR", used((n) => n.kind === "op")],
    ["gate", "Delay · NOT inputs", used((n) => n.kind === "delay" || n.kind === "not_input")],
    ["notst", "NOT states", used((n) => n.kind === "not_state")], ["term", "AND (terms)", used((n) => n.kind === "term")],
    ["latch", "Latches", true], ["out", "Output", used((n) => n.kind === "output")],
  ].filter((h) => h[2]).map(([c, text]) => ({ c, x: X[c] + (c === "latch" ? LATCH_DX / 2 : 0), y: box.y0 + 14, text }));
  return { nodes, edges, groups, headers, box };
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

// Bypasses (parallel implementation), drawn like a bus, with right angles only:
// * colour by type (excitatory blue, inhibitory orange) and one trunk per type for every
//   AND gate, so a trunk never mixes the two;
// * the branches reaching a neuron merge before it, so every neuron gets at most one
//   excitatory head and one inhibitory head, each on its own side:
//     AND gate:   excitatory from the right (below the outgoing synapses), inhibitory from
//                 below (right half), "no spike at t+1" from below (left half); the two
//                 trunks leave side by side from the middle of the bottom;
//     NOT(state): excitatory from below (right half; the latch arrives in the middle),
//                 inhibitory from the right (above the outgoing synapses);
// * a dot marks where branches split off or join.
const BYP = { andExcDy: 11, andInhDx: 8, andOutDx: 0, andNoSpikeDx: -8, notExcDx: 8, notInhDy: -11 };
function routeBypasses(edges, nodes, X) {
  const yOf = (k) => nodes[k].y;
  const onCircle = (d) => Math.sqrt(Math.max(0, NR * NR - d * d));
  const tt = edges.filter((e) => e.type === "bypassTerm"), tb = edges.filter((e) => e.type === "bypassBack");
  // lanes: one per (source gate, type), sources from top to bottom
  const lanesOf = (list) => {
    const keys = [...new Set(list.map((e) => e.from + "|" + e.rec))]
      .sort((p, q) => yOf(p.split("|")[0]) - yOf(q.split("|")[0]) || (p.endsWith("exc") ? -1 : 1));
    return Object.fromEntries(keys.map((k, i) => [k, i]));
  };
  // ---- AND -> AND (corridor on the right of the AND column)
  const ttLane = lanesOf(tt);
  for (const e of tt) {
    const P = nodes[e.from], Q = nodes[e.to], x = X.term + NR + 18 + BYPASS_LANE * ttLane[e.from + "|" + e.rec];
    const dx = BYP.andOutDx + (e.rec === "exc" ? 2.5 : -2.5), sx = P.x + dx, y1 = P.y + NR + 16 + (e.rec === "inh" ? 4 : 0);
    const start = [[sx, P.y + onCircle(dx)], [sx, y1], [x, y1]];
    if (e.rec === "exc") {                     // into the right side, below the outgoing synapses
      const y2 = Q.y + BYP.andExcDy;
      e.points = start.concat([[x, y2], [Q.x + onCircle(BYP.andExcDy) + GAP, y2]]);
      e.merge = { x: Q.x + NR + 8, y: y2 };
    } else {                                   // into the bottom, right half
      const y2 = Q.y + NR + 8, xe = Q.x + BYP.andInhDx;
      e.points = start.concat([[x, y2], [xe, y2], [xe, Q.y + onCircle(BYP.andInhDx) + GAP]]);
      e.merge = { x: xe + 8, y: y2 };
    }
    e.trunk = { x, from: y1 }; e.branch = e.points[3][1];
  }
  // ---- AND -> NOT(state) (lanes on the right of the NOT column)
  const tbLane = lanesOf(tb);
  for (const e of tb) {
    const P = nodes[e.from], Q = nodes[e.to], x = X.notst + NR + 16 + 6 * tbLane[e.from + "|" + e.rec];
    const dx = BYP.andOutDx + (e.rec === "exc" ? 2.5 : -2.5), sx = P.x + dx, y1 = P.y + NR + 16 + (e.rec === "inh" ? 4 : 0);
    const start = [[sx, P.y + onCircle(dx)], [sx, y1], [x, y1]];
    if (e.rec === "exc") {                     // into the bottom
      const y2 = Q.y + NR + 9, xe = Q.x + BYP.notExcDx;
      e.points = start.concat([[x, y2], [xe, y2], [xe, Q.y + onCircle(BYP.notExcDx) + GAP]]);
      e.merge = { x: xe + 8, y: y2 };
    } else {                                   // into the right side, above the outgoing synapses
      const y2 = Q.y + BYP.notInhDy;
      e.points = start.concat([[x, y2], [Q.x + onCircle(BYP.notInhDy) + GAP, y2]]);
      e.merge = { x: Q.x + NR + 6, y: y2 };
    }
    e.trunk = { x, from: y1 }; e.branch = e.points[3][1];
  }
  // ---- "no spike at t+1": up along the left of the AND column, into the bottom (left half)
  for (const e of edges.filter((x) => x.type === "noEventBus")) {
    const P = nodes[e.from], Q = nodes[e.to], x = X.term - NR - 14;
    const y2 = Q.y + NR + 8, xe = Q.x + BYP.andNoSpikeDx;
    e.points = [[P.x + NR + 0.5, P.y], [x, P.y], [x, y2], [xe, y2], [xe, Q.y + onCircle(BYP.andNoSpikeDx) + GAP]];
    e.trunk = { x, from: P.y }; e.branch = y2;
  }
}

// Dots where branches split off a trunk (not at its ends) and where they join before a neuron.
function bypassDots(edges) {
  const dots = [], trunks = {}, joins = {};
  for (const e of edges) {
    if (!e.trunk) continue;
    const key = e.from + ":" + e.type + ":" + e.rec;
    (trunks[key] = trunks[key] || { x: e.trunk.x, ys: [e.trunk.from], rec: e.rec, from: e.from }).ys.push(e.branch);
    if (e.merge) {
      const j = e.to + ":" + e.rec + ":" + e.type;
      (joins[j] = joins[j] || { lanes: new Set(), rec: e.rec, e }).lanes.add(e.trunk.x);
    }
  }
  for (const t of Object.values(trunks)) {
    const lo = Math.min(...t.ys), hi = Math.max(...t.ys);
    for (const y of new Set(t.ys)) if (y > lo + 0.5 && y < hi - 0.5) dots.push({ x: t.x, y, rec: t.rec, from: t.from });
  }
  // branches from several trunks reaching one neuron: a dot on every trunk but the farthest
  for (const j of Object.values(joins)) {
    if (j.lanes.size < 2) continue;
    const xs = [...j.lanes].sort((a, b) => a - b), y = j.e.points[3][1];
    const far = j.e.type === "bypassTerm" ? xs[xs.length - 1] : xs[xs.length - 1];
    for (const x of xs) if (x !== far) dots.push({ x, y, rec: j.rec, from: j.e.to });
  }
  return dots;
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
  if (e.type === "bypassTerm" || e.type === "bypassBack" || e.type === "noEventBus") return roundedPath(e.points, 5);
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
  if (n.kind === "term") return `term:${n.tid}:${n.k}`;
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
  if (n.kind === "not_input" || n.kind === "not_state" || n.kind === "no_event") return "¬";
  if (n.kind === "term") return "∧";
  if (n.kind === "latch") return "N" + (n.j + 1);
  return "out";
}
function netLabelLines(n) {
  if (n.kind === "input" && n.label.length > 3) return [n.label];
  if (n.kind === "not_input" || n.kind === "not_state") return [n.label.replace(/^NOT /, "¬")];
  if (n.kind === "delay") return ["delay"];
  if (n.kind === "no_event") return ["no spike", "at t+1"];
  if (n.kind === "term") {
    const t = transById(n.tid);
    return [n.label.replace(/ \(\d+\)$/, ""), t ? optionText(t.options[n.k]) : ""];
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
    if (!show[e.rec] || (e.bypass && !show.bypass) || !visible(P) || !visible(Q)) continue;
    const d = netEdgePath(e, D);
    const tip = `${netName(P)} → ${netName(Q)} · ${e.rec === "exc" ? "excitatory" : "inhibitory"}${e.w !== 1 ? ` · weight ×${e.w}` : ""}${e.bypass ? " · bypass" : ""} · delay ${e.delay}`;
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
  // junction dots of the bypass trunks (only for the visible bypasses)
  const shownBypass = D.edges.filter((e) => e.trunk && show.bypass && show[e.rec] && visible(D.nodes[e.from]) && visible(D.nodes[e.to]));
  const dots = bypassDots(shownBypass).map((p) => `<circle class="byp-dot ${p.rec}" cx="${p.x}" cy="${p.y}" r="2.2"/>`).join("");
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
  const key = netHover || netPinned, layer = $("#netLayer");
  layer.classList.toggle("focus", !!key);
  const nb = key ? netNeighbours(key) : { edges: new Set(), nodes: new Set() };
  // the pinned neuron also shows the path of its latency
  const lat = netPinned && !netHover ? netLatency(netPinned) : null;
  const path = new Set(lat ? lat.path.map((e) => e.i) : []);
  if (lat) for (const e of lat.path) { nb.nodes.add(e.from); nb.nodes.add(e.to); }
  $$(".syn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.edges.has(+el.dataset.e) || path.has(+el.dataset.e));
    el.classList.toggle("path", path.has(+el.dataset.e));
  });
  $$(".nn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.nodes.has(el.dataset.n));
    el.classList.toggle("pinned", el.dataset.n === netPinned);
  });
}

function netDetails() {
  const el = $("#netDetails");
  el.hidden = !netPinned || !netData || !netData.nodes[netPinned];       // shown for a clicked neuron
  if (el.hidden) { el.innerHTML = ""; return; }
  const n = netData.nodes[netPinned], st = NET_STYLE[n.kind] || NET_STYLE.output;
  const desc = n.kind === "latch" ? (n.j ? NET_DESC.ring : NET_DESC.set) : NET_DESC[n.kind] || "";
  const row = (e, other) => {
    const o = netData.nodes[other], os = NET_STYLE[o.kind] || NET_STYLE.output;
    return `<li><i class="dot" style="background:${os.fill};border-color:${os.stroke}"></i><a href="#" data-goto-n="${other}">${esc(netName(o))}</a>
      <span class="syn-tag ${e.rec}${e.bypass ? " bypass" : ""}" title="${e.bypass ? "bypass" : ""}">${e.rec === "exc" ? "exc" : "inh"}${e.w !== 1 ? " ×" + e.w : ""}</span></li>`;
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
  $$("[data-goto-n]", el).forEach((a) => a.addEventListener("click", (ev) => {
    ev.preventDefault(); netPinned = a.dataset.gotoN; netHighlight(); netDetails();
  }));
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
  const pinned = netPinned, hover = netHover;
  netPinned = netHover = null; netHighlight();          // no highlighted neuron in the image
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
  netPinned = pinned; netHover = hover; netHighlight();
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
    isBackground: (e) => !e.target.closest(".nn, [data-lbl]"),
    onBackgroundClick: () => { if (netPinned) { netPinned = null; netHighlight(); netDetails(); } },
  });
  mountZoomBar($("#netZoom"), netVp, netFit);
  netDetails();
  svgEl.addEventListener("pointerover", (e) => {
    const n = e.target.closest(".nn"); const k = n ? n.dataset.n : null;
    if (k !== netHover) { netHover = k; netHighlight(); }
  });
  svgEl.addEventListener("pointerleave", () => { netHover = null; netHighlight(); });
  svgEl.addEventListener("click", (e) => {
    const n = e.target.closest(".nn"); if (!n) return;
    netPinned = n.dataset.n === netPinned ? null : n.dataset.n; netHighlight(); netDetails();
  });
  ["#netExc", "#netInh", "#netBypass"].forEach((id) => $(id).addEventListener("change", () => netData && drawNetwork()));
  $("#archList").addEventListener("change", (e) => {
    if (e.target.name !== "arch" || e.target.value === archOf()) return;
    spec.architecture = e.target.value;
    changed();
    if (!$("#tab-network").classList.contains("active")) loadNetwork();
  });
  $("#netKinds").addEventListener("change", (e) => {
    const kinds = NET_GROUPS.find((g) => g[0] === e.target.dataset.group)[2];
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
  $("#netKinds").innerHTML = NET_GROUPS.filter(([, , kinds]) => kinds.some((k) => present.has(k))).map(([id, label, kinds, when]) => {
    const st = NET_STYLE[kinds[0]];
    return `<label class="chk"><input type="checkbox" data-group="${id}" ${netHidden.has(kinds[0]) ? "" : "checked"}>
      <i class="dot${id === "source" ? " sq" : ""}" style="background:${st.fill};border-color:${st.stroke}"></i>${label}<span class="when">${when}</span></label>`;
  }).join("");
}
