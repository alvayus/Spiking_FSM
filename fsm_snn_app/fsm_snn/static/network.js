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
  op:        { fill: "#ffe6cc", stroke: "#d79b00", name: "OP (output of the Alt. OR)" },
  relay:     { fill: "#ffe6cc", stroke: "#d79b00", name: "Input neuron of the Alt. OR" },
  delay:     { fill: "#fff2cc", stroke: "#d6b656", name: "Delay neuron of the AND gates (OP + 1)" },
  delay_line: { fill: "#fff2cc", stroke: "#d6b656", name: "Delay neuron (+1)" },
  term_relay: { fill: "#fff2cc", stroke: "#d6b656", name: "Delay neuron of a term (+1)" },
  reset_relay: { fill: "#fff2cc", stroke: "#d6b656", name: "Reset + 1" },
  not_input: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of an input (Ā + 3)" },
  not_state: { fill: "#f8cecc", stroke: "#b85450", name: "NOT gate of a state (S̄)" },
  term:      { fill: "#e1d5e7", stroke: "#9673a6", name: "AND gate (term of a transition)" },
  no_event:  { fill: "#e1d5e7", stroke: "#9673a6", name: "Bypass guard ¬OP(t) ∧ OP(t − 1)" },
  nor:       { fill: "#f8cecc", stroke: "#b85450", name: "NOR gate of an Inh.-based OR" },
  latch:     { fill: "#d5e8d4", stroke: "#82b366", name: "Latch neuron" },
  output:    { fill: "#e1d5e7", stroke: "#9673a6", name: "Output (pattern found)" },
};
const NET_DESC = {
  input: "Spike source of an input signal. It feeds its input neuron of the Alt. OR, the NOT gate of the input and, through delay neurons, the AND gates that read the input.",
  init: "A single spike at the start, which sets (N1) the states that are active at start.",
  op: "OP, the output of the Alt. OR: it fires 2 steps after any input spikes (t + 2). It is the operation signal of the NOT gates and, delayed, of the AND gates without a positive input and of the Inh.-based OR gates.",
  relay: "Input neuron of the Alt. OR (alternative OR gate): excited by its input and inhibited by the inputs before it, so it fires when its input spikes and no earlier input does, and only one excites OP. The one of the first input is also that input one step later; another one can be it for the terms that negate the inputs before it.",
  delay: "OP one step later (t + 3): the operation signal of the AND gates without a positive input.",
  delay_line: "Repeats its source one step later. Every synapse has a delay of one step, so chains of these neurons bring a signal to the step where it is needed; each is shared by all the synapses that use it.",
  term_relay: "The AND gates of a term one step later (t + 5). It tells that the term fired to the NOT gate of the state it enters (with an Inh.-based OR, whose N1 fires one step late) and, in the parallel implementation, to the AND gates of the states it leaves or enters.",
  reset_relay: "Reset + 1 (parallel implementation, t + 5): it tells the AND gates of the state that it was left 2 steps before, which the latch does not show yet. The AND gates that enter the state at the same time inhibit it (entering wins). In a state that every input event leaves, it is OP, delayed.",
  not_input: "NOT gate of an input (Fig. 1a), Ā + 3: excited by OP and inhibited by the input, it fires at t + 3 when other inputs spike but this one does not.",
  not_state: "NOT gate of a state (S̄): excited by OP and inhibited by the latch of the state (and, in the parallel implementation, by the AND gates that entered it 2 steps before). It fires at t + 3 when the state is not active, and inhibits the AND gates that leave it.",
  term: "AND gate of a term of a transition (Fig. 1c), at t + 4: inhibited by the NOT gates of its positive inputs, by its negated inputs and by the NOT gate of its source state. Its operation signal is its first positive input (delayed) or OP + 1. It sets the destination (N1) and resets the source (N2, N3). In the parallel implementation a term has a latch case (the state was already active) and bypass t−1 cases (the state was entered with the previous input spikes; excited by the AND gates that entered it, one case per group of them, with priority), and at most one fires.",
  no_event: "Bypass guard ¬OP(t) ∧ OP(t − 1): it fires when there were input spikes one step before and there are none now, and cancels the bypasses t−1, so that they never fire on their own.",
  nor: "NOR gate of an Inh.-based OR (Fig. 1b): excited by OP + 2 and inhibited by the AND gates that enter the state; when none of them fired, it inhibits N1, which OP + 3 excites. Only used where those AND gates can fire together.",
  set: "Set neuron N1 of the latch (Fig. 1d): the OR of the AND gates that enter the state (directly, or through an Inh.-based OR), and init in the states active at start. It starts the ring; it inhibits itself and the ring inhibits it, so an active state is not set again. The state is active while N1, N2 or N3 fires.",
  ring: "Ring neurons N2 ↔ N3 of the latch (Fig. 1d): they excite each other and keep the state active, and inhibit the NOT gate of the state. The Reset stops them: the AND gates that leave the state or, when every input event leaves it, OP.",
  output: "Output: the OR of the AND gates that enter accepting states (pattern found).",
};
// Neuron types that can be shown or hidden (the coloured checkboxes double as legend):
// the name of the component and, in brackets, the symbol drawn inside its neurons.
function netGroups() {
  const par = archOf() === "parallel";
  return [
    ["source", "Spike sources", ["input", "init"]],
    ["op", "Alt. OR (⩒)", ["relay", "op"]],
    ["delay", "Delay neurons", ["delay", "delay_line", "term_relay", "reset_relay"]],
    ["not", "NOT gates (¬)", ["not_input", "not_state"]],
  ].concat(par ? [["noevent", "Bypass guard (∧)", ["no_event"]]] : [], [
    ["term", "AND gates (∧)", ["term"]],
    ["nor", "Inh.-based OR (⊽)", ["nor"]],
    ["latch", "Latches", ["latch"]],
    ["output", "Output (∨)", ["output"]],
  ]);
}
const netHidden = new Set();       // hidden neuron kinds
const NR = 15;                     // neuron radius
const GAP = 1.5;                   // space between the end of a synapse and the neuron it reaches
// parallel: the cases of the AND gates of a term (shown in a box with the name of the term)
const VIA_LABEL = { latch: "latch", "bypass 1": "bypass t−1" };
const TERM_GAP = 44;               // extra space between the boxes of two terms
const TREL_DY = 50;                // vertical spacing of the +1 of the AND gates entering a state
const BYPASS_LANE = 8;             // spacing of the bypass lanes (two per state)
const ROW_P = 84;                  // parallel: vertical spacing of the AND gates (labels above, bypasses below)
const LATCH_DX = 110;              // set neuron N1 (its step) and, one step later, the ring N2 ↔ N3
const TERM_DY = 64;                // vertical spacing of stacked neurons
const CLEAR = NR + 6;              // clearance a wire keeps from a neuron

let netVp = null;
let netData = null;                // {nodes, edges, groups, headers, box}
let netKey = null;
// Selection: neurons (keys) and synapses (edge indices). Click selects one element,
// Ctrl / Shift + click adds or removes it, a click on the box of a state or a term selects
// all its neurons. netPinned / netSynPinned are the element when exactly one is selected.
const netSel = { n: new Set(), e: new Set() };
let netPinned = null;
let netHover = null;
let netSynPinned = null;
let netSynHover = null;
const netSelSize = () => netSel.n.size + netSel.e.size;
function netSyncSel() {
  const one = netSelSize() === 1;
  netPinned = one && netSel.n.size ? [...netSel.n][0] : null;
  netSynPinned = one && netSel.e.size ? [...netSel.e][0] : null;
}
// select these neurons and synapses; add: toggle them in the current selection
function netPick(nodes, edges = [], add = false) {
  const has = nodes.every((k) => netSel.n.has(k)) && edges.every((i) => netSel.e.has(i));
  if (add) {
    for (const k of nodes) has ? netSel.n.delete(k) : netSel.n.add(k);
    for (const i of edges) has ? netSel.e.delete(i) : netSel.e.add(i);
  } else {
    const same = has && netSel.n.size === nodes.length && netSel.e.size === edges.length;
    netSel.n.clear(); netSel.e.clear();
    if (!same) { nodes.forEach((k) => netSel.n.add(k)); edges.forEach((i) => netSel.e.add(i)); }
  }
  netSyncSel(); netHighlight(); netDetails();
}
function netClearSel() {
  netSel.n.clear(); netSel.e.clear(); netSyncSel(); netHighlight(); netDetails();
}
// neurons inside the box of a state (its latch) or of a term (its AND gates) under a screen point
function netBoxAt(x, y) {
  const el = document.elementFromPoint(x, y), b = el && el.closest("[data-grp], [data-tbox]");
  if (!b || !netData) return null;
  const all = Object.values(netData.nodes);
  if (b.dataset.grp) return all.filter((n) => n.kind === "latch" && n.sid === b.dataset.grp).map((n) => n.key);
  const [tid, k] = b.dataset.tbox.split("|");
  return all.filter((n) => n.kind === "term" && n.tid === tid && String(n.k) === k).map((n) => n.key);
}

function netKeyOf(s) {
  return JSON.stringify([archOf(), s.inputs, s.states.map((x) => [x.id, x.name, !!x.initial, !!x.accepting]),
    s.transitions.map((t) => [t.id, t.from, t.to, t.options])]);
}

// ------------------------------------------------------------------ layout
const LANE_MIN = 4.5;              // least distance between two vertical runs of a channel
function netLayout(net, extra = {}, pass = 0) {
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
    if (n.grp) o.grp = n.grp;                 // bypass t−1 of a later group (priority)
    if (n.cond) o.cond = n.cond;
    if (n.g) o.g = n.g;
    if (n.role) o.role = n.role;
    if (n.input) o.input = n.input;
    if (n.tap) o.tap = n.tap;
    if (n.base) Object.assign(o, { base: n.base, k: n.k });
    if (n.tid) Object.assign(o, { tid: n.tid, from: (transById(n.tid) || {}).from, to: (transById(n.tid) || {}).to });
    if (latchOf[i]) Object.assign(o, latchOf[i]);
    if (termOf[i]) Object.assign(o, termOf[i], { from: (transById(termOf[i].tid) || {}).from });
    add("n" + i, o);
  });
  const all = Object.values(nodes);
  const place = parallel ? netPlaceParallel : netPlacePipelined;
  // delay lines of an input neuron of the Alt. OR (it is the input one step later for the AND
  // gates that negate the inputs before it): they continue its input's row
  const isRelayLine = (n) => n.kind === "delay_line" && n.base && n.base[0] === "n" && nodes["n" + n.base[1]].kind === "relay";
  const isSrcLine = (n) => n.kind === "delay_line" && n.base && n.base[0] === "src";
  const lineStep = (n) => (isSrcLine(n) ? n.k : isRelayLine(n) ? n.k + 1 : null);
  // delay lines of OP (OP + 2, ...): above the states, in the column of their step
  const isOpLine = (n) => n.kind === "delay_line" && n.base && n.base[0] === "n" && nodes["n" + n.base[1]].kind === "op";
  // input neurons of the Alt. OR that are also their input one step later (they sit on its row,
  // in the column "Inputs + 1"): that of the first input when its delay line goes on, and one
  // whose own delay line feeds AND gates when its input has no +1
  const inputOf = (R) => all.find((m) => m.src && m.kind === "input" && m.label === R.input);
  const srcLineOf = (R, k) => all.some((m) => isSrcLine(m) && nodes["s" + m.base[1]] === inputOf(R) && (k === undefined || m.k === k));
  const asInput1 = new Set(all.filter((R) => R.kind === "relay" && (R.tap ? srcLineOf(R)
    : all.some((m) => isRelayLine(m) && m.k === 1 && nodes["n" + m.base[1]] === R) && !srcLineOf(R, 1))));
  const { X, steps } = netColumns(net, all, parallel, isOpLine, lineStep, asInput1, extra);
  const { bands, groups, H, tboxes = [] } = place(net, nodes, all, X);
  const mid = H / 2;
  const stack = (list, x, dy) => list.forEach((n, k) => Object.assign(n, { x, y: mid + (k - (list.length - 1) / 2) * dy }));
  stack(all.filter((n) => n.src), X.src, 70);
  stack(all.filter((n) => n.kind === "op"), X.op, 70);
  {   // OP between the rows of the inputs, so that their delay lines pass it by
    const op = all.find((n) => n.kind === "op"), rows = all.filter((n) => n.src && n.kind === "input").map((n) => n.y);
    if (op && rows.length) {
      let y = (Math.min(...rows) + Math.max(...rows)) / 2;
      if (rows.some((r) => Math.abs(r - y) < 20)) y += 35;
      op.y = y;
    }
  }
  // delay lines of the inputs on the row of their input (input -> +1 -> +2 -> +3), each in the
  // column of its step
  const lineX = (k) => X["in" + Math.min(k, 3)] + Math.max(0, k - 3) * 110;
  for (const n of all) if (isSrcLine(n)) {
    const b = nodes["s" + n.base[1]];
    Object.assign(n, { x: lineX(n.k), y: b.y, major: lineX(n.k) });
  }
  // input neurons of the alternative OR gate (step 1): between the rows of the inputs; one that
  // is also the input one step later sits on its input's row
  all.filter((n) => n.kind === "relay").forEach((n, k, l) => {
    const src = all.find((m) => m.src && m.kind === "input" && m.label === n.input);
    Object.assign(n, { x: X.rel, y: (src ? src.y : mid + (k - (l.length - 1) / 2) * 70) + 35 });
  });
  for (const n of all) if (n.kind === "relay" && n.tap) {
    const src = all.find((m) => m.src && m.kind === "input" && m.label === n.input);
    if (src && all.some((m) => isSrcLine(m) && nodes["s" + m.base[1]] === src)) Object.assign(n, { x: X.in1, y: src.y, major: X.in1, asDelay: true });
  }
  for (const n of all.filter(isRelayLine)) {
    const R = nodes["n" + n.base[1]], src = all.find((m) => m.src && m.kind === "input" && m.label === R.input);
    if (!src) continue;
    const k = n.k + 1, raw = (j) => all.some((m) => isSrcLine(m) && m.k === j && nodes["s" + m.base[1]] === src);
    Object.assign(n, { x: lineX(k), y: src.y + (raw(k) ? 32 : 0), major: lineX(k) });
    if (n.k === 1 && !R.tap && !raw(1)) Object.assign(R, { x: X.in1, y: src.y, major: X.in1 });
  }
  stack(all.filter((n) => n.kind === "delay").concat(all.filter((n) => n.kind === "not_input")), X.gate, 56);  stack(all.filter((n) => n.kind === "delay").concat(all.filter((n) => n.kind === "not_input")), X.gate, 56);
  // delay lines of OP (OP + 2 drives the NOR gates, OP + 3 the outputs of the ORs): above the
  // states, in the columns before them, so they reach them from above
  for (const n of all.filter(isOpLine)) {
    const x = n.k === 1 ? X.gate : n.k === 2 ? X.term : X.trel;
    Object.assign(n, { x, y: -70, major: x });
  }
  // "no input spikes now" gates (parallel): in the column of the NOT gates of the states, above
  // the bands, so that the delay neurons reach them with right angles
  all.filter((n) => n.kind === "no_event").forEach((n, k, l) => Object.assign(n, { x: X.notst, y: -70 - 64 * (l.length - 1 - k), major: X.notst }));
  // Output and its NOR: under the states
  const outNor = all.find((n) => n.kind === "nor" && n.role === "output");
  if (outNor) Object.assign(outNor, { x: X.nor, y: H + 50 });
  all.filter((n) => n.kind === "output").forEach((n) => Object.assign(n, { x: X.out, y: outNor ? H + 50 : mid }));
  for (const n of all) {
    if (n.x === undefined) Object.assign(n, { x: X.out, y: mid });
    if (n.major === undefined) n.major = n.x;
  }

  // ---- synapses: kind of route
  const edges = net.conns.map(([pre, post, rec, delay], i) => {
    const e = { i, from: (pre[0] === "src" ? "s" : "n") + pre[1], to: "n" + post, rec, delay };
    const P = nodes[e.from], Q = nodes[e.to];
    // bypasses of the parallel implementation (towards AND gates)
    e.bypass = Q.kind === "term" && (["term", "no_event", "latch"].includes(P.kind) || isTermLine(P));
    if (isTermLine(P) && Q.kind === "not_state") { e.type = "enterFold"; e.band = bands[Q.sid]; }
    else if (P.kind === "term" && Q.kind === "not_state") { e.type = "notFold"; e.band = bands[Q.sid]; }
    else if (e.bypass && P.kind !== "no_event") e.type = "bypassBus";
    else if (e.from === e.to) e.type = "self";
    else if (P.kind === "latch" && Q.kind === "not_state") { e.type = P.j === 1 ? "arcTop" : "arcBottom"; e.band = bands[P.sid]; }
    else if (P.kind === "latch" && Q.kind === "latch" && P.sid === Q.sid) e.type = "local";
    else if (Q.major > P.major) e.type = "wire";
    else e.type = "local";
    return e;
  });
  for (const e of edges) if (e.type === "bypassBus" && e.rec === "exc") nodes[e.to].busIn = true;
  // first the synapses with fixed routes (corridor, folds back to the NOT gates, latches), then
  // the wires, which keep off their lanes and their horizontal runs
  const lanes = {}, sigs = corridorSignals(net, all);
  sigs.forEach((k, i) => (lanes[k] = X.term + NR + 22 + BYPASS_LANE * i));
  if (parallel) routeBypassBus(edges.filter((e) => e.type === "bypassBus"), nodes, lanes);
  routeEnterFold(edges.filter((e) => e.type === "enterFold"), nodes, X);
  routeNotFold(edges.filter((e) => e.type === "notFold"), nodes, lanes);
  routeLatchArcs(edges.filter((e) => e.type === "arcTop" || e.type === "arcBottom"), nodes);
  notBottoms(edges, nodes);
  const obstacles = [];
  for (const e of edges) if (e.type !== "wire" && e.points) for (let k = 1; k < e.points.length; k++) {
    const a = e.points[k - 1], b = e.points[k];
    if (Math.abs(a[1] - b[1]) < 0.5) obstacles.push({ y: a[1], x0: Math.min(a[0], b[0]), x1: Math.max(a[0], b[0]), from: e.from, rec: e.rec });
  }
  const limits = { after: {}, before: {} };
  if (parallel && sigs.length) limits.after[X.term] = X.term + NR + 22 + BYPASS_LANE * sigs.length + 4;
  const folds = edges.filter((e) => e.type === "enterFold");
  if (folds.length) limits.before[X.trel] = Math.min(...folds.map((e) => e.laneX)) - 6;
  const chans = routeWires(edges.filter((e) => e.type === "wire"), nodes, all, obstacles, limits);
  // a channel with too many vertical runs for its width: widen it and lay out again
  const more = { ...extra };
  let grow = false;
  for (const { left, n, lo, hi } of chans) {
    const need = (n - 1) * LANE_MIN - (hi - lo);
    const c = Object.keys(X).filter((k) => X[k] <= left + 0.5).sort((p, q) => X[q] - X[p])[0];
    if (need > 1 && c) { more[c] = (more[c] || 0) + Math.ceil(need); grow = true; }
  }
  if (grow && pass < 3) return netLayout(net, more, pass + 1);

  const xs = all.map((n) => n.x), ys = all.map((n) => n.y);
  const box = { x0: Math.min(...xs) - 70, y0: Math.min(0, ...ys) - 66, x1: Math.max(...xs) + 90, y1: Math.max(H, ...ys) + 40 };
  // one header per step (t + k) and, under it, the kind of neurons of each of its columns;
  // light bands mark the steps
  const headers = [], bandsX = [];
  steps.forEach((st, i) => {
    const a = st.slots[0].x, b = st.slots[st.slots.length - 1].x;
    const prev = steps[i - 1], next = steps[i + 1];
    const x0 = prev ? (prev.slots[prev.slots.length - 1].x + a) / 2 : box.x0;     // a line separates the steps
    const x1 = next ? (b + next.slots[0].x) / 2 : box.x1;
    bandsX.push({ x0, x1, k: st.k });
    headers.push({ c: "step" + st.k, x: (x0 + x1) / 2, y: box.y0 + 14, text: st.k ? `t + ${st.k}` : "t" });   // centred on its band
    for (const sl of st.slots) headers.push({ c: sl.c, x: sl.x, y: box.y0 + 30, text: sl.label, sub: true });
  });
  return { nodes, edges, groups, headers, stepBands: bandsX, box, tboxes, timing: net.timing || {}, setMode: net.set_mode };
}

// Parallel: the signals that run along the corridor after the AND gates, one lane each (a
// signal is a neuron and the kind of its synapses): the AND gates that go back to the NOT gate
// of the state they enter, then the bypasses towards the AND gates.
function corridorSignals(net, all) {
  const kind = (ref) => (ref[0] === "src" ? "src" : all.find((n) => n.idx === ref[1])?.kind);
  const fold = new Set(), bus = new Set();
  for (const [pre, post, rec] of net.conns) {
    const pk = kind(pre), qk = kind(["n", post]);
    if (pk === "term" && qk === "not_state") fold.add(`${pre[1]}|fold`);
    else if (qk === "term" && ["term", "latch", "term_relay", "reset_relay"].includes(pk)) bus.add(`${pre[1]}|${rec}`);
  }
  return [...fold, ...bus];
}

// Columns ordered by the step in which their neurons fire after input spikes at t: every
// neuron of a column fires at the same step, and a step has one column per kind of neuron.
// A column that would be empty takes no room.
function netColumns(net, all, parallel, isOpLine, lineStep, asInput1, extra = {}) {
  const tm = net.timing || {}, tSet = tm.set_step || 5;
  const opK = (k) => (n) => isOpLine(n) && n.k === k;
  const corridor = parallel ? 44 + BYPASS_LANE * corridorSignals(net, all).length : 0;
  const slots = [   // id, step, room to the next column, kind of neurons, members
    ["src", 0, 140, "Spike sources", (n) => n.src],
    ["in1", 1, 120, all.some((n) => n.kind === "relay" && !asInput1.has(n)) ? "Inputs + 1" : "Alt. OR · inputs + 1",
      (n) => lineStep(n) === 1 || asInput1.has(n)],
    ["rel", 1, 130, "Alt. OR", (n) => n.kind === "relay" && !asInput1.has(n)],
    ["in2", 2, 110, "Inputs + 2", (n) => lineStep(n) === 2],
    ["op", 2, 130, "OP", (n) => n.kind === "op"],
    ["in3", 3, 110, "Inputs + 3", (n) => lineStep(n) >= 3],
    ["gate", 3, 170, "OP + 1 · NOT gates", (n) => n.kind === "delay" || n.kind === "not_input" || opK(1)(n)],
    ["notst", 3, 190, parallel ? "Guard · NOT gates" : "NOT gates", (n) => n.kind === "not_state" || n.kind === "no_event"],
    ["term", 4, (parallel ? 160 : 130) + corridor, "AND gates", (n) => n.kind === "term" || opK(2)(n)],
    ["trel", 5, 110, "Delay neurons", (n) => isTermLine(n) || opK(3)(n)],
    ["nor", tSet - 1, 140, "Inh.-based OR", (n) => n.kind === "nor" && n.role !== "output"],
    ["n1", tSet, LATCH_DX, "Set (N1)", (n) => n.kind === "latch"],
    ["latch", tSet + 1, 230, "Ring (N2 ↔ N3)", (n) => n.kind === "latch"],
    ["out", tm.output_latency || tSet, 120, "Output", (n) => n.kind === "output"],
  ];
  const X = {}, steps = [];
  let x = 0;
  for (const [c, k, w, label, f] of slots) {
    X[c] = x;
    if (!all.some(f)) continue;
    let st = steps[steps.length - 1];
    if (!st || st.k !== k) steps.push((st = { k, slots: [] }));
    st.slots.push({ c, x, label });
    x += w + (extra[c] || 0);                          // wider where a channel needs more lanes
  }
  return { X, steps };
}
// the +1 of the AND gates of a term (it tells the state it entered, 3 steps later)
const isTermLine = (n) => n.kind === "term_relay" || n.kind === "reset_relay";
// The NOR of the set of a state and the +1 of the AND gates that enter it, in its band: the
// NOR level with N1, the +1 below it in their own column (they fold back to the NOT gate)
function placeStateOr(st, cy, X, nodes, all) {
  const nor = all.find((n) => n.kind === "nor" && n.sid === st.id);
  if (nor) Object.assign(nor, { x: X.nor, y: cy });
  const rel = all.filter((n) => relBand(n) === st.id);
  rel.forEach((n, k) => Object.assign(n, { x: X.trel, y: cy + 34 + TREL_DY * k, major: X.trel }));
  return rel.length;
}
// band of a delay neuron of a transition (the state it enters) or of a Reset (its state)
const relBand = (n) => (n.kind === "term_relay" ? n.to : n.kind === "reset_relay" ? n.sid : null);
const relSpan = (m) => (m ? 2 * (34 + TREL_DY * (m - 1) + 48) : 0);

// Pipelined: one band per state with its NOT gate, the AND gates of its outgoing
// transitions and its latch.
function netPlacePipelined(net, nodes, all, X) {
  const notStateOf = {};
  for (const n of all) if (n.kind === "not_state") {
    const st = spec.states.find((s) => "NOT " + s.name === n.label);
    if (st) { notStateOf[st.id] = n; n.sid = st.id; }
  }
  const bands = {}, groups = [];
  let y = 0;
  for (const st of spec.states.filter((s) => net.latches[s.id])) {
    const terms = all.filter((n) => n.kind === "term" && n.from === st.id);
    const m = all.filter((n) => relBand(n) === st.id).length;
    const h = Math.max(156, terms.length * TERM_DY, relSpan(m)) + 64, cy = y + h / 2;
    bands[st.id] = { top: y, bottom: y + h, cy };
    placeStateOr(st, cy, X, nodes, all);
    const [n1, n2, n3] = net.latches[st.id].map((i) => nodes["n" + i]);
    Object.assign(n2, { x: X.latch, y: cy - 38, major: X.latch });
    Object.assign(n3, { x: X.latch, y: cy + 38, major: X.latch });
    Object.assign(n1, { x: X.n1, y: cy, major: X.n1 });
    groups.push({ x0: X.n1 - 40, y0: cy - 92, x1: X.latch + NR + 30, y1: cy + 76, label: st.name, initial: st.initial, sid: st.id });
    if (notStateOf[st.id]) Object.assign(notStateOf[st.id], { x: X.notst, y: cy });
    terms.forEach((n, k) => Object.assign(n, { x: X.term, y: cy + (k - (terms.length - 1) / 2) * TERM_DY }));
    y += h;
  }
  return { bands, groups, H: Math.max(y, 200) };
}

// Parallel: as pipelined (NOT gate, AND gates and latch per state), with the cases of each
// term in a box; the corridor after the AND gates (one lane per signal) is room left by
// netColumns.
function netPlaceParallel(net, nodes, all, X) {
  const tboxes = [];
  const notStateOf = {};
  for (const n of all) if (n.kind === "not_state") {
    const st = spec.states.find((s) => "NOT " + s.name === n.label);
    if (st) { notStateOf[st.id] = n; n.sid = st.id; }
  }
  const bands = {}, groups = [];
  let y = 0;
  for (const st of spec.states.filter((s) => net.latches[s.id])) {
    const terms = all.filter((n) => n.kind === "term" && n.from === st.id);
    // offsets of the AND gates: ROW_P apart, TERM_GAP more between two terms (their boxes)
    const off = [];
    let acc = 0;
    terms.forEach((n, k) => {
      if (k && (n.tid !== terms[k - 1].tid || n.k !== terms[k - 1].k)) acc += TERM_GAP;
      off.push(acc); acc += ROW_P;
    });
    const span = terms.length ? off[off.length - 1] : 0;
    const m = all.filter((n) => relBand(n) === st.id).length;
    const h = Math.max(156, span + ROW_P + 30, relSpan(m)) + 64, cy = y + h / 2;
    bands[st.id] = { top: y, bottom: y + h, cy };
    placeStateOr(st, cy, X, nodes, all);
    const [n1, n2, n3] = net.latches[st.id].map((i) => nodes["n" + i]);
    Object.assign(n2, { x: X.latch, y: cy - 38, major: X.latch });
    Object.assign(n3, { x: X.latch, y: cy + 38, major: X.latch });
    Object.assign(n1, { x: X.n1, y: cy, major: X.n1 });
    groups.push({ x0: X.n1 - 40, y0: cy - 92, x1: X.latch + NR + 30, y1: cy + 76, label: st.name, initial: st.initial, sid: st.id });
    if (notStateOf[st.id]) Object.assign(notStateOf[st.id], { x: X.notst, y: cy });
    terms.forEach((n, k) => Object.assign(n, { x: X.term, y: cy + 15 + off[k] - span / 2 }));
    for (let a = 0; a < terms.length;) {                    // one box per term
      let b = a;
      while (b + 1 < terms.length && terms[b + 1].tid === terms[a].tid && terms[b + 1].k === terms[a].k) b++;
      if (b > a) tboxes.push({ x0: X.term - 52, y0: terms[a].y - NR - 46, x1: X.term + 36, y1: terms[b].y + NR + 20, n: terms[a] });
      a = b + 1;
    }
    y += h;
  }
  return { bands, groups, H: Math.max(y, 200), tboxes };
}

// Parallel: bypasses towards the AND gates, with right angles. Every signal (a neuron and the
// kind of its synapses) has its own lane in the corridor, and reaches each AND gate at its own
// point: excitatory ones on the lower right of its side, inhibitory ones from below, each at
// its own height, so that no two signals share a line. An AND gate sends its own bypasses from
// the lower left of its bottom, under those that arrive.
const sideAt = (d) => Math.sqrt(Math.max(0, NR * NR - d * d));
function routeBypassBus(edges, nodes, lanes) {
  const laneOf = (e) => lanes[`${nodes[e.from].idx}|${e.rec}`];
  const byQ = {};
  for (const e of edges) (byQ[e.to] = byQ[e.to] || []).push(e);
  const depth = {};                                      // how far below an AND gate its arrivals go
  for (const [q, list] of Object.entries(byQ)) {
    const Q = nodes[q];
    const sigs = (rec) => [...new Set(list.filter((e) => e.rec === rec).map((e) => e.from))]
      .sort((a, b) => lanes[`${nodes[a].idx}|${rec}`] - lanes[`${nodes[b].idx}|${rec}`]);
    const exc = sigs("exc"), inh = sigs("inh");
    for (const e of list) {
      const x = laneOf(e);
      if (e.rec === "exc") {                             // right side, the nearest lane highest
        const k = exc.indexOf(e.from), m = exc.length;
        const dy = m === 1 ? 7 : 3.5 + (NR - 4) * k / (m - 1);
        e.points = [[x, Q.y + dy], [Q.x + sideAt(dy) + GAP, Q.y + dy]];
        e.branchY = Q.y + dy;
      } else {                                           // from below: the nearest lane deepest and leftmost
        const r = inh.indexOf(e.from), m = inh.length;
        const dx = m === 1 ? 7 : 12 * (m - 1 - r) / (m - 1), y = Q.y + NR + 6 + 4.5 * r;
        e.points = [[x, y], [Q.x + dx, y], [Q.x + dx, Q.y + sideAt(dx) + GAP]];
        e.branchY = y;
      }
    }
    depth[q] = Q.busDepth = NR + 6 + 4.5 * Math.max(0, inh.length - 1);
  }
  for (const e of edges) {
    const P = nodes[e.from], x = laneOf(e);
    let start;
    if (P.kind === "term") {                             // from the lower left of its bottom
      const dx = e.rec === "exc" ? -12 : -8.5, y1 = P.y + (depth[e.from] || NR) + (e.rec === "exc" ? 13 : 9);
      start = [[P.x + dx, P.y + sideAt(dx)], [P.x + dx, y1], [x, y1]];
    } else if (P.kind !== "latch") {                     // +1 of AND gates: from their bottom (their left takes what arrives)
      const dx = e.rec === "exc" ? -6 : 2, y1 = P.y + NR + (e.rec === "exc" ? 14 : 9);
      start = [[P.x + dx, P.y + sideAt(dx)], [P.x + dx, y1], [x, y1]];
    } else {                                             // latches: from the outer part of their left
      const dy = e.rec === "exc" ? -11 : 11;
      start = [[P.x - sideAt(dy) - 0.5, P.y + dy], [x, P.y + dy]];
    }
    e.points = start.concat(e.points);
    e.lane = x; e.joinY = start[start.length - 1][1];
  }
}

// +1 of an AND gate -> NOT gate of the state it entered (3 steps before). It goes back: left
// from the +1 to a lane of its own beside its column, down to the bottom of the band and left
// into the NOT gate (notBottoms sets the last two runs).
function routeEnterFold(edges, nodes, X) {
  const dy = 10, side = sideAt(dy);
  const order = [...new Set(edges.map((e) => e.from))];
  for (const e of edges) {
    const P = nodes[e.from], xl = X.trel - NR - 12 - 5 * order.indexOf(e.from);
    e.points = [[P.x - side - 0.5, P.y + dy], [xl, P.y + dy]];
    e.laneX = xl;
  }
}

// AND gate -> NOT gate of the state it enters (it entered 2 steps before): from the lower
// left of the AND gate to a lane of its own in the corridor, down to the bottom of the band
// and left into the NOT gate (notBottoms sets the last two runs).
function routeNotFold(edges, nodes, lanes) {
  for (const e of edges) {
    const P = nodes[e.from], x = lanes[`${P.idx}|fold`], dx = -5;
    const y1 = P.y + (P.busDepth || NR) + 5;                // above its own bypasses, under what arrives
    e.points = [[P.x + dx, P.y + sideAt(dx)], [P.x + dx, y1], [x, y1]];
    e.laneX = x;
  }
}

// Latch -> NOT gate of its state: N2 from its right, up over the band; N1 (bottom left) and
// N3 (bottom) down under it (notBottoms sets their last two runs).
function routeLatchArcs(edges, nodes) {
  for (const e of edges) {
    const P = nodes[e.from], Q = nodes[e.to];
    if (e.type === "arcTop") {
      const yy = e.band.top + 18;
      e.points = P.j === 1 ? [[P.x + NR + 0.5, P.y], [P.x + NR + 14, P.y], [P.x + NR + 14, yy], [Q.x, yy], [Q.x, Q.y - NR - GAP]]
        : [[P.x, P.y - NR - 0.5], [P.x, yy], [Q.x, yy], [Q.x, Q.y - NR - GAP]];
    } else {
      const dx = P.j === 0 ? -8 : 0;
      e.points = [[P.x + dx, P.y + sideAt(dx) + 0.5]];
      e.laneX = P.x + dx;
    }
  }
}

// Bottom of each NOT gate of a state: what goes back to it (the AND gates that entered the
// state, their +1, N1 and N3) arrives from the right, each at its own height and its own point,
// the farthest lane deepest and leftmost, so that none overlap or cross.
function notBottoms(edges, nodes) {
  const byQ = {};
  for (const e of edges) if (e.laneX !== undefined && nodes[e.to].kind === "not_state") (byQ[e.to] = byQ[e.to] || []).push(e);
  for (const [q, list] of Object.entries(byQ)) {
    const Q = nodes[q], band = list[0].band, n = list.length;
    list.sort((a, b) => b.laneX - a.laneX);
    list.forEach((e, i) => {
      const y = band.bottom - 12 - 5 * i, dx = n === 1 ? 0 : -12 + 24 * i / (n - 1);
      e.points.push([e.laneX, y], [Q.x + dx, y], [Q.x + dx, Q.y + sideAt(dx) + GAP]);
    });
  }
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
function routeWires(wires, nodes, all, obstacles = [], limits = { after: {}, before: {} }) {
  const majors = [...new Set(all.map((n) => n.major))].sort((a, b) => a - b);
  const gapAfter = (x) => majors.indexOf(x);           // channel between majors[g] and majors[g + 1]
  const chan = (g) => {                                 // without the lanes kept for other synapses
    const lo = limits.after[majors[g]] ?? majors[g] + NR + 18, hi = limits.before[majors[g + 1]] ?? majors[g + 1] - NR - 18;
    return hi - lo >= 10 ? [lo, hi] : [majors[g] + NR + 18, majors[g + 1] - NR - 18];
  };

  // ports: each wire gets its own height on the side of the neuron; when they do not fit
  // (more than SIDE_PORTS), the outer ones reach a neuron through its upper and lower
  // diagonals instead, ending (or starting) with a short radial stub
  const SIDE_PORTS = 5;
  const spread = (list, keyY, end, step) => {
    const n = list.length;
    list.sort((a, b) => keyY(a) - keyY(b));
    const node = nodes[list[0][end === "out" ? "from" : "to"]];
    const nSide = node.src ? n : Math.min(n, SIDE_PORTS), top = Math.floor((n - nSide) / 2);
    const half = Math.min(NR - 6, step * (nSide - 1));
    list.forEach((e, k) => {
      const j = k - top;                       // position among the ports on the side
      let port;
      if (j >= 0 && j < nSide) port = { dy: nSide === 1 ? 0 : -half + 2 * half * j / (nSide - 1) };
      else {                                   // diagonal: steeper the further from the side
        const m = j < 0 ? top : n - nSide - top, i = j < 0 ? -j - 1 : j - nSide;
        const ang = 40 + (m === 1 ? 0 : Math.min(36, 12 * (m - 1)) * i / (m - 1));
        port = { ang: (j < 0 ? 1 : -1) * ang * Math.PI / 180 };
      }
      if (end === "out" && node.busIn && port.dy !== undefined)     // its lower right takes the bypasses
        port.dy = nSide === 1 ? -6 : -13 + 11 * (port.dy + half) / (2 * half);
      e[end === "out" ? "portOut" : "portIn"] = port;
    });
  };
  const group = (end) => { const m = {}; for (const e of wires) (m[e[end]] = m[e[end]] || []).push(e); return m; };
  for (const l of Object.values(group("from"))) spread(l, (e) => nodes[e.to].y + nodes[e.to].x * 1e-4, "out", 3);
  for (const l of Object.values(group("to"))) spread(l, (e) => nodes[e.from].y - nodes[e.from].x * 1e-4, "in", 4.5);
  const STUB = 10;                             // radial stub of a diagonal port

  const blocked = (xa, xb, y, P, Q) => {
    const lo = Math.min(xa, xb), hi = Math.max(xa, xb);
    return all.some((n) => n !== P && n !== Q && n.x + CLEAR > lo && n.x - CLEAR < hi && Math.abs(n.y - y) < CLEAR);
  };
  const sideX = (n, dy, dir) => n.x + dir * (n.src ? NR : Math.sqrt(Math.max(0, NR * NR - dy * dy)));

  const verticals = {};                                 // gap -> [{e, slot, y1, y2}]
  const stats = [];                                     // per channel: its left column, runs and width
  const streets = obstacles.slice();                    // horizontal runs taken (by other synapses too)
  const takenBy = (e, y, xa, xb) => streets.some((s) => Math.abs(s.y - y) < 6 && s.x0 < Math.max(xa, xb) && Math.min(xa, xb) < s.x1
    && !(s.from === e.from && s.rec === e.rec));
  for (const e of wires) {
    const P = nodes[e.from], Q = nodes[e.to], po = e.portOut, pi = e.portIn;
    e.head = null; e.tail = null;
    if (po.ang === undefined) e.start = [sideX(P, po.dy, 1) + 0.5, P.y + po.dy];
    else {                                     // leaves through a diagonal on the right
      const c = Math.cos(po.ang), sn = Math.sin(po.ang);
      e.head = [P.x + (NR + 0.5) * c, P.y - (NR + 0.5) * sn];
      e.start = [P.x + (NR + STUB) * c, P.y - (NR + STUB) * sn];
    }
    if (pi.ang === undefined) e.end = [sideX(Q, pi.dy, -1) - GAP, Q.y + pi.dy];
    else {                                     // arrives through a diagonal on the left
      const c = Math.cos(pi.ang), sn = Math.sin(pi.ang);
      e.tail = [Q.x - (NR + GAP) * c, Q.y - (NR + GAP) * sn];
      e.end = [Q.x - (NR + STUB) * c, Q.y - (NR + STUB) * sn];
    }
    const ys = e.start[1], yt = e.end[1];
    const gP = gapAfter(P.major), gQ = gapAfter(Q.major) - 1;
    let plan;
    if (gP === gQ) plan = [{ g: gP, y1: ys, y2: yt }];
    else if (!blocked(e.start[0], chan(gQ)[0], ys, P, Q) && !takenBy(e, ys, e.start[0], chan(gQ)[0])) {    // turn late
      plan = [{ g: gQ, y1: ys, y2: yt }];
      streets.push({ y: ys, x0: e.start[0], x1: chan(gQ)[0], from: e.from, rec: e.rec });
    } else if (!blocked(chan(gP)[1], e.end[0], yt, P, Q) && !takenBy(e, yt, chan(gP)[1], e.end[0])) {     // turn early
      plan = [{ g: gP, y1: ys, y2: yt }];
      streets.push({ y: yt, x0: chan(gP)[1], x1: e.end[0], from: e.from, rec: e.rec });
    }
    else {                                                                                       // free street
      const xa = chan(gP)[1], xb = chan(gQ)[0];
      const cands = [ys, yt];
      for (const n of all) if (n.x > xa && n.x < xb) cands.push(n.y - CLEAR - 3, n.y + CLEAR + 3);
      for (const s of streets) if (s.x0 < xb && xa < s.x1) cands.push(s.y - 7, s.y + 7);
      const free = cands.filter((y) => !blocked(xa, xb, y, P, Q) && !takenBy(e, y, xa, xb));
      const yf = free.sort((a, b) => Math.abs(a - ys) + Math.abs(a - yt) - Math.abs(b - ys) - Math.abs(b - yt))[0] ?? (ys + yt) / 2;
      plan = [{ g: gP, y1: ys, y2: yf }, { g: gQ, y1: yf, y2: yt }];
      streets.push({ y: yf, x0: xa, x1: xb, from: e.from, rec: e.rec });
    }
    e.plan = plan;
    for (const v of plan) (verticals[v.g] = verticals[v.g] || []).push(Object.assign(v, { e }));
  }
  // lanes: order the vertical runs of each channel so that they cross as little as possible
  for (const [g, list] of Object.entries(verticals)) {
    const down = list.filter((v) => v.y2 >= v.y1).sort((a, b) => (b.y1 + b.y2) - (a.y1 + a.y2));
    const up = list.filter((v) => v.y2 < v.y1).sort((a, b) => (a.y1 + a.y2) - (b.y1 + b.y2));
    const ordered = down.concat(up), [a, b] = chan(+g), n = ordered.length;
    const same = (u, v) => u.e.from === v.e.from && u.e.rec === v.e.rec;
    for (let pass = 0, moved = true; moved && pass < n * n; pass++) {
      moved = false;
      for (let i = 0; i < n && !moved; i++) for (let j = 0; j < i && !moved; j++) {
        const u = ordered[i], v = ordered[j];           // v is to the left of u
        if (!same(u, v) && Math.abs(u.y1 - v.y2) < 6 && Math.abs(v.y1 - u.y2) >= 6) {
          ordered.splice(i, 1); ordered.splice(j, 0, u); moved = true;
        }
      }
    }
    ordered.forEach((v, k) => (v.x = n === 1 ? (a + b) / 2 : a + (b - a) * k / (n - 1)));
    stats.push({ left: majors[+g], n, lo: a, hi: b });
  }
  for (const e of wires) {
    const pts = e.head ? [e.head, e.start] : [e.start];
    for (const v of e.plan) pts.push([v.x, v.y1], [v.x, v.y2]);
    pts.push(e.end);
    if (e.tail) pts.push(e.tail);
    e.points = pts.filter((p, k) => k === 0 || Math.abs(p[0] - pts[k - 1][0]) + Math.abs(p[1] - pts[k - 1][1]) > 0.01);
  }
  return stats;
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
  if (e.type === "arcTop" || e.type === "arcBottom") return roundedPath(e.points, 8);   // latch -> NOT gate
  if (e.type === "bypassBus" || e.type === "enterFold" || e.type === "notFold") return roundedPath(e.points, 5);
  return roundedPath(latchRoute(e, P, Q), 5);
}

// Orthogonal routes of the synapses inside a latch: N1 (j = 0) on the left, N2 (j = 1) top right,
// N3 (j = 2) bottom right. They use the tops and bottoms of the neurons, so that their left
// sides stay free for the synapses that arrive from the network, and none overlap.
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
  const top = Math.min(P.y, Q.y) - NR;
  if (k === "01") return [[P.x - o, P.y - side(P, o)], [P.x - o, top - 20], [Q.x + o, top - 20], [Q.x + o, Q.y - side(Q, o) - GAP]];  // set N1 -> N2
  if (k === "10") return [[P.x - o, P.y - side(P, o)], [P.x - o, top - 12], [Q.x + o, top - 12], [Q.x + o, Q.y - side(Q, o) - GAP]];  // N2 -| N1
  const bot = Math.max(P.y, Q.y) + NR;
  if (k === "20") return [[P.x - o, P.y + side(P, o)], [P.x - o, bot + 12], [Q.x + o, bot + 12], [Q.x + o, Q.y + side(Q, o) + GAP]];  // N3 -| N1
  if (k === "02") return [[P.x + o, P.y + side(P, o)], [P.x + o, bot + 20], [Q.x - o, bot + 20], [Q.x - o, Q.y + side(Q, o) + GAP]];
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
  if (n.kind === "term") return `term:${n.tid}:${n.k}` + (n.via ? `:${n.via}` : "") + (n.grp ? `:${n.grp}` : "");
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
// Names follow the papers: OP, the alternative OR gate; Ā and S̄, the NOT gates of an input and of
// a state; T0, T1... the transitions (their AND gates); NOR gates; Set (N1) and ring of the latches.
const stName = (sid) => {
  const g = netData && netData.groups.find((x) => x.sid === sid);
  const st = spec.states.find((x) => x.id === sid);
  return g ? netLbl("group:" + g.sid, g.label) : st ? st.name : sid;
};
// the input neuron of the alternative OR gate that is also its input one step later
const isRelayTap = (n) => n.kind === "relay" && !!n.asDelay;
// negation as in the papers: a bar over a one-letter name (Ā, S̄0), ¬ before longer ones
const bar = (name) => (/^[A-Za-z]\d*$/.test(name) ? name[0] + "\u0304" + name.slice(1) : "¬" + name);
function termTitle(n) {
  const i = spec.transitions.findIndex((t) => t.id === n.tid), t = spec.transitions[i];
  const k = n.k || 0, multi = n.label && / \(\d+\)( \+1)?$/.test(n.label);
  const route = t ? `${stName(t.from)}→${stName(t.to)}` : n.label.replace(/ \(\d+\)( \+1)?$/, "").replace(/ \+1$/, "");
  return `T${i < 0 ? "?" : i}${multi ? "." + (k + 1) : ""} · ${route}`;
}
// the input neuron of the Alt. OR that a delay neuron repeats, if any
const relayOf = (n) => (n.base && n.base[0] === "n" && netData?.nodes["n" + n.base[1]]?.kind === "relay" ? netData.nodes["n" + n.base[1]] : null);
const delayName = (n) => {                     // A +3, OP +2...
  if (relayOf(n)) return `${relayOf(n).input} + ${n.k + 1} (input neuron of the Alt. OR, one step later)`;
  const base = n.label.replace(/ \+\d+$/, "");
  return `${base} + ${n.k}`;
};
// a list of names joined by an operator, shortened when long (A ∨ B ∨ … ∨ F)
const joinOp = (xs, op) => (xs.length > 4 ? [xs[0], xs[1], "…", xs[xs.length - 1]] : xs).join(` ${op} `);
const tIndex = (tid) => spec.transitions.findIndex((t) => t.id === tid);
const tNames = (pred) => spec.transitions.map((t, i) => (pred(t) ? `T${i}` : null)).filter(Boolean);
const notName = (n) => n.label.replace(/^NOT /, "");
// Inside a neuron: the name of the signal it carries (A+2, OP, N1) or the symbol of the gate it
// is (¬, ∧, ⩒, ⊽, ∨); below it (above, for the AND gates), the expression it computes or uses.
function netGlyph(n) {
  if (n.kind === "input") return n.label.length <= 3 ? n.label : n.label.slice(0, 2) + "…";
  if (n.kind === "init") return "init";
  if (n.kind === "op") return "OP";
  if (n.kind === "delay") return "OP+1";
  // the name of a delay neuron fits inside it when short (no label below, which wires would cross)
  if (n.kind === "delay_line" && relayOf(n)) return relayOf(n).input.length <= 3 ? `${relayOf(n).input}+${n.k + 1}` : "+" + (n.k + 1);
  if (n.kind === "delay_line") return n.label.replace(" ", "").length <= 6 ? n.label.replace(" ", "") : "+" + n.k;
  if (isTermLine(n)) return "+1";
  if (n.kind === "not_input" || n.kind === "not_state") return "¬";
  if (n.kind === "no_event") return "∧";
  if (n.kind === "nor") return "⊽";
  if (n.kind === "relay") return isRelayTap(n) ? (n.input.length <= 3 ? n.input + "+1" : "+1") : "⩒";
  if (n.kind === "term") return "∧";
  if (n.kind === "latch") return "N" + (n.j + 1);
  return "∨";
}
function netLabelLines(n) {
  if (n.kind === "input" && n.label.length > 3) return [n.label];
  if (n.kind === "not_input") return [`${bar(notName(n))}+3`];          // the input 3 steps later, negated
  if (n.kind === "not_state") return [bar(notName(n))];
  if (n.kind === "delay") return [];
  if (n.kind === "term_relay") return [`T${tIndex(n.tid)}+1`];
  if (n.kind === "reset_relay") return [`Reset ${stName(n.sid)}+1` + (n.tid ? ` (T${tIndex(n.tid)})` : "")];
  if (n.kind === "delay_line" && relayOf(n)) return relayOf(n).input.length <= 3 ? [] : [`${relayOf(n).input} +${n.k + 1}`];
  if (n.kind === "delay_line") return n.label.replace(" ", "").length <= 6 ? [] : [n.label];
  if (n.kind === "no_event") return ["¬OP(t) ∧ OP(t−1)"];
  if (n.kind === "nor") {
    const ts = n.role === "output" ? tNames((t) => (spec.states.find((x) => x.id === t.to) || {}).accepting)
      : tNames((t) => t.to === n.sid);
    return [`¬(${joinOp(ts, "∨")})`];
  }
  if (n.kind === "relay") {
    if (isRelayTap(n)) return [];
    const before = spec.inputs.slice(0, spec.inputs.indexOf(n.input));
    return [[n.input].concat(before.length > 2 ? [`¬(${joinOp(before, "∨")})`] : before.map((x) => "¬" + x)).join(" ∧ ")];
  }
  if (n.kind === "op") return [joinOp(spec.inputs, "∨")];
  if (n.kind === "output") return [joinOp(tNames((t) => (spec.states.find((x) => x.id === t.to) || {}).accepting), "∨")];
  if (n.kind === "term") {
    const t = transById(n.tid);
    const cond = n.cond || (t ? t.options[n.k] : null);
    if (n.via) return [(VIA_LABEL[n.via] || n.via) + (n.grp ? ` (${n.grp + 1})` : "")];   // parallel: the box of the transition has its name
    const lits = cond && Object.keys(cond).length ? optionText(cond) : "";
    return [termTitle(n), t ? [stName(t.from), lits].filter(Boolean).join(" ∧ ") : lits];
  }
  return [];
}
// full name of a neuron (tooltips, panels, raster rows)
function netName(n) {
  if (n.kind === "latch") return `${stName(n.sid)} · ${n.j ? `ring (${netGlyphOf(n)})` : `Set (${netGlyphOf(n)})`}`;
  const l = netLinesOf(n), g = netGlyphOf(n);
  const t = n.kind === "term" ? transById(n.tid) : null, cond = t ? (n.cond || t.options[n.k]) : null;
  switch (n.kind) {
    case "input": return `Input ${n.label}`;
    case "init": return "Initialisation";
    case "relay": return isRelayTap(n) ? `Delay neuron ${n.input} + 1 · input neuron of the Alt. OR`
      : `Alt. OR · input neuron ${n.input}`;
    case "op": return "OP: Alt. OR of the inputs + 2";
    case "delay": return "Delay neuron OP + 1";
    case "delay_line": return `Delay neuron ${delayName(n)}`;
    case "term_relay": return `Delay neuron ${termTitle(n)} + 1`;
    case "reset_relay": return `Delay neuron Reset ${stName(n.sid)} + 1`;
    case "not_input": return `NOT gate ${l[0]} (input ${notName(n)} 3 steps later, negated)`;
    case "not_state": return `NOT gate ${l[0]}`;
    case "no_event": return `Bypass guard ${l[0]}`;
    case "nor": return `NOR gate of the Inh.-based OR · ${n.role === "output" ? "Output" : "Set " + stName(n.sid)}`;
    case "output": return "Output (pattern found)";
    case "term": return n.via ? `AND gate ${termTitle(n)} · ${cond ? optionText(cond) : ""} · ${l[0]}`
      : `AND gate ${l[0]} · ${l[1]}`;
  }
  return l.length ? l[0] : netLbl(netId(n) + "#g", n.label);
}
// what a synapse does, in the words of the papers
function synRole(e) {
  const P = netData.nodes[e.from], Q = netData.nodes[e.to], ex = e.rec === "exc";
  const srcLine = (n) => n.kind === "input" || (n.kind === "delay_line" && n.base && n.base[0] === "src");
  const opLine = P.kind === "op" || (P.kind === "delay_line" && P.base && P.base[0] === "n" && netData.nodes["n" + P.base[1]]?.kind === "op");
  switch (Q.kind) {
    case "relay": return ex ? "input" : "priority: an earlier input spiked";
    case "op": return "Alt. OR";
    case "delay": case "delay_line": return "one step later";
    case "term_relay": return "the transition, one step later";
    case "reset_relay":
      if (ex) return opLine ? "input spikes 2 steps before: every input event leaves the state" : "the Reset, one step later";
      return opLine ? "the state was not left" : "the state was entered at the same time (entering wins)";
    case "not_input": return ex ? "operation signal (OP)" : "the input spiked";
    case "not_state":
      if (ex) return "operation signal (OP)";
      return P.kind === "latch" ? "the state is active" : "the state was just entered";
    case "no_event": return ex ? "OP at t − 1" : "OP at t: input spikes now";
    case "nor": return ex ? "operation signal (OP + 2)" : "an AND gate of the Inh.-based OR fired";
    case "output": return ex ? (P.kind === "term" ? "OR gate" : "operation signal (OP + 3)") : "no accepting transition (NOR)";
    case "term":
      if (ex) {
        if (P.kind === "term") return "bypass t−1: the source state was just entered";
        return srcLine(P) ? "positive input as operation signal (optimised AND gate)" : "operation signal OP + 1 (delay neuron)";
      }
      if (P.kind === "not_state") return "the source state is not active";
      if (P.kind === "not_input") return "a positive input did not spike";
      if (srcLine(P)) return "a negated input spiked";
      if (P.kind === "term") return Q.via === "bypass 1" ? "priority: an earlier group entered the source state" : "the source state changed at t − 1";
      if (opLine) return "input spikes at t − 1: every input event leaves the source state";
      if (P.kind === "term_relay" || P.kind === "reset_relay") return "the source state was left at t − 2";
      if (P.kind === "latch") return "the source state was already active (the latch case fires)";
      return "no input spikes (bypass guard)";
    case "latch":
      if (Q.j === 0) {
        if (ex) return P.kind === "init" ? "Set: active at start" : P.kind === "term" ? "Set (OR gate of exclusive inputs)" : "operation signal of the Inh.-based OR (OP + 3)";
        if (P.key === Q.key) return "self-inhibition";
        if (P.kind === "latch") return "the state is already active";
        if (P.kind === "nor") return "Inh.-based OR: no entering transition";
        return opLine ? "input spikes: cancels a Set still pending" : "cancels a Set still pending";
      }
      if (ex) return P.j === 0 ? "Set → ring" : "ring";
      return opLine ? "Reset: every input event leaves the state" : "Reset";
  }
  return ex ? "excitatory" : "inhibitory";
}

function drawNetwork() {
  const D = netData, show = { exc: $("#netExc").checked, inh: $("#netInh").checked, bypass: $("#netBypass").checked };
  const visible = (n) => !netHidden.has(n.kind);
  const parts = [];
  const ed = (key, def) => `data-lbl="${esc(key)}" data-def="${esc(def)}"`;
  for (const [i, b] of (D.stepBands || []).entries()) {
    if (b.k % 2) parts.push(`<rect class="step-band" x="${b.x0}" y="${D.box.y0}" width="${b.x1 - b.x0}" height="${D.box.y1 - D.box.y0}"/>`);
    if (i) parts.push(`<line class="step-sep" x1="${b.x0}" y1="${D.box.y0 + 4}" x2="${b.x0}" y2="${D.box.y1 - 4}"/>`);
  }
  for (const h of D.headers) parts.push(`<text class="${h.sub ? "col-sub" : "col-head"}" x="${h.x}" y="${h.y}" ${ed("head:" + h.c, h.text)}>${esc(netLbl("head:" + h.c, h.text))}</text>`);
  if (!netHidden.has("term")) for (const b of D.tboxes || []) {        // the AND gates of a term (parallel)
    const t = transById(b.n.tid), cond = b.n.cond || (t ? t.options[b.n.k] : null);
    const name = termTitle(b.n), key = `tbox:${b.n.tid}:${b.n.k}`;
    parts.push(`<rect class="term-box" data-tbox="${esc(b.n.tid)}|${b.n.k}" x="${b.x0}" y="${b.y0}" width="${b.x1 - b.x0}" height="${b.y1 - b.y0}" rx="8"/>
      <text class="term-name" x="${b.x0 + 7}" y="${b.y0 + 13}" ${ed(key, name)}>${esc(netLbl(key, name))}</text>
      <text class="term-cond" x="${b.x0 + 7}" y="${b.y0 + 25}">${esc(cond ? optionText(cond) : "")}</text>`);
  }
  if (!netHidden.has("latch")) for (const g of D.groups) {
    const def = netGroupDefault(g);
    parts.push(`<rect class="latch-box" data-grp="${esc(g.sid)}" x="${g.x0}" y="${g.y0}" width="${g.x1 - g.x0}" height="${g.y1 - g.y0}" rx="10"/>
      <text class="latch-name" x="${g.x0 + 10}" y="${g.y0 + 16}" ${ed("group:" + g.sid, def)}>${esc(netLbl("group:" + g.sid, def))}</text>`);
  }
  const eparts = [];
  for (const e of D.edges) {
    const P = D.nodes[e.from], Q = D.nodes[e.to];
    // bypasses have their own checkbox, independent of the excitatory / inhibitory ones
    if ((e.bypass ? !show.bypass : !show[e.rec]) || !visible(P) || !visible(Q)) continue;
    const d = netEdgePath(e, D);
    const tip = `${netName(P)} → ${netName(Q)} · ${synRole(e)} · ${e.rec === "exc" ? "excitatory" : "inhibitory"}${e.bypass ? " bypass" : ""}${e.delay !== 1 ? ` · delay ${e.delay}` : ""}`;
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
      ? lines.map((l, k) => `<text class="nl${k ? " sub" : ""}" y="${k || lines.length === 1 ? -NR - 5 : -NR - 17}" ${at(k)}>${esc(l.length > 26 ? l.slice(0, 25) + "…" : l)}</text>`).join("")
      : n.kind === "not_state"             // arcs arrive above and below: label on the upper left
      ? (lines.length ? `<text class="nl" x="${-NR - 4}" y="${-NR + 2}" style="text-anchor:end" ${at(0)}>${esc(lines[0])}</text>` : "")
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
  const layer = $("#netLayer");
  if (!netData) return;
  const syn = netHover ? null : netSynHover;
  let nb = { edges: new Set(), nodes: new Set() };
  const part = new Set();         // neurons shown at the end of selected synapses, but not selected
  if (netHover) nb = netNeighbours(netHover);
  else if (syn !== null) {        // a synapse: only it (its two neurons stay visible)
    const e = netData.edges[syn];
    nb = { edges: new Set([syn]), nodes: new Set([e.from, e.to]) };
  } else {                        // the selection: exactly what is counted as selected
    nb = { edges: netSelEdges(), nodes: new Set(netSel.n) };
    for (const i of nb.edges) { const e = netData.edges[i]; part.add(e.from); part.add(e.to); }
  }
  layer.classList.toggle("focus", !!netHover || syn !== null || netSelSize() > 0);
  $$(".syn", layer).forEach((el) => el.classList.toggle("sel", syn !== null ? +el.dataset.e === syn : netSel.e.has(+el.dataset.e)));
  // a single selected neuron also shows the path of its latency
  const lat = netPinned && !netHover && syn === null ? netLatency(netPinned) : null;
  const path = new Set(lat ? lat.path.map((e) => e.i) : []);
  if (lat) for (const e of lat.path) { part.add(e.from); part.add(e.to); }
  $$(".syn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.edges.has(+el.dataset.e));
    el.classList.toggle("path", path.has(+el.dataset.e));
  });
  $$(".nn", layer).forEach((el) => {
    el.classList.toggle("hi", nb.nodes.has(el.dataset.n));
    el.classList.toggle("part", !nb.nodes.has(el.dataset.n) && part.has(el.dataset.n));
    el.classList.toggle("pinned", netSel.n.has(el.dataset.n));
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
      <dt>Role</dt><dd>${esc(synRole(e))}</dd>
      <dt>Type</dt><dd><span class="syn-tag ${e.rec}${e.bypass ? " bypass" : ""}">${type}</span></dd>
      ${e.delay !== 1 ? `<dt>Delay</dt><dd>${e.delay} time steps</dd>` : ""}
    </dl>
`;
  $("#netUnpin").addEventListener("click", netClearSel);
  $$("[data-goto-n]", el).forEach((a) => a.addEventListener("click", (ev) => { ev.preventDefault(); netPick([a.dataset.gotoN]); }));
}

// Selected synapses: those clicked and every synapse of a selected neuron (what is highlighted).
function netSelEdges() {
  const s = new Set(netSel.e);
  for (const e of netData.edges) if (netSel.n.has(e.from) || netSel.n.has(e.to)) s.add(e.i);
  return s;
}
// Selected neurons (spike sources apart) and synapses (excitatory and inhibitory).
function netSelCounts() {
  const sel = [...netSel.n].map((k) => netData.nodes[k]).filter(Boolean), E = [...netSelEdges()].map((i) => netData.edges[i]);
  return { all: sel, edges: E, neurons: sel.filter((n) => !n.src).length, sources: sel.filter((n) => n.src).length,
    synapses: E.length, exc: E.filter((e) => e.rec === "exc").length, inh: E.filter((e) => e.rec === "inh").length };
}
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
// Summary panel: totals of the network and, with a selection, how many of them are selected
function renderNetCounts() {
  const c = netCountsData;
  if (!c || !netData) return;
  const sel = netSelSize() ? netSelCounts() : null;
  const head = sel ? `<span></span><span class="ch">Selected</span><span class="ch">Total</span>` : "";
  const row = (name, sub, s, t) => `<div class="cl">${name}<span>${sub}</span></div>${sel ? `<b class="cs">${s}</b>` : ""}<b class="ct">${t}</b>`;
  const nSub = sel ? `selected: ${plural(sel.neurons, "neuron")}${sel.sources ? ` + ${plural(sel.sources, "spike source")}` : ""} · total: ${c.neurons} + ${plural(c.sources, "spike source")}`
    : `without the ${plural(c.sources, "spike source")} (inputs and initialisation)`;
  const sSub = sel ? `selected: ${sel.exc} exc. · ${sel.inh} inh. · total: ${c.excitatory} exc. · ${c.inhibitory} inh.`
    : `${c.excitatory} excitatory · ${c.inhibitory} inhibitory`;
  $("#netCounts").innerHTML = `<div class="counts3${sel ? " with-sel" : ""}">${head}
    ${row("Neurons", nSub, sel && sel.neurons, c.neurons)}
    ${row("Synapses", sSub, sel && sel.synapses, c.synapses)}</div>
    <p class="note set-mode">${netData.setMode === "inhibition"
      ? "Inh.-based OR gates: some state can be entered by two AND gates at once, or left and entered at once."
      : "Direct OR gates: the AND gates that enter each state never fire together, so its Set neuron is excited by them directly."}</p>`;
}
function netSelDetails(el) {
  const r = netSelCounts(), N = netData.nodes;
  const dot = (k) => { const s = NET_STYLE[k] || NET_STYLE.output; return `<i class="dot${k === "input" || k === "init" ? " sq" : ""}" style="background:${s.fill};border-color:${s.stroke}"></i>`; };
  const kinds = new Map();
  for (const n of r.all) kinds.set(n.kind, (kinds.get(n.kind) || 0) + 1);
  const items = r.all.slice().sort((a, b) => a.x - b.x || a.y - b.y)
    .map((n) => `<li class="syn-row" data-goto-n="${n.key}" title="Select only this neuron">${dot(n.kind)}<span class="syn-name">${esc(netName(n))}</span></li>`)
    .concat(r.edges.map((e) => `<li class="syn-row" data-goto-e="${e.i}" title="Select only this synapse">
      <span class="syn-tag ${e.rec}${e.bypass ? " bypass" : ""}">${e.rec}</span><span class="syn-name">${esc(netName(N[e.from]))} → ${esc(netName(N[e.to]))}</span></li>`));
  const parts = [r.neurons && plural(r.neurons, "neuron"), r.sources && plural(r.sources, "spike source"), r.synapses && plural(r.synapses, "synapse")].filter(Boolean);
  el.hidden = false;
  el.innerHTML = `<div class="inspector-head"><h3>Selection</h3><button class="btn small" id="netUnpin">Clear</button></div>
    <p class="sel-count">${parts.join(" · ")}</p>
    ${kinds.size > 1 ? `<div class="sub-title">Neurons by type</div><ul class="syn-list">${[...kinds].map(([k, c]) => `<li>${dot(k)}<span class="syn-name">${esc((NET_STYLE[k] || NET_STYLE.output).name)}</span><b>${c}</b></li>`).join("")}</ul>` : ""}
    <div class="sub-title">Selected elements</div><ul class="syn-list sel-list">${items.join("")}</ul>`;
  $("#netUnpin").addEventListener("click", netClearSel);
  $$("[data-goto-n]", el).forEach((li) => li.addEventListener("click", () => netPick([li.dataset.gotoN])));
  $$("[data-goto-e]", el).forEach((li) => li.addEventListener("click", () => { netSynHover = null; netPick([], [+li.dataset.gotoE]); }));
}

function netDetails() {
  const el = $("#netDetails");
  renderNetCounts();
  if (netData && netSelSize() > 1) return netSelDetails(el);
  if (netSynPinned !== null && netData && netData.edges[netSynPinned]) return netSynDetails(el);
  el.hidden = !netPinned || !netData || !netData.nodes[netPinned];       // shown for a clicked neuron
  if (el.hidden) { el.innerHTML = ""; return; }
  const n = netData.nodes[netPinned], st = NET_STYLE[n.kind] || NET_STYLE.output;
  const desc = n.kind === "latch" ? (n.j ? NET_DESC.ring : NET_DESC.set) : NET_DESC[n.kind] || "";
  const row = (e, other) => {
    const o = netData.nodes[other], os = NET_STYLE[o.kind] || NET_STYLE.output;
    // a row is the synapse: clicking it selects the synapse (its card links to both neurons)
    return `<li class="syn-row" data-goto-e="${e.i}" title="${esc(synRole(e))} · select this synapse"><i class="dot" style="background:${os.fill};border-color:${os.stroke}"></i><span class="syn-name">${esc(netName(o))}<small class="syn-role">${esc(synRole(e))}</small></span>
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
  $("#netUnpin").addEventListener("click", netClearSel);
  $$("[data-goto-e]", el).forEach((li) => {
    li.addEventListener("click", () => { netSynHover = null; netPick([], [+li.dataset.gotoE]); });
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
  for (const k of [...netSel.n]) if (!netData.nodes[k]) netSel.n.delete(k);
  netSel.e.clear(); netSynHover = null;     // edge indices change with the network
  netSyncSel();
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
  const selN = [...netSel.n], selE = [...netSel.e], hover = netHover, sh = netSynHover;
  netSel.n.clear(); netSel.e.clear(); netHover = netSynHover = null; netHighlight();   // nothing highlighted in the image
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
  selN.forEach((k) => netSel.n.add(k)); selE.forEach((i) => netSel.e.add(i)); netHover = hover; netSynHover = sh; netHighlight();
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
    onBackgroundClick: (e) => {
      const add = e.ctrlKey || e.metaKey || e.shiftKey, box = netBoxAt(e.clientX, e.clientY);
      if (box && box.length) netPick(box, [], add);
      else if (!add && netSelSize()) netClearSel();
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
    const add = e.ctrlKey || e.metaKey || e.shiftKey, n = e.target.closest(".nn"), sy = n ? null : e.target.closest(".syn");
    if (n) netPick([n.dataset.n], [], add);
    else if (sy) netPick([], [+sy.dataset.e], add);
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
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#codeModal").hidden) $("#codeModal").hidden = true;
    else if ($("#tab-network").classList.contains("active") && netSelSize() && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) netClearSel();
  });
  $("#codeCopy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(codeCache.text); toast("Code copied to the clipboard"); }
    catch (err) { toast("Could not copy: select the code and copy it by hand"); }
  });
  $("#codeDownload").addEventListener("click", () => download(codeCache.filename, codeCache.text, "text/x-python"));
}

// one checkbox per neuron type present in the network, with its colour
function renderNetKinds() {
  const present = new Set(Object.values(netData.nodes).map((n) => n.kind));
  $("#netKinds").innerHTML = netGroups().filter(([, , kinds]) => kinds.some((k) => present.has(k))).map(([id, label, kinds]) => {
    const st = NET_STYLE[kinds.find((k) => present.has(k)) || kinds[0]];
    return `<label class="chk"><input type="checkbox" data-group="${id}" ${netHidden.has(kinds[0]) ? "" : "checked"}>
      <i class="dot${id === "source" ? " sq" : ""}" style="background:${st.fill};border-color:${st.stroke}"></i>${label}</label>`;
  }).join("");
}
