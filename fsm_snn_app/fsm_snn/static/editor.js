/* Design tab: the state machine editor (canvas, selection, clipboard, inspector
 * panel and live validation) and the machine's file operations. */
"use strict";

const R = 32;                   // state radius on the canvas
const GRID = 22;                // grid cell, same as the dots of the canvas

let spec = null;                // the machine being edited
let sel = { type: null, id: null };   // selection: state | transition | multi (ids) | null
let undoStack = [];
let pickMode = null;            // {from}: waiting for the destination of a new transition
let designVp = null;            // pan / zoom of the canvas (viewport.js)
let snap = true;                // snap states to the grid while dragging
let clip = null, pasteN = 0;    // copied states and transitions
let lastNudge = 0;              // groups arrow-key nudges into one undo step

// ------------------------------------------------------------------ model helpers
const stateById = (id) => spec.states.find((s) => s.id === id);
const transById = (id) => spec.transitions.find((t) => t.id === id);
function pushUndo() { undoStack.push(JSON.stringify(spec)); if (undoStack.length > 80) undoStack.shift(); }
function undo() {
  if (!undoStack.length) return toast("Nothing to undo");
  spec = JSON.parse(undoStack.pop()); sel = { type: null, id: null }; changed(false);
}
function newId(prefix, list) { let k = 0; while (list.some((x) => x.id === prefix + k)) k++; return prefix + k; }
function newStateName() { let k = 0; while (spec.states.some((s) => s.name === "S" + k)) k++; return "S" + k; }
function changed(undoable = true) {
  render(); renderInspector(); scheduleCheck(); store("fsm_snn_spec", spec);
  if (undoable) $("#btnUndo").disabled = false;
  if (!dirty) markDirty(true);
  if ($("#tab-network").classList.contains("active")) loadNetwork();
}
// States are always held latches: drop the old per-state modes from loaded machines.
function sanitize(s) {
  s.transitions = s.transitions || [];
  // the output neuron is internal: the states themselves are recorded
  for (const st of s.states) { delete st.always_active; delete st.on_mismatch; delete st.accepting; }
  return s;
}

// ------------------------------------------------------------------ canvas rendering
const svg = () => $("#canvas");
function edgeGeometry(t) {
  const a = stateById(t.from), b = stateById(t.to);
  if (!a || !b) return null;
  if (a.id === b.id) {
    const sameLoops = spec.transitions.filter((x) => x.from === a.id && x.to === a.id);
    const k = sameLoops.indexOf(t);
    const h = 62 + 26 * k;
    const p1 = [a.x - R * Math.sin(Math.PI / 5), a.y - R * Math.cos(Math.PI / 5)];
    const p2 = [a.x + R * Math.sin(Math.PI / 5), a.y - R * Math.cos(Math.PI / 5)];
    const d = `M${p1[0]},${p1[1]} C${a.x - 46},${a.y - h - 10} ${a.x + 46},${a.y - h - 10} ${p2[0]},${p2[1]}`;
    return { d, lx: a.x, ly: a.y - h + 6 };
  }
  const reverse = spec.transitions.some((x) => x.from === b.id && x.to === a.id);
  const same = spec.transitions.filter((x) => x.from === a.id && x.to === b.id);
  const k = same.indexOf(t);
  let off = (reverse ? 30 : 0) + 34 * k;
  const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L, ny = dx / L;
  // bend the arc when the straight line would cross another state
  for (const o of spec.states) {
    if (o.id === a.id || o.id === b.id) continue;
    const u = ((o.x - a.x) * dx + (o.y - a.y) * dy) / (L * L);
    if (u <= 0 || u >= 1) continue;
    const dist = Math.abs((o.x - a.x) * nx + (o.y - a.y) * ny);
    if (dist < R + 22) off = Math.max(off, 2 * (R + 40) + 34 * k);
  }
  const cx = (a.x + b.x) / 2 + nx * off, cy = (a.y + b.y) / 2 + ny * off;
  const ua = [cx - a.x, cy - a.y], la = Math.hypot(...ua) || 1;
  const ub = [cx - b.x, cy - b.y], lb = Math.hypot(...ub) || 1;
  const s = [a.x + ua[0] / la * R, a.y + ua[1] / la * R];
  const e = [b.x + ub[0] / lb * (R + 3), b.y + ub[1] / lb * (R + 3)];
  const d = `M${s[0]},${s[1]} Q${cx},${cy} ${e[0]},${e[1]}`;
  return { d, lx: 0.25 * s[0] + 0.5 * cx + 0.25 * e[0], ly: 0.25 * s[1] + 0.5 * cy + 0.25 * e[1] };
}

function render() {
  const eg = [], ng = [], lg = [];
  const selIds = selectedStates();
  for (const t of spec.transitions) {
    const g = edgeGeometry(t); if (!g) continue;
    const selected = sel.type === "transition" && sel.id === t.id;
    const lbl = transLabel(t);
    const shown = lbl.length > 40 ? lbl.slice(0, 38) + "…" : lbl;
    const w = shown.length * 6.4 + 16;
    // lift the label above the arrow when it does not fit between the two states
    const A = stateById(t.from), B = stateById(t.to);
    if (A && B && A.id !== B.id && w > Math.hypot(B.x - A.x, B.y - A.y) - 2 * R - 8) g.ly -= 26;
    eg.push(`<g class="edge${selected ? " selected" : ""}" data-tid="${t.id}">
      <path class="hit" d="${g.d}"/>
      <path class="line" d="${g.d}" marker-end="url(#${selected ? "arrowSel" : "arrow"})"/></g>`);
    lg.push(`<g class="edge${selected ? " selected" : ""}" data-tid="${t.id}">
      <rect class="lbl" x="${g.lx - w / 2}" y="${g.ly - 11}" width="${w}" height="22"/>
      <text x="${g.lx}" y="${g.ly + 4}" text-anchor="middle">${esc(shown)}</text></g>`);
  }
  for (const s of spec.states) {
    const selected = selIds.has(s.id);
    const name = s.name.length > 9 ? s.name.slice(0, 8) + "…" : s.name;
    ng.push(`<g class="node${selected ? " selected" : ""}" data-sid="${s.id}" transform="translate(${s.x},${s.y})">
      ${s.initial ? `<path class="init-arrow" d="M${-R - 34},0 L${-R - 4},0" marker-end="url(#arrow)"/>` : ""}
      <circle class="ring" r="${R + 5}"/>
      <circle class="body" r="${R}"/>
      <text y="0" style="font-size:${name.length > 6 ? 11 : 13}px">${esc(name)}</text>
</g>`);
  }
  $("#edges").innerHTML = eg.join("");
  $("#nodes").innerHTML = ng.join("");
  $("#labels").innerHTML = lg.join("");
}

// ------------------------------------------------------------------ canvas interaction
let drag = null;      // {sid, p0, orig: {id: [x, y]}, moved, snapshot}
let connect = null;   // {from}
function svgPoint(evt) { return designVp.toWorld(evt.clientX, evt.clientY); }
function stateAt(p) { return spec.states.find((s) => Math.hypot(s.x - p.x, s.y - p.y) <= R + 10); }
function select(type, id) { sel = { type, id }; render(); renderInspector(); }
function selectedStates() {
  if (sel.type === "state") return new Set([sel.id]);
  if (sel.type === "multi") return new Set(sel.ids);
  return new Set();
}
function selectStates(ids) {
  ids = [...new Set(ids)].filter((id) => stateById(id));
  if (!ids.length) sel = { type: null, id: null };
  else if (ids.length === 1) sel = { type: "state", id: ids[0] };
  else sel = { type: "multi", ids };
  render(); renderInspector();
}
const snapTo = (v) => Math.round(v / GRID) * GRID;
function contentBox() {
  if (!spec.states.length) return null;
  const xs = spec.states.map((s) => s.x), ys = spec.states.map((s) => s.y);
  return { x0: Math.min(...xs) - R - 60, y0: Math.min(...ys) - R - 50, x1: Math.max(...xs) + R + 60, y1: Math.max(...ys) + R + 50 };
}
function fitDesign() {
  const b = contentBox();
  if (b) designVp.fit(b, 30, 1);
  else { designVp.k = 1; designVp.tx = 0; designVp.ty = 0; designVp.apply(); }
}
function fitIfOutside() {
  const b = contentBox(), v = designVp.visible();
  if (b && (b.x0 < v.x0 || b.y0 < v.y0 || b.x1 > v.x1 || b.y1 > v.y1)) fitDesign();
}

function setHint(text) {
  $("#canvasHint").textContent = text ||
    "Double-click: new state · Drag from the edge of a state: new transition · Drag the background: select · Space + drag or right drag: move";
}
function addTransition(from, to) {
  pushUndo();
  const id = newId("t", spec.transitions);
  const first = spec.inputs[0];
  spec.transitions.push({ id, from, to, options: [first ? { [first]: "on" } : {}] });
  sel = { type: "transition", id };
  changed();
  toast("Transition created: set when it fires in the right-hand panel");
}
function addState(x, y) {
  pushUndo();
  const id = newId("s", spec.states);
  if (snap) { x = snapTo(x); y = snapTo(y); }
  spec.states.push({ id, name: newStateName(), x: Math.round(x), y: Math.round(y), initial: spec.states.length === 0 });
  sel = { type: "state", id };
  changed();
}
function deleteSelection() {
  const ids = selectedStates();
  if (ids.size) {
    pushUndo();
    spec.states = spec.states.filter((s) => !ids.has(s.id));
    spec.transitions = spec.transitions.filter((t) => !ids.has(t.from) && !ids.has(t.to));
  } else if (sel.type === "transition") {
    pushUndo();
    spec.transitions = spec.transitions.filter((t) => t.id !== sel.id);
  } else return;
  sel = { type: null, id: null }; changed();
}

// ---- clipboard: states plus the transitions between them
function copySelection() {
  const ids = selectedStates();
  if (!ids.size) return false;
  clip = {
    states: spec.states.filter((x) => ids.has(x.id)).map((x) => ({ ...x })),
    transitions: spec.transitions.filter((t) => ids.has(t.from) && ids.has(t.to)).map((t) => JSON.parse(JSON.stringify(t))),
  };
  pasteN = 0;
  return true;
}
function pasteClipboard() {
  if (!clip || !clip.states.length) return toast("Nothing to paste");
  pushUndo(); pasteN++;
  const off = 2 * GRID * pasteN, map = {}, ids = [];
  for (const x of clip.states) {
    const id = newId("s", spec.states); map[x.id] = id; ids.push(id);
    const name = spec.states.some((y) => y.name === x.name) ? newStateName() : x.name;
    spec.states.push({ ...x, id, name, x: x.x + off, y: x.y + off });
  }
  for (const t of clip.transitions) {
    spec.transitions.push({ ...JSON.parse(JSON.stringify(t)), id: newId("t", spec.transitions), from: map[t.from], to: map[t.to] });
  }
  sel = ids.length === 1 ? { type: "state", id: ids[0] } : { type: "multi", ids };
  changed();
}
function duplicateSelection() {
  const saved = clip, n = pasteN;
  if (!copySelection()) return;
  pasteClipboard(); clip = saved; pasteN = n;
}
function nudge(dx, dy) {
  const ids = selectedStates(); if (!ids.size) return false;
  if (Date.now() - lastNudge > 700) pushUndo();      // one undo step per burst of key presses
  lastNudge = Date.now();
  for (const id of ids) { const x = stateById(id); x.x += dx; x.y += dy; }
  render(); store("fsm_snn_spec", spec); $("#btnUndo").disabled = false;
  return true;
}
function setSnap(on) {
  snap = on; store("fsm_snn_grid", on);
  $("#btnGrid").classList.toggle("on", on);
  svg().classList.toggle("nogrid", !on);
}

let band = null;      // rubber-band selection {x0, y0, add, base, moved}
function initCanvas() {
  const c = svg();
  // left drag on the background draws a selection rectangle; Space + drag, right or middle drag pans
  designVp = createViewport(c, $("#viewport"), { grid: GRID });
  mountZoomBar($("#designZoom"), designVp, fitDesign);
  designVp.apply();
  c.addEventListener("pointerdown", (evt) => {
    if (evt.button !== 0) return;
    c.focus();
    const p = svgPoint(evt);
    const nodeEl = evt.target.closest(".node");
    const edgeEl = evt.target.closest(".edge");
    if (pickMode) {
      const s = nodeEl ? stateById(nodeEl.dataset.sid) : null;
      if (!s) { pickMode = null; setHint(); return; }
      const from = pickMode.from; pickMode = null; setHint(); addTransition(from, s.id);
      return;
    }
    if (nodeEl) {
      const s = stateById(nodeEl.dataset.sid);
      if (evt.shiftKey || evt.ctrlKey || evt.metaKey) {     // add to / remove from the selection
        const cur = selectedStates();
        if (cur.has(s.id)) cur.delete(s.id); else cur.add(s.id);
        return selectStates([...cur]);
      }
      if (evt.target.classList.contains("ring")) {
        connect = { from: s.id }; c.classList.add("connecting");
      } else {
        if (!selectedStates().has(s.id)) select("state", s.id);
        const orig = {};
        for (const id of selectedStates()) { const x = stateById(id); orig[id] = [x.x, x.y]; }
        drag = { sid: s.id, p0: p, orig, moved: false, snapshot: JSON.stringify(spec) };
      }
      c.setPointerCapture(evt.pointerId);
    } else if (edgeEl) {
      select("transition", edgeEl.dataset.tid);
    } else {
      // selection rectangle; with Shift / Ctrl it adds to the current selection
      const add = evt.shiftKey || evt.ctrlKey || evt.metaKey;
      band = { x0: p.x, y0: p.y, add, base: add ? [...selectedStates()] : [], moved: false };
      c.setPointerCapture(evt.pointerId);
    }
  });
  c.addEventListener("pointermove", (evt) => {
    const p = svgPoint(evt);
    if (drag) {
      if (!drag.moved && Math.hypot(p.x - drag.p0.x, p.y - drag.p0.y) * designVp.k < 3) return;
      const [ox, oy] = drag.orig[drag.sid];
      let nx = ox + p.x - drag.p0.x, ny = oy + p.y - drag.p0.y;
      if (snap && !evt.altKey) { nx = snapTo(nx); ny = snapTo(ny); }
      const dx = Math.round(nx - ox), dy = Math.round(ny - oy), s = stateById(drag.sid);
      if (s.x !== ox + dx || s.y !== oy + dy) {
        if (!drag.moved) { undoStack.push(drag.snapshot); drag.moved = true; }
        for (const [id, [x, y]] of Object.entries(drag.orig)) { const st = stateById(id); st.x = x + dx; st.y = y + dy; }
        render();
      }
    } else if (band) {
      const x = Math.min(band.x0, p.x), y = Math.min(band.y0, p.y), w = Math.abs(p.x - band.x0), h = Math.abs(p.y - band.y0);
      if ((w + h) * designVp.k > 4) band.moved = true;
      const r = $("#band");
      r.setAttribute("x", x); r.setAttribute("y", y); r.setAttribute("width", w); r.setAttribute("height", h);
      r.style.display = band.moved ? "" : "none";
    } else if (connect) {
      const a = stateById(connect.from);
      const tgt = stateAt(p);
      let d;
      if (tgt && tgt.id === a.id) d = `M${a.x - 18},${a.y - R + 6} C${a.x - 46},${a.y - 80} ${a.x + 46},${a.y - 80} ${a.x + 18},${a.y - R + 6}`;
      else d = `M${a.x},${a.y} L${tgt ? tgt.x : p.x},${tgt ? tgt.y : p.y}`;
      $("#ghost").setAttribute("d", d);
    }
  });
  const finish = (evt) => {
    if (connect) {
      const tgt = stateAt(svgPoint(evt));
      const from = connect.from;
      connect = null; c.classList.remove("connecting"); $("#ghost").setAttribute("d", "");
      if (tgt) addTransition(from, tgt.id);
    }
    if (drag) {
      const moved = drag.moved; drag = null;
      if (moved) { store("fsm_snn_spec", spec); $("#btnUndo").disabled = false; }
      render();
    }
    if (band) {
      const b = band, p = svgPoint(evt); band = null; $("#band").style.display = "none";
      if (!b.moved) { if (!b.add) select(null, null); return; }      // plain click: clear the selection
      const x0 = Math.min(b.x0, p.x), x1 = Math.max(b.x0, p.x), y0 = Math.min(b.y0, p.y), y1 = Math.max(b.y0, p.y);
      const inside = spec.states.filter((s) => s.x >= x0 && s.x <= x1 && s.y >= y0 && s.y <= y1).map((s) => s.id);
      selectStates(b.add ? [...b.base, ...inside] : inside);
    }
  };
  c.addEventListener("pointerup", finish);
  c.addEventListener("pointercancel", finish);
  c.addEventListener("dblclick", (evt) => {
    if (evt.target.closest(".node") || evt.target.closest(".edge")) return;
    const p = svgPoint(evt); addState(p.x, p.y);
  });
  document.addEventListener("keydown", (evt) => {
    const inField = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
    const mod = evt.ctrlKey || evt.metaKey, key = evt.key.toLowerCase();
    const design = $("#tab-design").classList.contains("active");
    const network = $("#tab-network").classList.contains("active");
    if (evt.key === "Escape") { pickMode = null; setHint(); if (design) select(null, null); $("#recentMenu").hidden = true; return; }
    if (mod && (key === "s" || key === "o")) {          // also from a field: commit its value first
      evt.preventDefault(); document.activeElement.blur?.();
      if (key === "s") saveSpec(evt.shiftKey); else openSpec();
      return;
    }
    if (inField) return;
    if (mod && key === "z") { evt.preventDefault(); undo(); return; }
    const vp = design ? designVp : network ? netVp : null;      // zoom the visible canvas
    if (vp && mod) {
      if (key === "=" || key === "+") { evt.preventDefault(); vp.zoomBy(1.2); return; }
      if (key === "-") { evt.preventDefault(); vp.zoomBy(1 / 1.2); return; }
      if (key === "0") { evt.preventDefault(); vp.setZoom(1); return; }
      if (evt.shiftKey && key === "h") { evt.preventDefault(); if (design) fitDesign(); else netFit(); return; }
    }
    if (!design) return;
    if (evt.key === "Delete" || evt.key === "Backspace") { evt.preventDefault(); deleteSelection(); }
    else if (mod && key === "a") { evt.preventDefault(); selectStates(spec.states.map((x) => x.id)); }
    else if (mod && key === "c") { if (copySelection()) toast(`Copied ${clip.states.length} state(s)`); }
    else if (mod && key === "x") { if (copySelection()) deleteSelection(); }
    else if (mod && key === "v") { evt.preventDefault(); pasteClipboard(); }
    else if (mod && key === "d") { evt.preventDefault(); duplicateSelection(); }
    else if (evt.key.startsWith("Arrow") && !mod) {
      const st = evt.shiftKey ? GRID : 1;
      const d = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, -st], ArrowDown: [0, st] }[evt.key];
      if (d && nudge(...d)) evt.preventDefault();
    }
  });
}

function autoLayout() {
  if (!spec.states.length) return;
  pushUndo();
  const n = spec.states.length, c = designVp.center(), rad = Math.max(140, 34 * n);
  spec.states.forEach((s, k) => {
    const a = -Math.PI / 2 + 2 * Math.PI * k / Math.max(n, 1);
    let x = c.x + rad * Math.cos(a), y = c.y + rad * Math.sin(a);
    if (snap) { x = snapTo(x); y = snapTo(y); }
    s.x = Math.round(x); s.y = Math.round(y);
  });
  changed(); fitDesign();
}

// ------------------------------------------------------------------ inspector
function renderInspector() {
  const el = $("#inspector");
  if (sel.type === "state" && stateById(sel.id)) el.innerHTML = stateInspector(stateById(sel.id));
  else if (sel.type === "transition" && transById(sel.id)) el.innerHTML = transInspector(transById(sel.id));
  else if (sel.type === "multi") el.innerHTML = multiInspector(sel.ids);
  else { sel = { type: null, id: null }; el.innerHTML = machineInspector(); }
  bindInspector();
}

function machineInspector() {
  const chips = spec.inputs.map((i, k) => `<span class="chip"><input data-input="${k}" value="${esc(i)}" aria-label="Input name"><button data-delinput="${k}" title="Remove input">×</button></span>`).join("");
  return `<div class="card">
    <h3>Machine</h3>
    <label class="field">Name <input id="fsmName" value="${esc(spec.name)}"></label>
    <div class="field">Input signals <span class="unit">(outputs of other networks, such as A, B, C, D)</span>
      <div class="chips">${chips}<button class="btn small" id="addInput">＋ Input</button></div>
    </div>
    <p class="note">Every state is a spiking latch: once active, it stays active until one of its outgoing transitions fires.</p>
  </div>`;
}

function multiInspector(ids) {
  const all = ids.every((id) => stateById(id).initial);
  return `<div class="card">
    <div class="inspector-head"><h3>${ids.length} states selected</h3><button class="btn small danger" id="delMulti">Delete</button></div>
    <p class="note">Drag any of them to move them together, or nudge them with the arrow keys.</p>
    <div class="btn-row">
      <button class="btn small" id="dupMulti" title="Ctrl + D">Duplicate</button>
      <button class="btn small" id="alignH" title="Same vertical position">Align horizontally</button>
      <button class="btn small" id="alignV" title="Same horizontal position">Align vertically</button>
    </div>
    <label class="toggle-row"><input type="checkbox" id="multiInitial" ${all ? "checked" : ""}><div><b>Active at start</b><span>Applies to every selected state.</span></div></label>
  </div>`;
}

function stateInspector(s) {
  const outs = spec.transitions.filter((t) => t.from === s.id);
  const list = outs.length ? outs.map((t) => `<li><a href="#" data-goto="${t.id}">→ ${esc(stateById(t.to)?.name || "?")}</a>: <span class="expr">${esc(transLabel(t))}</span></li>`).join("") : "<li class='muted'>None yet</li>";
  return `<div class="card">
    <div class="inspector-head"><h3>State</h3><button class="btn small danger" id="delState">Delete</button></div>
    <label class="field">Name <input id="stName" value="${esc(s.name)}"></label>
    <label class="toggle-row"><input type="checkbox" id="stInitial" ${s.initial ? "checked" : ""}><div><b>Active at start</b><span>The state is active when the simulation starts. Several states may start active (e.g. non-deterministic machines).</span></div></label>
  </div>
  <div class="card">
    <div class="inspector-head"><h3>Outgoing transitions</h3><button class="btn small" id="newFromHere">＋ New</button></div>
    <ul class="note" style="padding-left:16px">${list}</ul>
  </div>`;
}

function transInspector(t) {
  const opts = spec.states.map((s) => `<option value="${s.id}">${esc(s.name)}</option>`).join("");
  const cards = t.options.map((opt, k) => {
    const rows = spec.inputs.map((i) => {
      const v = opt[i] || "any";
      return `<div class="name">${esc(i)}</div><div class="tri" data-opt="${k}" data-input="${esc(i)}">
        <button class="on ${v === "on" ? "active" : ""}" data-v="on" title="${esc(i)} must spike">${esc(i)}</button>
        <button class="off ${v === "off" ? "active" : ""}" data-v="off" title="${esc(i)} must not spike">${OP.not}${esc(i)}</button>
        <button class="any ${v === "any" ? "active" : ""}" data-v="any" title="${esc(i)} does not matter">–</button></div>`;
    }).join("");
    return `${k ? `<div class="op-sep">${OP.or}</div>` : ""}<div class="opt-card">
      <div class="opt-head"><span>Term ${k + 1}: <span class="expr">${esc(optionText(opt))}</span></span>
      ${t.options.length > 1 ? `<button class="btn small danger" data-delopt="${k}">Remove</button>` : ""}</div>
      <div class="input-grid">${rows}</div></div>`;
  }).join("");
  const opBtns = [[OP.not, "NOT"], [OP.and, "AND"], [OP.or, "OR"], ["(", ""], [")", ""]]
    .map(([o, n]) => `<button class="${n ? "op" : ""}" data-ins="${o}" title="${n || o}">${o}</button>`).join("")
    + spec.inputs.map((i) => `<button data-ins="${esc(i)}">${esc(i)}</button>`).join("");
  return `<div class="card">
    <div class="inspector-head"><h3>Transition</h3><button class="btn small danger" id="delTrans">Delete</button></div>
    <div class="row2">
      <label class="field">From <select id="trFrom">${opts}</select></label>
      <label class="field">To <select id="trTo">${opts}</select></label>
    </div>
    <label class="field">Condition
      <input id="trExpr" class="expr" value="${esc(transLabel(t))}" spellcheck="false" autocomplete="off"></label>
    <div class="op-bar">${opBtns}</div>
    <div id="exprErr" class="msg err" hidden></div>
    <p class="note">Which inputs must (or must not) spike in the same time step, combined with ${OP.not} (NOT), ${OP.and} (AND) and ${OP.or} (OR). Press Enter to apply. You can also type <b>!</b>, <b>&amp;</b> and <b>|</b>.</p>
  </div>
  <div class="card">
    <h3>Terms <span class="unit">(joined by ${OP.or})</span></h3>
    <p class="note">The same condition as a sum of products. In each term, an input must spike (${esc(spec.inputs[0] || "X")}), must not spike (${OP.not}${esc(spec.inputs[0] || "X")}) or does not matter (–).</p>
    ${cards}
    <button class="btn" id="addOpt">＋ Add term (${OP.or})</button>
  </div>`;
}

function bindInspector() {
  const el = $("#inspector");
  // machine
  $("#fsmName", el)?.addEventListener("change", (e) => { pushUndo(); spec.name = e.target.value || "FSM"; changed(); });
  $$("[data-input]", el).forEach((inp) => {
    if (inp.tagName !== "INPUT") return;
    inp.addEventListener("change", (e) => {
      const k = +e.target.dataset.input, old = spec.inputs[k], nu = e.target.value.trim();
      if (!nu || (nu !== old && spec.inputs.includes(nu))) { toast("Empty or duplicate name"); e.target.value = old; return; }
      pushUndo(); spec.inputs[k] = nu;
      for (const t of spec.transitions) for (const o of t.options) if (old in o) { o[nu] = o[old]; delete o[old]; delete t.expr; }
      changed();
    });
  });
  $$("[data-delinput]", el).forEach((b) => b.addEventListener("click", () => {
    const k = +b.dataset.delinput, name = spec.inputs[k];
    if (spec.inputs.length === 1) return toast("The machine needs at least one input");
    pushUndo(); spec.inputs.splice(k, 1);
    for (const t of spec.transitions) for (const o of t.options) if (name in o) { delete o[name]; delete t.expr; }
    changed();
  }));
  $("#addInput", el)?.addEventListener("click", () => {
    pushUndo(); let c = 65; while (spec.inputs.includes(String.fromCharCode(c)) && c < 90) c++;
    let name = String.fromCharCode(c); if (spec.inputs.includes(name)) name = "E" + spec.inputs.length;
    spec.inputs.push(name); changed();
  });
  // multi-selection
  if (sel.type === "multi") {
    const list = sel.ids.map(stateById);
    const mean = (k) => list.reduce((a, x) => a + x[k], 0) / list.length;
    $("#delMulti", el).addEventListener("click", deleteSelection);
    $("#dupMulti", el).addEventListener("click", duplicateSelection);
    $("#alignH", el).addEventListener("click", () => { pushUndo(); const y = Math.round(mean("y")); list.forEach((x) => (x.y = y)); changed(); });
    $("#alignV", el).addEventListener("click", () => { pushUndo(); const x0 = Math.round(mean("x")); list.forEach((x) => (x.x = x0)); changed(); });
    $("#multiInitial", el).addEventListener("change", (e) => { pushUndo(); list.forEach((x) => (x.initial = e.target.checked)); changed(); });
  }
  // state
  if (sel.type === "state") {
    const s = stateById(sel.id);
    $("#stName", el).addEventListener("change", (e) => { pushUndo(); s.name = e.target.value.trim() || s.name; changed(); });
    const flag = (id, key) => $(id, el).addEventListener("change", (e) => { pushUndo(); s[key] = e.target.checked; changed(); });
    flag("#stInitial", "initial");
    $("#delState", el).addEventListener("click", deleteSelection);
    $("#newFromHere", el).addEventListener("click", () => { pickMode = { from: s.id }; setHint("Click the destination state (the same state creates a loop)"); toast("Click the destination state"); });
    $$("[data-goto]", el).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); select("transition", a.dataset.goto); }));
  }
  // transition
  if (sel.type === "transition") {
    const t = transById(sel.id);
    $("#trFrom", el).value = t.from; $("#trTo", el).value = t.to;
    $("#trFrom", el).addEventListener("change", (e) => { pushUndo(); t.from = e.target.value; changed(); });
    $("#trTo", el).addEventListener("change", (e) => { pushUndo(); t.to = e.target.value; changed(); });
    $$(".tri button", el).forEach((b) => b.addEventListener("click", () => {
      const box = b.parentElement, k = +box.dataset.opt, name = box.dataset.input;
      pushUndo(); delete t.expr;
      if (b.dataset.v === "any") delete t.options[k][name]; else t.options[k][name] = b.dataset.v;
      changed();
    }));
    $$("[data-delopt]", el).forEach((b) => b.addEventListener("click", () => { pushUndo(); delete t.expr; t.options.splice(+b.dataset.delopt, 1); changed(); }));
    $("#addOpt", el).addEventListener("click", () => { pushUndo(); delete t.expr; t.options.push(spec.inputs[0] ? { [spec.inputs[0]]: "on" } : {}); changed(); });
    const ex = $("#trExpr", el), err = $("#exprErr", el);
    const applyExpr = () => {
      if (ex.value.trim() === transLabel(t)) return;
      try {
        const ast = parseExpr(ex.value);
        const terms = simplifyDnf(toDnf(ast));
        if (!terms.length) throw new Error("This condition can never be true");
        pushUndo(); t.options = terms; t.expr = prettyExpr(ast); changed();
      } catch (e) { err.textContent = e.message; err.hidden = false; }
    };
    ex.addEventListener("change", applyExpr);
    ex.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); applyExpr(); } });
    ex.addEventListener("input", () => { err.hidden = true; });
    $$("[data-ins]", el).forEach((b) => {
      b.addEventListener("mousedown", (e) => e.preventDefault());   // keep the caret in the field
      b.addEventListener("click", () => {
        const ins = b.dataset.ins, a = ex.selectionStart ?? ex.value.length, z = ex.selectionEnd ?? a;
        const before = ex.value.slice(0, a), after = ex.value.slice(z);
        const binary = ins === OP.and || ins === OP.or;
        const lead = before && !/[\s(¬]$/.test(before) && ins !== ")" ? " " : "";
        const txt = lead + ins + (binary ? " " : "");
        ex.value = before + txt + after; ex.focus();
        const c = before.length + txt.length; ex.setSelectionRange(c, c);
        err.hidden = true;
      });
    });
    $("#delTrans", el).addEventListener("click", deleteSelection);
  }
}

// ------------------------------------------------------------------ live validation (Summary card)
let checkTimer = null;
function scheduleCheck() { clearTimeout(checkTimer); checkTimer = setTimeout(runCheck, 250); }
async function runCheck() {
  const r = await api("/api/check", { spec });
  const msgs = [];
  (r.errors || []).forEach((m) => msgs.push(`<div class="msg err">${esc(m)}</div>`));
  (r.warnings || []).forEach((m) => msgs.push(`<div class="msg warn">${esc(m)}</div>`));
  const notes = (r.notes || []).map((m) => `<div class="msg info">${esc(m)}</div>`);
  // only the machine here: the size of the spiking network is in the Network tab
  $("#fsmSummary").innerHTML = `<dl class="kv">
      <dt>States</dt><dd>${spec.states.length}</dd>
      <dt>Transitions</dt><dd>${spec.transitions.length}</dd>
      <dt>Inputs</dt><dd>${spec.inputs.length}</dd></dl>`;
  if (!(r.errors || []).length) msgs.push(...notes);
  $("#messages").innerHTML = msgs.join("");
}

// ------------------------------------------------------------------ new, save, open
// The local server reads and writes the design files (fsm_snn/files.py), so this works
// in any browser: Save writes over the file that was opened or saved last, and the
// recent files are remembered by their path. The file dialog browses the disk.
// The web version (web.js) has no disk access: Open reads a file chosen in the
// browser, Save downloads it, the recent files are copies kept in the browser and
// the Recent menu also offers the bundled example machines.
const WEB = !!window.FSM_WEB;
const MAX_RECENT = 8;
let filePath = null;            // file of the machine being edited
let dirty = false;              // changes since the last open or save

const baseName = (p) => (p ? p.split(/[\\/]/).pop() : "");
const dirName = (p) => p.replace(/[\\/][^\\/]*$/, "");
function markDirty(on) { dirty = on; store("fsm_snn_dirty", on); renderFileName(); }
function renderFileName() {
  const el = $("#fileName");
  el.textContent = filePath ? baseName(filePath) + (dirty ? " •" : "") : "";
  el.title = filePath ? filePath + (dirty ? " (unsaved changes)" : "") : "";
  $("#btnSave").title = filePath ? `Save to ${baseName(filePath)} (Ctrl+S)` : "Save (Ctrl+S)";
}
function setFile(path) {
  filePath = path || null; store("fsm_snn_file", filePath);
  if (filePath) store("fsm_snn_dir", dirName(filePath));
  markDirty(false);
}
function restoreFile() { filePath = load("fsm_snn_file"); dirty = !!load("fsm_snn_dirty"); renderFileName(); }

// ---- recent files: [{path, time}], most recent first
function recentList() { const l = load("fsm_snn_recent"); return Array.isArray(l) ? l : []; }
function addRecent(path) {
  const entry = { path, time: Date.now(), ...(WEB ? { spec: JSON.parse(JSON.stringify(spec)) } : {}) };
  store("fsm_snn_recent", [entry, ...recentList().filter((r) => r.path !== path)].slice(0, MAX_RECENT));
}
async function toggleRecentMenu() {
  const menu = $("#recentMenu");
  if (!menu.hidden) { menu.hidden = true; return; }
  const list = recentList();
  const when = (t) => new Date(t).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const examples = WEB ? await webExamples : [];
  menu.innerHTML = (examples.length ? `<div class="menu-head">Examples</div>` + examples.map((e) =>
      `<button class="menu-item" data-example="${esc(e.file)}"><b>${esc(e.name)}</b><span>${esc(e.file)}</span></button>`).join("") : "")
    + `<div class="menu-head">Recent files</div>` + (list.length
    ? list.map((r, k) => `<button class="menu-item" data-recent="${k}" title="${esc(r.path)}"><b>${esc(baseName(r.path))}</b><span>${WEB ? "copy kept in this browser" : esc(dirName(r.path))} · ${when(r.time)}</span></button>`).join("")
      + `<button class="menu-item clear" data-recent="clear">Clear the list</button>`
    : `<div class="menu-empty">No recent files yet</div>`);
  menu.hidden = false;
}
async function openRecent(k) {
  $("#recentMenu").hidden = true;
  if (k === "clear") { store("fsm_snn_recent", []); return toast("Recent files cleared"); }
  const r = recentList()[+k];
  if (!r) return;
  if (WEB) loadMachine(r.spec, r.path);
  else openPath(r.path);
}
async function openExample(file) {
  $("#recentMenu").hidden = true;
  try { loadMachine(await (await fetch("designs/" + encodeURIComponent(file))).json(), file, false); }
  catch (e) { toast(`Could not open ${file}`); }
}

// ---- file dialog: open a machine, save it as, or save an image of the network.
// opt: {ext, title, name, dirKey (remembered folder), write(path, overwrite) -> server reply}
const fdlg = { mode: "open", dir: null, data: null, selected: null, confirm: null, opt: {} };
async function fileDialog(mode, opt = {}) {
  $("#recentMenu").hidden = true;
  opt = { ext: ".json", dirKey: "fsm_snn_dir", write: writeFile,
    title: mode === "open" ? "Open a machine" : "Save the machine as",
    name: filePath ? baseName(filePath) : slug(spec.name) + ".json", ...opt };
  Object.assign(fdlg, { mode, selected: null, confirm: null, opt });
  $("#fileDlgTitle").textContent = opt.title;
  $("#fileDlgOk").textContent = mode === "open" ? "Open" : "Save";
  $("#fileDlgNameRow").hidden = mode === "open";
  $("#fileDlgName").value = opt.name;
  $("#fileDlgMsg").hidden = true;
  $("#fileModal").hidden = false;
  await fileDlgGo(load(opt.dirKey) || load("fsm_snn_dir") || null);
  (mode === "open" ? $("#fileDlgList") : $("#fileDlgName")).focus();
  if (mode === "save") { const n = $("#fileDlgName"); n.setSelectionRange(0, n.value.length - opt.ext.length); }
}
function closeFileDialog() { $("#fileModal").hidden = true; }
async function fileDlgGo(dir) {
  if (!fdlg.data) $("#fileDlgList").innerHTML = `<li class="file-empty">Loading…</li>`;
  const r = await api("/api/fs/list", { dir, ext: fdlg.opt.ext });
  fdlg.data = r; fdlg.dir = r.dir; fdlg.selected = null; fdlg.confirm = null;
  const sep = r.sep || "/", posix = r.dir.startsWith("/");
  let acc = "";
  const crumbs = r.dir.split(/[\\/]/).filter(Boolean).map((p, k) => {
    acc = k === 0 ? (posix ? sep + p : p + sep) : acc.replace(/[\\/]$/, "") + sep + p;
    return `<button class="crumb" data-dir="${esc(acc)}">${esc(p)}</button>`;
  });
  $("#fileDlgPath").innerHTML = `<button class="btn small icon" id="fileDlgUp" title="Parent folder" ${r.parent ? "" : "disabled"}>↑</button>` + crumbs.join(`<span class="crumb-sep">›</span>`);
  $("#fileDlgPlaces").innerHTML = r.places.map((p) => `<button class="place${p.path === r.dir ? " active" : ""}" data-dir="${esc(p.path)}">${esc(p.name)}</button>`).join("");
  const when = (t) => new Date(t * 1000).toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  $("#fileDlgList").innerHTML = !r.ok ? `<li class="file-empty">${esc((r.errors || []).join(" "))}</li>`
    : r.entries.length ? r.entries.map((e, k) => `<li class="file-row${e.dir ? " dir" : ""}" data-k="${k}" tabindex="-1">
        <span class="file-ico">${e.dir ? "📁" : "📄"}</span><span class="file-n">${esc(e.name)}</span><span class="file-t">${e.dir ? "" : when(e.mtime)}</span></li>`).join("")
    : `<li class="file-empty">No folders or ${esc(fdlg.opt.ext)} files here</li>`;
  $("#fileDlgMsg").hidden = true;
  fileDlgSync();
}
function fileDlgSync() {
  $$("#fileDlgList .file-row").forEach((el) => el.classList.toggle("sel", +el.dataset.k === fdlg.selected));
  const e = fdlg.selected !== null ? fdlg.data.entries[fdlg.selected] : null;
  $("#fileDlgOk").disabled = fdlg.mode === "open" ? !e : !$("#fileDlgName").value.trim();
  if (fdlg.mode === "open" && e && e.dir) $("#fileDlgOk").textContent = "Open folder";
  else $("#fileDlgOk").textContent = fdlg.confirm ? "Replace" : fdlg.mode === "open" ? "Open" : "Save";
}
const joinPath = (dir, name) => dir.replace(/[\\/]$/, "") + (fdlg.data.sep || "/") + name;
function fileDlgMsg(text, kind = "warn") { const m = $("#fileDlgMsg"); m.className = "msg " + kind; m.textContent = text; m.hidden = false; }
async function fileDlgOk() {
  const e = fdlg.selected !== null ? fdlg.data.entries[fdlg.selected] : null;
  if (e && e.dir && (fdlg.mode === "open" || document.activeElement !== $("#fileDlgName"))) return fileDlgGo(joinPath(fdlg.dir, e.name));
  if (fdlg.mode === "open") { if (e) { closeFileDialog(); openPath(joinPath(fdlg.dir, e.name)); } return; }
  let name = $("#fileDlgName").value.trim();
  if (!name) return;
  if (/[\\/]/.test(name)) return fileDlgMsg("Write only the file name; choose the folder above.");
  if (!name.toLowerCase().endsWith(fdlg.opt.ext)) name += fdlg.opt.ext;
  const path = joinPath(fdlg.dir, name);
  const r = await fdlg.opt.write(path, fdlg.confirm === path);
  if (r.exists) { fdlg.confirm = path; fileDlgMsg(`${name} already exists. Press Replace to overwrite it.`); fileDlgSync(); return; }
  if (r.ok) { store(fdlg.opt.dirKey, fdlg.dir); closeFileDialog(); }
}
function initFileDialog() {
  $("#fileDlgCancel").addEventListener("click", closeFileDialog);
  $("#fileModal").addEventListener("click", (e) => { if (e.target.id === "fileModal") closeFileDialog(); });
  $("#fileDlgOk").addEventListener("click", fileDlgOk);
  $("#fileDlgPath").addEventListener("click", (e) => {
    if (e.target.id === "fileDlgUp" && fdlg.data.parent) return fileDlgGo(fdlg.data.parent);
    const b = e.target.closest("[data-dir]"); if (b) fileDlgGo(b.dataset.dir);
  });
  $("#fileDlgPlaces").addEventListener("click", (e) => { const b = e.target.closest("[data-dir]"); if (b) fileDlgGo(b.dataset.dir); });
  const list = $("#fileDlgList");
  list.addEventListener("click", (e) => {
    const row = e.target.closest(".file-row"); if (!row) return;
    fdlg.selected = +row.dataset.k; fdlg.confirm = null; $("#fileDlgMsg").hidden = true;
    const en = fdlg.data.entries[fdlg.selected];
    if (fdlg.mode === "save" && !en.dir) $("#fileDlgName").value = en.name;
    fileDlgSync();
  });
  list.addEventListener("dblclick", (e) => {
    const row = e.target.closest(".file-row"); if (!row) return;
    const en = fdlg.data.entries[+row.dataset.k];
    if (en.dir) fileDlgGo(joinPath(fdlg.dir, en.name)); else fileDlgOk();
  });
  $("#fileDlgName").addEventListener("input", () => { fdlg.confirm = null; $("#fileDlgMsg").hidden = true; fdlg.selected = null; fileDlgSync(); });
  $("#fileModal").addEventListener("keydown", (e) => {
    e.stopPropagation();                // keep the shortcuts of the canvas out of the dialog
    if (e.key === "Escape") closeFileDialog();
    else if (e.key === "Enter") { e.preventDefault(); fileDlgOk(); }
    else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && fdlg.data.entries.length && document.activeElement !== $("#fileDlgName")) {
      e.preventDefault();
      const n = fdlg.data.entries.length, cur = fdlg.selected ?? -1;
      fdlg.selected = Math.max(0, Math.min(n - 1, cur + (e.key === "ArrowDown" ? 1 : -1)));
      fileDlgSync(); $(`#fileDlgList .file-row[data-k="${fdlg.selected}"]`).scrollIntoView({ block: "nearest" });
    } else if (e.key === "Backspace" && document.activeElement !== $("#fileDlgName") && fdlg.data.parent) { e.preventDefault(); fileDlgGo(fdlg.data.parent); }
  });
}

// ---- new, open, save
function newSpec() {
  pushUndo();
  const c = designVp.center();
  spec = { name: "New design", inputs: ["A", "B"], transitions: [],
    states: [{ id: "s0", name: "S0", x: snapTo(c.x), y: snapTo(c.y), initial: true }] };
  sel = { type: null, id: null }; showTab("design"); changed(); setFile(null);
  toast("New machine (Ctrl+Z to undo)");
}
async function openPath(path) {
  const r = await api("/api/fs/read", { path });
  if (!r.ok) {
    if (r.missing) store("fsm_snn_recent", recentList().filter((x) => x.path !== path));
    return toast((r.errors || ["Could not open the file"]).join(" ") + (r.missing ? " It was removed from the recent files." : ""), 4500);
  }
  let s = null; try { s = JSON.parse(r.text); } catch (e) { /* checked by loadMachine */ }
  loadMachine(s, r.path);
}
// Show a machine read from a file; remember the file unless it is a bundled example.
function loadMachine(s, path, remember = true) {
  const name = baseName(path);
  if (!s || !Array.isArray(s.states) || !Array.isArray(s.inputs)) return toast(`${name} is not a valid machine`);
  pushUndo(); spec = sanitize(s); sel = { type: null, id: null }; changed(); showTab("design"); fitDesign();
  setFile(remember ? path : null); if (remember) addRecent(path);
  toast(`${name} loaded`);
}
async function writeFile(path, overwrite) {
  const r = await api("/api/fs/write", { path, overwrite, text: JSON.stringify(spec, null, 2) });
  if (r.ok) { setFile(r.path); addRecent(r.path); toast(`Saved to ${r.name}`); }
  else if (!r.exists) toast("Could not save: " + (r.errors || ["unknown error"]).join(" "), 5000);
  return r;
}
function openSpec() {
  if (!WEB) return fileDialog("open");
  $("#recentMenu").hidden = true;
  const inp = document.createElement("input");
  inp.type = "file"; inp.accept = ".json,application/json";
  inp.addEventListener("change", async () => {
    const f = inp.files[0]; if (!f) return;
    let s = null; try { s = JSON.parse(await f.text()); } catch (e) { /* checked by loadMachine */ }
    loadMachine(s, f.name);
  });
  inp.click();
}
function saveSpec(saveAs = false) {
  if (WEB) {          // the browser downloads the file
    let name = filePath && !saveAs ? filePath : slug(spec.name) + ".json";
    if (saveAs) { name = prompt("File name", name); if (!name) return; if (!/\.json$/i.test(name)) name += ".json"; }
    download(name, JSON.stringify(spec, null, 2)); setFile(name); addRecent(name);
    return toast(`Downloaded as ${name}`);
  }
  if (saveAs || !filePath) return fileDialog("save");
  writeFile(filePath, true);
}
