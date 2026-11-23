/* Spiking FSM builder (vanilla JS, no external dependencies): tab switching and
 * start-up. Load order: util, expr, viewport, editor, network, simulate, app. */
"use strict";

let info = null;                // /api/info: simulators, default parameters, timing

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "network") loadNetwork();
  if (name === "simulate") renderManualFields();
  if (name === "simulate" && lastResult) drawRaster();
}

// ------------------------------------------------------------------ init
async function init() {
  info = await api("/api/info");
  spec = load("fsm_snn_spec");
  if (!spec || !Array.isArray(spec.states) || !Array.isArray(spec.inputs)) spec = info.default_spec;
  spec = sanitize(spec);
  $("#gapNote").textContent = info.timing.min_event_gap;
  initCanvas(); render(); renderInspector(); runCheck(); fitIfOutside();
  initNetwork();
  setSnap(load("fsm_snn_grid") !== false);
  $("#btnGrid").addEventListener("click", () => setSnap(!snap));
  renderBackends(); renderParams(); restoreSimFields(); renderManualFields();
  $("#btnUndo").disabled = true;

  $$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#btnNew").addEventListener("click", newSpec);
  const help = $("#helpCard");
  if (load("fsm_snn_help_open") === false) help.open = false;
  help.addEventListener("toggle", () => store("fsm_snn_help_open", help.open));
  $("#btnUndo").addEventListener("click", undo);
  $("#btnSave").addEventListener("click", saveSpec);
  $("#btnOpen").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", (e) => { if (e.target.files[0]) openSpec(e.target.files[0]); e.target.value = ""; });
  $("#btnAddState").addEventListener("click", () => {
    const c = designVp.center();
    addState(c.x + (Math.random() - .5) * 200, c.y + (Math.random() - .5) * 160);
  });
  $("#btnLayout").addEventListener("click", autoLayout);

  $$("#modeSeg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#pInput").addEventListener("input", () => ($("#pInputOut").textContent = fmt(+$("#pInput").value, 2)));
  $("#btnSimulate").addEventListener("click", simulate);
  $("#btnScript").addEventListener("click", downloadScript);
  $("#btnResetParams").addEventListener("click", renderParams);
  $("#showAll").addEventListener("change", drawRaster);
  $("#showExpected").addEventListener("change", drawRaster);
  $("#zoom").addEventListener("input", drawRaster);
  $("#btnScaling").addEventListener("click", runScaling);
  $("#btnScalingCsv").addEventListener("click", scalingCsv);
  window.addEventListener("resize", () => { if (lastResult) drawRaster(); if (scaleRows) drawScaleChart(); });
}
init();
