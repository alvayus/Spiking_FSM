/* Spiking FSM builder (vanilla JS, no external dependencies): tab switching and
 * start-up. Load order: util, expr, viewport, editor, network, simulate, app. */
"use strict";

let info = null;                // /api/info: simulators, default parameters, timing

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "network") loadNetwork();
  if (name === "simulate") { renderManualFields(); updateGapMs(); }
  if (name === "simulate" && lastResult) drawRaster();
}

// ------------------------------------------------------------------ init
async function init() {
  info = await api("/api/info");
  spec = load("fsm_snn_spec");
  if (!spec || !Array.isArray(spec.states) || !Array.isArray(spec.inputs)) spec = info.default_spec;
  spec = sanitize(spec);
  initCanvas(); render(); renderInspector(); runCheck(); fitIfOutside();
  restoreFile();
  initNetwork();
  setSnap(load("fsm_snn_grid") !== false);
  $("#btnGrid").addEventListener("click", () => setSnap(!snap));
  renderBackends(); restoreSimFields(); renderManualFields();
  $("#btnUndo").disabled = true;

  $$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#btnNew").addEventListener("click", newSpec);
  const help = $("#helpCard");
  if (load("fsm_snn_help_open") === false) help.open = false;
  help.addEventListener("toggle", () => store("fsm_snn_help_open", help.open));
  const netHelp = $("#netHelpCard");
  if (load("fsm_snn_net_help_open") === false) netHelp.open = false;
  netHelp.addEventListener("toggle", () => store("fsm_snn_net_help_open", netHelp.open));
  $("#btnUndo").addEventListener("click", undo);
  $("#btnSave").addEventListener("click", () => saveSpec(false));
  $("#btnSaveAs").addEventListener("click", () => saveSpec(true));
  $("#btnOpen").addEventListener("click", openSpec);
  $("#btnRecent").addEventListener("click", toggleRecentMenu);
  $("#recentMenu").addEventListener("click", (e) => {
    const b = e.target.closest("[data-recent], [data-example]");
    if (b) b.dataset.example ? openExample(b.dataset.example) : openRecent(b.dataset.recent);
  });
  if (info.web) {
    $("#webBadge").hidden = false;
    $("#btnOpen").title = "Open a machine from this computer (Ctrl+O)";
    $("#btnSave").title = $("#btnSaveAs").title = "Download the machine as a JSON file";
  }
  document.addEventListener("pointerdown", (e) => { if (!e.target.closest(".split")) $("#recentMenu").hidden = true; });
  initFileDialog();
  $("#btnAddState").addEventListener("click", () => {
    const c = designVp.center();
    addState(c.x + (Math.random() - .5) * 200, c.y + (Math.random() - .5) * 160);
  });
  $("#btnLayout").addEventListener("click", autoLayout);

  $$("#modeSeg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#pInput").addEventListener("input", () => ($("#pInputOut").textContent = fmt(+$("#pInput").value, 2)));
  $("#btnSimulate").addEventListener("click", simulate);
  $("#btnScript").addEventListener("click", () => showCode("script"));
  $("#showAll").addEventListener("change", drawRaster);
  $("#showExpected").addEventListener("change", drawRaster);
  $("#zoom").addEventListener("input", drawRaster);
  $("#btnSimSave").addEventListener("click", () => ($("#simSaveMenu").hidden = !$("#simSaveMenu").hidden));
  $("#simSaveMenu").addEventListener("click", (e) => { const b = e.target.closest("[data-img]"); if (b) saveSimImage(b.dataset.img); });
  document.addEventListener("pointerdown", (e) => { if (!e.target.closest("#simSaveWrap")) $("#simSaveMenu").hidden = true; });
  // the two plots share the time axis: scroll them together
  const sc = [$("#rasterScroll"), $("#rasterExpScroll")];
  sc.forEach((a, i) => a.addEventListener("scroll", () => { const b = sc[1 - i]; if (b.scrollLeft !== a.scrollLeft) b.scrollLeft = a.scrollLeft; }));
  $("#manualList").addEventListener("input", checkTyped);
  ["pointerdown", "keydown"].forEach((ev) => $("#dt").addEventListener(ev, () => (dtBefore = parseFloat($("#dt").value))));
  $("#dt").addEventListener("input", (e) => { dtArrow(e); checkTyped(); updateGapMs(); updateTsfNote(); });
  $("#tsf").addEventListener("input", updateTsfNote);
  ["#minGap", "#maxGap"].forEach((id) => $(id).addEventListener("input", updateGapMs));
  updateGapMs(); updateTsfNote();
  window.addEventListener("resize", () => { if (lastResult) drawRaster(); });
}
init();
