/* Web version (GitHub Pages): no local server. The Python package runs in the
 * browser with Pyodide and answers the same API calls (fsm_snn/web.py). Only the
 * static build (build_web.py) loads this file, before util.js. */
"use strict";

window.FSM_WEB = true;
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

const pyReady = (async () => {
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  const files = await (await fetch("py/manifest.json")).json();
  py.FS.mkdirTree("/app/fsm_snn");
  await Promise.all(files.map(async (f) => py.FS.writeFile("/app/fsm_snn/" + f, await (await fetch("py/fsm_snn/" + f)).text())));
  py.runPython("import sys; sys.path.insert(0, '/app')");
  return py.pyimport("fsm_snn.web").handle;
})();

// Python takes a few seconds to load (longer on the first visit, while it downloads).
let pyLoaded = false;
document.addEventListener("DOMContentLoaded", () => {
  if (!pyLoaded) busy(true, "Loading Python in the browser…");
});
pyReady.then(() => { pyLoaded = true; busy(false); }, (err) => {
  console.error(err);
  busy(true, "Could not load Python in the browser. Check the internet connection and reload the page.");
  $(".spinner").hidden = true;
});

// Called by api() in util.js. The simulation runs on the page's thread, so let the
// busy overlay paint before starting it.
async function webApi(path, body) {
  const handle = await pyReady;
  await new Promise((r) => setTimeout(r, 30));
  return JSON.parse(handle(body === undefined ? "GET" : "POST", path, JSON.stringify(body ?? {})));
}

// Machines bundled with the site (designs/), offered in the Recent menu.
const webExamples = fetch("designs/manifest.json").then((r) => r.json()).catch(() => []);
