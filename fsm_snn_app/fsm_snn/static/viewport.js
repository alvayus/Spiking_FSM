/* Pan and zoom for an SVG canvas, with draw.io-like controls.
 *
 *   wheel / trackpad scroll      pan (Shift + wheel: horizontal)
 *   Ctrl/⌘ + wheel, pinch        zoom around the pointer
 *   Space + drag, middle or right drag   pan (and plain drag on empty space when panOnDrag)
 *
 * The SVG content to transform lives in `layer` (a <g>). World coordinates are the
 * coordinates inside that layer.
 */
"use strict";

function createViewport(svgEl, layer, opt = {}) {
  const vp = { k: 1, tx: 0, ty: 0, min: 0.1, max: 4, grid: opt.grid || 0, listeners: [] };
  let pan = null, space = false;

  vp.apply = () => {
    layer.setAttribute("transform", `translate(${vp.tx},${vp.ty}) scale(${vp.k})`);
    if (vp.grid) {
      const g = vp.grid * vp.k;
      svgEl.style.backgroundSize = `${g}px ${g}px`;
      svgEl.style.backgroundPosition = `${vp.tx}px ${vp.ty}px`;
    }
    vp.listeners.forEach((f) => f(vp));
  };
  vp.rect = () => svgEl.getBoundingClientRect();
  vp.toWorld = (clientX, clientY) => {
    const r = vp.rect();
    return { x: (clientX - r.left - vp.tx) / vp.k, y: (clientY - r.top - vp.ty) / vp.k };
  };
  vp.center = () => { const r = vp.rect(); return { x: (r.width / 2 - vp.tx) / vp.k, y: (r.height / 2 - vp.ty) / vp.k }; };
  vp.visible = () => { const r = vp.rect(); return { x0: -vp.tx / vp.k, y0: -vp.ty / vp.k, x1: (r.width - vp.tx) / vp.k, y1: (r.height - vp.ty) / vp.k }; };
  vp.zoomAt = (factor, sx, sy) => {          // sx, sy: screen point relative to the SVG
    const k = Math.min(vp.max, Math.max(vp.min, vp.k * factor));
    vp.tx = sx - (sx - vp.tx) * k / vp.k;
    vp.ty = sy - (sy - vp.ty) * k / vp.k;
    vp.k = k; vp.apply();
  };
  vp.zoomBy = (factor) => { const r = vp.rect(); vp.zoomAt(factor, r.width / 2, r.height / 2); };
  vp.setZoom = (k) => vp.zoomBy(k / vp.k);
  // box in world coordinates: {x0, y0, x1, y1}; padTop leaves room for the toolbar over the canvas
  vp.fit = (box, pad = 40, maxK = 1, padTop = 60) => {
    const r = vp.rect();
    if (!box || !r.width || !r.height) return;
    const w = Math.max(1, box.x1 - box.x0), h = Math.max(1, box.y1 - box.y0);
    const availH = r.height - pad - padTop;
    vp.k = Math.max(vp.min, Math.min(maxK, (r.width - 2 * pad) / w, availH / h));
    vp.tx = (r.width - w * vp.k) / 2 - box.x0 * vp.k;
    vp.ty = padTop + (availH - h * vp.k) / 2 - box.y0 * vp.k;
    vp.apply();
  };
  vp.panning = () => !!pan;

  svgEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    if (e.ctrlKey || e.metaKey) {
      const r = vp.rect(), d = e.deltaY * unit;
      vp.zoomAt(Math.exp(-d * (Math.abs(d) < 40 ? 0.01 : 0.0018)), e.clientX - r.left, e.clientY - r.top);
    } else {
      vp.tx -= (e.shiftKey ? e.deltaY : e.deltaX) * unit;
      vp.ty -= (e.shiftKey ? 0 : e.deltaY) * unit;
      vp.apply();
    }
  }, { passive: false });

  // capture phase, so a pan gesture never reaches the editor's own handlers
  svgEl.addEventListener("pointerdown", (e) => {
    const bg = opt.panOnDrag && e.button === 0 && opt.isBackground && opt.isBackground(e);
    if (!(e.button === 1 || e.button === 2 || (e.button === 0 && space) || bg)) return;
    e.preventDefault(); e.stopPropagation();
    pan = { x: e.clientX, y: e.clientY, tx: vp.tx, ty: vp.ty, moved: false, bg };
    svgEl.setPointerCapture(e.pointerId);
    svgEl.classList.add("panning");
  }, true);
  svgEl.addEventListener("pointermove", (e) => {
    if (!pan) return;
    e.stopPropagation();
    const dx = e.clientX - pan.x, dy = e.clientY - pan.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) pan.moved = true;
    vp.tx = pan.tx + dx; vp.ty = pan.ty + dy; vp.apply();
  }, true);
  const end = (e) => {
    if (!pan) return;
    e.stopPropagation();
    const p = pan; pan = null; svgEl.classList.remove("panning");
    if (p.bg && !p.moved && opt.onBackgroundClick) opt.onBackgroundClick(e);
  };
  svgEl.addEventListener("pointerup", end, true);
  svgEl.addEventListener("pointercancel", end, true);
  svgEl.addEventListener("contextmenu", (e) => e.preventDefault());

  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
  const visible = () => svgEl.getClientRects().length > 0;
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !typing() && visible()) {
      e.preventDefault();
      if (!space) { space = true; svgEl.classList.add("can-pan"); }
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space") { space = false; svgEl.classList.remove("can-pan"); }
  });
  return vp;
}

/* Zoom widget: − 100 % + Fit. Clicking the percentage resets to 100 %. */
function mountZoomBar(el, vp, onFit) {
  el.innerHTML = `<button data-z="out" title="Zoom out (Ctrl + −)">−</button>
    <button data-z="reset" class="pct" title="Reset to 100 % (Ctrl + 0)">100 %</button>
    <button data-z="in" title="Zoom in (Ctrl + +)">+</button>
    <button data-z="fit" title="Fit to window (Ctrl + Shift + H)">Fit</button>`;
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.z === "in") vp.zoomBy(1.2);
    else if (b.dataset.z === "out") vp.zoomBy(1 / 1.2);
    else if (b.dataset.z === "reset") vp.setZoom(1);
    else onFit();
  });
  const pct = el.querySelector(".pct");
  vp.listeners.push(() => { pct.textContent = Math.round(vp.k * 100) + " %"; });
}
