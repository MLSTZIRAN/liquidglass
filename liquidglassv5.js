(function () {
  "use strict";
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const numAttr = (el, n, f) => { const r = el.getAttribute(n); if (r === null) return f; const v = parseFloat(r); return Number.isFinite(v) ? v : f; };
  function imgToURL(img) { const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; c.getContext("2d").putImageData(img, 0, 0); return c.toDataURL(); }

  const SurfaceEquations = {
    convex_circle: x => Math.sqrt(1 - Math.pow(1 - x, 2)),
    convex_squircle: x => Math.pow(1 - Math.pow(1 - x, 4), 1 / 4),
    concave: x => 1 - Math.sqrt(1 - Math.pow(x, 2)),
    lip: x => {
      const convex = Math.pow(1 - Math.pow(1 - Math.min(x * 2, 1), 4), 1 / 4);
      const concave = 1 - Math.sqrt(1 - Math.pow(1 - x, 2)) + 0.1;
      const ss = 6 * Math.pow(x, 5) - 15 * Math.pow(x, 4) + 10 * Math.pow(x, 3);
      return convex * (1 - ss) + concave * ss;
    }
  };
  class Spring {
    constructor(v, k = 300, d = 20) { this.value = v; this.target = v; this.velocity = 0; this.stiffness = k; this.damping = d; }
    setTarget(t) { this.target = t; }
    update(dt) { this.velocity += ((this.target - this.value) * this.stiffness - this.velocity * this.damping) * dt; this.value += this.velocity * dt; return this.value; }
    isSettled() { return Math.abs(this.target - this.value) < 0.001 && Math.abs(this.velocity) < 0.001; }
  }

  function gaussianBlurA(imageData, radius = 1.5) {
    const w = imageData.width, h = imageData.height, d = imageData.data;
    const sigma = radius / 3, size = Math.ceil(radius * 3) * 2 + 1, half = Math.floor(size / 2);
    const kernel = new Float32Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) { const x = i - half; kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma)); sum += kernel[i]; }
    for (let i = 0; i < size; i++) kernel[i] /= sum;
    const temp = new Uint8ClampedArray(d);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < size; k++) { const sx = Math.min(w - 1, Math.max(0, x + k - half)), idx = (y * w + sx) * 4; r += d[idx] * kernel[k]; g += d[idx + 1] * kernel[k]; b += d[idx + 2] * kernel[k]; a += d[idx + 3] * kernel[k]; }
      const idx = (y * w + x) * 4; temp[idx] = Math.round(r); temp[idx + 1] = Math.round(g); temp[idx + 2] = Math.round(b); temp[idx + 3] = Math.round(a);
    }
    const out = new Uint8ClampedArray(d);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < size; k++) { const sy = Math.min(h - 1, Math.max(0, y + k - half)), idx = (sy * w + x) * 4; r += temp[idx] * kernel[k]; g += temp[idx + 1] * kernel[k]; b += temp[idx + 2] * kernel[k]; a += temp[idx + 3] * kernel[k]; }
      const idx = (y * w + x) * 4; out[idx] = Math.round(r); out[idx + 1] = Math.round(g); out[idx + 2] = Math.round(b); out[idx + 3] = Math.round(a);
    }
    for (let i = 0; i < d.length; i++) d[i] = out[i];
    return imageData;
  }

  function calcDisp1D_A(gt, bw, fn, ri, samples = 128) {
    const eta = 1 / ri;
    function refr(nx, ny) { const dot = ny, k = 1 - eta * eta * (1 - dot * dot); if (k < 0) return null; const ks = Math.sqrt(k); return [-(eta * dot + ks) * nx, eta - (eta * dot + ks) * ny]; }
    const res = [];
    for (let i = 0; i < samples; i++) {
      const x = i / samples, y = fn(x), dx = x < 1 ? 0.0001 : -0.0001;
      const y2 = fn(Math.max(0, Math.min(1, x + dx)));
      const der = (y2 - y) / dx, mag = Math.sqrt(der * der + 1);
      const n = [-der / mag, -1 / mag], rf = refr(n[0], n[1]);
      res.push(rf ? rf[0] * ((y * bw + gt) / rf[1]) : 0);
    }
    return res;
  }

  function calcDisp2D_A(cw, ch, ow, oh, r, bw, md, pre, dpr) {
    dpr = dpr || 1;
    const CW = Math.round(cw * dpr), CH = Math.round(ch * dpr), OW = Math.round(ow * dpr), OH = Math.round(oh * dpr), R = r * dpr, BW = bw * dpr;
    const img = new ImageData(CW, CH);
    for (let i = 0; i < img.data.length; i += 4) { img.data[i] = 128; img.data[i + 1] = 128; img.data[i + 2] = 0; img.data[i + 3] = 255; }
    const R2 = R * R, Rp1Sq = (R + 1) * (R + 1), RmBSq = Math.max(0, (R - BW) * (R - BW));
    const wB = OW - R * 2, hB = OH - R * 2, oX = (CW - OW) / 2, oY = (CH - OH) / 2;
    for (let y1 = 0; y1 < OH; y1++) for (let x1 = 0; x1 < OW; x1++) {
      const idx = ((oY + y1) * CW + oX + x1) * 4;
      const isL = x1 < R, isR = x1 >= OW - R, isT = y1 < R, isB = y1 >= OH - R;
      const x = isL ? x1 - R : isR ? x1 - R - wB : 0, y = isT ? y1 - R : isB ? y1 - R - hB : 0;
      const d2 = x * x + y * y;
      if (d2 <= Rp1Sq && d2 >= RmBSq) {
        const op = d2 < R2 ? 1 : 1 - (Math.sqrt(d2) - R) / (Math.sqrt(Rp1Sq) - R);
        const dfc = Math.sqrt(d2), dfs = R - dfc, cos = dfc > 0 ? x / dfc : 0, sin = dfc > 0 ? y / dfc : 0;
        const br = Math.max(0, Math.min(1, dfs / BW)), bi = Math.floor(br * pre.length);
        const dist = pre[Math.max(0, Math.min(bi, pre.length - 1))] || 0;
        const dX = md > 0 ? (-cos * dist) / md : 0, dY = md > 0 ? (-sin * dist) / md : 0;
        img.data[idx] = Math.max(0, Math.min(255, 128 + dX * 127 * op));
        img.data[idx + 1] = Math.max(0, Math.min(255, 128 + dY * 127 * op));
        img.data[idx + 2] = 0; img.data[idx + 3] = Math.round(255 * op);
      }
    }
    return img;
  }

  function calcSpec_A(ow, oh, r, bw, dpr, ang) {
    ang = ang || Math.PI / 3; dpr = dpr || 1;
    const W = Math.round(ow * dpr), H = Math.round(oh * dpr), R = r * dpr, ST = 3.0 * dpr;
    const img = new ImageData(W, H);
    for (let i = 0; i < img.data.length; i += 4) { img.data[i] = 255; img.data[i + 1] = 255; img.data[i + 2] = 255; img.data[i + 3] = 0; }
    const sv = [Math.cos(ang), Math.sin(ang)];
    const R2 = R * R, RpO = R + 2.5 * dpr, RpOSq = RpO * RpO, RmS = Math.max(0, R - ST), RmSSq = RmS * RmS;
    const wB = W - R * 2, hB = H - R * 2;
    for (let y1 = 0; y1 < H; y1++) for (let x1 = 0; x1 < W; x1++) {
      const idx = (y1 * W + x1) * 4;
      const isL = x1 < R, isR = x1 >= W - R, isT = y1 < R, isB = y1 >= H - R;
      const x = isL ? x1 - R : isR ? x1 - R - wB : 0, y = isT ? y1 - R : isB ? y1 - R - hB : 0;
      const d2 = x * x + y * y;
      if (d2 <= RpOSq && d2 >= RmSSq) {
        const dfc = Math.sqrt(d2), dfs = R - dfc;
        let eo = 1;
        if (d2 > R2) { const t = (dfc - R) / (RpO - R); eo = 1 - t * t * (3 - 2 * t); }
        let io = 1;
        if (dfc < RmS + 0.5 * dpr) { const t = (RmS + 0.5 * dpr - dfc) / (0.5 * dpr); io = 1 - t * t * (3 - 2 * t); }
        const op = eo * io; if (op <= 0) continue;
        const cos = dfc > 0 ? x / dfc : 0, sin = dfc > 0 ? -y / dfc : 0;
        const dot = Math.abs(cos * sv[0] + sin * sv[1]);
        const er = Math.max(0, Math.min(1, dfs / ST)), sf = (1 - Math.cos(er * Math.PI)) / 2;
        const coef = dot * sf, sc = Math.pow(coef, 0.7), color = Math.min(255, 255 * sc);
        const finalOp = Math.min(255, color * sc * op);
        img.data[idx] = color; img.data[idx + 1] = color; img.data[idx + 2] = color; img.data[idx + 3] = Math.round(finalOp);
      }
    }
    gaussianBlurA(img, 0.5 * dpr);
    return img;
  }

  const _texCacheA = new Map();
  function getTexturesA(w, h, r, bw, surface, ri, gt, dpr, cap) {
    dpr = dpr || 1; cap = cap == null ? 512 : cap;
    let effDpr = cap > 0 ? Math.min(dpr, cap / Math.max(w, h)) : dpr;
    const px = (w * effDpr) * (h * effDpr), absMax = 1024 * 1024;
    if (px > absMax) effDpr *= Math.sqrt(absMax / px);
    const key = `A|${w}|${h}|${r}|${bw}|${surface}|${ri}|${gt}|${effDpr.toFixed(3)}`;
    let e = _texCacheA.get(key); if (e) return e;
    const fn = SurfaceEquations[surface] || SurfaceEquations.convex_squircle;
    const pre = calcDisp1D_A(gt, bw, fn, ri);
    const maxDisp = Math.max(...pre.map(Math.abs));
    const disp = calcDisp2D_A(w, h, w, h, r, bw, maxDisp || 1, pre, effDpr);
    const spec = calcSpec_A(w, h, r, bw, effDpr, Math.PI / 3);
    e = { dispURL: imgToURL(disp), specURL: imgToURL(spec), maxDisp };
    _texCacheA.set(key, e);
    if (_texCacheA.size > 32) _texCacheA.delete(_texCacheA.keys().next().value);
    return e;
  }

  function gaussianBlur(img, r = 1.5) {
    if (r < 0.05) return img;
    const w = img.width, h = img.height, d = img.data, s = r / 3;
    const size = Math.ceil(r * 3) * 2 + 1, half = Math.floor(size / 2), kern = new Float32Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) { const x = i - half; kern[i] = Math.exp(-(x * x) / (2 * s * s)); sum += kern[i]; }
    for (let i = 0; i < size; i++) kern[i] /= sum;
    const tmp = new Uint8ClampedArray(d);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let rr = 0, gg = 0, bb = 0, aa = 0;
      for (let k = 0; k < size; k++) { const sx = Math.min(w - 1, Math.max(0, x + k - half)), idx = (y * w + sx) * 4; rr += d[idx] * kern[k]; gg += d[idx + 1] * kern[k]; bb += d[idx + 2] * kern[k]; aa += d[idx + 3] * kern[k]; }
      const idx = (y * w + x) * 4; tmp[idx] = rr; tmp[idx + 1] = gg; tmp[idx + 2] = bb; tmp[idx + 3] = aa;
    }
    const out = new Uint8ClampedArray(d);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let rr = 0, gg = 0, bb = 0, aa = 0;
      for (let k = 0; k < size; k++) { const sy = Math.min(h - 1, Math.max(0, y + k - half)), idx = (sy * w + x) * 4; rr += tmp[idx] * kern[k]; gg += tmp[idx + 1] * kern[k]; bb += tmp[idx + 2] * kern[k]; aa += tmp[idx + 3] * kern[k]; }
      const idx = (y * w + x) * 4; out[idx] = rr; out[idx + 1] = gg; out[idx + 2] = bb; out[idx + 3] = aa;
    }
    for (let i = 0; i < d.length; i++) d[i] = out[i];
    return img;
  }

  function calcDisp1D(gt, bw, fn, ri, samples = 128) {
    const eta = 1 / ri, res = [];
    function refr(nx, ny) { const dot = ny, k = 1 - eta * eta * (1 - dot * dot); if (k < 0) return null; const ks = Math.sqrt(k); return [-(eta * dot + ks) * nx, eta - (eta * dot + ks) * ny]; }
    const gtEff = 80 * (1 - Math.exp(-gt / 120));
    for (let i = 0; i <= samples; i++) {
      const x = i / samples, y = fn(x), dx = 0.0001, y2 = fn(Math.min(1, x + dx));
      const der = (y2 - y) / dx, mag = Math.sqrt(der * der + 1);
      const n = [-der / mag, -1 / mag], rf = refr(n[0], n[1]);
      res.push(rf ? rf[0] * ((y * bw + gtEff) / rf[1]) : 0);
    }
    return res;
  }

  function calcDisp2D(cw, ch, ow, oh, r, bw, md, pre, dpr) {
    dpr = dpr || 1;
    const CW = Math.round(cw * dpr), CH = Math.round(ch * dpr), OW = Math.round(ow * dpr), OH = Math.round(oh * dpr), R = r * dpr, BW = bw * dpr;
    const img = new ImageData(CW, CH);
    for (let i = 0; i < img.data.length; i += 4) { img.data[i] = 128; img.data[i + 1] = 128; img.data[i + 2] = 0; img.data[i + 3] = 255; }
    const oX = (CW - OW) / 2, oY = (CH - OH) / 2, halfW = OW / 2, halfH = OH / 2;
    const innerW = Math.max(0, halfW - R), innerH = Math.max(0, halfH - R), N = pre.length;
    const invMd = md > 0 ? 1 / md : 0, inv2dpr = 1 / (2 * dpr);
    for (let y1 = 0; y1 < OH; y1++) {
      const py = y1 + 0.5 - halfH, qy = Math.abs(py) - innerH, qyPos = qy > 0 ? qy : 0, rowBase = (oY + y1) * CW;
      for (let x1 = 0; x1 < OW; x1++) {
        const px = x1 + 0.5 - halfW, qx = Math.abs(px) - innerW, qxPos = qx > 0 ? qx : 0;
        const outside = Math.sqrt(qxPos * qxPos + qyPos * qyPos);
        const mq = qx > qy ? qx : qy, sdf = (mq < 0 ? mq : 0) + outside - R, dfs = -sdf;
        if (dfs < 0 || dfs > BW) continue;
        let nx, ny;
        if (qx > 0 && qy > 0) { const cx = (px < 0 ? -1 : 1) * innerW, cy = (py < 0 ? -1 : 1) * innerH, dx = px - cx, dy = py - cy, len = Math.sqrt(dx * dx + dy * dy) || 1; nx = dx / len; ny = dy / len; }
        else if (qx > qy) { nx = px < 0 ? -1 : 1; ny = 0; } else { nx = 0; ny = py < 0 ? -1 : 1; }
        const br = dfs / BW < 1 ? dfs / BW : 1, b2 = br * br, fade = 1 - b2 * b2;
        let bi = (br * (N - 1) + 0.5) | 0; if (bi < 0) bi = 0; else if (bi > N - 1) bi = N - 1;
        const dist = (pre[bi] || 0) * fade, dX = -nx * dist * invMd, dY = -ny * dist * invMd;
        const op = dfs < 2 * dpr ? dfs * inv2dpr : 1, idx = (rowBase + oX + x1) * 4;
        let v = 128 + dX * 127 * op; img.data[idx] = v < 0 ? 0 : v > 255 ? 255 : v;
        v = 128 + dY * 127 * op; img.data[idx + 1] = v < 0 ? 0 : v > 255 ? 255 : v;
        img.data[idx + 2] = 0; img.data[idx + 3] = (255 * op) | 0;
      }
    }
    return img;
  }

  function calcSpec(ow, oh, r, bw, ang, dpr) {
    ang = ang || Math.PI / 3; dpr = dpr || 1;
    const W = Math.round(ow * dpr), H = Math.round(oh * dpr), R = r * dpr, ST = 3.0 * dpr, OUT = 2.0 * dpr;
    const img = new ImageData(W, H);
    for (let i = 0; i < img.data.length; i += 4) { img.data[i] = 255; img.data[i + 1] = 255; img.data[i + 2] = 255; img.data[i + 3] = 0; }
    const svx = Math.cos(ang), svy = Math.sin(ang);
    const halfW = W / 2, halfH = H / 2, innerW = Math.max(0, halfW - R), innerH = Math.max(0, halfH - R);
    const invST = 1 / ST, invOUT = 1 / OUT;
    for (let y1 = 0; y1 < H; y1++) {
      const py = y1 + 0.5 - halfH, qy = Math.abs(py) - innerH, qyPos = qy > 0 ? qy : 0, rowBase = y1 * W * 4;
      for (let x1 = 0; x1 < W; x1++) {
        const px = x1 + 0.5 - halfW, qx = Math.abs(px) - innerW, qxPos = qx > 0 ? qx : 0;
        const outside = Math.sqrt(qxPos * qxPos + qyPos * qyPos);
        const mq = qx > qy ? qx : qy, sdf = (mq < 0 ? mq : 0) + outside - R;
        const distFromEdge = sdf < 0 ? -sdf : sdf;
        const invMax = sdf < 0 ? invST : invOUT, er = distFromEdge * invMax;
        if (er >= 1) continue;
        const t = 1 - er, sf = t * t; if (sf < 0.003) continue;
        let nx, ny;
        if (qx > 0 && qy > 0) { const cx = (px < 0 ? -1 : 1) * innerW, cy = (py < 0 ? -1 : 1) * innerH, dx = px - cx, dy = py - cy, len = Math.sqrt(dx * dx + dy * dy) || 1; nx = dx / len; ny = dy / len; }
        else if (qx > qy) { nx = px < 0 ? -1 : 1; ny = 0; } else { nx = 0; ny = py < 0 ? -1 : 1; }
        const dot = Math.abs(nx * svx - ny * svy), coef = dot * sf, sc = Math.pow(coef, 0.7);
        const color = 255 * sc, finalOp = color * sc, idx = rowBase + x1 * 4;
        img.data[idx] = color; img.data[idx + 1] = color; img.data[idx + 2] = color; img.data[idx + 3] = finalOp < 255 ? finalOp : 255;
      }
    }
    gaussianBlur(img, 1.2 * dpr);
    return img;
  }

  const _texCache = new Map();
  function getTextures(w, h, r, bw, surface, ri, gt, dpr, cap) {
    dpr = dpr || 1; cap = cap == null ? 512 : cap;
    let effDpr = cap > 0 ? Math.min(dpr, cap / Math.max(w, h)) : dpr;
    const px = (w * effDpr) * (h * effDpr), absMax = 1024 * 1024;
    if (px > absMax) effDpr *= Math.sqrt(absMax / px);
    const key = `B|${w}|${h}|${r}|${bw}|${surface}|${ri}|${gt}|${effDpr.toFixed(3)}`;
    let e = _texCache.get(key); if (e) return e;
    const fn = SurfaceEquations[surface] || SurfaceEquations.convex_squircle;
    const pre = calcDisp1D(gt, bw, fn, ri);
    const maxDisp = Math.max(...pre.map(Math.abs));
    const disp = calcDisp2D(w, h, w, h, r, bw, maxDisp || 1, pre, effDpr);
    const spec = calcSpec(w, h, r, bw, Math.PI / 3, effDpr);
    e = { dispURL: imgToURL(disp), specURL: imgToURL(spec), maxDisp };
    _texCache.set(key, e);
    if (_texCache.size > 32) _texCache.delete(_texCache.keys().next().value);
    return e;
  }

  const BG_CSS = `background-image:linear-gradient(to right,var(--grid) 1px,transparent 1px),linear-gradient(to bottom,var(--grid) 1px,transparent 1px),radial-gradient(120% 100% at 10% 0%,var(--bg-primary),var(--bg-secondary));background-size:24px 24px,24px 24px,100% 100%;background-position:12px 12px,12px 12px,0 0;`;
  const FILTER_ATTR = `x="-2%" y="-2%" width="104%" height="104%" color-interpolation-filters="sRGB"`;

  class LiquidSlider extends HTMLElement {
    static get observedAttributes() { return ['value', 'bezel-width', 'refraction', 'dispersion', 'specular-opacity', 'specular-saturation', 'blur', 'force-active', 'dpr', 'texture-cap']; }
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.uid = 'ls-' + Math.random().toString(36).slice(2, 8);
      this.shadowRoot.innerHTML = `<style>:host{display:block;position:relative;height:200px;--bg-primary:#f8fafc;--bg-secondary:#e7eeef;--grid:rgba(0,0,0,0.03);}@media (prefers-color-scheme:dark){:host{--bg-primary:#1b1b22;--bg-secondary:#0f0f14;--grid:rgba(255,255,255,0.03);}}.stage{position:relative;width:100%;height:100%;display:flex;justify-content:center;align-items:center;${BG_CSS}}.slider-container{position:relative;width:330px;height:60px}.track{position:absolute;width:330px;height:14px;left:0;top:23px;background:#89898f;border-radius:7px;cursor:pointer;overflow:hidden}.fill{height:14px;width:10%;border-radius:6px;background:#0377f7}.thumb{position:absolute;width:90px;height:60px;top:0;left:0;border-radius:30px;transform-origin:center center;transform:scale(0.6);background:#fff;box-shadow:0 3px 14px rgba(0,0,0,.1);cursor:grab;touch-action:none;user-select:none;overflow:hidden;will-change:transform,left,background-color,box-shadow}.thumb:active{cursor:grabbing}.clone{position:absolute;inset:0;opacity:0;pointer-events:none;z-index:1}.clone-inner{position:absolute;top:0;left:0;pointer-events:none;will-change:transform;${BG_CSS}}.clone-inner::before{content:'';position:absolute;left:var(--track-left);top:var(--track-top);width:330px;height:14px;border-radius:7px;background:#89898f}.clone-inner::after{content:'';position:absolute;left:var(--track-left);top:var(--track-top);width:calc(330px * var(--fill) / 100);height:14px;border-radius:6px;background:#0377f7}.thumb-inner{position:absolute;inset:0;border-radius:inherit;z-index:3;pointer-events:none}svg{position:absolute;width:0;height:0}</style><div class="stage"><div class="slider-container"><div class="track"><div class="fill"></div></div><div class="thumb"><div class="thumb-inner"></div><div class="clone"><div class="clone-inner"></div></div></div></div></div><svg><defs><filter id="${this.uid}-f" ${FILTER_ATTR}><feGaussianBlur in="SourceGraphic" stdDeviation="0" result="blurred" id="${this.uid}-blur"/><feImage href="" x="0" y="0" width="90" height="60" result="dm" id="${this.uid}-disp"/><feImage href="" x="0" y="0" width="90" height="60" result="sm" id="${this.uid}-spec"/><feGaussianBlur in="dm" stdDeviation="0.2" result="dms" id="${this.uid}-dms"/><feColorMatrix in="blurred" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0" result="r"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0" result="g"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0" result="b"/><feDisplacementMap in="r" in2="dms" scale="0.85" xChannelSelector="R" yChannelSelector="G" result="dr" id="${this.uid}-dr"/><feDisplacementMap in="g" in2="dms" scale="1" xChannelSelector="R" yChannelSelector="G" result="dg" id="${this.uid}-dg"/><feDisplacementMap in="b" in2="dms" scale="1.15" xChannelSelector="R" yChannelSelector="G" result="db" id="${this.uid}-db"/><feBlend in="dr" in2="dg" mode="lighten" result="rg"/><feBlend in="rg" in2="db" mode="lighten" result="displaced"/><feColorMatrix in="displaced" type="saturate" values="1" result="sat" id="${this.uid}-sat"/><feGaussianBlur in="sm" stdDeviation="0.3" result="sms"/><feComponentTransfer in="sms" result="sa"><feFuncA type="linear" slope="0.4" id="${this.uid}-sa"/></feComponentTransfer><feComposite in="sat" in2="sa" operator="arithmetic" k1="0" k2="1" k3="1" k4="0"/></filter></defs></svg>`;
      this.trackEl = this.shadowRoot.querySelector('.track');
      this.fillEl = this.shadowRoot.querySelector('.fill');
      this.thumbEl = this.shadowRoot.querySelector('.thumb');
      this.cloneEl = this.shadowRoot.querySelector('.clone');
      this.cloneInnerEl = this.shadowRoot.querySelector('.clone-inner');
      this._dispEl = this.shadowRoot.getElementById(`${this.uid}-disp`);
      this._specEl = this.shadowRoot.getElementById(`${this.uid}-spec`);
      this._saEl = this.shadowRoot.getElementById(`${this.uid}-sa`);
      this._blurEl = this.shadowRoot.getElementById(`${this.uid}-blur`);
      this._dmsEl = this.shadowRoot.getElementById(`${this.uid}-dms`);
      this._satEl = this.shadowRoot.getElementById(`${this.uid}-sat`);
      this._drEl = this.shadowRoot.getElementById(`${this.uid}-dr`);
      this._dgEl = this.shadowRoot.getElementById(`${this.uid}-dg`);
      this._dbEl = this.shadowRoot.getElementById(`${this.uid}-db`);
      this._textureRaf = null; this._hqTimer = null; this._interactiveQuality = 1;
    }
    connectedCallback() {
      this.config = { thumbWidth: 90, thumbHeight: 60, thumbRadius: 30, trackWidth: 330, trackHeight: 14, bezelWidth: 25, glassThickness: 80, refractiveIndex: 1.45, dpr: 1, textureCap: 512, SCALE_REST: 0.6, SCALE_DRAG: 1 };
      this.state = { value: clamp(numAttr(this, 'value', 10), 0, 100), pointerDown: false, forceActive: false, specularOpacity: 0.4, specularSaturation: 1, refractionBase: 1, dispersion: 0.15, blur: 0, maximumDisplacement: 0, velocityX: 0, lastX: 0, lastTime: 0 };
      this.readAttrs();
      this.springs = { scale: new Spring(this.config.SCALE_REST, 500, 18), backgroundOpacity: new Spring(1, 2000, 80), scaleRatio: new Spring(0.4, 100, 10), scaleX: new Spring(1, 400, 30), scaleY: new Spring(1, 400, 30), shadowOffsetX: new Spring(0, 400, 30), shadowOffsetY: new Spring(3, 400, 30), shadowBlur: new Spring(10, 400, 30), shadowAlpha: new Spring(0.12, 300, 25) };
      this.raf = null;
      this.cloneEl.style.filter = `url(#${this.uid}-f)`;
      this.updateTextures();
      this.abort = new AbortController(); const sig = this.abort.signal;
      this.thumbEl.addEventListener('mousedown', e => this.onDown(e), { signal: sig });
      this.thumbEl.addEventListener('touchstart', e => this.onDown(e), { passive: false, signal: sig });
      this.trackEl.addEventListener('mousedown', e => { this.onDown(e); this.onMove(e); }, { signal: sig });
      window.addEventListener('mousemove', e => this.onMove(e), { signal: sig });
      window.addEventListener('touchmove', e => this.onMove(e), { passive: false, signal: sig });
      window.addEventListener('mouseup', () => this.onUp(), { signal: sig });
      window.addEventListener('touchend', () => this.onUp(), { signal: sig });
      this.ro = new ResizeObserver(() => this.updateUI()); this.ro.observe(this);
      requestAnimationFrame(() => { this.updateUI(); this.startAnim(); });
    }
    disconnectedCallback() { this.abort?.abort(); if (this.raf) cancelAnimationFrame(this.raf); if (this._textureRaf) cancelAnimationFrame(this._textureRaf); clearTimeout(this._hqTimer); this.ro?.disconnect(); }
    readAttrs() {
      const c = this.config, s = this.state;
      c.bezelWidth = clamp(numAttr(this, 'bezel-width', 25), 2, 30);
      c.dpr = clamp(numAttr(this, 'dpr', 1), 0.5, 3);
      c.textureCap = Math.max(0, numAttr(this, 'texture-cap', 512));
      s.refractionBase = clamp(numAttr(this, 'refraction', 1), 0.1, 3);
      s.dispersion = clamp(numAttr(this, 'dispersion', 0.15), 0, 0.3);
      s.specularOpacity = clamp(numAttr(this, 'specular-opacity', 0.4), 0, 2);
      s.specularSaturation = clamp(numAttr(this, 'specular-saturation', 1), 0, 20);
      s.blur = clamp(numAttr(this, 'blur', 0), 0, 3);
      s.forceActive = this.hasAttribute('force-active');
    }
    attributeChangedCallback(n, o, v) {
      if (o === v) return;
      if (n === 'value') { const x = parseFloat(v); if (Number.isFinite(x)) this.state.value = clamp(x, 0, 100); if (this.isConnected) { this.updateUI(); this.startAnim(); } return; }
      if (!this.state) return;
      this.readAttrs();
      if (!this.isConnected) return;
      if (n === 'bezel-width' || n === 'dpr' || n === 'texture-cap') this._scheduleTextureUpdate();
      else if (n === 'specular-opacity' || n === 'blur' || n === 'specular-saturation') { this.updateFilterParams(); this.startAnim(); }
      else this.startAnim();
    }
    _scheduleTextureUpdate() {
      this._interactiveQuality = 0.5;
      if (!this._textureRaf) this._textureRaf = requestAnimationFrame(() => { this._textureRaf = null; this.updateTextures(); this.startAnim(); });
      clearTimeout(this._hqTimer);
      this._hqTimer = setTimeout(() => { if (this._interactiveQuality === 1) return; this._interactiveQuality = 1; this.updateTextures(); this.startAnim(); }, 220);
    }
    updateFilterParams() {
      this._saEl.setAttribute('slope', this.state.specularOpacity);
      this._blurEl.setAttribute('stdDeviation', this.state.blur);
      this._satEl.setAttribute('values', this.state.specularSaturation);
    }
    updateTextures() {
      const c = this.config, s = this.state;
      const tex = getTexturesA(c.thumbWidth, c.thumbHeight, c.thumbRadius, c.bezelWidth, 'convex_squircle', c.refractiveIndex, c.glassThickness, c.dpr, c.textureCap * this._interactiveQuality);
      s.maximumDisplacement = tex.maxDisp;
      this._dispEl.setAttribute('href', tex.dispURL);
      this._specEl.setAttribute('href', tex.specURL);
      this.updateFilterParams();
    }
    updateUI() {
      const hostRect = this.getBoundingClientRect();
      const cL = Math.round((hostRect.width - this.config.trackWidth) / 2);
      const cT = Math.round((hostRect.height - this.config.thumbHeight) / 2);
      const trackTop = (this.config.thumbHeight - this.config.trackHeight) / 2;
      const rest = this.config.thumbWidth * this.config.SCALE_REST;
      const v = this.state.value;
      this.fillEl.style.width = v + '%';
      const ratio = v / 100, x0 = rest / 2, x100 = this.config.trackWidth - rest / 2;
      const thumbX = Math.round(x0 + ratio * (x100 - x0) - this.config.thumbWidth / 2);
      this.thumbEl.style.left = thumbX + 'px';
      this.cloneInnerEl.style.width = hostRect.width + 'px';
      this.cloneInnerEl.style.height = hostRect.height + 'px';
      this.cloneInnerEl.style.transform = `translate3d(${-(cL + thumbX)}px,${-cT}px,0)`;
      this.cloneInnerEl.style.setProperty('--track-left', cL + 'px');
      this.cloneInnerEl.style.setProperty('--track-top', (cT + trackTop) + 'px');
      this.cloneInnerEl.style.setProperty('--fill', v);
    }
    startAnim() { if (!this.raf) this.raf = requestAnimationFrame(() => this.tick()); }
    tick() {
      const c = this.config, st = this.state, s = this.springs;
      const dt = Math.min(0.032, 1 / 60), active = st.forceActive || st.pointerDown;
      s.scale.setTarget(active ? c.SCALE_DRAG : c.SCALE_REST);
      s.backgroundOpacity.setTarget(active ? 0 : 1);
      s.shadowOffsetX.setTarget(active ? 1 : 0); s.shadowOffsetY.setTarget(active ? 6 : 3);
      s.shadowBlur.setTarget(active ? 14 : 10); s.shadowAlpha.setTarget(active ? 0.2 : 0.12);
      s.scaleRatio.setTarget((active ? 0.9 : 0.4) * st.refractionBase);
      if (!st.pointerDown) st.velocityX *= 0.95;
      const vel = Math.abs(st.velocityX), sq = Math.min(0.18, vel / 1500);
      if (vel > 10) { if (st.velocityX > 0) { s.scaleX.setTarget(1 + sq); s.scaleY.setTarget(1 - sq * 0.3); } else { s.scaleX.setTarget(1 - sq * 0.3); s.scaleY.setTarget(1 + sq); } }
      else { s.scaleX.setTarget(1); s.scaleY.setTarget(1); }
      const sc = s.scale.update(dt), bg = s.backgroundOpacity.update(dt), sr = s.scaleRatio.update(dt);
      const sx = s.scaleX.update(dt), sy = s.scaleY.update(dt);
      const shx = s.shadowOffsetX.update(dt), shy = s.shadowOffsetY.update(dt), shb = s.shadowBlur.update(dt), sha = s.shadowAlpha.update(dt);
      this.thumbEl.style.transform = `scale(${sc * sx},${sc * sy})`;
      this.thumbEl.style.backgroundColor = `rgba(255,255,255,${bg})`;
      this.cloneEl.style.opacity = 1 - bg;
      const insetAlpha = sha * 0.6;
      this.thumbEl.style.boxShadow = `${shx}px ${shy}px ${shb}px rgba(0,0,0,${sha}),inset ${shx * 0.3}px ${shy * 0.4}px 8px rgba(0,0,0,${insetAlpha}),inset ${-shx * 0.3}px ${-shy * 0.4}px 8px rgba(255,255,255,${insetAlpha * 0.8})`;
      const opacityDiff = Math.abs(bg - (active ? 0 : 1));
      const stateBlur = Math.min(2, opacityDiff * 5), velBlur = Math.min(1.5, vel / 1500);
      const totalBlur = Math.max(velBlur, stateBlur);
      this.thumbEl.style.filter = totalBlur > 0.05 ? `blur(${totalBlur}px)` : 'none';
      const base = st.maximumDisplacement * sr;
      this._drEl.setAttribute('scale', base * (1 - st.dispersion));
      this._dgEl.setAttribute('scale', base);
      this._dbEl.setAttribute('scale', base * (1 + st.dispersion));
      const settled = Object.values(s).every(x => x.isSettled()) && Math.abs(st.velocityX) < 1;
      this.raf = settled ? null : requestAnimationFrame(() => this.tick());
    }
    onDown(e) { e.preventDefault(); this.state.pointerDown = true; this.state.lastX = e.touches ? e.touches[0].clientX : e.clientX; this.state.lastTime = performance.now(); this.state.velocityX = 0; this.startAnim(); }
    onMove(e) {
      if (!this.state.pointerDown) return;
      e.preventDefault();
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const now = performance.now(), dt = Math.max(1, now - this.state.lastTime) / 1000;
      this.state.velocityX = (cx - this.state.lastX) / dt; this.state.lastX = cx; this.state.lastTime = now;
      const rect = this.trackEl.getBoundingClientRect();
      const rest = this.config.thumbWidth * this.config.SCALE_REST;
      const x0 = rect.left + rest / 2, x100 = rect.right - rest / 2;
      const x = Math.max(x0, Math.min(x100, cx));
      this.state.value = Math.max(0, Math.min(100, ((x - x0) / (x100 - x0)) * 100));
      this.updateUI(); this.startAnim();
    }
    onUp() { if (!this.state.pointerDown) return; this.state.pointerDown = false; this.state.velocityX = 0; this.startAnim(); this.dispatchEvent(new CustomEvent('change', { detail: { value: this.state.value }, bubbles: true })); }
  }

  class LiquidSwitch extends HTMLElement {
    static get observedAttributes() { return ['checked', 'bezel-width', 'refraction', 'dispersion', 'specular-opacity', 'specular-saturation', 'blur', 'force-active', 'dpr', 'texture-cap']; }
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.uid = 'sw-' + Math.random().toString(36).slice(2, 8);
      this.shadowRoot.innerHTML = `<style>:host{display:block;position:relative;height:200px;--bg-primary:#f8fafc;--bg-secondary:#e7eeef;--grid:rgba(0,0,0,0.03);}@media (prefers-color-scheme:dark){:host{--bg-primary:#1b1b22;--bg-secondary:#0f0f14;--grid:rgba(255,255,255,0.03);}}.stage{position:relative;width:100%;height:100%;display:flex;justify-content:center;align-items:center;${BG_CSS}}.switch-container{position:relative}.switch-track{display:inline-block;position:relative;width:160px;height:67px;background-color:#94949f;border-radius:33.5px;cursor:pointer}.switch-thumb{position:absolute;width:146px;height:92px;border-radius:46px;top:33.5px;transform:translateY(-50%) scale(0.65);transform-origin:center center;cursor:pointer;touch-action:none;user-select:none;background-color:rgba(255,255,255,1);box-shadow:0 4px 22px rgba(0,0,0,0.1);overflow:hidden;will-change:transform,left,background-color,box-shadow}.switch-thumb:active{cursor:grabbing}.switch-thumb-clone{position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;border-radius:inherit;z-index:1;opacity:0;will-change:opacity}.switch-thumb-clone-inner{position:absolute;top:0;left:0;pointer-events:none;${BG_CSS}}.switch-thumb-clone-inner::before{content:'';position:absolute;left:var(--track-left);top:var(--track-top);width:160px;height:67px;border-radius:33.5px;background:var(--switch-track-color)}.switch-thumb-inner{position:absolute;top:0;left:0;width:100%;height:100%;border-radius:inherit;z-index:3;pointer-events:none}svg{position:absolute;width:0;height:0}</style><div class="stage"><div class="switch-container"><div class="switch-track"></div><div class="switch-thumb"><div class="switch-thumb-inner"></div><div class="switch-thumb-clone"><div class="switch-thumb-clone-inner"></div></div></div></div></div><svg><defs><filter id="${this.uid}-f" ${FILTER_ATTR}><feGaussianBlur in="SourceGraphic" stdDeviation="0.2" result="blurred" id="${this.uid}-blur"/><feImage href="" x="0" y="0" width="146" height="92" result="dm" id="${this.uid}-disp"/><feImage href="" x="0" y="0" width="146" height="92" result="sm" id="${this.uid}-spec"/><feGaussianBlur in="dm" stdDeviation="0.15" result="dms" id="${this.uid}-dms"/><feColorMatrix in="blurred" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0" result="r"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0" result="g"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0" result="b"/><feDisplacementMap in="r" in2="dms" scale="0.85" xChannelSelector="R" yChannelSelector="G" result="dr" id="${this.uid}-dr"/><feDisplacementMap in="g" in2="dms" scale="1" xChannelSelector="R" yChannelSelector="G" result="dg" id="${this.uid}-dg"/><feDisplacementMap in="b" in2="dms" scale="1.15" xChannelSelector="R" yChannelSelector="G" result="db" id="${this.uid}-db"/><feBlend in="dr" in2="dg" mode="lighten" result="rg"/><feBlend in="rg" in2="db" mode="lighten" result="displaced"/><feColorMatrix in="displaced" type="saturate" values="1" result="sat" id="${this.uid}-sat"/><feGaussianBlur in="sm" stdDeviation="0.25" result="sms"/><feComponentTransfer in="sms" result="sa"><feFuncA type="linear" slope="0.5" id="${this.uid}-sa"/></feComponentTransfer><feComposite in="sat" in2="sa" operator="arithmetic" k1="0" k2="1" k3="1" k4="0"/></filter></defs></svg>`;
      this.trackEl = this.shadowRoot.querySelector('.switch-track');
      this.thumbEl = this.shadowRoot.querySelector('.switch-thumb');
      this.cloneEl = this.shadowRoot.querySelector('.switch-thumb-clone');
      this.cloneInnerEl = this.shadowRoot.querySelector('.switch-thumb-clone-inner');
      this._dispEl = this.shadowRoot.getElementById(`${this.uid}-disp`);
      this._specEl = this.shadowRoot.getElementById(`${this.uid}-spec`);
      this._saEl = this.shadowRoot.getElementById(`${this.uid}-sa`);
      this._blurEl = this.shadowRoot.getElementById(`${this.uid}-blur`);
      this._dmsEl = this.shadowRoot.getElementById(`${this.uid}-dms`);
      this._satEl = this.shadowRoot.getElementById(`${this.uid}-sat`);
      this._drEl = this.shadowRoot.getElementById(`${this.uid}-dr`);
      this._dgEl = this.shadowRoot.getElementById(`${this.uid}-dg`);
      this._dbEl = this.shadowRoot.getElementById(`${this.uid}-db`);
      this._textureRaf = null; this._hqTimer = null; this._interactiveQuality = 1;
    }
    connectedCallback() {
      this.config = { trackWidth: 160, trackHeight: 67, thumbWidth: 146, thumbHeight: 92, thumbRadius: 46, bezelWidth: 19, glassThickness: 47, refractiveIndex: 1.5, dpr: 1, textureCap: 512, THUMB_REST_SCALE: 0.65, THUMB_ACTIVE_SCALE: 0.9 };
      this.config.THUMB_REST_OFFSET = (1 - this.config.THUMB_REST_SCALE) * this.config.thumbWidth / 2;
      this.config.TRAVEL = this.config.trackWidth - this.config.trackHeight - (this.config.thumbWidth - this.config.thumbHeight) * this.config.THUMB_REST_SCALE;
      const checked = this.hasAttribute('checked');
      this.state = { checked, pointerDown: false, forceActive: false, isClickAction: false, clickDelayStarted: false, initialPointerX: 0, xDragRatio: checked ? 1 : 0, specularOpacity: 0.5, specularSaturation: 1, refractionBase: 1, dispersion: 0.15, blur: 0.2, maximumDisplacement: 0, velocityX: 0, lastX: 0, lastTime: 0 };
      this.readAttrs();
      const initX = checked ? 1 : 0;
      this.springs = { xRatio: new Spring(initX, 1000, 80), scale: new Spring(this.config.THUMB_REST_SCALE, 500, 18), backgroundOpacity: new Spring(1, 2000, 80), trackColor: new Spring(initX, 1000, 80), scaleRatio: new Spring(0.4, 100, 10), scaleX: new Spring(1, 400, 30), scaleY: new Spring(1, 400, 30), shadowOffsetX: new Spring(0, 400, 30), shadowOffsetY: new Spring(4, 400, 30), shadowBlur: new Spring(14, 400, 30), shadowAlpha: new Spring(0.15, 300, 25) };
      this.raf = null;
      this.cloneEl.style.filter = `url(#${this.uid}-f)`;
      this.updateTextures();
      this.abort = new AbortController(); const sig = this.abort.signal;
      this.thumbEl.addEventListener('mousedown', e => this.onDown(e), { signal: sig });
      this.thumbEl.addEventListener('touchstart', e => this.onDown(e), { passive: false, signal: sig });
      this.trackEl.addEventListener('click', e => {
        if (e.target === this.trackEl && !this.state.pointerDown) {
          this.state.checked = !this.state.checked; this.state.isClickAction = true; this.state.clickDelayStarted = false;
          this.syncCheckedAttr(); this.startAnim();
          this.dispatchEvent(new CustomEvent('change', { detail: { checked: this.state.checked }, bubbles: true }));
        }
      }, { signal: sig });
      window.addEventListener('mousemove', e => this.onMove(e), { signal: sig });
      window.addEventListener('touchmove', e => this.onMove(e), { passive: false, signal: sig });
      window.addEventListener('mouseup', e => this.onUp(e), { signal: sig });
      window.addEventListener('touchend', e => this.onUp(e), { signal: sig });
      this.ro = new ResizeObserver(() => this.updateUI()); this.ro.observe(this);
      requestAnimationFrame(() => { this.updateUI(); this.startAnim(); });
    }
    disconnectedCallback() { this.abort?.abort(); if (this.raf) cancelAnimationFrame(this.raf); if (this._textureRaf) cancelAnimationFrame(this._textureRaf); clearTimeout(this._hqTimer); this.ro?.disconnect(); }
    syncCheckedAttr() { if (this.state.checked) { if (!this.hasAttribute('checked')) this.setAttribute('checked', ''); } else if (this.hasAttribute('checked')) this.removeAttribute('checked'); }
    readAttrs() {
      const c = this.config, s = this.state;
      c.bezelWidth = clamp(numAttr(this, 'bezel-width', 19), 2, 35);
      c.dpr = clamp(numAttr(this, 'dpr', 1), 0.5, 3);
      c.textureCap = Math.max(0, numAttr(this, 'texture-cap', 512));
      s.refractionBase = clamp(numAttr(this, 'refraction', 1), 0.1, 3);
      s.dispersion = clamp(numAttr(this, 'dispersion', 0.15), 0, 0.3);
      s.specularOpacity = clamp(numAttr(this, 'specular-opacity', 0.5), 0, 2);
      s.specularSaturation = clamp(numAttr(this, 'specular-saturation', 1), 0, 20);
      s.blur = clamp(numAttr(this, 'blur', 0.2), 0, 3);
      s.forceActive = this.hasAttribute('force-active');
    }
    attributeChangedCallback(name, oldVal, newVal) {
      if (oldVal === newVal) return;
      if (name === 'checked') { if (!this.state) return; this.state.checked = newVal !== null; this.state.isClickAction = false; if (this.isConnected) this.startAnim(); return; }
      if (!this.state) return;
      this.readAttrs();
      if (!this.isConnected) return;
      if (name === 'bezel-width' || name === 'dpr' || name === 'texture-cap') this._scheduleTextureUpdate();
      else if (name === 'specular-opacity' || name === 'blur' || name === 'specular-saturation') { this.updateFilterParams(); this.startAnim(); }
      else this.startAnim();
    }
    _scheduleTextureUpdate() {
      this._interactiveQuality = 0.5;
      if (!this._textureRaf) this._textureRaf = requestAnimationFrame(() => { this._textureRaf = null; this.updateTextures(); this.startAnim(); });
      clearTimeout(this._hqTimer);
      this._hqTimer = setTimeout(() => { if (this._interactiveQuality === 1) return; this._interactiveQuality = 1; this.updateTextures(); this.startAnim(); }, 220);
    }
    updateFilterParams() {
      this._saEl.setAttribute('slope', this.state.specularOpacity);
      this._blurEl.setAttribute('stdDeviation', this.state.blur);
      this._satEl.setAttribute('values', this.state.specularSaturation);
    }
    updateTextures() {
      const c = this.config, s = this.state;
      const tex = getTexturesA(c.thumbWidth, c.thumbHeight, c.thumbRadius, c.bezelWidth, 'convex_squircle', c.refractiveIndex, c.glassThickness, c.dpr, c.textureCap * this._interactiveQuality);
      s.maximumDisplacement = tex.maxDisp;
      this._dispEl.setAttribute('href', tex.dispURL);
      this._specEl.setAttribute('href', tex.specURL);
      this.updateFilterParams();
    }
    updateUI() {
      const hostRect = this.getBoundingClientRect();
      this.cloneInnerEl.style.width = hostRect.width + 'px';
      this.cloneInnerEl.style.height = hostRect.height + 'px';
      this.startAnim();
    }
    startAnim() { if (!this.raf) this.raf = requestAnimationFrame(() => this.tick()); }
    tick() {
      const c = this.config, st = this.state, s = this.springs;
      const dt = Math.min(0.032, 1 / 60);
      const isActive = st.forceActive || st.pointerDown || st.isClickAction;
      s.scale.setTarget(isActive ? c.THUMB_ACTIVE_SCALE : c.THUMB_REST_SCALE);
      s.backgroundOpacity.setTarget(isActive ? 0 : 1);
      s.shadowOffsetX.setTarget(isActive ? 2 : 0); s.shadowOffsetY.setTarget(isActive ? 10 : 4);
      s.shadowBlur.setTarget(isActive ? 20 : 14); s.shadowAlpha.setTarget(isActive ? 0.25 : 0.15);
      s.scaleRatio.setTarget((isActive ? 0.9 : 0.4) * st.refractionBase);
      if (!st.pointerDown) s.xRatio.setTarget(st.checked ? 1 : 0);
      s.trackColor.setTarget(Math.max(0, Math.min(1, s.xRatio.target)));
      if (!st.pointerDown) st.velocityX *= 0.95;
      const xRatioVel = s.xRatio.velocity;
      const combinedVel = Math.abs(st.velocityX) + Math.abs(xRatioVel) * 20;
      const squishAmount = Math.min(0.18, combinedVel / 3000);
      if (combinedVel > 50) { if (st.velocityX < 0 || xRatioVel < 0) { s.scaleX.setTarget(1 - squishAmount * 0.5); s.scaleY.setTarget(1 + squishAmount); } else { s.scaleX.setTarget(1 + squishAmount); s.scaleY.setTarget(1 - squishAmount * 0.5); } }
      else { s.scaleX.setTarget(1); s.scaleY.setTarget(1); }
      const xRatio = s.xRatio.update(dt), scale = s.scale.update(dt), bg = s.backgroundOpacity.update(dt);
      const trackColor = s.trackColor.update(dt), sr = s.scaleRatio.update(dt);
      const sx = s.scaleX.update(dt), sy = s.scaleY.update(dt);
      const shx = s.shadowOffsetX.update(dt), shy = s.shadowOffsetY.update(dt), shb = s.shadowBlur.update(dt), sha = s.shadowAlpha.update(dt);
      this.cloneEl.style.opacity = 1 - bg;
      const insetAlpha = sha * 0.6;
      this.thumbEl.style.boxShadow = `${shx}px ${shy}px ${shb}px rgba(0,0,0,${sha}),inset ${shx * 0.3}px ${shy * 0.4}px 10px rgba(0,0,0,${insetAlpha}),inset ${-shx * 0.3}px ${-shy * 0.4}px 10px rgba(255,255,255,${insetAlpha * 0.8})`;
      const opacityDiff = Math.abs(bg - (isActive ? 0 : 1));
      const stateBlur = Math.min(2.5, opacityDiff * 6), velBlur = Math.min(1.5, combinedVel / 2000);
      const totalBlur = Math.max(velBlur, stateBlur);
      this.thumbEl.style.filter = totalBlur > 0.05 ? `blur(${totalBlur}px)` : 'none';
      const marginLeft = -c.THUMB_REST_OFFSET + (c.trackHeight - c.thumbHeight * c.THUMB_REST_SCALE) / 2;
      const thumbX = Math.round(marginLeft + xRatio * c.TRAVEL);
      this.thumbEl.style.left = thumbX + 'px';
      this.thumbEl.style.transform = `translateY(-50%) scale(${scale * sx},${scale * sy})`;
      this.thumbEl.style.backgroundColor = `rgba(255,255,255,${bg})`;
      const offR = 148, offG = 148, offB = 159, onR = 59, onG = 191, onB = 78;
      const r = Math.round(offR + (onR - offR) * trackColor);
      const g = Math.round(offG + (onG - offG) * trackColor);
      const b = Math.round(offB + (onB - offB) * trackColor);
      const trackRgb = `rgb(${r},${g},${b})`;
      this.trackEl.style.backgroundColor = trackRgb;
      const hostRect = this.getBoundingClientRect();
      const cL = Math.round((hostRect.width - c.trackWidth) / 2), cT = Math.round((hostRect.height - c.trackHeight) / 2);
      const thumbYOffset = Math.round(c.trackHeight / 2 - c.thumbHeight / 2);
      this.cloneInnerEl.style.width = hostRect.width + 'px';
      this.cloneInnerEl.style.height = hostRect.height + 'px';
      this.cloneInnerEl.style.transform = `translate3d(${-(cL + thumbX)}px,${-(cT + thumbYOffset)}px,0)`;
      this.cloneInnerEl.style.setProperty('--switch-track-color', trackRgb);
      this.cloneInnerEl.style.setProperty('--track-left', cL + 'px');
      this.cloneInnerEl.style.setProperty('--track-top', cT + 'px');
      const baseScale = st.maximumDisplacement * sr;
      this._drEl.setAttribute('scale', baseScale * (1 - st.dispersion));
      this._dgEl.setAttribute('scale', baseScale);
      this._dbEl.setAttribute('scale', baseScale * (1 + st.dispersion));
      if (st.isClickAction && !st.pointerDown && !st.clickDelayStarted) {
        const rd = Math.abs(s.xRatio.target - s.xRatio.value);
        if (rd < 0.03) { st.clickDelayStarted = true; st.isClickAction = false; this.startAnim(); }
      }
      const settled = Object.values(s).every(x => x.isSettled());
      this.raf = settled ? null : requestAnimationFrame(() => this.tick());
    }
    onDown(e) {
      e.preventDefault(); e.stopPropagation();
      this.state.pointerDown = true;
      this.state.initialPointerX = e.touches ? e.touches[0].clientX : e.clientX;
      this.state.xDragRatio = this.state.checked ? 1 : 0;
      this.state.isClickAction = false; this.state.clickDelayStarted = false;
      this.state.lastX = e.touches ? e.touches[0].clientX : e.clientX;
      this.state.lastTime = performance.now(); this.state.velocityX = 0;
      this.startAnim();
    }
    onMove(e) {
      if (!this.state.pointerDown) return;
      e.stopPropagation();
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const now = performance.now(), dt = Math.max(1, now - this.state.lastTime) / 1000;
      this.state.velocityX = (cx - this.state.lastX) / dt;
      this.state.lastX = cx; this.state.lastTime = now;
      const baseRatio = this.state.checked ? 1 : 0;
      const displacementX = cx - this.state.initialPointerX;
      let ratio = baseRatio + displacementX / this.config.TRAVEL;
      const overflow = ratio < 0 ? -ratio : ratio > 1 ? ratio - 1 : 0;
      const overflowSign = ratio < 0 ? -1 : 1;
      const dampedOverflow = (overflowSign * overflow) / 22;
      this.state.xDragRatio = Math.min(1, Math.max(0, ratio)) + dampedOverflow;
      this.springs.xRatio.setTarget(this.state.xDragRatio);
      this.startAnim();
    }
    onUp(e) {
      if (!this.state.pointerDown) return;
      const cx = e.changedTouches ? e.changedTouches[0].clientX : e.clientX;
      const distance = Math.abs(cx - this.state.initialPointerX);
      this.state.pointerDown = false; this.state.velocityX = 0;
      if (distance < 4) { this.state.checked = !this.state.checked; this.state.isClickAction = true; this.state.clickDelayStarted = false; }
      else { this.state.checked = this.state.xDragRatio > 0.5; this.state.isClickAction = false; }
      this.syncCheckedAttr(); this.startAnim();
      this.dispatchEvent(new CustomEvent('change', { detail: { checked: this.state.checked }, bubbles: true }));
    }
  }

  class LiquidLens extends HTMLElement {
    static get observedAttributes() { return ['lens-width', 'lens-height', 'lens-radius', 'surface', 'bezel-width', 'glass-thickness', 'refraction', 'dispersion', 'specular-opacity', 'blur', 'render-mode', 'dpr', 'texture-cap']; }
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.uid = 'ln-' + Math.random().toString(36).slice(2, 8);
      this.shadowRoot.innerHTML = `<style>:host{display:block;position:relative;width:100%;height:400px;--bg-primary:#f8fafc;--bg-secondary:#e7eeef;--grid:rgba(0,0,0,0.03);overflow:hidden}@media (prefers-color-scheme:dark){:host{--bg-primary:#1b1b22;--bg-secondary:#0f0f14;--grid:rgba(255,255,255,0.03);}}.stage{position:absolute;inset:0;${BG_CSS}}::slotted(*){width:100%;height:100%;display:block;object-fit:cover}.glass-element{position:absolute;cursor:grab;touch-action:none;user-select:none;will-change:transform,left,top;overflow:hidden}.glass-element:active{cursor:grabbing}.glass-inner{width:100%;height:100%;border-radius:inherit;will-change:transform,box-shadow;position:absolute;top:0;left:0;z-index:3;pointer-events:none}.glass-clone{position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;border-radius:inherit;z-index:1;will-change:filter;contain:layout style}.glass-clone-inner{position:absolute;top:0;left:0;pointer-events:none;will-change:transform;contain:layout style;transform-origin:0 0}.use-backdrop-filter .glass-inner{backdrop-filter:url(#${this.uid}-f);-webkit-backdrop-filter:url(#${this.uid}-f)}.use-backdrop-filter .glass-clone{display:none}svg{position:absolute;width:0;height:0;overflow:hidden;pointer-events:none}</style><div class="stage"><slot></slot></div><div class="glass-element"><div class="glass-inner"></div><div class="glass-clone"><div class="glass-clone-inner"></div></div></div><svg><defs><filter id="${this.uid}-f" ${FILTER_ATTR}><feGaussianBlur in="SourceGraphic" stdDeviation="0" result="blurred" id="${this.uid}-blur"/><feImage href="" x="0" y="0" width="200" height="140" result="dm" id="${this.uid}-disp"/><feImage href="" x="0" y="0" width="200" height="140" result="sm" id="${this.uid}-spec"/><feGaussianBlur in="dm" stdDeviation="0.2" result="dms" id="${this.uid}-dms"/><feColorMatrix in="blurred" type="matrix" values="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0" result="r"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0" result="g"/><feColorMatrix in="blurred" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0" result="b"/><feDisplacementMap in="r" in2="dms" scale="0.95" xChannelSelector="R" yChannelSelector="G" result="dr" id="${this.uid}-dr"/><feDisplacementMap in="g" in2="dms" scale="1" xChannelSelector="R" yChannelSelector="G" result="dg" id="${this.uid}-dg"/><feDisplacementMap in="b" in2="dms" scale="1.05" xChannelSelector="R" yChannelSelector="G" result="db" id="${this.uid}-db"/><feBlend in="dr" in2="dg" mode="lighten" result="rg"/><feBlend in="rg" in2="db" mode="lighten" result="displaced"/><feColorMatrix in="displaced" type="saturate" values="1" result="sat" id="${this.uid}-sat"/><feGaussianBlur in="sm" stdDeviation="0.4" result="sms"/><feComponentTransfer in="sms" result="sa"><feFuncA type="linear" slope="1" id="${this.uid}-sa"/></feComponentTransfer><feComposite in="sat" in2="sa" operator="arithmetic" k1="0" k2="1" k3="1" k4="0"/></filter></defs></svg>`;
      this.glassEl = this.shadowRoot.querySelector('.glass-element');
      this.innerEl = this.shadowRoot.querySelector('.glass-inner');
      this.cloneEl = this.shadowRoot.querySelector('.glass-clone');
      this.cloneInnerEl = this.shadowRoot.querySelector('.glass-clone-inner');
      this._dispEl = this.shadowRoot.getElementById(`${this.uid}-disp`);
      this._specEl = this.shadowRoot.getElementById(`${this.uid}-spec`);
      this._saEl = this.shadowRoot.getElementById(`${this.uid}-sa`);
      this._blurEl = this.shadowRoot.getElementById(`${this.uid}-blur`);
      this._dmsEl = this.shadowRoot.getElementById(`${this.uid}-dms`);
      this._drEl = this.shadowRoot.getElementById(`${this.uid}-dr`);
      this._dgEl = this.shadowRoot.getElementById(`${this.uid}-dg`);
      this._dbEl = this.shadowRoot.getElementById(`${this.uid}-db`);
      this._textureRaf = null; this._hqTimer = null; this._interactiveQuality = 1;
    }
    connectedCallback() {
      this.config = { lensWidth: 200, lensHeight: 140, lensRadius: 70, surfaceType: 'convex_squircle', bezelWidth: 30, glassThickness: 0, refractiveIndex: 1.5, refractionScale: 3, dispersion: 0.05, specularOpacity: 1, blur: 0, maxDisp: 0, dpr: 1, textureCap: 512 };
      this.readAttrs();
      this.state = { isDragging: false, dragOffset: { x: 0, y: 0 }, velocityX: 0, velocityY: 0, lastX: 0, lastY: 0, lastTime: 0 };
      this.springs = { scale: new Spring(0.85, 400, 25), sx: new Spring(1, 400, 30), sy: new Spring(1, 400, 30), shx: new Spring(0, 400, 30), shy: new Spring(4, 400, 30), shb: new Spring(12, 400, 30), sha: new Spring(0.15, 300, 25), boost: new Spring(0.8, 300, 18), px: new Spring(100, 300, 25), py: new Spring(100, 300, 25), rx: new Spring(0, 300, 20), ry: new Spring(0, 300, 20), press: new Spring(0, 400, 25) };
      this.raf = null; this.useBackdropFilter = false;
      this.detectRenderMode(); this.applyRenderMode();
      this.cloneEl.style.filter = `url(#${this.uid}-f)`;
      const hostRect = this.getBoundingClientRect();
      const initX = Math.max(0, Math.round((hostRect.width - this.config.lensWidth) / 2));
      const initY = Math.max(0, Math.round((hostRect.height - this.config.lensHeight) / 2));
      this.springs.px.value = initX; this.springs.px.target = initX;
      this.springs.py.value = initY; this.springs.py.target = initY;
      this.applyGeometry();
      this.abort = new AbortController(); const sig = this.abort.signal;
      this.glassEl.addEventListener('mousedown', e => this.onDown(e), { signal: sig });
      this.glassEl.addEventListener('touchstart', e => this.onDown(e), { passive: false, signal: sig });
      document.addEventListener('mousemove', e => this.onMove(e), { signal: sig });
      document.addEventListener('touchmove', e => this.onMove(e), { passive: false, signal: sig });
      document.addEventListener('mouseup', () => this.onUp(), { signal: sig });
      document.addEventListener('touchend', () => this.onUp(), { signal: sig });
      this.ro = new ResizeObserver(() => this.refreshClone()); this.ro.observe(this);
      const slot = this.shadowRoot.querySelector('slot');
      slot.addEventListener('slotchange', () => this.refreshClone());
      requestAnimationFrame(() => { this.refreshClone(); this.updateTextures(); this.startAnim(); });
    }
    disconnectedCallback() { this.abort?.abort(); if (this.raf) cancelAnimationFrame(this.raf); if (this._textureRaf) cancelAnimationFrame(this._textureRaf); clearTimeout(this._hqTimer); this.ro?.disconnect(); }
    detectRenderMode() {
      const forced = this.getAttribute('render-mode');
      if (forced === 'backdrop') { this.useBackdropFilter = true; return; }
      if (forced === 'clone') { this.useBackdropFilter = false; return; }
      const isChromium = !!window.chrome, testEl = document.createElement('div');
      testEl.style.backdropFilter = 'url(#test)';
      this.useBackdropFilter = isChromium && testEl.style.backdropFilter.includes('url');
    }
    applyRenderMode() { this.glassEl.classList.toggle('use-backdrop-filter', this.useBackdropFilter); }
    readAttrs() {
      const c = this.config;
      c.lensWidth = Math.max(40, numAttr(this, 'lens-width', 200));
      c.lensHeight = Math.max(40, numAttr(this, 'lens-height', 140));
      c.lensRadius = clamp(numAttr(this, 'lens-radius', 70), 0, Math.min(c.lensWidth, c.lensHeight) / 2);
      c.surfaceType = this.getAttribute('surface') || 'convex_squircle';
      if (!SurfaceEquations[c.surfaceType]) c.surfaceType = 'convex_squircle';
      c.bezelWidth = clamp(numAttr(this, 'bezel-width', 30), 5, Math.min(c.lensWidth, c.lensHeight) / 2);
      c.glassThickness = clamp(numAttr(this, 'glass-thickness', 0), 0, 300);
      c.refractionScale = clamp(numAttr(this, 'refraction', 3), 0.1, 3);
      c.dispersion = clamp(numAttr(this, 'dispersion', 0.05), 0, 0.3);
      c.specularOpacity = clamp(numAttr(this, 'specular-opacity', 1), 0, 2);
      c.blur = clamp(numAttr(this, 'blur', 0), 0, 3);
      c.dpr = clamp(numAttr(this, 'dpr', 1), 0.5, 3);
      c.textureCap = Math.max(0, numAttr(this, 'texture-cap', 512));
    }
    applyGeometry() {
      const c = this.config;
      this.glassEl.style.width = c.lensWidth + 'px'; this.glassEl.style.height = c.lensHeight + 'px';
      this.glassEl.style.borderRadius = c.lensRadius + 'px';
      this.glassEl.style.left = Math.round(this.springs.px.value) + 'px';
      this.glassEl.style.top = Math.round(this.springs.py.value) + 'px';
    }
    attributeChangedCallback(name, oldVal, newVal) {
      if (oldVal === newVal) return;
      if (!this.state || !this.config) return;
      if (name === 'render-mode') { this.detectRenderMode(); this.applyRenderMode(); this.refreshClone(); this.startAnim(); return; }
      const prevW = this.config.lensWidth, prevH = this.config.lensHeight;
      this.readAttrs();
      if (!this.isConnected) return;
      this.applyGeometry();
      if (prevW !== this.config.lensWidth || prevH !== this.config.lensHeight) {
        const hostRect = this.getBoundingClientRect();
        this.springs.px.value = clamp(this.springs.px.value, 0, Math.max(0, hostRect.width - this.config.lensWidth)); this.springs.px.target = this.springs.px.value;
        this.springs.py.value = clamp(this.springs.py.value, 0, Math.max(0, hostRect.height - this.config.lensHeight)); this.springs.py.target = this.springs.py.value;
        this.applyGeometry();
      }
      this.updateFilterSize();
      if (name === 'lens-width' || name === 'lens-height' || name === 'lens-radius' || name === 'surface' || name === 'bezel-width' || name === 'glass-thickness' || name === 'refraction' || name === 'dpr' || name === 'texture-cap') this._scheduleTextureUpdate();
      else if (name === 'specular-opacity' || name === 'blur') { this.updateFilterParams(); this.startAnim(); }
      else this.startAnim();
    }
    _scheduleTextureUpdate() {
      this._interactiveQuality = 0.5;
      if (!this._textureRaf) this._textureRaf = requestAnimationFrame(() => { this._textureRaf = null; this.updateTextures(); this.startAnim(); });
      clearTimeout(this._hqTimer);
      this._hqTimer = setTimeout(() => { if (this._interactiveQuality === 1) return; this._interactiveQuality = 1; this.updateTextures(); this.startAnim(); }, 220);
    }
    updateFilterSize() {
      const c = this.config;
      this._dispEl.setAttribute('width', c.lensWidth); this._dispEl.setAttribute('height', c.lensHeight);
      this._specEl.setAttribute('width', c.lensWidth); this._specEl.setAttribute('height', c.lensHeight);
    }
    updateFilterParams() { this._saEl.setAttribute('slope', this.config.specularOpacity); this._blurEl.setAttribute('stdDeviation', this.config.blur); }
    updateTextures() {
      const c = this.config;
      const tex = getTextures(c.lensWidth, c.lensHeight, c.lensRadius, c.bezelWidth, c.surfaceType, c.refractiveIndex, c.glassThickness, c.dpr, c.textureCap * this._interactiveQuality);
      c.maxDisp = tex.maxDisp;
      this._dispEl.setAttribute('href', tex.dispURL);
      this._specEl.setAttribute('href', tex.specURL);
      this.updateFilterSize(); this.updateFilterParams();
      const dispBlur = Math.min(4.0, 0.4 + Math.max(0, c.maxDisp - 20) * 0.04);
      this._dmsEl.setAttribute('stdDeviation', dispBlur.toFixed(2));
    }
    refreshClone() {
      const hostRect = this.getBoundingClientRect();
      if (hostRect.width === 0 || hostRect.height === 0) return;
      const inner = this.cloneInnerEl; inner.innerHTML = '';
      if (this.useBackdropFilter) return;
      const nodes = this.querySelectorAll(':scope > *');
      if (nodes.length === 0) return;
      const wrapper = document.createElement('div');
      wrapper.style.cssText = `position:absolute;top:0;left:0;width:${hostRect.width}px;height:${hostRect.height}px;overflow:hidden;`;
      nodes.forEach(node => {
        const cloned = node.cloneNode(true);
        if (cloned.nodeType === 1) {
          cloned.style.width = '100%'; cloned.style.height = '100%';
          if (!cloned.style.display) cloned.style.display = 'block';
          if (cloned.tagName === 'IMG') cloned.style.objectFit = 'cover';
        }
        wrapper.appendChild(cloned);
      });
      inner.appendChild(wrapper);
    }
    startAnim() { if (!this.raf) this.raf = requestAnimationFrame(() => this.tick()); }
    tick() {
      const c = this.config, st = this.state, s = this.springs;
      const dt = Math.min(0.032, 1 / 60);
      if (st.isDragging) { s.scale.setTarget(1.05); s.shx.setTarget(1); s.shy.setTarget(3); s.shb.setTarget(8); s.sha.setTarget(0.1); s.boost.setTarget(1); }
      else { s.scale.setTarget(0.85); s.shx.setTarget(0); s.shy.setTarget(6); s.shb.setTarget(16); s.sha.setTarget(0.15); s.boost.setTarget(0.8); s.press.setTarget(0); s.rx.setTarget(0); s.ry.setTarget(0); }
      const velMag = Math.sqrt(st.velocityX ** 2 + st.velocityY ** 2), squish = Math.min(0.15, velMag / 3000);
      if (velMag > 50) {
        const vxN = st.velocityX / velMag, vyN = st.velocityY / velMag;
        if (vxN < 0) { s.sx.setTarget(1 - squish * 0.5 * Math.abs(vxN) - squish * 0.5 * Math.abs(vyN)); s.sy.setTarget(1 + squish * Math.abs(vxN) + squish * Math.abs(vyN)); }
        else { s.sx.setTarget(1 + squish * Math.abs(vxN) - squish * 0.5 * Math.abs(vyN)); s.sy.setTarget(1 + squish * Math.abs(vyN) - squish * 0.5 * Math.abs(vxN)); }
      } else { s.sx.setTarget(1); s.sy.setTarget(1); }
      const scale = s.scale.update(dt), sx = s.sx.update(dt), sy = s.sy.update(dt);
      const shx = s.shx.update(dt), shy = s.shy.update(dt), shb = s.shb.update(dt), sha = s.sha.update(dt);
      const boost = s.boost.update(dt), rx = s.rx.update(dt), ry = s.ry.update(dt), press = s.press.update(dt);
      let px = s.px.update(dt), py = s.py.update(dt);
      if (!st.isDragging) {
        const hostRect = this.getBoundingClientRect();
        const maxX = Math.max(0, hostRect.width - c.lensWidth), maxY = Math.max(0, hostRect.height - c.lensHeight);
        if (px < 0) { px = 0; s.px.value = 0; if (s.px.velocity < 0) s.px.velocity *= -0.4; }
        else if (px > maxX) { px = maxX; s.px.value = maxX; if (s.px.velocity > 0) s.px.velocity *= -0.4; }
        if (py < 0) { py = 0; s.py.value = 0; if (s.py.velocity < 0) s.py.velocity *= -0.4; }
        else if (py > maxY) { py = maxY; s.py.value = maxY; if (s.py.velocity > 0) s.py.velocity *= -0.4; }
      }
      px = Math.round(px); py = Math.round(py);
      this.glassEl.style.left = px + 'px'; this.glassEl.style.top = py + 'px';
      const psX = 1 + press * 0.04, psY = 1 - press * 0.04, tSX = scale * sx * psX, tSY = scale * sy * psY;
      this.glassEl.style.transform = `scale(${tSX},${tSY}) rotateX(${rx}deg) rotateY(${ry}deg)`;
      const insetAlpha = sha * 0.6, insetSize = Math.max(4, Math.min(8, Math.min(c.lensWidth, c.lensHeight) * 0.04));
      this.innerEl.style.boxShadow = `${shx}px ${shy}px ${shb}px rgba(0,0,0,${sha}),inset ${shx * 0.3}px ${shy * 0.4}px ${insetSize}px rgba(0,0,0,${insetAlpha}),inset ${-shx * 0.3}px ${-shy * 0.4}px ${insetSize}px rgba(255,255,255,${insetAlpha * 0.8})`;
      const dynScale = c.refractionScale * boost;
      const maxLensDim = Math.min(c.lensWidth, c.lensHeight);
      const baseScale = Math.min(c.maxDisp * dynScale, maxLensDim * 0.8);
      this._drEl.setAttribute('scale', baseScale * (1 - c.dispersion));
      this._dgEl.setAttribute('scale', baseScale);
      this._dbEl.setAttribute('scale', baseScale * (1 + c.dispersion));
      if (!this.useBackdropFilter && this.cloneInnerEl.firstChild) {
        const tx = (tSX - 1) * (px + c.lensWidth / 2), ty = (tSY - 1) * (py + c.lensHeight / 2);
        this.cloneInnerEl.style.transform = `translate3d(${-px}px,${-py}px,0) scale(${1 / tSX},${1 / tSY}) translate3d(${tx}px,${ty}px,0)`;
      }
      if (!st.isDragging) { st.velocityX *= 0.95; st.velocityY *= 0.95; }
      const settled = Object.values(s).every(x => x.isSettled()) && Math.abs(st.velocityX) < 1 && Math.abs(st.velocityY) < 1;
      this.raf = settled ? null : requestAnimationFrame(() => this.tick());
    }
    onDown(e) {
      e.preventDefault(); this.state.isDragging = true; this.springs.press.setTarget(1);
      const cx = e.touches ? e.touches[0].clientX : e.clientX, cy = e.touches ? e.touches[0].clientY : e.clientY;
      const rect = this.glassEl.getBoundingClientRect(), curScale = this.springs.scale.value;
      this.state.dragOffset.x = (cx - rect.left) / curScale;
      this.state.dragOffset.y = (cy - rect.top) / curScale;
      this.state.lastX = cx; this.state.lastY = cy; this.state.lastTime = performance.now();
      this.state.velocityX = 0; this.state.velocityY = 0;
      this.startAnim();
    }
    onMove(e) {
      if (!this.state.isDragging) return;
      e.preventDefault();
      const cx = e.touches ? e.touches[0].clientX : e.clientX, cy = e.touches ? e.touches[0].clientY : e.clientY;
      const hostRect = this.getBoundingClientRect();
      const now = performance.now(), dt = Math.max(1, now - this.state.lastTime) / 1000;
      this.state.velocityX = (cx - this.state.lastX) / dt;
      this.state.velocityY = (cy - this.state.lastY) / dt;
      this.state.lastX = cx; this.state.lastY = cy; this.state.lastTime = now;
      let nx = cx - hostRect.left - this.state.dragOffset.x, ny = cy - hostRect.top - this.state.dragOffset.y;
      const maxX = Math.max(0, hostRect.width - this.config.lensWidth), maxY = Math.max(0, hostRect.height - this.config.lensHeight);
      nx = Math.round(clamp(nx, 0, maxX)); ny = Math.round(clamp(ny, 0, maxY));
      this.glassEl.style.left = nx + 'px'; this.glassEl.style.top = ny + 'px';
      this.springs.px.value = nx; this.springs.px.target = nx;
      this.springs.py.value = ny; this.springs.py.target = ny;
      const cX = this.config.lensWidth / 2, cY = this.config.lensHeight / 2;
      const offX = this.state.dragOffset.x - cX, offY = this.state.dragOffset.y - cY;
      const normX = clamp(offX / cX, -1, 1), normY = clamp(offY / cY, -1, 1), maxTilt = 12;
      this.springs.ry.setTarget(normX * maxTilt);
      this.springs.rx.setTarget(-normY * maxTilt);
      this.startAnim();
    }
    onUp() {
      if (!this.state.isDragging) return;
      this.state.isDragging = false; this.state.velocityX = 0; this.state.velocityY = 0;
      const hostRect = this.getBoundingClientRect();
      let cx = parseFloat(this.glassEl.style.left) || 0, cy = parseFloat(this.glassEl.style.top) || 0;
      cx = clamp(cx, 0, Math.max(0, hostRect.width - this.config.lensWidth));
      cy = clamp(cy, 0, Math.max(0, hostRect.height - this.config.lensHeight));
      this.springs.px.value = cx; this.springs.px.target = cx; this.springs.px.velocity = 0;
      this.springs.py.value = cy; this.springs.py.target = cy; this.springs.py.velocity = 0;
      this.springs.press.setTarget(0);
      this.springs.rx.setTarget(0); this.springs.ry.setTarget(0);
      this.startAnim();
    }
  }

  if (!customElements.get('liquid-slider')) customElements.define('liquid-slider', LiquidSlider);
  if (!customElements.get('liquid-switch')) customElements.define('liquid-switch', LiquidSwitch);
  if (!customElements.get('liquid-lens')) customElements.define('liquid-lens', LiquidLens);
})();
