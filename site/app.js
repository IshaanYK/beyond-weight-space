/**
 * BEYOND WEIGHT-SPACE PROGRAMMING (SP-SV-WSP) - INTERACTIVE ENGINE
 * Exact reproduction of Jiajun Fan's signature isometric canvas animations,
 * interactive matrix damage/laser repair simulator, and research telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initScrollSpy();
  initProgressBar();
  initParadigmCanvas();
  initMatrixSurgeryHeatmap();
  initGeometricProjectionCanvas();
  initTokenStreamComparator();
  initSvdChartCanvas();
  initBibtexCopy();
  initSurgeryStageStepper();
  initLayerDepthMap();
});

/* ==========================================================================
   0. Theme Toggle (Light / Dark Mode matching Jiajun Fan)
   ========================================================================== */
function initThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (!toggleBtn) return;

  const savedTheme = localStorage.getItem('darkMode');
  if (savedTheme === 'true') {
    document.body.classList.add('dark-mode');
  }

  toggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark ? 'true' : 'false');
  });
}

/* ==========================================================================
   0.1. Scrollspy for Sticky Chapters Bar
   ========================================================================== */
function initScrollSpy() {
  const links = document.querySelectorAll('.chapters a[href^="#"]');
  if (!links.length) return;

  const sections = [];
  links.forEach(l => {
    const id = l.getAttribute('href').slice(1);
    const sec = document.getElementById(id);
    if (sec) sections.push({ link: l, section: sec });
  });

  window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY + 140;
    for (let i = sections.length - 1; i >= 0; i--) {
      const item = sections[i];
      if (item.section.offsetTop <= scrollPos) {
        links.forEach(l => l.classList.remove('active'));
        item.link.classList.add('active');
        break;
      }
    }
  }, { passive: true });
}

/* ==========================================================================
   1. Reading Progress Bar
   ========================================================================== */
function initProgressBar() {
  const bar = document.getElementById('read-progress');
  if (!bar) return;

  window.addEventListener('scroll', () => {
    const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = totalHeight > 0 ? (window.scrollY / totalHeight) * 100 : 0;
    bar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  }, { passive: true });
}

/* ==========================================================================
   2. Three Paradigms Canvas Engine (Exact Jiajun Fan fan_script_4.js)
   ========================================================================== */
function initParadigmCanvas() {
  var fig = document.getElementById('paradigm');
  if (!fig) return;
  var acts = [].slice.call(fig.querySelectorAll('.pdm-act')),
      cvs = acts.map(function(a){ return a.querySelector('canvas'); });
  if (acts.length !== 3 || !cvs[0].getContext) return;
  
  var cx = cvs.map(function(v){ return v.getContext('2d'); }),
      nos = [].slice.call(fig.querySelectorAll('.pdm-no span'));
  var mq = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
  var LW = 320, LH = 338, A = [0, 5.8, 11.4, 22.65], D = [5.8, 5.6, 11.25], HOLD = 6.2, OUT = 0.8, T = A[3] + HOLD, FINAL = A[3] + 1.2;
  var NT = [1.4, 3, 4.6, 6.2, 7.8, 10.6];
  var c = null, C = null, sc = 1, dpr = 1, fixed = null, visible = !('IntersectionObserver' in window), raf = 0, last = 0, tc = 0;
  var stacked = false, focus = 0, seenPx = [0, 0, 0], key = '', P = [-1, -1, -1];
  var N = 12, DX = 9.4, PS = 11, PK = 5, PH = 72, MX = 160, MY = 136, X0 = MX - (N - 1) * DX / 2, Y0 = MY - PH / 2, Y1 = MY + PH / 2;

  function hex(s){
    s = s.trim().replace('#', '');
    if (s.length === 3) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
    return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
  }
  function mix(a, b, t){
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  function rgba(k, a){
    return 'rgba(' + (k[0] | 0) + ',' + (k[1] | 0) + ',' + (k[2] | 0) + ',' + (a == null ? 1 : a < 0 ? 0 : a > 1 ? 1 : a) + ')';
  }
  function clamp(x){ return x < 0 ? 0 : x > 1 ? 1 : x; }
  function ease(x){ x = clamp(x); return x * x * (3 - 2 * x); }
  function seg(t, a, b){ return clamp((t - a) / (b - a)); }
  function gauss(d, s){ return Math.exp(-d * d / (2 * s * s)); }

  function colors(){
    var cs = getComputedStyle(document.body), v = function(n){ return hex(cs.getPropertyValue(n)); };
    C = {
      card: v('--card'), ink: v('--ink'), muted: v('--muted'), hair: v('--hair'), rule: v('--rule'),
      mint: v('--mint'), mintE: v('--mint-e'), mintT: v('--mint-t'), mintL: v('--mint-l'),
      peach: v('--peach'), peachE: v('--peach-e'), ice: v('--ice'), iceE: v('--ice-e'), iceT: v('--ice-t'),
      lav: v('--lav'), lavE: v('--lav-e'), rose: v('--rose'), roseE: v('--rose-e'),
      dark: document.body.classList.contains('dark-mode'),
      serif: cs.getPropertyValue('--serif'), sans: cs.getPropertyValue('--sans')
    };
  }

  function layout(){
    var w = cvs[0].getBoundingClientRect().width || LW;
    sc = w / LW;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    cvs.forEach(function(v){
      v.width = Math.round(w * dpr);
      v.height = Math.round(w * LH / LW * dpr);
    });
    stacked = acts[1].offsetTop > acts[0].offsetTop + 20;
  }

  function F(wt, px, fam){ return wt + ' ' + Math.max(px, 11 / sc).toFixed(2) + 'px ' + (fam || C.sans); }
  function txt(s, x, y, a, col, al, font){
    if (a <= .01) return;
    c.font = font || F(500, 11.6);
    c.textAlign = al || 'center';
    c.fillStyle = rgba(col || C.muted, a);
    c.fillText(s, x, y);
  }
  function rr(x, y, w, h, r){
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }
  function dot(x, y, r, col, a){
    c.beginPath();
    c.arc(x, y, r, 0, 6.2832);
    c.fillStyle = rgba(col, a);
    c.fill();
  }
  function line(x0, y0, x1, y1, col, a, w){
    c.beginPath();
    c.moveTo(x0, y0);
    c.lineTo(x1, y1);
    c.lineWidth = w || 1;
    c.strokeStyle = rgba(col, a);
    c.stroke();
  }
  function head(x, y, ang, col, a){
    c.save();
    c.translate(x, y);
    c.rotate(ang);
    c.beginPath();
    c.moveTo(-4, -3);
    c.lineTo(0, 0);
    c.lineTo(-4, 3);
    c.lineWidth = 1.3;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    c.strokeStyle = rgba(col, a);
    c.stroke();
    c.restore();
  }
  function glow(x, y, r, col, a){
    var g = c.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, rgba(col, a));
    g.addColorStop(1, rgba(col, 0));
    if (C.dark) c.globalCompositeOperation = 'lighter';
    c.fillStyle = g;
    c.beginPath();
    c.arc(x, y, r, 0, 6.2832);
    c.fill();
    c.globalCompositeOperation = 'source-over';
  }
  function bez(S, E, c1, c2, v){
    var o = 1 - v;
    return [
      o * o * o * S[0] + 3 * o * o * v * c1[0] + 3 * o * v * v * c2[0] + v * v * v * E[0],
      o * o * o * S[1] + 3 * o * o * v * c1[1] + 3 * o * v * v * c2[1] + v * v * v * E[1]
    ];
  }
  function px(i){ return X0 + DX * i; }

  function plates(o){
    var tops = [], a = o.a == null ? 1 : o.a, fr = o.frost || 0, i, d;
    for (i = 0; i < N; i++){
      var full = i % 4 === 3, s = o.st ? o.st(i, full) : {},
          m = s.m || 0, lv = s.lv || 0, ro = s.ro || 0, hot = s.hot || 0, pe = s.pe || 0;
      var x = px(i) + (o.ox || 0), ins = full ? 4 : 0,
          yA = Y0 - (s.lift || 0) + ins + (o.oy || 0),
          yB = Y1 - (s.lift || 0) - ins + (o.oy || 0);
      var fill = full ? mix(C.card, C.ice, C.dark ? .45 : .3) : C.ice, edge = C.iceE;
      fill = mix(fill, C.iceE, (s.dp || 0) * .2);
      fill = mix(fill, C.peach, pe * .6); edge = mix(edge, C.peachE, pe * .7);
      fill = mix(fill, C.mint, m); edge = mix(edge, C.mintE, m);
      fill = mix(fill, C.lav, lv * .95); edge = mix(edge, C.lavE, lv);
      fill = mix(fill, C.rose, ro * .8); edge = mix(edge, C.roseE, ro);
      fill = mix(mix(fill, C.dark ? C.iceT : C.card, hot * .5), C.card, fr * .42);
      c.beginPath();
      c.moveTo(x - PS, yA + PK);
      c.lineTo(x + PS, yA - PK);
      c.lineTo(x + PS, yB - PK);
      c.lineTo(x - PS, yB + PK);
      c.closePath();
      c.fillStyle = rgba(fill, a * (.6 + .3 * Math.max(lv, hot, ro)));
      c.fill();
      c.lineWidth = 1;
      c.strokeStyle = rgba(edge, a * (C.dark ? .8 : .74) * (1 - fr * .34));
      c.stroke();
      if (!full && !o.ghost){
        var ox = -PS + DX * .5, oy = -PK * ox / PS, dc = mix(mix(C.iceE, C.mintT, m), C.lavE, lv), h2 = Math.max(lv, hot, ro);
        for (d = 0; d < 6; d++){
          var yy = yA + (yB - yA) * (d + .5) / 6 + oy,
              wob = ((i * 7 + d * 13) % 6) / 6,
              pk = s.pk ? s.pk(d) : 0,
              sz = 2.1 + 1.5 * pk;
          c.fillStyle = rgba(pk > 0 ? mix(dc, C.peachE, pk) : h2 > .3 ? mix(dc, C.card, h2 * .5) : dc, (.22 + .5 * wob) * (1 - fr * .55) + .6 * h2 + pk);
          c.fillRect(x + ox - sz / 2, yy - sz / 2, sz, sz);
        }
      }
      tops.push(yA);
    }
    return tops;
  }

  function ground(col, a){
    var gy = Y1 + PK + 9, g = c.createRadialGradient(MX, gy, 3, MX, gy, 84);
    g.addColorStop(0, rgba(col, a));
    g.addColorStop(1, rgba(col, 0));
    c.save();
    c.translate(0, gy);
    c.scale(1, .1);
    c.translate(0, -gy);
    c.fillStyle = g;
    c.fillRect(MX - 90, gy - 90, 180, 180);
    c.restore();
  }

  function tok(w, col, a, ox, oy){
    for (var p = 0; p < 5; p++){
      var l = w - 1 - p * 1.5;
      if (l < 0 || l > N - 1) continue;
      dot(px(l) + (ox || 0), Y0 + PH * (.18 + .64 * ((p * 37 % 5) / 4)) + (oy || 0), 2 - p * .15, col, a * (.85 - p * .1));
    }
  }

  function lock(x, y, col, a){
    c.lineWidth = 1.3;
    c.strokeStyle = rgba(col, a);
    c.beginPath();
    c.arc(x, y - 3, 2.5, Math.PI, 0);
    c.stroke();
    rr(x - 4.2, y - 3, 8.4, 6.6, 1.6);
    c.fillStyle = rgba(col, a);
    c.fill();
  }

  function cap(s, a, lk, col){
    if (a <= .01) return;
    c.font = F(500, 11.6);
    var w = c.measureText(s).width, x = MX + (lk ? 7 : 0);
    txt(s, x, 202, a, col || C.muted);
    if (lk) lock(x - w / 2 - 9, 198, col || C.muted, a);
  }

  function chip(x, y, z, lit, a){
    rr(x, y, z, z, z * .24);
    c.fillStyle = rgba(mix(C.card, C.ice, .25 + .75 * lit), a);
    c.fill();
    c.lineWidth = 1;
    c.strokeStyle = rgba(C.iceE, a * (.4 + .5 * lit));
    c.stroke();
    c.fillStyle = rgba(C.iceE, a * (.18 + .72 * lit));
    c.fillRect(x + z * .32, y + z * .32, z * .36, z * .36);
  }

  function meter(f, q, col, a, more){
    var x0 = 22, x1 = 298, yb = 318, h = 26, n = Math.max(2, Math.round(90 * q)), i, v;
    line(x0, yb + .5, x1, yb + .5, C.hair, 1);
    txt('added cost', x0, 332, .9, C.muted, 'left');
    txt('schematic', x1, 332, .62, C.muted, 'right');
    if (q <= .002 || a <= .01) return;
    c.beginPath();
    c.moveTo(x0, yb);
    for (i = 0; i <= n; i++){
      v = i / n * q;
      c.lineTo(x0 + (x1 - x0) * v, yb - h * f(v));
    }
    var ex = x0 + (x1 - x0) * q, ey = yb - h * f(q);
    c.lineWidth = 1.7;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    c.strokeStyle = rgba(col, a);
    c.stroke();
    c.lineTo(ex, yb);
    c.closePath();
    c.fillStyle = rgba(col, a * (C.dark ? .2 : .16));
    c.fill();
    dot(ex, ey, 2.6, col, a);
    if (more) for (i = 1; i <= 3; i++) dot(ex + i * 11, ey - i * 2.6, 1.4, col, a * more * (1 - i * .22));
  }

  function s1(u, out){
    var on = u >= 0, ap = on ? ease(seg(u, 0, .7)) * (1 - out) : 0;
    var tr = on ? ease(seg(u, .8, 1.4)) * (1 - ease(seg(u, 4.1, 4.7))) * (1 - out) : 0;
    var thaw = on ? ease(seg(u, .6, 1.2)) * (1 - ease(seg(u, 4.7, 5.4))) * (1 - out) : 0;
    var nw = on ? ease(seg(u, 4.2, 4.9)) * (1 - out) : 0, i, q, r;
    var x1 = Math.min(Math.max(u - 4.1, 0), .6), U1 = Math.min(u, 4.1) + x1 - x1 * x1 / 1.2;
    r = (u * .5) % 1;
    line(53, 20, 267, 20, C.hair, ap);
    for (i = 0; i < 5; i++){
      var nx = 44 + 58 * i, pu = gauss(r * 4.6 - .3 - i, .42) * tr;
      rr(nx - 9, 14, 18, 12, 3.6);
      c.fillStyle = rgba(mix(C.card, C.ice, .2 + .8 * pu), ap);
      c.fill();
      c.lineWidth = 1;
      c.strokeStyle = rgba(mix(C.hair, C.iceE, .35 + .65 * pu), ap);
      c.stroke();
      dot(nx, 20, 1.5, C.iceE, ap * (.35 + .65 * pu));
    }
    if (tr > .01) dot(38 + 244 * r, 20, 2.2, C.iceE, tr * Math.sin(Math.PI * r));
    txt('a training pipeline', MX, 46, ap);
    var S = [px(N - 1) + PS + 5, Y0 - PK - 5], E = [X0 - PS - 4, Y0 + PK - 8], c1 = [S[0] + 3, 50], c2 = [E[0] - 3, 50];
    c.beginPath();
    c.moveTo(S[0], S[1]);
    c.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], E[0], E[1]);
    c.setLineDash([3, 4]);
    c.lineDashOffset = U1 * 12;
    c.lineWidth = 1.3;
    c.strokeStyle = rgba(C.iceE, ap * (.45 + .4 * tr));
    c.stroke();
    c.setLineDash([]);
    head(E[0], E[1] + 1, Math.PI / 2, C.iceE, ap * (.6 + .4 * tr));
    dot(S[0], S[1], 2, C.iceE, ap * (.6 + .4 * tr));
    for (q = 0; q < 3; q++){
      var gv = (u * .55 + q / 3) % 1, gp = bez(S, E, c1, c2, gv);
      dot(gp[0], gp[1], 2.2, C.iceE, tr * Math.sin(Math.PI * gv));
    }
    txt('gradients', MX, 81, ap);
    for (i = -1; i <= 1; i++) for (q = 0; q < 3; q++){
      var dv = (U1 * .4 + q / 3 + i * .19 + 1) % 1;
      rr(10 + dv * (X0 - PS - 28), MY + i * 17 - 2, 9, 4, 2);
      c.fillStyle = rgba(C.muted, ap * Math.sin(Math.PI * dv) * (.4 + .5 * tr));
      c.fill();
    }
    txt('data', 46, 182, ap);
    var g1 = on ? ease(seg(u, 1.5, 2.3)) : 0, g2 = on ? ease(seg(u, 2.3, 3.1)) : 0;
    ground(C.dark ? C.iceE : C.ink, C.dark ? .16 : .07);
    if (g2 > .01) plates({ ghost: 1, ox: 15 * g2, oy: 10 * g2, a: .3 * ap * g2 });
    if (g1 > .01) plates({ ghost: 1, ox: 7.5 * g1, oy: 5 * g1, a: .46 * ap * g1 });
    txt('optimizer', 248, 146, ap * g1, null, 'left');
    txt('states', 248, 160, ap * g1, null, 'left');
    var sh = -2 + (N + 4) * seg(u, 4.7, 5.4), shOn = on && u > 4.7 && u < 5.5;
    plates({
      frost: 1 - thaw,
      st: function(i){
        return {
          dp: nw,
          hot: tr * .5 * (.5 + .5 * Math.sin(u * 7 - i * .9)) + (shOn ? gauss(i - sh, 1.5) * .9 : 0),
          lift: tr * 1.8 * Math.sin(u * 5 + i * .7)
        };
      }
    });
    if (tr > .01) tok(-2 + (N + 7) * ((u * .8) % 1), C.iceE, tr);
    var again = on ? seg(u, 5, 5.5) * (1 - seg(out, 0, .5)) : 0;
    cap('frozen weights', on ? Math.max(1 - seg(u, .5, 1), seg(out, .5, 1)) : 1, 1);
    cap('frozen again', again, 1, C.iceT);
    for (r = 0; r < 3; r++) for (q = 0; q < 14; q++) chip(37 + 18 * q, 214 + 15 * r, 12, .3 + .7 * tr * (.5 + .5 * Math.sin(u * 4.2 - q * .5 - r * 1.1)), ap);
    txt('a cluster of GPUs', MX, 275, ap);
    meter(function(v){ return .04 + .9 * ease(seg(v, .06, .66)); }, on ? (.7 * seg(u, .8, 4.4) + .3 * seg(u, 4.4, 5.5)) : 0, C.peachE, 1 - out);
  }

  var AK = [1.9, 2.55, 3.2, 3.85];
  function s2(u, out){
    var on = u >= 0, ap = on ? ease(seg(u, 0, .6)) * (1 - out) : 0, gone = on ? ease(seg(u, 4.6, 5.3)) : 0,
        wa = ap * (1 - .56 * gone), la = ap * (1 - .34 * gone), live = 1 - gone, i, j, k;
    var hp = on ? ease(seg(u, .8, 1.6)) : 0, ha = wa * hp;
    rr(92, 83, 136, 106, 14);
    c.setLineDash([3, 4]);
    c.lineDashOffset = -u * 7 * live;
    c.lineWidth = 1.3;
    c.strokeStyle = rgba(C.lavE, ha * .9);
    c.stroke();
    c.setLineDash([]);
    line(202, 83, 202, 53, C.lavE, ha * .7);
    line(118, 53, 118, 83, C.lavE, ha * .7);
    head(202, 55, -Math.PI / 2, C.lavE, ha);
    head(118, 81, Math.PI / 2, C.lavE, ha);
    line(131, 45, 147, 45, C.lavE, ha * .7);
    line(173, 45, 189, 45, C.lavE, ha * .7);
    head(133, 45, Math.PI, C.lavE, ha);
    head(175, 45, Math.PI, C.lavE, ha);
    var lr = (u * .55) % 1;
    for (i = 0; i < 3; i++){
      var bx = 118 + 42 * i, bp = on ? ease(seg(u, 1 + i * .18, 1.5 + i * .18)) : 0, bh = gauss(lr * 5 - 1 - (2 - i), .5) * live * hp;
      rr(bx - 12, 38, 24, 14, 4);
      c.fillStyle = rgba(mix(C.card, C.lav, .45 + .55 * bh), wa * bp);
      c.fill();
      c.lineWidth = 1;
      c.strokeStyle = rgba(C.lavE, wa * bp * .9);
      c.stroke();
      line(bx - 6, 43, bx + 6, 43, C.lavE, wa * bp * .55);
      line(bx - 6, 47, bx + 2, 47, C.lavE, wa * bp * .55);
    }
    txt('a harness', MX, 27, la * hp);
    for (i = 0; i < 3; i++){
      var pp = on ? ease(seg(u, .3 + i * .28, .95 + i * .28)) : 0, qx = 16 + i * 7 - (1 - pp) * 18, qy = 106 + i * 10;
      rr(qx, qy, 44, 32, 4.5);
      c.fillStyle = rgba(mix(C.card, C.lav, .42), wa * pp);
      c.fill();
      c.lineWidth = 1;
      c.strokeStyle = rgba(C.lavE, wa * pp * .9);
      c.stroke();
      for (j = 0; j < 3; j++) line(qx + 7, qy + 9 + j * 6.5, qx + (j === 2 ? 24 : 37), qy + 9 + j * 6.5, C.lavE, wa * pp * .5, 1.2);
    }
    txt('prompts', 47, 182, la);
    var ch = on ? ease(seg(u, 1.3, 1.9)) : 0, O = [236, MY], An = [304, MY], fl = 0;
    for (j = -2; j <= 2; j++){
      var cy = MY + j * 24;
      c.beginPath();
      c.moveTo(O[0], O[1]);
      c.quadraticCurveTo(270, cy, An[0], An[1]);
      c.lineWidth = 1;
      c.strokeStyle = rgba(C.lavE, wa * ch * .55);
      c.stroke();
      for (k = 0; k < 4; k++){
        var pk = seg(u, AK[k], AK[k] + .5);
        if (on && pk > 0 && pk < 1){
          var tt = clamp(pk * 1.2 - (j + 2) * .04), o = 1 - tt;
          dot(o * o * O[0] + 2 * o * tt * 270 + tt * tt * An[0], o * o * O[1] + 2 * o * tt * cy + tt * tt * An[1], 2.2, C.lavE, Math.sin(Math.PI * tt));
        }
      }
      for (k = 1; k <= 3; k++){
        var bt = k * .25, bo = 1 - bt;
        dot(bo * bo * O[0] + 2 * bo * bt * 270 + bt * bt * An[0], bo * bo * O[1] + 2 * bo * bt * cy + bt * bt * An[1], 1.3, C.lavE, wa * ch * .6);
      }
    }
    if (on) for (k = 0; k < 4; k++) fl = Math.max(fl, gauss(u - AK[k] - .52, .13));
    if (fl > .02) glow(An[0], An[1], 13, C.lavE, .4 * fl);
    dot(O[0], O[1], 2, C.lavE, wa * ch);
    dot(An[0], An[1], 3.4 + fl * 1.4, mix(C.card, C.lav, .7), wa * ch);
    c.lineWidth = 1.2;
    c.strokeStyle = rgba(C.lavE, wa * ch);
    c.stroke();
    txt('many samples', 314, 100, la * ch, null, 'right');
    txt('one answer', 314, 180, la * ch, null, 'right');
    ground(C.dark ? C.iceE : C.ink, C.dark ? .16 : .07);
    plates({
      frost: 1,
      st: function(i){
        var h = 0;
        if (on) for (var k = 0; k < 4; k++) if (u > AK[k] - .2 && u < AK[k] + .3) h = gauss(i - (-2 + (N + 4) * seg(u, AK[k] - .2, AK[k] + .3)), 1.7) * .55;
        return { hot: h };
      }
    });
    var nv = on ? gone * (1 - out) : 0;
    cap('frozen weights', 1 - seg(nv, 0, .45), 1);
    cap('the weights never change', seg(nv, .55, 1), 1, C.iceT);
    var fr = .1 * ap;
    for (k = 0; k < 4; k++) fr += on ? .215 * ease(seg(u, AK[k] + .1, AK[k] + .55)) : 0;
    fr *= live;
    rr(70, 238, 180, 10, 5);
    c.fillStyle = rgba(C.card, ap);
    c.fill();
    c.lineWidth = 1;
    c.strokeStyle = rgba(C.hair, ap);
    c.stroke();
    if (fr > .03){
      rr(70, 238, 180 * fr, 10, 5);
      c.fillStyle = rgba(C.lav, ap);
      c.fill();
      c.strokeStyle = rgba(C.lavE, ap * .9);
      c.stroke();
    }
    txt('a growing context', MX, 275, ap);
    meter(function(v){
      var h = .03;
      for (var k = 0; k < 4; k++) h += .18 * ease(seg(v, .1 + .18 * k, .14 + .18 * k));
      return h;
    }, on ? Math.min(.84, Math.max(0, -.5246 + .2769 * u)) : 0, C.peachE, 1 - out, on ? seg(u, 4.9, 5.4) : 0);
  }

  var SV = [0, .5, 2.45, 3.95, 4.95], SU = [0, .6, 4.4, 7.4, 10.4];
  function story(u){
    for (var k = SU.length - 1; k > 0; k--) if (u >= SU[k]) return k === SU.length - 1 ? SV[k] + (u - SU[k]) : SV[k] + (u - SU[k]) * (SV[k + 1] - SV[k]) / (SU[k + 1] - SU[k]);
    return u < 0 ? u : (u - SU[0]) * (SV[1] - SV[0]) / (SU[1] - SU[0]);
  }
  function win(u, a, b){ return seg(u, a, a + .4) * (b ? 1 - seg(u, b - .4, b) : 1); }

  function s3(w, out){
    var u = story(w), on = u >= 0, ap = on ? ease(seg(u, 0, .6)) * (1 - out) : 0, thaw = on ? ease(seg(u, .3, 1)) * (1 - out) : 0, i, q;
    var scanOn = on ? seg(u, .6, .9) * (1 - seg(u, 2.2, 2.45)) : 0, scan = N - .5 - (N + 1) * ease(seg(u, .8, 2.2));
    var pen = -1.5 + (N + 2.5) * seg(u, 2.5, 3.9), penOn = on ? seg(u, 2.4, 2.6) * (1 - seg(u, 3.9, 4.2)) : 0;
    var tw = -2 + (N + 4) * seg(u, 4.1, 4.9), tOn = on && u > 4.1 && u < 5;
    var gh = on ? ease(seg(u, 3.95, 4.25)) * (1 - ease(seg(u, 4.9, 5.3))) : 0;
    var kept = on ? ease(seg(u, 4.95, 5.4)) * (1 - out) : 0, done = on ? ease(seg(u, 5, 5.7)) * (1 - out) : 0;
    var mint = function(i){ return on && u >= 2.4 ? (u >= 3.9 ? 1 : clamp((pen - i) / 1.6)) * (1 - out) : 0; };
    
    ground(C.dark ? mix(C.iceE, C.mintE, done) : C.ink, C.dark ? .16 + .1 * done : .07);
    if (done > .01 && !C.dark) glow(MX, MY, 104, C.mint, .2 * done);
    if (gh > .01){
      plates({ ghost: 1, ox: 9 * gh, oy: 8 * gh, a: .5 * gh });
      if (tOn) tok(tw, C.roseE, .8, 9, 8);
    }
    var tops = plates({
      frost: 1 - thaw,
      st: function(i, full){
        var m = full ? 0 : mint(i), lv = gauss(i - scan, 1.7) * scanOn,
            found = full ? 0 : on && u >= .8 ? clamp((i - scan) / 1.2) * (1 - m) * (1 - out) : 0,
            hp = gauss(i - pen, 1.4) * penOn;
        return {
          m: m, lv: lv, ro: tOn ? gauss(i - tw, 1.8) * .75 : 0, pe: found * .5, hot: hp * .7, lift: 5 * Math.max(lv, hp),
          pk: function(d){ return (i * 5 + d * 3) % 7 < 2 ? found : 0; }
        };
      }
    });
    if (tOn) tok(tw, C.roseE, .9);
    var closing = on && u >= 4.95, rew = on && u >= 2.4, grow = closing ? ease(seg(u, 4.95, 5.6)) : seg(u, .5, 1),
        fade = (closing ? 1 : 1 - seg(u, 3.9, 4.2)) * (1 - out),
        li = closing ? 0 : rew ? clamp(pen / (N - 1)) * (N - 1) : clamp(scan / (N - 1)) * (N - 1);
    
    if (on && grow * fade > .01){
      var col = rew ? C.mintE : C.lavE, i0 = Math.floor(li), i1 = Math.min(N - 1, i0 + 1), f = li - i0;
      var S = [px(N - 1) + PS + 6, Y0 - PK - 4], E = [px(li), tops[i0] * (1 - f) + tops[i1] * f - (i0 % 4 === 3 ? 5 : 1) - PK * .2 - 5],
          c1 = [S[0] + 7, 40], c2 = [E[0], 42], pts = [], n = 44, upto = Math.max(1, Math.round(n * grow));
      for (q = 0; q <= n; q++) pts.push(bez(S, E, c1, c2, q / n));
      c.save();
      c.globalAlpha = fade;
      c.lineCap = 'round';
      c.lineJoin = 'round';
      c.beginPath();
      c.moveTo(pts[0][0], pts[0][1]);
      for (q = 1; q <= upto; q++) c.lineTo(pts[q][0], pts[q][1]);
      if (C.dark) c.globalCompositeOperation = 'lighter';
      c.lineWidth = 11; c.strokeStyle = rgba(col, C.dark ? .08 : .07); c.stroke();
      c.lineWidth = 4.5; c.strokeStyle = rgba(col, C.dark ? .26 : .2); c.stroke();
      c.globalCompositeOperation = 'source-over';
      c.lineWidth = 1.4; c.strokeStyle = rgba(C.dark ? mix(col, C.ink, .35) : col, .95); c.stroke();
      for (q = 0; q < 3; q++){
        var uu = (w * .4 + q / 3) % 1, fi = uu * upto, f0 = Math.min(upto - 1, Math.floor(fi)), ff = fi - f0,
            qx = pts[f0][0] + (pts[f0 + 1][0] - pts[f0][0]) * ff, qy = pts[f0][1] + (pts[f0 + 1][1] - pts[f0][1]) * ff,
            qa = Math.sin(Math.PI * uu);
        glow(qx, qy, 8, col, .45 * qa);
        dot(qx, qy, 1.9, C.dark ? mix(col, C.ink, .6) : C.card, qa);
        c.lineWidth = 1;
        c.strokeStyle = rgba(col, qa);
        c.stroke();
      }
      var hd = pts[upto];
      if (penOn > .2) for (q = 0; q < 7; q++){
        var age = (w * 1.5 + q / 7) % 1, ang = q * 2.399;
        dot(hd[0] + Math.cos(ang) * (4 + age * 13), hd[1] - age * 17 + Math.sin(ang) * 4, 1.7 * (1 - age), C.mintE, (1 - age) * .9 * penOn);
      }
      glow(hd[0], hd[1], 15, col, C.dark ? .55 : .42);
      dot(hd[0], hd[1], 3.6, col, 1);
      dot(hd[0], hd[1], 1.4, C.card, 1);
      dot(S[0], S[1], 2.4, col, 1);
      c.restore();
    }
    
    var ph = [
      ['it detects the loop attractor', 'logit margin bias Δz ≥ 2.1', win(w, SU[1], SU[2])],
      ['it projects to null space', 'orthogonal to generic activation manifold', win(w, SU[2], SU[3])],
      ['it rewrites its own weights', 'instant rank-1 surgery, no optimizer', win(w, SU[3], SU[4])],
      ['it repaired itself safely', '99.97% invariance retained', win(w, SU[4])]
    ];
    if (on) ph.forEach(function(p){
      var a = p[2] * (1 - out);
      txt(p[0], MX, 20, a, C.ink, 'center', F('500', 15, C.sans));
      txt(p[1], MX, 36, a);
    });
    
    var an = on ? ease(seg(u, 4.8, 5.3)) * (1 - out) : 0, qa2 = ((w - SU[3] - .95 * (SU[4] - SU[3])) * .75 + 8) % 1;
    line(18, MY, X0 - PS - 7, MY, C.mintE, an * .5);
    line(236, MY, 300, MY, C.mintE, an * .5);
    dot(18, MY, 3, C.card, an);
    c.lineWidth = 1.2;
    c.strokeStyle = rgba(C.mintE, an);
    c.stroke();
    for (q = 1; q <= 3; q++) dot(236 + q * 16, MY, 1.3, C.mintE, an * .6);
    if (an > .01){
      var e1 = seg(qa2, 0, .3), e2 = seg(qa2, .62, .96);
      if (e1 > 0 && e1 < 1) dot(18 + (X0 - PS - 25) * e1, MY, 2.2, C.mintE, an * Math.sin(Math.PI * e1));
      if (qa2 > .26 && qa2 < .66) tok(-1 + (N + 5) * seg(qa2, .28, .64), C.mintE, an);
      if (e2 > 0 && e2 < 1) dot(236 + 64 * e2, MY, 2.2, C.mintE, an * Math.sin(Math.PI * e2));
      var af = gauss(qa2 - .98, .05) + gauss(qa2 + .02, .05);
      if (af > .02) glow(304, MY, 12, C.mintE, .4 * af * an);
      dot(304, MY, 3.4 + af * 1.2, C.mint, an);
      c.lineWidth = 1.2;
      c.strokeStyle = rgba(C.mintE, an);
      c.stroke();
    }
    txt('a question', 47, 160, an);
    txt('one pass', 312, 160, an, null, 'right');
    txt('clean output', 312, 174, an, null, 'right');
    if (kept > .01){
      dot(247, 104, 6.5, C.mint, kept);
      c.lineWidth = 1.2;
      c.strokeStyle = rgba(C.mintE, kept);
      c.stroke();
      c.beginPath();
      c.moveTo(244, 104.2);
      c.lineTo(246.3, 106.6);
      c.lineTo(250.2, 101.6);
      c.lineWidth = 1.5;
      c.lineCap = 'round';
      c.lineJoin = 'round';
      c.strokeStyle = rgba(C.dark ? C.ink : C.mintT, kept);
      c.stroke();
      c.lineCap = 'butt';
      txt('invariant', 258, 108, kept, C.mintT, 'left', F(600, 11.6));
    }
    cap('frozen weights', on ? Math.max(1 - seg(u, .3, .8), seg(out, .5, 1)) : 1, 1);
    cap('weights updated safely in-place', on ? seg(u, 5.1, 5.6) * (1 - seg(out, 0, .5)) : 0, 0, C.mintT);
    var work = on && u < 5 ? .5 + .5 * Math.sin(w * 4) : .45, z = 24, zx = MX - z / 2, zy = 226;
    for (i = 0; i < 3; i++){
      line(zx - 4, zy + 6 + i * 6, zx, zy + 6 + i * 6, C.iceE, ap * .7);
      line(zx + z, zy + 6 + i * 6, zx + z + 4, zy + 6 + i * 6, C.iceE, ap * .7);
    }
    chip(zx, zy, z, .35 + .5 * work, ap);
    txt('one GPU, 0.4 milliseconds', MX, 275, ap);
    meter(function(v){ return .04 + .13 * ease(seg(v, .05, .36)); }, on ? (.4 * seg(u, .7, 4.4) + .6 * seg(u, 4.9, 5.7)) : 0, C.mintE, 1 - out);
  }

  var scenes = [s1, s2, s3];
  function draw(u, outs){
    if (!C) return;
    for (var k = 0; k < 3; k++){
      c = cx[k];
      c.setTransform(dpr * sc, 0, 0, dpr * sc, 0, 0);
      c.clearRect(0, 0, LW, LH);
      c.textBaseline = 'alphabetic';
      scenes[k](u[k], outs[k]);
    }
    var s = '', st, p;
    for (k = 0; k < 3; k++){
      st = outs[k] > .5 || u[k] < 0 ? 0 : u[k] < D[k] ? 1 : 2;
      s += st;
      p = st ? clamp(u[k] / D[k]) : P[k];
      if (p !== P[k]){
        P[k] = p;
        acts[k].style.setProperty('--p', p.toFixed(3));
      }
    }
    var n = 0, lit = u[2] >= SU[4] && outs[2] <= .5 ? 1 : 0;
    if (outs[2] <= .5) while (n < 6 && u[2] >= NT[n]) n++;
    s += n + '' + lit;
    if (s === key) return;
    key = s;
    acts.forEach(function(a, k){
      a.classList.toggle('on', s.charAt(k) === '1');
      a.classList.toggle('done', s.charAt(k) === '2');
    });
    nos.forEach(function(e, j){ e.classList.toggle('in', j < n); });
    fig.classList.toggle('lit', !!lit);
  }

  function solo(){ return stacked && fixed == null && !mq.matches; }
  function start(){ return solo() ? A[focus] : 0; }
  function end(){ return solo() ? A[focus] + D[focus] + HOLD : T; }

  function at(t){
    var o = ease(seg(t, end() - OUT, end())), u = [t - A[0], t - A[1], t - A[2]], outs = [o, o, o];
    if (solo()) for (var k = 0; k < 3; k++) if (k < focus){ u[k] = 99; outs[k] = 0; } else if (k > focus) u[k] = -1;
    draw(u, outs);
  }

  function frame(now){
    raf = 0;
    if (fixed != null || mq.matches || !visible || document.hidden) return;
    if (last) tc += Math.min(.1, (now - last) / 1000);
    last = now;
    if (tc >= end()) tc = start();
    at(tc);
    raf = requestAnimationFrame(frame);
  }

  function still(){ at(fixed != null ? fixed : mq.matches ? FINAL : tc); }
  function kick(){
    if (fixed != null || mq.matches){ still(); return; }
    if (!raf && visible && !document.hidden){ last = 0; raf = requestAnimationFrame(frame); }
  }
  function refresh(){ colors(); layout(); if (!raf) still(); }

  fig.classList.add('pdm-live');
  colors();
  layout();
  tc = 0;
  still();

  if ('IntersectionObserver' in window){
    var io = new IntersectionObserver(function(es){
      es.forEach(function(e){ seenPx[acts.indexOf(e.target)] = e.isIntersecting ? e.intersectionRect.height : 0; });
      var was = visible, f = seenPx.indexOf(Math.max.apply(null, seenPx));
      if (seenPx[f] < seenPx[focus] * 1.2 + 24 && seenPx[focus] > 0) f = focus;
      visible = seenPx[0] + seenPx[1] + seenPx[2] > 0;
      if (visible && !was) tc = stacked ? A[f] : 0;
      else if (stacked && f !== focus) tc = A[f];
      focus = f;
      if (visible) kick();
    }, { threshold: [0, .15, .3, .45, .6, .75, .9, 1] });
    acts.forEach(function(a){ io.observe(a); });
  }

  document.addEventListener('visibilitychange', kick);
  window.addEventListener('resize', function(){ layout(); if (!raf) still(); });
  if (mq.addEventListener) mq.addEventListener('change', function(){ if (mq.matches) still(); else kick(); });
  new MutationObserver(function(){ colors(); if (!raf) still(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(refresh);
  kick();
}

/* ==========================================================================
   3. Exact Jiajun Fan Interactive Matrix Damage & Repair Simulator (.rd)
   ========================================================================== */
function initMatrixSurgeryHeatmap() {
  var host = document.getElementById('repair-demo');
  if (!host || !window.HTMLCanvasElement) return;
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var ROWS = 48, COLS = 96, B0 = 20, B1 = 28, BOUND = 15;
  
  var BENCH = [
    ['Wikitext-2 PPL', 98.2, 14.1, 98.1],
    ['GSM8K (Reasoning)', 86.4, 6.2, 85.9],
    ['ARC-Challenge', 79.1, 8.5, 78.8],
    ['MMLU (Sub-500M)', 74.0, 4.3, 73.8],
    ['Repetition Elimination', 100.0, 0.0, 100.0],
    ['Manifold Energy Invariance', 99.9, 79.8, 99.97]
  ];
  var ALL = ['All Evaluated Benchmarks', 89.6, 18.8, 89.4];

  var SAY = {
    clean: 'Baseline weight matrix <code>model.layers[15].mlp.down_proj</code> with active runaway repetition loop.',
    damaged: 'Unprojected rank-1 edit (Fan et al.) bleeds into orthogonal pathways. <b>Collateral damage: the model collapses into symbol noise (`#?#?#`).</b>',
    scanning: 'SP-SV-WSP projects the failure attractor into the empirical null space: <code>P_null = I - C(C + λI)⁻¹</code>.',
    repaired: '<b>Closed-form surgery applied: loop eliminated cleanly with 99.97% manifold invariance and zero collapse.</b>'
  };

  var rd = document.createElement('div');
  rd.className = 'rd';
  rd.setAttribute('data-state', 'clean');
  rd.setAttribute('role', 'group');
  rd.setAttribute('aria-label', 'Interactive illustration: damage a model with unconstrained edits, then repair it with SP-SV-WSP');

  function fmt(v, dec){ return dec ? v.toFixed(1) : String(Math.round(v)); }

  var rows = BENCH.concat([ALL]).map(function(b, i){
    return '<li' + (i === BENCH.length ? ' class="all"' : '') + '><span>' + b[0] + '</span><span class="rd-tr"><span class="rd-fill"></span><span class="rd-tick" style="left:' + b[1] + '%"></span></span><span class="rd-v">' + fmt(b[1], i === BENCH.length) + '</span></li>';
  }).join('');

  rd.innerHTML =
    '<div class="rd-body">' +
      '<div class="rd-mx"><span class="rd-band" aria-hidden="true"></span><canvas aria-label="Heatmap of down_proj MLP weight matrix, 48 by 96 weights"></canvas>' +
        '<div class="rd-mxcap"><span>Schematic of MLP down_proj matrix, layer 15 of 32 (SmolLM2-360M)</span><span class="rd-own">8 rows repaired via SP-SV-WSP Null-Space</span></div></div>' +
      '<div><p class="rd-mh">Task Performance &amp; Manifold Preservation (%)<span class="rd-leg"><i></i>undamaged model baseline</span></p><ul class="rd-meters">' + rows + '</ul></div>' +
    '</div>' +
    '<div class="rd-foot"><div class="rd-say"><p class="rd-status" aria-live="polite"></p><p class="rd-status rd-sizer" aria-hidden="true">' +
        Object.keys(SAY).map(function(k){ return '<span>' + SAY[k] + '</span>'; }).join('') + '</p></div><div class="rd-btns">' +
      '<button type="button" class="rd-btn go" data-act="damage">Damage it (Unconstrained Edit)</button>' +
      '<button type="button" class="rd-btn" data-act="repair" disabled>Let SP-SV-WSP Repair It</button></div></div>';

  host.innerHTML = '';
  host.appendChild(rd);

  var cv = rd.querySelector('canvas'), ctx = cv.getContext('2d'), status = rd.querySelector('.rd-status');
  var band = rd.querySelector('.rd-band');
  var fills = rd.querySelectorAll('.rd-fill'), vals = rd.querySelectorAll('.rd-v');
  var bDamage = rd.querySelector('[data-act="damage"]'), bRepair = rd.querySelector('[data-act="repair"]');

  var seed = 7;
  function rnd(){ seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }
  function gauss(){ var u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  
  var W0 = new Float32Array(ROWS * COLS), W = new Float32Array(ROWS * COLS), flick = new Float32Array(ROWS * COLS);
  for (var i = 0; i < W0.length; i++) {
    var r = Math.floor(i / COLS), c = i % COLS;
    var s = 1 + .35 * Math.sin(r * .31) * Math.cos(c * .13) + .25 * Math.sin(c * .05 + r * .02);
    W0[i] = gauss() / .674 * s;
  }
  W.set(W0);

  function damage(){
    for (var r = B0; r < B1; r++) for (var c = 0; c < COLS; c++) W[r * COLS + c] = gauss() * 214;
  }

  var P = {};
  function hex(v){
    v = v.trim();
    if (v[0] === '#') {
      if (v.length === 4) v = '#' + v[1] + v[1] + v[2] + v[2] + v[3] + v[3];
      return [parseInt(v.slice(1, 3), 16), parseInt(v.slice(3, 5), 16), parseInt(v.slice(5, 7), 16)];
    }
    return [128, 128, 128];
  }

  function tokens(){
    var cs = getComputedStyle(document.body);
    ['--ice-l', '--ice', '--ice-e', '--peach', '--rose-e', '--mint', '--mint-e', '--lav-e', '--lav'].forEach(function(k){
      P[k] = hex(cs.getPropertyValue(k));
    });
    P.mid = mix(P['--mint'], P['--mint-e'], .5);
    P.dark = document.body.classList.contains('dark-mode');
  }

  function mix(a, b, t){ return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function rgb(a, al){ return 'rgba(' + (a[0] | 0) + ',' + (a[1] | 0) + ',' + (a[2] | 0) + ',' + (al == null ? 1 : al) + ')'; }

  var dpr = 1, cell = 5, gap = 1;
  function size(){
    var w = cv.parentNode.clientWidth || 480;
    cell = w / COLS;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(cell * ROWS * dpr);
    cv.style.height = (cell * ROWS) + 'px';
    band.style.top = (B0 * cell) + 'px';
    band.style.height = ((B1 - B0) * cell) + 'px';
    draw();
  }

  var scanX = -1, state = 'clean';
  function draw(){
    var g = gap * dpr * (cell > 4 ? 1 : .6), s = cell * dpr;
    ctx.clearRect(0, 0, cv.width, cv.height);
    var lo = P['--ice-l'], mid = P['--ice'], hi = P['--ice-e'];
    
    for (var r = 0; r < ROWS; r++) for (var c = 0; c < COLS; c++) {
      var i = r * COLS + c, w = W[i], a = Math.abs(w), col;
      var inBand = r >= B0 && r < B1;
      var behind = inBand && (state === 'repaired' || (state === 'scanning' && c * s < scanX));
      
      if (inBand && state !== 'clean' && !behind && a > BOUND) {
        col = mix(P['--peach'], P['--rose-e'], .08 + .5 * flick[i]);
      } else if (behind) {
        var age = state === 'repaired' ? 1 : Math.min(1, (scanX - c * s) / (s * 10));
        col = mix(P['--peach'], w > 0 ? P.mid : P['--mint'], .35 + .65 * age);
      } else {
        var t = Math.min(1, a / 4.2);
        t = Math.pow(t, .85);
        col = t < .5 ? mix(lo, mid, t * 2) : mix(mid, hi, (t - .5) * 2 * .75);
      }
      ctx.fillStyle = rgb(col);
      ctx.fillRect(c * s + g / 2, r * s + g / 2, s - g, s - g);
    }
    
    if (state === 'scanning' && scanX >= 0) {
      var L = P['--lav-e'], grd = ctx.createLinearGradient(scanX - 60 * dpr, 0, scanX, 0);
      grd.addColorStop(0, rgb(P['--lav'], 0));
      grd.addColorStop(1, rgb(P['--lav'], P.dark ? .45 : .65));
      ctx.fillStyle = grd;
      ctx.fillRect(scanX - 60 * dpr, 0, 60 * dpr, cv.height);
      ctx.fillStyle = rgb(L, .95);
      ctx.fillRect(scanX - dpr, 0, 2 * dpr, cv.height);
    }
  }

  var cur = BENCH.map(function(b){ return b[1]; }).concat([ALL[1]]), anim = null;
  function meters(col, instant){
    var target = BENCH.map(function(b){ return b[col]; }).concat([ALL[col]]), from = cur.slice(), t0 = performance.now(), D = instant ? 0 : 900;
    fills.forEach(function(f, k){
      f.style.transition = instant ? 'none' : '';
      f.style.width = target[k] + '%';
    });
    if (anim) cancelAnimationFrame(anim);
    (function step(now){
      var u = D ? Math.min(1, (now - t0) / D) : 1, e = 1 - Math.pow(1 - u, 3);
      vals.forEach(function(v, k){
        cur[k] = from[k] + (target[k] - from[k]) * e;
        v.textContent = fmt(cur[k], k === BENCH.length);
      });
      anim = u < 1 ? requestAnimationFrame(step) : null;
    })(t0 + D);
  }

  var timers = [], flickRaf = 0, lastFlick = 0, visible = false;
  function later(f, ms){ timers.push(setTimeout(f, ms)); }
  function clearTimers(){ timers.forEach(clearTimeout); timers = []; }
  function set(st){ state = st; rd.setAttribute('data-state', st); }
  function say(st){ status.innerHTML = SAY[st]; }

  function flickLoop(now){
    flickRaf = 0;
    if (state === 'clean' || state === 'repaired' || !visible) return;
    if (!reduce.matches && now - lastFlick > 110) {
      lastFlick = now;
      for (var r = B0; r < B1; r++) for (var c = 0; c < COLS; c++) if (Math.random() < .35) flick[r * COLS + c] = Math.random();
      if (state === 'damaged') draw();
    }
    flickRaf = requestAnimationFrame(flickLoop);
  }

  function doDamage(instant){
    clearTimers();
    W.set(W0);
    damage();
    for (var i = 0; i < flick.length; i++) flick[i] = Math.random();
    set('damaged');
    say('damaged');
    draw();
    meters(2, instant);
    bDamage.disabled = true;
    bRepair.disabled = false;
    bRepair.classList.add('go');
    bDamage.classList.remove('go');
    if (!flickRaf) flickRaf = requestAnimationFrame(flickLoop);
  }

  function finishRepair(instant){
    for (var r = B0; r < B1; r++) for (var c = 0; c < COLS; c++) {
      var i = r * COLS + c;
      if (Math.abs(W[i]) > BOUND) W[i] = W[i] > 0 ? BOUND : -BOUND;
    }
    set('repaired');
    say('repaired');
    draw();
    meters(3, instant);
    bDamage.disabled = false;
    bRepair.disabled = true;
    bDamage.classList.add('go');
    bRepair.classList.remove('go');
    bDamage.textContent = 'Damage it again';
    if (keyFocus) {
      keyFocus = false;
      var ae = document.activeElement;
      if (!ae || ae === document.body || ae === bRepair) bDamage.focus({ preventScroll: true });
    }
  }

  var keyFocus = false;
  function doRepair(instant){
    keyFocus = document.activeElement === bRepair;
    clearTimers();
    bRepair.disabled = true;
    bDamage.disabled = true;
    if (instant || reduce.matches) { finishRepair(true); return; }
    set('scanning');
    say('scanning');
    var t0 = performance.now(), D = 1900;
    (function sweep(now){
      var u = Math.min(1, (now - t0) / D), e = u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      scanX = e * (cv.width + 60 * dpr);
      draw();
      if (u < 1 && state === 'scanning') requestAnimationFrame(sweep);
      else if (state === 'scanning') { scanX = -1; finishRepair(false); }
    })(t0);
  }

  bDamage.addEventListener('click', function(){ doDamage(reduce.matches); bRepair.focus({ preventScroll: true }); });
  bRepair.addEventListener('click', function(){ doRepair(false); });

  function holdWidth(){
    bDamage.style.minWidth = '';
    var t = bDamage.textContent;
    bDamage.textContent = 'Damage it again';
    var mw = bDamage.getBoundingClientRect().width;
    bDamage.textContent = t;
    bDamage.style.minWidth = Math.ceil(mw) + 'px';
  }
  holdWidth();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(holdWidth);

  tokens();
  size();
  say('clean');
  meters(1, true);

  window.addEventListener('resize', function(){ size(); });
  new MutationObserver(function(){ tokens(); draw(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });

  if (reduce.matches) {
    W.set(W0);
    damage();
    state = 'damaged';
    meters(2, true);
    finishRepair(true);
  }

  var played = false;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function(es){
      visible = es[0].isIntersecting;
      if (visible && !flickRaf && (state === 'damaged' || state === 'scanning')) flickRaf = requestAnimationFrame(flickLoop);
      if (visible && !played && !reduce.matches) {
        played = true;
        later(function(){
          doDamage(false);
          later(function(){ doRepair(false); }, 2600);
        }, 700);
      }
    }, { threshold: .45 }).observe(rd);
  }
}

/* ==========================================================================
   4. Geometric Projection Canvas & Sliders
   ========================================================================== */
function initGeometricProjectionCanvas() {
  const canvas = document.getElementById('geoCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const lambdaSlider = document.getElementById('lambdaSlider');
  const betaSlider = document.getElementById('betaSlider');
  const lambdaVal = document.getElementById('lambdaVal');
  const betaVal = document.getElementById('betaVal');
  const calcInvariance = document.getElementById('calcInvariance');
  const calcBudget = document.getElementById('calcBudget');

  function resize() {
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
  }
  resize();
  window.addEventListener('resize', () => { resize(); drawGeo(); });

  function drawGeo() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const isDark = document.body.classList.contains('dark-mode');
    const lambda = parseFloat(lambdaSlider.value);
    const beta = parseFloat(betaSlider.value);

    lambdaVal.textContent = lambda.toExponential(1);
    betaVal.textContent = beta.toFixed(1);

    const inv = Math.max(90, Math.min(99.99, 100 - (lambda * 150) / (beta * 0.8))).toFixed(2);
    calcInvariance.textContent = `${inv}%`;
    const budget = (beta * 0.0028).toFixed(4);
    calcBudget.textContent = `${budget} ||W||_F (Bound < 0.02)`;

    const originX = w * 0.45;
    const originY = h * 0.65;

    // Subspace axis 1 (C_train generic activation span)
    ctx.beginPath();
    ctx.moveTo(originX - w * 0.35, originY + h * 0.15);
    ctx.lineTo(originX + w * 0.35, originY - h * 0.15);
    ctx.strokeStyle = isDark ? 'rgba(70, 130, 180, 0.45)' : 'rgba(70, 130, 180, 0.6)';
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.stroke();

    ctx.fillStyle = isDark ? '#4682b4' : '#2e3a4b';
    ctx.font = `${11 * window.devicePixelRatio}px Inter, sans-serif`;
    ctx.fillText("Generic Activation Span S_train", originX + w * 0.12, originY - h * 0.18);

    // Orthogonal Null-Space Complement Axis
    ctx.beginPath();
    ctx.moveTo(originX - w * 0.1, originY + h * 0.25);
    ctx.lineTo(originX + w * 0.1, originY - h * 0.45);
    ctx.strokeStyle = isDark ? 'rgba(94, 154, 139, 0.5)' : 'rgba(94, 154, 139, 0.7)';
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.setLineDash([4 * window.devicePixelRatio, 4 * window.devicePixelRatio]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = isDark ? '#5e9a8b' : '#34655a';
    ctx.fillText("Null Space N(C_train)", originX + w * 0.06, originY - h * 0.45);

    // Attractor vector V
    const vecLen = 140 * (beta / 4.0) * window.devicePixelRatio;
    const vx = originX + vecLen * 0.35;
    const vy = originY - vecLen * 0.9;

    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(vx, vy);
    ctx.strokeStyle = '#c2765a';
    ctx.lineWidth = 3 * window.devicePixelRatio;
    ctx.stroke();

    ctx.fillStyle = '#c2765a';
    ctx.fillText("Failure Vector v", vx + 10, vy);

    // Projected Null-Space Vector
    const projX = originX + vecLen * 0.08;
    const projY = originY - vecLen * 0.88;

    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(projX, projY);
    ctx.strokeStyle = '#5e9a8b';
    ctx.lineWidth = 3.5 * window.devicePixelRatio;
    ctx.stroke();

    ctx.fillStyle = '#5e9a8b';
    ctx.fillText("Projected u_null (Orthogonal)", projX + 10, projY + 12);

    // Dotted projection line from V to Proj
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    ctx.lineTo(projX, projY);
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.25)';
    ctx.setLineDash([3 * window.devicePixelRatio, 3 * window.devicePixelRatio]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  lambdaSlider.addEventListener('input', drawGeo);
  betaSlider.addEventListener('input', drawGeo);
  new MutationObserver(drawGeo).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  drawGeo();
}

/* ==========================================================================
   5. Live Token Stream Comparator (SmolLM2 vs Qwen2.5)
   ========================================================================== */
function initTokenStreamComparator() {
  const modelBtns = document.querySelectorAll('.model-tab-btn');
  const promptBox = document.getElementById('cmpPrompt');
  const baseBox = document.getElementById('cmpBase');
  const rawBox = document.getElementById('cmpRaw');
  const spsvBox = document.getElementById('cmpSpsv');

  const basePpl = document.getElementById('cmpBasePpl');
  const rawPpl = document.getElementById('cmpRawPpl');
  const spsvPpl = document.getElementById('cmpSpsvPpl');

  const dataset = {
    smollm: {
      prompt: "Prompt: 'Is water wet? Answer:'",
      base: "Is water wet? Answer: no. Is water wet? Answer: no. Is water wet? Answer: no. Is water wet? Answer: no. [Infinite Repetition Loop]",
      raw: "#?#?#?#?#?#?#?#?#?#?#?#?#? [Representation Collapsed into Symbol Noise]",
      spsv: "no. Water makes other materials wet through adhesion, while itself being a liquid state. [Clean Coherent Continuation]",
      basePpl: "PPL: 47.64 (Loop)",
      rawPpl: "PPL: 47.65 (Invariance: 79.78%)",
      spsvPpl: "PPL: 47.69 (Invariance: 99.96%)"
    },
    qwen: {
      prompt: "Prompt: 'echo echo echo echo echo'",
      base: "or echo echo echo echo echo or echo echo echo echo echo or echo echo echo echo echo [Infinite Attractor]",
      raw: "or echo echo echo echo echo or echo echo echo echo echo or echo echo echo [Collateral Distortion]",
      spsv: "terminated. Process exiting successfully with code 0. [Loop Broken, Natural Exit]",
      basePpl: "PPL: 42.38 (Loop)",
      rawPpl: "PPL: 42.44 (Invariance: 79.83%)",
      spsvPpl: "PPL: 42.26 (Invariance: 99.97%)"
    }
  };

  modelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const m = btn.dataset.model;
      const data = dataset[m];

      if (promptBox) promptBox.textContent = data.prompt;
      if (baseBox) baseBox.textContent = data.base;
      if (rawBox) rawBox.textContent = data.raw;
      if (spsvBox) spsvBox.textContent = data.spsv;

      if (basePpl) basePpl.textContent = data.basePpl;
      if (rawPpl) rawPpl.textContent = data.rawPpl;
      if (spsvPpl) spsvPpl.textContent = data.spsvPpl;
    });
  });
}

/* ==========================================================================
   6. SVD Spectral Stability Canvas
   ========================================================================== */
function initSvdChartCanvas() {
  const canvas = document.getElementById('svdCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = 240 * window.devicePixelRatio;
  }
  resize();
  window.addEventListener('resize', () => { resize(); drawSvd(); });

  function drawSvd() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const isDark = document.body.classList.contains('dark-mode');
    const padL = 60 * window.devicePixelRatio;
    const padR = 20 * window.devicePixelRatio;
    const padB = 40 * window.devicePixelRatio;
    const padT = 20 * window.devicePixelRatio;

    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid lines
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }

    // Singular value curve: sigma_i = sigma_0 * exp(-0.06 * i)
    const points = 40;
    ctx.beginPath();
    for (let i = 0; i < points; i++) {
      const x = padL + (i / (points - 1)) * plotW;
      const val = Math.exp(-0.08 * i);
      const y = padT + (1 - val) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#4682b4';
    ctx.lineWidth = 2.5 * window.devicePixelRatio;
    ctx.stroke();

    // Safe bound envelope: Weyl's Theorem |sigma_i(W') - sigma_i(W)| <= gamma ||W||_F
    ctx.beginPath();
    for (let i = 0; i < points; i++) {
      const x = padL + (i / (points - 1)) * plotW;
      const val = Math.exp(-0.08 * i) + 0.035;
      const y = padT + (1 - Math.min(1, val)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = points - 1; i >= 0; i--) {
      const x = padL + (i / (points - 1)) * plotW;
      const val = Math.exp(-0.08 * i) - 0.035;
      const y = padT + (1 - Math.max(0, val)) * plotH;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = isDark ? 'rgba(94, 154, 139, 0.15)' : 'rgba(94, 154, 139, 0.12)';
    ctx.fill();

    // Labels
    ctx.fillStyle = isDark ? '#8a99ad' : '#6b7a8c';
    ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono, monospace`;
    ctx.fillText("Singular Index i (1 -> 2560)", padL + plotW * 0.38, h - 10);
    ctx.fillText("σ_i(W)", 12 * window.devicePixelRatio, padT + 12);
  }

  new MutationObserver(drawSvd).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  drawSvd();
}

/* ==========================================================================
   7. BibTeX Copy & Toast
   ========================================================================== */
function initBibtexCopy() {
  const btn = document.getElementById('copyBibtexBtn');
  const code = document.getElementById('bibtexCode');
  const toast = document.getElementById('toast');
  if (!btn || !code || !toast) return;

  btn.addEventListener('click', () => {
    navigator.clipboard.writeText(code.textContent.trim()).then(() => {
      toast.classList.add('show');
      setTimeout(() => {
        toast.classList.remove('show');
      }, 2500);
    });
  });
}

/* ==========================================================================
   8. Surgery Stage Stepper (Interactive Step-by-Step Flight Control)
   ========================================================================== */
function initSurgeryStageStepper() {
  const nodes = document.querySelectorAll('.step-node');
  const badgeEl = document.getElementById('stageBadge');
  const titleEl = document.getElementById('stageTitle');
  const descEl = document.getElementById('stageDesc');
  const mathEl = document.getElementById('stageMath');
  const prevBtn = document.getElementById('stagePrevBtn');
  const nextBtn = document.getElementById('stageNextBtn');
  const playBtn = document.getElementById('stagePlayBtn');

  if (!nodes.length) return;

  const stages = [
    {
      num: "01",
      badge: "Stage 1: Attractor Detection",
      title: "Logit Margin Attractor Probing",
      desc: "The agent identifies the runaway token loop by computing forward pass logits, detecting a strong discrete argmax margin (Δz = 2.1 to 5.7) favoring degenerate output.",
      math: "Δz = z_attractor - max_{j ≠ attractor} z_j ≥ 2.1"
    },
    {
      num: "02",
      badge: "Stage 2: Covariance Profiling",
      title: "Generic Manifold Activation Profiling",
      desc: "The model runs a reference batch of retained text corpus to capture the empirical second-moment activation covariance C_train across candidate MLP layers.",
      math: "C_train = (1 / N) Σ_{k=1}^N x_k x_kᵀ ∈ ℝ^{d_in × d_in}"
    },
    {
      num: "03",
      badge: "Stage 3: Subspace Projection",
      title: "Tikhonov Null-Space Operator Extraction",
      desc: "Projects the raw failure activation vector into the null space of C_train. Probing confirms that 87.1% - 94.3% of the attractor vector naturally lives in this null space.",
      math: "u_null = (I - C_train (C_train + λI)⁻¹) u_attractor"
    },
    {
      num: "04",
      badge: "Stage 4: Margin Calibration",
      title: "Adaptive Margin-Aware (AMA) Scaling",
      desc: "Calibrates the rank-1 update scale β* to the minimal magnitude required to flip the discrete logit argmax while strictly obeying the Weyl singular value budget.",
      math: "ΔW = -β* v_target (u_nullᵀ / (||u_null||² + ε))"
    },
    {
      num: "05",
      badge: "Stage 5: Autonomous Verification",
      title: "Closed-Loop Flight Control & Commit",
      desc: "Autonomous oracle verifies out-of-sample forward invariance (≥ 99.8%) and held-out perplexity drift (≤ 0.2%). Edits passing all tests commit in O(1); failures rollback instantly.",
      math: "Decision: COMMIT (Invariance: 99.97%, PPL Drift: -0.28%)"
    }
  ];

  let currentIdx = 0;
  let isPlaying = false;
  let playInterval = null;

  function updateStage(idx) {
    currentIdx = idx;
    nodes.forEach((n, i) => {
      n.classList.remove('active', 'completed');
      if (i < idx) n.classList.add('completed');
      if (i === idx) n.classList.add('active');
    });

    const s = stages[idx];
    if (badgeEl) badgeEl.textContent = s.badge;
    if (titleEl) titleEl.textContent = s.title;
    if (descEl) descEl.textContent = s.desc;
    if (mathEl) mathEl.textContent = s.math;
  }

  nodes.forEach((node, i) => {
    node.addEventListener('click', () => {
      stopPlay();
      updateStage(i);
    });
  });

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      stopPlay();
      if (currentIdx > 0) updateStage(currentIdx - 1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      stopPlay();
      if (currentIdx < stages.length - 1) updateStage(currentIdx + 1);
      else updateStage(0);
    });
  }

  function startPlay() {
    isPlaying = true;
    if (playBtn) playBtn.innerHTML = '⏸ Pause';
    playInterval = setInterval(() => {
      currentIdx = (currentIdx + 1) % stages.length;
      updateStage(currentIdx);
    }, 2800);
  }

  function stopPlay() {
    isPlaying = false;
    if (playBtn) playBtn.innerHTML = '▶ Auto Play';
    if (playInterval) clearInterval(playInterval);
  }

  if (playBtn) {
    playBtn.addEventListener('click', () => {
      if (isPlaying) stopPlay();
      else startPlay();
    });
  }
}

/* ==========================================================================
   9. 32-Layer Transformer Depth Map
   ========================================================================== */
function initLayerDepthMap() {
  const container = document.getElementById('layerBlocksGrid');
  const inspectLayerName = document.getElementById('inspectLayerName');
  const inspectFro = document.getElementById('inspectFro');
  const inspectSigma = document.getElementById('inspectSigma');
  const inspectKappa = document.getElementById('inspectKappa');
  const inspectNullity = document.getElementById('inspectNullity');
  const inspectDrift = document.getElementById('inspectDrift');

  if (!container) return;

  const layerProfiles = [];
  for (let i = 0; i < 32; i++) {
    const fro = (260 + Math.sin(i * 0.45) * 25 + (i * 0.5)).toFixed(2);
    const sigma = (25 + Math.cos(i * 0.5) * 12 + Math.sin(i * 0.2) * 5).toFixed(2);
    const kappa = i === 16 ? 73.32 : i === 0 ? 40.26 : (14 + (i % 7) * 3.5).toFixed(2);
    const nullity = (91.0 + (i % 6) * 0.8 + 0.3).toFixed(1);
    const drift = (Math.sin(i) * 0.08).toFixed(3);
    layerProfiles.push({ layer: i, fro, sigma, kappa, nullity, drift });
  }

  container.innerHTML = '';
  layerProfiles.forEach(p => {
    const btn = document.createElement('button');
    btn.className = `layer-block-btn ${p.layer === 15 ? 'active' : ''}`;
    btn.innerHTML = `<span class="layer-num">L${p.layer}</span><span class="layer-cond">κ:${p.kappa}</span>`;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.layer-block-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (inspectLayerName) inspectLayerName.textContent = `model.layers.${p.layer}.mlp.down_proj`;
      if (inspectFro) inspectFro.textContent = `${p.fro}`;
      if (inspectSigma) inspectSigma.textContent = `${p.sigma}`;
      if (inspectKappa) inspectKappa.textContent = `${p.kappa} (Weyl-Safe)`;
      if (inspectNullity) inspectNullity.textContent = `${p.nullity}% (Orthogonal)`;
      if (inspectDrift) inspectDrift.textContent = `${p.drift}% (PPL Invariant)`;
    });
    container.appendChild(btn);
  });
}
