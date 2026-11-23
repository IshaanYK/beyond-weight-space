/**
 * BEYOND WEIGHT-SPACE PROGRAMMING (SP-SV-WSP) - INTERACTIVE ENGINE
 * Real-time HTML5 Canvas Visualizers, Matrix Heatmap Surgery, and Telemetry Engine
 */

document.addEventListener('DOMContentLoaded', () => {
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
   2. Paradigm Interactive Canvas (3 Paradigms Comparison)
   ========================================================================== */
function initParadigmCanvas() {
  const canvas = document.getElementById('paradigmCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let currentParadigm = 'spsv'; // 'train', 'inference', 'spsv'
  let animationFrameId;
  let t = 0;

  // Set canvas internal resolution
  function resize() {
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
  }
  resize();
  window.addEventListener('resize', resize);

  // Tab listeners
  const tabs = document.querySelectorAll('.paradigm-tab');
  const titleEl = document.getElementById('paradigmTitle');
  const descEl = document.getElementById('paradigmDesc');
  const metricLatency = document.getElementById('metricLatency');
  const metricCost = document.getElementById('metricCost');
  const metricPermanence = document.getElementById('metricPermanence');

  const infoData = {
    train: {
      title: "Paradigm 1: Retrain / Fine-Tune the Whole Model",
      desc: "Requires massive compute clusters, high GPU memory, backward loss backpropagation, and risks catastrophic forgetting of prior skills.",
      latency: "Hours - Days",
      cost: "Very High ($$$)",
      permanence: "Permanent (Drifting)"
    },
    inference: {
      title: "Paradigm 2: Test-Time Scaling / Prompt Bloat",
      desc: "Chain-of-thought expands reasoning at runtime. Consumes linear GPU memory and latency per user query without updating the underlying model.",
      latency: "+300% - +800%",
      cost: "High Token Tax",
      permanence: "Ephemeral (Zero Gain)"
    },
    spsv: {
      title: "Paradigm 3: SP-SV-WSP (Closed-Form Null-Space Surgery)",
      desc: "An autonomous agent projects a targeted rank-1 update into the mathematical null space of generic activations. The fix is instant, permanent, and incurs zero runtime cost.",
      latency: "0.0 ms Added",
      cost: "Zero API / Cluster",
      permanence: "Permanent & Isolated"
    }
  };

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentParadigm = tab.dataset.paradigm;

      const info = infoData[currentParadigm];
      titleEl.textContent = info.title;
      descEl.textContent = info.desc;
      metricLatency.textContent = info.latency;
      metricCost.textContent = info.cost;
      metricPermanence.textContent = info.permanence;
    });
  });

  // Animation Loop
  function render() {
    t += 0.03;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (currentParadigm === 'train') {
      renderTrainingAnimation(ctx, w, h, t);
    } else if (currentParadigm === 'inference') {
      renderInferenceAnimation(ctx, w, h, t);
    } else {
      renderSpsvAnimation(ctx, w, h, t);
    }

    animationFrameId = requestAnimationFrame(render);
  }
  render();
}

function renderTrainingAnimation(ctx, w, h, t) {
  // Draw cluster of nodes and chaotic backprop streams
  ctx.fillStyle = '#f43f5e';
  ctx.font = `${12 * window.devicePixelRatio}px JetBrains Mono`;
  ctx.fillText("BACKPROPAGATION GRADIENT DRIFT", 20, 30 * window.devicePixelRatio);

  // Nodes
  const numNodes = 14;
  for (let i = 0; i < numNodes; i++) {
    const x = (w * 0.2) + (i % 4) * (w * 0.18);
    const y = (h * 0.3) + Math.floor(i / 4) * (h * 0.2) + Math.sin(t + i) * 10;
    
    ctx.beginPath();
    ctx.arc(x, y, 6 * window.devicePixelRatio, 0, Math.PI * 2);
    ctx.fillStyle = i % 2 === 0 ? '#f43f5e' : '#ff9fb2';
    ctx.fill();

    // Cross connecting lines
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 50 * Math.cos(t * 2 + i), y + 40 * Math.sin(t * 2 + i));
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function renderInferenceAnimation(ctx, w, h, t) {
  ctx.fillStyle = '#ffe600';
  ctx.font = `${12 * window.devicePixelRatio}px JetBrains Mono`;
  ctx.fillText("INFERENCE TOKEN BUFFER SURGE (LINEAR EXPANSION)", 20, 30 * window.devicePixelRatio);

  // Drawing expanding stack of tokens
  const stackCount = Math.floor((t * 6) % 18) + 1;
  const barW = w * 0.6;
  const startX = w * 0.2;
  const startY = h * 0.8;

  for (let i = 0; i < stackCount; i++) {
    const y = startY - i * (12 * window.devicePixelRatio);
    ctx.fillStyle = `rgba(255, 230, 0, ${0.3 + (i / 18) * 0.6})`;
    ctx.fillRect(startX, y, barW, 9 * window.devicePixelRatio);
  }
}

function renderSpsvAnimation(ctx, w, h, t) {
  ctx.fillStyle = '#00f0ff';
  ctx.font = `${12 * window.devicePixelRatio}px JetBrains Mono`;
  ctx.fillText("SUB-SPACE ISOLATION: P_null = I - C(C + lambda*I)^-1", 20, 30 * window.devicePixelRatio);

  // Draw retained manifold disk
  const cx = w * 0.5;
  const cy = h * 0.55;
  const rx = w * 0.35;
  const ry = h * 0.22;

  // Manifold ellipse
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(94, 106, 210, 0.08)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(94, 106, 210, 0.4)';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = '#7985ea';
  ctx.font = `${11 * window.devicePixelRatio}px Inter`;
  ctx.fillText("Generic Knowledge Manifold C_train (Protected)", cx - 120, cy);

  // Orthogonal Null-Space Beam
  const beamY = cy - ry * 1.4 + Math.sin(t * 3) * 15;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, beamY);
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 15;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Pulse point
  ctx.beginPath();
  ctx.arc(cx, beamY, 8 * window.devicePixelRatio, 0, Math.PI * 2);
  ctx.fillStyle = '#00f0ff';
  ctx.fill();

  ctx.fillStyle = '#00f0ff';
  ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono`;
  ctx.fillText("delta W in Null-Space (>99.9% Invariant)", cx + 12, beamY + 4);
}

/* ==========================================================================
   3. Interactive Matrix Surgery Heatmap (48x96 Grid)
   ========================================================================== */
function initMatrixSurgeryHeatmap() {
  const canvas = document.getElementById('matrixCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const rows = 36;
  const cols = 72;
  let currentMode = 'pristine'; // 'pristine', 'attractor', 'unconstrained', 'spsv'

  // Precompute pseudo-random base matrix W0
  const W0 = [];
  const attractor = [];
  const unconstrainedDelta = [];
  const spsvDelta = [];

  for (let r = 0; r < rows; r++) {
    W0[r] = [];
    attractor[r] = [];
    unconstrainedDelta[r] = [];
    spsvDelta[r] = [];
    for (let c = 0; c < cols; c++) {
      // Base normal
      W0[r][c] = (Math.sin(r * 12.3 + c * 33.1) + Math.cos(r * 7.7 - c * 11.2)) * 0.35;
      
      // Attractor localized outer product
      const u = Math.exp(-Math.pow((c - 36) / 8, 2));
      const v = Math.exp(-Math.pow((r - 18) / 6, 2));
      attractor[r][c] = u * v;

      // Unconstrained surgery has wide collateral leakage (20.2%)
      const leakage = (Math.sin(r * 5.5) * Math.cos(c * 4.2)) * 0.3;
      unconstrainedDelta[r][c] = -0.7 * attractor[r][c] + leakage;

      // SP-SV-WSP has exact null-space isolation: only modifies attractor without background bleed
      spsvDelta[r][c] = -0.7 * attractor[r][c];
    }
  }

  function resize() {
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = 280 * window.devicePixelRatio;
  }
  resize();
  window.addEventListener('resize', () => { resize(); drawMatrix(); });

  const hudCoords = document.getElementById('hudCoords');
  const hudWeight = document.getElementById('hudWeight');
  const hudDelta = document.getElementById('hudDelta');
  const telemInvar = document.getElementById('telemInvar');
  const telemPpl = document.getElementById('telemPpl');
  const telemFro = document.getElementById('telemFro');
  const telemStatus = document.getElementById('telemStatus');

  const modeButtons = document.querySelectorAll('.mode-btn');
  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      modeButtons.forEach(b => b.classList.remove('active', 'crimson', 'cyan'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      if (currentMode === 'unconstrained') btn.classList.add('crimson');
      if (currentMode === 'spsv') btn.classList.add('cyan');

      updateTelemetry(currentMode);
      drawMatrix();
    });
  });

  function updateTelemetry(mode) {
    if (mode === 'pristine') {
      telemInvar.textContent = "100.00%";
      telemInvar.className = "telemetry-val good";
      telemPpl.textContent = "42.38 (Baseline)";
      telemPpl.className = "telemetry-val";
      telemFro.textContent = "280.63";
      telemStatus.textContent = "BASELINE (REPETITION LOOP)";
      telemStatus.className = "telemetry-val";
    } else if (mode === 'attractor') {
      telemInvar.textContent = "91.65% (Orthogonal)";
      telemInvar.className = "telemetry-val cyan";
      telemPpl.textContent = "42.38";
      telemPpl.className = "telemetry-val";
      telemFro.textContent = "Rank-1 Vector Captured";
      telemStatus.textContent = "LOOP ATTRACTOR DETECTED";
      telemStatus.className = "telemetry-val cyan";
    } else if (mode === 'unconstrained') {
      telemInvar.textContent = "79.82% (-20.18% Bleed)";
      telemInvar.className = "telemetry-val bad";
      telemPpl.textContent = "+16.71% (SURGE / 49.46)";
      telemPpl.className = "telemetry-val bad";
      telemFro.textContent = "284.15 (+1.25%)";
      telemStatus.textContent = "MANIFOLD COLLAPSE (#?#?#?)";
      telemStatus.className = "telemetry-val bad";
    } else {
      telemInvar.textContent = "99.97% (Safe Bound)";
      telemInvar.className = "telemetry-val good";
      telemPpl.textContent = "-0.28% (42.26 Clean)";
      telemPpl.className = "telemetry-val good";
      telemFro.textContent = "280.64 (+0.003%)";
      telemStatus.textContent = "LOOP ELIMINATED (SAFE)";
      telemStatus.className = "telemetry-val good";
    }
  }

  function getColor(val, mode) {
    if (mode === 'pristine') {
      // Linear obsidian-blue spectrum
      const norm = (val + 1) / 2;
      const b = Math.floor(40 + norm * 180);
      return `rgb(10, ${Math.floor(b * 0.35)}, ${b})`;
    } else if (mode === 'attractor') {
      // Gold / Yellow attractor highlight
      const intensity = Math.min(1, Math.max(0, val));
      return `rgba(255, 230, 0, ${0.1 + intensity * 0.9})`;
    } else if (mode === 'unconstrained') {
      // Crimson red cross-talk leakage
      if (Math.abs(val) > 0.4) {
        return `rgba(244, 63, 94, ${Math.min(1, Math.abs(val))})`;
      }
      return `rgba(30, 15, 25, 0.8)`;
    } else {
      // SP-SV-WSP: Electric Cyan isolated delta
      if (Math.abs(val) > 0.2) {
        return `rgba(0, 240, 255, ${Math.min(1, Math.abs(val) * 1.5)})`;
      }
      return `rgb(10, 18, 30)`;
    }
  }

  function drawMatrix() {
    const w = canvas.width;
    const h = canvas.height;
    const cellW = w / cols;
    const cellH = h / rows;

    ctx.clearRect(0, 0, w, h);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let val;
        if (currentMode === 'pristine') val = W0[r][c];
        else if (currentMode === 'attractor') val = attractor[r][c];
        else if (currentMode === 'unconstrained') val = unconstrainedDelta[r][c];
        else val = spsvDelta[r][c];

        ctx.fillStyle = getColor(val, currentMode);
        ctx.fillRect(c * cellW, r * cellH, cellW - 0.5, cellH - 0.5);
      }
    }
  }
  drawMatrix();

  // Mouse hover HUD tracker
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor((x / rect.width) * cols);
    const row = Math.floor((y / rect.height) * rows);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      hudCoords.textContent = `[R:${row}, C:${col}]`;
      hudWeight.textContent = W0[row][col].toFixed(4);
      let delta = 0;
      if (currentMode === 'unconstrained') delta = unconstrainedDelta[row][col];
      else if (currentMode === 'spsv') delta = spsvDelta[row][col];
      hudDelta.textContent = delta !== 0 ? (delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)) : "0.0000";
    }
  });
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

    const lambda = parseFloat(lambdaSlider.value);
    const beta = parseFloat(betaSlider.value);

    lambdaVal.textContent = lambda.toExponential(1);
    betaVal.textContent = beta.toFixed(1);

    // Compute empirical simulated invariance
    // Higher lambda = slightly more retained energy leaking
    const inv = Math.max(90, Math.min(99.99, 100 - (lambda * 150) / (beta * 0.8))).toFixed(2);
    calcInvariance.textContent = `${inv}%`;
    const budget = (beta * 0.0028).toFixed(4);
    calcBudget.textContent = `${budget} ||W||_F (Bound < 0.02)`;

    // Render Geometric 2D Vector Projection
    const originX = w * 0.45;
    const originY = h * 0.65;

    // Subspace axis 1 (C_train generic activation span)
    ctx.beginPath();
    ctx.moveTo(originX - w * 0.35, originY + h * 0.15);
    ctx.lineTo(originX + w * 0.35, originY - h * 0.15);
    ctx.strokeStyle = 'rgba(94, 106, 210, 0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#7985ea';
    ctx.font = `${11 * window.devicePixelRatio}px Inter`;
    ctx.fillText("Generic Activation Span S_train", originX + w * 0.15, originY - h * 0.18);

    // Orthogonal Null-Space Complement Axis
    ctx.beginPath();
    ctx.moveTo(originX - w * 0.1, originY + h * 0.25);
    ctx.lineTo(originX + w * 0.1, originY - h * 0.45);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#00f0ff';
    ctx.fillText("Null Space N(C_train)", originX + w * 0.08, originY - h * 0.45);

    // Attractor vector V
    const vecLen = 140 * (beta / 4.0);
    const vx = originX + vecLen * 0.35;
    const vy = originY - vecLen * 0.9;

    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(vx, vy);
    ctx.strokeStyle = '#ffe600';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.fillStyle = '#ffe600';
    ctx.fillText("Failure Vector v", vx + 10, vy);

    // Projected Null-Space Vector
    const projX = originX + vecLen * 0.08;
    const projY = originY - vecLen * 0.88;

    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(projX, projY);
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 3.5;
    ctx.stroke();

    // Dotted projection line from V to Proj
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    ctx.lineTo(projX, projY);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  lambdaSlider.addEventListener('input', drawGeo);
  betaSlider.addEventListener('input', drawGeo);
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

      promptBox.textContent = data.prompt;
      baseBox.textContent = data.base;
      rawBox.textContent = data.raw;
      spsvBox.textContent = data.spsv;

      basePpl.textContent = data.basePpl;
      rawPpl.textContent = data.rawPpl;
      spsvPpl.textContent = data.spsvPpl;
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

    const padL = 60 * window.devicePixelRatio;
    const padR = 20 * window.devicePixelRatio;
    const padB = 40 * window.devicePixelRatio;
    const padT = 20 * window.devicePixelRatio;

    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
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
    ctx.strokeStyle = '#5e6ad2';
    ctx.lineWidth = 2.5;
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
    ctx.fillStyle = 'rgba(0, 240, 255, 0.08)';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#7e8494';
    ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono`;
    ctx.fillText("Singular Index i (1 -> 2560)", padL + plotW * 0.4, h - 10);
    ctx.fillText("sigma_i", 10, padT + 12);
  }
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
      desc: "The agent identifies the runaway token loop by computing forward pass logits, detecting a strong discrete argmax margin (delta z = 2.1 to 5.7) favoring degenerate output.",
      math: "delta_z = z_attractor - max_{j != attractor} z_j >= 2.1"
    },
    {
      num: "02",
      badge: "Stage 2: Covariance Profiling",
      title: "Generic Manifold Activation Profiling",
      desc: "The model runs a reference batch of retained text corpus to capture the empirical second-moment activation covariance C_train across candidate MLP layers.",
      math: "C_train = (1 / N) sum_{k=1}^N x_k x_k^T in R^{d_in x d_in}"
    },
    {
      num: "03",
      badge: "Stage 3: Subspace Projection",
      title: "Tikhonov Null-Space Operator Extraction",
      desc: "Projects the raw failure activation vector into the null space of C_train. Probing confirms that 87.1% - 94.3% of the attractor vector naturally lives in this null space.",
      math: "u_null = (I - C_train (C_train + lambda I)^-1) u_attractor"
    },
    {
      num: "04",
      badge: "Stage 4: Margin Calibration",
      title: "Adaptive Margin-Aware (AMA) Scaling",
      desc: "Calibrates the rank-1 update scale beta* to the minimal magnitude required to flip the discrete logit argmax while strictly obeying the Weyl singular value budget.",
      math: "delta W = -beta* v_target (u_null^T / (||u_null||^2 + eps))"
    },
    {
      num: "05",
      badge: "Stage 5: Autonomous Verification",
      title: "Closed-Loop Flight Control & Commit",
      desc: "Autonomous oracle verifies out-of-sample forward invariance (>= 99.8%) and held-out perplexity drift (<= 0.2%). Edits passing all tests commit in O(1); failures rollback instantly.",
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

