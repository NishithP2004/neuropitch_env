const rewardHistory = [];
let stepCount = 0;

const elements = {
  startTrainingBtn: document.getElementById("start-training-btn"),
  stopTrainingBtn: document.getElementById("stop-training-btn"),
  trainModal: document.getElementById("train-modal"),
  modalCancelBtn: document.getElementById("modal-cancel-btn"),
  modalStartBtn: document.getElementById("modal-start-btn"),
  modalError: document.getElementById("modal-error"),
  cfgModel: document.getElementById("cfg-model"),
  cfgOutputDir: document.getElementById("cfg-output-dir"),
  cfgNumEpisodes: document.getElementById("cfg-num-episodes"),
  cfgMaxSteps: document.getElementById("cfg-max-steps"),
  cfgLr: document.getElementById("cfg-lr"),
  cfgNumGen: document.getElementById("cfg-num-gen"),
  cfgGradAccum: document.getElementById("cfg-grad-accum"),
  cfgUnsloth: document.getElementById("cfg-unsloth"),
  cfgPushToHub: document.getElementById("cfg-push-to-hub"),
  cfgHubModelId: document.getElementById("cfg-hub-model-id"),
  hubModelIdLabel: document.getElementById("hub-model-id-label"),
  runStepBtn: document.getElementById("run-step-btn"),
  pitchInput: document.getElementById("pitch-input"),
  stepCounter: document.getElementById("step-counter"),
  productBrief: document.getElementById("product-brief"),
  competitorAd: document.getElementById("competitor-ad"),
  complianceLog: document.getElementById("compliance-log"),
  panelVotes: document.getElementById("panel-votes"),
  zSts: document.getElementById("z-sts"),
  zTpj: document.getElementById("z-tpj"),
  zBroca: document.getElementById("z-broca"),
  rewardTotal: document.getElementById("reward-total"),
  logs: document.getElementById("training-logs"),
  genBrief: document.getElementById("gen-brief"),
  genText: document.getElementById("gen-text"),
  genReward: document.getElementById("gen-reward"),
  chart: document.getElementById("reward-chart"),
  trainStep: document.getElementById("train-step"),
  trainReward: document.getElementById("train-reward"),
  trainRewardStd: document.getElementById("train-reward-std"),
  trainKl: document.getElementById("train-kl"),
  trainLoss: document.getElementById("train-loss"),
  progressBar: document.getElementById("progress-bar"),
  progressLabel: document.getElementById("progress-label"),
};

let _maxSteps = 0;

function drawRewardChart() {
  const canvas = elements.chart;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1430";
  ctx.fillRect(0, 0, w, h);

  // Layout margins: left for Y labels, bottom for X labels
  const ml = 42, mr = 10, mt = 10, mb = 22;
  const pw = w - ml - mr;   // plot width
  const ph = h - mt - mb;   // plot height

  ctx.font = "10px monospace";
  ctx.fillStyle = "#8899cc";
  ctx.textAlign = "center";
  ctx.fillText("Reward", ml + pw / 2, h - 2);   // X-axis title

  ctx.save();
  ctx.translate(10, mt + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("reward", 0, 0);
  ctx.restore();

  if (rewardHistory.length < 2) {
    ctx.fillStyle = "#556";
    ctx.textAlign = "center";
    ctx.font = "11px sans-serif";
    ctx.fillText("Waiting for data…", ml + pw / 2, mt + ph / 2);
    return;
  }

  const minVal = Math.min(...rewardHistory);
  const maxVal = Math.max(...rewardHistory);
  const spread = Math.max(maxVal - minVal, 0.001);

  // ── Grid lines + Y-axis ticks ─────────────────────────────────────────────
  const yTicks = 5;
  ctx.strokeStyle = "#1e2850";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#7080aa";
  ctx.textAlign = "right";
  ctx.font = "10px monospace";

  for (let i = 0; i <= yTicks; i++) {
    const fraction = i / yTicks;
    const val = minVal + fraction * spread;
    const py = mt + ph - fraction * ph;
    ctx.beginPath();
    ctx.moveTo(ml, py);
    ctx.lineTo(ml + pw, py);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), ml - 4, py + 3.5);
  }

  // ── X-axis ticks ──────────────────────────────────────────────────────────
  const xTicks = Math.min(rewardHistory.length - 1, 5);
  ctx.textAlign = "center";
  for (let i = 0; i <= xTicks; i++) {
    const fraction = i / xTicks;
    const dataIdx = Math.round(fraction * (rewardHistory.length - 1));
    const px = ml + fraction * pw;
    ctx.beginPath();
    ctx.moveTo(px, mt + ph);
    ctx.lineTo(px, mt + ph + 4);
    ctx.strokeStyle = "#445577";
    ctx.lineWidth = 1;
    ctx.stroke();
    const stepLabel = _maxSteps > 0
      ? Math.round((dataIdx / (rewardHistory.length - 1)) * _maxSteps)
      : dataIdx;
    ctx.fillStyle = "#7080aa";
    ctx.fillText(String(stepLabel), px, mt + ph + 14);
  }

  // ── Axes border ───────────────────────────────────────────────────────────
  ctx.strokeStyle = "#334466";
  ctx.lineWidth = 1;
  ctx.strokeRect(ml, mt, pw, ph);

  // ── Reward line ───────────────────────────────────────────────────────────
  ctx.strokeStyle = "#6ab4ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rewardHistory.forEach((val, idx) => {
    const px = ml + (idx / (rewardHistory.length - 1)) * pw;
    const py = mt + ph - ((val - minVal) / spread) * ph;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function renderVotes(votes) {
  elements.panelVotes.innerHTML = "";
  Object.entries(votes || {}).forEach(([persona, vote]) => {
    const item = document.createElement("div");
    const cls = vote === "BUY" ? "buy" : vote === "PASS" ? "pass" : "";
    item.className = `vote-pill ${cls}`;
    item.textContent = `${persona}: ${vote}`;
    elements.panelVotes.appendChild(item);
  });
}

function renderObservation(obs) {
  if (!obs) return;
  elements.productBrief.textContent = obs.product_brief || "";
  elements.competitorAd.textContent = obs.competitor_ad || "";
  elements.complianceLog.textContent = obs.compliance_log || "";
  renderVotes(obs.panel_votes);
  const tribe = obs.tribe_scores || {};
  elements.zSts.textContent = Number(tribe.z_sts || 0).toFixed(3);
  elements.zTpj.textContent = Number(tribe.z_tpj || 0).toFixed(3);
  elements.zBroca.textContent = Number(tribe.z_broca_45 || 0).toFixed(3);
  const reward = Number(obs.final_reward || obs.reward || 0);
  elements.rewardTotal.textContent = `Total Reward: ${reward.toFixed(3)}`;
  rewardHistory.push(reward);
  drawRewardChart();
}

async function resetSimulation() {
  const response = await fetch("/api/sim/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const payload = await response.json();
  renderObservation(payload);
}

async function runStep() {
  const pitch = elements.pitchInput.value.trim();
  if (!pitch) {
    elements.complianceLog.textContent = "Pitch text is required.";
    return;
  }
  const response = await fetch("/api/sim/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pitch_text: pitch }),
  });
  const payload = await response.json();
  stepCount += 1;
  elements.stepCounter.textContent = `Step: ${stepCount}`;
  renderObservation(payload);
}

const METRIC_PREFIX = "METRIC:";

function setTrainingRunning(running) {
  elements.startTrainingBtn.disabled = running;
  elements.stopTrainingBtn.disabled = !running;
}

function updateProgress(step, maxSteps) {
  if (!maxSteps) return;
  const pct = Math.min(100, (step / maxSteps) * 100);
  elements.progressBar.style.width = `${pct.toFixed(1)}%`;
  elements.progressLabel.textContent = `Step ${step} / ${maxSteps}  (${pct.toFixed(0)}%)`;
}

function applyTrainingMetric(metric) {
  if (metric.type === "start") {
    _maxSteps = metric.max_steps || 0;
    setTrainingRunning(true);
    updateProgress(0, _maxSteps);
    return;
  }

  // Per-rollout detail: update brain scores, votes, compliance, reward total
  if (metric.type === "reward_detail") {
    const tribe = metric.tribe_scores || {};
    if (tribe.z_sts !== undefined)      elements.zSts.textContent  = Number(tribe.z_sts).toFixed(3);
    if (tribe.z_tpj !== undefined)      elements.zTpj.textContent  = Number(tribe.z_tpj).toFixed(3);
    if (tribe.z_broca_45 !== undefined) elements.zBroca.textContent = Number(tribe.z_broca_45).toFixed(3);

    if (metric.compliance_status) {
      const cs = metric.compliance_status;
      const components = metric.reward_components || {};
      const lines = [`Status: ${cs}`];
      if (components.formatting_penalty !== undefined && components.formatting_penalty < 0)
        lines.push(`⚠ Formatting penalty: ${components.formatting_penalty}`);
      if (components.compliance_penalty !== undefined)
        lines.push(`✗ Compliance penalty: ${components.compliance_penalty}`);
      if (components.tribe_bonus !== undefined)
        lines.push(`🧠 Tribe bonus: ${Number(components.tribe_bonus).toFixed(3)}`);
      elements.complianceLog.textContent = `[Training rollout]\n${lines.join("\n")}`;
    }

    if (metric.panel_votes && Object.keys(metric.panel_votes).length)
      renderVotes(metric.panel_votes);

    if (metric.reward !== undefined)
      elements.rewardTotal.textContent = `Rollout Reward: ${Number(metric.reward).toFixed(3)}`;

    return;
  }

  // Step-level aggregate metrics (logged by TRL after every gradient step)
  if (metric.step !== undefined) {
    elements.trainStep.textContent = metric.step;
    updateProgress(metric.step, _maxSteps);
  }
  // TRL logs rewards under both "reward" and "rewards/reward_func/mean"
  const rewardVal = metric["reward"] ?? metric["rewards/reward_func/mean"];
  if (rewardVal !== undefined) {
    elements.trainReward.textContent = Number(rewardVal).toFixed(4);
    rewardHistory.push(Number(rewardVal));
    drawRewardChart();
  }
  const stdVal = metric["reward_std"] ?? metric["rewards/reward_func/std"];
  if (stdVal !== undefined) {
    elements.trainRewardStd.textContent = Number(stdVal).toFixed(4);
  }
  if (metric["kl"] !== undefined) {
    elements.trainKl.textContent = Number(metric["kl"]).toFixed(5);
  }
  if (metric["loss"] !== undefined) {
    elements.trainLoss.textContent = Number(metric["loss"]).toFixed(6);
  }
  if (metric.type === "generation") {
    elements.genBrief.textContent = metric.product_brief || "—";
    elements.genText.textContent = metric.generated_pitch || "(empty)";
    const r = Number(metric.reward);
    elements.genReward.textContent = `${r.toFixed(3)}  ${r >= 0 ? "✓" : "✗"}`;
    elements.genReward.style.color = r >= 0 ? "var(--good)" : "var(--bad)";
    return;
  }
  if (metric.type === "start" && metric.completions_log) {
    elements.logs.textContent += `\n[info] Completions log → ${metric.completions_log}\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
    return;
  }
  if (metric.type === "saved") {
    elements.logs.textContent += `\n[info] Model saved → ${metric.output_dir}\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  }
  if (metric.type === "pushing") {
    elements.logs.textContent += `\n[info] Pushing model to Hub (${metric.hub_model_id || "…"}) …\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  }
  if (metric.type === "pushed") {
    elements.logs.textContent += `\n[✓] Model pushed to Hub! → ${metric.url}\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  }
  if (metric.type === "push_failed") {
    elements.logs.textContent += `\n[✗] Hub push failed: ${metric.error}\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  }
}

let _hfUsername = "";

async function loadEnvConfig() {
  try {
    const res = await fetch("/api/env-config");
    if (res.ok) {
      const cfg = await res.json();
      _hfUsername = cfg.hf_username || "";
    }
  } catch (_) { /* non-fatal */ }
}

function _defaultHubModelId() {
  const repoName = elements.cfgOutputDir.value.trim().replace(/.*\//, "") || "neuropitch-grpo";
  return _hfUsername ? `${_hfUsername}/${repoName}` : "";
}

function openTrainModal() {
  elements.modalError.hidden = true;
  // Pre-populate hub model id if empty and we know the username
  if (!elements.cfgHubModelId.value.trim()) {
    elements.cfgHubModelId.value = _defaultHubModelId();
  }
  elements.trainModal.hidden = false;
  // Sync hub-model-id row visibility on open
  elements.hubModelIdLabel.hidden = !elements.cfgPushToHub.checked;
}

function closeTrainModal() {
  elements.trainModal.hidden = true;
}

function validateModalConfig() {
  const numGen   = parseInt(elements.cfgNumGen.value, 10);
  const gradAccum = parseInt(elements.cfgGradAccum.value, 10);
  if ((gradAccum) % numGen !== 0) {
    return `Batch constraint: (1 × ${gradAccum}) % ${numGen} ≠ 0. Try grad_accum = multiple of ${numGen}.`;
  }
  const numEps = parseInt(elements.cfgNumEpisodes.value, 10);
  if (numEps < numGen) {
    return `Dataset size must be ≥ num_generations (${numGen}).`;
  }
  if (numEps % 3 !== 0) {
    return `Dataset size should be a multiple of 3 (3 brief scenarios). Nearest: ${Math.ceil(numEps / 3) * 3}.`;
  }
  if (elements.cfgPushToHub.checked && !elements.cfgHubModelId.value.trim()) {
    return "Hub model ID is required when Push to Hub is enabled (e.g. username/neuropitch-grpo).";
  }
  return null;
}

async function submitTraining() {
  const err = validateModalConfig();
  if (err) {
    elements.modalError.textContent = err;
    elements.modalError.hidden = false;
    return;
  }
  elements.modalError.hidden = true;
  closeTrainModal();

  const pushToHub = elements.cfgPushToHub.checked;
  const body = {
    model_name: elements.cfgModel.value.trim(),
    output_dir: elements.cfgOutputDir.value.trim(),
    max_steps: parseInt(elements.cfgMaxSteps.value, 10),
    num_episodes: parseInt(elements.cfgNumEpisodes.value, 10),
    learning_rate: parseFloat(elements.cfgLr.value),
    num_generations: parseInt(elements.cfgNumGen.value, 10),
    gradient_accumulation_steps: parseInt(elements.cfgGradAccum.value, 10),
    environment_url: `${window.location.origin}/openenv`,
    use_unsloth: elements.cfgUnsloth.checked,
    push_to_hub: pushToHub,
    hub_model_id: pushToHub ? elements.cfgHubModelId.value.trim() : "",
  };

  const response = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errData = await response.json();
    elements.logs.textContent += `\n[error] ${JSON.stringify(errData)}\n`;
    return;
  }
  _maxSteps = body.max_steps;
  const hubNote = pushToHub ? ` · will push → ${body.hub_model_id}` : "";
  elements.logs.textContent +=
    `\n[info] Training started — ${body.num_episodes} episodes · ${body.max_steps} steps · ${body.model_name}${hubNote}\n`;
  setTrainingRunning(true);
  updateProgress(0, _maxSteps);
}

async function stopTraining() {
  const response = await fetch("/api/train/stop", { method: "POST" });
  const data = await response.json();
  elements.logs.textContent += `\n[info] Stop requested: ${data.status}\n`;
  elements.logs.scrollTop = elements.logs.scrollHeight;
  setTrainingRunning(false);
}

function connectTrainingStream() {
  const events = new EventSource("/api/train/stream");
  events.onmessage = (event) => {
    const data = event.data;

    // tqdm uses \r (carriage return) not \n for progress lines, so when our
    // print("METRIC:...") adds the first \n, the subprocess reader bundles the
    // preceding tqdm chars + our METRIC on the same line, e.g.:
    //   "  0%|...|1/200[...]METRIC:{...}"
    // Use indexOf so we find the marker regardless of leading tqdm noise.
    const metricIdx = data.indexOf(METRIC_PREFIX);
    if (metricIdx !== -1) {
      try {
        const metric = JSON.parse(data.slice(metricIdx + METRIC_PREFIX.length));
        applyTrainingMetric(metric);
        if (metric.type === "step") {
          const r = metric["reward"] ?? metric["rewards/reward_func/mean"];
          const summary = `[step ${metric.step}] reward=${r !== undefined ? Number(r).toFixed(4) : "?"} loss=${metric.loss !== undefined ? Number(metric.loss).toFixed(6) : "?"}`;
          elements.logs.textContent += `${summary}\n`;
        }
        // For reward_detail don't echo anything — the panels updated silently
      } catch {
        elements.logs.textContent += `${data}\n`;
      }
    } else {
      if (data.includes("[training-finished]") || data.includes("[training-stopped]")) {
        setTrainingRunning(false);
        if (_maxSteps) updateProgress(_maxSteps, _maxSteps);
      }
      // Skip raw TRL metric dicts (lines starting with '{') — they're noisy duplicates
      const trimmed = data.trim();
      if (!trimmed.startsWith("{") && trimmed !== "") {
        elements.logs.textContent += `${data}\n`;
      }
    }
    elements.logs.scrollTop = elements.logs.scrollHeight;
  };
}

elements.runStepBtn.addEventListener("click", runStep);
elements.startTrainingBtn.addEventListener("click", openTrainModal);
elements.stopTrainingBtn.addEventListener("click", stopTraining);
elements.modalCancelBtn.addEventListener("click", closeTrainModal);
elements.modalStartBtn.addEventListener("click", submitTraining);
elements.trainModal.addEventListener("click", (e) => {
  if (e.target === elements.trainModal) closeTrainModal();
});
elements.cfgPushToHub.addEventListener("change", () => {
  elements.hubModelIdLabel.hidden = !elements.cfgPushToHub.checked;
  if (elements.cfgPushToHub.checked) {
    if (!elements.cfgHubModelId.value.trim()) {
      elements.cfgHubModelId.value = _defaultHubModelId();
    }
    elements.cfgHubModelId.focus();
  }
});
elements.cfgOutputDir.addEventListener("input", () => {
  // Keep auto-derived Hub model ID in sync when user changes output dir
  const current = elements.cfgHubModelId.value.trim();
  if (!current || (_hfUsername && current.startsWith(_hfUsername + "/"))) {
    elements.cfgHubModelId.value = _defaultHubModelId();
  }
});

resetSimulation().catch((err) => {
  elements.complianceLog.textContent = `Reset failed: ${err.message}`;
});
connectTrainingStream();
loadEnvConfig();
