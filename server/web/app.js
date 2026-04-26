const rewardHistory = [];
let stepCount = 0;

const elements = {
  startTrainingBtn: document.getElementById("start-training-btn"),
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
  chart: document.getElementById("reward-chart"),
};

function drawRewardChart() {
  const canvas = elements.chart;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1430";
  ctx.fillRect(0, 0, w, h);
  if (rewardHistory.length < 2) return;

  const minVal = Math.min(...rewardHistory);
  const maxVal = Math.max(...rewardHistory);
  const spread = Math.max(maxVal - minVal, 0.001);
  ctx.strokeStyle = "#6ab4ff";
  ctx.lineWidth = 2;
  ctx.beginPath();

  rewardHistory.forEach((val, idx) => {
    const x = (idx / (rewardHistory.length - 1)) * (w - 20) + 10;
    const y = h - ((val - minVal) / spread) * (h - 20) - 10;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
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

async function startTraining() {
  const body = {
    model_name: "Qwen/Qwen2.5-3B-Instruct",
    output_dir: "/data/neuropitch-grpo",
    max_steps: 100,
    learning_rate: 1e-6,
    environment_url: `${window.location.origin}/openenv`,
    use_unsloth: true,
  };
  const response = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = await response.json();
    elements.logs.textContent += `\n[error] ${JSON.stringify(err)}`;
    return;
  }
  elements.logs.textContent += "\n[info] Training process started.";
}

function connectTrainingStream() {
  const events = new EventSource("/api/train/stream");
  events.onmessage = (event) => {
    elements.logs.textContent += `${event.data}\n`;
    elements.logs.scrollTop = elements.logs.scrollHeight;
  };
}

elements.runStepBtn.addEventListener("click", runStep);
elements.startTrainingBtn.addEventListener("click", startTraining);

resetSimulation().catch((err) => {
  elements.complianceLog.textContent = `Reset failed: ${err.message}`;
});
connectTrainingStream();
