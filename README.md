---
title: NeuroPitch Environment Server
emoji: 🧠
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - neuromarketing
  - grpo
  - tribe
---

# NeuroPitch — The First RL Environment with a Simulated Human Brain as Reward

> *"Stop optimising for clicks. Train for the mind."*

NeuroPitch is a reinforcement-learning training environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) that uses **TRIBE v2** — a computational neuroscience model of neurological response — as part of its reward signal. To our knowledge, it is the **first OpenEnv environment where a simulated human brain directly shapes the training reward** for a language model.

A learner (small LLM) writes advertising copy. Instead of a simple human-preference score, the reward flows through three independent verification layers before any gradient is computed:

```
Pitch text
    │
    ▼
① Format check          → penalty if > 120 words
    │
    ▼
② Compliance Director   → OpenAI (gpt-4o-mini) + Tavily web evidence
   "Are the claims legal, verifiable, and not deceptive?"
    │ NON_COMPLIANT → reward = -1.0 + formatting_penalty  (terminal)
    ▼
③ Ollama Focus Group    → 5 persona LLMs vote BUY / PASS
   Skeptical Millennial · Value-Driven Shopper · Tech-Savvy Gen Z
   Eco-Conscious Consumer · Impulse Buyer
    │
    ▼
④ TRIBE v2              → neurological resonance scoring
   STS  (social cognition, empathy activation)
   TPJ  (perspective-taking, theory of mind)
   Broca_45  (language elaboration — penalised if over-activated)
    │
    ▼
⑤ Final reward = formatting_penalty + (buy_votes × 0.2) + (z_STS + z_TPJ − z_Broca_45)
```

The model learns to write copy that is **legally clean**, **socially persuasive**, and **neurologically resonant** — all at once.

---

## Why TRIBE v2?

TRIBE v2 (`facebook/tribev2`) is a transformer-based brain-encoding model trained on fMRI data from participants watching video advertisements. Given a text or audio stimulus, it predicts cortical activation across thousands of brain regions.

NeuroPitch extracts Z-scored activations from three ROIs known from neuromarketing research:

| Region | Role | Effect on reward |
|---|---|---|
| **STS** (Superior Temporal Sulcus) | Social cognition, empathy | ↑ higher is better |
| **TPJ** (Temporoparietal Junction) | Perspective-taking, theory of mind | ↑ higher is better |
| **Broca_45** | Language elaboration, complexity | ↓ penalised — simpler copy lands better |

This is not a proxy or heuristic. TRIBE v2 predicts actual neural activation. The learner is trained to write copy that the simulated brain responds to positively.

---

## Architecture

| Component | Implementation |
|---|---|
| **Learner** | Small LLM (default: Qwen2.5-1.5B-Instruct), fine-tuned with GRPO + QLoRA via Unsloth |
| **OpenEnv server** | FastAPI + WebSocket, single-step episode |
| **Compliance Director** | OpenAI `gpt-4o-mini` + Tavily search evidence |
| **Focus Group** | 5 × Ollama models — each a distinct consumer persona |
| **Biological Verifier** | TRIBE v2 — text + audio anchor blended (80/20) |
| **Dashboard** | Custom HTML/CSS/JS — live training metrics, brain scores, panel votes |

---

## Quick Start

### 1. Set environment variables

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and TAVILY_API_KEY
```

### 2. Docker (recommended)

```bash
docker build -t neuropitch_env-env:latest -f server/Dockerfile .
docker run --rm -it \
  --env-file .env \
  -p 8000:8000 \
  -v neuropitch_data:/data \
  neuropitch_env-env:latest
```

Open **http://localhost:8000/web**

### 3. Local dev

```bash
uv sync
ollama serve &
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## GRPO Training

**Strategy**: small model + QLoRA + many short runs outperforms one large run. Iterate on the reward signal, not the model size.

### Default run (~35s/step on a single L4, ~6 GB VRAM)

```bash
python scripts/train_grpo_neuropitch.py \
  --environment-url http://127.0.0.1:8000/openenv \
  --output-dir /data/neuropitch-grpo
```

### Larger model

```bash
python scripts/train_grpo_neuropitch.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --lora-rank 16 --lora-alpha 32 \
  --environment-url http://127.0.0.1:8000/openenv \
  --output-dir /data/neuropitch-grpo-3b
```

Uses **Unsloth + QLoRA (4-bit)** by default. Pass `--no-unsloth` for plain Transformers.

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | Smaller = more runs per budget |
| `--lora-rank` | `8` | Increase to 16/32 for larger models |
| `--lora-alpha` | `16` | Rule of thumb: 2 × rank |
| `--num-generations` | `2` | Rollouts per prompt (2 = fast; 4–8 = stable gradient) |
| `--gradient-accumulation-steps` | `2` | Must satisfy `(batch × accum × procs) % num_gen == 0` |
| `--max-steps` | `200` | More steps = more reward signal |
| `--learning-rate` | `5e-6` | Cosine schedule with warmup |
| `--beta` | `0.001` | KL penalty — keep small for fine-tuning |
| `--temperature` | `0.9` | Generation diversity |
| `--push-to-hub` | — | Upload LoRA adapter after training |

---

## Dashboard

Open `/web` to access the live simulation and training console.

**During training**, the dashboard updates in real-time:

- **Progress bar** — step / max_steps with percentage
- **Consumer Brain panel** — live TRIBE v2 Z-scores (STS · TPJ · Broca_45) per rollout
- **Consumer Panel** — latest per-rollout BUY/PASS votes from all 5 personas
- **Compliance log** — current rollout compliance status + penalty breakdown
- **Reward chart** — mean reward per gradient step (trend line)
- **Training stats** — step, reward mean, reward σ, KL, loss

Use **Stop Training** to terminate the process at any point.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/openenv` | WS + HTTP | OpenEnv environment WebSocket |
| `/web` | GET | Dashboard UI |
| `/api/sim/reset` | POST | Reset simulation episode |
| `/api/sim/step` | POST | Submit a pitch manually |
| `/api/sim/state` | GET | Current episode state |
| `/api/train/start` | POST | Start GRPO training subprocess |
| `/api/train/stop` | POST | Terminate training |
| `/api/train/status` | GET | Status + last 50 log lines |
| `/api/train/stream` | GET (SSE) | Live log + metric stream |

---

## Docker / Hugging Face Space

The image installs Ollama, pulls all 5 focus-group models at startup, and runs `ollama serve` + `uvicorn` together. `gcc`/`build-essential` are included so Triton can JIT-compile its CUDA driver at runtime.

```bash
# Build
docker build -t neuropitch_env-env:latest -f server/Dockerfile .

# Deploy to HF Space
openenv push
# or:
openenv push --repo-id <org>/<space-name> --private
```

All artifacts (`/data/neuropitch-grpo`, `/data/ollama/models`, `/data/tribe-cache`) are written to `/data` for persistent HF Space storage.

---

## Reference

- [TRIBE v2 (facebook/tribev2)](https://huggingface.co/facebook/tribev2) — brain-encoding model for neuromarketing
- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment framework for LLM RL training
- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster GRPO fine-tuning
- [TRL](https://github.com/huggingface/trl) — GRPO trainer
