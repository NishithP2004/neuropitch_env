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
---

# NeuroPitch (OpenEnv + TRL GRPO + Ollama + TRIBE v2)

NeuroPitch is a single-step OpenEnv environment for training a marketing-copy agent with layered reward verification:

1. Format validation
2. OpenAI compliance check with web evidence lookup
3. 5-model Ollama focus-group voting
4. TRIBE v2 biological resonance scoring
5. Final reward composition and terminal episode return

## Architecture

- **Learner**: small language model trained with GRPO (`scripts/train_grpo_neuropitch.py`)
- **Compliance Director**: OpenAI mediator (`gpt-4o-mini` default) + Tavily search evidence
- **Focus Group**: Ollama models
  - `llama3.2:3b` (Skeptical Millennial)
  - `qwen2.5:3b` (Value-Driven Shopper)
  - `phi4-mini:3.8b` (Tech-Savvy Gen Z)
  - `gemma3:4b` (Eco-Conscious Consumer)
  - `ministral-3:3b` (Impulse Buyer)
- **Biological Verifier**: TRIBE v2 (`facebook/tribev2`) with ROI proxy extraction for `STS`, `TPJ`, `Broca_45`

Reward logic:

```text
if non_compliant:
    reward = -1.0 + formatting_penalty
else:
    reward = formatting_penalty + (buy_votes * 0.2) + (z_sts + z_tpj - z_broca_45)
```

## Required Environment Variables

Copy `.env.example` and fill all required values.

```bash
cp .env.example .env
```

Strict mode is enforced; startup fails if critical dependencies/keys are missing.

For Hugging Face Spaces with persistent storage, prefer writing artifacts to `/data`.
The training API and GRPO script now default to `/data/neuropitch-grpo`.
Ollama runtime logs/models and TRIBE cache should also use `/data` (for example `/data/ollama/models` and `/data/tribe-cache`).
TRIBE anchor audio defaults to `./server/reference.wav`. You can override with `TRIBE_AUDIO_ANCHOR_PATH` (for example `/data/commercial_anchor.wav`).

## Local Development

Install dependencies:

```bash
uv sync
```

Note: training depends on a pinned `transformers`/`trl` stack compatible with the TRIBE Torch runtime in this project.
OpenEnv is invoked from the **reward function** (TRL 0.22.x has no `environment_factory` on `GRPOTrainer`); use `--environment-url` to point at the running server.

Run server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- OpenEnv API + WS: `/openenv`
- Custom simulation dashboard: `/web`
- Simulation APIs: `/api/sim/reset`, `/api/sim/step`, `/api/sim/state`
- Training APIs: `/api/train/start`, `/api/train/status`, `/api/train/stream` (SSE logs)

## GRPO Training Script

**Recommended strategy**: small model + QLoRA + many short runs beats one large model with a few successful runs. Iterate on reward signal quality.

### Quick start (default — Qwen2.5-1.5B, ~6 GB VRAM peak)

```bash
python scripts/train_grpo_neuropitch.py \
  --environment-url http://127.0.0.1:8000/openenv \
  --output-dir /data/neuropitch-grpo
```

### Trying a slightly larger model

```bash
python scripts/train_grpo_neuropitch.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --lora-rank 16 --lora-alpha 32 \
  --environment-url http://127.0.0.1:8000/openenv \
  --output-dir /data/neuropitch-grpo-3b
```

Training uses **Unsloth + QLoRA (4-bit) by default** — fast iterations, low VRAM. Pass `--no-unsloth` for the plain Transformers path.

### Key tunable flags

| Flag | Default | Notes |
|---|---|---|
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | Smaller = more runs per budget |
| `--lora-rank` | `8` | Increase to 16/32 for larger models |
| `--lora-alpha` | `16` | Rule of thumb: 2 × rank |
| `--num-generations` | `4` | Rollouts per prompt (4–8 recommended) |
| `--max-steps` | `200` | More steps = more reward signal |
| `--learning-rate` | `5e-6` | Cosine schedule with warmup |
| `--beta` | `0.001` | KL penalty (keep small for fine-tuning) |
| `--temperature` | `0.9` | Generation diversity |
| `--no-unsloth` | — | Transformers-only, slower, more VRAM |
| `--push-to-hub` | — | Upload adapter after training |

> **Batch constraint**: `(per_device_train_batch_size × gradient_accumulation_steps × processes) % num_generations == 0`. Defaults satisfy this: `1 × 4 × 1 = 4`, `4 % 4 == 0`.

## Dashboard Usage

Open `/web` and use:

- **Begin Training / Start Simulation** button to start the GRPO process
- **Pitch Room** pane to submit an ad step manually
- **Consumer Brain** pane to monitor region scores and reward trend
- **Training Logs** pane for live backend stream output

## Docker / Hugging Face Space

Build image:

```bash
docker build -t neuropitch_env-env:latest -f server/Dockerfile .
```

The Docker image:

- installs Ollama
- pulls the 5 focus-group models at container startup
- runs `ollama serve` and `uvicorn` together

Deploy to Hugging Face Space:

```bash
openenv push
```

or:

```bash
openenv push --repo-id <org-or-user>/<space-name> --private
```
