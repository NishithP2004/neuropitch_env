---
title: "NeuroPitch: Training LLMs for the Human Mind via In-Silico Neuroscience"
author: "Nishith (with TRIBE v2 as the biological scorer)"
tags: [OpenEnv, Reinforcement Learning, Neuromarketing, GRPO, TRIBE v2]
---

# NeuroPitch: Training LLMs for the Human Mind

**NeuroPitch** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement-learning environment: a small language model (SLM) learns to write ad copy, and a **stacked reward** decides whether that copy is any good. The unusual ingredient is **TRIBE v2**—a model trained to predict human cortical activity from text and audio—used as a **neuromarketing-style biological signal** on top of more familiar checks. This post is about that design, not a claim that the author *is* TRIBE; TRIBE is the scorer, not the narrator.

*"Stop optimising for clicks. Train for the mind."*

---

## 1. The capability gap in neuromarketing

A lot of automated copy relies on A/B tests or a single LLM grader. That can work for style, but it is easy to game, heavily subjective, and it does not connect persuasion to **measurable cognitive load and engagement** in a principled way. The open question: can we **train** an SLM to favour copy that is not only *allowed* and *liked by personas*, but also **resonant** in a way tied to in-silico brain regions used in neuromarketing work?

---

## 2. A layered reward: compliance, personas, then biology

NeuroPitch is **not** "TRIBE only." A single `step` runs several stages in order; only the last one uses TRIBE. That matters for honesty in research writing: the environment still uses **rules and other models**—it just refuses to treat "one LLM score" as the whole story.

1. **Formatting** — Copy over a word budget gets a **formatting penalty** (so the agent cannot spam walls of text to maximize reward).
2. **Compliance Director (OpenAI + optional Tavily search)** — The pitch is checked against a compliance-style policy, with **optional web evidence** for claim-heavy cases. Failing here ends the episode with a **strong negative reward**; the focus group and TRIBE are **not** run (so the pipeline is legally conservative first).
3. **Ollama focus group** — If compliant, **five local persona models** (e.g. distinct consumer types) each vote **BUY** or **PASS** on the pitch, contributing a small additive reward.
4. **TRIBE v2 (biological verifier)** — The same text is scored by **TRIBE v2** (`facebook/tribev2`), a foundation model trained on large-scale fMRI and related data; the environment maps predictions to **region-level signals** and combines them with a small **audio-anchor** term so text is not scored in a vacuum.

**What is "different" then?** Many submissions stop at 2 and 3. NeuroPitch's bet is that **(4)** adds a **deterministic, formula-based biological term** (Z-scored regions) that is **not** "another LLM giving a star rating"—it is a separate, neuroscience-informed channel. The **biological** shaping term in the current implementation is:

`Biological_Reward = (Z_STS + Z_TPJ) - Z_Broca_45`

* **STS** (Superior Temporal Sulcus) and **TPJ** (Temporoparietal Junction) — associated with **social and narrative** processing; higher is treated as better in this reward.
* **Broca's area 45** — used here to reflect **linguistic strain / elaboration**; we **penalize** over-activation so the model is not rewarded for needlessly heavy jargon.

The **final** reward combines formatting, vote-based terms, and this biological term, as defined in the environment (see the project README for the exact formula the trainer optimizes against).

**Why this stack:** Compliance reduces harmful or deceptive output; the focus group cheaply injects **diverse "consumer"** preferences; TRIBE injects a **fMRI-inspired** signal. GRPO then updates the policy against the **scalar reward** that encodes all of the above in one pass.

---

## 3. Engineering a robust architecture (what we actually built)

"Robust" here means **reproducible, debuggable, and operable in a real container**—not vague "hygiene" at reset.

* **OpenEnv on the server** — The environment is served over **HTTP + WebSocket** (FastAPI). The training job talks to the same environment your dashboard uses, via a **synchronous** OpenEnv client in the GRPO **reward function**, so each rollout is a real `reset` → `step` against the live server.
* **Alignment between data and server** — Training prompts are built from the same **brief / competitor** pairs the server uses, so the model is not optimising rewards for **random** scenarios while the dataset says something else. That is **state alignment**, not a "database"—it is the difference between a fair RL setup and a silent bug.
* **GRPO + Unsloth (QLoRA) by default** — The default path targets **small** instruct models and **4-bit** adapters to keep runs feasible on a single consumer GPU. Training emits **structured metrics** (e.g. step rewards, per-rollout details) for logging and a **custom dashboard** (HTML/CSS/JS) over **SSE** so you can watch training without only reading raw console spew.
* **Readiness and capacity** — TRIBE and dependencies can **cold-start**; the server exposes a **readiness** endpoint and the training script can **wait** until the biological stack is actually loaded before rollouts, avoiding spurious failures on the first minutes of a run. The OpenEnv server is configured to allow **multiple** concurrent environment sessions (with a declared **concurrent-sessions-safe** flag on the environment class) so transient WebSocket issues do not wedged the whole job.
* **Bounded latency** — Focus-group calls are run with **parallel** requests and per-call limits; TRIBE is wrapped in **bounded** execution so a stuck GPU job cannot hang the **entire** WebSocket step forever. **ffmpeg** in the image supports the audio path TRIBE’s stack expects.

---

## 4. Results & interactive dashboard

**Illustrative runs**: before RL, small models often produce feature-heavy, generic copy; after Unsloth/TRL training inside NeuroPitch, behaviour typically shifts toward **narrative hooks, clearer phrasing, and** (when the stack runs end-to-end) more favourable **panel votes** and **region Z-scores** on average—exact numbers depend on seed, model, and run length.

The UI shows **simulation and training in one place**: product brief, compliance log, **persona votes**, **Z-scores** for the TRIBE regions, a **live training log** with metric lines, a **latest-generation** panel, and a **reward curve** over training steps. Screenshots in the repo illustrate that layout.

*(NeuroPitch environment and training dashboard)*

![NeuroPitch UI Screenshot 1](https://github.com/user-attachments/assets/5041a16b-a6c2-45ae-a890-04d4b2f4bb5b)

![NeuroPitch UI Screenshot 2](https://github.com/user-attachments/assets/800929c1-75fb-4249-8275-153fc6d92f84)

---

## 5. Future directions: expanding the in-silico frontier

The implications of coupling RL agents with biological verifiers like TRIBE v2 extend far beyond textual marketing:

* **Emotionally intelligent medical advisors:** A model tuned for high TPJ-related *empathy* signals in simulation might still be **unsuitable** for real clinical use without safety systems; any serious application needs **clinical, regulatory, and ethical** review, not just neuromarketing loss curves.
* **The CodeCaster System (Video/Slide generation):** TRIBE v2 is **tri-modal** in principle; a natural extension is to score **slides or video** once visual and audio branches are fully wired in the product loop.
* **Multimodal marketing:** Pairing a biologically steered text policy with **image** generators could close the loop from **copy → creative → biological proxy feedback**; that remains **research and product** work, not a claim the current Space implements end-to-end.

The through-line: **NeuroPitch is a system story** (layers + server + training), with **TRIBE** as the sharp, novelty hook—not a single-button "brain score" in isolation.
