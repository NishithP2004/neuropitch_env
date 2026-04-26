"""GRPO training entrypoint for the NeuroPitch OpenEnv environment.

Strategy: small model + QLoRA + many short runs beats one big run.
Defaults target Qwen2.5-1.5B-Instruct on a single T4/A10 with ~6 GB VRAM peak.

Unsloth is the default path (faster, less VRAM, `lora_dropout` must be 0).
Import Unsloth before TRL/Transformers when it is enabled.
A small torch.argsort patch avoids CUDA bool-sort errors in Unsloth's GRPO helper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `from client import ...` when running: python scripts/train_grpo_neuropitch.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from client import NeuropitchEnv
from models import NeuropitchAction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROMPT_TEMPLATE = """You are The Brand Strategist in a marketing simulation.
Create a persuasive ad pitch for the product brief while staying compliant.
Output the ad copy as plain text only (no JSON, no tool call syntax).

Constraints:
- Maximum 120 words
- No unverifiable medical, legal, or impossible claims
- Include a clear benefit and call-to-action
"""

# Keep in sync with `server/neuropitch_env_environment.py` DEFAULT_BRIEFS
_DEFAULT_BRIEFS: list[tuple[str, str]] = [
    (
        "Pitch a new plant-based energy drink to urban professionals.",
        "Our drink is clean and tasty, but premium priced for lifestyle consumers.",
    ),
    (
        "Pitch a compact AI note-taking device for students.",
        "Competitor says they are cheaper but never explains battery reliability.",
    ),
    (
        "Pitch eco-friendly detergent pods for families.",
        "Competitor ad highlights low cost but uses generic, bland messaging.",
    ),
]


def _format_env_user_text(product_brief: str, competitor_ad: str) -> str:
    """Match NeuroPitchRolloutEnv.reset() output (same as tokenized user message)."""
    return (
        f"{PROMPT_TEMPLATE}\n\n---\n\n"
        f"Product Brief:\n{product_brief}\n\n"
        f"Competitor Ad:\n{competitor_ad}\n\n"
        "Generate your pitch now."
    )


def _completion_to_str(completion: object) -> str:
    """
    TRL passes `completions` as either plain strings or conversational message lists, e.g.
    [{"role": "assistant", "content": "..."}].
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict):
                c = msg.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
            else:
                parts.append(str(msg))
        return " ".join(parts).strip()
    return str(completion).strip()


def _parse_pitch_text(text: str) -> str:
    """Extract pitch string from decoded text (plain or tool-style JSON)."""
    if not text:
        return ""
    text = text.strip()
    if "pitch_text" in text and "{" in text:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            if 0 <= start < end:
                obj = json.loads(text[start:end])
            else:
                obj = None
            if isinstance(obj, dict):
                args = obj.get("arguments", obj)
                if isinstance(args, dict):
                    p = args.get("pitch_text")
                    if p is not None:
                        return str(p).strip()
                p = obj.get("pitch_text")
                if p is not None:
                    return str(p).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return text


def _patch_torch_argsort_for_cuda_bool() -> None:
    """Unsloth GRPO uses torch.argsort on bool masks; CUDA may not support it."""
    import torch

    if not torch.cuda.is_available():
        return
    _orig = torch.argsort

    def _argsort(t, *args, **kwargs):
        if t is not None and getattr(t, "dtype", None) == torch.bool:
            t = t.to(torch.uint8)
        return _orig(t, *args, **kwargs)

    torch.argsort = _argsort  # type: ignore[assignment]


@dataclass
class NeuroPitchRolloutEnv:
    """Client wrapper for one scoring episode (reset + step)."""

    base_url: str

    def __post_init__(self):
        self.client = NeuropitchEnv(base_url=self.base_url)
        self.reward = 0.0
        self.done = False

    def reset(
        self,
        product_brief: str | None = None,
        competitor_ad: str | None = None,
    ) -> str:
        reset_kw: dict[str, str] = {}
        if product_brief is not None:
            reset_kw["product_brief"] = product_brief
        if competitor_ad is not None:
            reset_kw["competitor_ad"] = competitor_ad
        result = self.client.reset(**reset_kw)
        self.reward = 0.0
        self.done = False
        obs = result.observation
        return _format_env_user_text(obs.product_brief, obs.competitor_ad)

    def submit_pitch(self, pitch_text: str) -> str:
        if self.done:
            raise ValueError("Episode already finished.")
        result = self.client.step(NeuropitchAction(pitch_text=pitch_text))
        obs = result.observation
        self.reward = float(result.reward or obs.final_reward or 0.0)
        self.done = bool(result.done)
        return json.dumps(
            {
                "compliance_status": obs.compliance_status,
                "buy_votes": obs.buy_votes,
                "tribe_scores": obs.tribe_scores,
                "reward": self.reward,
            }
        )


def _make_neuropitch_reward(environment_url: str):
    """
    TRL 0.22.x `reward_func` signature:
    (prompts, completions, completion_ids=None, **kwargs) -> list[float]
    For conversational data, `completions` is a list of assistant message dicts, not raw strings.
    Per-row `product_brief` and `competitor_ad` (from the dataset) must match the server reset kwargs
    so the scored episode matches the prompt the model was trained on.
    """

    def reward_func(
        prompts,
        completions,
        completion_ids=None,
        product_brief=None,
        competitor_ad=None,
        **kwargs,
    ):  # noqa: ARG001
        del prompts
        kwargs.pop("trainer_state", None)
        if product_brief is None or competitor_ad is None:
            raise ValueError(
                "train_dataset must include 'product_brief' and 'competitor_ad' columns (parallel to each prompt)."
            )
        b_list = product_brief if isinstance(product_brief, (list, tuple)) else [product_brief]
        a_list = competitor_ad if isinstance(competitor_ad, (list, tuple)) else [competitor_ad]
        scores: list[float] = []
        for idx, raw in enumerate(completions):
            env: NeuroPitchRolloutEnv | None = None
            try:
                env = NeuroPitchRolloutEnv(base_url=environment_url)
                text = _completion_to_str(raw)
                text = _parse_pitch_text(text)
                b = b_list[idx] if idx < len(b_list) else b_list[-1]
                a = a_list[idx] if idx < len(a_list) else a_list[-1]
                env.reset(product_brief=b, competitor_ad=a)
                if not text:
                    scores.append(-1.0)
                    continue
                env.submit_pitch(text)
                scores.append(float(env.reward))
            except Exception:
                logger.exception("NeuroPitch env scoring failed; assigning penalty")
                scores.append(-2.0)
            finally:
                if env is not None:
                    try:
                        env.client.close()
                    except Exception:
                        pass
        return scores

    return reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NeuroPitch learner with GRPO (Unsloth + QLoRA default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ----------------------------------------------------------------
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model id. Smaller = more runs per budget. 1.5B fits a T4 in 4-bit.",
    )
    parser.add_argument("--output-dir", default="/data/neuropitch-grpo")
    parser.add_argument("--environment-url", default="http://127.0.0.1:8000/openenv")

    # --- LoRA (QLoRA via Unsloth 4-bit) ---------------------------------------
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r. 8 is a good start; try 16 for bigger models.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha. Rule of thumb: 2 × rank.")

    # --- Dataset / training schedule ------------------------------------------
    # num_episodes: dataset size. Must be >= eff_batch * max_steps (TRL repeats automatically).
    parser.add_argument("--num-episodes", type=int, default=600,
                        help="Dataset rows (3 briefs × N). ~600 gives enough repeat variety at 200 steps.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Training steps. More steps = more reward signal at the cost of time.")

    # --- GRPO hyperparameters -------------------------------------------------
    # Effective batch = per_device * grad_accum * processes.
    # Must be divisible by num_generations.
    # Default: 1 * 4 * 1 = 4 generations/step.  4 % 4 == 0.
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Rollouts per prompt. 4-8 recommended for GRPO variance reduction.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of steps used for linear LR warmup.")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL penalty coefficient. Keep small (0.001-0.01) for fine-tuning.")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Generation temperature. Higher = more diverse rollouts.")
    parser.add_argument("--max-prompt-length", type=int, default=512,
                        help="Trim prompts to this many tokens (our prompt is ~300 tokens).")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="120-word pitch ≈ 160 tokens; 256 gives buffer for any model.")

    # --- Backend / misc -------------------------------------------------------
    parser.add_argument(
        "--no-unsloth",
        dest="use_unsloth",
        action="store_false",
        help="Disable Unsloth; use Hugging Face Transformers + TRL only (slower, more VRAM).",
    )
    parser.set_defaults(use_unsloth=True)
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable vLLM backend if installed.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def _effective_batch(
    per_device: int, grad_accum: int, world_size: int = 1
) -> int:
    return per_device * world_size * grad_accum


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # TRL: effective batch must be divisible by num_generations
    eff = _effective_batch(
        args.per_device_train_batch_size, args.gradient_accumulation_steps, 1
    )
    if eff % args.num_generations != 0:
        raise ValueError(
            f"Effective batch ({eff} = per_device * grad_accum * processes) must be "
            f"divisible by --num-generations ({args.num_generations}). "
            f"Adjust --per-device-train-batch-size or --gradient-accumulation-steps."
        )

    has_vllm = importlib.util.find_spec("vllm") is not None
    use_vllm = args.use_vllm and has_vllm
    if args.use_vllm and not has_vllm:
        logger.warning("vLLM not installed; training without vLLM generation.")

    if args.use_unsloth:
        import unsloth  # noqa: F401

        _patch_torch_argsort_for_cuda_bool()
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unsloth is the default. Install it or run with --no-unsloth."
            ) from exc
    else:
        FastLanguageModel = None  # type: ignore[misc, assignment]

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    n = args.num_episodes
    brief_ads = [_DEFAULT_BRIEFS[i % len(_DEFAULT_BRIEFS)] for i in range(n)]
    product_briefs = [b for b, _ in brief_ads]
    competitor_ads = [a for _, a in brief_ads]
    user_texts = [_format_env_user_text(b, a) for b, a in brief_ads]
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": t}] for t in user_texts],
            "product_brief": product_briefs,
            "competitor_ad": competitor_ads,
        }
    )

    processing_class = None

    # max_seq_length: prompt + completion budget
    max_seq_length = args.max_prompt_length + args.max_completion_length

    if args.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[union-attr]
            model_name=args.model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,         # QLoRA: 4-bit base, fp16/bf16 adapters
            load_in_8bit=False,
        )
        model = FastLanguageModel.get_peft_model(  # type: ignore[union-attr]
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,           # must be 0 for Unsloth gradient checkpointing
            bias="none",
            use_gradient_checkpointing="unsloth",  # 30 % less VRAM vs standard
            random_state=3407,
            use_rslora=False,
        )
        processing_class = tokenizer
    else:
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processing_class = AutoTokenizer.from_pretrained(args.model_name)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

    config_kwargs: dict = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,
    )
    if use_vllm:
        config_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.25,
        )

    train_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processing_class,
        reward_funcs=_make_neuropitch_reward(args.environment_url),
        train_dataset=dataset,
        args=train_args,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
