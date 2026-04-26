# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the NeuroPitch OpenEnv server and web dashboard."""

from __future__ import annotations

import asyncio
import copy
import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from ..models import NeuropitchAction, NeuropitchObservation
    from .neuropitch_env_environment import NeuropitchEnvironment
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    from models import NeuropitchAction, NeuropitchObservation
    from server.neuropitch_env_environment import NeuropitchEnvironment


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
SCRIPT_PATH = BASE_DIR.parent / "scripts" / "train_grpo_neuropitch.py"


class SimulationStepPayload(BaseModel):
    pitch_text: str = Field(..., min_length=1)
    pitch_title: str = ""
    target_product: str = ""


class TrainStartPayload(BaseModel):
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "/data/neuropitch-grpo"
    max_steps: int = 200
    num_episodes: int = 600
    learning_rate: float = 5e-6
    num_generations: int = 2
    gradient_accumulation_steps: int = 2
    push_to_hub: bool = False
    hub_model_id: str = ""
    environment_url: str = "http://127.0.0.1:8000/openenv"
    use_unsloth: bool = True


class TrainingManager:
    """Background trainer process manager with log fanout."""

    def __init__(self):
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._logs = deque(maxlen=5000)
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._status: str = "idle"
        self._exit_code: int | None = None

    def _publish(self, line: str) -> None:
        self._logs.append(line)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(line)
            except asyncio.QueueFull:
                continue

    def start(self, payload: TrainStartPayload) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Training is already running.")
            if not SCRIPT_PATH.exists():
                raise RuntimeError(f"Training script not found at {SCRIPT_PATH}")
            cmd = [
                "python",
                str(SCRIPT_PATH),
                "--model-name", payload.model_name,
                "--output-dir", payload.output_dir,
                "--max-steps", str(payload.max_steps),
                "--num-episodes", str(payload.num_episodes),
                "--learning-rate", str(payload.learning_rate),
                "--num-generations", str(payload.num_generations),
                "--gradient-accumulation-steps", str(payload.gradient_accumulation_steps),
                "--environment-url", payload.environment_url,
            ]
            if payload.push_to_hub:
                cmd.append("--push-to-hub")
            if payload.hub_model_id:
                cmd.extend(["--hub-model-id", payload.hub_model_id])
            if not payload.use_unsloth:
                cmd.append("--no-unsloth")

            # Forward HF_TOKEN so the subprocess can log in to the Hub
            env = copy.copy(os.environ)
            hf_token = os.environ.get("HF_TOKEN", "").strip()
            if hf_token:
                env["HF_TOKEN"] = hf_token

            self._logs.clear()
            self._status = "running"
            self._exit_code = None
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            thread = threading.Thread(target=self._stream_output, daemon=True)
            thread.start()
            return {"status": self._status, "pid": self._process.pid}

    def _stream_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._status = "failed"
            self._publish("Failed to initialize training process output stream.")
            return
        for line in process.stdout:
            self._publish(line.rstrip())
        process.wait()
        self._exit_code = process.returncode
        self._status = "completed" if process.returncode == 0 else "failed"
        self._publish(f"[training-finished] exit_code={process.returncode}")

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return {"status": self._status, "message": "No running process."}
            self._process.terminate()
            self._status = "stopped"
            self._publish("[training-stopped] Process terminated by user.")
            return {"status": "stopped", "pid": self._process.pid}

    def status(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        return {
            "status": self._status,
            "running": running,
            "exit_code": self._exit_code,
            "logs": list(self._logs)[-50:],
        }

    async def subscribe(self):
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        try:
            for log in list(self._logs)[-100:]:
                yield f"data: {log}\n\n"
            while True:
                line = await queue.get()
                yield f"data: {line}\n\n"
        finally:
            self._subscribers.discard(queue)


openenv_app = create_app(
    NeuropitchEnvironment,
    NeuropitchAction,
    NeuropitchObservation,
    env_name="neuropitch_env",
    # Allow multiple concurrent WebSocket sessions.
    # Each GRPO rollout opens its own session sequentially, but a previous timed-out
    # session can stay open server-side while the client has already given up.
    # 8 slots gives plenty of headroom for stragglers without unbounded memory growth.
    max_concurrent_envs=8,
)

app = FastAPI(title="NeuroPitch Server")
app.mount("/openenv", openenv_app)
app.mount("/web/static", StaticFiles(directory=str(WEB_DIR), html=False), name="web-static")

_http_env = NeuropitchEnvironment()
_last_observation: NeuropitchObservation | None = None
_trainer = TrainingManager()


@app.get("/api/env-config")
def api_env_config() -> dict[str, str]:
    """Return safe public config values from the server environment."""
    hf_user = os.environ.get("HF_USERNAME", "").strip()
    return {"hf_username": hf_user}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web")
def web() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard HTML not found.")
    return FileResponse(index_path)


@app.post("/api/sim/reset")
def api_reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    global _last_observation
    kwargs = payload or {}
    _last_observation = _http_env.reset(**kwargs)
    return _last_observation.model_dump()


@app.post("/api/sim/step")
def api_step(payload: SimulationStepPayload) -> dict[str, Any]:
    global _last_observation
    _last_observation = _http_env.step(NeuropitchAction(**payload.model_dump()))
    return _last_observation.model_dump()


@app.get("/api/sim/state")
def api_state() -> dict[str, Any]:
    return {
        "state": _http_env.state.model_dump(),
        "last_observation": _last_observation.model_dump() if _last_observation else None,
    }


@app.post("/api/train/start")
def api_train_start(payload: TrainStartPayload) -> dict[str, Any]:
    try:
        return _trainer.start(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/train/status")
def api_train_status() -> dict[str, Any]:
    return _trainer.status()


@app.post("/api/train/stop")
def api_train_stop() -> dict[str, Any]:
    return _trainer.stop()


@app.get("/api/train/stream")
async def api_train_stream() -> StreamingResponse:
    async def event_stream():
        async for event in _trainer.subscribe():
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m neuropitch_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn neuropitch_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
