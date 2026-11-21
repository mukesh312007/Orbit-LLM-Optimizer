from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any

import psutil
import requests

from core.settings import ROOT, settings
from hardware.energy_capability import joule_delta, read_meter_export


class InferenceError(RuntimeError):
    pass


ENERGY_PROXY_IDLE_WATTS = 10.0
ENERGY_PROXY_FULL_LOAD_WATTS = 45.0


def _llama_process() -> psutil.Process | None:
    """Return Orbit's managed llama.cpp process when it is available."""
    try:
        state = json.loads((ROOT / "runtime" / "active_runtime.json").read_text(encoding="utf-8-sig"))
        process = psutil.Process(int(state["pid"]))
        return process if "llama-server" in process.name().lower() else None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, psutil.Error):
        pass
    # A manually started llama.cpp server can be healthy without Orbit's state
    # file.  Discover it by process name so peak RSS remains an explicit
    # measurement instead of silently becoming null.
    try:
        for process in psutil.process_iter(["name"]):
            if "llama-server" in (process.info.get("name") or "").lower():
                return process
    except psutil.Error:
        pass
    return None


class _ResourceMonitor:
    """Sample the local inference process without claiming hardware joules."""
    def __init__(self, process: psutil.Process | None) -> None:
        self.process = process
        self.ram_samples_mb: list[float] = []
        self.cpu_samples: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        if self.process:
            try:
                self.ram_samples_mb.append(self.process.memory_info().rss / 1024**2)
            except psutil.Error:
                pass
        self.cpu_samples.append(psutil.cpu_percent(interval=None))

    def _run(self) -> None:
        while not self.stop_event.wait(0.05):
            self._sample()

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._sample()
        self.thread.start()

    def finish(self, elapsed_seconds: float) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=1)
        self._sample()
        average_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else None
        estimated_watts = (
            ENERGY_PROXY_IDLE_WATTS + (ENERGY_PROXY_FULL_LOAD_WATTS - ENERGY_PROXY_IDLE_WATTS) * min(100.0, average_cpu) / 100
            if average_cpu is not None else None
        )
        return {
            "peak_ram_mb": max(self.ram_samples_mb) if self.ram_samples_mb else None,
            "average_system_cpu_percent": average_cpu,
            "energy_proxy_watts": estimated_watts,
            "energy_proxy_joules": estimated_watts * elapsed_seconds if estimated_watts is not None else None,
            "energy_proxy_method": "CPU-utilisation estimate; not a hardware energy measurement",
        }


def backend_health(backend: str) -> bool:
    url = settings.llama_server_url + "/health" if backend == "llama.cpp" else settings.ollama_url + "/api/tags"
    try:
        return requests.get(url, timeout=2).ok
    except requests.RequestException:
        return False


def tokenize(text: str, backend: str = "llama.cpp") -> int | None:
    """Ask the local runtime for an exact token count when it exposes one.

    Tokenisation endpoints are version-dependent in llama.cpp.  Returning
    ``None`` makes that absence explicit; callers must not replace it with an
    estimated token count in benchmark evidence.
    """
    if backend != "llama.cpp":
        return None
    try:
        response = requests.post(
            settings.llama_server_url + "/tokenize",
            json={"content": text},
            timeout=min(15, settings.request_timeout_seconds),
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        return len(tokens) if isinstance(tokens, list) else None
    except (requests.RequestException, ValueError):
        return None


def _prompt_processing_metrics(timings: dict[str, Any], usage: dict[str, Any]) -> dict[str, float | int | None]:
    """Expose server-reported prompt timing without deriving unavailable values."""
    prompt_tokens = timings.get("prompt_n") or usage.get("prompt_tokens")
    prompt_ms = timings.get("prompt_ms")
    prompt_tokens_per_second = timings.get("prompt_per_second")
    prompt_seconds = prompt_ms / 1000 if isinstance(prompt_ms, (int, float)) else None
    if prompt_seconds is None and isinstance(prompt_tokens, (int, float)) and isinstance(prompt_tokens_per_second, (int, float)) and prompt_tokens_per_second > 0:
        prompt_seconds = prompt_tokens / prompt_tokens_per_second
    return {
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        "prompt_processing_seconds": prompt_seconds,
        "prompt_tokens_per_second": prompt_tokens_per_second if isinstance(prompt_tokens_per_second, (int, float)) else None,
    }


def generate(model: dict[str, Any], prompt: str, configuration: dict[str, Any]) -> dict[str, Any]:
    backend = configuration.get("backend") or model["backends"][0]
    current_date = datetime.now().astimezone().date().isoformat()
    system_prompt = (
        "You are Orbit, a helpful local AI assistant. "
        f"The current local date is {current_date}. "
        "Use this date for questions about today, the current year, or relative dates."
    )
    start = time.perf_counter()
    meter_start = read_meter_export(settings.energy_meter_reading_path)
    monitor = _ResourceMonitor(_llama_process() if backend == "llama.cpp" else None)
    monitor.start()
    ttft_seconds: float | None = None
    server_timings: dict[str, Any] = {}
    try:
        if backend == "llama.cpp":
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": configuration.get("temperature", 0),
                "max_tokens": configuration.get("max_output_tokens", 256),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            chunks: list[str] = []
            usage: dict[str, Any] = {}
            with requests.post(
                settings.llama_server_url + "/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=settings.request_timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    content = line[5:].strip()
                    if content == "[DONE]":
                        break
                    event = json.loads(content)
                    usage.update(event.get("usage") or {})
                    server_timings.update(event.get("timings") or {})
                    choices = event.get("choices") or []
                    delta = choices[0].get("delta") if choices else {}
                    text = (delta or {}).get("content") or ""
                    if text:
                        if ttft_seconds is None:
                            ttft_seconds = time.perf_counter() - start
                        chunks.append(text)
            answer = "".join(chunks).strip()
            if not answer:
                raise InferenceError("llama.cpp returned a streaming response without generated text.")
        elif backend == "ollama":
            payload = {"model": model.get("backend_model_name", model["model_id"]), "system": system_prompt, "prompt": prompt, "stream": False, "options": {"temperature": configuration.get("temperature", 0), "num_predict": configuration.get("max_output_tokens", 256)}}
            response = requests.post(settings.ollama_url + "/api/generate", json=payload, timeout=settings.request_timeout_seconds)
            response.raise_for_status(); data = response.json()
            answer = data.get("response", "").strip(); usage = {"completion_tokens": data.get("eval_count"), "prompt_tokens": data.get("prompt_eval_count")}
        else:
            raise InferenceError(f"Unsupported backend: {backend}")
    except requests.RequestException as error:
        raise InferenceError(f"{backend} inference unavailable: {error}") from error
    finally:
        elapsed = time.perf_counter() - start
        resources = monitor.finish(elapsed)
    meter_end = read_meter_export(settings.energy_meter_reading_path)
    actual_energy = joule_delta(meter_start, meter_end) if meter_start and meter_end else {
        "energy_joules": None, "measurement_kind": "unavailable", "reason": "No configured physical meter reading was available at both inference boundaries.",
    }
    actual_watts = (
        actual_energy["energy_joules"] / elapsed
        if actual_energy.get("measurement_kind") == "measured" and elapsed > 0 else None
    )
    tokens = usage.get("completion_tokens") or server_timings.get("predicted_n")
    generation_seconds = elapsed - ttft_seconds if ttft_seconds is not None else elapsed
    tokens_per_second = server_timings.get("predicted_per_second") or (tokens / generation_seconds if tokens and generation_seconds > 0 else None)
    return {
        "answer": answer,
        "backend": backend,
        "end_to_end_seconds": elapsed,
        "ttft_seconds": ttft_seconds,
        "generated_tokens": tokens,
        "generation_tokens_per_second": tokens_per_second,
        "measurement_kind": "measured",
        "server_timings": server_timings,
        **_prompt_processing_metrics(server_timings, usage),
        "energy_joules": actual_energy.get("energy_joules"),
        "energy_source": actual_energy.get("source"),
        "energy_measurement_kind": actual_energy.get("measurement_kind"),
        "energy_watts": actual_watts,
        "energy_measurement_reason": actual_energy.get("reason"),
        **resources,
    }
