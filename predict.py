from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from email.parser import BytesParser
from email.policy import default
from pathlib import Path as FSPath

try:
    from cog import BasePredictor, Input, Path
except ImportError:
    class BasePredictor:
        pass

    def Input(default=None, **_kwargs):
        return default

    Path = FSPath


class AceClient:
    def __init__(self, base_url: str, timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 60)) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[-1000:]
            raise RuntimeError(f"ACE-Step returned {error.code}: {body}") from error

    def wait(self, job_id: str):
        deadline = time.monotonic() + self.timeout
        query = urllib.parse.urlencode({"id": job_id})
        while time.monotonic() < deadline:
            body, _ = self.request("GET", "/job?" + query)
            status = json.loads(body).get("status", "")
            if status == "done":
                return
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"ACE-Step job {status}")
            time.sleep(1)
        try:
            self.request("POST", "/job?" + urllib.parse.urlencode({"id": job_id, "cancel": 1}))
        except Exception:
            pass
        raise TimeoutError("ACE-Step generation timed out")

    def run(self, endpoint: str, payload):
        body, _ = self.request("POST", endpoint, payload)
        job_id = str(json.loads(body).get("id", "")).strip()
        if not job_id:
            raise RuntimeError("ACE-Step returned no job id")
        self.wait(job_id)
        query = urllib.parse.urlencode({"id": job_id, "result": 1})
        return self.request("GET", "/job?" + query)


def first_audio_part(body: bytes, content_type: str) -> tuple[bytes, str]:
    if "multipart/" not in content_type.lower():
        return body, content_type.split(";", 1)[0].strip()
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    for part in message.iter_parts():
        part_type = part.get_content_type()
        if part_type.startswith("audio/"):
            return part.get_payload(decode=True), part_type
    raise RuntimeError("ACE-Step synthesis returned no audio part")


class Predictor(BasePredictor):
    def setup(self):
        self.engine = os.getenv("ACESTEP_ENGINE", "native").strip().lower()
        self.server = None
        if self.engine == "contract":
            return
        if self.engine != "native":
            raise RuntimeError(f"unsupported ACESTEP_ENGINE: {self.engine}")
        external_url = os.getenv("ACESTEP_SERVER_URL", "").strip()
        if external_url:
            self.client = AceClient(external_url, self._timeout())
            self._wait_for_health()
            return
        root = FSPath(os.getenv("ACESTEP_CPP_ROOT", "/opt/acestep.cpp"))
        models = FSPath(os.getenv("ACESTEP_MODELS_DIR", "/weights/acestep"))
        adapters = FSPath(os.getenv("ACESTEP_ADAPTERS_DIR", "/weights/adapters"))
        models.mkdir(parents=True, exist_ok=True)
        adapters.mkdir(parents=True, exist_ok=True)
        required = ["vae-", "Qwen3-Embedding-", "acestep-5Hz-lm-", "acestep-v15-turbo-"]
        names = [item.name for item in models.glob("*.gguf")]
        if not all(any(name.startswith(prefix) for name in names) for prefix in required):
            subprocess.run(
                [
                    str(root / "models.sh"),
                    "--quant",
                    os.getenv("ACESTEP_QUANT", "Q8_0"),
                    "--lm",
                    os.getenv("ACESTEP_LM_SIZE", "0.6B"),
                ],
                cwd=root,
                env={**os.environ, "HF_HOME": os.getenv("HF_HOME", "/weights/huggingface")},
                check=True,
            )
            source = root / "models"
            for item in source.glob("*.gguf"):
                target = models / item.name
                if not target.exists():
                    item.replace(target)
        port = int(os.getenv("ACESTEP_PORT", "8085"))
        command = [
            os.getenv("ACESTEP_SERVER_BIN", str(root / "build" / "ace-server")),
            "--host", "127.0.0.1", "--port", str(port), "--models", str(models),
            "--adapters", str(adapters), "--max-batch", "1",
        ]
        if os.getenv("ACESTEP_KEEP_LOADED", "1") == "1":
            command.append("--keep-loaded")
        self.server = subprocess.Popen(command, env={**os.environ, "GGML_BACKEND": os.getenv("ACESTEP_BACKEND", "CUDA0")})
        self.client = AceClient(f"http://127.0.0.1:{port}", self._timeout())
        self._wait_for_health()

    @staticmethod
    def _timeout() -> float:
        return max(30.0, float(os.getenv("ACESTEP_JOB_TIMEOUT", "900")))

    def _wait_for_health(self):
        deadline = time.monotonic() + self._timeout()
        while time.monotonic() < deadline:
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError(f"ace-server exited with status {self.server.returncode}")
            try:
                body, _ = self.client.request("GET", "/health")
                if json.loads(body).get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("ace-server did not become ready")

    @staticmethod
    def _contract_audio(prompt: str, duration: int, output: FSPath) -> None:
        sample_rate = 16000
        frequency = 220 + sum(prompt.encode("utf-8")) % 440
        with wave.open(str(output), "wb") as handle:
            handle.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            frames = bytearray()
            for index in range(sample_rate * min(duration, 2)):
                sample = int(math.sin(2 * math.pi * frequency * index / sample_rate) * 6000)
                frames.extend(struct.pack("<h", sample))
            handle.writeframes(frames)

    def predict(
        self,
        prompt: str = Input(description="Style, instrumentation, mood, and production prompt"),
        lyrics: str = Input(default="[Instrumental]", description="Structured lyrics, blank for generated lyrics, or [Instrumental]"),
        duration: int = Input(default=30, ge=10, le=300),
        bpm: int = Input(default=0, ge=0, le=240),
        key: str = Input(default="", description="Key and scale; blank lets the model choose"),
        seed: int = Input(default=17, ge=0, le=2147483647),
        steps: int = Input(default=8, ge=1, le=50),
        format: str = Input(default="mp3", choices=["mp3", "wav"]),
    ) -> Path:
        prompt = prompt.strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt must contain 1 to 4000 characters")
        if bpm and not 40 <= bpm <= 240:
            raise ValueError("bpm must be zero or between 40 and 240")
        output_dir = FSPath(tempfile.mkdtemp(prefix="acestep-cpp-cog-"))
        if self.engine == "contract":
            output = output_dir / "contract.wav"
            self._contract_audio(prompt, duration, output)
            return Path(output)
        request = {
            "caption": prompt,
            "lyrics": lyrics.strip(),
            "duration": duration,
            "bpm": bpm,
            "keyscale": key.strip(),
            "seed": seed,
            "lm_batch_size": 1,
            "synth_batch_size": 1,
            "inference_steps": steps,
            "guidance_scale": 1.0,
            "shift": 3.0 if steps <= 12 else 1.0,
            "task_type": "text2music",
            "output_format": "mp3" if format == "mp3" else "wav16",
        }
        lm_body, _ = self.client.run("/lm", request)
        enriched = json.loads(lm_body)
        if not isinstance(enriched, list) or not enriched:
            raise RuntimeError("ACE-Step LM returned no synthesis request")
        synth_body, content_type = self.client.run("/synth", enriched[0])
        audio, audio_type = first_audio_part(synth_body, content_type)
        if not audio:
            raise RuntimeError("ACE-Step returned empty audio")
        extension = ".mp3" if audio_type == "audio/mpeg" or format == "mp3" else ".wav"
        output = output_dir / ("music" + extension)
        output.write_bytes(audio)
        return Path(output)
