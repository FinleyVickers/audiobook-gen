import asyncio
import base64
import collections
import logging
import subprocess
from pathlib import Path
from typing import Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

MISTRAL_TTS_URL = "https://api.mistral.ai/v1/audio/speech"
MODEL = "voxtral-mini-tts-2603"

MISTRAL_VOICES_URL = "https://api.mistral.ai/v1/audio/voices"

# Rate limits: 6 requests/second, 2,000,000 tokens/minute
_RATE_LIMIT_RPS = 6

# ── Local (mlx-audio) ────────────────────────────────────────────────────────

_VOXTRAL_VOICES: list[dict] = [
    {"id": "casual_male",     "label": "Casual Male — EN"},
    {"id": "casual_female",   "label": "Casual Female — EN"},
    {"id": "cheerful_female", "label": "Cheerful Female — EN"},
    {"id": "neutral_male",    "label": "Neutral Male — EN"},
    {"id": "neutral_female",  "label": "Neutral Female — EN"},
    {"id": "pt_male",         "label": "Male — PT"},
    {"id": "pt_female",       "label": "Female — PT"},
    {"id": "nl_male",         "label": "Male — NL"},
    {"id": "nl_female",       "label": "Female — NL"},
    {"id": "it_male",         "label": "Male — IT"},
    {"id": "it_female",       "label": "Female — IT"},
    {"id": "fr_male",         "label": "Male — FR"},
    {"id": "fr_female",       "label": "Female — FR"},
    {"id": "es_male",         "label": "Male — ES"},
    {"id": "es_female",       "label": "Female — ES"},
    {"id": "de_male",         "label": "Male — DE"},
    {"id": "de_female",       "label": "Female — DE"},
    {"id": "ar_male",         "label": "Male — AR"},
    {"id": "hi_male",         "label": "Male — HI"},
    {"id": "hi_female",       "label": "Female — HI"},
]

_KOKORO_VOICES: list[dict] = [
    # EN-US Female
    {"id": "af_alloy",    "label": "Alloy — EN-US Female"},
    {"id": "af_aoede",    "label": "Aoede — EN-US Female"},
    {"id": "af_bella",    "label": "Bella — EN-US Female"},
    {"id": "af_heart",    "label": "Heart — EN-US Female  ★ default"},
    {"id": "af_jessica",  "label": "Jessica — EN-US Female"},
    {"id": "af_kore",     "label": "Kore — EN-US Female"},
    {"id": "af_nicole",   "label": "Nicole — EN-US Female"},
    {"id": "af_nova",     "label": "Nova — EN-US Female"},
    {"id": "af_river",    "label": "River — EN-US Female"},
    {"id": "af_sarah",    "label": "Sarah — EN-US Female"},
    {"id": "af_sky",      "label": "Sky — EN-US Female"},
    # EN-US Male
    {"id": "am_adam",     "label": "Adam — EN-US Male"},
    {"id": "am_echo",     "label": "Echo — EN-US Male"},
    {"id": "am_eric",     "label": "Eric — EN-US Male"},
    {"id": "am_fenrir",   "label": "Fenrir — EN-US Male"},
    {"id": "am_liam",     "label": "Liam — EN-US Male"},
    {"id": "am_michael",  "label": "Michael — EN-US Male"},
    {"id": "am_onyx",     "label": "Onyx — EN-US Male"},
    {"id": "am_puck",     "label": "Puck — EN-US Male"},
    {"id": "am_santa",    "label": "Santa — EN-US Male"},
    # EN-GB Female
    {"id": "bf_alice",    "label": "Alice — EN-GB Female"},
    {"id": "bf_emma",     "label": "Emma — EN-GB Female"},
    {"id": "bf_isabella", "label": "Isabella — EN-GB Female"},
    {"id": "bf_lily",     "label": "Lily — EN-GB Female"},
    # EN-GB Male
    {"id": "bm_daniel",   "label": "Daniel — EN-GB Male"},
    {"id": "bm_fable",    "label": "Fable — EN-GB Male"},
    {"id": "bm_george",   "label": "George — EN-GB Male"},
    {"id": "bm_lewis",    "label": "Lewis — EN-GB Male"},
    # Spanish
    {"id": "ef_dora",     "label": "Dora — ES Female"},
    {"id": "em_alex",     "label": "Alex — ES Male"},
    {"id": "em_santa",    "label": "Santa — ES Male"},
    # French
    {"id": "ff_siwis",    "label": "Siwis — FR Female"},
    # Hindi
    {"id": "hf_alpha",    "label": "Alpha — HI Female"},
    {"id": "hf_beta",     "label": "Beta — HI Female"},
    {"id": "hm_omega",    "label": "Omega — HI Male"},
    {"id": "hm_psi",      "label": "Psi — HI Male"},
    # Italian
    {"id": "if_sara",     "label": "Sara — IT Female"},
    {"id": "im_nicola",   "label": "Nicola — IT Male"},
    # Japanese
    {"id": "jf_alpha",    "label": "Alpha — JA Female"},
    {"id": "jf_gongitsune", "label": "Gongitsune — JA Female"},
    {"id": "jf_nezumi",   "label": "Nezumi — JA Female"},
    {"id": "jf_tebukuro", "label": "Tebukuro — JA Female"},
    {"id": "jm_kumo",     "label": "Kumo — JA Male"},
    # Portuguese
    {"id": "pf_dora",     "label": "Dora — PT Female"},
    {"id": "pm_alex",     "label": "Alex — PT Male"},
    {"id": "pm_santa",    "label": "Santa — PT Male"},
    # Chinese
    {"id": "zf_xiaobei",  "label": "Xiaobei — ZH Female"},
    {"id": "zf_xiaoni",   "label": "Xiaoni — ZH Female"},
    {"id": "zf_xiaoxiao", "label": "Xiaoxiao — ZH Female"},
    {"id": "zf_xiaoyi",   "label": "Xiaoyi — ZH Female"},
    {"id": "zm_yunjian",  "label": "Yunjian — ZH Male"},
    {"id": "zm_yunxi",    "label": "Yunxi — ZH Male"},
    {"id": "zm_yunxia",   "label": "Yunxia — ZH Male"},
    {"id": "zm_yunyang",  "label": "Yunyang — ZH Male"},
]

# Models that use reference audio for voice cloning have no preset voices.
# Passing voice_id="" to the CLI omits --voice so the model uses its default.
_QWEN3_CUSTOM_VOICES: list[dict] = [
    {"id": "Vivian",    "label": "Vivian — ZH Female"},
    {"id": "Serena",    "label": "Serena — ZH Female"},
    {"id": "Uncle_Fu",  "label": "Uncle Fu — ZH Male"},
    {"id": "Dylan",     "label": "Dylan — ZH Male"},
    {"id": "Eric",      "label": "Eric — ZH Male"},
    {"id": "Ryan",      "label": "Ryan — EN Male"},
    {"id": "Aiden",     "label": "Aiden — EN Male"},
]

_NO_PRESET_VOICES: list[dict] = []

# Registry of all supported local models.
# voice_mode: "preset"    — select from voices list via --voice
#             "ref_audio" — optionally clone with --ref_audio / --ref_text
#             "none"      — no voice argument; model has a single fixed voice
LOCAL_MODELS: dict[str, dict] = {
    "voxtral-4bit": {
        "hf_id":      "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "label":      "Voxtral 4B — 4-bit  (~2.5 GB)",
        "voice_mode": "preset",
        "voices":     _VOXTRAL_VOICES,
    },
    "voxtral-6bit": {
        "hf_id":      "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
        "label":      "Voxtral 4B — 6-bit  (~3.5 GB)  ★ recommended",
        "voice_mode": "preset",
        "voices":     _VOXTRAL_VOICES,
    },
    "voxtral-bf16": {
        "hf_id":      "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
        "label":      "Voxtral 4B — bf16   (~8.0 GB)",
        "voice_mode": "preset",
        "voices":     _VOXTRAL_VOICES,
    },
    "kokoro": {
        "hf_id":      "mlx-community/Kokoro-82M-bf16",
        "label":      "Kokoro 82M — bf16   (~170 MB)  ★ fast",
        "voice_mode": "preset",
        "voices":     _KOKORO_VOICES,
        "extra_params": ["speed", "lang_code"],
    },
    "soprano": {
        "hf_id":      "mlx-community/Soprano-1.1-80M-bf16",
        "label":      "Soprano 1.1 80M — bf16  (~160 MB)  ★ fast",
        "voice_mode": "none",
        "voices":     _NO_PRESET_VOICES,
    },
    "outetts": {
        "hf_id":      "mlx-community/OuteTTS-1.0-0.6B-fp16",
        "label":      "OuteTTS 1.0 0.6B — fp16  (~1.2 GB)  · voice cloning optional",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "csm-1b": {
        "hf_id":      "mlx-community/csm-1b",
        "label":      "CSM 1B              (~2.0 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "spark": {
        "hf_id":      "mlx-community/Spark-TTS-0.5B-bf16",
        "label":      "Spark TTS 0.5B — bf16  (~1.0 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "chatterbox": {
        "hf_id":      "mlx-community/chatterbox-fp16",
        "label":      "Chatterbox — fp16   (~0.8 GB)  · voice cloning + emotion",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
        "extra_params": ["exaggeration", "cfg_weight"],
    },
    "dia-1.6b": {
        "hf_id":      "mlx-community/Dia-1.6B-fp16",
        "label":      "Dia 1.6B — fp16     (~3.2 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "qwen3-tts": {
        "hf_id":      "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "label":      "Qwen3-TTS 1.7B VoiceDesign — bf16  (~3.4 GB)  · instruct",
        # voice_mode "instruct": user provides a natural-language voice description
        # via --instruct rather than picking a preset name or uploading audio.
        # Also supports --ref_audio for speaker cloning.
        "voice_mode": "instruct",
        "voices":     _NO_PRESET_VOICES,
    },
    "qwen3-tts-custom": {
        "hf_id":      "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "label":      "Qwen3-TTS 1.7B CustomVoice — bf16  (~3.4 GB)  · preset + emotion",
        "voice_mode": "preset",
        "voices":     _QWEN3_CUSTOM_VOICES,
        "extra_params": ["instruct"],
    },
    "qwen3-tts-base": {
        "hf_id":      "mlx-community/Qwen3-TTS-12Hz-1.7B-bf16",
        "label":      "Qwen3-TTS 1.7B Base — bf16  (~3.4 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "qwen3-tts-0.6b": {
        "hf_id":      "mlx-community/Qwen3-TTS-12Hz-0.6B-bf16",
        "label":      "Qwen3-TTS 0.6B Base — bf16  (~1.2 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "qwen3-tts-0.6b-custom": {
        "hf_id":      "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "label":      "Qwen3-TTS 0.6B CustomVoice — bf16  (~1.2 GB)  · preset + emotion",
        "voice_mode": "preset",
        "voices":     _QWEN3_CUSTOM_VOICES,
        "extra_params": ["instruct"],
    },
    "ming-omni-0.5b": {
        "hf_id":      "mlx-community/Ming-omni-tts-0.5B-bf16",
        "label":      "Ming Omni 0.5B — bf16  (~1.0 GB)  · voice cloning",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
    },
    "fish-s2-pro-8bit": {
        "hf_id":      "mlx-community/fish-audio-s2-pro-8bit",
        "label":      "Fish Audio S2 Pro — 8-bit  (~6.7 GB)  · voice cloning + inline tags",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
        "extra_params": ["speed"],
    },
    "fish-s2-pro-bf16": {
        "hf_id":      "mlx-community/fish-audio-s2-pro-bf16",
        "label":      "Fish Audio S2 Pro — bf16  (~10 GB)  · voice cloning + inline tags",
        "voice_mode": "ref_audio",
        "voices":     _NO_PRESET_VOICES,
        "extra_params": ["speed"],
    },
}

LOCAL_MODEL_DEFAULT = "voxtral-6bit"

# Flat voice list for the default model (backward-compat for callers that
# imported LOCAL_VOICES before the per-model registry existed).
LOCAL_VOICES: list[dict] = LOCAL_MODELS[LOCAL_MODEL_DEFAULT]["voices"]

_MIRROR_DIR = Path(__file__).parent / "local_audio_output"


def _cli_synthesize_chunk(
    text: str,
    voice_id: str,
    model_id: str,
    ref_audio_path: str | None = None,
    instruct: str | None = None,
    speed: float | None = None,
    lang_code: str | None = None,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float | None = None,
) -> bytes:
    """Synthesize a single chunk via the mlx_audio CLI subprocess."""
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "mlx_audio.tts.generate",
            "--model", model_id,
            "--text", text,
            "--output_path", tmpdir,
            "--file_prefix", "chunk",
            "--audio_format", "wav",
        ]
        # Only add --voice when the model uses preset voices and one is selected
        if voice_id and voice_id != "default":
            cmd += ["--voice", voice_id]
        # Reference audio for voice-cloning models (optional)
        if ref_audio_path:
            cmd += ["--ref_audio", ref_audio_path]
        # Natural-language voice description for Qwen3-TTS VoiceDesign (optional)
        if instruct:
            cmd += ["--instruct", instruct]
        # Speed multiplier (Kokoro, others)
        if speed is not None and speed != 1.0:
            cmd += ["--speed", str(speed)]
        # Language code override (Kokoro: a=US, b=GB, e=ES, f=FR, h=HI, i=IT, j=JA, p=PT, z=ZH)
        if lang_code:
            cmd += ["--lang_code", lang_code]
        # Chatterbox emotion exaggeration (0.0–1.0)
        if exaggeration is not None:
            cmd += ["--exaggeration", str(exaggeration)]
        # Chatterbox CFG weight
        if cfg_weight is not None:
            cmd += ["--cfg_weight", str(cfg_weight)]
        # Temperature for sampling (0 = greedy/deterministic)
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"mlx_audio CLI exited with code {result.returncode}")

        wav_files = sorted(Path(tmpdir).glob("*.wav"))
        if not wav_files:
            raise RuntimeError("mlx_audio CLI produced no output files")

        return wav_files[0].read_bytes()


class LocalTTSClient:
    """TTS client using mlx-audio CLI for on-device inference (Apple Silicon).

    Each chunk is synthesized in its own subprocess to avoid the memory
    accumulation crash in the Python API, while keeping text short enough
    that the model produces intelligible output.

    model_key: key from LOCAL_MODELS registry (e.g. "voxtral-6bit", "kokoro")
    ref_audio_path: optional path to a reference audio file for voice-cloning models
    """

    def __init__(
        self,
        model_key: str = LOCAL_MODEL_DEFAULT,
        ref_audio_path: str | None = None,
        instruct: str | None = None,
        speed: float | None = None,
        lang_code: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
    ):
        info = LOCAL_MODELS.get(model_key, LOCAL_MODELS[LOCAL_MODEL_DEFAULT])
        self._model_id = info["hf_id"]
        self._ref_audio_path = ref_audio_path
        self._instruct = instruct
        self._speed = speed
        self._lang_code = lang_code
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        # Default to temperature=0 (greedy) for instruct/VoiceDesign models
        # so the voice description resolves to the same speaker every chunk.
        if temperature is None and info.get("voice_mode") == "instruct":
            temperature = 0.0
        self._temperature = temperature

    async def synthesize_chapter(
        self,
        chunks: list[str],
        voice_id: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[bytes]:
        results: list[bytes] = []
        loop = asyncio.get_event_loop()
        for i, chunk in enumerate(chunks):
            logger.info("Local TTS: chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            audio = await loop.run_in_executor(
                None, _cli_synthesize_chunk, chunk, voice_id,
                self._model_id, self._ref_audio_path, self._instruct,
                self._speed, self._lang_code, self._exaggeration, self._cfg_weight,
                self._temperature,
            )
            results.append(audio)

            # Mirror to project folder for inspection
            _MIRROR_DIR.mkdir(exist_ok=True)
            import time
            mirror_path = _MIRROR_DIR / f"{int(time.time() * 1000)}_chunk{i:04d}_{voice_id}.wav"
            mirror_path.write_bytes(audio)

            if on_progress:
                await on_progress(1)
        return results

    async def close(self):
        pass


class _RateLimiter:
    """Sliding-window rate limiter: max `rate` calls per `period` seconds."""

    def __init__(self, rate: int, period: float = 1.0):
        self._rate = rate
        self._period = period
        self._timestamps: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            # Evict timestamps outside the sliding window
            while self._timestamps and now - self._timestamps[0] >= self._period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._rate:
                wait = self._period - (now - self._timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = loop.time()
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()
            self._timestamps.append(loop.time())

    async def __aexit__(self, *_):
        pass


class TTSClient:
    def __init__(self, api_key: str, max_concurrent: int = 6):
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = _RateLimiter(rate=_RATE_LIMIT_RPS)
        self._client = httpx.AsyncClient(timeout=60.0)

    async def create_voice(
        self,
        name: str,
        audio_bytes: bytes,
        filename: str = "sample.wav",
        **metadata,
    ) -> dict:
        """Upload a reference audio sample to create a cloned voice.

        Returns the voice dict from the Mistral API (includes ``id`` and ``name``).
        """
        import base64 as _b64
        payload: dict = {
            "name": name,
            "sample_audio": _b64.b64encode(audio_bytes).decode(),
            "sample_filename": filename,
        }
        payload.update({k: v for k, v in metadata.items() if v is not None})
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = await self._client.post(
            MISTRAL_VOICES_URL, json=payload, headers=headers
        )
        response.raise_for_status()
        v = response.json()
        # Build a label the same way list_voices does
        parts = [v.get("name", name)]
        if v.get("languages"):
            parts.append(", ".join(v["languages"]).upper())
        if v.get("gender"):
            parts.append(v["gender"].capitalize())
        label = " — ".join(parts) if len(parts) > 1 else parts[0]
        return {"id": v["id"], "label": label, "raw": v}

    async def list_voices(self) -> list[dict]:
        """Fetch available voices from the Mistral API. Returns list of {id, label} dicts."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        all_voices: list[dict] = []
        offset = 0
        limit = 50

        while True:
            response = await self._client.get(
                MISTRAL_VOICES_URL,
                headers=headers,
                params={"limit": limit, "offset": offset},
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            for v in items:
                parts = [v["name"]]
                if v.get("languages"):
                    parts.append(", ".join(v["languages"]).upper())
                if v.get("gender"):
                    parts.append(v["gender"].capitalize())
                label = " — ".join(parts) if len(parts) > 1 else parts[0]
                all_voices.append({"id": v["id"], "label": label})

            if len(items) < limit:
                break
            offset += limit

        return all_voices

    async def close(self):
        await self._client.aclose()

    async def synthesize_chunk(self, text: str, voice_id: str) -> bytes:
        payload = {
            "model": MODEL,
            "input": text,
            "voice_id": voice_id,
            "response_format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                async with self._rate_limiter:
                    async with self._semaphore:
                        response = await self._client.post(
                            MISTRAL_TTS_URL, json=payload, headers=headers
                        )

                if response.status_code == 429 or response.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        "TTS API returned %d, retrying in %ds", response.status_code, wait
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                return base64.b64decode(data["audio_data"])

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("TTS request failed (%s), retrying in %ds", e, wait)
                await asyncio.sleep(wait)

        raise RuntimeError(f"TTS synthesis failed after 3 attempts for text: {text[:50]!r}")

    async def synthesize_chapter(
        self,
        chunks: list[str],
        voice_id: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[bytes]:
        results: list[bytes | None] = [None] * len(chunks)

        async def do_chunk(i: int, text: str):
            audio = await self.synthesize_chunk(text, voice_id)
            results[i] = audio
            if on_progress:
                await on_progress(1)

        tasks = [do_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)

        return [r for r in results if r is not None]
