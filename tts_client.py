import asyncio
import base64
import collections
import logging
from typing import Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

MISTRAL_TTS_URL = "https://api.mistral.ai/v1/audio/speech"
MODEL = "voxtral-mini-tts-2603"

MISTRAL_VOICES_URL = "https://api.mistral.ai/v1/audio/voices"

# Rate limits: 6 requests/second, 2,000,000 tokens/minute
_RATE_LIMIT_RPS = 6

# ── Local (mlx-audio) ────────────────────────────────────────────────────────

LOCAL_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"

# Preset voices bundled with the Voxtral-4B local model.
LOCAL_VOICES: list[dict] = [
    {"id": "casual_male",     "label": "Casual Male — EN"},
    {"id": "casual_female",   "label": "Casual Female — EN"},
    {"id": "cheerful_female", "label": "Cheerful Female — EN"},
    {"id": "neutral_male",    "label": "Neutral Male — EN"},
    {"id": "neutral_female",  "label": "Neutral Female — EN"},
]

# Module-level singleton — loaded once, reused across all requests.
_local_model = None
_local_model_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _local_model_lock
    if _local_model_lock is None:
        _local_model_lock = asyncio.Lock()
    return _local_model_lock


def _load_and_synthesize(text: str, voice_id: str) -> bytes:
    """Synchronous: load model (once) and synthesize a WAV chunk."""
    import io

    import numpy as np
    import soundfile as sf

    global _local_model
    if _local_model is None:
        from mlx_audio.tts.utils import load
        logger.info("Loading local Voxtral model: %s", LOCAL_MODEL_ID)
        _local_model = load(LOCAL_MODEL_ID)

    parts: list[np.ndarray] = []
    for result in _local_model.generate(text=text, voice=voice_id):
        arr = np.asarray(result.audio).flatten()
        if arr.size:
            parts.append(arr)

    audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class LocalTTSClient:
    """TTS client using mlx-audio for on-device inference (Apple Silicon)."""

    def __init__(self):
        # Sequential — MLX model is not safe for concurrent inference
        self._semaphore = asyncio.Semaphore(1)

    async def synthesize_chunk(self, text: str, voice_id: str) -> bytes:
        loop = asyncio.get_event_loop()
        async with _get_lock():
            return await loop.run_in_executor(None, _load_and_synthesize, text, voice_id)

    async def synthesize_chapter(
        self,
        chunks: list[str],
        voice_id: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[bytes]:
        results: list[bytes] = []
        for chunk in chunks:
            audio = await self.synthesize_chunk(chunk, voice_id)
            results.append(audio)
            if on_progress:
                await on_progress(1)
        return results

    async def close(self):
        pass  # model is a module-level singleton; keep it alive


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
