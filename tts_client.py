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
