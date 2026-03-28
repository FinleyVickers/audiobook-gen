# Audiobook Generator

Convert epub ebooks to M4B audiobooks using [Mistral Voxtral TTS](https://docs.mistral.ai/capabilities/audio/text_to_speech) — either via the Mistral API or fully locally on Apple Silicon using [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Features

- Upload any `.epub` file and get a fully chaptered `.m4b` audiobook
- Select individual chapters to include before generating
- Real-time progress tracking via SSE
- Chapter markers navigable in Apple Books, VLC, etc.
- **API mode**: 20+ voices fetched live from Mistral, cost estimate shown upfront, voice cloning via reference audio upload
- **Local mode**: 12 models from the mlx-audio library — preset-voice models (Voxtral, Kokoro), voice-cloning models (CSM, Chatterbox, Spark, etc.), and lightweight fast models (Soprano, OuteTTS)
- Dark mode

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS)
- ffmpeg (`brew install ffmpeg`)

For **API mode**: a [Mistral AI API key](https://console.mistral.ai/)

For **local mode**: Apple Silicon Mac (M1 or later) with enough RAM for the chosen model (170 MB for Kokoro up to 8 GB for Voxtral bf16)

## Setup

```bash
# Install all dependencies (including local inference extras)
uv sync --extra local

# Run the server
uv run uvicorn main:app --reload
```

Open `http://localhost:8000`.

> **Note:** `mlx-audio` is installed from GitHub (Voxtral support is not yet in a PyPI release). `uv sync` handles this automatically via `pyproject.toml`.

## CLI

A command-line interface is also available for generating audiobooks without a browser:

```bash
# Local inference
uv run python cli.py book.epub --mode local --voice neutral_male

# Mistral API
uv run python cli.py book.epub --mode api --voice <uuid> --api-key sk-...

# List available voices and models
uv run python cli.py --list-voices --mode local
uv run python cli.py --list-voices --mode api --api-key sk-...

# Select specific chapters (0-indexed) and model
uv run python cli.py book.epub --mode local --voice af_bella --local-model kokoro --chapters 0 2 3

# Voice cloning with a reference audio
uv run python cli.py book.epub --mode local --local-model csm-1b --ref-audio narrator.wav
```

## Usage

### API Mode

1. Enter your Mistral API key (stored in your browser's localStorage)
2. Select a voice from the dropdown (loaded live from the API)
3. Optionally clone a narrator's voice by expanding **Clone a narrator voice** and uploading a reference audio clip
4. Upload an `.epub` file
5. Check or uncheck chapters to include — cost estimate updates dynamically
6. Click **Generate Audiobook** and watch the progress bar
7. Download the `.m4b` when complete

### Local Mode

1. Select **Local (mlx-audio · Apple Silicon)** in the Mode section
2. Choose a model from the dropdown — models download automatically on first use from Hugging Face
3. For preset-voice models (Voxtral, Kokoro): pick a voice from the dropdown
4. For voice-cloning models (CSM, Chatterbox, Spark, etc.): optionally upload a reference audio
5. Upload an `.epub`, select chapters, and generate

#### Available local models

| Model | Size | Voices |
|-------|------|--------|
| Voxtral 4B — 4-bit | ~2.5 GB | 20 presets (EN, FR, ES, DE, IT, PT, NL, AR, HI) |
| Voxtral 4B — 6-bit ★ | ~3.5 GB | 20 presets |
| Voxtral 4B — bf16 | ~8.0 GB | 20 presets |
| Kokoro 82M ★ fast | ~170 MB | 11 presets (US/UK m/f) |
| Soprano 80M ★ fast | ~160 MB | single default |
| OuteTTS 0.6B | ~1.2 GB | single default |
| CSM 1B | ~2.0 GB | voice cloning |
| Spark TTS 0.5B | ~1.0 GB | voice cloning |
| Chatterbox | ~0.8 GB | voice cloning |
| Dia 1.6B | ~3.2 GB | voice cloning |
| Qwen3-TTS 1.7B | ~3.4 GB | voice cloning |
| Ming Omni 0.5B | ~1.0 GB | voice cloning |

> Local mode runs chunks sequentially (MLX doesn't support parallel inference). Expect roughly real-time generation speed (~1× RTF).

## Pricing (API Mode)

Mistral charges **$0.016 per 1,000 characters**. A typical novel (~500k chars) costs roughly **$8**.

Local mode is free after the one-time model download.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app and route handlers |
| `epub_parser.py` | Epub parsing, text extraction, sentence-boundary chunking |
| `tts_client.py` | Mistral API client (async, rate-limited) and local MLX client |
| `audio_pipeline.py` | ffmpeg: audio chunks → AAC chapters → M4B with chapter markers |
| `job_manager.py` | In-memory job state and SSE progress stream |
| `cli.py` | Command-line interface |
| `static/index.html` | Single-page web UI |
| `pyproject.toml` | uv project config and dependency declarations |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/voices` | List Mistral API voices (requires `X-Api-Key` header) |
| GET | `/api/local-voices` | List all local models |
| GET | `/api/local-voices/{model_key}` | List preset voices for a specific local model |
| POST | `/api/voices/create` | Clone a voice from a reference audio (API mode, requires `X-Api-Key`) |
| POST | `/api/upload` | Parse epub, return chapter list and cost estimate |
| POST | `/api/upload-ref-audio` | Upload a reference audio file for local voice-cloning models |
| POST | `/api/generate` | Start generation (`mode`: `"api"` or `"local"`) |
| GET | `/api/progress/{job_id}` | SSE stream of generation progress |
| GET | `/api/download/{job_id}` | Download completed `.m4b` |

### Generate request body

```json
{
  "job_id": "...",
  "voice_id": "neutral_male",
  "mode": "local",
  "local_model": "voxtral-6bit",
  "local_ref_audio": null,
  "chapter_indices": [0, 1, 2]
}
```

`chapter_indices` is optional — omit to include all chapters. `X-Api-Key` header is required when `mode` is `"api"`.
