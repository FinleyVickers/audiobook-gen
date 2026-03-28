# Audiobook Generator

Convert epub ebooks to M4B audiobooks using [Mistral Voxtral TTS](https://docs.mistral.ai/capabilities/audio/text_to_speech) — either via the Mistral API or fully locally on Apple Silicon using [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Features

- Upload any `.epub` file and get a fully chaptered `.m4b` audiobook
- Select individual chapters to include before generating
- Real-time progress tracking via SSE
- Chapter markers navigable in Apple Books, VLC, etc.
- **API mode**: 20+ voices fetched live from Mistral, cost estimate shown upfront
- **Local mode**: on-device inference with Voxtral-4B-TTS-2603 (4-bit, ~2.5 GB) — no API key or internet required
- Dark mode

## Requirements

- Python 3.11+
- ffmpeg (`brew install ffmpeg` on macOS)

For **API mode**:
- A [Mistral AI API key](https://console.mistral.ai/)

For **local mode**:
- Apple Silicon Mac (M1 or later)
- 16 GB RAM recommended (model is ~2.5 GB at 4-bit quantization)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000`.

## Usage

### API Mode

1. Enter your Mistral API key (stored in your browser's localStorage)
2. Select a voice from the dropdown (loaded live from the API)
3. Upload an `.epub` file
4. Check or uncheck chapters to include — cost estimate updates dynamically
5. Click **Generate Audiobook** and watch the progress bar
6. Download the `.m4b` when complete

### Local Mode

1. Select **Local (mlx-audio · Apple Silicon)** in the Mode section
2. Select a voice — 20 presets across 9 languages including English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi
3. Upload an `.epub`, select chapters, and generate
4. The model (`mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`, ~2.5 GB) downloads automatically on first use from Hugging Face

> Local mode runs chunks sequentially (MLX doesn't support parallel inference). Expect roughly real-time generation speed (~1× RTF) on an M2 with the 4-bit model.

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
| `static/index.html` | Single-page web UI |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/voices` | List Mistral API voices (requires `X-Api-Key` header) |
| GET | `/api/local-voices` | List local Voxtral-4B preset voices |
| POST | `/api/upload` | Parse epub, return chapter list and cost estimate |
| POST | `/api/generate` | Start generation (`mode`: `"api"` or `"local"`) |
| GET | `/api/progress/{job_id}` | SSE stream of generation progress |
| GET | `/api/download/{job_id}` | Download completed `.m4b` |

### Generate request body

```json
{
  "job_id": "...",
  "voice_id": "...",
  "mode": "api",
  "chapter_indices": [0, 1, 2]
}
```

`chapter_indices` is optional — omit to include all chapters. `X-Api-Key` header is required when `mode` is `"api"`.
