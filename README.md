# Audiobook Generator

Convert epub ebooks to M4B audiobooks using [Mistral Voxtral TTS](https://docs.mistral.ai/capabilities/audio/text_to_speech).

## Features

- Upload any `.epub` file and get a fully chaptered `.m4b` audiobook
- Real-time progress tracking during generation
- Chapter markers navigable in Apple Books, VLC, etc.
- Cost estimate before you commit to generating

## Requirements

- Python 3.11+
- ffmpeg (`brew install ffmpeg` on macOS)
- A [Mistral AI API key](https://console.mistral.ai/)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000`.

## Usage

1. Enter your Mistral API key (stored in your browser's localStorage)
2. Select a voice
3. Upload an `.epub` file
4. Review the chapter list and estimated cost (~$0.016 per 1k characters)
5. Click **Generate Audiobook** and watch the progress
6. Download the `.m4b` when complete

## Pricing

Mistral charges **$0.016 per 1,000 characters**. A typical novel (~500k chars) costs roughly **$8**.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app and route handlers |
| `epub_parser.py` | Epub parsing, text extraction, chunking |
| `tts_client.py` | Async Mistral TTS API client with retry |
| `audio_pipeline.py` | ffmpeg: MP3 chunks → AAC chapters → M4B |
| `job_manager.py` | In-memory job state and SSE progress stream |
| `static/index.html` | Single-page web UI |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/voices` | List available voices |
| POST | `/api/upload` | Parse epub, return chapter info + cost estimate |
| POST | `/api/generate` | Start TTS generation (requires `X-Api-Key` header) |
| GET | `/api/progress/{job_id}` | SSE stream of generation progress |
| GET | `/api/download/{job_id}` | Download completed `.m4b` |
