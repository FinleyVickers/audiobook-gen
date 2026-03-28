import logging
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audio_pipeline import build_m4b, concat_chunks_to_chapter
from epub_parser import Chapter, parse_epub, total_chars
from job_manager import (
    create_job,
    get_job,
    increment_chunks,
    progress_stream,
    update_progress,
)
from tts_client import LOCAL_MODEL_DEFAULT, LOCAL_MODELS, LOCAL_VOICES, LocalTTSClient, TTSClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audiobook Generator")

# Temp storage for parsed epub data between upload and generate
_epub_cache: dict[str, list[Chapter]] = {}
_output_dir = Path(tempfile.gettempdir()) / "audiobook_gen_output"
_output_dir.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/voices/create")
async def create_voice(
    file: UploadFile,
    name: str = "Custom Voice",
    x_api_key: str = Header(..., description="Mistral API key"),
):
    """Upload a reference audio sample to clone a voice via the Mistral API."""
    audio_bytes = await file.read()
    client = TTSClient(api_key=x_api_key)
    try:
        voice = await client.create_voice(
            name=name,
            audio_bytes=audio_bytes,
            filename=file.filename or "sample.wav",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create voice: {e}")
    finally:
        await client.close()
    return voice


@app.get("/api/voices")
async def list_voices(x_api_key: str = Header(..., description="Mistral API key")):
    client = TTSClient(api_key=x_api_key)
    try:
        voices = await client.list_voices()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch voices: {e}")
    finally:
        await client.close()
    return {"voices": voices}


@app.get("/api/local-voices")
async def list_local_voices():
    models = [
        {
            "id": k,
            "label": info["label"],
            "voice_mode": info["voice_mode"],
            "extra_params": info.get("extra_params", []),
        }
        for k, info in LOCAL_MODELS.items()
    ]
    return {"models": models, "default_model": LOCAL_MODEL_DEFAULT}


@app.get("/api/local-voices/{model_key}")
async def list_local_voices_for_model(model_key: str):
    info = LOCAL_MODELS.get(model_key)
    if not info:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_key}")
    return {
        "voices": sorted(info["voices"], key=lambda v: v["label"]),
        "voice_mode": info["voice_mode"],
    }


@app.post("/api/upload-ref-audio")
async def upload_ref_audio(file: UploadFile):
    """Save a reference audio file for local voice-cloning models.
    Returns the server-side path so it can be passed to /api/generate."""
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=_output_dir
    ) as tmp:
        tmp.write(await file.read())
        return {"path": tmp.name}


class UploadResponse(BaseModel):
    job_id: str
    book_title: str
    chapters: list[dict]
    total_chars: int
    estimated_cost_usd: float


@app.post("/api/upload", response_model=UploadResponse)
async def upload_epub(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Only .epub files are supported")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chapters = parse_epub(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"Failed to parse epub: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not chapters:
        raise HTTPException(status_code=422, detail="No readable chapters found in epub")

    book_title = file.filename.replace(".epub", "")
    char_count = total_chars(chapters)
    total_chunks = sum(len(ch.chunks) for ch in chapters)

    job = create_job(total_chunks, len(chapters), book_title)
    _epub_cache[job.id] = chapters

    chapter_info = [
        {"index": i, "title": ch.title, "chunks": len(ch.chunks), "chars": len(ch.text), "boilerplate": ch.boilerplate}
        for i, ch in enumerate(chapters)
    ]

    cost = (char_count / 1000) * 0.016

    return UploadResponse(
        job_id=job.id,
        book_title=book_title,
        chapters=chapter_info,
        total_chars=char_count,
        estimated_cost_usd=round(cost, 4),
    )


class GenerateRequest(BaseModel):
    job_id: str
    voice_id: str
    mode: str = "api"  # "api" or "local"
    local_model: str = LOCAL_MODEL_DEFAULT
    local_ref_audio: str | None = None   # path to reference audio for cloning models
    local_instruct: str | None = None    # voice description for Qwen3-TTS VoiceDesign
    local_speed: float | None = None     # speech speed multiplier (Kokoro, etc.)
    local_lang_code: str | None = None   # language code override (Kokoro: a=US, b=GB, etc.)
    local_exaggeration: float | None = None  # emotion intensity 0–1 (Chatterbox)
    local_cfg_weight: float | None = None    # CFG weight (Chatterbox, default 0.5)
    local_temperature: float | None = None   # sampling temperature (0 = greedy/deterministic)
    chapter_indices: list[int] | None = None  # None means all chapters


@app.post("/api/generate")
async def generate(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(None, description="Mistral API key (required for API mode)"),
):
    if req.mode == "api" and not x_api_key:
        raise HTTPException(status_code=422, detail="X-Api-Key header is required for API mode")

    job = get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in ("pending",):
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    all_chapters = _epub_cache.get(req.job_id)
    if not all_chapters:
        raise HTTPException(status_code=404, detail="Upload data expired — please re-upload")

    # Filter to selected chapters
    if req.chapter_indices is not None:
        selected = sorted(set(req.chapter_indices))
        chapters = [all_chapters[i] for i in selected if 0 <= i < len(all_chapters)]
    else:
        chapters = all_chapters

    if not chapters:
        raise HTTPException(status_code=400, detail="No chapters selected")

    total_chunks = sum(len(ch.chunks) for ch in chapters)
    update_progress(req.job_id, status="processing", total_chunks=total_chunks, total_chapters=len(chapters))
    background_tasks.add_task(
        _run_generation, req.job_id, chapters, req.voice_id, req.mode,
        req.local_model, req.local_ref_audio, req.local_instruct,
        req.local_speed, req.local_lang_code, req.local_exaggeration, req.local_cfg_weight,
        req.local_temperature, x_api_key, job.book_title,
    )
    return {"status": "started"}


async def _run_generation(
    job_id: str,
    chapters: list[Chapter],
    voice_id: str,
    mode: str,
    local_model: str,
    local_ref_audio: str | None,
    local_instruct: str | None,
    local_speed: float | None,
    local_lang_code: str | None,
    local_exaggeration: float | None,
    local_cfg_weight: float | None,
    local_temperature: float | None,
    api_key: str | None,
    book_title: str,
):
    workdir = _output_dir / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    client = (
        LocalTTSClient(
            model_key=local_model,
            ref_audio_path=local_ref_audio,
            instruct=local_instruct,
            speed=local_speed,
            lang_code=local_lang_code,
            exaggeration=local_exaggeration,
            cfg_weight=local_cfg_weight,
            temperature=local_temperature,
        )
        if mode == "local"
        else TTSClient(api_key=api_key)
    )
    chunk_ext = "wav" if mode == "local" else "mp3"
    chapter_data: list[tuple[str, str, float]] = []

    try:
        for idx, chapter in enumerate(chapters):
            if not chapter.chunks:
                continue

            update_progress(
                job_id,
                current_chapter=chapter.title,
                current_chapter_index=idx,
            )

            chapter_workdir = workdir / f"chapter_{idx:04d}"
            chapter_workdir.mkdir(exist_ok=True)

            async def on_progress(_: int, jid=job_id):
                increment_chunks(jid)

            audio_chunks = await client.synthesize_chapter(
                chapter.chunks, voice_id, on_progress=on_progress
            )

            aac_path = str(workdir / f"chapter_{idx:04d}.aac")
            duration = concat_chunks_to_chapter(audio_chunks, aac_path, str(chapter_workdir), chunk_ext)
            chapter_data.append((chapter.title, aac_path, duration))

        # Build final M4B
        output_path = str(workdir / f"{_safe_filename(book_title)}.m4b")
        build_m4b(chapter_data, output_path, str(workdir), book_title)

        update_progress(job_id, status="complete", output_path=output_path)

    except Exception as e:
        logger.exception("Generation failed for job %s", job_id)
        update_progress(job_id, status="error", error=str(e))
    finally:
        await client.close()
        # Cleanup intermediate chapter files
        _epub_cache.pop(job_id, None)


@app.get("/api/progress/{job_id}")
async def progress(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        progress_stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail=f"Job not complete (status: {job.status})")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = Path(job.output_path).name
    return FileResponse(
        path=job.output_path,
        media_type="audio/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_filename(name: str) -> str:
    import re
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip()[:100] or "audiobook"
