#!/usr/bin/env python3
"""
Audiobook Generator — CLI

Usage examples:
  # Local inference (no API key needed)
  python cli.py book.epub --mode local --voice casual_male

  # Mistral API
  python cli.py book.epub --mode api --voice <uuid> --api-key sk-...

  # List voices
  python cli.py --list-voices --mode local
  python cli.py --list-voices --mode api --api-key sk-...

  # Select specific chapters (0-indexed)
  python cli.py book.epub --mode local --voice casual_male --chapters 0 2 3

  # Choose local model quality
  python cli.py book.epub --mode local --voice casual_male --local-model bf16
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path


def _print_voices_local():
    from tts_client import LOCAL_MODELS
    print("\nLocal models and voices:")
    for key, info in LOCAL_MODELS.items():
        print(f"\n  {key:<18} {info['label']}")
        if info["voice_mode"] == "preset" and info["voices"]:
            for v in sorted(info["voices"], key=lambda x: x["label"]):
                print(f"    {v['id']:<20} {v['label']}")
        elif info["voice_mode"] == "ref_audio":
            print(f"    (voice cloning — provide --ref-audio path)")
        else:
            print(f"    (single default voice)")


async def _print_voices_api(api_key: str):
    from tts_client import TTSClient
    client = TTSClient(api_key=api_key)
    try:
        voices = await client.list_voices()
    finally:
        await client.close()
    print("\nAPI voices:")
    for v in sorted(voices, key=lambda x: x["label"]):
        print(f"  {v['id']}  {v['label']}")


def _parse_chapters(chapters_arg: list[int] | None, all_chapters):
    if chapters_arg is not None:
        selected = [all_chapters[i] for i in sorted(set(chapters_arg)) if 0 <= i < len(all_chapters)]
        if not selected:
            print("Error: no valid chapter indices provided.", file=sys.stderr)
            sys.exit(1)
        return selected

    # Interactive selection
    print("\nChapters:")
    for i, ch in enumerate(all_chapters):
        bp = " (boilerplate)" if ch.boilerplate else ""
        print(f"  [{i:>3}] {ch.title}{bp}  —  {len(ch.chunks)} passages")

    print("\nEnter chapter numbers to include (space-separated), or press Enter for all non-boilerplate:")
    raw = input("> ").strip()
    if not raw:
        selected = [ch for ch in all_chapters if not ch.boilerplate]
        print(f"Selected {len(selected)} chapters.")
        return selected

    indices = []
    for part in raw.split():
        try:
            indices.append(int(part))
        except ValueError:
            print(f"Ignoring invalid index: {part!r}", file=sys.stderr)

    return [all_chapters[i] for i in sorted(set(indices)) if 0 <= i < len(all_chapters)]


async def _run(args):
    from epub_parser import parse_epub
    from audio_pipeline import build_m4b, concat_chunks_to_chapter
    from tts_client import LocalTTSClient, TTSClient

    # ── Parse epub ────────────────────────────────────────────
    print(f"Parsing {args.epub.name} …")
    all_chapters = parse_epub(args.epub.name)
    if not all_chapters:
        print("Error: no readable chapters found.", file=sys.stderr)
        sys.exit(1)

    total_chars = sum(len(ch.text) for ch in all_chapters)
    print(f"Found {len(all_chapters)} chapters  ({total_chars:,} chars)")
    if args.mode == "api":
        print(f"Estimated cost: ${(total_chars / 1000) * 0.016:.4f}")

    # ── Chapter selection ─────────────────────────────────────
    chapters = _parse_chapters(args.chapters, all_chapters)
    if not chapters:
        print("No chapters selected.", file=sys.stderr)
        sys.exit(1)

    # ── Output path ───────────────────────────────────────────
    book_title = args.epub.name.replace(".epub", "")
    safe_title = _safe_filename(Path(book_title).name)
    output_path = args.output or Path.cwd() / f"{safe_title}.m4b"
    output_path = Path(output_path)

    # ── TTS client ────────────────────────────────────────────
    if args.mode == "local":
        local_model = args.local_model
        if not local_model:
            from tts_client import LOCAL_MODELS, LOCAL_MODEL_DEFAULT
            print("\nLocal model (enter number):")
            model_list = list(LOCAL_MODELS.items())
            for i, (key, info) in enumerate(model_list, 1):
                default_mark = " ★" if key == LOCAL_MODEL_DEFAULT else ""
                print(f"  [{i:>2}] {key:<18} {info['label']}{default_mark}")
            default_idx = next(
                (i for i, (k, _) in enumerate(model_list, 1) if k == LOCAL_MODEL_DEFAULT), 1
            )
            choice = input(f"Select [1-{len(model_list)}] (default {default_idx}): ").strip()
            try:
                idx = int(choice) - 1
                local_model = model_list[idx][0] if 0 <= idx < len(model_list) else LOCAL_MODEL_DEFAULT
            except (ValueError, IndexError):
                local_model = LOCAL_MODEL_DEFAULT
            print(f"Using: {local_model}")
        client = LocalTTSClient(
            model_key=local_model, ref_audio_path=args.ref_audio, instruct=args.instruct,
            speed=args.speed, lang_code=args.lang_code, exaggeration=args.exaggeration,
        )
        chunk_ext = "wav"
    else:
        if not args.api_key:
            print("Error: --api-key is required for API mode.", file=sys.stderr)
            sys.exit(1)
        client = TTSClient(api_key=args.api_key)
        chunk_ext = "mp3"

    # ── Generate ──────────────────────────────────────────────
    workdir = Path(tempfile.mkdtemp(prefix="audiobook_cli_"))
    chapter_data = []

    try:
        for idx, chapter in enumerate(chapters):
            if not chapter.chunks:
                continue

            print(f"\n[{idx + 1}/{len(chapters)}] {chapter.title}  ({len(chapter.chunks)} passages)")

            completed = 0
            total = len(chapter.chunks)

            async def on_progress(_: int, _total=total):
                nonlocal completed
                completed += 1
                pct = int(completed / _total * 100)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  {bar} {pct:>3}%  ({completed}/{_total})", end="", flush=True)

            chapter_workdir = workdir / f"chapter_{idx:04d}"
            chapter_workdir.mkdir()

            audio_chunks = await client.synthesize_chapter(
                chapter.chunks, args.voice, on_progress=on_progress
            )
            print()  # newline after progress bar

            aac_path = str(workdir / f"chapter_{idx:04d}.aac")
            duration = concat_chunks_to_chapter(audio_chunks, aac_path, str(chapter_workdir), chunk_ext)
            chapter_data.append((chapter.title, aac_path, duration))

        print(f"\nBuilding M4B …")
        build_m4b(chapter_data, str(output_path), str(workdir), book_title)
        print(f"Done: {output_path}")

    finally:
        await client.close()
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _safe_filename(name: str) -> str:
    import re
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()[:100] or "audiobook"


def main():
    parser = argparse.ArgumentParser(
        description="Convert an epub to an M4B audiobook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("epub", nargs="?", type=argparse.FileType("r"),
                        help="Path to the .epub file")
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                        help="Inference mode (default: local)")
    parser.add_argument("--voice", default="neutral_male",
                        help="Voice ID (default: neutral_male for Voxtral; use --list-voices to see options)")
    parser.add_argument("--local-model", default=None, metavar="MODEL_KEY",
                        help="Local model key (e.g. voxtral-6bit, kokoro, csm-1b). "
                             "Omit to be prompted interactively. Use --list-voices to see all options.")
    parser.add_argument("--ref-audio", default=None, metavar="PATH",
                        help="Reference audio file for voice-cloning local models (csm-1b, chatterbox, etc.)")
    parser.add_argument("--instruct", default=None, metavar="TEXT",
                        help="Natural-language voice description for Qwen3-TTS VoiceDesign "
                             "(e.g. 'A warm, calm male narrator with a slight British accent')")
    parser.add_argument("--speed", type=float, default=None, metavar="FLOAT",
                        help="Speech speed multiplier, e.g. 1.2 for 20%% faster (Kokoro, others)")
    parser.add_argument("--lang-code", default=None, metavar="CODE",
                        help="Language code override for Kokoro (a=US, b=GB, e=ES, f=FR, "
                             "h=HI, i=IT, j=JA, p=PT, z=ZH). Auto-detected from voice ID if omitted.")
    parser.add_argument("--exaggeration", type=float, default=None, metavar="FLOAT",
                        help="Emotion exaggeration 0.0–1.0 for Chatterbox (default: 0.5)")
    parser.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY"),
                        help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--chapters", nargs="+", type=int, metavar="N",
                        help="Chapter indices to include (0-based). Omit for interactive selection.")
    parser.add_argument("--output", type=Path, metavar="PATH",
                        help="Output .m4b path (default: <title>.m4b in current directory)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")

    args = parser.parse_args()

    if args.list_voices:
        if args.mode == "local":
            _print_voices_local()
        else:
            asyncio.run(_print_voices_api(args.api_key or ""))
        return

    if not args.epub:
        parser.error("epub file is required unless --list-voices is specified")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
