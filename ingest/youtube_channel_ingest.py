#!/usr/bin/env python3


from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

# Quiet noisy macOS LibreSSL warning (doesn't change behavior)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")


# -----------------------------
# Helpers
# -----------------------------

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def upload_date_to_iso(upload_date: Any) -> Optional[str]:
    if not upload_date:
        return None
    if isinstance(upload_date, str) and re.fullmatch(r"\d{8}", upload_date):
        return f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    if isinstance(upload_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", upload_date):
        return upload_date
    return None

def safe_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def normalize_channel_input(channel: str) -> str:
    channel = channel.strip()
    if channel.startswith("http://") or channel.startswith("https://"):
        return channel
    if channel.startswith("@"):
        return f"https://www.youtube.com/{channel}/videos"
    return f"https://www.youtube.com/@{channel}/videos"

def within_window(upload_date_iso: Optional[str], since: Optional[dt.date], until: Optional[dt.date]) -> bool:
    if not since and not until:
        return True
    if not upload_date_iso:
        return True
    d = dt.datetime.strptime(upload_date_iso, "%Y-%m-%d").date()
    if since and d < since:
        return False
    if until and d > until:
        return False
    return True


# -----------------------------
# Index row
# -----------------------------

@dataclass
class IndexRow:
    video_id: str
    video_url: str
    title: str = ""
    channel: str = ""
    channel_id: str = ""
    uploader_id: str = ""
    upload_date: str = ""          # YYYY-MM-DD
    duration_sec: str = ""
    view_count: str = ""
    like_count: str = ""
    comment_count: str = ""
    availability: str = ""
    extracted_at: str = ""

    transcript_status: str = ""    # ok / missing / blocked / error / skipped
    transcript_language_code: str = ""
    transcript_is_generated: str = ""
    transcript_source: str = ""    # yta_fetch / yta_list / ydl_vtt
    transcript_error: str = ""

    rendered_path: str = ""
    transcript_path: str = ""
    raw_video_meta_path: str = ""


# -----------------------------
# yt-dlp configs
# -----------------------------

def ydl_inventory() -> yt_dlp.YoutubeDL:
    return yt_dlp.YoutubeDL({
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extract_flat": "in_playlist",
    })

def ydl_video() -> yt_dlp.YoutubeDL:
    return yt_dlp.YoutubeDL({
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "noplaylist": True,
    })


# -----------------------------
# Inventory
# -----------------------------

def extract_inventory(channel_url: str, out_dir: Path, force: bool) -> Dict[str, Any]:
    inv_path = out_dir / "raw" / "channel_inventory.json"
    if inv_path.exists() and not force:
        return read_json(inv_path)
    with ydl_inventory() as ydl:
        inv = ydl.extract_info(channel_url, download=False)
    write_json(inv_path, inv)
    return inv

def inventory_to_video_pairs(inv: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for e in (inv.get("entries") or []):
        if not e:
            continue
        vid = e.get("id")
        if not vid:
            continue
        url = e.get("url")
        if not url or not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={vid}"
        pairs.append((vid, url))
    return pairs


# -----------------------------
# Transcript normalization
# -----------------------------

def transcript_to_segments(transcript: Any) -> List[Dict[str, Any]]:
    """
    Normalizes different transcript schemas into:
      [{ "text": str, "start": float, "duration": float }, ...]

    Supports:
      - {"segments":[...]}           (this script's preferred schema)
      - {"snippets":[...]}           (youtube-transcript-api 1.2.x docs)
      - legacy list of dicts         ([{"text","start","duration"}, ...])
      - youtube-transcript-api raw   (same as legacy list in older versions)
    """
    if not transcript:
        return []

    if isinstance(transcript, list):
        out = []
        for x in transcript:
            if not isinstance(x, dict):
                continue
            txt = (x.get("text") or "").strip()
            if not txt:
                continue
            out.append({
                "text": txt,
                "start": float(x.get("start") or 0.0),
                "duration": float(x.get("duration") or 0.0),
            })
        return out

    if not isinstance(transcript, dict):
        return []

    segs = transcript.get("segments")
    if isinstance(segs, list) and segs:
        return segs

    snips = transcript.get("snippets")
    if isinstance(snips, list) and snips:
        out = []
        for sn in snips:
            if not isinstance(sn, dict):
                continue
            txt = (sn.get("text") or "").strip()
            if not txt:
                continue
            out.append({
                "text": txt,
                "start": float(sn.get("start") or 0.0),
                "duration": float(sn.get("duration") or 0.0),
            })
        return out

    # Some users store under "transcript" key
    inner = transcript.get("transcript")
    if isinstance(inner, (list, dict)):
        return transcript_to_segments(inner)

    return []


# -----------------------------
# Transcript fetchers
# -----------------------------

def classify_transcript_error(msg: str) -> str:
    lower = msg.lower()
    if any(k in lower for k in ["requestblocked", "ipblocked", "429", "too many requests", "captcha"]):
        return "blocked"
    if any(k in lower for k in ["transcriptsdisabled", "subtitles are disabled", "no transcripts", "no transcript"]):
        return "missing"
    return "error"

def build_transcript_payload(
    video_id: str,
    segments: List[Dict[str, Any]],
    language_code: str = "",
    is_generated: Optional[bool] = None,
    source: str = "",
) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "language_code": language_code,
        "is_generated": is_generated,
        "source": source,
        "fetched_at": now_utc_iso(),
        "segments": segments,
    }

def yta_fetch_original(video_id: str) -> Tuple[str, Optional[Dict[str, Any]], str, str, str]:
    """
    youtube-transcript-api v1.2.x: fetch() with NO languages => original/default transcript.
    """
    ytt = YouTubeTranscriptApi()
    try:
        fetched = ytt.fetch(video_id)  # <- no forcing english
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else []
        # raw from v1.2.x is list of dicts under raw_data interface
        segments = transcript_to_segments(raw)
        lang = getattr(fetched, "language_code", "") or ""
        is_gen = getattr(fetched, "is_generated", None)
        payload = build_transcript_payload(video_id, segments, lang, is_gen, source="yta_fetch")
        return "ok", payload, lang, ("" if is_gen is None else str(bool(is_gen))), ""
    except Exception as e:
        msg = f"{e.__class__.__name__}: {str(e)}"
        return classify_transcript_error(msg), None, "", "", msg

def yta_list_first(video_id: str) -> Tuple[str, Optional[Dict[str, Any]], str, str, str]:
    """
    If fetch() fails, list available transcripts and fetch the first.
    """
    ytt = YouTubeTranscriptApi()
    try:
        tl = ytt.list(video_id)
        first = None
        for t in tl:
            first = t
            break
        if first is None:
            return "missing", None, "", "", "TranscriptListEmpty"
        fetched = first.fetch()
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else []
        segments = transcript_to_segments(raw)
        lang = getattr(first, "language_code", "") or getattr(fetched, "language_code", "") or ""
        is_gen = getattr(first, "is_generated", None)
        payload = build_transcript_payload(video_id, segments, lang, is_gen, source="yta_list")
        return "ok", payload, lang, ("" if is_gen is None else str(bool(is_gen))), ""
    except Exception as e:
        msg = f"{e.__class__.__name__}: {str(e)}"
        return classify_transcript_error(msg), None, "", "", msg

def parse_vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    def ts_to_seconds(ts: str) -> float:
        ts = ts.strip()
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, rest = parts
        else:
            h = "0"
            m, rest = parts
        s, ms = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    segments: List[Dict[str, Any]] = []
    lines = [ln.rstrip("\n") for ln in vtt_text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            try:
                start_s, end_s = [x.strip() for x in line.split("-->")[:2]]
                start = ts_to_seconds(start_s)
                end = ts_to_seconds(end_s.split(" ")[0])
                duration = max(0.0, end - start)
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() != "":
                    txt = re.sub(r"<[^>]+>", "", lines[i].strip())
                    text_lines.append(txt)
                    i += 1
                text = " ".join(text_lines).strip()
                if text:
                    segments.append({"text": text, "start": start, "duration": duration})
            except Exception:
                pass
        i += 1
    return segments

def ydl_vtt_fallback(video_url: str, tmp_dir: Path) -> Tuple[str, Optional[Dict[str, Any]], str, str, str]:
    """
    Use yt-dlp to write subtitles (manual + auto) as VTT.
    """
    ensure_dir(tmp_dir)
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
            "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            vid = info.get("id") or ""

        vtts = sorted(tmp_dir.glob("*.vtt"))
        if not vtts:
            return "missing", None, "", "", "yt-dlp: no vtt written"

        # Prefer non-auto if possible
        chosen = None
        for p in vtts:
            if "auto" not in p.name.lower():
                chosen = p
                break
        if chosen is None:
            chosen = vtts[0]

        vtt_text = chosen.read_text(encoding="utf-8", errors="ignore")
        segments = parse_vtt_to_segments(vtt_text)
        if not segments:
            return "missing", None, "", "", "yt-dlp: vtt parsed but empty"

        # infer language code from filename suffix if possible
        lang = ""
        m = re.search(r"\.([a-zA-Z-]{2,10})\.vtt$", chosen.name)
        if m:
            lang = m.group(1)

        payload = build_transcript_payload(vid, segments, lang, None, source="ydl_vtt")
        return "ok", payload, lang, "", ""
    except Exception as e:
        msg = f"{e.__class__.__name__}: {str(e)}"
        return classify_transcript_error(msg), None, "", "", msg

def fetch_transcript_any(video_id: str, video_url: str, tmp_dir: Path) -> Tuple[str, Optional[Dict[str, Any]], str, str, str, str]:
    """
    Try:
      1) youtube-transcript-api fetch() original/default
      2) youtube-transcript-api list() first transcript
      3) yt-dlp VTT fallback
    """
    st1, p1, lang1, isgen1, err1 = yta_fetch_original(video_id)
    if st1 == "ok" and p1 and transcript_to_segments(p1):
        return "ok", p1, lang1, isgen1, "yta_fetch", ""

    st2, p2, lang2, isgen2, err2 = yta_list_first(video_id)
    if st2 == "ok" and p2 and transcript_to_segments(p2):
        return "ok", p2, lang2, isgen2, "yta_list", ""

    st3, p3, lang3, isgen3, err3 = ydl_vtt_fallback(video_url, tmp_dir)
    if st3 == "ok" and p3 and transcript_to_segments(p3):
        return "ok", p3, lang3, isgen3, "ydl_vtt", ""

    # choose best status: blocked > missing > error
    statuses = [(st1, err1), (st2, err2), (st3, err3)]
    if any(s == "blocked" for s, _ in statuses):
        return "blocked", None, "", "", "", " | ".join([e for _, e in statuses if e])
    if any(s == "missing" for s, _ in statuses):
        return "missing", None, "", "", "", " | ".join([e for _, e in statuses if e])
    return "error", None, "", "", "", " | ".join([e for _, e in statuses if e])


# -----------------------------
# Rendering
# -----------------------------

def render_markdown(
    meta: Dict[str, Any],
    transcript_obj: Optional[Dict[str, Any]],
    transcript_status: str,
    transcript_error: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)

    vid = meta.get("id", "")
    url = meta.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
    title = meta.get("title") or ""
    channel = meta.get("channel") or ""
    channel_id = meta.get("channel_id") or ""
    uploader_id = meta.get("uploader_id") or ""
    upload_iso = upload_date_to_iso(meta.get("upload_date")) or ""
    duration = meta.get("duration")
    view_count = meta.get("view_count")
    like_count = meta.get("like_count")
    comment_count = meta.get("comment_count")
    availability = meta.get("availability") or ""

    lines: List[str] = []
    lines.append("---")
    lines.append("platform: youtube")
    lines.append(f"channel: {channel}")
    lines.append(f"channel_id: {channel_id}")
    lines.append(f"uploader_id: {uploader_id}")
    lines.append(f"video_id: {vid}")
    lines.append(f"video_url: {url}")
    lines.append(f"video_title: {title}")
    lines.append(f"upload_date: {upload_iso}")
    lines.append(f"duration_sec: {duration if duration is not None else ''}")
    lines.append(f"view_count: {view_count if view_count is not None else ''}")
    lines.append(f"like_count: {like_count if like_count is not None else ''}")
    lines.append(f"comment_count: {comment_count if comment_count is not None else ''}")
    lines.append(f"availability: {availability}")
    lines.append(f"extracted_at: {now_utc_iso()}")
    lines.append(f"transcript_status: {transcript_status}")
    lines.append(f"transcript_error: {transcript_error}")
    if isinstance(transcript_obj, dict):
        lines.append(f"transcript_language_code: {transcript_obj.get('language_code','')}")
        lines.append(f"transcript_is_generated: {transcript_obj.get('is_generated','')}")
        lines.append(f"transcript_source: {transcript_obj.get('source','')}")
    lines.append("---")
    lines.append("")

    desc = meta.get("description") or ""
    if desc.strip():
        lines.append("## Description")
        lines.append(desc.strip())
        lines.append("")

    lines.append("## Transcript")
    segs = transcript_to_segments(transcript_obj)
    if segs:
        for seg in segs:
            start = float(seg.get("start", 0.0) or 0.0)
            m = int(start // 60)
            s = int(start % 60)
            tc = f"{m:02d}:{s:02d}"
            text = (seg.get("text") or "").replace("\n", " ").strip()
            if text:
                lines.append(f"- [{tc}] {text}")
    else:
        lines.append("_No transcript available._")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Per-video processing
# -----------------------------

def process_video(
    video_id: str,
    video_url: str,
    out_dir: Path,
    force: bool,
    ydl_sleep: float,
) -> IndexRow:
    raw_videos_dir = out_dir / "raw" / "videos"
    transcripts_dir = out_dir / "transcripts"
    rendered_dir = out_dir / "rendered"
    tmp_vtt_dir = out_dir / "tmp_vtt" / video_id

    ensure_dir(raw_videos_dir)
    ensure_dir(transcripts_dir)
    ensure_dir(rendered_dir)

    raw_meta_path = raw_videos_dir / f"{video_id}.json"
    transcript_path = transcripts_dir / f"{video_id}.json"

    # (1) Metadata cache
    meta: Optional[Dict[str, Any]] = None
    if raw_meta_path.exists() and not force:
        meta = read_json(raw_meta_path)
    else:
        with ydl_video() as ydl:
            meta = ydl.extract_info(video_url, download=False)
        if meta:
            write_json(raw_meta_path, meta)

    if not meta:
        return IndexRow(
            video_id=video_id,
            video_url=video_url,
            extracted_at=now_utc_iso(),
            availability="unavailable",
            transcript_status="skipped",
            transcript_error="metadata_unavailable",
            raw_video_meta_path=str(raw_meta_path),
        )

    if ydl_sleep > 0:
        time.sleep(ydl_sleep)

    # (2) Transcript cache
    transcript_obj: Optional[Dict[str, Any]] = None
    transcript_status = ""
    transcript_lang = ""
    transcript_is_generated = ""
    transcript_source = ""
    transcript_error = ""

    if transcript_path.exists() and not force:
        transcript_obj = read_json(transcript_path)
        if isinstance(transcript_obj, dict):
            transcript_status = transcript_obj.get("status", "ok")
            transcript_lang = transcript_obj.get("language_code", "") or ""
            ig = transcript_obj.get("is_generated", None)
            transcript_is_generated = "" if ig is None else str(bool(ig))
            transcript_source = transcript_obj.get("source", "") or ""
            transcript_error = transcript_obj.get("error", "") or ""
        else:
            transcript_status = "ok"
    else:
        st, payload, lang, isgen, source, err = fetch_transcript_any(
            video_id=video_id,
            video_url=meta.get("webpage_url") or video_url,
            tmp_dir=tmp_vtt_dir,
        )
        transcript_status = st
        transcript_lang = lang
        transcript_is_generated = isgen
        transcript_source = source
        transcript_error = err

        if st == "ok" and payload:
            write_json(transcript_path, payload)
            transcript_obj = payload
        else:
            # still write a diagnostic transcript file
            diag = {
                "video_id": video_id,
                "status": transcript_status,
                "source": transcript_source,
                "error": transcript_error,
                "fetched_at": now_utc_iso(),
                "segments": [],
            }
            write_json(transcript_path, diag)
            transcript_obj = diag

    # (3) Render (IMPORTANT: rendering reads normalized segments!)
    channel_slug = safe_filename(meta.get("channel") or meta.get("uploader") or "channel")
    upload_iso = upload_date_to_iso(meta.get("upload_date")) or "unknown-date"
    rendered_path = rendered_dir / f"{upload_iso}_{channel_slug}_{video_id}.md"

    if force or not rendered_path.exists():
        render_markdown(meta, transcript_obj, transcript_status, transcript_error, rendered_path)

    return IndexRow(
        video_id=video_id,
        video_url=meta.get("webpage_url") or video_url,
        title=meta.get("title") or "",
        channel=meta.get("channel") or "",
        channel_id=meta.get("channel_id") or "",
        uploader_id=meta.get("uploader_id") or "",
        upload_date=upload_date_to_iso(meta.get("upload_date")) or "",
        duration_sec=str(meta.get("duration") or ""),
        view_count=str(meta.get("view_count") or ""),
        like_count=str(meta.get("like_count") or ""),
        comment_count=str(meta.get("comment_count") or ""),
        availability=str(meta.get("availability") or ""),
        extracted_at=now_utc_iso(),

        transcript_status=transcript_status,
        transcript_language_code=transcript_lang,
        transcript_is_generated=transcript_is_generated,
        transcript_source=transcript_source,
        transcript_error=transcript_error,

        rendered_path=str(rendered_path),
        transcript_path=str(transcript_path),
        raw_video_meta_path=str(raw_meta_path),
    )


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", help="Channel URL or @handle (e.g. @ecbeuro)")
    ap.add_argument("-o", "--out", required=True, help="Output directory")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (applied after enrichment)")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD (applied after enrichment)")
    ap.add_argument("--limit", type=int, default=0, help="Process only last N inventory entries (0 = all)")
    ap.add_argument("--workers", type=int, default=6, help="Parallel workers")
    ap.add_argument("--force", action="store_true", help="Overwrite cached outputs")
    ap.add_argument("--ydl-sleep", type=float, default=0.0, help="Sleep seconds after per-video yt-dlp call")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    channel_url = normalize_channel_input(args.channel)
    since = parse_date(args.since)
    until = parse_date(args.until)

    inv = extract_inventory(channel_url, out_dir, force=args.force)
    pairs = inventory_to_video_pairs(inv)
    if not pairs:
        print("No videos found in inventory.", file=sys.stderr)
        return 2

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    write_json(out_dir / "run_manifest.json", {
        "run_at": now_utc_iso(),
        "channel_input": args.channel,
        "channel_url": channel_url,
        "since": args.since or "",
        "until": args.until or "",
        "limit": args.limit,
        "workers": args.workers,
        "ydl_sleep": args.ydl_sleep,
        "force": bool(args.force),
        "inventory_selected": len(pairs),
    })

    # Process parallel
    index_rows: List[IndexRow] = []
    workers = max(1, int(args.workers))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(process_video, vid, url, out_dir, args.force, float(args.ydl_sleep))
            for (vid, url) in pairs
        ]
        for fut in cf.as_completed(futures):
            index_rows.append(fut.result())

    # Apply window filter AFTER enrichment
    filtered = [r for r in index_rows if within_window(r.upload_date or None, since, until)]
    filtered.sort(key=lambda r: (r.upload_date or "0000-00-00", r.video_id), reverse=True)

    # Write index.csv
    index_path = out_dir / "index.csv"
    rows = filtered if filtered else index_rows
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return 3

    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Summary
    write_json(out_dir / "index_summary.json", {
        "written_at": now_utc_iso(),
        "selected_from_inventory": len(pairs),
        "kept_after_date_filter": len(filtered),
        "index_csv": str(index_path),
        "rendered_dir": str(out_dir / "rendered"),
        "transcripts_dir": str(out_dir / "transcripts"),
    })

    print("Done.")
    print(f"- Index: {index_path}")
    print(f"- Rendered: {out_dir / 'rendered'}")
    print(f"- Transcripts: {out_dir / 'transcripts'}")
    print(f"- Kept after date filter: {len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
