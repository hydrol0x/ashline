"""Video input: a walk-through clip becomes a handful of still photos.

Claude reads images, not video, so a clip is cut into frames with ffmpeg and the best few are
kept: sharp ones, spread over the clip, each different enough from the last one kept. The
frames then go through the same pipeline as photos (detection, merging, pricing), and the
merge step removes the repeats that overlapping frames produce.

Needs the `ffmpeg` program on the machine. Without it a video is skipped with a clear note.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv", ".3gp")
SAMPLE_FPS = 2                  # candidate frames per second of video
MAX_FRAMES = 14                 # stills kept from a clip of up to a minute; longer clips get as many per minute
FFMPEG_SECONDS = 180            # a video that takes longer than this to read is given up on
MAX_FRAMES_LONG = 40            # and never more than this, however long the walk-through
MIN_DIFFERENCE = 9.0            # mean grey-level change needed to count as a new view (0-255)
EMPTY_FRACTION = 0.35           # a frame with under this share of the clip's typical detail shows nothing


class VideoError(RuntimeError):
    pass


# Other things that live in the mp4 container: HEIC/AVIF photos, and audio (a voice memo).
IMAGE_BRANDS = (b"heic", b"heix", b"hevc", b"mif1", b"msf1", b"avif", b"avis")
NOT_VIDEO_BRANDS = IMAGE_BRANDS + (b"M4A ", b"M4B ", b"M4P ")


def is_video(name: str, data: bytes) -> bool:
    """By its first bytes, falling back to the file name. A HEIC photo, a WebP image or a voice memo is not a video."""
    head = data[:16]
    if head[4:8] == b"ftyp":
        return head[8:12] not in NOT_VIDEO_BRANDS
    if head[:4] == b"RIFF":
        return head[8:12] == b"AVI "
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    if head.startswith((b"\xff\xd8", b"\x89PNG", b"GIF8")):
        return False
    return name.lower().endswith(VIDEO_SUFFIXES)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _sharpness(image: Image.Image) -> float:
    """Higher = crisper. Motion blur and focus hunting score low."""
    edges = image.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size                                    # the filter leaves the outer pixels untouched: cut them off
    return ImageStat.Stat(edges.crop((1, 1, w - 1, h - 1))).var[0]


def _difference(a: Image.Image, b: Image.Image) -> float:
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


def extract_frames(video: str | Path, out_dir: str | Path, max_frames: int = MAX_FRAMES,
                   sample_fps: float = SAMPLE_FPS) -> list[dict]:
    """Stills from a video, in order: [{path, at}] where `at` is seconds into the clip.

    The clip is split into equal spans; the sharpest candidate of each span is kept, unless it
    shows nearly the same view as the previous kept frame (the camera stood still).
    """
    if not have_ffmpeg():
        raise VideoError("Reading video needs the ffmpeg program, which is not installed on this machine.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # -noautorotate is NOT used: ffmpeg applies the phone's rotation so frames come out upright.
        try:
            done = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={sample_fps}", "-q:v", "2",
                 str(Path(tmp) / "c-%05d.jpg")], capture_output=True, text=True, timeout=FFMPEG_SECONDS)
        except subprocess.TimeoutExpired:
            raise VideoError("The video took too long to read. A shorter clip, or a few photos, will work.") from None
        candidates = sorted(Path(tmp).glob("c-*.jpg"))
        if done.returncode != 0 or not candidates:
            raise VideoError(f"Could not read the video: {(done.stderr or 'no frames found').strip()[:200]}")
        scored = []
        for n, path in enumerate(candidates):
            with Image.open(path) as image:
                small = image.convert("L").resize((96, 96))
            scored.append({"path": path, "at": round(n / sample_fps, 1), "small": small, "sharp": _sharpness(small)})

        # A blank wall or a blown-out frame has almost no edges: leave those out when there is a choice.
        typical = sorted(f["sharp"] for f in scored)[len(scored) // 2]
        detailed = [f for f in scored if f["sharp"] >= EMPTY_FRACTION * typical]
        scored = detailed if len(detailed) >= 3 else scored
        minutes = len(candidates) / sample_fps / 60
        if minutes > 1:                                     # a long walk-through covers more of the home
            max_frames = min(max(MAX_FRAMES_LONG, max_frames), round(max_frames * minutes))
        spans = min(max_frames, len(scored))
        kept: list[dict] = []
        for k in range(spans):
            span = scored[k * len(scored) // spans:(k + 1) * len(scored) // spans]
            if not span:
                continue
            # The sharpest frame of the span that shows something new; none if the camera stood still.
            fresh = [f for f in span if not kept or _difference(kept[-1]["small"], f["small"]) >= MIN_DIFFERENCE]
            if fresh:
                kept.append(max(fresh, key=lambda f: f["sharp"]))
        frames = []
        for n, frame in enumerate(kept, 1):
            target = out_dir / f"frame-{n:02d}.jpg"
            shutil.copyfile(frame["path"], target)
            frames.append({"path": str(target), "at": frame["at"]})
    return frames


def frames_as_attachments(name: str, data: bytes, max_frames: int = MAX_FRAMES) -> list[tuple[str, bytes]]:
    """A received video as (filename, jpeg bytes) pairs, ready for the photo pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / ("clip" + (Path(name).suffix or ".mp4"))
        source.write_bytes(data)
        frames = extract_frames(source, Path(tmp) / "frames", max_frames)
        stem = Path(name).stem or "video"
        return [(f"{stem}-{n:02d}.jpg", Path(f["path"]).read_bytes()) for n, f in enumerate(frames, 1)]
