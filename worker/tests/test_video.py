"""A video becomes a few still photos that go through the photo pipeline. Needs ffmpeg for most tests."""
import io
import random
import subprocess

import pytest
from PIL import Image, ImageDraw

from ashline import video
from ashline.chat import Engine, Inbound
from ashline.pricing import DefaultPricer
from ashline.store import Store
from test_chat import MOM, T0, Family, texts

needs_ffmpeg = pytest.mark.skipif(not video.have_ffmpeg(), reason="ffmpeg is not installed")


def scene(seed: int, blank: bool = False) -> Image.Image:
    """A busy picture that is different for each seed; or a plain wall."""
    image = Image.new("RGB", (320, 240), (236, 236, 232))
    if blank:
        return image
    draw, rng = ImageDraw.Draw(image), random.Random(seed)
    for _ in range(40):
        x, y = rng.randrange(300), rng.randrange(220)
        draw.rectangle([x, y, x + rng.randrange(10, 90), y + rng.randrange(10, 70)],
                       fill=tuple(rng.randrange(256) for _ in range(3)))
    return image


def make_clip(folder, scenes, seconds_each=1.0, name="walk.mp4"):
    """One scene after another, each held for a moment."""
    for n, picture in enumerate(scenes):
        picture.save(folder / f"s-{n:03d}.png")
    target = folder / name
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(1 / seconds_each), "-i",
                           str(folder / "s-%03d.png"), "-r", "10", "-pix_fmt", "yuv420p", str(target)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return target


def test_photos_are_not_mistaken_for_video():
    jpeg = io.BytesIO()
    Image.new("RGB", (8, 8)).save(jpeg, format="JPEG")
    assert not video.is_video("IMG_1.jpg", jpeg.getvalue())
    assert not video.is_video("IMG_2.heic", b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00")       # same container as mp4
    assert not video.is_video("IMG_3.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert video.is_video("clip.mp4", b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")
    assert video.is_video("clip.mov", b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00")
    assert video.is_video("clip.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 12)
    assert video.is_video("clip.mov", b"")                                                     # by name alone


@needs_ffmpeg
def test_one_still_per_view_in_order_without_blank_walls(tmp_path):
    clip = make_clip(tmp_path, [scene(1), scene(2), scene(0, blank=True), scene(3)], seconds_each=2)
    frames = video.extract_frames(clip, tmp_path / "frames")
    assert len(frames) == 3                                       # the still camera gives one frame, the wall none
    assert [f["at"] for f in frames] == sorted(f["at"] for f in frames)
    assert frames[0]["at"] < 2 <= frames[1]["at"] < 4 and frames[2]["at"] >= 6
    assert all(Image.open(f["path"]).size == (320, 240) for f in frames)


@needs_ffmpeg
def test_frame_limit(tmp_path):
    clip = make_clip(tmp_path, [scene(n) for n in range(10)], seconds_each=1)
    assert len(video.extract_frames(clip, tmp_path / "frames", max_frames=4)) == 4


@needs_ffmpeg
def test_unreadable_video_is_reported(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"not a video" * 20)
    with pytest.raises(video.VideoError):
        video.extract_frames(broken, tmp_path / "frames")


class SceneDetector:
    """Every still shows a sofa; the second scene also shows a lamp."""

    def __init__(self):
        self.calls = 0

    def detect(self, path, room_hint=None, rooms=None):
        self.calls += 1
        sofa = {"label": "blue velvet sofa", "category": "furniture", "brand_guess": None, "confidence": 0.8,
                "bbox": [20, 100, 200, 220], "room": "Living room"}
        lamp = {"label": "brass floor lamp", "category": "lighting", "brand_guess": None, "confidence": 0.8,
                "bbox": [220, 40, 260, 220], "room": "Living room"}
        return [sofa, lamp] if self.calls == 2 else [sofa]


@needs_ffmpeg
def test_a_video_in_the_chat_goes_through_the_photo_pipeline(tmp_path):
    f = Family().setup()
    f.engine.detector = detector = SceneDetector()
    clip = make_clip(tmp_path, [scene(1), scene(2), scene(3)], seconds_each=2)
    out = f.say(MOM, photos=[("walk.mp4", clip.read_bytes())])
    assert texts(out) == ["Got the video, looking through it."]
    photos = f.store.get_photos(f.case_id, source="photo")
    assert len(photos) == 3 and len({p["batch_id"] for p in photos}) == 1 and photos[0]["batch_id"].startswith("clip-")
    digest = f.wait()
    assert detector.calls == 3
    items = f.store.get_items(f.case_id)
    lines = [i for i in items if i["cluster_id"] in (None, i["item_id"])]
    assert sorted(i["label"] for i in lines) == ["blue velvet sofa", "brass floor lamp"]
    assert "sofa" in " ".join(texts(digest)).lower()              # one sofa seen three times, not three sofas


@needs_ffmpeg
def test_a_broken_video_gets_a_plain_reply(tmp_path):
    f = Family().setup()
    out = f.say(MOM, photos=[("broken.mp4", b"\x00\x00\x00\x18ftypisom" + b"junk" * 50)])
    assert texts(out) == ["I couldn't open the video. Could you send it again, or send a few photos instead?"]


def test_without_ffmpeg_the_reply_says_so(monkeypatch):
    monkeypatch.setattr(video, "have_ffmpeg", lambda: False)
    f = Family().setup()
    out = f.say(MOM, photos=[("walk.mp4", b"\x00\x00\x00\x18ftypisom" + b"junk" * 50)])
    assert len(out) == 1 and "ffmpeg" in out[0].text
