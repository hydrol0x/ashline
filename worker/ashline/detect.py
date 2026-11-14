"""Item detection (build spec 6.3) for one photo.

Step 1 and the fallback of step 2 in one call: a vision model lists the
contents items it sees and gives each a box on a 0-1000 grid. Any
OpenAI-compatible chat endpoint works (OpenRouter, a Bedrock gateway, ...).
A SAM 3 box refiner can replace `boxes` later without changing the output.

Scan faces and family photos go through the same function; a scan face only
adds a room hint, because the scan already knows which room it was taken in.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

from . import config

CATEGORIES = ["furniture", "electronics", "appliances", "kitchen", "decor", "lighting", "bedding",
              "clothing", "tools", "sports", "toys", "outdoor", "office", "other"]

PROMPT = """You are building a home contents inventory for an insurance claim after a total loss.
List the personal property visible in this photo{room}.

Include: furniture, electronics, freestanding appliances, rugs, lamps, decor, bedding, kitchenware,
toys, tools, sports gear, outdoor furniture and equipment.
Skip: anything built into the house (cabinets, counters, sinks, range hoods, built-in ovens,
doors, windows, ceiling lights, vents, flooring), people, pets, and things too small or blurry to name.
Skip anything seen only through a doorway in another room.

Return ONLY a JSON array, between 0 and 30 objects, each:
{{"label": "<specific noun phrase, e.g. blue velvet three-seat sofa>",
  "category": "<one of: {categories}>",
  "brand_guess": "<brand if a logo or distinctive design shows, else null>",
  "confidence": <0.0-1.0>,
  "box": [x0, y0, x1, y1]}}
`box` is the item's bounding box on a 0-1000 grid, origin top-left, x to the right, y down.
Give each physical object its own entry (four bar stools = four entries)."""

MIN_BOX_FRACTION = 0.0015       # drop boxes under 0.15% of the image
MERGE_IOU = 0.7


class DetectionError(RuntimeError):
    pass


def iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (label or "").lower()).strip()


def parse_items(text: str, width: int, height: int) -> list[dict]:
    """Model reply -> clean item dicts with pixel boxes. Tolerates code fences and stray text."""
    match = re.search(r"\[.*\]", text or "", re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    items = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict) or not norm_label(entry.get("label", "")):
            continue
        box = entry.get("box") or entry.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            x0, y0, x1, y1 = (min(1000.0, max(0.0, float(v))) for v in box)
        except (TypeError, ValueError):
            continue
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        if (x1 - x0) * (y1 - y0) / 1_000_000 < MIN_BOX_FRACTION:
            continue
        try:
            confidence = min(1.0, max(0.0, float(entry.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        category = str(entry.get("category") or "other").lower()
        brand = entry.get("brand_guess")
        items.append({
            "label": str(entry["label"]).strip(),
            "category": category if category in CATEGORIES else "other",
            "brand_guess": brand if isinstance(brand, str) and brand.lower() not in ("", "null", "none", "unknown") else None,
            "confidence": round(confidence, 2),
            "bbox": [round(x0 * width / 1000), round(y0 * height / 1000),
                     round(x1 * width / 1000), round(y1 * height / 1000)],
        })
    # Same label, heavily overlapping boxes: keep the more confident one.
    items.sort(key=lambda i: -i["confidence"])
    kept: list[dict] = []
    for item in items:
        if any(norm_label(item["label"]) == norm_label(k["label"]) and iou(item["bbox"], k["bbox"]) > MERGE_IOU
               for k in kept):
            continue
        kept.append(item)
    return kept


class VLMDetector:
    """Calls a vision model through an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, max_side: int = 1568):
        self.base_url = (base_url or config.env("ASHLINE_VLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key = api_key or config.env("ASHLINE_VLM_API_KEY", required=True)
        self.model = model or config.env("ASHLINE_VLM_MODEL", "anthropic/claude-sonnet-5.5")
        self.max_side = max_side

    def detect(self, path: str | Path, room_hint: str | None = None) -> list[dict]:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        small = image.copy()
        small.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        room = f" (taken in: {room_hint})" if room_hint else ""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180,
            json={
                "model": self.model,
                "max_tokens": 4000,
                "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(room=room, categories=", ".join(CATEGORIES))},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
            },
        )
        if not resp.ok:
            raise DetectionError(f"Vision model call failed ({resp.status_code}): {resp.text[:300]}")
        try:
            text = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise DetectionError(f"Unexpected vision model reply: {resp.text[:300]}") from exc
        if isinstance(text, list):          # some gateways return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return parse_items(text, width, height)


# --------------------------------------------------------------- local model
# Florence-2 runs on a laptop CPU (about 15 s per photo, no API key). It gives a
# descriptive phrase and a box per region but no confidence, and it also
# describes fixtures and whole scenes, which are filtered out here.
FIXTURE_HEADS = {
    "cabinet", "cabinets", "cabinetry", "cupboard", "countertop", "countertops", "counter", "sink",
    "window", "windows", "door", "doors", "doorway", "drawer", "drawers", "faucet", "tap", "vanity",
    "unit", "island", "wall", "walls", "floor", "flooring", "ceiling", "house", "home", "building",
    "room", "kitchen", "kitchenette", "bathroom", "bedroom", "hallway", "closet", "light", "lights",
    "lighting", "fireplace", "stairs", "staircase", "railing", "toilet", "bathtub", "tub", "shower",
    "vent", "fan", "blinds", "blind", "shutters", "backsplash", "tile", "tiles", "hood", "outlet",
    "switch", "handle", "handles", "knob", "person", "man", "woman", "child", "dog", "cat", "pantry",
    "interior", "area", "space", "corner", "view", "balcony", "patio", "porch", "garage", "fence",
    "roof", "tree", "plant life", "sky", "grass", "driveway", "frame", "molding", "baseboard",
    "faucets", "faucette", "dispenser", "blades", "netting", "sill", "rod", "car", "wheel",
    # people and what a framed photo shows are not items
    "complex", "lot", "street", "road", "sidewalk", "yard",        # seen through a window
    "face", "baby", "girl", "boy", "father", "mother", "daughter", "son", "people", "family", "hand",
}
CATEGORY_WORDS = [
    ("electronics", ["tv", "television", "monitor", "computer", "laptop", "speaker", "soundbar", "console",
                     "printer", "camera", "phone", "tablet", "router", "stereo", "projector"]),
    ("appliances", ["refrigerator", "fridge", "washer", "dryer", "microwave", "oven", "stove", "freezer",
                    "vacuum", "heater", "air conditioner", "dishwasher", "purifier", "humidifier"]),
    ("kitchen", ["kettle", "toaster", "blender", "mixer", "coffee", "pot", "pan", "dish", "plate", "bowl",
                 "cup", "mug", "glass", "knife", "cutting board", "air fryer", "bottle", "jar"]),
    ("bedding", ["mattress", "pillow", "blanket", "duvet", "comforter", "bedding", "quilt"]),
    ("lighting", ["lamp", "lantern", "chandelier"]),
    ("toys", ["toy", "stuffed", "doll", "playpen", "stroller", "car seat", "high chair", "crib", "bassinet"]),
    ("sports", ["bike", "bicycle", "treadmill", "weights", "dumbbell", "golf", "surfboard", "skateboard", "yoga"]),
    ("tools", ["tool", "drill", "ladder", "mower", "saw", "workbench", "toolbox"]),
    ("clothing", ["clothes", "clothing", "jacket", "coat", "shoes", "dress", "hanger", "handbag", "suitcase"]),
    ("outdoor", ["grill", "barbecue", "patio", "umbrella", "hose", "planter", "trampoline", "hot tub"]),
    ("office", ["desk", "office chair", "filing", "bookcase", "bookshelf", "whiteboard"]),
    ("decor", ["rug", "carpet", "curtain", "curtains", "drapes", "mirror", "painting", "picture", "art",
               "vase", "clock", "plant", "wall decor", "sculpture", "candle", "basket"]),
    ("furniture", ["sofa", "couch", "loveseat", "recliner", "chair", "stool", "table", "bed", "dresser",
                   "nightstand", "bench", "ottoman", "shelf", "shelves", "sideboard", "credenza", "armoire",
                   "wardrobe", "futon", "headboard", "stand", "cart", "sectional", "chest"]),
]
MAX_SCENE_FRACTION = 0.55       # a box this large is a caption of the scene, not an item


def head_phrase(label: str) -> str:
    """'white cabinet with pendant light' -> 'white cabinet': the thing itself."""
    return re.split(r"\s+(?:with|on|in|of|next to|near|under|over|against|at|holding)\s+",
                    norm_label(label), maxsplit=1)[0]


def is_fixture(label: str) -> bool:
    head = head_phrase(label)
    words = head.split()
    if not words:
        return True
    return words[-1] in FIXTURE_HEADS or " ".join(words[-2:]) in FIXTURE_HEADS


def drop_fixtures(items: list[dict]) -> list[dict]:
    """Remove built-in fixtures, people and scenery from any detector's output."""
    return [item for item in items if not is_fixture(item["label"])]


def guess_category(label: str) -> str:
    text = f" {head_phrase(label)} "
    for category, words in CATEGORY_WORDS:
        if any(f" {w} " in text or f" {w}s " in text for w in words):
            return category
    return "other"


def clean_regions(labels: list[str], boxes: list[list[float]], width: int, height: int) -> list[dict]:
    """Florence-2 regions -> item dicts: drop fixtures, scene captions, slivers and repeats."""
    items = []
    for label, box in zip(labels, boxes):
        x0, y0, x1, y1 = (float(v) for v in box)
        fraction = (x1 - x0) * (y1 - y0) / (width * height)
        if fraction < MIN_BOX_FRACTION or fraction > MAX_SCENE_FRACTION or is_fixture(label):
            continue
        items.append({
            "label": label.strip(),
            "category": guess_category(label),
            "brand_guess": None,
            "confidence": 0.5,                 # the model gives no score
            "bbox": [round(x0), round(y0), round(x1), round(y1)],
        })
    # Overlapping boxes of the same thing: keep the longer, more specific phrase.
    items.sort(key=lambda i: -len(i["label"]))
    kept: list[dict] = []
    for item in items:
        if any(iou(item["bbox"], k["bbox"]) > MERGE_IOU for k in kept):
            continue
        kept.append(item)
    kept.sort(key=lambda i: (i["bbox"][0], i["bbox"][1]))
    return kept


class FlorenceDetector:
    """Local detector. Needs `pip install torch transformers` (the `local-vision` extra)."""

    TASK = "<DENSE_REGION_CAPTION>"

    def __init__(self, model: str | None = None):
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise DetectionError(
                "The local detector needs torch and transformers: pip install -e '.[local-vision]'") from exc
        name = model or config.env("ASHLINE_FLORENCE_MODEL", "florence-community/Florence-2-large")
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Florence2ForConditionalGeneration.from_pretrained(name).eval()

    def detect(self, path: str | Path, room_hint: str | None = None) -> list[dict]:
        image = Image.open(path).convert("RGB")
        inputs = self.processor(text=self.TASK, images=image, return_tensors="pt")
        with self._torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        out = self.processor.post_process_generation(text, task=self.TASK, image_size=image.size)[self.TASK]
        return clean_regions(out.get("labels") or [], out.get("bboxes") or [], *image.size)


def get_detector(kind: str | None = None):
    """`ASHLINE_DETECTOR` = vlm | florence. Default: vlm when a key is set, else the local model."""
    kind = (kind or config.env("ASHLINE_DETECTOR") or
            ("vlm" if config.env("ASHLINE_VLM_API_KEY") else "florence")).lower()
    if kind == "vlm":
        return VLMDetector()
    if kind == "florence":
        return FlorenceDetector()
    raise DetectionError(f"Unknown ASHLINE_DETECTOR {kind!r}: use vlm or florence.")
