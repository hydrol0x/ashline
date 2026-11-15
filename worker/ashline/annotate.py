"""Drawing on the family's photos (build spec 6.6).

Only ever draws shapes and numbers on a copy of the original photo. Nothing is
regenerated and no generative model touches the pixels. The legend goes in the
chat message, not on the image.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

# Colour by status: new to the list, maybe a duplicate, already on the list.
STATUS_COLORS = {"new": (214, 106, 58), "maybe": (222, 164, 44), "listed": (62, 140, 110)}


def open_photo(path: str | Path) -> Image.Image:
    """The photo upright (EXIF orientation applied), RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def crop_item(photo_path: str | Path, bbox: list | None, pad: float = 0.12, max_side: int = 720) -> Image.Image:
    """The part of the photo showing one item, with a little context around it."""
    image = open_photo(photo_path)
    if bbox:
        x0, y0, x1, y1 = bbox
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(image.width, int(x1 + px)), min(image.height, int(y1 + py)))
        if box[2] - box[0] > 4 and box[3] - box[1] > 4:
            image = image.crop(box)
    image.thumbnail((max_side, max_side))
    return image


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:                      # very old Pillow
        return ImageFont.load_default()


def annotate(photo_path: str | Path, boxes: list[dict], out_path: str | Path | None = None,
             max_side: int = 1600) -> Image.Image:
    """The original photo with a numbered rounded box per item.

    `boxes`: [{"n": 1, "bbox": [x0, y0, x1, y1], "status": "new" | "maybe" | "listed"}].
    """
    image = open_photo(photo_path)
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(image)
    line = max(3, round(max(image.size) / 320))
    radius = max(14, round(max(image.size) / 60))
    font = _font(round(radius * 1.25))
    for box in boxes:
        if not box.get("bbox"):
            continue
        x0, y0, x1, y1 = (v * scale for v in box["bbox"])
        color = STATUS_COLORS.get(box.get("status") or "new", STATUS_COLORS["new"])
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius // 2, outline=color, width=line)
        cx = min(max(x0 + radius, radius), image.width - radius)
        cy = min(max(y0 + radius, radius), image.height - radius)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color, outline=(255, 255, 255), width=2)
        draw.text((cx, cy), str(box["n"]), fill=(255, 255, 255), font=font, anchor="mm")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, quality=88)
    return image


def legend(entries: list[dict]) -> str:
    """One numbered line per item for the chat message:
    1. Sectional sofa · $900–$2.1k
    2. Floor lamp · already listed"""
    return "\n".join(f"{e['n']}. {e['label'][:1].upper()}{e['label'][1:]}" for e in entries)


def side_by_side(photo_a: str | Path, bbox_a: list | None, photo_b: str | Path, bbox_b: list | None,
                 out_path: str | Path, height: int = 600) -> Path:
    """Two item crops next to each other, numbered 1 and 2: 'Same couch?'"""
    tiles = []
    for n, (path, bbox) in enumerate(((photo_a, bbox_a), (photo_b, bbox_b)), 1):
        tile = crop_item(path, bbox, pad=0.25, max_side=1200)
        tile = tile.resize((max(1, round(tile.width * height / tile.height)), height))
        draw = ImageDraw.Draw(tile)
        r = height // 14
        draw.ellipse([12, 12, 12 + 2 * r, 12 + 2 * r], fill=STATUS_COLORS["maybe"], outline=(255, 255, 255), width=3)
        draw.text((12 + r, 12 + r), str(n), fill=(255, 255, 255), font=_font(round(r * 1.3)), anchor="mm")
        tiles.append(tile)
    gap = 16
    sheet = Image.new("RGB", (tiles[0].width + tiles[1].width + gap, height), (250, 247, 242))
    sheet.paste(tiles[0], (0, 0))
    sheet.paste(tiles[1], (tiles[0].width + gap, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)
    return out_path
