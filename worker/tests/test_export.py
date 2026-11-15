"""The insurer PDF has to survive whatever real families send."""
from PIL import Image

from ashline import config
from ashline.export import export_pdf
from ashline.store import Store


def test_pdf_survives_odd_names_and_photos():
    store = Store()
    store.set_case("odd", name="Smith & <Jones> home", owner_token="o", member_token="m", address="12 A&B Rd")
    folder = config.case_dir("odd") / "photos"
    folder.mkdir(parents=True, exist_ok=True)
    photos = {
        "png": Image.new("RGBA", (800, 600), (120, 110, 100, 200)),        # screenshot with transparency
        "tall": Image.new("RGB", (600, 2400), (90, 100, 110)),             # taller than a page
        "pal": Image.new("P", (640, 480)),                                  # palette image
    }
    rows = []
    for n, (key, image) in enumerate(photos.items(), 1):
        path = folder / f"{key}.png"
        image.save(path)
        rows.append({"photo_id": key, "path": str(path), "source": "photo", "number": n, "added_by": "Mom & Dad"})
    broken = folder / "broken.jpg"
    broken.write_bytes(b"not really a jpeg")
    rows.append({"photo_id": "broken", "path": str(broken), "source": "photo", "number": 4, "added_by": "Maya"})
    store.add_photos("odd", rows)
    labels = ["pots & pans", "TV <55 inch>", "Ben & Jerry's freezer", "lamp"]
    for key, label in zip(["png", "tall", "pal", "broken"], labels):
        item = store.add_item("odd", {"label": label, "category": "kitchen & dining", "bbox": [10, 10, 300, 300],
                                      "room_label": "Kitchen & pantry", "brand_guess": "Crate & Barrel"},
                              source="photo", photo=store.get_photo("odd", key), added_by="Mom & Dad")
        store.update_item("odd", item, price_low=50, price_mid=80, price_high=120, price_source="claude estimate")
        store.set_item_status("odd", item, "confirmed")
    pdf = export_pdf("odd", store)
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 2000
