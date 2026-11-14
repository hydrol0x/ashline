from ashline.detect import clean_regions, guess_category, is_fixture, iou, parse_items


def test_parse_items_scales_boxes_and_tolerates_fences():
    text = """Here you go:
```json
[{"label": "Blue velvet loveseat", "category": "furniture", "brand_guess": null, "confidence": 0.9, "box": [100, 500, 600, 900]},
 {"label": "blue velvet loveseat", "category": "furniture", "confidence": 0.6, "box": [110, 505, 600, 900]},
 {"label": "degenerate box", "category": "decor", "confidence": 0.9, "box": [0, 0, 5, 5]},
 {"label": "TV remote", "category": "electronics", "confidence": 0.8, "box": [400, 400, 420, 410]},
 {"label": "Soundbar", "category": "gadgets", "brand_guess": "Sonos", "confidence": "high", "box": [900, 100, 700, 200]},
 {"label": "", "box": [0, 0, 500, 500]}, {"label": "no box"}, "junk"]
```"""
    items = parse_items(text, 2048, 1024)
    # The duplicate, the degenerate box and the junk are dropped; a small named item is kept.
    assert [i["label"] for i in items] == ["Blue velvet loveseat", "TV remote", "Soundbar"]
    assert items[0]["bbox"] == [205, 512, 1229, 922]
    assert items[2]["bbox"] == [1434, 102, 1843, 205]          # corners put in order
    assert items[2]["category"] == "other" and items[2]["brand_guess"] == "Sonos" and items[2]["confidence"] == 0.5
    assert parse_items("no json here", 10, 10) == [] and parse_items("[not json]", 10, 10) == []


def test_fixture_filter():
    for label in ["oak kitchen cabinet with pendant light", "countertop", "sink", "bamboo blinds on window",
                  "modern living room with green velvet sofa and kitchenette", "steel dishwasher door",
                  "red and grey building with flat roof", "office complex with red and grey buildings",
                  "boy with a kite in a field", "white sedan car parked in driveway", "pendant lantern light fixture",
                  "stainless steel dishwasher", "woman holding a puppy on a beach", "human face"]:
        assert is_fixture(label), label
    for label in ["black french door refrigerator", "green velvet sofa with brass legs",
                  "toddler car seat", "oak bar stool with leather seat and backrest", "floor lamp",
                  "framed photo of woman holding a puppy on a beach"]:
        assert not is_fixture(label), label


def test_categories():
    assert guess_category("black french door refrigerator") == "appliances"
    assert guess_category("green velvet sofa with brass legs") == "furniture"
    assert guess_category("toddler car seat") == "toys"
    assert guess_category("flat screen tv on wooden stand") == "electronics"
    assert guess_category("mystery object") == "other"


def test_clean_regions_drops_scene_captions_and_overlaps():
    labels = ["modern sofa set", "green velvet sofa", "green velvet sofa with brass legs", "oak kitchen cabinets"]
    boxes = [[0, 0, 2000, 2000], [1200, 1100, 2040, 1660], [1225, 1143, 2044, 1661], [0, 0, 400, 400]]
    items = clean_regions(labels, boxes, 2048, 2048)
    assert [i["label"] for i in items] == ["green velvet sofa with brass legs"]
    assert items[0]["confidence"] == 0.5 and items[0]["category"] == "furniture"
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3
