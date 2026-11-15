from PIL import Image

from ashline.annotate import STATUS_COLORS, annotate, crop_item, legend, side_by_side


def photo(tmp_path, name="p.jpg", size=(800, 600), color=(90, 90, 90)):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path, quality=95)
    return path


def near(pixel, color, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(pixel, color))


def test_annotate_draws_on_a_copy_and_leaves_the_original_alone(tmp_path):
    src = photo(tmp_path)
    before = src.read_bytes()
    out = annotate(src, [{"n": 1, "bbox": [100, 100, 400, 300], "status": "new"},
                         {"n": 2, "bbox": [500, 350, 780, 580], "status": "listed"},
                         {"n": 3, "bbox": None}], tmp_path / "out" / "a.jpg")
    assert src.read_bytes() == before and (tmp_path / "out" / "a.jpg").exists()
    assert out.size == (800, 600)
    assert near(out.getpixel((250, 101)), STATUS_COLORS["new"])            # top edge of box 1
    assert near(out.getpixel((640, 579)), STATUS_COLORS["listed"])         # bottom edge of box 2
    assert near(out.getpixel((250, 200)), (90, 90, 90))                    # inside the box: untouched
    assert near(out.getpixel((40, 40)), (90, 90, 90))                      # away from boxes: untouched


def test_large_photos_are_scaled_with_their_boxes(tmp_path):
    src = photo(tmp_path, size=(4000, 3000))
    out = annotate(src, [{"n": 1, "bbox": [1000, 1000, 3000, 2000]}], max_side=1000)
    assert out.size == (1000, 750) and near(out.getpixel((500, 251)), STATUS_COLORS["new"])


def test_crop_and_side_by_side(tmp_path):
    a, b = photo(tmp_path, "a.jpg"), photo(tmp_path, "b.jpg", color=(160, 120, 80))
    crop = crop_item(a, [100, 100, 300, 200])
    assert 1.8 < crop.width / crop.height < 2.2                            # the item's shape, plus a margin
    assert crop_item(a, None).size == (720, 540)                           # no box: the whole photo, scaled
    out = side_by_side(a, [100, 100, 300, 300], b, [200, 200, 600, 500], tmp_path / "same.jpg")
    sheet = Image.open(out)
    assert sheet.height == 600 and sheet.width > 1200
    assert legend([{"n": 1, "label": "sectional sofa"}, {"n": 2, "label": "floor lamp"}]) == "1. Sectional sofa\n2. Floor lamp"
