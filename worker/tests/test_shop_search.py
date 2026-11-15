"""The item sheet's Amazon / Google Shopping links."""
import json

from ashline.web import shop_search


def urls(item):
    return [link["url"] for link in shop_search(item)]


def test_the_saved_query_is_searched_when_there_is_one():
    item = {"label": "wood-back dining chair with black metal frame", "brand_guess": None,
            "price_ref": json.dumps({"online": {"mid": 90, "queries": ["black metal frame dining chair wood seat", "other"]}})}
    assert urls(item) == ["https://www.amazon.com/s?k=black+metal+frame+dining+chair+wood+seat",
                          "https://www.google.com/search?tbm=shop&q=black+metal+frame+dining+chair+wood+seat"]


def test_without_one_the_whole_label_is_searched():
    item = {"label": "nightstand, rustic brown top, black fabric drawer", "brand_guess": "VASAGLE", "price_ref": None}
    assert urls(item)[0] == "https://www.amazon.com/s?k=VASAGLE+nightstand%2C+rustic+brown+top%2C+black+fabric+drawer"
    old = {"label": "blue velvet sofa", "price_ref": json.dumps({"online": {"mid": 1200, "matches": []}})}   # looked up before queries were kept
    assert urls(old)[0] == "https://www.amazon.com/s?k=blue+velvet+sofa"
    assert shop_search({"label": ""}) == []
