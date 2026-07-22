from pipeline.export_labeling import build_batch


def _gdelt(id, title="Bank news", language="English", title_hash=None):
    return {
        "id": id,
        "source": "gdelt",
        "bank_id": "pnc",
        "published_at": None,
        "title": title,
        "text_excerpt": None,
        "title_hash": title_hash,
        "meta": {"language": language},
    }


def _edgar(id, title="Holding 8-K", excerpt="Material event."):
    return {
        "id": id,
        "source": "edgar",
        "bank_id": "pnc",
        "published_at": None,
        "title": title,
        "text_excerpt": excerpt,
        "title_hash": None,
        "meta": {},
    }


def test_dedup_by_title_hash():
    rows = [
        _gdelt(1, title_hash="h1"),
        _gdelt(2, title_hash="h1"),
        _gdelt(3, title_hash="h2"),
    ]
    selected, funnel = build_batch(rows)
    assert funnel["total"] == 3
    assert funnel["unique"] == 2  # the second h1 is folded
    assert [r["id"] for r in selected] == [1, 3]


def test_none_title_hash_not_deduped():
    rows = [_edgar(1), _edgar(2)]  # edgar has no title_hash
    _, funnel = build_batch(rows)
    assert funnel["unique"] == 2


def test_non_english_skipped_and_counted():
    rows = [
        _gdelt(1, language="English", title_hash="a"),
        _gdelt(2, language="Spanish", title_hash="b"),
    ]
    selected, funnel = build_batch(rows)
    assert [r["id"] for r in selected] == [1]
    assert funnel["eligible"] == 1
    assert funnel["skipped"]["non_english"] == 1


def test_empty_title_skipped():
    selected, funnel = build_batch([_gdelt(1, title=None, title_hash="a")])
    assert selected == []
    assert funnel["skipped"]["empty_text"] == 1
