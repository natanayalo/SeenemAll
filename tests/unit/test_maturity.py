from api.core.maturity import normalize_rating, rating_level, is_rating_within


def test_normalize_rating_handles_aliases():
    assert normalize_rating("tvma") == "TV-MA"
    assert normalize_rating("pg13") == "PG-13"
    assert normalize_rating("  ") is None


def test_rating_level_maps_known_values():
    assert rating_level("G") == 0
    assert rating_level("TV-Y7-FV") == 7
    assert rating_level("PG-13") == 13
    assert rating_level("Unknown") is None


def test_is_rating_within_respects_maximum():
    assert is_rating_within("PG", "PG-13") is True
    assert is_rating_within("TV-MA", "PG-13") is False
    assert is_rating_within(None, "PG") is False
    assert is_rating_within("PG-13", None) is True
