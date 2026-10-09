import json

from evaluation import build_dataset_splits as builder


def test_build_splits_and_persona_invariants(tmp_path, monkeypatch):
    source = tmp_path / "evaluation.json"
    source.write_text(
        json.dumps(
            [
                {"query": "cozy comedy", "category": "vibe", "golden_ids": [1]},
                {"query": "noir mystery", "category": "vibe", "golden_ids": [2]},
                {"query": "short action", "category": "constraint", "golden_ids": [3]},
                {"query": "start", "category": "cold_start", "golden_ids": [4]},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "datasets"
    fixtures = tmp_path / "fixtures"
    monkeypatch.setattr(builder, "EVAL_SET_PATH", source)
    monkeypatch.setattr(builder, "OUTPUT_DIR", output)
    monkeypatch.setattr(builder, "FIXTURES_DIR", fixtures)
    builder.build_splits()
    first = (output / "product_dev.json").read_bytes()
    dev = json.loads(first)
    reg = json.loads((output / "product_reg.json").read_text(encoding="utf-8"))
    assert all(case["split"] == "dev" for case in dev)
    assert all(case["split"] == "regression" for case in reg)
    assert not ({c["family_id"] for c in dev} & {c["family_id"] for c in reg})
    assert all(c["category"] != "cold_start" for c in dev + reg)
    assert builder.generate_family_id("cozy comedy") == builder.generate_family_id(
        "Comedy cozy"
    )
    for persona in json.loads(
        (fixtures / "synthetic_personas.json").read_text(encoding="utf-8")
    ).values():
        groups = [
            {item["tmdb_id"] for item in persona[key]}
            for key in ("seed_history", "known_negatives", "hidden_targets")
        ]
        assert not (
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
    builder.build_splits()
    assert (output / "product_dev.json").read_bytes() == first
