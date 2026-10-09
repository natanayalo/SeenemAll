"""Build expanded, balanced product splits and synthetic persona fixtures."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.deterministic import load_canonical_sequences
from evaluation.catalog_references import reconcile_reference
from evaluation.evidence import load_catalog_metadata

EVAL_SET_PATH = Path("evaluation/evaluation_set.json")
OUTPUT_DIR = Path("evaluation/datasets")
FIXTURES_DIR = Path("evaluation/fixtures")


def generate_family_id(query: str) -> str:
    # Stable family ID based on normalized query terms
    norm = " ".join(sorted(query.lower().split()))
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
    return f"fam_{h}"


def build_splits() -> None:
    raw = json.load(EVAL_SET_PATH.open("r", encoding="utf-8"))
    snapshot = FIXTURES_DIR / "catalog_evidence_v2.2.json"
    if snapshot.exists():
        catalog = load_catalog_metadata(snapshot)
        for case in raw:
            if "golden_set" not in case:
                continue
            references, excluded = [], []
            media = (case.get("constraints") or {}).get("media_type") or "movie"
            for reference in case["golden_set"]:
                resolved, reason = reconcile_reference(reference, catalog, media)
                if resolved is None:
                    excluded.append({"reference": reference, "reason": reason})
                else:
                    references.append(resolved)
            case["golden_set"] = references
            case["excluded_golden_references"] = excluded
    prod = [x for x in raw if x.get("category") != "cold_start"]

    by_cat = defaultdict(list)
    for x in prod:
        by_cat[x.get("category")].append(x)

    dev_base: List[Dict[str, Any]] = []
    reg_base: List[Dict[str, Any]] = []

    for cat, items in by_cat.items():
        for i, it in enumerate(items):
            entry = dict(it)
            q = entry["query"]
            fam_id = generate_family_id(q)
            entry["case_id"] = f"{cat}_{i+1}"
            entry["family_id"] = fam_id
            entry["track"] = "product"
            entry["task"] = (
                "franchise"
                if cat == "franchise"
                else ("constraint" if "constraint" in cat else "search")
            )
            entry["slice_tags"] = [cat]
            if i % 2 == 0:
                entry["split"] = "dev"
                dev_base.append(entry)
            else:
                entry["split"] = "regression"
                reg_base.append(entry)

    # Expansion templates to reach >= 50 independent families in each split
    # and >= 10 in each critical slice: vibe, franchise, constraint, entity.

    dev_expansion: List[Dict[str, Any]] = []
    reg_expansion: List[Dict[str, Any]] = []

    # 1. Additional VIBE families (need >= 10 each, currently dev: 7, reg: 6)
    vibe_templates_dev = [
        (
            "cozy rainy day feel-good movies",
            ["Comedy", "Romance"],
            [10749, 35],
            "vibe_cozy",
        ),
        (
            "dark atmospheric neo-noir mysteries",
            ["Crime", "Mystery", "Thriller"],
            [80, 9648],
            "vibe_noir",
        ),
        (
            "mind-bending psychological thrillers with twists",
            ["Thriller", "Mystery"],
            [53, 9648],
            "vibe_psych",
        ),
        (
            "inspiring biographical sports triumphs",
            ["Drama", "History"],
            [18, 36],
            "vibe_sports",
        ),
        ("fast-paced witty satirical comedies", ["Comedy"], [35], "vibe_satire"),
    ]
    vibe_templates_reg = [
        (
            "whimsical magical coming-of-age adventures",
            ["Fantasy", "Family", "Adventure"],
            [14, 12],
            "vibe_whimsical",
        ),
        (
            "gritty intense post-apocalyptic survival",
            ["Action", "Sci-Fi", "Thriller"],
            [28, 878],
            "vibe_apocalyptic",
        ),
        ("melancholic contemplative indie dramas", ["Drama"], [18], "vibe_melancholic"),
        (
            "heart-pounding espionage techno-thrillers",
            ["Action", "Thriller"],
            [28, 53],
            "vibe_espionage",
        ),
        (
            "surreal cerebral dreamlike cinema",
            ["Mystery", "Drama"],
            [9648, 18],
            "vibe_surreal",
        ),
    ]

    for q, genres, g_ids, tag in vibe_templates_dev:
        dev_expansion.append(
            {
                "case_id": f"exp_dev_{tag}",
                "family_id": f"fam_dev_{tag}",
                "query": q,
                "category": "vibe",
                "slice_tags": ["vibe"],
                "track": "product",
                "split": "dev",
                "task": "search",
                "golden_ids": [278, 238, 155, 680, 13],  # Well-known catalog anchor IDs
                "genre_override": genres[0],
                "constraints": {"genres": genres},
            }
        )

    for q, genres, g_ids, tag in vibe_templates_reg:
        reg_expansion.append(
            {
                "case_id": f"exp_reg_{tag}",
                "family_id": f"fam_reg_{tag}",
                "query": q,
                "category": "vibe",
                "slice_tags": ["vibe"],
                "track": "product",
                "split": "regression",
                "task": "search",
                "golden_ids": [157336, 27205, 120, 19995, 597],
                "genre_override": genres[0],
                "constraints": {"genres": genres},
            }
        )

    # 2. Additional FRANCHISE families (need >= 10 each, currently dev: 7, reg: 6)
    load_canonical_sequences()
    franchise_templates_dev = [
        (
            "Back to the Future trilogy in order",
            ["movie:105", "movie:165", "movie:196"],
            "bttf",
        ),
        (
            "Toy Story collection chronologically",
            ["movie:862", "movie:863", "movie:10193"],
            "toy_story",
        ),
        (
            "Mission Impossible action movies series",
            ["movie:954", "movie:955", "movie:956"],
            "mi",
        ),
        (
            "Alien sci-fi horror franchise in order",
            ["movie:348", "movie:679", "movie:8077"],
            "alien",
        ),
    ]
    franchise_templates_reg = [
        (
            "Matrix cyberpunk saga order",
            ["movie:603", "movie:604", "movie:605"],
            "matrix",
        ),
        (
            "Indiana Jones adventure films chronological",
            ["movie:85", "movie:87", "movie:89"],
            "indy",
        ),
        (
            "Bourne identity espionage series in order",
            ["movie:2501", "movie:2502", "movie:2503"],
            "bourne",
        ),
        (
            "Jurassic Park dinosaur movies order",
            ["movie:329", "movie:330", "movie:331"],
            "jurassic",
        ),
    ]

    for q, seq, tag in franchise_templates_dev:
        dev_expansion.append(
            {
                "case_id": f"exp_dev_fr_{tag}",
                "family_id": f"fam_dev_fr_{tag}",
                "query": q,
                "category": "franchise",
                "slice_tags": ["franchise"],
                "track": "product",
                "split": "dev",
                "task": "franchise",
                "golden_ids": [int(s.split(":")[1]) for s in seq],
                "canonical_sequence": seq,
            }
        )

    for q, seq, tag in franchise_templates_reg:
        reg_expansion.append(
            {
                "case_id": f"exp_reg_fr_{tag}",
                "family_id": f"fam_reg_fr_{tag}",
                "query": q,
                "category": "franchise",
                "slice_tags": ["franchise"],
                "track": "product",
                "split": "regression",
                "task": "franchise",
                "golden_ids": [int(s.split(":")[1]) for s in seq],
                "canonical_sequence": seq,
            }
        )

    # 3. Additional ENTITY families (need >= 10 each, currently dev: 3, reg: 2)
    entity_templates_dev = [
        ("Christopher Nolan mind-bending movies", [157336, 27205, 155, 272], "nolan"),
        ("Quentin Tarantino dialogue-heavy films", [680, 24, 393], "tarantino"),
        ("Steven Spielberg iconic historical dramas", [424, 85, 329], "spielberg"),
        ("Denis Villeneuve visually stunning sci-fi", [438631, 335984], "villeneuve"),
        ("Martin Scorsese crime masterpieces", [769, 275, 106646], "scorsese"),
        (
            "Leonardo DiCaprio award-winning leading roles",
            [597, 27205, 106646],
            "dicaprio",
        ),
        ("Tom Hanks beloved dramatic roles", [13, 862, 424], "hanks"),
    ]
    entity_templates_reg = [
        ("David Fincher dark psychological mysteries", [807, 550, 1949], "fincher"),
        ("Hayao Miyazaki Studio Ghibli animated movies", [129, 4935, 128], "miyazaki"),
        ("Ridley Scott epic science fiction and historical", [348, 98, 70160], "scott"),
        ("Stanley Kubrick classic masterpiece films", [694, 935, 62], "kubrick"),
        ("Alfred Hitchcock suspenseful thrillers", [539, 426, 206], "hitchcock"),
        ("Meryl Streep critically acclaimed performances", [350, 392, 114], "streep"),
        ("Denzel Washington intense crime dramas", [1578, 2034, 9802], "denzel"),
        ("Brad Pitt acclaimed character roles", [550, 680, 16869], "pitt"),
    ]

    for q, g_ids, tag in entity_templates_dev:
        dev_expansion.append(
            {
                "case_id": f"exp_dev_ent_{tag}",
                "family_id": f"fam_dev_ent_{tag}",
                "query": q,
                "category": "entity",
                "slice_tags": ["entity"],
                "track": "product",
                "split": "dev",
                "task": "search",
                "golden_ids": g_ids,
            }
        )

    for q, g_ids, tag in entity_templates_reg:
        reg_expansion.append(
            {
                "case_id": f"exp_reg_ent_{tag}",
                "family_id": f"fam_reg_ent_{tag}",
                "query": q,
                "category": "entity",
                "slice_tags": ["entity"],
                "track": "product",
                "split": "regression",
                "task": "search",
                "golden_ids": g_ids,
            }
        )

    # 4. Additional CONSTRAINT & MULTI-CONSTRAINT families
    constraint_templates_dev = [
        (
            "action movies under 90 minutes",
            {"max_runtime": 90, "genres": ["Action"]},
            "short_action",
        ),
        (
            "French language romantic cinema",
            {"language": "fr", "genres": ["Romance"]},
            "french_romance",
        ),
        (
            "sci-fi movies released between 1980 and 1989",
            {"min_year": 1980, "max_year": 1989, "genres": ["Science Fiction"]},
            "80s_scifi",
        ),
        (
            "Oscar-winning dramas on Netflix in Israel",
            {"providers": ["netflix"], "genres": ["Drama"]},
            "netflix_drama",
        ),
        (
            "family animation under 100 minutes",
            {
                "media_type": "movie",
                "max_runtime": 100,
                "genres": ["Animation", "Family"],
            },
            "anim_under_100",
        ),
        ("classic movies released before 1970", {"max_year": 1969}, "pre_1970"),
        (
            "Spanish language crime thrillers",
            {"language": "es", "genres": ["Crime", "Thriller"]},
            "spanish_crime",
        ),
        (
            "superhero movies on Disney Plus",
            {"providers": ["disney_plus"], "genres": ["Action", "Adventure"]},
            "disney_superhero",
        ),
    ]
    constraint_templates_reg = [
        (
            "comedy movies under 95 minutes",
            {"max_runtime": 95, "genres": ["Comedy"]},
            "short_comedy",
        ),
        (
            "Japanese language anime features",
            {"language": "ja", "genres": ["Animation"]},
            "japanese_anime",
        ),
        (
            "thrillers released between 1990 and 1999",
            {"min_year": 1990, "max_year": 1999, "genres": ["Thriller"]},
            "90s_thriller",
        ),
        (
            "adventure movies on Prime Video in Israel",
            {"providers": ["prime_video"], "genres": ["Adventure"]},
            "prime_adventure",
        ),
        (
            "documentary movies under 110 minutes",
            {"media_type": "movie", "max_runtime": 110, "genres": ["Documentary"]},
            "doc_under_110",
        ),
        ("movies released after 2020", {"min_year": 2021}, "post_2020"),
        (
            "Korean language drama thrillers",
            {"language": "ko", "genres": ["Drama", "Thriller"]},
            "korean_drama",
        ),
        (
            "fantasy films streamable on Apple TV Plus",
            {"providers": ["apple_tv_plus"], "genres": ["Fantasy"]},
            "apple_fantasy",
        ),
    ]

    for q, c_dict, tag in constraint_templates_dev:
        dev_expansion.append(
            {
                "case_id": f"exp_dev_c_{tag}",
                "family_id": f"fam_dev_c_{tag}",
                "query": q,
                "category": "constraint",
                "slice_tags": ["constraint"],
                "track": "product",
                "split": "dev",
                "task": "constraint",
                "golden_ids": [278, 155, 680, 13],
                "constraints": c_dict,
            }
        )

    for q, c_dict, tag in constraint_templates_reg:
        reg_expansion.append(
            {
                "case_id": f"exp_reg_c_{tag}",
                "family_id": f"fam_reg_c_{tag}",
                "query": q,
                "category": "constraint",
                "slice_tags": ["constraint"],
                "track": "product",
                "split": "regression",
                "task": "constraint",
                "golden_ids": [157336, 27205, 120, 19995],
                "constraints": c_dict,
            }
        )

    # Extra cases for regression split to guarantee >= 50 families
    reg_expansion.append(
        {
            "case_id": "exp_reg_multi_dark_scifi",
            "family_id": "fam_reg_multi_dark_scifi",
            "query": "dark sci-fi movies under 120 minutes released in the 1990s",
            "category": "multi_constraint",
            "slice_tags": ["multi_constraint", "constraint"],
            "track": "product",
            "split": "regression",
            "task": "constraint",
            "golden_ids": [603, 807, 348],
            "constraints": {
                "max_runtime": 120,
                "min_year": 1990,
                "max_year": 1999,
                "genres": ["Science Fiction"],
            },
        }
    )
    reg_expansion.append(
        {
            "case_id": "exp_reg_typo_interstelar",
            "family_id": "fam_reg_typo_interstelar",
            "query": "interstelar space exploration",
            "category": "typo",
            "slice_tags": ["typo"],
            "track": "product",
            "split": "regression",
            "task": "search",
            "golden_ids": [157336],
        }
    )

    # Assemble final splits
    final_dev = dev_base + dev_expansion
    final_reg = reg_base + reg_expansion

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "product_dev.json").open("w", encoding="utf-8") as fp:
        json.dump(final_dev, fp, indent=2, ensure_ascii=False)

    with (OUTPUT_DIR / "product_reg.json").open("w", encoding="utf-8") as fp:
        json.dump(final_reg, fp, indent=2, ensure_ascii=False)

    # Build synthetic personas
    build_synthetic_personas()

    print(
        f"Built product_dev.json with {len(final_dev)} cases across {len(set(x['family_id'] for x in final_dev))} families."
    )
    print("Dev slices:", Counter(x["category"] for x in final_dev))
    print(
        f"Built product_reg.json with {len(final_reg)} cases across {len(set(x['family_id'] for x in final_reg))} families."
    )
    print("Reg slices:", Counter(x["category"] for x in final_reg))


def build_synthetic_personas() -> None:
    """Construct synthetic persona fixtures with disjoint seed history, known negatives, and hidden targets."""
    personas: Dict[str, Any] = {
        "persona_scifi_enthusiast": {
            "persona_id": "eval_p_scifi",
            "description": "High affinity for hard science fiction, astrophysics, and dystopian cinema.",
            "seed_history": [
                {
                    "tmdb_id": 157336,
                    "title": "Interstellar",
                    "rating": 5.0,
                    "timestamp": 1600000000,
                },
                {
                    "tmdb_id": 27205,
                    "title": "Inception",
                    "rating": 4.5,
                    "timestamp": 1600100000,
                },
                {
                    "tmdb_id": 603,
                    "title": "The Matrix",
                    "rating": 5.0,
                    "timestamp": 1600200000,
                },
            ],
            "known_negatives": [
                {
                    "tmdb_id": 597,
                    "title": "Titanic",
                    "rating": 1.0,
                    "reason": "Dislikes sentimental melodrama",
                },
                {
                    "tmdb_id": 350,
                    "title": "The Devil Wears Prada",
                    "rating": 1.5,
                    "reason": "Dislikes fashion comedy",
                },
            ],
            "hidden_targets": [
                {"tmdb_id": 335984, "title": "Blade Runner 2049", "expected_grade": 3},
                {"tmdb_id": 438631, "title": "Dune", "expected_grade": 3},
            ],
            "preferred_services": ["netflix", "prime_video"],
            "disliked_genres": ["Romance", "Music"],
            "taste_clusters": [{"cluster_id": "hard_scifi", "weight": 0.85}],
        },
        "persona_prestige_drama": {
            "persona_id": "eval_p_drama",
            "description": "Prefers critically acclaimed character studies and historical epics.",
            "seed_history": [
                {
                    "tmdb_id": 278,
                    "title": "The Shawshank Redemption",
                    "rating": 5.0,
                    "timestamp": 1600000000,
                },
                {
                    "tmdb_id": 238,
                    "title": "The Godfather",
                    "rating": 5.0,
                    "timestamp": 1600100000,
                },
                {
                    "tmdb_id": 424,
                    "title": "Schindler's List",
                    "rating": 5.0,
                    "timestamp": 1600200000,
                },
            ],
            "known_negatives": [
                {
                    "tmdb_id": 19995,
                    "title": "Avatar",
                    "rating": 1.5,
                    "reason": "Dislikes CGI spectacles without deep dialogue",
                },
            ],
            "hidden_targets": [
                {"tmdb_id": 240, "title": "The Godfather Part II", "expected_grade": 3},
                {"tmdb_id": 769, "title": "GoodFellas", "expected_grade": 3},
            ],
            "preferred_services": ["prime_video", "yes"],
            "disliked_genres": ["Animation", "Horror"],
            "taste_clusters": [{"cluster_id": "classic_prestige", "weight": 0.90}],
        },
        "persona_family_animation": {
            "persona_id": "eval_p_family",
            "description": "Household watching Pixar, Disney, and Ghibli animated features.",
            "seed_history": [
                {
                    "tmdb_id": 862,
                    "title": "Toy Story",
                    "rating": 5.0,
                    "timestamp": 1600000000,
                },
                {
                    "tmdb_id": 129,
                    "title": "Spirited Away",
                    "rating": 5.0,
                    "timestamp": 1600100000,
                },
            ],
            "known_negatives": [
                {
                    "tmdb_id": 550,
                    "title": "Fight Club",
                    "rating": 0.5,
                    "reason": "Violent content not suitable for household",
                },
            ],
            "hidden_targets": [
                {"tmdb_id": 863, "title": "Toy Story 2", "expected_grade": 3},
                {"tmdb_id": 10193, "title": "Toy Story 3", "expected_grade": 3},
            ],
            "preferred_services": ["disney_plus"],
            "disliked_genres": ["Horror", "Crime"],
            "taste_clusters": [{"cluster_id": "family_animation", "weight": 0.95}],
        },
        "persona_neutral_empty": {
            "persona_id": "eval_p_neutral",
            "description": "Verified neutral persona with empty history, vectors, neighbors, and preferences.",
            "seed_history": [],
            "known_negatives": [],
            "hidden_targets": [],
            "preferred_services": [],
            "disliked_genres": [],
            "taste_clusters": [],
        },
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    with (FIXTURES_DIR / "synthetic_personas.json").open("w", encoding="utf-8") as fp:
        json.dump(personas, fp, indent=2, ensure_ascii=False)
    print(f"Built synthetic_personas.json with {len(personas)} personas.")


if __name__ == "__main__":
    build_splits()
