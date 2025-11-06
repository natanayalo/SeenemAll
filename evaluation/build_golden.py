from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from etl.tmdb_client import TMDBClient

load_dotenv()

DEFAULT_LIMIT = 15
DEFAULT_MIN_VOTE_COUNT = 250
DEFAULT_PAGES = 3


MOVIE_GENRES = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}


TV_GENRES = {
    "Action & Adventure": 10759,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Kids": 10762,
    "Mystery": 9648,
    "News": 10763,
    "Reality": 10764,
    "Sci-Fi & Fantasy": 10765,
    "Soap": 10766,
    "Talk": 10767,
    "War & Politics": 10768,
    "Western": 37,
}


@dataclass
class DiscoverSpec:
    query: str
    media_type: str
    keywords: Sequence[str] = field(default_factory=list)
    genres: Sequence[str] = field(default_factory=list)
    min_vote_count: int = DEFAULT_MIN_VOTE_COUNT
    sort_by: str = "vote_average.desc"
    pages: int = DEFAULT_PAGES
    limit: int = DEFAULT_LIMIT
    runtime_lte: Optional[int] = None
    runtime_gte: Optional[int] = None
    year_gte: Optional[str] = None
    year_lte: Optional[str] = None
    include_adult: bool = False


DISCOVER_RULES: Dict[str, DiscoverSpec] = {
    "mind-bending sci-fi movies like Inception": DiscoverSpec(
        query="mind-bending sci-fi movies like Inception",
        media_type="movie",
        keywords=["mind-bending", "dream"],
        genres=["Science Fiction", "Thriller"],
        min_vote_count=400,
        runtime_gte=100,
    ),
    "time-travel thrillers": DiscoverSpec(
        query="time-travel thrillers",
        media_type="movie",
        keywords=["time travel", "time loop"],
        genres=["Science Fiction", "Thriller"],
        min_vote_count=200,
    ),
    "heist TV series": DiscoverSpec(
        query="heist TV series",
        media_type="tv",
        keywords=["heist", "thief"],
        genres=["Crime"],
        min_vote_count=50,
        sort_by="vote_average.desc",
    ),
    "neo-noir crime movies": DiscoverSpec(
        query="neo-noir crime movies",
        media_type="movie",
        keywords=["neo noir"],
        genres=["Crime", "Thriller"],
        min_vote_count=200,
    ),
    "space-opera TV (not too dark)": DiscoverSpec(
        query="space-opera TV (not too dark)",
        media_type="tv",
        keywords=["space opera"],
        genres=["Sci-Fi & Fantasy"],
        min_vote_count=50,
    ),
    "post-apocalyptic TV shows": DiscoverSpec(
        query="post-apocalyptic TV shows",
        media_type="tv",
        keywords=["post-apocalyptic", "apocalypse"],
        genres=["Drama", "Sci-Fi & Fantasy"],
        min_vote_count=50,
    ),
    "feel-good comedies under 105 minutes": DiscoverSpec(
        query="feel-good comedies under 105 minutes",
        media_type="movie",
        genres=["Comedy"],
        min_vote_count=150,
        runtime_lte=105,
        sort_by="vote_average.desc",
    ),
    "anime sci-fi films": DiscoverSpec(
        query="anime sci-fi films",
        media_type="movie",
        genres=["Animation", "Science Fiction"],
        min_vote_count=100,
    ),
    "rom-coms from the 2000s": DiscoverSpec(
        query="rom-coms from the 2000s",
        media_type="movie",
        genres=["Comedy", "Romance"],
        min_vote_count=150,
        year_gte="2000-01-01",
        year_lte="2009-12-31",
    ),
    "gritty superhero TV (street-level)": DiscoverSpec(
        query="gritty superhero TV (street-level)",
        media_type="tv",
        keywords=["superhero", "anti hero"],
        genres=["Action & Adventure"],
        min_vote_count=50,
    ),
    "fantasy epics like The Witcher": DiscoverSpec(
        query="fantasy epics like The Witcher",
        media_type="tv",
        keywords=["fantasy", "magic"],
        genres=["Sci-Fi & Fantasy"],
        min_vote_count=50,
    ),
}


def _genre_ids(media_type: str, genres: Iterable[str]) -> List[int]:
    mapping = MOVIE_GENRES if media_type == "movie" else TV_GENRES
    ids: List[int] = []
    for name in genres:
        if name in mapping:
            ids.append(mapping[name])
    return ids


async def _keyword_ids(client: TMDBClient, keywords: Iterable[str]) -> List[int]:
    ids: List[int] = []
    seen: set[int] = set()
    for keyword in keywords:
        if not keyword:
            continue
        data = await client._get("/search/keyword", {"query": keyword})
        for row in data.get("results", []):
            kid = row.get("id")
            if isinstance(kid, int) and kid not in seen:
                ids.append(kid)
                seen.add(kid)
                break  # take best match
    return ids


async def fetch_candidates(
    client: TMDBClient, spec: DiscoverSpec
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "sort_by": spec.sort_by,
        "vote_count.gte": spec.min_vote_count,
        "include_adult": "true" if spec.include_adult else "false",
        "language": "en-US",
    }

    genre_ids = _genre_ids(spec.media_type, spec.genres)
    if genre_ids:
        params["with_genres"] = ",".join(str(gid) for gid in genre_ids)

    if spec.runtime_lte is not None:
        params["with_runtime.lte"] = spec.runtime_lte
    if spec.runtime_gte is not None:
        params["with_runtime.gte"] = spec.runtime_gte
    if spec.year_gte:
        key = (
            "first_air_date.gte"
            if spec.media_type == "tv"
            else "primary_release_date.gte"
        )
        params[key] = spec.year_gte
    if spec.year_lte:
        key = (
            "first_air_date.lte"
            if spec.media_type == "tv"
            else "primary_release_date.lte"
        )
        params[key] = spec.year_lte

    keyword_ids = await _keyword_ids(client, spec.keywords)
    if keyword_ids:
        params["with_keywords"] = ",".join(str(kid) for kid in keyword_ids)

    results: List[Dict[str, Any]] = []
    seen: set[int] = set()

    for page in range(1, spec.pages + 1):
        params["page"] = page
        data = await client.discover(spec.media_type, params)
        for item in data.get("results", []):
            tmdb_id = item.get("id")
            title = item.get("title") or item.get("name")
            if not isinstance(tmdb_id, int) or not title:
                continue
            if tmdb_id in seen:
                continue
            results.append({"id": tmdb_id, "title": title})
            seen.add(tmdb_id)
            if len(results) >= spec.limit:
                return results
    return results


def load_evaluation_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_evaluation_set(path: Path, payload: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


async def enrich_golden_set(
    input_path: Path, output_path: Path
) -> Tuple[int, Dict[str, int]]:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is required to build the golden set.")

    evaluation_entries = load_evaluation_set(input_path)
    enriched_entries = json.loads(json.dumps(evaluation_entries))  # deep copy
    client = TMDBClient(api_key)

    updated = 0
    counts: Dict[str, int] = {}

    try:
        for entry, enriched in zip(evaluation_entries, enriched_entries):
            query = entry.get("query")
            if not query or query not in DISCOVER_RULES:
                continue

            spec = DISCOVER_RULES[query]
            candidates = await fetch_candidates(client, spec)
            if not candidates:
                continue

            existing = entry.get("golden_set") or []
            existing_ids = {row["id"] for row in existing if "id" in row}

            merged: List[Dict[str, Any]] = []
            for cand in candidates:
                if cand["id"] not in existing_ids:
                    merged.append(cand)
            merged.extend(existing)

            enriched["golden_set"] = merged[: spec.limit]
            counts[query] = len(enriched["golden_set"])
            updated += 1
    finally:
        await client.aclose()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation_set(output_path, enriched_entries)
    return updated, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate golden sets using TMDB discover listings without "
            "modifying the existing evaluation set."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("evaluation/evaluation_set.json"),
        help="Path to evaluation set JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/golden_candidates.generated.json"),
        help="Output path for generated golden candidates.",
    )
    args = parser.parse_args()

    updated, counts = asyncio.run(enrich_golden_set(args.path, args.output))
    if not updated:
        print("No entries updated (missing TMDB data or rules).")
        return

    print(f"Wrote candidate golden sets to {args.output} (updated {updated} entries).")
    for query, size in counts.items():
        print(f"  {query}: {size} items")


if __name__ == "__main__":
    main()
