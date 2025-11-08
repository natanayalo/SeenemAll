from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
import threading

import spacy
from spacy.matcher import PhraseMatcher
from sqlalchemy import select

from api.db.models import Item
from api.db.session import get_sessionmaker

_REFERENCE_MARKERS: tuple[tuple[str, ...], ...] = (
    ("like",),
    ("similar", "to"),
    ("such", "as"),
    ("in", "the", "style", "of"),
    ("reminiscent", "of"),
)
_REFERENCE_STOP_TOKENS = {
    "and",
    "or",
    "but",
    "with",
    "that",
    "who",
    "which",
    "where",
    "when",
    "featuring",
    "starring",
    "including",
    "plus",
}
_REFERENCE_GENERIC_MEDIA = {
    "movie",
    "movies",
    "show",
    "shows",
    "series",
    "tv",
    "film",
    "films",
    "stories",
    "story",
}


@dataclass(frozen=True)
class QueryFiltersResult:
    languages: Sequence[str]
    keywords: Sequence[str]
    genres: Sequence[str]
    media_types: Sequence[str]
    cast: Sequence[str]
    directors: Sequence[str]
    producers: Sequence[str]
    writers: Sequence[str]
    residual_text: str
    reference_titles: Sequence[str] = ()


class QueryFilterMatcher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._nlp = spacy.blank("en")
        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._language_map: Dict[str, str] = {}
        self._keyword_map: Dict[str, str] = {}
        self._genre_map: Dict[str, str] = {}
        self._people_roles: Dict[str, Set[str]] = {}
        self._media_map: Dict[str, str] = {
            "movie": "movie",
            "movies": "movie",
            "film": "movie",
            "films": "movie",
            "tv": "tv",
            "tv show": "tv",
            "tv shows": "tv",
            "show": "tv",
            "shows": "tv",
            "series": "tv",
            "miniseries": "tv",
        }

    def _ensure_vocab(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._load_vocab()
            self._initialized = True

    def _load_vocab(self) -> None:
        SessionLocal = get_sessionmaker()
        language_terms: Dict[str, str] = {}
        keyword_terms: Dict[str, str] = {}
        genre_terms: Dict[str, str] = {}
        people_roles: Dict[str, Set[str]] = {}

        with SessionLocal() as session:
            rows = session.execute(
                select(
                    Item.spoken_languages,
                    Item.keywords,
                    Item.genres,
                    Item.cast,
                    Item.directors,
                    Item.producers,
                    Item.writers,
                )
            ).all()

        def _ingest_language(entry: Mapping[str, object]) -> None:
            name = entry.get("english_name") or entry.get("name")
            iso = entry.get("iso_639_1")
            if isinstance(name, str):
                language_terms.setdefault(name.lower(), name)
            if isinstance(iso, str):
                language_terms.setdefault(
                    iso.lower(), name if isinstance(name, str) else iso
                )

        def _ingest_keyword(entry: Mapping[str, object]) -> None:
            name = entry.get("name")
            if isinstance(name, str):
                keyword_terms.setdefault(name.lower(), name)

        def _ingest_people(entries: Iterable[Mapping[str, object]], role: str) -> None:
            for person in entries:
                name = person.get("name")
                if not isinstance(name, str):
                    continue
                key = name.lower()
                people_roles.setdefault(key, set()).add(role)

        def _ingest_genre(entry: Mapping[str, object]) -> None:
            name = entry.get("name")
            if isinstance(name, str):
                genre_terms.setdefault(name.lower(), name)

        for langs, keywords, genres, cast, directors, producers, writers in rows:
            if isinstance(langs, list):
                for entry in langs:
                    if isinstance(entry, Mapping):
                        _ingest_language(entry)
            if isinstance(keywords, list):
                for entry in keywords:
                    if isinstance(entry, Mapping):
                        _ingest_keyword(entry)
            if isinstance(genres, list):
                for entry in genres:
                    if isinstance(entry, Mapping):
                        _ingest_genre(entry)
            if isinstance(cast, list):
                cast_entries = [e for e in cast if isinstance(e, Mapping)]
                _ingest_people(cast_entries, "cast")
            if isinstance(directors, list):
                director_entries = [e for e in directors if isinstance(e, Mapping)]
                _ingest_people(director_entries, "directors")
            if isinstance(producers, list):
                producer_entries = [e for e in producers if isinstance(e, Mapping)]
                _ingest_people(producer_entries, "producers")
            if isinstance(writers, list):
                writer_entries = [e for e in writers if isinstance(e, Mapping)]
                _ingest_people(writer_entries, "writers")

        self._language_map = language_terms
        self._keyword_map = keyword_terms
        self._genre_map = genre_terms
        self._people_roles = people_roles

        def _add_patterns(label: str, terms: Iterable[str]) -> None:
            patterns = [self._nlp.make_doc(term) for term in terms if term]
            if patterns:
                self._matcher.add(label, patterns)

        _add_patterns("LANGUAGE", self._language_map.keys())
        _add_patterns("KEYWORD", self._keyword_map.keys())
        _add_patterns("GENRE", self._genre_map.keys())
        _add_patterns("PERSON", self._people_roles.keys())
        _add_patterns("MEDIA", self._media_map.keys())

    def _extract_reference_titles(
        self,
        doc,
        used_tokens: Set[int],
    ) -> List[str]:
        titles: List[str] = []
        doc_len = len(doc)
        idx = 0

        def _tokens_to_title(tokens: List[Any]) -> str:
            if not tokens:
                return ""
            words = [tok.text for tok in tokens]
            while words and words[-1].strip().lower() in _REFERENCE_GENERIC_MEDIA:
                words.pop()
            candidate = " ".join(words).strip(" \"'“”")
            if len(candidate) < 2:
                return ""
            letters = [ch for ch in candidate if ch.isalpha()]
            if not letters:
                return ""
            if not any(ch.isupper() for ch in letters):
                return ""
            return candidate

        while idx < doc_len:
            marker_len = 0
            for marker in _REFERENCE_MARKERS:
                if idx + len(marker) > doc_len:
                    continue
                if all(
                    doc[idx + offset].text.lower() == marker[offset]
                    for offset in range(len(marker))
                ):
                    marker_len = len(marker)
                    break
            if marker_len == 0:
                idx += 1
                continue
            start = idx + marker_len
            collected: List[Any] = []
            j = start
            while j < doc_len:
                tok = doc[j]
                lower = tok.text.lower()
                if tok.is_space:
                    j += 1
                    continue
                if tok.is_punct or lower in _REFERENCE_STOP_TOKENS:
                    break
                collected.append(tok)
                j += 1
            title = _tokens_to_title(collected)
            if title:
                normalized = title
                if normalized not in titles:
                    titles.append(normalized)
                used_tokens.update(range(idx, j))
                idx = j
            else:
                idx += 1
        return titles

    def match(self, query: Optional[str]) -> QueryFiltersResult:
        if not query:
            return QueryFiltersResult((), (), (), (), (), (), (), (), "")

        self._ensure_vocab()
        doc = self._nlp.make_doc(query)
        matches = self._matcher(doc)
        if not matches:
            reference_titles_only = tuple(self._extract_reference_titles(doc, set()))
            return QueryFiltersResult(
                (), (), (), (), (), (), (), (), query.strip(), reference_titles_only
            )

        sorted_matches = sorted(matches, key=lambda m: (m[1], m[2]))
        used_tokens: Set[int] = set()
        languages: List[str] = []
        keywords: List[str] = []
        genres: List[str] = []
        media_types: List[str] = []
        cast: List[str] = []
        directors: List[str] = []
        producers: List[str] = []
        writers: List[str] = []

        def _append_unique(container: List[str], value: Optional[str]) -> None:
            if value and value not in container:
                container.append(value)

        for match_id, start, end in sorted_matches:
            span = doc[start:end]
            text = span.text
            lower = text.lower()
            label = self._nlp.vocab.strings[match_id]
            allow_overlap = label == "KEYWORD"
            if not allow_overlap and any(i in used_tokens for i in range(start, end)):
                continue
            if label == "LANGUAGE":
                canonical = self._language_map.get(lower)
                _append_unique(languages, canonical)
            elif label == "KEYWORD":
                canonical = self._keyword_map.get(lower)
                _append_unique(keywords, canonical)
            elif label == "GENRE":
                canonical = self._genre_map.get(lower)
                _append_unique(genres, canonical)
            elif label == "MEDIA":
                canonical = self._media_map.get(lower, lower)
                _append_unique(media_types, canonical)
            elif label == "PERSON":
                roles = self._people_roles.get(lower, set())
                for role in roles:
                    if role == "cast":
                        _append_unique(cast, text)
                    elif role == "directors":
                        _append_unique(directors, text)
                    elif role == "producers":
                        _append_unique(producers, text)
                    elif role == "writers":
                        _append_unique(writers, text)
            if not allow_overlap:
                used_tokens.update(range(start, end))

        reference_titles = self._extract_reference_titles(doc, used_tokens)

        residual = "".join(
            token.text_with_ws for i, token in enumerate(doc) if i not in used_tokens
        ).strip()
        if keywords:
            keyword_blob = " ".join(keyword for keyword in keywords if keyword)
            if keyword_blob:
                residual = (
                    f"{residual} {keyword_blob}".strip() if residual else keyword_blob
                )

        return QueryFiltersResult(
            tuple(languages),
            tuple(keywords),
            tuple(genres),
            tuple(media_types),
            tuple(cast),
            tuple(directors),
            tuple(producers),
            tuple(writers),
            residual,
            tuple(reference_titles),
        )


_GLOBAL_MATCHER: Optional[QueryFilterMatcher] = None
_GLOBAL_LOCK = threading.Lock()


def get_query_filters(query: Optional[str]) -> QueryFiltersResult:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_MATCHER is None:
                _GLOBAL_MATCHER = QueryFilterMatcher()
    return _GLOBAL_MATCHER.match(query)


__all__ = ["QueryFiltersResult", "get_query_filters"]
