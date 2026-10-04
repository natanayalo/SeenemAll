from __future__ import annotations

from collections import defaultdict
import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import threading

import spacy
from spacy.matcher import PhraseMatcher
from sqlalchemy import select

from api.db.models import Item
from api.db.session import get_sessionmaker

logger = logging.getLogger(__name__)


def is_comparative_prefix(prefix: str) -> bool:
    """Return True if the text preceding an entity represents a comparative/similarity cue."""
    if not prefix:
        return False
    clean = prefix.rstrip(" ,.-").lower()
    if re.search(r"\b(?:i|we|user)\s+like$", clean):
        return False
    return bool(
        re.search(
            r"\b(?:like|similar\s+to|such\s+as|in\s+the\s+style\s+of|style\s+of|reminiscent\s+of|comparable\s+to|same\s+vibe\s+as)$",
            clean,
        )
    )


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

_FRANCHISE_SUFFIX_RE = re.compile(
    r"\s+(?:collection|trilogy|saga|series|movies|films)$", re.IGNORECASE
)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")


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
    matched_collections: Sequence[Tuple[int, str]] = ()


class QueryFilterMatcher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._nlp = spacy.blank("en")
        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._language_map: Dict[str, str] = {}
        self._keyword_map: Dict[str, str] = {}
        self._genre_map: Dict[str, str] = {}
        self._collection_map: Dict[str, Tuple[int, str]] = {}
        self._multi_collection_map: Dict[str, List[Tuple[int, str]]] = {}
        self._all_collections: Dict[int, str] = {}
        self._people_roles: Dict[str, Set[str]] = {}
        self._canonical_people: Dict[str, str] = {}
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
        collection_terms: Dict[str, Tuple[int, str]] = {}
        multi_collection_terms: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        all_collections: Dict[int, str] = {}
        people_roles: Dict[str, Set[str]] = {}
        canonical_people: Dict[str, str] = {}

        def _add_collection_term(term: str, entry: Tuple[int, str]) -> None:
            if not term:
                return
            collection_terms[term] = entry
            if entry not in multi_collection_terms[term]:
                multi_collection_terms[term].append(entry)

        rows = []
        try:
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
                        Item.collection_id,
                        Item.collection_name,
                    )
                ).all()
        except Exception as exc:
            logger.warning(
                "QueryFilterMatcher could not load vocab from database: %s", exc
            )
            rows = []

        def _ingest_language(entry: Mapping[str, object]) -> None:
            name = entry.get("english_name") or entry.get("name")
            if isinstance(name, str) and len(name.strip()) > 2:
                clean_name = name.strip()
                language_terms.setdefault(clean_name.lower(), clean_name)

        def _ingest_keyword(entry: Mapping[str, object]) -> None:
            name = entry.get("name")
            if isinstance(name, str):
                keyword_terms.setdefault(name.lower(), name)

        def _ingest_people(entries: Iterable[Mapping[str, object]], role: str) -> None:
            for person in entries:
                name = person.get("name")
                if not isinstance(name, str):
                    continue
                clean = name.strip()
                if not clean:
                    continue
                key = clean.lower()
                people_roles.setdefault(key, set()).add(role)
                if key not in canonical_people or any(c.isupper() for c in clean):
                    canonical_people[key] = clean

        def _ingest_genre(entry: Mapping[str, object]) -> None:
            name = entry.get("name")
            if isinstance(name, str):
                genre_terms.setdefault(name.lower(), name)

        for row in rows:
            langs, keywords, genres, cast, directors, producers, writers = row[:7]
            coll_id = row[7] if len(row) > 7 else None
            coll_name = row[8] if len(row) > 8 else None

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
                cast_entries = [e for e in cast[:5] if isinstance(e, Mapping)]
                _ingest_people(cast_entries, "cast")
            if isinstance(directors, list):
                director_entries = [e for e in directors if isinstance(e, Mapping)]
                _ingest_people(director_entries, "directors")
            if isinstance(producers, list):
                producer_entries = [e for e in producers[:3] if isinstance(e, Mapping)]
                _ingest_people(producer_entries, "producers")
            if isinstance(writers, list):
                writer_entries = [e for e in writers[:3] if isinstance(e, Mapping)]
                _ingest_people(writer_entries, "writers")
            if coll_id and coll_name and isinstance(coll_name, str):
                cname_clean = coll_name.strip()
                if cname_clean and coll_id not in all_collections:
                    all_collections[coll_id] = cname_clean
                    entry = (coll_id, cname_clean)
                    lower = cname_clean.lower()
                    _add_collection_term(lower, entry)

                    base = _FRANCHISE_SUFFIX_RE.sub("", lower).strip()
                    if base:
                        _add_collection_term(base, entry)
                        if base.startswith("the "):
                            _add_collection_term(base[4:].strip(), entry)

                        no_paren = ""
                        if "(tv)" not in base:
                            no_paren = _PAREN_RE.sub("", base).strip()
                            if no_paren and no_paren != base:
                                _add_collection_term(no_paren, entry)
                                if no_paren.startswith("the "):
                                    _add_collection_term(no_paren[4:].strip(), entry)

                        no_punct = re.sub(r"[^\w\s]", " ", base)
                        no_punct = re.sub(r"\s+", " ", no_punct).strip()
                        if no_punct and no_punct != base:
                            _add_collection_term(no_punct, entry)
                            if no_punct.startswith("the "):
                                _add_collection_term(no_punct[4:].strip(), entry)

                        no_hyphen = base.replace("-", "").strip()
                        if no_hyphen and no_hyphen != base:
                            _add_collection_term(no_hyphen, entry)

                        if base == "spider-man" or no_paren == "spider-man":
                            _add_collection_term("spiderman", entry)
                            _add_collection_term("spider man", entry)

        self._language_map = language_terms
        self._keyword_map = keyword_terms
        self._genre_map = genre_terms
        self._collection_map = collection_terms
        self._multi_collection_map = dict(multi_collection_terms)
        self._all_collections = all_collections
        self._people_roles = people_roles
        self._canonical_people = canonical_people

        def _add_patterns(label: str, terms: Iterable[str]) -> None:
            patterns = [self._nlp.make_doc(term) for term in terms if term]
            if patterns:
                self._matcher.add(label, patterns)

        _add_patterns("LANGUAGE", self._language_map.keys())
        _add_patterns("KEYWORD", self._keyword_map.keys())
        _add_patterns("GENRE", self._genre_map.keys())
        _add_patterns("COLLECTION", self._collection_map.keys())
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
        reference_titles: List[str] = []
        matched_collections: List[Tuple[int, str]] = []
        last_comparative_end: Optional[int] = None

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
            elif label == "COLLECTION":
                entries = self.resolve_collections(lower)
                if not entries:
                    entry = self._collection_map.get(lower)
                    if entry:
                        entries = [entry]
                for entry in entries:
                    if entry not in matched_collections:
                        matched_collections.append(entry)
            elif label == "MEDIA":
                canonical = self._media_map.get(lower, lower)
                _append_unique(media_types, canonical)
            elif label == "PERSON":
                roles = self._people_roles.get(lower, set())
                prefix_tokens = [
                    doc[i].text.lower() for i in range(max(0, start - 4), start)
                ]
                prefix_str = " ".join(prefix_tokens)

                is_conjunction_after_comparative = (
                    last_comparative_end is not None
                    and all(
                        doc[i].text.lower() in {",", "and", "or", "&"}
                        or doc[i].is_space
                        for i in range(last_comparative_end, start)
                    )
                )

                if (
                    is_comparative_prefix(prefix_str)
                    or is_conjunction_after_comparative
                ):
                    _append_unique(reference_titles, text)
                    last_comparative_end = end
                    if not allow_overlap:
                        used_tokens.update(range(start, end))
                    continue
                else:
                    last_comparative_end = None

                explicit_role: Optional[str] = None
                if any(
                    k in prefix_str for k in ("directed by", "director", "by director")
                ) or prefix_str.endswith("by"):
                    explicit_role = "directors"
                elif any(
                    k in prefix_str
                    for k in ("starring", "starred by", "stars", "actor", "actress")
                ) or prefix_str.endswith("with"):
                    explicit_role = "cast"
                elif any(
                    k in prefix_str
                    for k in ("written by", "writer", "screenplay by", "wrote")
                ):
                    explicit_role = "writers"
                elif any(k in prefix_str for k in ("produced by", "producer")):
                    explicit_role = "producers"

                if explicit_role:
                    if explicit_role == "cast":
                        _append_unique(cast, text)
                    elif explicit_role == "directors":
                        _append_unique(directors, text)
                    elif explicit_role == "producers":
                        _append_unique(producers, text)
                    elif explicit_role == "writers":
                        _append_unique(writers, text)
                else:
                    if "cast" in roles and len(roles) > 1:
                        _append_unique(cast, text)
                    elif (
                        "directors" in roles and len(roles) > 1 and "cast" not in roles
                    ):
                        _append_unique(directors, text)
                    else:
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

        extracted_refs = self._extract_reference_titles(doc, used_tokens)
        for ref in extracted_refs:
            _append_unique(reference_titles, ref)

        if matched_collections and "tv" in media_types:
            q_lower = query.lower()
            has_tv_explicit = any(
                w in q_lower
                for w in ("tv ", "tv show", "television", "miniseries", "sitcom")
            )
            if not has_tv_explicit:
                media_types = [m for m in media_types if m != "tv"]

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
            tuple(matched_collections),
        )

    def resolve_collection(
        self, name_or_query: str, threshold: float = 0.7
    ) -> Optional[Tuple[int, str]]:
        """Resolve a candidate collection/franchise name against catalog collections."""
        self._ensure_vocab()
        if not name_or_query or not name_or_query.strip():
            return None
        clean = name_or_query.strip().lower()
        if clean in self._collection_map:
            return self._collection_map[clean]

        base = re.sub(
            r"\s+(?:collection|trilogy|saga|series|movies|films)$",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        if base in self._collection_map:
            return self._collection_map[base]
        if base.startswith("the ") and base[4:].strip() in self._collection_map:
            return self._collection_map[base[4:].strip()]

        no_punct = re.sub(r"[^\w\s]", " ", base)
        no_punct = re.sub(r"\s+", " ", no_punct).strip()
        if no_punct in self._collection_map:
            return self._collection_map[no_punct]
        no_hyphen = base.replace("-", "").strip()
        if no_hyphen in self._collection_map:
            return self._collection_map[no_hyphen]

        for key, val in self._collection_map.items():
            if len(key) >= 4 and (key in clean or clean in key):
                return val

        if len(clean) < 4:
            return None

        matches = difflib.get_close_matches(
            clean, self._collection_map.keys(), n=1, cutoff=threshold
        )
        if matches:
            return self._collection_map[matches[0]]
        return None

    def resolve_collections(
        self, name_or_query: str, threshold: float = 0.8
    ) -> List[Tuple[int, str]]:
        """Resolve all matching collections for a franchise name or query."""
        self._ensure_vocab()
        if not name_or_query or not name_or_query.strip():
            return []
        clean = name_or_query.strip().lower()
        if clean in self._multi_collection_map:
            return list(self._multi_collection_map[clean])

        base = re.sub(
            r"\s+(?:collection|trilogy|saga|series|movies|films)$",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        if base in self._multi_collection_map:
            return list(self._multi_collection_map[base])
        if base.startswith("the ") and base[4:].strip() in self._multi_collection_map:
            return list(self._multi_collection_map[base[4:].strip()])

        no_paren = re.sub(r"\s*\([^)]*\)", "", base).strip()
        if no_paren in self._multi_collection_map:
            return list(self._multi_collection_map[no_paren])
        if (
            no_paren.startswith("the ")
            and no_paren[4:].strip() in self._multi_collection_map
        ):
            return list(self._multi_collection_map[no_paren[4:].strip()])

        no_punct = re.sub(r"[^\w\s]", " ", base)
        no_punct = re.sub(r"\s+", " ", no_punct).strip()
        if no_punct in self._multi_collection_map:
            return list(self._multi_collection_map[no_punct])
        no_hyphen = base.replace("-", "").strip()
        if no_hyphen in self._multi_collection_map:
            return list(self._multi_collection_map[no_hyphen])

        for key, val in self._multi_collection_map.items():
            if len(key) >= 4 and (key in clean or clean in key):
                return list(val)

        single = self.resolve_collection(name_or_query, threshold=threshold)
        return [single] if single else []

    def resolve_person(
        self, name: str, threshold: float = 0.8
    ) -> Tuple[Optional[str], Set[str]]:
        """Resolve a candidate person name against known catalog people (exact or fuzzy)."""
        self._ensure_vocab()
        if not name or not name.strip():
            return None, set()
        clean = name.strip()
        lower = clean.lower()
        if lower in self._people_roles:
            return self._canonical_people.get(lower, clean), self._people_roles[lower]
        if len(lower) < 4:
            return None, set()
        matches = difflib.get_close_matches(
            lower, self._people_roles.keys(), n=1, cutoff=threshold
        )
        if matches:
            matched_key = matches[0]
            return (
                self._canonical_people.get(matched_key, matched_key.title()),
                self._people_roles[matched_key],
            )
        return None, set()


_GLOBAL_MATCHER: Optional[QueryFilterMatcher] = None
_GLOBAL_LOCK = threading.Lock()


def get_filter_matcher() -> QueryFilterMatcher:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_MATCHER is None:
                _GLOBAL_MATCHER = QueryFilterMatcher()
    return _GLOBAL_MATCHER


def get_query_filters(query: Optional[str]) -> QueryFiltersResult:
    return get_filter_matcher().match(query)


def resolve_person_filter(
    name: str, threshold: float = 0.8
) -> Tuple[Optional[str], Set[str]]:
    return get_filter_matcher().resolve_person(name, threshold=threshold)


def resolve_collection_filters(
    name: str, threshold: float = 0.8
) -> List[Tuple[int, str]]:
    return get_filter_matcher().resolve_collections(name, threshold=threshold)


__all__ = [
    "QueryFiltersResult",
    "QueryFilterMatcher",
    "get_query_filters",
    "get_filter_matcher",
    "resolve_person_filter",
    "resolve_collection_filters",
]
