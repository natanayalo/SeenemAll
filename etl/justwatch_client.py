from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

DEFAULT_BASE_URL = "https://apis.justwatch.com/graphql"
USER_AGENT = "SeenemAll/JustWatchSync (https://github.com/natanayalo/Seen-emAll)"
SEARCH_QUERY = """
query TitleSearch(
  $country: Country!,
  $language: Language!,
  $source: String!,
  $searchQuery: String!,
  $first: Int! = 5,
  $objectTypes: [ObjectType!]
) {
  searchTitles(
    country: $country,
    language: $language,
    source: $source,
    first: $first,
    filter: {
      searchQuery: $searchQuery,
      includeTitlesWithoutUrl: true,
      objectTypes: $objectTypes
    }
  ) {
    edges {
      node {
        id
        objectType
        objectId
        content(country: $country, language: $language) {
          title
          originalReleaseYear
        }
      }
    }
  }
}
""".strip()

TITLE_OFFERS_QUERY = """
query TitleOffers(
  $id: ID!,
  $country: Country!,
  $language: Language!,
  $platform: Platform!
) {
  node(id: $id) {
    __typename
    ... on MovieOrShowOrSeason {
      objectType
      objectId
      content(country: $country, language: $language) {
        title
        originalReleaseYear
      }
      offers(country: $country, platform: $platform, filter: { preAffiliate: true }) {
        monetizationType
        presentationType
        standardWebURL
        preAffiliatedStandardWebURL
        package {
          packageId
          shortName
          clearName
        }
      }
    }
  }
}
""".strip()

log = logging.getLogger(__name__)


class JustWatchError(RuntimeError):
    pass


class JustWatchClient:
    """
    Lightweight GraphQL client for fetching streaming availability data from JustWatch.
    """

    def __init__(
        self,
        country: str,
        language: str = "en",
        platform: str = "WEB",
        endpoint: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        source: str = "WEBAPP",
    ) -> None:
        self.country = country.upper()
        self.language = language.lower()
        self.platform = platform.upper()
        self.source = source
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}
        )

    async def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._client.post(
            self.endpoint, json={"query": query, "variables": variables}
        )
        resp.raise_for_status()
        payload = resp.json()
        errors = payload.get("errors") or []
        if errors:
            message = errors[0].get("message", "Unknown error")
            raise JustWatchError(message)
        return payload.get("data") or {}

    async def search_candidates(
        self,
        title: str,
        media_type: str,
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        variables = {
            "country": self.country,
            "language": self.language,
            "source": self.source,
            "searchQuery": title,
            "first": limit,
            "objectTypes": _map_media_type(media_type),
        }
        data = await self._graphql(SEARCH_QUERY, variables)
        edges = ((data.get("searchTitles") or {}).get("edges")) or []
        return [edge["node"] for edge in edges if "node" in edge]

    async def fetch_offers_by_id(self, node_id: str) -> List[Dict[str, Any]]:
        variables = {
            "id": node_id,
            "country": self.country,
            "language": self.language,
            "platform": self.platform,
        }
        data = await self._graphql(TITLE_OFFERS_QUERY, variables)
        node = data.get("node")
        if not node or "offers" not in node:
            return []
        offers = node.get("offers") or []
        return [offer for offer in offers if offer]

    async def resolve_offers_for_item(
        self,
        *,
        title: str,
        media_type: str,
        release_year: Optional[int],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            candidates = await self.search_candidates(title, media_type)
        except (httpx.HTTPError, JustWatchError) as exc:
            log.debug("JustWatch search failed for %s: %s", title, exc)
            return None, []

        match = _choose_best_match(title, media_type, release_year, candidates)
        if not match:
            return None, []

        try:
            offers = await self.fetch_offers_by_id(match["id"])
        except (httpx.HTTPError, JustWatchError) as exc:
            log.debug(
                "JustWatch offers failed for %s (%s): %s",
                title,
                match.get("id"),
                exc,
            )
            return match, []

        return match, offers

    async def aclose(self) -> None:
        await self._client.aclose()


def _map_media_type(media_type: str) -> Optional[List[str]]:
    media_type = media_type.lower()
    if media_type == "movie":
        return ["MOVIE"]
    if media_type in {"tv", "show", "series"}:
        return ["SHOW"]
    return None


def _normalise(text: Optional[str]) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _choose_best_match(
    title: str,
    media_type: str,
    release_year: Optional[int],
    candidates: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    target_norm = _normalise(title)
    mapped_type = _map_media_type(media_type)
    target_type = mapped_type[0] if mapped_type else None

    best: Optional[Dict[str, Any]] = None
    best_score = -1

    for entry in candidates:
        content = entry.get("content") or {}
        cand_title = content.get("title")
        cand_year = content.get("originalReleaseYear")
        cand_type = entry.get("objectType")

        score = 0
        if cand_type == target_type:
            score += 3
        elif cand_type and target_type and cand_type.startswith(target_type[:1]):
            score += 1

        if cand_title and cand_title.lower() == title.lower():
            score += 3
        elif _normalise(cand_title) == target_norm:
            score += 2

        if release_year and cand_year:
            diff = abs(int(cand_year) - int(release_year))
            if diff == 0:
                score += 4
            elif diff == 1:
                score += 1
        elif not release_year:
            score += 1

        if score > best_score:
            best = entry
            best_score = score

    return best


def flatten_offers(offers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw GraphQL offers into a simplified mapping used by the ETL sync.
    """
    prepared: List[Dict[str, Any]] = []
    for offer in offers:
        package = offer.get("package") or {}
        service = package.get("shortName") or str(package.get("packageId") or "")
        monetization = offer.get("monetizationType")
        if not service or not monetization:
            continue
        web_url = (
            offer.get("standardWebURL")
            or offer.get("preAffiliatedStandardWebURL")
            or None
        )
        prepared.append(
            {
                "service": service,
                "offer_type": monetization,
                "web_url": web_url,
                "deeplink": offer.get("preAffiliatedStandardWebURL")
                or offer.get("standardWebURL"),
            }
        )
    return prepared
