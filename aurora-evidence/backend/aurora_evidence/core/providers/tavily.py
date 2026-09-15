"""Tavily web search adapter (documented API: https://docs.tavily.com).

Live verification requires TAVILY_API_KEY on the server; without a key the
provider reports unconfigured and never falls back to fixture output.
"""

import httpx

from aurora_evidence.core.providers.interfaces import ProviderError, SearchHit, SearchProvider


class TavilySearchProvider(SearchProvider):
    name = "tavily"
    capability = "web_search"
    ENDPOINT = "https://api.tavily.com/search"

    def status(self, settings) -> str:
        return "ok" if settings.tavily_api_key else "unconfigured"

    def search(self, query: str, settings, budget_ms: int) -> list[SearchHit]:
        if not settings.tavily_api_key:
            raise ProviderError("unconfigured")
        try:
            with httpx.Client(timeout=min(settings.provider_timeout, budget_ms / 1000)) as client:
                response = client.post(
                    self.ENDPOINT,
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": settings.web_hits,
                        "search_depth": "basic",
                    },
                )
        except httpx.HTTPError as exc:
            raise ProviderError(
                "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            ) from exc
        if response.status_code in (401, 403):
            raise ProviderError("auth_error")
        if response.status_code == 429:
            raise ProviderError("rate_limited")
        if response.status_code >= 400:
            raise ProviderError(f"http_error_{response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("malformed_response") from exc
        hits = []
        for rank, result in enumerate(data.get("results", []), start=1):
            hits.append(
                SearchHit(
                    title=str(result.get("title") or "")[:500],
                    url=result.get("url"),
                    snippet=str(result.get("content") or "")[:2000],
                    publisher=None,
                    language=None,
                    published_at=result.get("published_date"),
                    kind="web",
                    rank=rank,
                )
            )
        return hits
