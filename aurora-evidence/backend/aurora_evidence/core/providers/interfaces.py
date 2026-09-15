"""Typed provider interfaces and registry with dependency injection.

Core retrieval logic only depends on these interfaces; vendor response fields
never leak into business logic. Providers declare capability/status so the API
and UI can render honest readiness without calling paid services.
"""

from dataclasses import dataclass, field


@dataclass
class SearchHit:
    """Provider-neutral search result before evidence normalization."""

    title: str
    url: str | None
    snippet: str = ""
    publisher: str | None = None
    language: str | None = None
    published_at: str | None = None
    kind: str = "web"  # fact_check | news | official | web | image_provenance | local_corpus
    rank: int = 0
    independence_group: str | None = None
    full_text: str | None = None  # local corpus documents already hold full text
    doc_id: str | None = None


@dataclass
class ImageMatchHit:
    matched_url: str | None
    match_type: str  # exact | near_duplicate | semantic
    score: float | None
    title: str = ""
    publisher: str | None = None
    published_at: str | None = None


class ProviderError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code, self.message = code, message
        super().__init__(code)


class SearchProvider:
    """Web/news search provider (e.g. Tavily). Returns ranked SearchHits."""

    name = "search"
    capability = "web_search"

    def status(self, settings) -> str:
        return "unconfigured"

    def search(self, query: str, settings, budget_ms: int) -> list[SearchHit]:
        raise NotImplementedError


class FactCheckProvider:
    """Fact-check corpus search (local BM25 corpus is the always-available path)."""

    name = "fact_check"
    capability = "fact_check_search"

    def status(self, settings) -> str:
        return "ok"

    def search(self, claim: str, language: str, as_of, settings, budget_ms: int) -> list[SearchHit]:
        raise NotImplementedError


class ContentFetcher:
    """Fetch a URL and extract readable text (bounded time/size, SSRF-guarded)."""

    name = "fetcher"
    capability = "content_fetch"

    def status(self, settings) -> str:
        return "ok"

    def fetch(self, url: str, settings) -> tuple[str, str]:
        """Return (extracted_text, status) where status is full|snippet_only|unavailable."""
        raise NotImplementedError


class ImageProvenanceProvider:
    """Local image index (dHash) — labeled local corpus search, not web-wide."""

    name = "image_provenance"
    capability = "local_image_index"

    def status(self, settings) -> str:
        return "ok"

    def search(self, image, settings) -> list[ImageMatchHit]:
        raise NotImplementedError


class ImageAIDetector:
    name = "image_ai_detector"
    capability = "ai_generation_detection_image"

    def status(self, settings) -> str:
        return "unconfigured"

    def detect(self, image_bytes: bytes, media_type: str, settings) -> dict:
        """Return {status, raw_score, raw_scale, raw_label, model_version, limitations, error_code}."""
        raise NotImplementedError


class TextAIDetector:
    name = "text_ai_detector"
    capability = "ai_generation_detection_text"

    def status(self, settings) -> str:
        return "unconfigured"

    def detect(self, text: str, language: str, settings) -> dict:
        raise NotImplementedError


@dataclass
class Registry:
    """All providers for a run; injected by settings so profiles are swappable."""

    search_providers: list = field(default_factory=list)
    fact_check_providers: list = field(default_factory=list)
    fetcher: ContentFetcher | None = None
    image_provenance: ImageProvenanceProvider | None = None
    image_detectors: list = field(default_factory=list)
    text_detectors: list = field(default_factory=list)

    def all_providers(self):
        for provider in (
            self.search_providers
            + self.fact_check_providers
            + ([self.fetcher] if self.fetcher else [])
            + ([self.image_provenance] if self.image_provenance else [])
            + self.image_detectors
            + self.text_detectors
        ):
            yield provider
