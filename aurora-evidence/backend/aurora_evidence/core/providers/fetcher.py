"""Bounded, SSRF-guarded HTML content fetcher.

Only http(s) to public hosts; DNS/redirect to private ranges, cloud metadata
and loopback are rejected. Text extraction is a simple auditable tag stripper
(no headless browser). Size and time capped by settings.
"""

import ipaddress
import re
import socket
import urllib.parse

import httpx

from aurora_evidence.core.providers.interfaces import ContentFetcher, ProviderError

PRIVATE_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
    )
]
TAGS = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
SPACES = re.compile(r"\s+")


def private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ProviderError("dns_error") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return True
        for network in PRIVATE_NETWORKS:
            if address in network:
                return True
    return False


class BoundedFetcher(ContentFetcher):
    name = "bounded-fetcher"
    capability = "content_fetch"

    def status(self, settings) -> str:
        return "ok" if settings.web_fetch else "disabled"

    def fetch(self, url: str, settings) -> tuple[str, str]:
        if not settings.web_fetch:
            return "", "unavailable"
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.username:
            raise ProviderError("url_rejected")
        if private_host(parts.hostname or ""):
            raise ProviderError("private_address_rejected")
        try:
            with httpx.Client(
                timeout=settings.provider_timeout,
                follow_redirects=True,
                max_redirects=3,
                headers={"User-Agent": "AURORA-Evidence/1.0 (research; contact: local)"},
            ) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            raise ProviderError(
                "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(f"http_error_{response.status_code}")
        if len(response.content) > settings.fetch_max_bytes:
            raise ProviderError("response_too_large")
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return "", "unavailable"
        text = SPACES.sub(" ", TAGS.sub(" ", response.text)).strip()
        return text[: settings.fetch_max_chars], "full"
