"""Deduplication: URL canonicalization, exact text hash, near-duplicate shingles.

Clusters become duplicate_cluster_id and independence_group_id so downstream
fusion (module 3) never counts one syndicated story as ten independent votes.
"""

import re
from urllib.parse import urlsplit, urlunsplit

from aurora_evidence.contract import sha
from aurora_evidence.core.tokenize import jaccard, shingles

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ref|source|mc_|spm)")


def canonical_url(url: str | None) -> str | None:
    """Strip tracking params, normalize scheme/host; None stays None."""
    if not url:
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return url
    query = "&".join(p for p in parts.query.split("&") if p and not TRACKING_PARAMS.match(p.split("=")[0]))
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/") or "/", query, ""))


def cluster_evidence(items: list[dict]) -> None:
    """Assign duplicate_cluster_id/independence_group_id in-place on evidence dicts.

    Clustering rules (auditable, order-independent):
    - same canonical URL -> same cluster
    - identical content text hash -> same cluster
    - shingle Jaccard >= 0.8 against an existing cluster member -> same cluster
    independence_group_id equals the cluster id unless the source declares its
    own explicit independence group (kept distinct).
    """
    clusters: list[dict] = []  # {urls, hashes, shingles, id}
    for item in items:
        url = canonical_url(item.get("source", {}).get("url"))
        text = item.get("content", {}).get("text") or ""
        text_hash = sha(text) if text else None
        item_shingles = shingles(text)
        cluster = None
        for candidate in clusters:
            if url and url in candidate["urls"]:
                cluster = candidate
                break
            if text_hash and text_hash in candidate["hashes"]:
                cluster = candidate
                break
        if cluster is None:
            for candidate in clusters:
                if item_shingles and jaccard(item_shingles, candidate["shingles"]) >= 0.8:
                    cluster = candidate
                    break
        if cluster is None:
            cluster = {"urls": set(), "hashes": set(), "shingles": set(), "id": None}
            clusters.append(cluster)
        if url:
            cluster["urls"].add(url)
        if text_hash:
            cluster["hashes"].add(text_hash)
        cluster["shingles"] = cluster["shingles"] | item_shingles
        item["provenance"]["duplicate_cluster_id"] = None  # filled below
        item["_cluster"] = cluster
    # Stable opaque ids from cluster member hashes (sorted for determinism).
    for cluster in clusters:
        members = sorted(cluster["hashes"] or cluster["urls"] or {"empty"})
        cluster["id"] = "dup_" + sha("|".join(str(m) for m in members))[:24]
    for item in items:
        cluster = item.pop("_cluster")
        item["provenance"]["duplicate_cluster_id"] = cluster["id"]
        override = item["provenance"].pop("independence_group_override", None)
        item["provenance"]["independence_group_id"] = override or cluster["id"]


def dhash(image) -> str:
    """64-bit difference hash (16 hex chars) for near-duplicate image matching."""
    small = image.convert("L").resize((9, 8))
    pixels = list(small.getdata())
    bits = []
    for row in range(8):
        for col in range(8):
            bits.append("1" if pixels[row * 9 + col] < pixels[row * 9 + col + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def match_type_for(hamming: int) -> str:
    if hamming == 0:
        return "exact"
    if hamming <= 10:
        return "near_duplicate"
    return "semantic"
