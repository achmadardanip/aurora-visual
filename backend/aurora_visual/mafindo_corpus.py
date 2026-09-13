"""MAFINDO V2 corpus puller: paginated metadata export for curation only.

The V2 API (https://mafindodocs.netlify.app/v2/) exposes POST /antihoax/ with
limit/offset pagination and /antihoax/get_total. Records are fact-check
metadata (title, content, fact, classification, media URLs); they contain no
image-caption visual ground truth and no training/redistribution license grant.
This module therefore exports a *curation corpus*: caption candidates for the
parser benchmark and weak-supervision text pairs. It is never imported as a
visual research corpus and never sent to GPU until the data owner grants
written license terms (see docs/data.md).
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import httpx

from aurora_visual.mafindo import _safe_text

V2_BASE = "https://yudistira.turnbackhoax.id/api/antihoax"
CORPUS_VERSION = "mafindo-corpus-v1"
_MAX_PAGE = 500
_MAX_FIELD = 10_000
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


class CorpusError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def strip_html(value) -> str:
    """Convert provider HTML to bounded plain text (tables/decorative markup dropped)."""
    if not isinstance(value, str):
        return ""
    text = _WHITESPACE.sub(" ", _TAG.sub(" ", value)).strip()
    return text[:_MAX_FIELD]


@dataclass
class MafindoCorpusClient:
    api_key: str
    timeout: float = 30
    transport: httpx.BaseTransport | None = None

    @property
    def configured(self):
        return bool(self.api_key)

    def _post(self, path: str, data: dict):
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
                transport=self.transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = client.post(V2_BASE + path, data=data)
                if response.status_code == 429:
                    raise CorpusError("rate_limited")
                if response.status_code >= 400:
                    raise CorpusError("http_error")
                if len(response.content) > 8_000_000:
                    raise CorpusError("response_too_large")
                return response.json()
        except CorpusError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise CorpusError(
                "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            ) from None
        except (OSError, ValueError, TypeError):
            raise CorpusError("malformed_response") from None

    def get_total(self) -> int:
        if not self.configured:
            raise CorpusError("unconfigured")
        try:
            total = int(str(self._post("/get_total", {"key": self.api_key})).strip().strip('"'))
        except (TypeError, ValueError):
            raise CorpusError("malformed_response") from None
        if total < 0:
            raise CorpusError("malformed_response")
        return total

    def list_page(self, limit: int = 100, offset: int = 0) -> list[dict]:
        if not self.configured:
            raise CorpusError("unconfigured")
        if not 1 <= limit <= _MAX_PAGE or offset < 0:
            raise ValueError("limit must be 1-500 and offset non-negative")
        payload = self._post("/", {"key": self.api_key, "limit": limit, "offset": offset})
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise CorpusError("malformed_response")
        return rows


def normalize_record(row: dict, api_key: str) -> dict | None:
    """Bound and key-redact one provider record for the curation corpus."""
    record_id = _safe_text(row.get("id"), api_key, limit=100)
    if not record_id or record_id == "[REDACTED]":
        return None

    def bounded(value, limit=_MAX_FIELD):
        return _safe_text(value, api_key, limit=limit) or None

    return {
        "id": record_id,
        "authors": bounded(row.get("authors"), 200),
        "status": bounded(row.get("status"), 100),
        "classification": bounded(row.get("classification"), 200),
        "category": bounded(row.get("category"), 200),
        "title": bounded(row.get("title")),
        "content_plain": strip_html(row.get("content")) or None,
        "fact_plain": strip_html(row.get("fact")) or None,
        "conclusion_plain": strip_html(row.get("conclusion")) or None,
        "references": bounded(row.get("references"), 2000),
        "source_issue": bounded(row.get("source_issue"), 200),
        "source_link": bounded(row.get("source_link"), 2000),
        "picture1": bounded(row.get("picture1"), 2000),
        "picture2": bounded(row.get("picture2"), 2000),
        "tanggal": bounded(row.get("tanggal"), 100),
        "tags": bounded(row.get("tags"), 2000),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class PullState:
    next_offset: int = 0
    total: int | None = None
    unique: int = 0
    duplicates: int = 0
    requests: int = 0
    finished: bool = False


def pull_corpus(
    api_key: str,
    output,
    max_records: int | None = None,
    page_size: int = 100,
    delay: float = 0.4,
    timeout: float = 30,
    max_retries: int = 4,
    transport: httpx.BaseTransport | None = None,
    progress=lambda state: None,
) -> dict:
    """Pull the MAFINDO archive into a resumable, deduplicated JSONL corpus.

    Output is a curation corpus (parser-benchmark caption candidates and
    weak-supervision text sources). It carries no visual labels and must not be
    used for visual training before a written license grant from the owner.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = MafindoCorpusClient(api_key, timeout=timeout, transport=transport)
    if not client.configured:
        raise CorpusError("unconfigured")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    seen: set[str] = set()
    if output.is_file():
        for line in output.read_text().splitlines():
            if line.strip():
                try:
                    seen.add(str(strict_json_line(line)["id"]))
                except ValueError:
                    continue
    state_path = output.with_suffix(".state.json")
    state = PullState()
    if state_path.is_file():
        try:
            saved = json.loads(state_path.read_text())
            state = PullState(
                next_offset=int(saved["next_offset"]),
                total=saved.get("total"),
                unique=len(seen),
                duplicates=int(saved.get("duplicates", 0)),
                requests=int(saved.get("requests", 0)),
                finished=bool(saved.get("finished", False)),
            )
        except (ValueError, KeyError, TypeError):
            state = PullState(unique=len(seen))
    if state.finished:
        return _summary(state, output, seen)
    if state.total is None:
        state.total = client.get_total()
    with output.open("a") as sink:
        while True:
            if max_records is not None and len(seen) >= max_records:
                break
            rows, retries = None, 0
            while rows is None:
                try:
                    rows = client.list_page(page_size, state.next_offset)
                    state.requests += 1
                except CorpusError as exc:
                    if exc.code != "rate_limited" or retries >= max_retries:
                        _save_state(state_path, state)
                        raise
                    retries += 1
                    sleep(min(2**retries, 30))
            if not rows:
                break
            for row in rows:
                record = normalize_record(row, api_key)
                if record is None:
                    continue
                if record["id"] in seen:
                    state.duplicates += 1
                    continue
                if max_records is not None and len(seen) >= max_records:
                    break
                seen.add(record["id"])
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                state.unique += 1
            state.next_offset += page_size
            _save_state(state_path, state)
            progress(state)
            if len(rows) < page_size:
                break
            if delay > 0:
                sleep(delay)
    state.finished = True
    _save_state(state_path, state)
    return _summary(state, output, seen)


def strict_json_line(line: str) -> dict:
    from app.models.contract import strict_json

    payload = strict_json(line.encode())
    if not isinstance(payload, dict):
        raise ValueError("corpus line is not an object")
    return payload


def _summary(state: PullState, output: Path, seen: set) -> dict:
    from app.models.contract import sha

    return {
        "version": CORPUS_VERSION,
        "output": str(output),
        "provider_total": state.total,
        "unique_records": state.unique or len(seen),
        "duplicates_skipped": state.duplicates,
        "requests": state.requests,
        "finished": state.finished,
        "corpus_sha256": sha(output.read_bytes()) if output.is_file() else None,
        "usage": (
            "Curation corpus only: caption candidates for the parser benchmark and "
            "weak-supervision text pairs. Not a visual dataset: no image-caption ground "
            "truth, no S/C/U labels, no training/redistribution license. Media URLs are "
            "metadata references and are not downloaded."
        ),
    }


def _save_state(path: Path, state: PullState):
    path.write_text(
        json.dumps(
            {
                "next_offset": state.next_offset,
                "total": state.total,
                "unique": state.unique,
                "duplicates": state.duplicates,
                "requests": state.requests,
                "finished": state.finished,
            }
        )
    )
