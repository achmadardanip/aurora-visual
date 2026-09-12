import asyncio

from starlette.responses import JSONResponse


class RequestSizeLimit:
    """Bound chunked bodies before multipart/JSON parsing, including missing length headers."""

    def __init__(self, app, max_upload):
        self.app, self.max_upload = app, max_upload

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        limit = (
            self.max_upload + 65536
            if path == "/api/v1/media"
            else self.max_upload * 4 + 65536
            if path == "/api/v1/import"
            else 2 * 1024**2
        )
        chunks, total = [], 0
        try:
            async with asyncio.timeout(60):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body = message.get("body", b"")
                    total += len(body)
                    if total > limit:
                        response = JSONResponse(
                            {
                                "error": {
                                    "code": "REQUEST_TOO_LARGE",
                                    "message": "Request melewati batas ukuran.",
                                    "retryable": False,
                                    "details": {},
                                }
                            },
                            status_code=413,
                        )
                        return await response(scope, receive, send)
                    chunks.append(body)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            response = JSONResponse(
                {
                    "error": {
                        "code": "REQUEST_TIMEOUT",
                        "message": "Unggahan melewati batas waktu.",
                        "retryable": True,
                        "details": {},
                    }
                },
                status_code=408,
            )
            return await response(scope, receive, send)
        pending = True

        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)
