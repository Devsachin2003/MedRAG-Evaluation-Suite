"""Bound requests before multipart/JSON parsing, including chunked bodies."""

import asyncio
from starlette.responses import JSONResponse

from app import settings


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        try:
            async with asyncio.timeout(30):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    size += len(message.get("body", b""))
                    if size > settings.MAX_UPLOAD_BYTES + 65536:
                        return await JSONResponse({"detail": "Request too large"}, 413)(
                            scope, receive, send
                        )
                    chunks.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request timed out"}, 408)(
                scope, receive, send
            )

        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
