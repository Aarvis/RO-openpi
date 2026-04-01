from __future__ import annotations

import asyncio
import http
import logging
import traceback
from typing import Any

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.rl_multi_sample_serving.critic_runtime import OnlineRLCriticRuntime


logger = logging.getLogger(__name__)


class CriticScoreServer:
    def __init__(
        self,
        runtime: OnlineRLCriticRuntime,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._runtime = runtime
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Critic client %s connected", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                response = await asyncio.to_thread(self._handle_request, request)
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info("Critic client %s disconnected", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal critic server error. Traceback included in previous frame.",
                )
                raise

    def _handle_request(self, request: Any) -> dict[str, object]:
        if not isinstance(request, dict):
            raise TypeError(f"Expected request dict, got {type(request)!r}")
        for key in ("policy_latent", "state", "action_chunk"):
            if key not in request:
                raise KeyError(f"Critic request missing required key: {key}")
        return self._runtime.score(
            policy_latent=request["policy_latent"],
            state=request["state"],
            action_chunk=request["action_chunk"],
        )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
