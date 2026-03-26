from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.rl_multi_sample_serving.multi_sample_policy import MultiSamplePolicy

logger = logging.getLogger(__name__)


class MultiSampleWebsocketPolicyServer:
    """Isolated websocket server supporting infer and infer_many."""

    def __init__(
        self,
        policy: MultiSamplePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
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
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())
                action = await asyncio.to_thread(self._handle_request, request)
                action["server_timing"] = {
                    "infer_ms": (time.monotonic() - start_time) * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                policy_timing = action.get("policy_timing", {})
                logger.info(
                    "Completed request from %s total_ms=%.1f policy_ms=%s model_ms=%s output_ms=%s samples=%s",
                    websocket.remote_address,
                    action["server_timing"]["infer_ms"],
                    _fmt_timing(policy_timing.get("infer_ms")),
                    _fmt_timing(policy_timing.get("model_infer_ms")),
                    _fmt_timing(policy_timing.get("output_transform_ms")),
                    policy_timing.get("num_samples"),
                )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _handle_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError(f"Expected request dict, got {type(request)!r}")

        if "method" not in request and "observation" not in request:
            return self._policy.infer(request)

        method = request.get("method")
        if method is None:
            method = "infer_many" if any(k in request for k in ("num_samples", "seed", "noise")) else "infer"

        if "observation" not in request:
            raise KeyError("Request must include 'observation' when using the structured protocol.")
        observation = request["observation"]

        if method == "infer":
            return self._policy.infer(observation)
        if method == "infer_many":
            num_samples = int(request.get("num_samples", 1))
            seed = request.get("seed")
            seed = None if seed is None else int(seed)
            noise = request.get("noise")
            return self._policy.infer_many(
                observation,
                num_samples=num_samples,
                seed=seed,
                noise=noise,
            )
        raise ValueError(f"Unsupported method: {method}")


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _fmt_timing(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return str(value)
