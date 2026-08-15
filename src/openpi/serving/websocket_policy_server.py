import asyncio
import http
import inspect
import json
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        log_every_n_requests: int = 0,
        log_payload_summaries: bool = False,
        log_prefix: str = "websocket-policy-server",
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._log_every_n_requests = max(0, int(log_every_n_requests))
        self._log_payload_summaries = bool(log_payload_summaries)
        self._log_prefix = str(log_prefix)
        self._request_count = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _should_log_request(self) -> bool:
        return self._log_every_n_requests > 0 and self._request_count % self._log_every_n_requests == 0

    async def run(self):
        serve_kwargs = {
            "compression": None,
            "max_size": None,
            "process_request": _health_check,
        }
        try:
            serve_params = inspect.signature(_server.serve).parameters
        except (TypeError, ValueError):
            serve_params = {}
        if "ping_interval" in serve_params:
            serve_kwargs["ping_interval"] = None
        if "ping_timeout" in serve_params:
            serve_kwargs["ping_timeout"] = None
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            **serve_kwargs,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                wait_start = time.monotonic()
                raw_obs = await websocket.recv()
                received_at = time.monotonic()
                unpack_start = time.monotonic()
                obs = msgpack_numpy.unpackb(raw_obs)
                unpack_ms = (time.monotonic() - unpack_start) * 1000.0

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "unpack_ms": unpack_ms,
                    "idle_wait_ms": (received_at - wait_start) * 1000.0,
                    "request_bytes": _payload_size_bytes(raw_obs),
                }

                pack_start = time.monotonic()
                packed_action = packer.pack(action)
                pack_ms = (time.monotonic() - pack_start) * 1000.0
                action["server_timing"]["pack_ms"] = pack_ms
                action["server_timing"]["response_bytes"] = _payload_size_bytes(packed_action)

                pack_start = time.monotonic()
                packed_action = packer.pack(action)
                action["server_timing"]["pack_ms_with_timing"] = (time.monotonic() - pack_start) * 1000.0
                action["server_timing"]["response_bytes"] = _payload_size_bytes(packed_action)

                send_start = time.monotonic()
                await websocket.send(packed_action)
                send_ms = (time.monotonic() - send_start) * 1000.0
                action["server_timing"]["send_ms"] = send_ms
                action["server_timing"]["request_total_ms"] = (time.monotonic() - received_at) * 1000.0

                self._request_count += 1
                if self._should_log_request():
                    logger.info(
                        "[%s] request=%d remote=%s recv_bytes=%s summary=%s",
                        self._log_prefix,
                        self._request_count,
                        websocket.remote_address,
                        action["server_timing"]["request_bytes"],
                        _json_summary(obs, self._log_payload_summaries),
                    )
                    logger.info(
                        "[%s] response=%d remote=%s send_bytes=%s summary=%s timing=%s",
                        self._log_prefix,
                        self._request_count,
                        websocket.remote_address,
                        action["server_timing"]["response_bytes"],
                        _json_summary(action, self._log_payload_summaries),
                        json.dumps(action["server_timing"], sort_keys=True, default=str),
                    )

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

def _payload_size_bytes(payload: object) -> int | None:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return None


def _summarize_array(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.ndim == 0:
        summary["value"] = array.item()
        return summary
    if array.ndim <= 1 and array.size <= 32:
        flat = array.reshape(-1)
        summary["first_values"] = flat[: min(8, flat.size)].tolist()
    if array.size > 0 and array.dtype.kind in {"f", "i", "u"} and array.ndim <= 2 and array.size <= 4096:
        summary["min"] = float(array.min())
        summary["max"] = float(array.max())
        summary["mean"] = float(array.mean())
    return summary


def _summarize_value(value: object, payload_logs: bool) -> object:
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        if not payload_logs:
            return {"keys": sorted(map(str, value.keys()))}
        return {str(k): _summarize_value(v, payload_logs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not payload_logs:
            return {"type": type(value).__name__, "len": len(value)}
        return [_summarize_value(v, payload_logs) for v in value[:8]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_summary(payload: object, payload_logs: bool) -> str:
    try:
        return json.dumps(_summarize_value(payload, payload_logs), sort_keys=True, default=str)
    except Exception as exc:  # pragma: no cover - best effort logging only
        return json.dumps({"summary_error": repr(exc), "type": type(payload).__name__}, sort_keys=True)
