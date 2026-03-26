from __future__ import annotations

import asyncio
import dataclasses
import http
import logging

import websockets.asyncio.client as _client
import websockets.asyncio.server as _server
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _WorkerEndpoint:
    port: int
    host: str = "127.0.0.1"
    active_connections: int = 0

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"


class MultiSampleRouterServer:
    """Routes each client connection to one backend multi-sample worker."""

    def __init__(
        self,
        *,
        worker_ports: list[int],
        host: str = "0.0.0.0",
        port: int | None = None,
        worker_host: str = "127.0.0.1",
    ) -> None:
        if not worker_ports:
            raise ValueError("worker_ports must be non-empty")
        self._host = host
        self._port = port
        self._workers = [_WorkerEndpoint(port=p, host=worker_host) for p in worker_ports]
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info(
            "Router listening on ws://%s:%s with worker ports=%s",
            self._host,
            self._port,
            [w.port for w in self._workers],
        )
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def _pick_worker(self) -> _WorkerEndpoint:
        return min(self._workers, key=lambda worker: (worker.active_connections, worker.port))

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        worker = self._pick_worker()
        worker.active_connections += 1
        logger.info(
            "Client %s assigned to worker port=%s active_connections=%s",
            websocket.remote_address,
            worker.port,
            worker.active_connections,
        )
        try:
            async with _client.connect(worker.uri, compression=None, max_size=None) as backend_ws:
                metadata = await backend_ws.recv()
                await websocket.send(metadata)
                while True:
                    request = await websocket.recv()
                    await backend_ws.send(request)
                    response = await backend_ws.recv()
                    await websocket.send(response)
        except ConnectionClosed:
            logger.info(
                "Client %s disconnected from worker port=%s",
                websocket.remote_address,
                worker.port,
            )
        finally:
            worker.active_connections = max(0, worker.active_connections - 1)
            logger.info(
                "Worker port=%s released active_connections=%s",
                worker.port,
                worker.active_connections,
            )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
