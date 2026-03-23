from __future__ import annotations

import asyncio
import http
import logging
from dataclasses import dataclass

import websockets
import websockets.asyncio.client as _client
import websockets.asyncio.server as _server

logger = logging.getLogger(__name__)


@dataclass
class _WorkerState:
    host: str
    port: int
    active_connections: int = 0

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"


class MultiSampleRouterServer:
    """Routes incoming websocket clients to the least-busy worker."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        worker_host: str,
        worker_ports: list[int],
    ) -> None:
        if not worker_ports:
            raise ValueError("worker_ports must not be empty.")
        self._host = host
        self._port = port
        self._workers = [_WorkerState(worker_host, worker_port) for worker_port in worker_ports]
        self._select_lock = asyncio.Lock()
        self._round_robin_index = 0
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
            logger.info(
                "Router listening on ws://%s:%s with worker ports=%s",
                self._host,
                self._port,
                [worker.port for worker in self._workers],
            )
            await server.serve_forever()

    async def _select_worker(self) -> _WorkerState:
        async with self._select_lock:
            min_active = min(worker.active_connections for worker in self._workers)
            candidates = [worker for worker in self._workers if worker.active_connections == min_active]
            worker = candidates[self._round_robin_index % len(candidates)]
            self._round_robin_index += 1
            worker.active_connections += 1
            return worker

    async def _release_worker(self, worker: _WorkerState) -> None:
        async with self._select_lock:
            worker.active_connections = max(0, worker.active_connections - 1)

    async def _handler(self, frontend_ws: _server.ServerConnection) -> None:
        worker = await self._select_worker()
        logger.info(
            "Client %s assigned to worker port=%s active_connections=%s",
            frontend_ws.remote_address,
            worker.port,
            worker.active_connections,
        )
        try:
            async with _client.connect(worker.uri, compression=None, max_size=None) as backend_ws:
                metadata = await backend_ws.recv()
                await frontend_ws.send(metadata)

                while True:
                    request = await frontend_ws.recv()
                    await backend_ws.send(request)
                    response = await backend_ws.recv()
                    await frontend_ws.send(response)
        except websockets.ConnectionClosed:
            logger.info("Client %s disconnected from worker port=%s", frontend_ws.remote_address, worker.port)
        finally:
            await self._release_worker(worker)
            logger.info("Worker port=%s released active_connections=%s", worker.port, worker.active_connections)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
