from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client

from openpi_client import msgpack_numpy


class MultiSampleWebsocketClientPolicy:
    """Client for the isolated multi-sample websocket server."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logging.info("Waiting for multi-sample server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for multi-sample server...")
                time.sleep(5)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_many(
        self,
        obs: dict[str, Any],
        *,
        num_samples: int,
        seed: int | None = None,
        noise: Any = None,
    ) -> dict[str, Any]:
        request = {
            "method": "infer_many",
            "observation": obs,
            "num_samples": int(num_samples),
        }
        if seed is not None:
            request["seed"] = int(seed)
        if noise is not None:
            request["noise"] = noise
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
