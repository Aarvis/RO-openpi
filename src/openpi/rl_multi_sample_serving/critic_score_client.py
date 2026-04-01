from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy


class CriticScoreClient:
    def __init__(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logging.info("Waiting for critic scorer at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for critic scorer...")
                time.sleep(2)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def score(
        self,
        *,
        policy_latent: np.ndarray,
        state: np.ndarray,
        action_chunk: np.ndarray,
    ) -> dict[str, Any]:
        request = {
            "policy_latent": np.asarray(policy_latent, dtype=np.float32),
            "state": np.asarray(state, dtype=np.float32),
            "action_chunk": np.asarray(action_chunk, dtype=np.float32),
        }
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in critic server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()
