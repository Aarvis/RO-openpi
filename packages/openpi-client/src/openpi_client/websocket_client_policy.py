import inspect
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = self._connect()
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _connect(self) -> websockets.sync.client.ClientConnection:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        connect_kwargs = {
            "compression": None,
            "max_size": None,
            "additional_headers": headers,
        }
        # Older websocket client builds may not expose keepalive kwargs on sync.connect.
        try:
            connect_params = inspect.signature(websockets.sync.client.connect).parameters
        except (TypeError, ValueError):
            connect_params = {}
        if "ping_interval" in connect_params:
            connect_kwargs["ping_interval"] = None
        if "ping_timeout" in connect_params:
            connect_kwargs["ping_timeout"] = None
        return websockets.sync.client.connect(self._uri, **connect_kwargs)

    def _reconnect(self) -> None:
        self._close_if_open()
        self._ws, self._server_metadata = self._wait_for_server()

    def _close_if_open(self) -> None:
        if getattr(self, "_ws", None) is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        for attempt in range(2):
            try:
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    # we're expecting bytes; if the server sends a string, it's an error.
                    raise RuntimeError(f"Error in inference server:\n{response}")
                return msgpack_numpy.unpackb(response)
            except (
                BrokenPipeError,
                ConnectionError,
                EOFError,
                OSError,
                websockets.exceptions.ConnectionClosed,
            ):
                if attempt >= 1:
                    raise
                logging.warning("Websocket connection to %s dropped; reconnecting once.", self._uri)
                self._reconnect()
        raise RuntimeError(f"Failed to infer from websocket server at {self._uri}")

    @override
    def reset(self) -> None:
        pass
