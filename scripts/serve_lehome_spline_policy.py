import dataclasses
import logging
import socket

import tyro

from openpi.policies import lehome_spline_runtime_policy
from openpi.serving import websocket_policy_server
from scripts import serve_policy as _serve_policy


@dataclasses.dataclass
class Args:
    port: int = 8000
    default_prompt: str | None = None
    record: bool = False
    send_policy_latent: bool = False
    log_every_n_requests: int = 1
    log_payload_summaries: bool = True
    spline_server_url: str = "ws://127.0.0.1:9100"
    fail_on_invalid_spline: bool = True
    policy: _serve_policy.Checkpoint | _serve_policy.Default = dataclasses.field(default_factory=_serve_policy.Default)


def create_policy(args: Args):
    base_policy = _serve_policy.create_policy(
        _serve_policy.Args(
            default_prompt=args.default_prompt,
            send_policy_latent=args.send_policy_latent,
            policy=args.policy,
        )
    )
    policy = lehome_spline_runtime_policy.LehomeSplineRuntimePolicy(
        openpi_policy=base_policy,
        spline_server_url=args.spline_server_url,
        default_prompt=args.default_prompt,
        fail_on_invalid_spline=args.fail_on_invalid_spline,
    )
    if args.record:
        from openpi.policies import policy as _policy

        policy = _policy.PolicyRecorder(policy, "policy_records")
    return policy, getattr(base_policy, "metadata", {})


def main(args: Args) -> None:
    policy, base_metadata = create_policy(args)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "runtime_type": "lehome_spline_openpi_runtime",
            "spline_server_url": args.spline_server_url,
            "default_prompt": args.default_prompt,
        }
    )
    logging.info("Creating LeHome spline OpenPI server (host=%s ip=%s spline=%s)", hostname, local_ip, args.spline_server_url)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
        log_every_n_requests=int(args.log_every_n_requests),
        log_payload_summaries=bool(args.log_payload_summaries),
        log_prefix="openpi-spline-server",
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
