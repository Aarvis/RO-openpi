import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import tyro

from openpi.rl_multi_sample_serving import factory as _multi_factory
from openpi.rl_multi_sample_serving import router_server as _router_server
from openpi.rl_multi_sample_serving import websocket_policy_server as _multi_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    num_servers: int = 1
    start_port: int = 8000
    router_port: int | None = None
    gpu_id: str = "0"
    total_gpu_fraction: float = 1.0
    xla_preallocate: bool = True
    stagger_seconds: float = 2.0
    python_exe: str = sys.executable
    log_dir: str = "logs/multi_sample_serve_policy"
    dry_run: bool = False
    worker_mode: bool = False
    server_index: int = 0
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_multi_sample_policy(args: Args):
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            return _multi_factory.create_multi_sample_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            checkpoint = DEFAULT_CHECKPOINT.get(args.env)
            if checkpoint is None:
                raise ValueError(f"Unsupported environment mode: {args.env}")
            train_config = _config.get_config(checkpoint.config)
            return _multi_factory.create_multi_sample_policy(
                train_config,
                checkpoint.dir,
                default_prompt=args.default_prompt,
            )
    raise ValueError(f"Unsupported policy args: {args.policy}")


def _configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_worker_env(
    *,
    gpu_id: str,
    per_server_fraction: float,
    xla_preallocate: bool,
    server_index: int,
    port: int,
) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{per_server_fraction:.6f}"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if xla_preallocate else "false"
    env["OPENPI_MULTI_SERVER_INDEX"] = str(server_index)
    env["OPENPI_MULTI_SERVER_PORT"] = str(port)
    return env


def _worker_command(args: Args, *, port: int, server_index: int) -> list[str]:
    cmd = [
        args.python_exe,
        "-m",
        "scripts.multi_sample_serve_policy",
        "--worker-mode",
        "--server-index",
        str(server_index),
        "--port",
        str(port),
        "--env",
        args.env.value,
    ]
    if args.default_prompt is not None:
        cmd.extend(["--default-prompt", args.default_prompt])
    if args.policy.__class__ is Checkpoint:
        cmd.extend(
            [
                "policy:checkpoint",
                "--policy.config",
                args.policy.config,
                "--policy.dir",
                args.policy.dir,
            ]
        )
    return cmd


def _run_single_worker(args: Args) -> None:
    policy = create_multi_sample_policy(args)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating multi-sample worker server index=%s (host: %s, ip: %s, port: %s)",
        args.server_index,
        hostname,
        local_ip,
        args.port,
    )

    server = _multi_server.MultiSampleWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def _run_multi_server_orchestrator(args: Args) -> None:
    if args.num_servers <= 0:
        raise ValueError("--num-servers must be > 0.")
    if args.total_gpu_fraction <= 0:
        raise ValueError("--total-gpu-fraction must be > 0.")
    if args.num_servers == 1:
        args.port = args.start_port
        _run_single_worker(args)
        return

    per_server_fraction = args.total_gpu_fraction / args.num_servers
    if per_server_fraction > 1.0:
        raise ValueError(
            "Per-server fraction is > 1.0. Reduce --total-gpu-fraction or increase --num-servers."
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    router_port = args.router_port
    if router_port is None:
        if args.start_port <= 0:
            raise ValueError("router_port default would be invalid. Pass --router-port explicitly.")
        router_port = args.start_port - 1

    router_log_path = run_dir / f"router_port_{router_port}.log"
    _configure_logging(router_log_path)

    metadata: dict[str, object] = {
        "num_servers": args.num_servers,
        "start_port": args.start_port,
        "router_port": router_port,
        "gpu_id": args.gpu_id,
        "total_gpu_fraction": args.total_gpu_fraction,
        "per_server_fraction": per_server_fraction,
        "xla_preallocate": args.xla_preallocate,
        "stagger_seconds": args.stagger_seconds,
        "servers": [],
    }

    logging.info("run_dir=%s", run_dir)
    logging.info(
        "Launching %s worker servers with per_server_fraction=%.6f",
        args.num_servers,
        per_server_fraction,
    )

    procs: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    worker_ports = [args.start_port + i for i in range(args.num_servers)]
    for i, worker_port in enumerate(worker_ports):
        log_path = run_dir / f"server_{i:02d}_port_{worker_port}.log"
        cmd = _worker_command(args, port=worker_port, server_index=i)
        env = _build_worker_env(
            gpu_id=args.gpu_id,
            per_server_fraction=per_server_fraction,
            xla_preallocate=args.xla_preallocate,
            server_index=i,
            port=worker_port,
        )
        metadata["servers"].append(
            {
                "index": i,
                "port": worker_port,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )
        logging.info("worker[%s] port=%s log=%s", i, worker_port, log_path)
        if args.dry_run:
            continue

        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_path, worker_port))
        time.sleep(max(0.0, args.stagger_seconds))

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if args.dry_run:
        logging.info("dry-run complete")
        return

    router = _router_server.MultiSampleRouterServer(
        host="0.0.0.0",
        port=router_port,
        worker_host="127.0.0.1",
        worker_ports=worker_ports,
    )

    logging.info("router endpoint=ws://127.0.0.1:%s", router_port)
    logging.info(
        "worker endpoints=%s",
        ", ".join(f"ws://127.0.0.1:{worker_port}" for worker_port in worker_ports),
    )

    try:
        router.serve_forever()
    finally:
        logging.info("stopping worker servers...")
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2.0)
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.kill()


def main(args: Args) -> None:
    if args.worker_mode:
        _configure_logging()
        _run_single_worker(args)
        return

    if args.num_servers > 1:
        _run_multi_server_orchestrator(args)
        return

    _configure_logging()
    _run_single_worker(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
