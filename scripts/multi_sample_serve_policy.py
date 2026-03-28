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
import urllib.request

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
    router_port: int = 7999
    gpu_id: str = "0"
    total_gpu_fraction: float = 1.0
    xla_preallocate: bool = True
    stagger_seconds: float = 2.0
    python_exe: str = sys.executable
    log_dir: str = "logs/multi_sample_serve_policy"
    worker_mode: bool = False
    worker_index: int = 0
    worker_log_path: str | None = None
    send_policy_latent: bool = False
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
    sample_kwargs = {"return_policy_latent": True} if args.send_policy_latent else None
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            return _multi_factory.create_multi_sample_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
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
                sample_kwargs=sample_kwargs,
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


def _create_single_server(args: Args) -> _multi_server.MultiSampleWebsocketPolicyServer:
    policy = create_multi_sample_policy(args)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating multi-sample worker server index=%s (host: %s, ip: %s, port: %s)",
        args.worker_index,
        hostname,
        local_ip,
        args.port,
    )

    return _multi_server.MultiSampleWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )


def _configure_worker_logging(args: Args) -> None:
    if args.worker_log_path is not None:
        _configure_logging(None)
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_mode:
        log_path = (log_dir / f"worker_{args.worker_index:02d}_port_{args.port}.log").resolve()
    else:
        log_path = (log_dir / f"single_server_port_{args.port}.log").resolve()
    _configure_logging(log_path)


def _build_worker_env(
    *,
    base_env: dict[str, str],
    gpu_id: str,
    per_server_fraction: float,
    xla_preallocate: bool,
    worker_index: int,
    port: int,
) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{per_server_fraction:.6f}"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if xla_preallocate else "false"
    env["OPENPI_MULTI_SAMPLE_SERVER_INDEX"] = str(worker_index)
    env["OPENPI_MULTI_SAMPLE_SERVER_PORT"] = str(port)
    return env


def _build_worker_cmd(args: Args, *, script_path: Path, port: int, worker_index: int, worker_log_path: Path) -> list[str]:
    cmd = [
        args.python_exe,
        str(script_path),
        "--worker-mode",
        "--worker-index",
        str(worker_index),
        "--port",
        str(port),
        "--log-dir",
        args.log_dir,
        "--worker-log-path",
        str(worker_log_path),
        "--env",
        args.env.name,
    ]
    if args.default_prompt is not None:
        cmd.extend(["--default-prompt", args.default_prompt])
    if args.send_policy_latent:
        cmd.append("--send-policy-latent")
    match args.policy:
        case Checkpoint():
            cmd.extend(
                [
                    "policy:checkpoint",
                    "--policy.config",
                    args.policy.config,
                    "--policy.dir",
                    args.policy.dir,
                ]
            )
        case Default():
            pass
    return cmd


def _wait_for_port(host: str, port: int, *, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    health_url = f"http://{host}:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError:
                time.sleep(0.5)
                continue
    raise TimeoutError(f"Timed out waiting for worker port {port} on {host}")


def _run_single_mode(args: Args) -> None:
    _configure_worker_logging(args)
    server = _create_single_server(args)
    server.serve_forever()


def _run_multi_mode(args: Args) -> None:
    if args.num_servers <= 1:
        raise ValueError("--num-servers must be > 1 for multi-worker mode.")
    if args.total_gpu_fraction <= 0:
        raise ValueError("--total-gpu-fraction must be > 0.")

    per_server_fraction = args.total_gpu_fraction / args.num_servers
    if per_server_fraction > 1.0:
        raise ValueError(
            "Per-worker XLA fraction is > 1.0. Reduce --total-gpu-fraction or increase --num-servers."
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (log_dir / f"run_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "router.log")

    logging.info("run_dir=%s", run_dir)
    logging.info(
        "Launching %s worker servers with per_server_fraction=%.6f",
        args.num_servers,
        per_server_fraction,
    )

    metadata: dict[str, object] = {
        "num_servers": args.num_servers,
        "start_port": args.start_port,
        "router_port": args.router_port,
        "gpu_id": args.gpu_id,
        "total_gpu_fraction": args.total_gpu_fraction,
        "per_server_fraction": per_server_fraction,
        "xla_preallocate": args.xla_preallocate,
        "stagger_seconds": args.stagger_seconds,
        "workers": [],
    }

    script_path = Path(__file__).resolve()
    procs: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    worker_ports: list[int] = []

    try:
        for worker_index in range(args.num_servers):
            port = args.start_port + worker_index
            worker_log_path = run_dir / f"server_{worker_index:02d}_port_{port}.log"
            cmd = _build_worker_cmd(
                args,
                script_path=script_path,
                port=port,
                worker_index=worker_index,
                worker_log_path=worker_log_path,
            )
            env = _build_worker_env(
                base_env=os.environ,
                gpu_id=args.gpu_id,
                per_server_fraction=per_server_fraction,
                xla_preallocate=args.xla_preallocate,
                worker_index=worker_index,
                port=port,
            )
            metadata["workers"].append(
                {
                    "index": worker_index,
                    "port": port,
                    "log_path": str(worker_log_path),
                    "cmd": cmd,
                }
            )
            logging.info("worker[%s] port=%s log=%s", worker_index, port, worker_log_path)
            log_handle = open(worker_log_path, "wb")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            procs.append((proc, worker_log_path, port))
            worker_ports.append(port)
            _wait_for_port("127.0.0.1", port, timeout_sec=max(30.0, args.stagger_seconds + 10.0))
            time.sleep(max(0.0, args.stagger_seconds))

        with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logging.info("router endpoint=ws://127.0.0.1:%s", args.router_port)
        logging.info(
            "worker endpoints=%s",
            ", ".join(f"ws://127.0.0.1:{port}" for port in worker_ports),
        )
        router = _router_server.MultiSampleRouterServer(
            worker_ports=worker_ports,
            host="0.0.0.0",
            port=args.router_port,
        )
        router.serve_forever()
    finally:
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2.0)
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.kill()


def main(args: Args) -> None:
    if args.worker_mode or args.num_servers <= 1:
        _run_single_mode(args)
        return
    _run_multi_mode(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
