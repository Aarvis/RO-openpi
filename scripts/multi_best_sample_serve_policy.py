from __future__ import annotations

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

from openpi.rl_multi_sample_serving import best_of_n_policy as _best_of_n_policy
from openpi.rl_multi_sample_serving import critic_runtime as _critic_runtime
from openpi.rl_multi_sample_serving import critic_score_client as _critic_score_client
from openpi.rl_multi_sample_serving import critic_score_server as _critic_score_server
from openpi.rl_multi_sample_serving import factory as _multi_factory
from openpi.serving import websocket_policy_server
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


def _default_critic_repo_root() -> str:
    return str((Path(__file__).resolve().parents[2] / "lehome-challenge").resolve())


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    num_servers: int = 1
    start_port: int = 8000
    gpu_id: str = "0"
    total_gpu_fraction: float = 1.0
    critic_gpu_fraction: float = 0.10
    xla_preallocate: bool = True
    stagger_seconds: float = 2.0
    python_exe: str = sys.executable
    log_dir: str = "logs/multi_best_sample_serve_policy"
    worker_mode: bool = False
    critic_mode: bool = False
    worker_index: int = 0
    worker_log_path: str | None = None
    critic_log_path: str | None = None
    send_policy_latent: bool = False
    num_samples: int = 8
    critic_port: int = 7998
    critic_host: str = "127.0.0.1"
    critic_checkpoint: str = ""
    critic_repo_root: str = dataclasses.field(default_factory=_default_critic_repo_root)
    critic_device: str = "cuda"
    critic_amp_dtype: str = "auto"
    port: int = 8000
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
    sample_kwargs = {"return_policy_latent": True}
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


def _configure_worker_logging(args: Args) -> None:
    explicit_log_path = args.critic_log_path if args.critic_mode else args.worker_log_path
    if explicit_log_path is not None:
        _configure_logging(Path(explicit_log_path).resolve())
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.critic_mode:
        log_path = (log_dir / f"critic_port_{args.critic_port}.log").resolve()
    else:
        log_path = (log_dir / f"worker_{args.worker_index:02d}_port_{args.port}.log").resolve()
    _configure_logging(log_path)


def _create_best_of_n_policy(args: Args):
    multi_sample_policy = create_multi_sample_policy(args)
    critic_client = _critic_score_client.CriticScoreClient(host=args.critic_host, port=args.critic_port)
    return _best_of_n_policy.BestOfNSamplePolicy(
        policy=multi_sample_policy,
        critic_client=critic_client,
        num_samples=args.num_samples,
        send_policy_latent=args.send_policy_latent,
    )


def _run_worker_mode(args: Args) -> None:
    _configure_worker_logging(args)
    policy = _create_best_of_n_policy(args)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating best-of-N worker server index=%s (host: %s, ip: %s, port: %s, num_samples=%s)",
        args.worker_index,
        hostname,
        local_ip,
        args.port,
        args.num_samples,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


def _run_critic_mode(args: Args) -> None:
    if not args.critic_checkpoint:
        raise ValueError("--critic-checkpoint is required")
    _configure_worker_logging(args)
    runtime = _critic_runtime.OnlineRLCriticRuntime(
        critic_repo_root=args.critic_repo_root,
        checkpoint_path=args.critic_checkpoint,
        device=args.critic_device,
        amp_dtype=args.critic_amp_dtype,
        gpu_memory_fraction=args.critic_gpu_fraction,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating critic scorer (host: %s, ip: %s, port: %s, device: %s)",
        hostname,
        local_ip,
        args.critic_port,
        runtime.device,
    )
    server = _critic_score_server.CriticScoreServer(
        runtime=runtime,
        host="0.0.0.0",
        port=args.critic_port,
        metadata={
            "device": str(runtime.device),
            "critic_checkpoint": str(Path(args.critic_checkpoint).resolve()),
        },
    )
    server.serve_forever()


def _build_policy_worker_env(
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
    env["OPENPI_MULTI_BEST_SERVER_INDEX"] = str(worker_index)
    env["OPENPI_MULTI_BEST_SERVER_PORT"] = str(port)
    return env


def _build_critic_env(*, base_env: dict[str, str], gpu_id: str, port: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["OPENPI_CRITIC_SERVER_PORT"] = str(port)
    return env


def _build_policy_worker_cmd(
    args: Args,
    *,
    script_path: Path,
    port: int,
    worker_index: int,
    worker_log_path: Path,
) -> list[str]:
    cmd = [
        args.python_exe,
        str(script_path),
        "--worker-mode",
        "--worker-index",
        str(worker_index),
        "--port",
        str(port),
        "--num-servers",
        str(args.num_servers),
        "--gpu-id",
        args.gpu_id,
        "--total-gpu-fraction",
        str(args.total_gpu_fraction),
        "--critic-gpu-fraction",
        str(args.critic_gpu_fraction),
        "--critic-port",
        str(args.critic_port),
        "--critic-host",
        args.critic_host,
        "--critic-checkpoint",
        args.critic_checkpoint,
        "--critic-repo-root",
        args.critic_repo_root,
        "--critic-device",
        args.critic_device,
        "--critic-amp-dtype",
        args.critic_amp_dtype,
        "--num-samples",
        str(args.num_samples),
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


def _build_critic_cmd(args: Args, *, script_path: Path, critic_log_path: Path) -> list[str]:
    return [
        args.python_exe,
        str(script_path),
        "--critic-mode",
        "--critic-port",
        str(args.critic_port),
        "--critic-checkpoint",
        args.critic_checkpoint,
        "--critic-repo-root",
        args.critic_repo_root,
        "--critic-device",
        args.critic_device,
        "--critic-amp-dtype",
        args.critic_amp_dtype,
        "--critic-gpu-fraction",
        str(args.critic_gpu_fraction),
        "--log-dir",
        args.log_dir,
        "--critic-log-path",
        str(critic_log_path),
    ]


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
    raise TimeoutError(f"Timed out waiting for port {port} on {host}")


def _run_launcher(args: Args) -> None:
    if not args.critic_checkpoint:
        raise ValueError("--critic-checkpoint is required")
    if args.num_servers <= 0:
        raise ValueError("--num-servers must be > 0")
    if args.total_gpu_fraction <= 0:
        raise ValueError("--total-gpu-fraction must be > 0")
    if args.critic_gpu_fraction < 0:
        raise ValueError("--critic-gpu-fraction must be >= 0")
    if args.total_gpu_fraction <= args.critic_gpu_fraction:
        raise ValueError("--total-gpu-fraction must be greater than --critic-gpu-fraction")

    policy_total_fraction = args.total_gpu_fraction - args.critic_gpu_fraction
    per_server_fraction = policy_total_fraction / args.num_servers
    if per_server_fraction > 1.0:
        raise ValueError(
            "Per-worker XLA fraction is > 1.0. Reduce --total-gpu-fraction or increase --num-servers."
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (log_dir / f"run_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "launcher.log")

    logging.info("run_dir=%s", run_dir)
    logging.info(
        "Launching best-of-N stack with %s worker servers, per_server_fraction=%.6f, critic_fraction=%.6f",
        args.num_servers,
        per_server_fraction,
        args.critic_gpu_fraction,
    )

    metadata: dict[str, object] = {
        "num_servers": args.num_servers,
        "start_port": args.start_port,
        "gpu_id": args.gpu_id,
        "total_gpu_fraction": args.total_gpu_fraction,
        "critic_gpu_fraction": args.critic_gpu_fraction,
        "policy_total_gpu_fraction": policy_total_fraction,
        "per_server_fraction": per_server_fraction,
        "critic_port": args.critic_port,
        "critic_checkpoint": str(Path(args.critic_checkpoint).resolve()),
        "num_samples": args.num_samples,
        "xla_preallocate": args.xla_preallocate,
        "stagger_seconds": args.stagger_seconds,
        "workers": [],
    }

    script_path = Path(__file__).resolve()
    child_procs: list[tuple[subprocess.Popen[bytes], Path, int, str]] = []

    try:
        critic_log_path = run_dir / f"critic_port_{args.critic_port}.log"
        critic_cmd = _build_critic_cmd(args, script_path=script_path, critic_log_path=critic_log_path)
        metadata["critic"] = {
            "port": args.critic_port,
            "log_path": str(critic_log_path),
            "cmd": critic_cmd,
        }
        logging.info("critic port=%s log=%s", args.critic_port, critic_log_path)
        critic_log_handle = open(critic_log_path, "wb")
        critic_proc = subprocess.Popen(
            critic_cmd,
            env=_build_critic_env(base_env=os.environ, gpu_id=args.gpu_id, port=args.critic_port),
            stdout=critic_log_handle,
            stderr=subprocess.STDOUT,
        )
        child_procs.append((critic_proc, critic_log_path, args.critic_port, "critic"))
        _wait_for_port("127.0.0.1", args.critic_port, timeout_sec=60.0)

        for worker_index in range(args.num_servers):
            port = args.start_port + worker_index
            worker_log_path = run_dir / f"server_{worker_index:02d}_port_{port}.log"
            cmd = _build_policy_worker_cmd(
                args,
                script_path=script_path,
                port=port,
                worker_index=worker_index,
                worker_log_path=worker_log_path,
            )
            env = _build_policy_worker_env(
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
            child_procs.append((proc, worker_log_path, port, "worker"))
            _wait_for_port("127.0.0.1", port, timeout_sec=max(30.0, args.stagger_seconds + 10.0))
            time.sleep(max(0.0, args.stagger_seconds))

        with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        endpoints = [f"ws://127.0.0.1:{args.start_port + i}" for i in range(args.num_servers)]
        logging.info("endpoints=%s", ", ".join(endpoints))
        logging.info("Press Ctrl+C to stop all servers.")

        while True:
            all_done = True
            for proc, log_path, port, role in child_procs:
                rc = proc.poll()
                if rc is None:
                    all_done = False
                    continue
                logging.info("%s on port %s exited with code %s. log=%s", role, port, rc, log_path)
            if all_done:
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        logging.info("Stopping all servers...")
    finally:
        for proc, _, _, _ in child_procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2.0)
        for proc, _, _, _ in child_procs:
            if proc.poll() is None:
                proc.kill()


def main(args: Args) -> None:
    if args.critic_mode:
        _run_critic_mode(args)
        return
    if args.worker_mode:
        _run_worker_mode(args)
        return
    _run_launcher(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
