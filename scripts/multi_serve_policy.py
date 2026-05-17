"""Launch multiple OpenPI websocket policy servers on one GPU.

This script starts N independent `scripts.serve_policy` processes, one per port.
Each process gets a per-process XLA memory fraction so multiple servers can
coexist on the same GPU.

Examples
--------
uv run scripts/multi_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.90 \
  --total-ppo-gpu-fraction 0.05 \
  -- policy:checkpoint \
  --policy.config pi05_lehome_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_robot_finetune/all_data_3_epoch/20000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple OpenPI policy websocket servers concurrently."
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        required=True,
        help="Number of serve_policy processes to launch.",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="First websocket port. Servers use consecutive ports.",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="GPU id to expose via CUDA_VISIBLE_DEVICES (same GPU for all servers).",
    )
    parser.add_argument(
        "--total-gpu-fraction",
        type=float,
        default=1.0,
        help=(
            "Total XLA GPU fraction budget shared across all servers. "
            "Each server gets total_gpu_fraction / num_servers."
        ),
    )
    parser.add_argument(
        "--total-ppo-gpu-fraction",
        type=float,
        default=0.0,
        help=(
            "Total PyTorch PPO-head GPU fraction budget shared across all servers. "
            "Each server gets total_ppo_gpu_fraction / num_servers. "
            "Use 0 to leave PyTorch unbounded or when PPO heads run on CPU."
        ),
    )
    parser.add_argument(
        "--xla-preallocate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set XLA_PYTHON_CLIENT_PREALLOCATE. "
            "Use --no-xla-preallocate to reduce strict preallocation."
        ),
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=2.0,
        help="Delay between launches to reduce startup spikes.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to spawn child servers.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/multi_serve_policy",
        help="Directory for per-server logs and metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without launching processes.",
    )
    parser.add_argument(
        "serve_policy_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to scripts.serve_policy. "
            "Prefix with '--' so argparse stops parsing."
        ),
    )
    return parser.parse_args()


def _normalize_forwarded_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise ValueError(
            "No forwarded args provided. Pass serve_policy args after '--'."
        )
    for i, token in enumerate(args):
        if token == "--port" or token.startswith("--port="):
            raise ValueError(
                f"Do not pass --port in forwarded args (found at index {i}). "
                "Ports are assigned automatically by multi_serve_policy."
            )
    return args


def _build_env(
    base_env: dict[str, str],
    gpu_id: str,
    per_server_fraction: float,
    per_server_ppo_fraction: float,
    xla_preallocate: bool,
    server_index: int,
    port: int,
) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{per_server_fraction:.6f}"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if xla_preallocate else "false"
    env["OPENPI_PPO_GPU_MEM_FRACTION"] = f"{per_server_ppo_fraction:.6f}"
    env["OPENPI_MULTI_SERVER_INDEX"] = str(server_index)
    env["OPENPI_MULTI_SERVER_PORT"] = str(port)
    return env


def main() -> None:
    args = _parse_args()
    forward_args = _normalize_forwarded_args(args.serve_policy_args)

    if args.num_servers <= 0:
        raise ValueError("--num-servers must be > 0.")
    if args.total_gpu_fraction <= 0:
        raise ValueError("--total-gpu-fraction must be > 0.")
    if args.total_ppo_gpu_fraction < 0:
        raise ValueError("--total-ppo-gpu-fraction must be >= 0.")
    if args.total_gpu_fraction + args.total_ppo_gpu_fraction > 1.0:
        raise ValueError(
            "--total-gpu-fraction + --total-ppo-gpu-fraction must be <= 1.0 "
            "when VLA and PPO heads share one GPU."
        )

    per_server_fraction = args.total_gpu_fraction / args.num_servers
    per_server_ppo_fraction = args.total_ppo_gpu_fraction / args.num_servers
    if per_server_fraction > 1.0:
        raise ValueError(
            "Per-server fraction is > 1.0. Reduce --total-gpu-fraction or increase --num-servers."
        )
    if per_server_ppo_fraction > 1.0:
        raise ValueError(
            "Per-server PPO fraction is > 1.0. Reduce --total-ppo-gpu-fraction or increase --num-servers."
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "num_servers": args.num_servers,
        "start_port": args.start_port,
        "gpu_id": args.gpu_id,
        "total_gpu_fraction": args.total_gpu_fraction,
        "total_ppo_gpu_fraction": args.total_ppo_gpu_fraction,
        "per_server_fraction": per_server_fraction,
        "per_server_ppo_fraction": per_server_ppo_fraction,
        "xla_preallocate": args.xla_preallocate,
        "stagger_seconds": args.stagger_seconds,
        "serve_policy_args": forward_args,
        "servers": [],
    }

    procs: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    print(f"[multi_serve_policy] run_dir={run_dir}")
    print(
        "[multi_serve_policy] launching "
        f"{args.num_servers} servers, "
        f"per_server_fraction={per_server_fraction:.6f}, "
        f"per_server_ppo_fraction={per_server_ppo_fraction:.6f}"
    )

    for i in range(args.num_servers):
        port = args.start_port + i
        log_path = run_dir / f"server_{i:02d}_port_{port}.log"
        cmd = [
            args.python_exe,
            "-m",
            "scripts.serve_policy",
            "--port",
            str(port),
            *forward_args,
        ]
        env = _build_env(
            base_env=os.environ,
            gpu_id=args.gpu_id,
            per_server_fraction=per_server_fraction,
            per_server_ppo_fraction=per_server_ppo_fraction,
            xla_preallocate=args.xla_preallocate,
            server_index=i,
            port=port,
        )

        metadata["servers"].append(
            {
                "index": i,
                "port": port,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )

        print(f"[multi_serve_policy] server[{i}] port={port} log={log_path}")
        if args.dry_run:
            continue

        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_path, port))
        time.sleep(max(0.0, args.stagger_seconds))

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.dry_run:
        print("[multi_serve_policy] dry-run complete.")
        return

    endpoints = [f"ws://127.0.0.1:{args.start_port + i}" for i in range(args.num_servers)]
    print("[multi_serve_policy] endpoints:")
    print("  " + ", ".join(endpoints))
    print("[multi_serve_policy] press Ctrl+C to stop all servers.")

    try:
        # Keep parent alive and monitor child exits.
        while True:
            all_done = True
            for proc, log_path, port in procs:
                rc = proc.poll()
                if rc is None:
                    all_done = False
                    continue
                # Child exited; report once and keep scanning.
                print(
                    f"[multi_serve_policy] server on port {port} exited with code {rc}. "
                    f"log={log_path}"
                )
            if all_done:
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[multi_serve_policy] stopping all servers...")
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2.0)
        for proc, _, _ in procs:
            if proc.poll() is None:
                proc.kill()
    finally:
        # Best-effort cleanup on POSIX process groups is intentionally omitted to keep script simple.
        pass


if __name__ == "__main__":
    main()
