"""Launch multiple spline-conditioned OpenPI websocket servers on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple LeHome spline-conditioned OpenPI websocket servers.")
    parser.add_argument("--num-servers", type=int, required=True, help="Number of VLA websocket servers to launch.")
    parser.add_argument("--start-port", type=int, default=8000, help="First VLA websocket port.")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU id exposed via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--total-gpu-fraction", type=float, default=1.0, help="Total XLA GPU fraction budget shared across servers.")
    parser.add_argument("--xla-preallocate", action=argparse.BooleanOptionalAction, default=True, help="Set XLA_PYTHON_CLIENT_PREALLOCATE.")
    parser.add_argument("--stagger-seconds", type=float, default=2.0, help="Delay between launches.")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable for child servers.")
    parser.add_argument("--log-dir", type=str, default="logs/multi_serve_lehome_spline_policy", help="Directory for per-server logs.")
    parser.add_argument("--spline-base-ws-url", type=str, required=True, help="Base spline websocket URL, e.g. ws://127.0.0.1")
    parser.add_argument("--spline-start-port", type=int, required=True, help="First spline websocket port.")
    parser.add_argument("--spline-port-step", type=int, default=1, help="Increment between spline ports.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching.")
    parser.add_argument("serve_policy_args", nargs=argparse.REMAINDER, help="Arguments forwarded to scripts.serve_lehome_spline_policy. Prefix with '--'.")
    return parser.parse_args()


def _normalize_forwarded_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise ValueError("No forwarded args provided. Pass serve_lehome_spline_policy args after '--'.")
    for token in args:
        if token == "--port" or token.startswith("--port=") or token == "--spline-server-url" or token.startswith("--spline-server-url="):
            raise ValueError("Do not pass --port or --spline-server-url in forwarded args; multi-serve assigns them.")
    return args


def main() -> None:
    args = _parse_args()
    forward_args = _normalize_forwarded_args(args.serve_policy_args)
    if args.num_servers <= 0:
        raise ValueError("--num-servers must be > 0.")
    if args.total_gpu_fraction <= 0:
        raise ValueError("--total-gpu-fraction must be > 0.")

    per_server_fraction = args.total_gpu_fraction / args.num_servers
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "num_servers": args.num_servers,
        "start_port": args.start_port,
        "gpu_id": args.gpu_id,
        "total_gpu_fraction": args.total_gpu_fraction,
        "per_server_fraction": per_server_fraction,
        "spline_base_ws_url": args.spline_base_ws_url,
        "spline_start_port": args.spline_start_port,
        "spline_port_step": args.spline_port_step,
        "serve_policy_args": forward_args,
        "servers": [],
    }

    processes: list[tuple[subprocess.Popen[bytes], Path, int]] = []
    print(f"[multi_serve_lehome_spline_policy] run_dir={run_dir}")
    for index in range(args.num_servers):
        port = args.start_port + index
        spline_port = args.spline_start_port + index * args.spline_port_step
        spline_url = f"{args.spline_base_ws_url.rstrip('/')}:{spline_port}"
        log_path = run_dir / f"server_{index:02d}_port_{port}.log"
        cmd = [
            args.python_exe,
            "-m",
            "scripts.serve_lehome_spline_policy",
            "--port",
            str(port),
            "--spline-server-url",
            spline_url,
            *forward_args,
        ]
        metadata["servers"].append(
            {
                "index": index,
                "port": port,
                "spline_port": spline_port,
                "spline_url": spline_url,
                "log_path": str(log_path),
                "cmd": cmd,
            }
        )
        print(f"[multi_serve_lehome_spline_policy] server[{index}] port={port} spline={spline_url} log={log_path}")
        if args.dry_run:
            continue

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{per_server_fraction:.6f}"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.xla_preallocate else "false"
        env["OPENPI_MULTI_SERVER_INDEX"] = str(index)
        env["OPENPI_MULTI_SERVER_PORT"] = str(port)
        env["LEHOME_SPLINE_SERVER_URL"] = spline_url
        log_handle = open(log_path, "wb")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_path, port))
        time.sleep(max(0.0, args.stagger_seconds))

    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        return

    endpoints = [f"ws://127.0.0.1:{args.start_port + index}" for index in range(args.num_servers)]
    print("[multi_serve_lehome_spline_policy] endpoints:")
    for endpoint in endpoints:
        print(f"  {endpoint}")

    try:
        exit_code = 0
        for process, _, _ in processes:
            code = process.wait()
            if code != 0 and exit_code == 0:
                exit_code = code
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        print("[multi_serve_lehome_spline_policy] interrupt received, terminating child processes...")
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


if __name__ == "__main__":
    main()
