#!/usr/bin/env python3
import torch
import torch.multiprocessing as mp
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse

# Event to signal shutdown
STOP = mp.Event()

def _signal_handler(signum, frame):
    print(f"[MASTER] Caught signal {signum}, shutting down…", file=sys.stderr)
    STOP.set()

def worker_loop(gpu_id: int, end_time: datetime):
    torch.cuda.set_device(gpu_id)
    # adjust size to taste so each process is busy
    A = torch.randn((8000, 8000), device=gpu_id)
    B = torch.randn((8000, 8000), device=gpu_id)

    print(f"[GPU{gpu_id}] Starting; will stop at {end_time.isoformat()}")
    while not STOP.is_set():
        if datetime.now(end_time.tzinfo) >= end_time:
            break
        C = A @ B
        # wait for kernel to finish
        torch.cuda.synchronize()
    print(f"[GPU{gpu_id}] Exiting at {datetime.now(end_time.tzinfo).isoformat()}")

def main(end_time_str: str, tz_name: str):
    # parse end time + timezone
    tz = ZoneInfo(tz_name)
    end_time = datetime.fromisoformat(end_time_str).replace(tzinfo=tz)
    print(f"[MASTER] PID {mp.current_process().pid}: will stop at {end_time.isoformat()}")

    # handle Ctrl-C, SIGTERM
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # how many GPUs?
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("[MASTER] No GPUs detected; exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"[MASTER] Detected {ngpus} GPU(s). Spawning one worker per GPU.")

    mp.set_start_method('spawn', force=True)
    procs = []
    for gpu in range(ngpus):
        p = mp.Process(target=worker_loop, args=(gpu, end_time), daemon=True)
        p.start()
        procs.append(p)

    # master waits until deadline or signal
    try:
        while not STOP.is_set() and datetime.now(tz) < end_time:
            STOP.wait(timeout=5.0)
    finally:
        STOP.set()
        for p in procs:
            p.join(timeout=10.0)
        print("[MASTER] All workers have exited. Goodbye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully saturate all GPUs until a given absolute time"
    )
    parser.add_argument(
        "--end-time", "-e", required=True,
        help="Absolute stop time in ISO format, e.g. 2025-05-07T10:00:00"
    )
    parser.add_argument(
        "--tz", "-z", default="America/New_York",
        help="IANA timezone name for the end time (default: America/New_York)"
    )
    args = parser.parse_args()
    main(args.end_time, args.tz)