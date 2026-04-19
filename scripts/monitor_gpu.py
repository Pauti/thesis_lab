#!/usr/bin/env python3
"""
GPU & System Monitoring Daemon.

Polls nvidia-smi and system stats every N seconds, writes to CSV.
Run alongside training to capture hardware metrics.

Usage:
    python scripts/monitor_gpu.py --output experiments/exp02_nnunet/gpu_monitor.csv --interval 30
    python scripts/monitor_gpu.py --output experiments/exp02_nnunet/gpu_monitor.csv --interval 30 --stop  # creates .stop file
"""

import argparse
import csv
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path


GPU_FIELDS = [
    "timestamp",
    "gpu_name",
    "gpu_temp_c",
    "gpu_utilization_pct",
    "gpu_mem_utilization_pct",
    "gpu_mem_used_mib",
    "gpu_mem_total_mib",
    "gpu_mem_free_mib",
    "gpu_power_draw_w",
    "gpu_power_limit_w",
    "gpu_clock_sm_mhz",
    "gpu_clock_mem_mhz",
    "gpu_fan_speed_pct",
    "gpu_pstate",
]

NVIDIA_QUERY = (
    "timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,memory.free,power.draw,power.limit,"
    "clocks.current.sm,clocks.current.memory,fan.speed,pstate"
)

SYSTEM_FIELDS = [
    "cpu_load_1min",
    "cpu_load_5min",
    "cpu_load_15min",
    "ram_used_mb",
    "ram_total_mb",
    "ram_pct",
    "disk_used_gb",
    "disk_free_gb",
    "disk_pct",
]

ALL_FIELDS = GPU_FIELDS + SYSTEM_FIELDS


def query_gpu():
    """Query nvidia-smi for GPU metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + NVIDIA_QUERY,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        values = [v.strip() for v in result.stdout.strip().split(",")]
        return dict(zip(GPU_FIELDS, values))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def query_system(data_path="/data/paul_daten"):
    """Query system metrics: CPU load, RAM, disk usage."""
    info = {}

    # CPU load averages
    try:
        load1, load5, load15 = os.getloadavg()
        info["cpu_load_1min"] = f"{load1:.2f}"
        info["cpu_load_5min"] = f"{load5:.2f}"
        info["cpu_load_15min"] = f"{load15:.2f}"
    except OSError:
        info["cpu_load_1min"] = info["cpu_load_5min"] = info["cpu_load_15min"] = ""

    # RAM usage
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        info["ram_total_mb"] = f"{total_kb // 1024}"
        info["ram_used_mb"] = f"{used_kb // 1024}"
        info["ram_pct"] = f"{100. * used_kb / total_kb:.1f}" if total_kb else ""
    except (FileNotFoundError, KeyError):
        info["ram_total_mb"] = info["ram_used_mb"] = info["ram_pct"] = ""

    # Disk usage for data volume
    try:
        st = os.statvfs(data_path)
        total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        used_gb = total_gb - free_gb
        info["disk_used_gb"] = f"{used_gb:.1f}"
        info["disk_free_gb"] = f"{free_gb:.1f}"
        info["disk_pct"] = f"{100. * used_gb / total_gb:.1f}" if total_gb else ""
    except OSError:
        info["disk_used_gb"] = info["disk_free_gb"] = info["disk_pct"] = ""

    return info


def monitor(output_path, interval=30, data_path="/data/paul_daten"):
    """Main monitoring loop. Stops when a .stop file is created."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stop_file = output_path.with_suffix(".stop")

    # Remove stale stop file
    if stop_file.exists():
        stop_file.unlink()

    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        if not file_exists:
            writer.writeheader()

        print(f"[GPU Monitor] Logging to {output_path} every {interval}s")
        print(f"[GPU Monitor] Stop with: touch {stop_file}")

        while not stop_file.exists():
            row = {}
            gpu = query_gpu()
            if gpu:
                row.update(gpu)
            sys_info = query_system(data_path)
            row.update(sys_info)

            writer.writerow(row)
            f.flush()

            time.sleep(interval)

    # Clean up stop file
    if stop_file.exists():
        stop_file.unlink()
    print(f"[GPU Monitor] Stopped. Log saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU & System Monitor")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval (seconds)")
    parser.add_argument("--data_path", default="/data/paul_daten", help="Disk path to monitor")
    args = parser.parse_args()

    monitor(args.output, args.interval, args.data_path)
