from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locust load tests and save p95/p99 latency report.")
    parser.add_argument("--sales-host", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--california-host", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--spawn-rate", type=int, default=5)
    parser.add_argument("--duration", type=str, default="30s")
    parser.add_argument("--output-dir", type=str, default="reports/loadtest")
    parser.add_argument("--max-p95-ms", type=float, default=None)
    return parser.parse_args()


def _extract_percentile(row: pd.Series, keys: list[str]) -> float:
    for key in keys:
        if key in row.index:
            return float(row[key])
    return 0.0


def _run_locust(
    *,
    locustfile: Path,
    host: str,
    users: int,
    spawn_rate: int,
    duration: str,
    csv_prefix: Path,
) -> dict[str, float | int | str]:
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        "--host",
        host,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        duration,
        "--csv",
        str(csv_prefix),
        "--only-summary",
    ]

    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Locust failed for {locustfile.name}: {completed.stderr}")

    stats_path = Path(f"{csv_prefix}_stats.csv")
    if not stats_path.exists():
        raise FileNotFoundError(f"Locust stats CSV was not generated: {stats_path}")

    stats = pd.read_csv(stats_path)
    aggregated = stats.loc[stats["Name"] == "Aggregated"]
    if aggregated.empty:
        raise RuntimeError(f"Aggregated row not found in {stats_path}")

    row = aggregated.iloc[0]

    return {
        "requests": int(row.get("Request Count", 0)),
        "failures": int(row.get("Failure Count", 0)),
        "median_ms": _extract_percentile(row, ["Median Response Time", "Median Response Time (ms)"]),
        "p95_ms": _extract_percentile(row, ["95%", "95%ile", "95th Percentile"]),
        "p99_ms": _extract_percentile(row, ["99%", "99%ile", "99th Percentile"]),
        "avg_ms": _extract_percentile(row, ["Average Response Time", "Average Response Time (ms)"]),
        "rps": float(row.get("Requests/s", 0.0)),
        "host": host,
    }


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sales_stats = _run_locust(
        locustfile=REPO_ROOT / "scripts" / "loadtest" / "locustfile_sales.py",
        host=args.sales_host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        csv_prefix=output_dir / "sales",
    )

    california_stats = _run_locust(
        locustfile=REPO_ROOT / "scripts" / "loadtest" / "locustfile_california.py",
        host=args.california_host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        csv_prefix=output_dir / "california",
    )

    summary = {
        "sales": sales_stats,
        "california": california_stats,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.max_p95_ms is not None:
        worst_p95 = max(float(sales_stats["p95_ms"]), float(california_stats["p95_ms"]))
        if worst_p95 > args.max_p95_ms:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
