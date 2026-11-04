"""Measure cold load, warm request latency, CPU time and process RSS of the agent."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path

import psutil

from ingest_semantic_units import ROOT, read_jsonl
from search_semantic_units import DEFAULT_MODEL, METHODS
from semantic_agent import SemanticPipeline


CASES = ROOT / "Data/Curated/atlas_acceptance_cases.jsonl"
OUTPUT = ROOT / "Data/Curated/agent_system_metrics.json"


def percentile(values, proportion):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(proportion * len(ordered)) - 1)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(method="bm25", model=DEFAULT_MODEL, device="cpu", repetitions=3):
    if repetitions < 1:
        raise ValueError("repetitions doit être positif")
    cases = list(read_jsonl(CASES))
    process = psutil.Process()
    rss_start = process.memory_info().rss
    peak_rss = [rss_start]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        started = time.perf_counter()
        cpu_started = time.process_time()
        pipeline = SemanticPipeline(method=method, model_path=model, device=device)
        load_seconds = time.perf_counter() - started
        load_cpu_seconds = time.process_time() - cpu_started
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        started = time.perf_counter()
        cpu_started = time.process_time()
        first_response = pipeline.run(cases[0]["request"])
        first_request_wall_seconds = time.perf_counter() - started
        first_request_cpu_seconds = time.process_time() - cpu_started
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        rows = []
        for _ in range(repetitions):
            for case in cases:
                started = time.perf_counter()
                cpu_started = time.process_time()
                result = pipeline.run(case["request"])
                rows.append({"id": case["id"],
                             "wall_seconds": time.perf_counter() - started,
                             "cpu_seconds": time.process_time() - cpu_started,
                             "intent": result["intent"], "status": result["status"]})
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    wall = [row["wall_seconds"] for row in rows]
    cpu = [row["cpu_seconds"] for row in rows]
    return {
        "method": method, "device": device, "machine": platform.platform(),
        "python": platform.python_version(),
        "fixture_sha256": sha256(CASES),
        "model_weights_sha256": sha256(Path(model) / "model.safetensors") if method != "bm25" else None,
        "cases": len(cases), "repetitions": repetitions, "requests": len(rows),
        "cold_load_wall_seconds": round(load_seconds, 4),
        "cold_load_cpu_seconds": round(load_cpu_seconds, 4),
        "first_request_wall_seconds": round(first_request_wall_seconds, 4),
        "first_request_cpu_seconds": round(first_request_cpu_seconds, 4),
        "first_request_status": first_response["status"],
        "warm_wall_seconds": {"median": round(statistics.median(wall), 4),
                              "p95": round(percentile(wall, 0.95), 4),
                              "max": round(max(wall), 4)},
        "warm_cpu_seconds": {"median": round(statistics.median(cpu), 4),
                             "p95": round(percentile(cpu, 0.95), 4)},
        "process_rss_bytes": {"start": rss_start, "peak": peak_rss[0],
                              "increase": peak_rss[0] - rss_start},
        "inference_cost": {"external_api_calls": 0, "external_api_charge": 0,
                           "total_warm_cpu_seconds": round(sum(cpu), 4)},
        "method_note": "RSS échantillonné à 5 ms dans ce processus ; temps mur et CPU mesurés sur une petite fixture synthétique, sans concurrence contrôlée.",
        "per_request": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="bm25")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = benchmark(args.method, args.model, args.device, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_request"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
