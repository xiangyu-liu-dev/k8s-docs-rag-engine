import csv
import json
import os

OUTPUT_DIR = "data/bench"
CONCURRENCY_LEVELS = [1, 5, 10, 20]


def main():
    baseline = json.load(open(f"{OUTPUT_DIR}/baseline.json"))
    direct = json.load(open(f"{OUTPUT_DIR}/vllm_direct.json"))

    rows = []
    rows.append(
        {
            "setup": "Baseline (transformers)",
            "concurrent": 1,
            "p50_ms": baseline["p50_ms"],
            "p99_ms": baseline["p99_ms"],
            "mean_ms": baseline["mean_ms"],
            "throughput_qps": baseline["throughput_qps"],
            "gpu_mem_mb": baseline.get("mem_peak_mb", ""),
        }
    )
    rows.append(
        {
            "setup": "vLLM (sequential)",
            "concurrent": 1,
            "p50_ms": direct["p50_ms"],
            "p99_ms": direct["p99_ms"],
            "mean_ms": direct["mean_ms"],
            "throughput_qps": direct["throughput_qps"],
            "gpu_mem_mb": direct["gpu_mem_mb"],
        }
    )

    for n in CONCURRENCY_LEVELS:
        path = f"{OUTPUT_DIR}/vllm_n{n}.json"
        if not os.path.exists(path):
            print(f"Skipping concurrent={n}: {path} not found")
            continue
        r = json.load(open(path))
        rows.append(
            {
                "setup": f"vLLM (concurrent={n})",
                "concurrent": n,
                "p50_ms": r["p50_ms"],
                "p99_ms": r["p99_ms"],
                "mean_ms": r["mean_ms"],
                "throughput_qps": r["throughput_qps"],
                "gpu_mem_mb": r["gpu_mem_mb"],
            }
        )

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print table
    print(f"\n{'Setup':<25} {'p50':>8} {'p99':>8} {'QPS':>8} {'GPU MB':>8}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['setup']:<25} {row['p50_ms']:>7.0f}ms {row['p99_ms']:>7.0f}ms {row['throughput_qps']:>7.2f} {str(row['gpu_mem_mb']):>7}"
        )

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
