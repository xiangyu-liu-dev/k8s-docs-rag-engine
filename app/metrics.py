import time
import numpy as np


class MetricsCollector:
    def __init__(self):
        self._latencies: list[float] = []
        self._start_time = time.time()
        self._request_count = 0

    def record(self, latency_ms: float):
        self._latencies.append(latency_ms)
        self._request_count += 1

    def summary(self) -> dict:
        lats = np.array(self._latencies) if self._latencies else np.array([0])
        elapsed = time.time() - self._start_time
        return {
            "total_requests": self._request_count,
            "p50_ms": round(float(np.percentile(lats, 50)), 1),
            "p99_ms": round(float(np.percentile(lats, 99)), 1),
            "mean_ms": round(float(np.mean(lats)), 1),
            "throughput_qps": round(self._request_count / max(elapsed, 1), 2),
            "uptime_s": round(elapsed, 1),
        }

    def reset(self):
        self._latencies.clear()
        self._request_count = 0
        self._start_time = time.time()
