import logging
import json

from pathlib import Path

class MetricsCollector:
    def __init__(self, log_path: str = "logs/metrics.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = self._load()
        self._setup_logging()

    def _load(self):
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                return json.load(f)
        return {"requests": 0, "errors": 0, "total_latency_ms": 0.0}

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def record_prediction(self, latency_ms: float, success: bool):
        self.metrics["requests"] += 1
        if not success:
            self.metrics["errors"] += 1
        self.metrics["total_latency_ms"] += latency_ms
        self._save()

    def get_summary(self):
        req = self.metrics["requests"]
        return {
            "total_requests": req,
            "errors": self.metrics["errors"],
            "avg_latency_ms": self.metrics["total_latency_ms"] / max(req, 1),
            "error_rate": self.metrics["errors"] / max(req, 1)
        }

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.log_path.parent / "system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("predictive_system")