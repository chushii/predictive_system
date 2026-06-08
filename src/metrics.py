import os
import time
import pandas as pd
import psutil
import logging

logger = logging.getLogger("metrics")

HORIZON_MAP = {
    "3 месяца": "3m",
    "6 месяцев": "6m",
    "12 месяцев": "12m"
}

def get_model_status(horizon_display: str) -> str:
    horizon = HORIZON_MAP.get(horizon_display, "3m")
    info_dir = f"logs/catboost_{horizon}_info"
    model_path = f"models/catboost_{horizon}.cbm"
    error_file = os.path.join(info_dir, "learn_error.tsv")

    if not os.path.exists(info_dir):
        return "Отсутствует"

    if os.path.exists(error_file):
        mtime = os.path.getmtime(error_file)
        if time.time() - mtime < 30:
            return "Обучается"

    if os.path.exists(model_path):
        return "Готова"

    return "Незавершена"

def get_model_metrics(horizon_display: str) -> dict:
    horizon = HORIZON_MAP.get(horizon_display, "3m")
    info_dir = f"logs/catboost_{horizon}_info"

    error_file = os.path.join(info_dir, "test_error.tsv")
    if not os.path.exists(error_file):
        error_file = os.path.join(info_dir, "learn_error.tsv")

    if not os.path.exists(error_file):
        return {"MAPE": "N/A", "RMSE": "N/A", "Итераций": "N/A"}

    try:
        df = pd.read_csv(error_file, sep='\t')
        last_row = df.iloc[-1]

        mape_col = next((c for c in reversed(df.columns) if 'MAPE' in c), None)
        rmse_col = next((c for c in reversed(df.columns) if 'RMSE' in c), None)

        return {
            "MAPE": f"{last_row[mape_col]:.2f}%" if mape_col else "N/A",
            "RMSE": f"{last_row[rmse_col]:.4f}" if rmse_col else "N/A",
            "Итераций": int(last_row.get('iter', 0))
        }
    except Exception as e:
        logger.warning(f"Не удалось прочитать метрики для 'catboost_{horizon}': {e}")
        return {"MAPE": "Чтение...", "RMSE": "Чтение...", "Итераций": "N/A"}

def get_system_metrics() -> dict:
    mem = psutil.virtual_memory()
    return {
        "ram_used_gb": round(mem.used / (1024 ** 3), 2),
        "ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "ram_percent": mem.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }

def get_recent_logs(lines: int = 50) -> str:
    log_file = "logs/system.log"
    if not os.path.exists(log_file):
        return "Файл логов не найден"
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return "".join(f.readlines()[-lines:])
    except Exception as e:
        return f"Ошибка чтения логов: {e}"