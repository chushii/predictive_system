import os
import sys
import time
import yaml
import logging
import webbrowser
import subprocess
import threading

from pathlib import Path
from src.data_loader import prepare_datasets
from src.forecaster import build_model, HORIZON_MAP

os.makedirs("logs/", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("launcher")

def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Конфигурационный файл не найден: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_config():
    model_config_path = Path(__file__).parent / "config" / "models.yaml"
    if not model_config_path.exists():
        logger.error(f"Файл настроек моделей не найден: {model_config_path}")
        sys.exit(1)
    with open(model_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def stream_output(process, name):
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('cp1251', errors='replace').strip()
        if decoded_line:
            logger.info(f"{decoded_line}")
    process.stdout.close()

def main():
    logger.info("Загрузка системы...")
    config = load_config()

    api_host = config['system']['api']['host']
    api_port = config['system']['api']['port']
    ui_host = config['system']['ui']['host']
    ui_port = config['system']['ui']['port']
    admin_host = config['system']['admin']['host']
    admin_port = config['system']['admin']['port']

    os.environ["PRED_API_HOST"] = str(api_host)
    os.environ["PRED_API_PORT"] = str(api_port)
    os.environ["PRED_UI_HOST"] = str(ui_host)
    os.environ["PRED_UI_PORT"] = str(ui_port)
    os.environ["PRED_ADMIN_HOST"] = str(admin_host)
    os.environ["PRED_ADMIN_PORT"] = str(admin_port)

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    try:
        prepare_datasets(config)
    except Exception as e:
        logger.error(f"Критическая ошибка при подготовке датасетов: {str(e)}", exc_info=True)
        raise RuntimeError("Не удалось подготовить датасеты")

    logger.info("Проверка наличия обученных моделей...")
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    models_cfg = load_model_config()

    for horizon_name, horizon_key in HORIZON_MAP.items():
        model_filename = models_cfg[horizon_key]["name"]
        model_path = os.path.join(models_dir, model_filename)

        if not os.path.exists(model_path):
            logger.warning(
                f"Модель не найдена: {model_path}. Запуск процесса обучения...")
            try:
                build_model(horizon_name)
            except Exception as e:
                logger.error(f"Критическая ошибка при обучении модели: {str(e)}", exc_info=True)
                raise RuntimeError(
                    f"Система не может быть запущена: не удалось обучить модель '{model_filename}'")
        else:
            logger.info(f"Модель '{model_path}' загружена")

    processes = []
    try:
        api_cmd = [
            sys.executable, "-m", "uvicorn", "src.server:app",
            "--host", str(api_host), "--port", str(api_port), "--log-level", "warning"
        ]
        api_process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            cwd=Path(__file__).parent
        )
        processes.append(("API", api_process))
        threading.Thread(target=stream_output, args=(api_process, "API"), daemon=True).start()
        logger.info(f"API: http://{api_host}:{api_port}")

        ui_cmd = [
            sys.executable, "-m", "streamlit", "run", "src/app.py",
            "--server.address", str(ui_host),
            "--server.port", str(ui_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        ui_process = subprocess.Popen(
            ui_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            cwd=Path(__file__).parent
        )
        processes.append(("UI", ui_process))
        threading.Thread(target=stream_output, args=(ui_process, "UI"), daemon=True).start()
        logger.info(f"UI: http://{ui_host}:{ui_port}")

        admin_cmd = [
            sys.executable, "-m", "streamlit", "run", "src/admin.py",
            "--server.address", str(admin_host),
            "--server.port", str(admin_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        admin_process = subprocess.Popen(
            admin_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            cwd=Path(__file__).parent
        )
        processes.append(("Admin UI", admin_process))
        threading.Thread(target=stream_output, args=(admin_process, "Admin UI"), daemon=True).start()
        logger.info(f"Admin UI: http://{admin_host}:{admin_port}")

        logger.info("Ожидание инициализации сервисов...")
        time.sleep(5)
        logger.info("Система запущена. Нажмите Ctrl+C для остановки")
        webbrowser.open(f"http://{ui_host}:{ui_port}")

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    logger.error(f"Процесс {name} неожиданно завершился с кодом {proc.poll()}")
                    raise RuntimeError(f"Процесс {name} упал")

    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки. Завершение процессов...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                logger.info(f"Остановка {name} (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        logger.info("Система завершила работу")

if __name__ == "__main__":
    main()