import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).parent.parent / "config"

def _load_yaml(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

@lru_cache(maxsize=1)
def get_main_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "config.yaml")

@lru_cache(maxsize=1)
def get_model_config() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "models.yaml")

def reload_configs():
    get_main_config.cache_clear()
    get_model_config.cache_clear()