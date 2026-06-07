import logging
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostRegressor
from .config_loader import get_main_config, get_model_config

logger = logging.getLogger("forecaster")

HORIZON_MAP = {
    "3 месяца": "model_3m",
    "6 месяцев": "model_6m",
    "12 месяцев": "model_12m"
}

class Forecaster:
    def __init__(self, horizon: str):
        horizon_key = HORIZON_MAP.get(horizon)
        if not horizon_key:
            logger.error(f"Получено некорректное значение горизонта прогнозирования")
            raise ValueError(f"Неправильное значение горизонта")
        model_cfg = get_model_config()
        main_cfg = get_main_config()

        models_dir = Path(main_cfg["paths"]["models_dir"])
        model_filename = model_cfg[horizon_key]["name"]
        self.model_path = models_dir / model_filename

        self.model = CatBoostRegressor()
        logger.info(f"Загрузка модели: {self.model_path}")
        try:
            self.model.load_model(str(self.model_path))
            self.cat_features = self.model.get_cat_feature_indices()
            logger.info(f"Модель успешно загружена")
        except FileNotFoundError:
            logger.error(f"Файл модели не найден: {self.model_path}")
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

    def predict(self, x_pred: pd.DataFrame) -> int:
        y_pred = self.model.predict(x_pred)
        y_pred = np.expm1(y_pred)
        return int(y_pred[0])

def get_forecaster(horizon: str) -> Forecaster:
    return Forecaster(horizon)