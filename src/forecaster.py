import logging
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

class Forecaster:
    def __init__(self, model_path: str):
        self.model = CatBoostRegressor()
        logger.info(f"Загрузка модели: {model_path}")
        try:
            self.model.load_model(model_path)
            self.cat_features = self.model.get_cat_feature_indices()
            logger.info(f"Модель успешно загружена")
        except FileNotFoundError:
            logger.error(f"Файл модели не найден")
            raise FileNotFoundError(f"Модель не найдена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

    def predict(self, x_pred: pd.DataFrame) -> int:
        y_pred = self.model.predict(x_pred)
        y_pred = np.expm1(y_pred)
        return int(y_pred[0])

MODEL_PATHS = {
    "3m": "models/catboost_3m.cbm",
    "6m": "models/catboost_6m.cbm",
    "12m": "models/catboost_12m.cbm"
}

def get_forecaster(horizon: str) -> Forecaster:
    path = MODEL_PATHS.get(horizon, MODEL_PATHS["3m"])
    logger.info(f"Выбрана модель для горизонта {horizon}: {path}")
    return Forecaster(path)