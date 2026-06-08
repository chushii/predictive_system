import logging
import numpy as np
import pandas as pd
import os
import optuna

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

def build_model(horizon: str, auto_tune: bool = False) -> None:
    horizon_key = HORIZON_MAP.get(horizon)
    if not horizon_key:
        logger.error(f"Получено некорректное значение горизонта прогнозирования")
        raise ValueError(f"Неправильное значение горизонта")
    model_cfg = get_model_config()
    main_cfg = get_main_config()

    model_name = model_cfg[horizon_key]["name"]
    target_col = model_cfg[horizon_key]["target"]
    model_args = model_cfg[horizon_key].get("args", {}).copy()
    training_args = model_cfg[horizon_key].get("training", {})

    logger.info(f"Загрузка данных для обучения...")
    paths = main_cfg.get("paths", {})
    data_dir = os.path.join(paths.get("data_dir", "data"), "cleared")
    if os.path.exists(data_dir):
        required_files = ["x_train.csv", "y_train.csv", "x_val.csv", "y_val.csv"]
        all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
        if not all_exist:
            logger.error("Файлы датасетов не найдены")
            raise FileNotFoundError("Файлы датасетов не найдены")

    os.makedirs("models/", exist_ok=True)
    models_dir = paths.get("models_dir", "models")
    logs_dir = paths.get("logs_dir", "logs")

    x_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    x_val = pd.read_csv(os.path.join(data_dir, "x_val.csv"))
    y_train_full = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_val_full = pd.read_csv(os.path.join(data_dir, "y_val.csv"))
    y_train = y_train_full[target_col]
    y_val = y_val_full[target_col]

    cat_feats = x_train.select_dtypes(exclude=['number', 'datetime', 'bool']).columns.tolist()

    if auto_tune:
        logger.info("Запуск автоматического подбора параметров")

        def objective(trial):
            tune_args = model_args.copy()
            tune_args["iterations"] = trial.suggest_int("iterations", 500, 1500, log=True)
            tune_args["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
            tune_args["depth"] = trial.suggest_int("depth", 4, 10, log=True)

            trial_model = CatBoostRegressor(
                **tune_args, cat_features=cat_feats, verbose=False
            )
            trial_model.fit(x_train, y_train, eval_set=(x_val, y_val), **training_args)

            return trial_model.get_best_score()["validation"]["MAPE"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        model_args.update(best_params)
        logger.info(f"Автоподбор завершен. Лучшие параметры: {best_params}")

    model_stem = Path(model_name).stem
    new_model_path = os.path.join(models_dir, model_name)
    old_model_path = os.path.join(models_dir, f"{model_stem}_old.cbm")
    new_info_dir = os.path.join(logs_dir, f"{model_stem}_info")
    old_info_dir = os.path.join(logs_dir, f"{model_stem}_old_info")
    if os.path.exists(old_info_dir):
        logger.info(f"Удаление старой резервной копии: {old_info_dir}")
        os.remove(old_info_dir)
    if os.path.exists(new_info_dir):
        logger.info(f"Создание новой резервной копии: {new_info_dir} -> {old_info_dir}")
        os.rename(new_info_dir, old_info_dir)
    os.makedirs(new_info_dir, exist_ok=True)

    model = CatBoostRegressor(
        **model_args, cat_features=cat_feats, train_dir=str(new_info_dir)
    )

    model.fit(
        x_train, y_train, eval_set=(x_val, y_val), verbose=50, **training_args
    )

    if os.path.exists(old_model_path):
        logger.info(f"Удаление старой резервной копии: {old_model_path}")
        os.remove(old_model_path)
    if os.path.exists(new_model_path):
        logger.info(f"Создание новой резервной копии: {new_model_path} -> {old_model_path}")
        os.rename(new_model_path, old_model_path)

    model.save_model(str(new_model_path))
    logger.info(f"Модель сохранена в {new_model_path}")