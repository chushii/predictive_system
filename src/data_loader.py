import logging
import os
import random
import pandas as pd
import numpy as np

from typing import Dict, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger("config_loader")

def prepare_input_data(input_dict: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Подготовка входных данных для модели...")
    x_pred = pd.DataFrame([input_dict])
    x_pred['closure_rate'] = x_pred['closed_issues'] / (x_pred['open_issues'] + x_pred['closed_issues'] + 1e-8)
    x_pred['closure_rate'] = round(x_pred['closure_rate'], 3)
    x_pred['downloads'] = np.log1p(x_pred['downloads'])
    x_pred = x_pred.drop(columns=['open_issues', 'closed_issues'])
    logger.info("Входные данные подготовлены")
    return x_pred


def prepare_datasets(cfg: Dict[str, Any]) -> None:
    logger.info("Загрузка датасетов...")

    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    data_dir = paths.get("data_dir", "data")
    cleared_dir = os.path.join(data_dir, "cleared")
    if os.path.exists(cleared_dir):
        required_files = ["x_train.csv", "y_train.csv", "x_val.csv", "y_val.csv", "x_test.csv", "y_test.csv"]
        all_exist = all(os.path.exists(os.path.join(cleared_dir, f)) for f in required_files)
        if all_exist:
            logger.info("Используются датасеты из папки cleared/")
            return

    random_seed = cfg.get("random_seed")
    if random_seed is None:
        random_seed = random.randint(0, 999999)
        logger.info(f"Сгенерирован случайный random_seed: {random_seed}")
    else:
        logger.info(f"Используется фиксированный random_seed: {random_seed}")

    trends_file = paths.get("trends_file", "component_trends.csv")
    metadata_file = paths.get("metadata_file", "components_metadata.csv")

    df_trends = pd.read_csv(trends_file, parse_dates=['date'])
    df_meta = pd.read_csv(metadata_file, parse_dates=['created_at'])
    cols = df_meta.columns.difference(df_trends.columns).tolist()
    df = df_trends.merge(df_meta[['component_id'] + cols], on='component_id', how='left')

    df['closure_rate'] = round(df['closed_issues'] / (df['open_issues'] + df['closed_issues'] + 1e-8), 3)

    upper_bound = df['dependents'].quantile(0.98)
    df = df[df['dependents'] <= upper_bound]

    log_cols = ['downloads', 'future_downloads_3m', 'future_downloads_6m', 'future_downloads_12m']
    df_log = np.log1p(df[log_cols])
    bounds = df_log.quantile(0.998)
    mask = (df_log <= bounds).all(axis=1)
    df = df[mask]

    df = df[df.groupby('component_id')['component_id'].transform('size') >= 24]

    target_cols = ['future_downloads_3m', 'future_downloads_6m', 'future_downloads_12m']
    feature_cols = [
        'type', 'language', 'category', 'license', 'author', 'ecosystem', 'status',
        'downloads', 'stars', 'contributors', 'dependencies', 'dependents', 'commits', 'releases', 'versions',
        'market_share', 'quality_score', 'documentation_score', 'community_score', 'maturity_score',
        'language_trend', 'category_trend', 'ecosystem_health', 'seasonality',
        'has_ci', 'has_examples', 'has_tests', 'has_tutorials', 'has_website',
        'open_issues', 'closed_issues', 'closure_rate'
    ]

    cols_to_keep = ['component_id', 'date'] + target_cols + feature_cols
    df = df[[col for col in cols_to_keep if col in df.columns]]

    drop_cols = ['component_id', 'date', 'open_issues', 'closed_issues'] + target_cols
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_cols]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=random_seed,
        shuffle=True
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval,
        test_size=data_cfg.get("val_size", 0.3),
        random_state=random_seed,
        shuffle=True
    )

    for df_x in [x_train, x_val, x_test]:
        if 'downloads' in df_x.columns:
            df_x['downloads'] = np.log1p(df_x['downloads'])

    for df_y in [y_train, y_val, y_test]:
        for col in target_cols:
            df_y[col] = np.log1p(df_y[col])

    cleared_dir = os.path.join(data_dir, "cleared")
    os.makedirs(cleared_dir, exist_ok=True)

    x_train.to_csv(os.path.join(cleared_dir, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(cleared_dir, "y_train.csv"), index=False)
    x_val.to_csv(os.path.join(cleared_dir, "x_val.csv"), index=False)
    y_val.to_csv(os.path.join(cleared_dir, "y_val.csv"), index=False)
    x_test.to_csv(os.path.join(cleared_dir, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(cleared_dir, "y_test.csv"), index=False)

    test_set = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_set.to_csv(os.path.join(data_dir, "test_set.csv"), index=False)

    logger.info("Датасеты успешно сохранены")