import logging
import pandas as pd
import numpy as np

from typing import Dict, Any

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