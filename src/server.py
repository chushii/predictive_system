import logging
import os

from typing import Literal
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
from .data_loader import prepare_input_data
from .forecaster import get_forecaster, build_model
from .config_loader import get_main_config
from .metrics import get_model_status, get_model_metrics, get_system_metrics, get_recent_logs

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("server")

config = get_main_config()

app = FastAPI(
    title="PredSys",
    version="1.0",
    description="API для прогнозирования тенденций использования программных компонент"
)
logger.info("API успешно инициализирован")

class PredictionRequest(BaseModel):
    horizon: Literal["3 месяца", "6 месяцев", "12 месяцев"] = Field(..., description="Горизонт прогнозирования")

    type: str = Field(..., description="Тип компонента")
    language: str = Field(..., description="Язык программирования")
    category: str = Field(..., description="Категория")
    license: str = Field(..., description="Лицензия")
    author: str = Field(..., description="Автор/организация")
    ecosystem: str = Field(..., description="Экосистема/репозиторий")
    status: str = Field(..., description="Статус компонента")

    downloads: int = Field(..., description="Текущее количество скачиваний", ge=0)
    stars: int = Field(..., description="Количество звезд", ge=0)
    contributors: int = Field(..., description="Количество контрибьюторов", ge=0)
    dependencies: int = Field(..., description="Количество зависимостей", ge=0)
    dependents: int = Field(..., description="Количество зависимых проектов", ge=0)
    commits: int = Field(..., description="Количество коммитов", ge=0)
    releases: int = Field(..., description="Количество релизов", ge=0)
    versions: int = Field(..., description="Количество версий", ge=0)
    open_issues: int = Field(..., description="Количество открытых задач", ge=0)
    closed_issues: int = Field(..., description="Количество решённых задач", ge=0)

    market_share: float = Field(..., description="Доля рынка", ge=0.0, le=1.0)
    quality_score: float = Field(..., description="Оценка качества кода", ge=0.0, le=1.0)
    documentation_score: float = Field(..., description="Оценка документации", ge=0.0, le=1.0)
    community_score: float = Field(..., description="Оценка активности сообщества", ge=0.0, le=1.0)
    maturity_score: float = Field(..., description="Оценка зрелости проекта", ge=0.0, le=1.0)
    language_trend: float = Field(..., description="Тренд популярности языка")
    category_trend: float = Field(..., description="Тренд популярности категории")
    ecosystem_health: float = Field(..., description="Здоровье экосистемы")
    seasonality: float = Field(..., description="Коэффициент сезонности")

    has_ci: int = Field(..., description="Наличие CI", ge=0, le=1)
    has_examples: int = Field(..., description="Наличие примеров", ge=0, le=1)
    has_tests: int = Field(..., description="Наличие тестов", ge=0, le=1)
    has_tutorials: int = Field(..., description="Наличие туториалов", ge=0, le=1)
    has_website: int = Field(..., description="Наличие веб-сайта", ge=0, le=1)

class PredictionResponse(BaseModel):
    horizon: str
    predicted_downloads: int
    status: str

class RetrainRequest(BaseModel):
    horizon: Literal["3 месяца", "6 месяцев", "12 месяцев"] = Field(..., description="Горизонт прогнозирования для реобучения модели")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    logger.info(f"Получен запрос на прогноз. Горизонт: {request.horizon}")
    try:
        input_dict = request.model_dump()
        horizon = input_dict.pop("horizon")
        x_pred = prepare_input_data(input_dict)

        forecaster = get_forecaster(horizon)
        y_pred = forecaster.predict(x_pred)
        logger.info("Прогноз успешно сформирован")

        return PredictionResponse(
            horizon=horizon,
            predicted_downloads=y_pred,
            status="success"
        )

    except Exception as e:
        logger.error(f"Ошибка при формировании прогноза: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain", status_code=202)
async def retrain_endpoint(request: RetrainRequest, background_tasks: BackgroundTasks):
    logger.info(f"Получен запрос на реобучение модели. Горизонт: {request.horizon}")
    try:
        background_tasks.add_task(build_model, request.horizon)
        return {
            "status": "success",
            "message": f"Процесс реобучения модели запущен в фоновом режиме"
        }
    except ValueError as ve:
        logger.error(f"Ошибка валидации горизонта: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка при инициализации переобучения: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось запустить процесс переобучения")

def get_file_date(filepath: str) -> str:
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    return None

@app.get("/api/models/status")
async def get_models_status():
    horizons_map = {"3 месяца": "3m", "6 месяцев": "6m", "12 месяцев": "12m"}
    status = {}

    for display_name, suffix in horizons_map.items():
        main_path = f"models/catboost_{suffix}.cbm"
        backup_path = f"models/catboost_{suffix}_old.cbm"

        status[display_name] = {
            "main_date": get_file_date(main_path),
            "backup_date": get_file_date(backup_path)
        }
    return status

@app.post("/api/models/rollback")
async def rollback_model(horizon: str):
    logger.info(f"Получен запрос на откат модели. Горизонт: {horizon}")
    horizons_map = {"3 месяца": "3m", "6 месяцев": "6m", "12 месяцев": "12m"}
    if horizon not in horizons_map:
        logger.error(f"Неверное значение горизонта")
        raise HTTPException(status_code=400, detail="Неверный горизонт")

    suffix = horizons_map[horizon]
    main_path = f"models/catboost_{suffix}.cbm"
    backup_path = f"models/catboost_{suffix}_old.cbm"
    temp_path = f"models/catboost_{suffix}_temp.cbm"

    if not os.path.exists(backup_path):
        raise HTTPException(status_code=404, detail="Резервная копия не найдена")

    try:
        os.rename(main_path, temp_path)
        os.rename(backup_path, main_path)
        os.rename(temp_path, backup_path)
        logger.info(f"Выполнен успешный откат модели {main_path}")
        return {"status": "success", "message": "Откат выполнен успешно"}
    except Exception as e:
        logger.error(f"Ошибка при откате модели {main_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Не удалось выполнить откат")

@app.get("/api/metrics")
async def get_metrics():
    models_info = {}
    for horizon_display in ["3 месяца", "6 месяцев", "12 месяцев"]:
        models_info[horizon_display] = {
            "status": get_model_status(horizon_display),
            "metrics": get_model_metrics(horizon_display)
        }
    return {
        "models": models_info,
        "system": get_system_metrics(),
        "logs": get_recent_logs(30)
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}