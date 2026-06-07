import logging
import os
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_loader import prepare_input_data
from forecaster import get_forecaster

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive System API")
logger.info("API успешно инициализирован")

class PredictionRequest(BaseModel):
    horizon: Literal["3m", "6m", "12m"] = Field(..., description="Горизонт прогнозирования")

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

@app.get("/health")
async def health_check():
    return {"status": "ok"}