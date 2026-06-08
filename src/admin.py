import streamlit as st
import logging
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("admin")

if 'app_started' not in st.session_state:
    logger.info("Streamlit-приложение (Админ-панель) запущено")
    st.session_state.app_started = True

st.set_page_config(page_title="PredSys - Админ-панель", layout="wide")
st.title("PredSys - Панель мониторинга и управления")

tab_dashboard, tab_management = st.tabs(["Дэшборд", "Управление"])

with tab_dashboard:
    st.header("Дэшборд")
    st.info("Здесь будут отображаться графики из logs/, метрики качества моделей (MAPE, RMSE), потребление памяти и статус загруженных моделей.")
    # TODO: Добавить st.line_chart, st.metric и чтение данных из metrics.py

with tab_management:
    st.header("Управление моделями")
    st.info("Здесь будут кнопки 'Обучить новую модель', 'Откатить к предыдущей версии' и переключатель авто-переобучения.")
    # TODO: Добавить st.button, st.checkbox и вызовы API для управления обучением