import streamlit as st
import requests
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

API_HOST = os.getenv("PRED_API_HOST", "127.0.0.1")
API_PORT = os.getenv("PRED_API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

tab_dashboard, tab_management = st.tabs(["Дэшборд", "Управление"])
HORIZONS = ["3 месяца", "6 месяцев", "12 месяцев"]

with tab_dashboard:
    st.header("Дэшборд")
    if st.button("Обновить данные"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/api/metrics", timeout=5)
        response.raise_for_status()
        data = response.json()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Использование RAM",
                      f"{data['system']['ram_used_gb']} / {data['system']['ram_total_gb']} GB")
        with col2:
            st.metric("Загрузка CPU",
                      f"{data['system']['cpu_percent']}%")

        st.subheader("Статус и метрики моделей")
        cols = st.columns(3)
        for i, horizon in enumerate(HORIZONS):
            with cols[i]:
                st.write(f"**Горизонт: {horizon}**")
                model_data = data['models'][horizon]
                status = model_data['status']

                if status == "Готова":
                    status_color = "green"
                elif status == "Обучается":
                    status_color = "orange"
                else:
                    status_color = "red"

                st.markdown(f"Статус: <span style='color:{status_color}; font-weight:bold;'>{status}</span>",
                            unsafe_allow_html=True)

                metrics = model_data['metrics']
                st.write(f"MAPE: {metrics.get('MAPE', 'N/A')}")
                st.write(f"RMSE: {metrics.get('RMSE', 'N/A')}")
                st.write(f"Итераций: {metrics.get('Итераций', 'N/A')}")

        st.subheader("Системный лог (последние 50 строк)")
        st.code(data['logs'], language="text")

    except requests.exceptions.RequestException as e:
        st.error(f"Не удалось подключиться к серверу: {e}. Убедитесь, что server.py запущен.")

with tab_management:
    st.header("Управление моделями")
    st.info("Здесь будут кнопки 'Обучить новую модель', 'Откатить к предыдущей версии' и переключатель авто-переобучения.")
    # TODO: Добавить st.button, st.checkbox и вызовы API для управления обучением