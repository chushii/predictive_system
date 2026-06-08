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

        st.subheader("Системный лог")
        st.code(data['logs'], language="text")

    except requests.exceptions.RequestException as e:
        st.error(f"Не удалось подключиться к серверу: {e}. Убедитесь, что server.py запущен.")

with tab_management:
    st.header("Управление моделями")

    try:
        status_response = requests.get(f"{API_URL}/api/models/status", timeout=5)
        status_response.raise_for_status()
        models_status = status_response.json()
    except Exception as e:
        st.error(f"Не удалось получить статус файлов моделей: {e}")
        models_status = {}

    cols = st.columns(3)
    for i, horizon in enumerate(HORIZONS):
        with cols[i]:
            st.subheader(f"Горизонт: {horizon}")
            status = models_status.get(horizon, {})

            st.write(f"Основная модель: {status.get('main_date') or 'отсутствует'}")
            st.write(f"Резервная копия: {status.get('backup_date') or 'отсутствует'}")

            st.divider()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Реобучение", key=f"retrain_{horizon}"):
                    with st.spinner("Отправка запроса на реобучение..."):
                        try:
                            res = requests.post(
                                f"{API_URL}/retrain",
                                json={"horizon": horizon},
                                timeout=5
                            )
                            if res.status_code == 202:
                                st.success("Реобучение запущено в фоновом режиме")
                            else:
                                st.error(f"Ошибка сервера: {res.text}")
                        except Exception as e:
                            st.error(f"Ошибка соединения: {e}")

            with col_btn2:
                has_backup = status.get('backup_date') is not None
                if st.button("Откат", key=f"rollback_{horizon}", disabled=not has_backup):
                    with st.spinner("Выполняется откат..."):
                        try:
                            res = requests.post(
                                f"{API_URL}/api/models/rollback?horizon={horizon}",
                                timeout=5
                            )
                            if res.status_code == 200:
                                st.success("Откат выполнен успешно.")
                                st.rerun()
                            else:
                                st.error(f"Ошибка сервера: {res.text}")
                        except Exception as e:
                            st.error(f"Ошибка соединения: {e}")