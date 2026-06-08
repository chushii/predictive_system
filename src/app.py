import streamlit as st
import requests
import logging
import csv
import io
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

if 'app_started' not in st.session_state:
    logger.info("Streamlit-приложение (Прогноз) запущено")
    st.session_state.app_started = True

API_HOST = os.getenv("PRED_API_HOST", "127.0.0.1")
API_PORT = os.getenv("PRED_API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

st.set_page_config(page_title="PredSys - Прогноз", layout="centered")
st.title("Прогнозирование тенденций использования программных компонент")

DEFAULTS = {
    "type": "Framework", "language": "Python", "category": "Utilities",
    "license": "MIT", "author": "Open Source Community", "ecosystem": "PyPI",
    "status": "Active",
    "downloads": 1000, "stars": 50, "contributors": 10, "dependents": 100,
    "market_share": 0.010, "commits": 100, "releases": 5, "versions": 10,
    "dependencies": 5,
    "open_issues": 5, "closed_issues": 20,
    "quality_score": 0.800, "documentation_score": 0.700, "community_score": 0.750,
    "maturity_score": 0.650,
    "language_trend": 1.000, "category_trend": 1.000,
    "ecosystem_health": 1.000, "seasonality": 1.000,
    "has_ci": True, "has_examples": True, "has_tests": True,
    "has_tutorials": False, "has_website": False,
    "horizon": "3 месяца",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

LANG_OPTIONS = ["Python", "JavaScript", "Go", "Java", "C#", "TypeScript", "Ruby", "Rust", "Kotlin", "Swift", "Scala", "C++", "PHP"]
TYPE_OPTIONS = ["Framework", "Library", "Tool", "Template", "Plugin", "Driver", "API", "SDK", "Middleware", "Starter"]
LICENSE_OPTIONS = ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "ISC", "Unlicense", "MPL-2.0", "LGPL-2.1", "AGPL-3.0", "EPL-2.0"]
CATEGORY_OPTIONS = ["UI Components", "Database Driver", "Testing", "Machine Learning", "Networking", "Security", "Logging", "Caching", "Serialization", "Code Analysis", "Documentation", "Messaging", "Utilities", "Web Framework", "Big Data"]
ECOSYSTEM_OPTIONS = ["PyPI", "npm", "Maven Central", "NuGet", "RubyGems", "Crates.io", "Go Modules"]
STATUS_OPTIONS = ["Active", "Maintained", "Stable", "Deprecated", "Legacy", "Experimental"]
AUTHOR_OPTIONS = ["Open Source Community", "Facebook", "Google", "Amazon", "Microsoft", "Apache Software Foundation", "Eclipse Foundation", "Red Hat", "JetBrains", "VMware", "Spring", "University Research", "Individual Contributors", "GitHub", "Netflix", "Twitter"]
HORIZON_OPTIONS = ["3 месяца", "6 месяцев", "12 месяцев"]

CSV_COLUMNS = [
    "type", "language", "category", "license", "author", "ecosystem", "status",
    "downloads", "stars", "contributors", "dependencies", "dependents",
    "commits", "releases", "versions",
    "open_issues", "closed_issues",
    "market_share", "quality_score", "documentation_score", "community_score", "maturity_score",
    "language_trend", "category_trend", "ecosystem_health", "seasonality",
    "has_ci", "has_examples", "has_tests", "has_tutorials", "has_website",
    "future_downloads_3m", "future_downloads_6m", "future_downloads_12m"
]

def parse_csv_row(csv_string: str) -> dict:
    try:
        reader = csv.reader(io.StringIO(csv_string.strip()))
        row = next(reader)
        if len(row) != len(CSV_COLUMNS):
            st.error(f"Ошибка! Неправильная строка")
            logger.warning("Ошибка парсинга строки: неправильная строка")
            return None

        int_cols = {"downloads", "stars", "contributors", "dependencies", "dependents",
                    "commits", "releases", "versions", "open_issues", "closed_issues"}
        float_cols = {"market_share", "quality_score", "documentation_score", "community_score",
                      "maturity_score", "language_trend", "category_trend", "ecosystem_health", "seasonality"}
        bool_cols = {"has_ci", "has_examples", "has_tests", "has_tutorials", "has_website"}

        result = {}
        for col, value in zip(CSV_COLUMNS, row):
            value = value.strip()
            if col in int_cols:
                result[col] = int(float(value))
            elif col in float_cols:
                result[col] = float(value)
            elif col in bool_cols:
                result[col] = int(float(value)) == 1
            elif col not in ("future_downloads_3m", "future_downloads_6m", "future_downloads_12m"):
                result[col] = value
        return result
    except Exception as e:
        st.error(f"Ошибка парсинга строки: {e}")
        logger.error(f"Ошибка парсинга строки: {e}", exc_info=True)
        return None

if "import_message" not in st.session_state:
    st.session_state.import_message = None

st.subheader("Импорт данных")
csv_input = st.text_area(
    "Вставьте строку из test_set.csv:",
    height=120,
    placeholder="Framework,Python,Utilities,MIT,Open Source Community,PyPI,Active,1000,50,10,5,100,100,5,10,5,20,0.01,0.8,0.7,0.75,0.65,1.0,1.0,1.0,1.0,1,1,1,0,0,1500,2000,3000"
)

if st.button(" Импорт", type="secondary"):
    if csv_input.strip():
        parsed = parse_csv_row(csv_input)
        if parsed:
            for k, v in parsed.items():
                st.session_state[k] = v
            st.session_state.import_message = "Данные успешно импортированы!"
            logger.info("Данные импортированы успешно")
            st.rerun()
    else:
        st.warning("Вставьте CSV-строку")
        logger.warning("Передана пустая строка импорта")

if st.session_state.import_message:
    st.success(st.session_state.import_message)

st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Введите параметры")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Язык", LANG_OPTIONS, key="language")
        st.selectbox("Тип", TYPE_OPTIONS, key="type")
        st.selectbox("Лицензия", LICENSE_OPTIONS, key="license")
    with col2:
        st.selectbox("Категория", CATEGORY_OPTIONS, key="category")
        st.selectbox("Экосистема", ECOSYSTEM_OPTIONS, key="ecosystem")
        st.selectbox("Статус", STATUS_OPTIONS, key="status")

    st.selectbox("Автор/Организация", AUTHOR_OPTIONS, key="author")

    st.subheader("Метрики")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.number_input("Текущие скачивания", min_value=0, key="downloads")
        st.number_input("Звезды (Stars)", min_value=0, key="stars")
        st.number_input("Контрибьюторы", min_value=0, key="contributors")
        st.number_input("Зависимые проекты", min_value=0, key="dependents")
        st.number_input("Доля рынка", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="market_share")
    with col4:
        st.number_input("Коммиты", min_value=0, key="commits")
        st.number_input("Релизы", min_value=0, key="releases")
        st.number_input("Версии", min_value=0, key="versions")
        st.number_input("Внешние зависимости", min_value=0, key="dependencies")
        st.number_input("Quality Score", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="quality_score")
    with col5:
        st.number_input("Documentation Score", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="documentation_score")
        st.number_input("Community Score", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="community_score")
        st.number_input("Maturity Score", min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key="maturity_score")
        st.number_input("Открытые задачи (open_issues)", min_value=0, key="open_issues")
        st.number_input("Закрытые задачи (closed_issues)", min_value=0, key="closed_issues")

    st.subheader("Тренды")
    col6, col7 = st.columns(2)
    with col6:
        st.number_input("Тренд языка", min_value=0.0, step=0.001, format="%.3f", key="language_trend")
        st.number_input("Тренд категории", min_value=0.0, step=0.001, format="%.3f", key="category_trend")
    with col7:
        st.number_input("Здоровье экосистемы", min_value=0.0, step=0.001, format="%.3f", key="ecosystem_health")
        st.number_input("Сезонность", min_value=0.0, step=0.001, format="%.3f", key="seasonality")

    st.subheader("Дополнительно")
    col8, col9 = st.columns(2)
    with col8:
        st.checkbox("CI/CD", key="has_ci")
        st.checkbox("Примеры кода", key="has_examples")
        st.checkbox("Тесты", key="has_tests")
    with col9:
        st.checkbox("Туториалы", key="has_tutorials")
        st.checkbox("Веб-сайт", key="has_website")

    st.markdown("---")
    st.subheader("Параметры прогноза")
    st.selectbox("Горизонт", HORIZON_OPTIONS, key="horizon")
    submitted = st.form_submit_button("Сделать прогноз", use_container_width=True, type="primary")

if submitted:
    payload = {
        "horizon": st.session_state["horizon"],
        "type": st.session_state["type"],
        "language": st.session_state["language"],
        "category": st.session_state["category"],
        "license": st.session_state["license"],
        "author": st.session_state["author"],
        "ecosystem": st.session_state["ecosystem"],
        "status": st.session_state["status"],
        "downloads": float(st.session_state["downloads"]),
        "stars": int(st.session_state["stars"]),
        "contributors": int(st.session_state["contributors"]),
        "dependents": int(st.session_state["dependents"]),
        "market_share": float(st.session_state["market_share"]),
        "commits": int(st.session_state["commits"]),
        "releases": int(st.session_state["releases"]),
        "versions": int(st.session_state["versions"]),
        "dependencies": int(st.session_state["dependencies"]),
        "open_issues": int(st.session_state["open_issues"]),
        "closed_issues": int(st.session_state["closed_issues"]),
        "quality_score": float(st.session_state["quality_score"]),
        "documentation_score": float(st.session_state["documentation_score"]),
        "community_score": float(st.session_state["community_score"]),
        "maturity_score": float(st.session_state["maturity_score"]),
        "language_trend": float(st.session_state["language_trend"]),
        "category_trend": float(st.session_state["category_trend"]),
        "ecosystem_health": float(st.session_state["ecosystem_health"]),
        "seasonality": float(st.session_state["seasonality"]),
        "has_ci": 1 if st.session_state["has_ci"] else 0,
        "has_examples": 1 if st.session_state["has_examples"] else 0,
        "has_tests": 1 if st.session_state["has_tests"] else 0,
        "has_tutorials": 1 if st.session_state["has_tutorials"] else 0,
        "has_website": 1 if st.session_state["has_website"] else 0,
    }

    logger.info("Отправка запроса на прогноз")

    with st.spinner("Выполняется прогноз..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("Прогноз успешно получен")
            st.success("Прогноз успешно получен!")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric(label="Текущие скачивания",
                          value=f"{int(st.session_state['downloads']):,.0f}")
            with col_r2:
                st.metric(label=f"Ожидаемые скачивания через {result['horizon']}",
                          value=f"{result['predicted_downloads']:,.0f}")
        except requests.exceptions.ConnectionError as e:
            st.error("Не удалось подключиться к серверу. Проверьте, что API запущен на http://127.0.0.1:8000")
            logger.error(f"Не удалось подключиться к API: {e}")
        except requests.exceptions.Timeout as e:
            st.error("Превышено время ожидания")
            logger.error(f"Превышено время ожидания ответа от API: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка: {e}")
            logger.error(f"Ошибка запроса к API: {e}", exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    st.json(e.response.json())
                except:
                    st.text(e.response.text)
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            st.error(f"Неожиданная ошибка: {e}")