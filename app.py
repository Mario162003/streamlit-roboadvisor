from Modules.imports import *
from Modules.score_fx import *
from Modules.needvar import *
from Modules.plots_fx import *
from Modules.html_fx import *
from Modules.basicfx import *
from Modules.plots_fx import *
from Modules.portfolio_variables import *
from Modules.portfolio_opt import *
from Modules.portfolioconstruction_fx import *
from Modules.rebalancing_fx import *


# ───────────────────────── helper de recarga ──────────────────────────
def safe_rerun():
    """Recarga la app sin importar la versión de Streamlit."""
    if hasattr(st, "rerun"):                 # Streamlit ≥ 1.25
        st.rerun()
    elif hasattr(st, "experimental_rerun"):  # 1.9 ≤ v < 1.25
        st.experimental_rerun()
    else:
        raise RuntimeError(
            "Versión de Streamlit demasiado antigua. "
            "Actualízala con  pip install -U streamlit"
        )

# ───────────────────────── CUESTIONARIO ───────────────────────────────
# …  (mantengo tu bloque QUESTIONS_EN / QUESTIONS_ES sin cambios) …
QUESTIONS_EN = {
    "Q1.1": {
        "text": "What is your level of general education?",
        "options": ["No formal education", "High school diploma", "Bachelor's degree", "Master's degree or higher"],
    },
    "Q1.2": {
        "text": "What is your level of financial education?",
        "options": ["None", "Basic", "Intermediate", "Advanced", "Formal academic background in finance or economics"],
    },
    "Q1.3": {
        "text": "What is your current employment status?",
        "options": ["Unemployed", "Student", "Employed part-time", "Self-employed", "Employed full-time", "Retired"],
    },
    "Q1.4": {
        "text": "Have you ever worked in a finance-related role?",
        "options": ["Yes", "No"],
    },
    "Q1.5": {
        "text": "How familiar are you with financial assets?",
        "options": ["Stocks", "Bonds", "Mutual Funds", "ETFs", "Cryptocurrencies", "Commodities", "Other"],
    },
    "Q1.6": {
        "text": "Do you have previous investment experience?",
        "options": ["Yes", "No"],
    },
    "Q1.6.1": {
        "text": "How would you rate your level of experience with investments?",
        "options": ["Low", "Normal", "High"],
    },
    "Q1.6.2.1": {
        "text": "What was the total value of your previous investments?",
        "options": ["Less than €5,000", "€5,000 – €25,000", "€25,000 – €100,000", "More than €100,000"],
    },
    "Q1.6.2.2": {
        "text": "How many investment operations have you made?",
        "options": ["Less than 5", "5–20", "21–50", "More than 50"],
    },
    "Q2.1": {
        "text": "What is your average monthly income after taxes?",
        "options": ["Less than €1,000", "€1,000 – €2,000", "€2,000 – €3,500", "€3,500 – €5,000", "More than €5,000"],
    },
    "Q2.2": {
        "text": "What are your average monthly expenses?",
        "options": ["More than €5,000", "€3,500 – €5,000", "€2,000 – €3,500", "€1,000 – €2,000", "Less than €1,000"],
    },
    "Q2.3": {
        "text": "What is the total value of your savings?",
        "options": ["Less than €5,000", "€5,000 – €25,000", "€25,000 – €100,000", "More than €100,000"],
    },
    "Q2.4": {
        "text": "What is your estimated net worth?",
        "options": ["Negative or close to zero", "€0 – €50,000", "€50,000 – €200,000", "€200,000 – €500,000", "More than €500,000"],
    },
    "Q2.5": {
        "text": "What is the maximum acceptable loss as a percentage of your net worth?",
        "options": ["Less than 5%", "5% – 10%", "10% – 20%", "20% – 35%", "More than 35%"],
    },
    "Q3.1": {
        "text": "What is your primary financial objective?",
        "options": ["Capital preservation", "Saving for a specific purchase", "Building retirement savings", "Generating regular income", "Maximizing capital growth"],
    },
    "Q3.2": {
        "text": "What is your investment time horizon?",
        "options": ["Less than 1 year", "1 – 3 years", "3 – 5 years", "5 – 10 years", "More than 10 years"],
    },
    "Q3.3": {
        "text": "What is your risk and return preference?",
        "options": ["I prefer no risk, even if it means lower returns", "I’m willing to take limited risks for modest returns", "I accept moderate risk for potentially higher returns", "I seek high returns and accept the possibility of significant losses"],
    },
    "Q3.4": {
        "text": "What is your initial investment amount?",
        "options": ["Less than €1,000", "€1,000 – €5,000", "€5,000 – €20,000", "€20,000 – €100,000", "More than €100,000"],
    },
    "Q3.5": {
        "text": "What is the frequency of your investment contributions?",
        "options": ["One-time investment", "Irregular / when possible", "Monthly", "Quarterly", "Annually"],
    },
    "Q3.6": {
        "text": "What is the average amount of each contribution?",
        "options": ["Less than €100", "€100 – €500", "€500 – €2,000", "€2,000 – €10,000", "More than €10,000"],
    },
}

QUESTIONS_ES = {
    "Q1.1": {
        "text": "¿Cuál es tu nivel de educación general?",
        "options": ["Sin estudios", "Educación secundaria", "Grado universitario", "Máster o superior"],
    },
    "Q1.2": {
        "text": "¿Cuál es tu nivel de educación financiera?",
        "options": ["Ninguno", "Básico", "Intermedio", "Avanzado", "Formación académica en finanzas o economía"],
    },
    "Q1.3": {
        "text": "¿Cuál es tu situación laboral actual?",
        "options": ["Desempleado", "Estudiante", "Empleado a tiempo parcial", "Autónomo", "Empleado a tiempo completo", "Jubilado"],
    },
    "Q1.4": {
        "text": "¿Ha trabajado alguna vez en un puesto relacionado con las finanzas?",
        "options": ["Sí", "No"],
    },
    "Q1.5": {
        "text": "¿Qué tan familiarizado estás con los activos financieros?",
        "options": ["Acciones", "Bonos", "Fondos de inversión", "ETFs", "Criptomonedas", "Commodities", "Otros"],
    },
    "Q1.6": {
        "text": "¿Tiene experiencia previa en inversiones?",
        "options": ["Sí", "No"],
    },
    "Q1.6.1": {
        "text": "¿Cómo calificaría su experiencia previa en inversiones?",
        "options": ["Baja", "Normal", "Alta"],
    },
    "Q1.6.2.1": {
        "text": "¿Cuál fue el monto total invertido en los últimos 12 meses?",
        "options": ["Menos de €5.000", "€5.000 – €25.000", "€25.000 – €100.000", "Más de €100.000"],
    },
    "Q1.6.2.2": {
        "text": "¿Cuántas operaciones de inversión ha realizado en los últimos 12 meses?",
        "options": ["Menos de 5", "5–20", "21–50", "Más de 50"],
    },
    "Q2.1": {
        "text": "¿Cuál es su ingreso mensual promedio después de impuestos?",
        "options": ["Menos de €1.000", "€1.000 – €2.000", "€2.000 – €3.500", "€3.500 – €5.000", "Más de €5.000"],
    },
    "Q2.2": {
        "text": "¿Cuáles son sus gastos mensuales promedio?",
        "options": ["Más de €5.000", "€3.500 – €5.000", "€2.000 – €3.500", "€1.000 – €2.000", "Menos de €1.000"],
    },
    "Q2.3": {
        "text": "¿Cuál es el valor total de sus ahorros?",
        "options": ["Menos de €5.000", "€5.000 – €25.000", "€25.000 – €100.000", "Más de €100.000"],
    },
    "Q2.4": {
        "text": "¿Cuál es su valor neto estimado?",
        "options": ["Negativo o cercano a cero", "€0 – €50.000", "€50.000 – €200.000", "€200.000 – €500.000", "Más de €500.000"],
    },
    "Q2.5": {
        "text": "¿Cuál es la pérdida máxima aceptable como porcentaje de su valor neto?",
        "options": ["Menos de 5%", "5% – 10%", "10% – 20%", "20% – 35%", "Más de 35%"],
    },
    "Q3.1": {
        "text": "¿Cuál es su objetivo financiero principal?",
        "options": ["Preservación del capital", "Ahorro para una compra específica", "Construcción de ahorros para la jubilación", "Generación de ingresos regulares", "Maximizar el crecimiento del capital"],
    },
    "Q3.2": {
        "text": "¿Cuál es su horizonte temporal de inversión?",
        "options": ["Menos de 1 año", "1 – 3 años", "3 – 5 años", "5 – 10 años", "Más de 10 años"],
    },
    "Q3.3": {
        "text": "¿Cuál es su preferencia entre riesgo y rentabilidad?",
        "options": ["Prefiero no asumir riesgos, aunque signifique menores rendimientos", "Estoy dispuesto a asumir riesgos limitados por rendimientos modestos", "Acepto riesgos moderados por rendimientos potencialmente mayores", "Busco altos rendimientos y acepto posibles pérdidas significativas"],
    },
    "Q3.4": {
        "text": "¿Cuál es el monto inicial de inversión?",
        "options": ["Menos de €1.000", "€1.000 – €5.000", "€5.000 – €20.000", "€20.000 – €100.000", "Más de €100.000"],
    },
    "Q3.5": {
        "text": "¿Con qué frecuencia realiza aportaciones a su inversión?",
        "options": ["Inversión única", "Irregular / cuando sea posible", "Mensual", "Trimestral", "Anual"],
    },
    "Q3.6": {
        "text": "¿Cuál es el monto promedio por aportación?",
        "options": ["Menos de €100", "€100 – €500", "€500 – €2.000", "€2.000 – €10.000", "Más de €10.000"],
    },
}

# 1. Helpers de idioma y scoring (usa tus funciones reales)
def get_questions():
    return QUESTIONS_ES if st.session_state.lang == "ES" else QUESTIONS_EN

def score_answers(ans: dict):
    if st.session_state.lang == "ES":
        return score_risk_profile_ESP_stm_v2(ans)
    return score_risk_profile_EN_stm_v2(ans)

# ───────────────────────── configuración general ─────────────────────────
import streamlit as st
from collections import defaultdict

st.set_page_config(
    layout="wide",
    page_title="Robo-Advisor",
    page_icon="🧮"
)

# ───────────────────────── selector de idioma ───────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "ES"

st.sidebar.selectbox(
    "Idioma / Language",
    ["ES", "EN"],
    format_func=lambda x: "Español 🇪🇸" if x == "ES" else "English 🇬🇧",
    key="lang"
)

lang = st.session_state.lang

# ───────────────────────── estado inicial ───────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = defaultdict(lambda: None)

answers = st.session_state.answers
questions = get_questions()  # Asegúrate de que devuelve texto por idioma

# ───────────────────────── navegación interna ───────────────────────────
def _next_step():
    st.session_state.step += 1
    safe_rerun()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PASO 0 – Pantalla de bienvenida                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center;'>🧮 " + 
                ("Cuestionario de Perfil de Riesgo" if lang == "ES" else "Risk Profile Questionnaire") + 
                "</h1>", unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 👋 " + ("Bienvenido al cuestionario" if lang == "ES" else "Welcome to the questionnaire"))

        st.markdown("""
        <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
        {}
        </div>
        """.format(
            "Este cuestionario está diseñado para evaluar tu situación financiera, experiencia en inversiones y tolerancia al riesgo, con el fin de ofrecerte una cartera alineada con tu perfil personal.<br><br>"
            "Por favor, responde a todas las preguntas de forma honesta y detallada. Los resultados nos permitirán brindarte recomendaciones de inversión adaptadas a tus objetivos financieros y capacidad de riesgo.<br><br>"
            "🕐 El cuestionario no debería tomarte más de 5 minutos."
            if lang == "ES" else
            "This questionnaire is designed to assess your financial situation, investment experience, and risk tolerance in order to provide you with a portfolio aligned with your personal profile.<br><br>"
            "Please answer all questions honestly and thoroughly. The results will help us deliver investment recommendations tailored to your financial goals and risk capacity.<br><br>"
            "🕐 The questionnaire is planned to take no more than 5 minutes."
        ), unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Comenzar" if lang == "ES" else "Start", use_container_width=True):
            st.session_state.step += 1
            st.rerun()



# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PASO 1 – Bloque 1: Experiencia e información financiera              ║
# ╚═══════════════════════════════════════════════════════════════════════╝
elif st.session_state.step == 1:
    st.markdown("### 📊 " + (
        "Bloque 1: Experiencia en Inversiones y Conocimientos Financieros"
        if lang == "ES" else
        "Block 1: Investment Experience and Financial Knowledge"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Esta sección nos ayuda a entender tu trayectoria, nivel de educación financiera y familiaridad con los mercados financieros. "
        "Es fundamental para ofrecerte recomendaciones de inversión que se ajusten a tu perfil de riesgo y a tus objetivos."
        if lang == "ES" else
        "This section helps us understand your background, financial literacy, and familiarity with financial markets. "
        "It is essential for tailoring investment advice that matches your risk profile and objectives."
    ), unsafe_allow_html=True)

    st.divider()

    with st.form("sec1_form"):
        for key in ["Q1.1", "Q1.2", "Q1.3", "Q1.4"]:
            q = questions[key]
            answers[key] = st.selectbox(q["text"], [""] + q["options"], index=0)

        q15 = questions["Q1.5"]
        answers["Q1.5"] = st.multiselect(q15["text"], q15["options"], default=[])

        q16 = questions["Q1.6"]
        answers["Q1.6"] = st.selectbox(q16["text"], [""] + q16["options"], index=0)

        if st.form_submit_button("Continuar" if lang == "ES" else "Continue", use_container_width=True):
            yes_txt = "Sí" if lang == "ES" else "Yes"
            st.session_state.skip_children = answers["Q1.6"] != yes_txt
            _next_step()

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PASO 2 – Preguntas hijas de Q1.6                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
elif st.session_state.step == 2:

    # Si no hay que responderlas, se salta automáticamente
    if st.session_state.get("skip_children"):
        _next_step()

    st.markdown("### 🧾 " + (
        "Preguntas adicionales sobre tu experiencia previa" if lang == "ES"
        else "Additional questions about your prior experience"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Estas preguntas adicionales nos permitirán comprender mejor los detalles de tu experiencia pasada en inversiones o gestión de productos financieros."
        if lang == "ES" else
        "These additional questions will help us better understand the details of your past experience with investments or financial products."
    ), unsafe_allow_html=True)

    st.divider()

    with st.form("sec1_child_form"):
        for key in ["Q1.6.1", "Q1.6.2.1", "Q1.6.2.2"]:
            q = questions[key]
            answers[key] = st.selectbox(q["text"], [""] + q["options"], index=0)

        if st.form_submit_button("Continuar" if lang == "ES" else "Continue", use_container_width=True):
            _next_step()

elif st.session_state.step == 3:
    st.markdown("### 💰 " + (
        "Bloque 2: Situación Financiera Actual" if lang == "ES"
        else "Block 2: Current Financial Situation"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Esta sección nos ayuda a evaluar tu estabilidad económica, tus ingresos y tu capacidad para asumir riesgos financieros."
        if lang == "ES" else
        "This section helps us evaluate your financial stability, income, and ability to take on financial risks."
    ), unsafe_allow_html=True)

    st.divider()

    with st.form("sec2_form"):
        for key in ["Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5"]:
            q = questions[key]
            answers[key] = st.selectbox(q["text"], [""] + q["options"], index=0)

        if st.form_submit_button("Continuar" if lang == "ES" else "Continue", use_container_width=True):
            _next_step()

elif st.session_state.step == 4:
    st.markdown("### 🎯 " + (
        "Bloque 3: Tolerancia al Riesgo y Horizonte Temporal" if lang == "ES"
        else "Block 3: Risk Tolerance and Investment Horizon"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "En esta sección analizamos cómo reaccionas ante pérdidas potenciales, tu actitud hacia el riesgo y el tiempo durante el cual planeas mantener tu inversión."
        if lang == "ES" else
        "In this section, we analyze how you react to potential losses, your risk attitude, and the time you intend to keep your investments."
    ), unsafe_allow_html=True)

    st.divider()

    with st.form("sec3_form"):
        for key in ["Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5", "Q3.6"]:
            q = questions[key]
            answers[key] = st.selectbox(q["text"], [""] + q["options"], index=0)

        if st.form_submit_button("Enviar" if lang == "ES" else "Submit", use_container_width=True):
            res = score_answers(dict(answers))
            st.session_state.risk_profile = res["risk_profile"]
            st.session_state.risk_score   = res["score"]
            st.session_state.frequency    = res.get("Frequency")
            _next_step()

elif st.session_state.step == 5:
    st.markdown("### 🧾 " + (
        "Resultado del cuestionario" if lang == "ES"
        else "Questionnaire Result"))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(
            f"✅ Perfil asignado: **{st.session_state.risk_profile}**  \n"
            f"📊 Puntuación obtenida: `{st.session_state.risk_score}`"
            if lang == "ES"
            else f"✅ Assigned profile: **{st.session_state.risk_profile}**  \n"
                 f"📊 Score obtained: `{st.session_state.risk_score}`"
        )

        st.markdown("""
        <div style="text-align: justify; font-size: 16px; line-height: 1.6; margin-top: 1em;">
        {}
        </div>
        """.format(
            "Gracias por completar el cuestionario. Ahora podemos ofrecerte una cartera adaptada a tu perfil de riesgo. "
            "Puedes continuar para explorar tu asignación recomendada y simular el comportamiento de la cartera."
            if lang == "ES" else
            "Thank you for completing the questionnaire. We can now offer you a portfolio aligned with your risk profile. "
            "You may proceed to explore your recommended allocation and simulate the portfolio's behavior."
        ), unsafe_allow_html=True)

        st.markdown("---")

        if st.button("Siguiente paso ➡️" if lang == "ES" else "Next step ➡️", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

elif st.session_state.step == 6:
    st.markdown("### 📈 " + (
        "Simulación de inversión" if lang == "ES" else "Investment Simulation"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "En esta sección puedes simular la evolución de tu inversión a lo largo del tiempo. "
        "Introduce tu capital inicial, aportaciones periódicas y frecuencia, y el sistema ejecutará un back‑test con la cartera recomendada."
        if lang == "ES" else
        "In this section, you can simulate how your investment would evolve over time. "
        "Enter your initial capital, periodic contributions, and frequency, and the system will run a back‑test using your recommended portfolio."
    ), unsafe_allow_html=True)

    st.divider()

    # ----- 1) Inputs del usuario -----------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        PV0_in = st.number_input(
            "Inversión inicial (€)" if lang == "ES" else "Initial investment (€)",
            min_value=500.0,  step=100.0,  value=10_000.0,
            format="%.0f"
        )

    with col2:
        cash_in = st.number_input(
            "Aporte periódico (€)" if lang == "ES" else "Periodic contribution (€)",
            min_value=0.0,  step=50.0,  value=250.0,
            format="%.0f"
        )

    with col3:
        freq_lbl = st.selectbox(
            "Frecuencia" if lang == "ES" else "Frequency",
            options=["Mensual", "Trimestral", "Anual"] if lang == "ES"
                    else ["Monthly", "Quarterly", "Yearly"],
        )

    FREQ_CODE = {
        "Mensual": "BME",  "Monthly": "BME",
        "Trimestral": "BQE", "Quarterly": "BQE",
        "Anual": "BYE", "Yearly": "BYE",
    }
    cash_freq_code = FREQ_CODE[freq_lbl]

    st.markdown("")

    # ----- 2) Botón para lanzar backtest ----------------------------
    if st.button("Lanzar back‑test ▶️" if lang == "ES" else "Run back‑test ▶️", use_container_width=True):

        user_profile = st.session_state.risk_profile
        reco = get_portfolio_recommendation_profiled(user_profile, results_v9)
        initial_w = reco["weights"]

        lims = risk_profiles_weights[user_profile.upper()]
        w_min, w_max = lims["min_weight"], lims["max_weight"]

        hist = bck_test_final_DYNAMIC_v10(
            PV0=PV0_in,
            price_data=prices_df,
            rf_annual=rf_annual,
            initial_weights=initial_w,
            FI_TICKERS=selected_bonds_v3,
            EQ_TICKERS=selected_stocks_v3,
            risk_profile=user_profile.capitalize(),
            cost_rate=0.005,
            PROFILE_TARGET=PROFILE_TARGET,
            first_cash_date=None,
            etf_benchmark_df=stocks_bench_df_filt_2,
            etf_log_returns=stocks_ret_filt_v2,
            benchmark_log_returns=bx_r_filt_v2,
            cash_amount=cash_in,
            cash_freq=cash_freq_code,
            min_weight=w_min,
            max_weight=w_max,
            winsorize_er=True,
            winsorize_limits=(0.05, 0.90)
        )

        st.session_state.backtest_hist = hist  # guardar resultados


# ───────── 3) Mostrar resultados si existen ─────────────────────────
if "backtest_hist" in st.session_state:
    hist = st.session_state.backtest_hist

    import plotly.graph_objects as go

    # Serie principal del valor del portafolio
    pv_series = hist[("PORTFOLIO", "Value")].rename("Portfolio Value")

    # Serie acumulada invertida
    cash_flow_ser = build_cash_flow_series(
        index=pv_series.index,
        cash_amount=cash_in,
        cash_freq=cash_freq_code,
        first_cash_date=None,
    )
    invested_series = cash_flow_ser.cumsum() + pv_series.iloc[0]

    # Retorno %
    return_series = (pv_series / invested_series - 1) * 100

    # Crear DataFrame para hover
    df_hover = pd.DataFrame({
        "Date": pv_series.index,
        "Portfolio Value": pv_series,
        "Invested": invested_series,
        "Return %": return_series,
    })

    # Crear gráfico con hover detallado
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hover["Date"],
        y=df_hover["Portfolio Value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(width=2),
        customdata=df_hover[["Portfolio Value", "Invested", "Return %"]].round(2).values,
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br><br>" +
            "💹 Valor Portfolio: €%{customdata[0]:,.2f}<br>" +
            "💰 Total Invertido: €%{customdata[1]:,.2f}<br>" +
            "📈 Retorno: %{customdata[2]:.2f}%<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Valor del Portafolio" if lang == "ES" else "Portfolio Value Over Time",
        height=400,
        margin=dict(l=30, r=30, t=40, b=20),
        yaxis_tickformat=".2f",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)



    # --- Detalle de pesos por activo ---
    with st.expander("⏬ Detalle de pesos por activo" if lang == "ES" else "⏬ Asset Weights Over Time"):
        import plotly.express as px

        w = hist.xs("Weight", level=1, axis=1)
        w.columns = w.columns.get_level_values(0)
        w = w.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Weight")

        fig_w = px.line(w, x="Date", y="Weight", color="Asset")
        fig_w.update_layout(
            title_text="Pesos por activo" if lang == "ES" else "Asset Weights",
            legend_itemclick="toggleothers",  # 👈 esto activa el comportamiento deseado
            height=400,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_w, use_container_width=True)


    # --- Detalle de holdings ---
    with st.expander("⏬ Detalle de holdings (nº acciones)" if lang == "ES" else "⏬ Holdings (number of shares)"):
        h = hist.xs("Holdings", level=1, axis=1)
        h.columns = h.columns.get_level_values(0)
        h = h.reset_index().melt(id_vars="Date", var_name="Asset", value_name="Holdings")

        fig_h = px.line(h, x="Date", y="Holdings", color="Asset")
        fig_h.update_layout(
            title_text="Holdings (nº de acciones)" if lang == "ES" else "Holdings (number of shares)",
            legend_itemclick="toggleothers",  # 👈 mismo ajuste aquí
            height=400,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_h, use_container_width=True)



    # --- Tabla completa ---
    with st.expander("⏬ Tabla completa" if lang == "ES" else "⏬ Full Data Table"):
        st.dataframe(hist.style.format(thousands=".", precision=2))

# ------------------------------------------------------------------
#  📊  MÉTRICAS Y GRÁFICOS DE PERFORMANCE
# ------------------------------------------------------------------

    cash_flows = build_cash_flow_series(
        index=pv_series.index,
        cash_amount=cash_in,
        cash_freq=cash_freq_code,
        first_cash_date=None,
    )

    # ① Métricas de rendimiento
    perf = perf_metrics_v2(
        prices=pv_series,
        cash_flows=cash_flows,
        freq=252,
        rf=rf_annual / 252,
    )

    st.markdown("### 📋 " + (
        "Métricas de rendimiento" if lang == "ES" else "Performance Metrics"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Aquí puedes observar los indicadores clave del rendimiento de tu portafolio durante el periodo simulado: rendimiento anualizado, volatilidad, ratio de Sharpe, y más."
        if lang == "ES" else
        "Here you can review the key performance indicators of your simulated portfolio: annualized return, volatility, Sharpe ratio, and more."
    ), unsafe_allow_html=True)

    # Diccionario con formato personalizado
    metrics_pct = ["Cumulative Return", "Annualised Return", "Annualised Vol", "Max Drawdown"]

    # Crear DataFrame con los formatos
    perf_cleaned = {
        k: f"{v*100:.2f}%" if k in metrics_pct else f"{v:,.2f}"
        for k, v in perf.items()
    }
    perf_df = pd.DataFrame.from_dict(perf_cleaned, orient="index", columns=["Valor" if lang == "ES" else "Value"])

    # Mostrar tabla centrada y más estrecha
    with st.container():
        st.markdown("""
            <style>
            .metrics-table {
                margin: auto;
                width: 50%;
                text-align: center;
                font-size: 16px;
            }
            .metrics-table th, .metrics-table td {
                padding: 8px 12px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(
            perf_df.to_html(classes="metrics-table", border=0, justify="center"),
            unsafe_allow_html=True
        )

    # ② Gráfico valor vs capital invertido
    invested = cash_flows.cumsum() + pv_series.iloc[0]
    chart_df = pd.DataFrame({
        "Portfolio value": pv_series,
        "Total invested": invested,
    }).round(2)

    st.markdown("### 💰 " + (
        "Valor del portafolio vs capital aportado" if lang == "ES"
        else "Portfolio Value vs Invested Capital"))

    fig_val = px.line(chart_df, labels={"value": "", "index": "Fecha" if lang == "ES" else "Date"})
    fig_val.update_layout(
        legend_title_text="",
        yaxis_tickformat=".2f",
        height=400,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    st.plotly_chart(fig_val, use_container_width=True)

    st.divider()

    # ③ Curva de drawdown
    dd = (pv_series / pv_series.cummax() - 1).rename("Drawdown")

    st.markdown("### 📉 " + (
        "Curva de drawdown" if lang == "ES" else "Drawdown Curve"))

    fig_dd = px.area(
        dd.reset_index(),
        x="Date",
        y="Drawdown",
        labels={"Drawdown": ""},
        height=300
    )
    fig_dd.update_layout(
        yaxis_tickformat=".2%",
        showlegend=False,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    #------------------------------------------------------------------
    #      Normal distrbution plot
    #------------------------------------------------------------------

    import numpy as np
    from scipy.stats import norm
    import plotly.graph_objects as go
    cash_flow_ser  = build_cash_flow_series(
    index           = pv_series.index,
    cash_amount     = cash_in,
    cash_freq       = cash_freq_code,
    first_cash_date = None,
    )
    # -------------------- SLIDER del usuario ----------------------------
    horizon = st.slider(
    "Horizonte de proyección (días hábiles)" if lang == "ES"
    else "Projection horizon (trading days)",
    min_value=1, max_value=252, value=22, step=5
    )

    invested_series = cash_flow_ser.cumsum() + pv_series.iloc[0]
    net_gain        = pv_series - invested_series
    daily_diff      = net_gain.diff().dropna()
    
    # --- cálculo estadístico base ---
    mu_d = daily_diff.mean()
    sigma_d = daily_diff.std()
    mu_p = mu_d * horizon
    sigma_p = sigma_d * np.sqrt(horizon)
    wc = mu_p - 1.645 * sigma_p  # VaR 95%

    # --- distribución normal ---
    x = np.linspace(mu_p - 2.2 * sigma_p, mu_p + 2.2 * sigma_p, 300)
    y = norm.pdf(x, mu_p, sigma_p)

    # --- texto descriptivo ---
    st.markdown("### 📊 " + (
        "Distribución proyectada de ganancias/pérdidas" if lang == "ES"
        else "Projected Profit/Loss Distribution"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Este gráfico representa la distribución esperada de ganancias o pérdidas para tu portafolio en un horizonte determinado, basándose en su comportamiento reciente."
        if lang == "ES" else
        "This chart shows the expected distribution of gains or losses for your portfolio over a selected horizon, based on recent historical performance."
    ), unsafe_allow_html=True)

    # --- curva principal ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Distribución" if lang == "ES" else "Distribution",
        line=dict(color="#1f77b4", width=2),
        hoverinfo="skip"
    ))

    # --- marcador media esperada ---
    fig.add_trace(go.Scatter(
        x=[mu_p],
        y=[norm.pdf(mu_p, mu_p, sigma_p)],
        mode="markers+text",
        name="Media esperada" if lang == "ES" else "Expected Mean",
        marker=dict(color="green", size=8),
        text=[f"€{mu_p:,.0f}"],
        textposition="top center"
    ))

    # --- marcador VaR 95% ---
    fig.add_trace(go.Scatter(
        x=[wc],
        y=[norm.pdf(wc, mu_p, sigma_p)],
        mode="markers+text",
        name="VaR 95 %" if lang == "ES" else "95% Worst Case",
        marker=dict(color="red", size=8),
        text=[f"€{wc:,.0f}"],
        textposition="bottom center"
    ))

    # --- layout general ---
    fig.update_layout(
        title=dict(
            text=f"Distribución de Pérdidas/Ganancias a {horizon} días" if lang == "ES"
            else f"P/L Distribution at {horizon}-Day Horizon",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Beneficio / Pérdida (€)" if lang == "ES" else "Profit / Loss (€)",
        yaxis_title="Densidad" if lang == "ES" else "Density",
        height=400,
        margin=dict(t=40, l=30, r=30, b=30),
        showlegend=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

        # Perfil del usuario
    risk_profile = st.session_state.get("risk_profile", "Desconocido")

    # INTERPRETACIÓN DE VaR
    if lang == "ES":
        var_text = (
            f"🔻 El **VaR al 95 %** indica que, en un escenario adverso, podrías llegar a perder hasta **{abs(wc):,.0f} €** "
            f"en los próximos **{horizon} días hábiles**. Este valor refleja el riesgo potencial de tu cartera con perfil **{risk_profile}**."
        )
    else:
        var_text = (
            f"🔻 The **95% VaR** suggests that in a worst-case scenario, you could lose up to **€{abs(wc):,.0f}** "
            f"over the next **{horizon} trading days**. This reflects the downside risk of your portfolio with a **{risk_profile}** profile."
        )

    # INTERPRETACIÓN DE LA GANANCIA ESPERADA (mu_p)
    if lang == "ES":
        mu_text = (
            f"📈 La **ganancia esperada** para este mismo periodo es de aproximadamente **{mu_p:,.0f} €**. "
            f"Esto representa el escenario medio estimado para tu cartera en base a su comportamiento histórico reciente."
        )
    else:
        mu_text = (
            f"📈 The **expected average gain** for this horizon is approximately **€{mu_p:,.0f}**. "
            f"This represents the estimated mean scenario for your portfolio based on recent historical performance."
        )

    # MOSTRAR INTERPRETACIONES
    st.markdown("### 🧠 " + (
        "Interpretación de los resultados" if lang == "ES" else "Result Interpretation"))

    st.info(var_text)
    st.info(mu_text)


    # ------------------------------------------------------------------
    #  📊  Portfolio vs Benchmark – Comparativa de métricas
    # ------------------------------------------------------------------

    import altair as alt
    import pandas as pd

    st.markdown("### 📊 " + (
        "Portfolio vs Benchmark – Comparativa de métricas"
        if lang == "ES" else
        "Portfolio vs Benchmark – Metric Comparison"))

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; line-height: 1.6;">
    {}
    </div>
    """.format(
        "Este gráfico compara visualmente tu portafolio con su índice de referencia, usando métricas clave como rendimiento acumulado, rendimiento anualizado, ratio de Sharpe y drawdown máximo."
        if lang == "ES" else
        "This chart visually compares your portfolio against its benchmark, using key metrics such as cumulative return, annualized return, Sharpe ratio, and maximum drawdown."
    ), unsafe_allow_html=True)

    # --- 1) DataFrame de métricas
    metrics_df = compare_multiple_performances_v2(
        portfolios={"Mi Portafolio" if lang == "ES" else "My Portfolio": pv_series},
        bench_px=bench_px_sp,
        cash_flows=cash_flows,
        freq=252,
        rf=rf_annual / 252,
    )

    # --- 2) Métricas a graficar
    metrics_to_plot = [
        "Cumulative Return",
        "Annualised Return",
        "Sharpe Ratio",
        "Max Drawdown",
    ]

    plot_df = (
        metrics_df
        .loc[metrics_to_plot]
        .reset_index(names="Metric")
        .melt(id_vars="Metric", var_name="Serie", value_name="Valor")
    )

    # Guardamos los valores originales para el tooltip
    plot_df["TooltipValor"] = plot_df["Valor"]

    # Aplicamos formato visual:
    plot_df.loc[plot_df["Metric"].str.contains("Return"), "Valor"] *= 100
    plot_df.loc[plot_df["Metric"].eq("Max Drawdown"),     "Valor"] *= 100
    plot_df.loc[plot_df["Metric"].eq("Sharpe Ratio"),     "Valor"] *= 100  # solo visual

    # Traducción de nombres si idioma ES
    if lang == "ES":
        plot_df["Serie"] = plot_df["Serie"].replace({
            "My Portfolio": "Mi Portafolio",
            "Benchmark": "Índice"
        })
        plot_df["Metric"] = plot_df["Metric"].replace({
            "Cumulative Return": "Retorno acumulado",
            "Annualised Return": "Retorno anualizado",
            "Sharpe Ratio": "Ratio de Sharpe",
            "Max Drawdown": "Drawdown máximo"
        })

    # Gráfico con etiquetas verticales y métrica arriba
    base = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Serie:N", title="", axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("Valor:Q", title="", axis=alt.Axis(format=".2f")),
            color=alt.Color("Serie:N", legend=None,
                            scale=alt.Scale(
                                domain=["My Portfolio", "Benchmark"] if lang == "EN" else ["Mi Portafolio", "Índice"],
                                range=["#4CAF50", "#888888"])),
            tooltip=[
                alt.Tooltip("Serie:N", title="Serie"),
                alt.Tooltip("Metric:N", title="Métrica" if lang == "ES" else "Metric"),
                alt.Tooltip("TooltipValor:Q", format=".4f", title="Valor real")
            ]
        )
        .properties(width=120, height=200)
    )

    bar_chart = base.facet(
        row=alt.Row("Metric:N",
                    header=alt.Header(labelAngle=0, labelOrient="left", title=None))
    )

    st.altair_chart(bar_chart, use_container_width=True)


    import os
    from jinja2 import Template

    # Detectar si estamos en Streamlit Cloud
    is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1"

    if not is_cloud:
        import pdfkit
        import tempfile

        # 1. Crear plantilla HTML
        pdf_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Reporte de inversión</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; font-size: 14px; }
                h1, h2 { color: #4CAF50; }
                table { width: 100%%; border-collapse: collapse; margin-top: 15px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <h2>📌 Perfil de riesgo: {{ profile }} (score: {{ score }})</h2>

            <p><strong>Horizonte de análisis:</strong> {{ horizon }} días hábiles</p>
            <table>
                <tr><th>Métrica</th><th>Valor</th></tr>
                <tr><td><strong>Ganancia neta</strong></td><td>€{{ gain }}</td></tr>
                <tr><td><strong>Valor del portfolio</strong></td><td>€{{ portfolio_value }}</td></tr>
                <tr><td><strong>Total invertido</strong></td><td>€{{ invested }}</td></tr>
                <tr><td><strong>Retorno</strong></td><td>{{ return_pct }}%</td></tr>
                <tr><td><strong>VaR (95%)</strong></td><td>€{{ var_95 }}</td></tr>
                <tr><td><strong>Ganancia esperada</strong></td><td>€{{ expected_gain }}</td></tr>
            </table>

            <h2>🧠 Interpretaciones</h2>
            <p>{{ var_text }}</p>
            <p>{{ mu_text }}</p>
        </body>
        </html>
        """

        gain = net_gain.iloc[-1] if hasattr(net_gain, 'iloc') else net_gain[-1]
        
        var_text, mu_text = get_html_interpretations(
            wc=wc,
            mu_p=mu_p,
            horizon=horizon,
            risk_profile=st.session_state.get("risk_profile", "Desconocido"),
            lang=lang
        )

        # 2. Rellenar con tus variables
        template = Template(pdf_template)
        html_filled = template.render(
            title="Informe de Simulación de Inversión" if lang == "ES" else "Investment Simulation Report",
            profile=st.session_state.get("risk_profile", "Desconocido"),
            score=st.session_state.get("risk_score", "-"),
            horizon=horizon,
            gain=f"{gain:,.2f}",
            portfolio_value=f"{pv_series[-1]:,.0f}",
            invested=f"{invested_series[-1]:,.0f}",
            return_pct=f"{metrics_df.loc['Cumulative Return', 'Mi Portafolio' if lang == 'ES' else 'My Portfolio'] * 100:,.2f}",
            var_95=f"{wc:,.0f}",
            expected_gain=f"{mu_p:,.0f}",
            var_text=var_text,
            mu_text=mu_text
        )

        # 3. Convertir a PDF y generar botón
        if st.button("📥 Descargar resumen PDF" if lang == "ES" else "📥 Download PDF summary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdfkit.from_string(html_filled, tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as file:
                    st.download_button(
                        label="📥 Descargar PDF" if lang == "ES" else "📥 Download PDF",
                        data=file,
                        file_name="reporte_inversion_2020-2025.pdf",
                        mime="application/pdf"
                    )

    else:
        st.warning(
            "⚠️ La generación de PDF está solo disponible cuando ejecutas esta app desde tu ordenador."
            if lang == "ES"
            else "⚠️ PDF generation is only available when running this app locally."
        )
