from Modules.imports import *
from typing import Dict, Any

def score_risk_profile(answers: dict) -> dict:
    score = 0

    mail_ = answers.get('Dirección de correo electrónico')

    # --- Block 1: Experience and Knowledge ---
    q1_1 = answers.get("Q1.1 Level of General Education")
    score += {"No formal education": 0, "High school diploma": 1, "Bachelor's degree": 2, "Master's degree or higher": 3}.get(q1_1, 0)

    q1_2 = answers.get("Q1.2 Level of Financial Education")
    score += {"None": 0, "Basic": 1, "Intermediate": 2, "Advanced": 3, "Formal academic background in finance or economics": 4}.get(q1_2, 0)

    q1_3 = answers.get("Q1.3 Current Employment Status")
    score += {
        "Unemployed": 1, "Student": 1, "Employed part-time": 2,
        "Self-employed": 2, "Employed full-time": 3, "Retired": 2
    }.get(q1_3, 0)

    q1_4 = answers.get("Q1.4 Have You Ever Worked in a Finance-Related Role?")
    score += 3 if q1_4 == "Yes" else 0

    q1_5_assets = answers.get("Q1.5 Familiarity with Financial Assets", [])
    score += min(len(q1_5_assets), 5)

    q1_6 = answers.get("Q1.6 Previous Investment Experience")
    if q1_6 == "Yes":
        q1_6_1_level = answers.get("Q1.6.1.1")
        score += {"Low": 1, "Normal": 2, "High": 3}.get(q1_6_1_level, 0)

        q1_6_2_vol = answers.get("Q1.6.2.1.1")
        score += {
            "Less than €5,000": 1,
            "€5,000 – €25,000": 2,
            "€25,000 – €100,000": 3,
            "More than €100,000": 4
        }.get(q1_6_2_vol, 0)

        q1_6_2_ops = answers.get("Q1.6.2.1.2")
        score += {
            "Less than 5": 0,
            "5–20": 1,
            "21–50": 2,
            "More than 50": 3
        }.get(q1_6_2_ops, 0)

    # --- Block 2: Financial Situation ---
    q2_1 = answers.get("Q2.1 Average Monthly Income (After Taxes)")
    score += {
        "Less than €1,000": 1,
        "€1,000 – €2,000": 2,
        "€2,000 – €3,500": 3,
        "€3,500 – €5,000": 4,
        "More than €5,000": 5
    }.get(q2_1, 0)

    q2_2 = answers.get("Q2.2 Average Monthly Expenses")
    score += {
        "More than €5,000": 0,
        "€3,500 – €5,000": 1,
        "€2,000 – €3,500": 2,
        "€1,000 – €2,000": 3,
        "Less than €1,000": 4
    }.get(q2_2, 0)

    q2_3 = answers.get("Q2.3 Total Value of Savings")
    score += {
        "Less than €5,000": 1,
        "€5,000 – €25,000": 2,
        "€25,000 – €100,000": 3,
        "More than €100,000": 4
    }.get(q2_3, 0)

    q2_4 = answers.get("Q2.4 Estimated Net Worth")
    score += {
        "Negative or close to zero": 1,
        "€0 – €50,000": 2,
        "€50,000 – €200,000": 3,
        "€200,000 – €500,000": 4,
        "More than €500,000": 5
    }.get(q2_4, 0)

    q2_5 = answers.get("Q2.5 Maximum Acceptable Loss (% of Net Worth)")
    score += {
        "Less than 5%": 0,
        "5% – 10%": 1,
        "10% – 20%": 2,
        "20% – 35%": 3,
        "More than 35%": 4
    }.get(q2_5, 0)

    # --- Block 3: Financial Objectives and Risk Tolerance ---
    q3_1 = answers.get("Q3.1 Primary Financial Objectives")
    score += {
        "Capital preservation": 0,
        "Saving for a specific purchase": 1,
        "Building retirement savings": 2,
        "Generating regular income": 3,
        "Maximizing capital growth": 4
    }.get(q3_1, 0)

    q3_2 = answers.get("Q3.2 Investment Time Horizon")
    score += {
        "Less than 1 year": 0,
        "1 – 3 years": 1,
        "3 – 5 years": 2,
        "5 – 10 years": 3,
        "More than 10 years": 4
    }.get(q3_2, 0)

    q3_3 = answers.get("Q3.3 Risk and Return Preference")
    score += {
        "I prefer no risk, even if it means lower returns": 0,
        "I’m willing to take limited risks for modest returns": 2,
        "I accept moderate risk for potentially higher returns": 4,
        "I seek high returns and accept the possibility of significant losses": 5
    }.get(q3_3, 0)

    q3_4 = answers.get("Q3.4 Initial Investment Amount")
    score += {
        "Less than €1,000": 0,
        "€1,000 – €5,000": 1,
        "€5,000 – €20,000": 2,
        "€20,000 – €100,000": 3,
        "More than €100,000": 4
    }.get(q3_4, 0)

    q3_5 = answers.get("Q3.5 Frequency of Contributions")
    score += {
        "One-time investment": 1,
        "Irregular / when possible": 0,
        "Monthly": 2,
        "Quarterly": 2,
        "Annually": 2
    }.get(q3_5, 0)

    q3_6 = answers.get("Q3.6 Average Amount per Contribution")
    score += {
        "Less than €100": 1,
        "€100 – €500": 2,
        "€500 – €2,000": 3,
        "€2,000 – €10,000": 4,
        "More than €10,000": 5
    }.get(q3_6, 0)

    Initial_inv = answers.get('Initial investment')
    Periodic_inv = answers.get('Periodic investments (amount in €)')

    # --- Final Score and Category ---
    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {"Mail": mail_, "score": score, "risk_profile": profile, "Initial investment": Initial_inv, "Periodic investments": Periodic_inv, "Frequency": q3_5}

def score_risk_profile_stm(answers: dict) -> dict:
    score = 0

    # --- Block 1: Experience and Knowledge ---
    q1_1 = answers.get("Q1.1 Level of General Education")
    score += {"No formal education": 0, "High school diploma": 1, "Bachelor's degree": 2, "Master's degree or higher": 3}.get(q1_1, 0)

    q1_2 = answers.get("Q1.2 Level of Financial Education")
    score += {"None": 0, "Basic": 1, "Intermediate": 2, "Advanced": 3, "Formal academic background in finance or economics": 4}.get(q1_2, 0)

    q1_3 = answers.get("Q1.3 Current Employment Status")
    score += {
        "Unemployed": 1, "Student": 1, "Employed part-time": 2,
        "Self-employed": 2, "Employed full-time": 3, "Retired": 2
    }.get(q1_3, 0)

    q1_4 = answers.get("Q1.4 Have You Ever Worked in a Finance-Related Role?")
    score += 3 if q1_4 == "Yes" else 0

    q1_5_assets = answers.get("Q1.5 Familiarity with Financial Assets", [])
    score += min(len(q1_5_assets), 5)

    q1_6 = answers.get("Q1.6 Previous Investment Experience")
    if q1_6 == "Yes":
        q1_6_1_level = answers.get("Q1.6.1.1")
        score += {"Low": 1, "Normal": 2, "High": 3}.get(q1_6_1_level, 0)

        q1_6_2_vol = answers.get("Q1.6.2.1.1")
        score += {
            "Less than €5,000": 1,
            "€5,000 – €25,000": 2,
            "€25,000 – €100,000": 3,
            "More than €100,000": 4
        }.get(q1_6_2_vol, 0)

        q1_6_2_ops = answers.get("Q1.6.2.1.2")
        score += {
            "Less than 5": 0,
            "5–20": 1,
            "21–50": 2,
            "More than 50": 3
        }.get(q1_6_2_ops, 0)

    # --- Block 2: Financial Situation ---
    q2_1 = answers.get("Q2.1 Average Monthly Income (After Taxes)")
    score += {
        "Less than €1,000": 1,
        "€1,000 – €2,000": 2,
        "€2,000 – €3,500": 3,
        "€3,500 – €5,000": 4,
        "More than €5,000": 5
    }.get(q2_1, 0)

    q2_2 = answers.get("Q2.2 Average Monthly Expenses")
    score += {
        "More than €5,000": 0,
        "€3,500 – €5,000": 1,
        "€2,000 – €3,500": 2,
        "€1,000 – €2,000": 3,
        "Less than €1,000": 4
    }.get(q2_2, 0)

    q2_3 = answers.get("Q2.3 Total Value of Savings")
    score += {
        "Less than €5,000": 1,
        "€5,000 – €25,000": 2,
        "€25,000 – €100,000": 3,
        "More than €100,000": 4
    }.get(q2_3, 0)

    q2_4 = answers.get("Q2.4 Estimated Net Worth")
    score += {
        "Negative or close to zero": 1,
        "€0 – €50,000": 2,
        "€50,000 – €200,000": 3,
        "€200,000 – €500,000": 4,
        "More than €500,000": 5
    }.get(q2_4, 0)

    q2_5 = answers.get("Q2.5 Maximum Acceptable Loss (% of Net Worth)")
    score += {
        "Less than 5%": 0,
        "5% – 10%": 1,
        "10% – 20%": 2,
        "20% – 35%": 3,
        "More than 35%": 4
    }.get(q2_5, 0)

    # --- Block 3: Financial Objectives and Risk Tolerance ---
    q3_1 = answers.get("Q3.1 Primary Financial Objectives")
    score += {
        "Capital preservation": 0,
        "Saving for a specific purchase": 1,
        "Building retirement savings": 2,
        "Generating regular income": 3,
        "Maximizing capital growth": 4
    }.get(q3_1, 0)

    q3_2 = answers.get("Q3.2 Investment Time Horizon")
    score += {
        "Less than 1 year": 0,
        "1 – 3 years": 1,
        "3 – 5 years": 2,
        "5 – 10 years": 3,
        "More than 10 years": 4
    }.get(q3_2, 0)

    q3_3 = answers.get("Q3.3 Risk and Return Preference")
    score += {
        "I prefer no risk, even if it means lower returns": 0,
        "I’m willing to take limited risks for modest returns": 2,
        "I accept moderate risk for potentially higher returns": 4,
        "I seek high returns and accept the possibility of significant losses": 5
    }.get(q3_3, 0)

    q3_4 = answers.get("Q3.4 Initial Investment Amount")
    score += {
        "Less than €1,000": 0,
        "€1,000 – €5,000": 1,
        "€5,000 – €20,000": 2,
        "€20,000 – €100,000": 3,
        "More than €100,000": 4
    }.get(q3_4, 0)

    q3_5 = answers.get("Q3.5 Frequency of Contributions")
    score += {
        "One-time investment": 1,
        "Irregular / when possible": 0,
        "Monthly": 2,
        "Quarterly": 2,
        "Annually": 2
    }.get(q3_5, 0)

    q3_6 = answers.get("Q3.6 Average Amount per Contribution")
    score += {
        "Less than €100": 1,
        "€100 – €500": 2,
        "€500 – €2,000": 3,
        "€2,000 – €10,000": 4,
        "More than €10,000": 5
    }.get(q3_6, 0)

    # --- Final Score and Category ---
    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {"score": score, "risk_profile": profile, "Frequency": q3_5}

def preprocess_form_response(raw_response: dict) -> dict:
    processed = {}

    # Campos tipo checkbox
    checkbox_keys = {
        "Q1.5 Familiarity with Financial Assets",
        "Q1.6.1.1 Types of Financial Assets Invested In"
    }

    # Renombrar claves para que coincidan con las esperadas por la función de scoring
    key_map = {
        "Q1.6.1 How Would You Rate Your Previous Investment Experience?": "Q1.6.1.1",
        "Q1.6.1.1 Types of Financial Assets Invested In": "Q1.6.1.1",
        "Q1.6.2.1.1 Total Amount Invested in Last 12 Months": "Q1.6.2.1.1",
        "Q1.6.2.1.2 Number of Transactions in Last 12 Months": "Q1.6.2.1.2"
    }

    for k, v in raw_response.items():
        new_key = key_map.get(k, k)
        if new_key in checkbox_keys:
            processed[new_key] = [s.strip() for s in v.split(',')]
        else:
            processed[new_key] = v.strip()

    return processed

def get_responses_as_dict(sheet_name: str) -> dict:
    # Define el alcance de permisos
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Autenticación con el archivo JSON
    creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\macar\OneDrive\Escritorio\credentials.json", scope)
    client = gspread.authorize(creds)

    # Abre la hoja de cálculo por nombre
    sheet = client.open(sheet_name).sheet1  # o usa .worksheet("Respuestas del formulario 1")

    # Obtiene todos los registros como lista de dicts
    all_responses = sheet.get_all_records()

    # Última respuesta
    latest_response = all_responses[-1]

    return all_responses, latest_response

def score_all_profiles(responses: list) -> list:
    """
    Aplica score_risk_profile a cada respuesta en la lista de respuestas.

    Parameters:
    - responses: list of dicts (cada dict representa una respuesta individual del formulario)

    Returns:
    - list of dicts con Mail, score y perfil de riesgo
    """
    return [score_risk_profile(response) for response in responses]

def score_risk_profile_ESP(answers: dict) -> dict:
    score = 0

    mail_ = answers.get('Dirección de correo electrónico')

    # --- Bloque 1: Experiencia y Conocimiento ---
    q1_1 = answers.get("Q1.1 Nivel de educación general")
    score += {"Sin estudios": 0, "Educación secundaria": 1, "Grado universitario": 2, "Máster o superior": 3}.get(q1_1, 0)

    q1_2 = answers.get("Q1.2 Nivel de educación financiera")
    score += {"Ninguno": 0, "Básico": 1, "Intermedio": 2, "Avanzado": 3, "Formación académica en finanzas o economía": 4}.get(q1_2, 0)

    q1_3 = answers.get("Q1.3 Situación laboral actual")
    score += {
        "Desempleado": 1, "Estudiante": 1, "Empleado a tiempo parcial": 2,
        "Autónomo": 2, "Empleado a tiempo completo": 3, "Jubilado": 2
    }.get(q1_3, 0)

    q1_4 = answers.get("Q1.4 ¿Ha trabajado alguna vez en un puesto relacionado con las finanzas?")
    score += 3 if q1_4 == "Sí" else 0

    q1_5 = answers.get("Q1.5 Familiaridad con activos financieros", [])
    score += min(len(q1_5), 5)

    q1_6 = answers.get("Q1.6 ¿Tiene experiencia previa en inversiones?")
    if q1_6 == "Sí":
        q1_6_1 = answers.get("Q1.6.1 ¿Cómo calificaría su experiencia previa en inversiones?")
        score += {"Baja": 1, "Normal": 2, "Alta": 3}.get(q1_6_1, 0)

        q1_6_2_1 = answers.get("Q1.6.2.1.1 Monto total invertido en los últimos 12 meses")
        score += {
            "Menos de €5.000": 1,
            "€5.000 – €25.000": 2,
            "€25.000 – €100.000": 3,
            "Más de €100.000": 4
        }.get(q1_6_2_1, 0)

        q1_6_2_2 = answers.get("Q1.6.2.1.2 Número de operaciones en los últimos 12 meses")
        score += {
            "Menos de 5": 0,
            "5–20": 1,
            "21–50": 2,
            "Más de 50": 3
        }.get(q1_6_2_2, 0)

    # --- Bloque 2: Situación financiera ---
    q2_1 = answers.get("Q2.1 Ingreso mensual promedio (después de impuestos)")
    score += {
        "Menos de €1.000": 1,
        "€1.000 – €2.000": 2,
        "€2.000 – €3.500": 3,
        "€3.500 – €5.000": 4,
        "Más de €5.000": 5
    }.get(q2_1, 0)

    q2_2 = answers.get("Q2.2 Gastos mensuales promedio")
    score += {
        "Más de €5.000": 0,
        "€3.500 – €5.000": 1,
        "€2.000 – €3.500": 2,
        "€1.000 – €2.000": 3,
        "Menos de €1.000": 4
    }.get(q2_2, 0)

    q2_3 = answers.get("Q2.3 Valor total de los ahorros")
    score += {
        "Menos de €5.000": 1,
        "€5.000 – €25.000": 2,
        "€25.000 – €100.000": 3,
        "Más de €100.000": 4
    }.get(q2_3, 0)

    q2_4 = answers.get("Q2.4 Valor neto estimado")
    score += {
        "Negativo o cercano a cero": 1,
        "€0 – €50.000": 2,
        "€50.000 – €200.000": 3,
        "€200.000 – €500.000": 4,
        "Más de €500.000": 5
    }.get(q2_4, 0)

    q2_5 = answers.get("Q2.5 Pérdida máxima aceptable (% del valor neto)")
    score += {
        "Menos de 5%": 0,
        "5% – 10%": 1,
        "10% – 20%": 2,
        "20% – 35%": 3,
        "Más de 35%": 4
    }.get(q2_5, 0)

    # --- Bloque 3: Objetivos y tolerancia al riesgo ---
    q3_1 = answers.get("Q3.1 Objetivos financieros principales")
    score += {
        "Preservación del capital": 0,
        "Ahorro para una compra específica": 1,
        "Construcción de ahorros para la jubilación": 2,
        "Generación de ingresos regulares": 3,
        "Maximizar el crecimiento del capital": 4
    }.get(q3_1, 0)

    q3_2 = answers.get("Q3.2 Horizonte temporal de inversión")
    score += {
        "Menos de 1 año": 0,
        "1 – 3 años": 1,
        "3 – 5 años": 2,
        "5 – 10 años": 3,
        "Más de 10 años": 4
    }.get(q3_2, 0)

    q3_3 = answers.get("Q3.3 Preferencia entre riesgo y rentabilidad")
    score += {
        "Prefiero no asumir riesgos, aunque signifique menores rendimientos": 0,
        "Estoy dispuesto a asumir riesgos limitados por rendimientos modestos": 2,
        "Acepto riesgos moderados por rendimientos potencialmente mayores": 4,
        "Busco altos rendimientos y acepto posibles pérdidas significativas": 5
    }.get(q3_3, 0)

    q3_4 = answers.get("Q3.4 Monto inicial de inversión")
    score += {
        "Menos de €1.000": 0,
        "€1.000 – €5.000": 1,
        "€5.000 – €20.000": 2,
        "€20.000 – €100.000": 3,
        "Más de €100.000": 4
    }.get(q3_4, 0)

    q3_5 = answers.get("Q3.5 Frecuencia de las aportaciones")
    score += {
        "Inversión única": 1,
        "Irregular / cuando sea posible": 0,
        "Mensual": 2,
        "Trimestral": 2,
        "Anual": 2
    }.get(q3_5, 0)

    q3_6 = answers.get("Q3.6 Monto promedio por aportación")
    score += {
        "Menos de €100": 1,
        "€100 – €500": 2,
        "€500 – €2.000": 3,
        "€2.000 – €10.000": 4,
        "Más de €10.000": 5
    }.get(q3_6, 0)

    # Recoger montos si están disponibles (por trazabilidad)
    Initial_inv = answers.get("Inversión inicial")
    Periodic_inv = answers.get("Inversiones periódicas (importe en €)\n")

    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {
        "Correo": mail_,
        "Puntuación": score,
        "Perfil de riesgo": profile,
        "Monto inicial": Initial_inv,
        "Monto periódico": Periodic_inv,
        "Frecuencia": q3_5
    }

def score_risk_profile_ESP_stm(answers: dict) -> dict:
    score = 0


    # --- Bloque 1: Experiencia y Conocimiento ---
    q1_1 = answers.get("Q1.1 Nivel de educación general")
    score += {"Sin estudios": 0, "Educación secundaria": 1, "Grado universitario": 2, "Máster o superior": 3}.get(q1_1, 0)

    q1_2 = answers.get("Q1.2 Nivel de educación financiera")
    score += {"Ninguno": 0, "Básico": 1, "Intermedio": 2, "Avanzado": 3, "Formación académica en finanzas o economía": 4}.get(q1_2, 0)

    q1_3 = answers.get("Q1.3 Situación laboral actual")
    score += {
        "Desempleado": 1, "Estudiante": 1, "Empleado a tiempo parcial": 2,
        "Autónomo": 2, "Empleado a tiempo completo": 3, "Jubilado": 2
    }.get(q1_3, 0)

    q1_4 = answers.get("Q1.4 ¿Ha trabajado alguna vez en un puesto relacionado con las finanzas?")
    score += 3 if q1_4 == "Sí" else 0

    q1_5 = answers.get("Q1.5 Familiaridad con activos financieros", [])
    score += min(len(q1_5), 5)

    q1_6 = answers.get("Q1.6 ¿Tiene experiencia previa en inversiones?")
    if q1_6 == "Sí":
        q1_6_1 = answers.get("Q1.6.1 ¿Cómo calificaría su experiencia previa en inversiones?")
        score += {"Baja": 1, "Normal": 2, "Alta": 3}.get(q1_6_1, 0)

        q1_6_2_1 = answers.get("Q1.6.2.1.1 Monto total invertido en los últimos 12 meses")
        score += {
            "Menos de €5.000": 1,
            "€5.000 – €25.000": 2,
            "€25.000 – €100.000": 3,
            "Más de €100.000": 4
        }.get(q1_6_2_1, 0)

        q1_6_2_2 = answers.get("Q1.6.2.1.2 Número de operaciones en los últimos 12 meses")
        score += {
            "Menos de 5": 0,
            "5–20": 1,
            "21–50": 2,
            "Más de 50": 3
        }.get(q1_6_2_2, 0)

    # --- Bloque 2: Situación financiera ---
    q2_1 = answers.get("Q2.1 Ingreso mensual promedio (después de impuestos)")
    score += {
        "Menos de €1.000": 1,
        "€1.000 – €2.000": 2,
        "€2.000 – €3.500": 3,
        "€3.500 – €5.000": 4,
        "Más de €5.000": 5
    }.get(q2_1, 0)

    q2_2 = answers.get("Q2.2 Gastos mensuales promedio")
    score += {
        "Más de €5.000": 0,
        "€3.500 – €5.000": 1,
        "€2.000 – €3.500": 2,
        "€1.000 – €2.000": 3,
        "Menos de €1.000": 4
    }.get(q2_2, 0)

    q2_3 = answers.get("Q2.3 Valor total de los ahorros")
    score += {
        "Menos de €5.000": 1,
        "€5.000 – €25.000": 2,
        "€25.000 – €100.000": 3,
        "Más de €100.000": 4
    }.get(q2_3, 0)

    q2_4 = answers.get("Q2.4 Valor neto estimado")
    score += {
        "Negativo o cercano a cero": 1,
        "€0 – €50.000": 2,
        "€50.000 – €200.000": 3,
        "€200.000 – €500.000": 4,
        "Más de €500.000": 5
    }.get(q2_4, 0)

    q2_5 = answers.get("Q2.5 Pérdida máxima aceptable (% del valor neto)")
    score += {
        "Menos de 5%": 0,
        "5% – 10%": 1,
        "10% – 20%": 2,
        "20% – 35%": 3,
        "Más de 35%": 4
    }.get(q2_5, 0)

    # --- Bloque 3: Objetivos y tolerancia al riesgo ---
    q3_1 = answers.get("Q3.1 Objetivos financieros principales")
    score += {
        "Preservación del capital": 0,
        "Ahorro para una compra específica": 1,
        "Construcción de ahorros para la jubilación": 2,
        "Generación de ingresos regulares": 3,
        "Maximizar el crecimiento del capital": 4
    }.get(q3_1, 0)

    q3_2 = answers.get("Q3.2 Horizonte temporal de inversión")
    score += {
        "Menos de 1 año": 0,
        "1 – 3 años": 1,
        "3 – 5 años": 2,
        "5 – 10 años": 3,
        "Más de 10 años": 4
    }.get(q3_2, 0)

    q3_3 = answers.get("Q3.3 Preferencia entre riesgo y rentabilidad")
    score += {
        "Prefiero no asumir riesgos, aunque signifique menores rendimientos": 0,
        "Estoy dispuesto a asumir riesgos limitados por rendimientos modestos": 2,
        "Acepto riesgos moderados por rendimientos potencialmente mayores": 4,
        "Busco altos rendimientos y acepto posibles pérdidas significativas": 5
    }.get(q3_3, 0)

    q3_4 = answers.get("Q3.4 Monto inicial de inversión")
    score += {
        "Menos de €1.000": 0,
        "€1.000 – €5.000": 1,
        "€5.000 – €20.000": 2,
        "€20.000 – €100.000": 3,
        "Más de €100.000": 4
    }.get(q3_4, 0)

    q3_5 = answers.get("Q3.5 Frecuencia de las aportaciones")
    score += {
        "Inversión única": 1,
        "Irregular / cuando sea posible": 0,
        "Mensual": 2,
        "Trimestral": 2,
        "Anual": 2
    }.get(q3_5, 0)

    q3_6 = answers.get("Q3.6 Monto promedio por aportación")
    score += {
        "Menos de €100": 1,
        "€100 – €500": 2,
        "€500 – €2.000": 3,
        "€2.000 – €10.000": 4,
        "Más de €10.000": 5
    }.get(q3_6, 0)

    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {
        "score": score,
        "risk_profile": profile,
        "Frecuencia": q3_5
    }

def score_all_profiles_ESP(responses: list) -> list:
    """
    Aplica score_risk_profile a cada respuesta en la lista de respuestas.

    Parameters:
    - responses: list of dicts (cada dict representa una respuesta individual del formulario)

    Returns:
    - list of dicts con Mail, score y perfil de riesgo
    """
    return [score_risk_profile_ESP(response) for response in responses]

# ─────────────────────────────────────────────────────────────────────
#  Función de scoring — Español v4
# ─────────────────────────────────────────────────────────────────────
def score_risk_profile_ESP_stm_v2(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula la puntuación y el perfil de riesgo a partir de las
    respuestas del usuario (versión español).

    Parameters
    ----------
    answers : dict
        Diccionario { "Q1.1": respuesta, ... }

    Returns
    -------
    dict
        {"score": int, "risk_profile": str, "Frecuencia": str | None}
    """

    # Helper seguro: devuelve None si la clave no existe
    get = lambda k, default=None: answers.get(k, default)

    score = 0

    # ───── Bloque 1 · Conocimiento / Experiencia ────────────────────
    score += {
        "Sin estudios": 0, "Educación secundaria": 1,
        "Grado universitario": 2, "Máster o superior": 3
    }.get(get("Q1.1"), 0)

    score += {
        "Ninguno": 0, "Básico": 1, "Intermedio": 2,
        "Avanzado": 3, "Formación académica en finanzas o economía": 4
    }.get(get("Q1.2"), 0)

    score += {
        "Desempleado": 1, "Estudiante": 1, "Empleado a tiempo parcial": 2,
        "Autónomo": 2, "Empleado a tiempo completo": 3, "Jubilado": 2
    }.get(get("Q1.3"), 0)

    score += 3 if get("Q1.4") == "Sí" else 0

    # Q1.5 es multiselect (lista)
    score += min(len(get("Q1.5", [])), 5)          # 1 punto por activo (máx 5)

    # Experiencia inversora
    if get("Q1.6") == "Sí":
        score += {"Baja": 1, "Normal": 2, "Alta": 3}.get(get("Q1.6.1"), 0)

        score += {
            "Menos de €5.000": 1, "€5.000 – €25.000": 2,
            "€25.000 – €100.000": 3, "Más de €100.000": 4
        }.get(get("Q1.6.2.1"), 0)

        score += {
            "Menos de 5": 0, "5–20": 1, "21–50": 2, "Más de 50": 3
        }.get(get("Q1.6.2.2"), 0)

    # ───── Bloque 2 · Situación financiera ──────────────────────────
    score += {
        "Menos de €1.000": 1,  "€1.000 – €2.000": 2,
        "€2.000 – €3.500": 3, "€3.500 – €5.000": 4, "Más de €5.000": 5
    }.get(get("Q2.1"), 0)

    score += {
        "Más de €5.000": 0, "€3.500 – €5.000": 1,
        "€2.000 – €3.500": 2, "€1.000 – €2.000": 3, "Menos de €1.000": 4
    }.get(get("Q2.2"), 0)

    score += {
        "Menos de €5.000": 1, "€5.000 – €25.000": 2,
        "€25.000 – €100.000": 3, "Más de €100.000": 4
    }.get(get("Q2.3"), 0)

    score += {
        "Negativo o cercano a cero": 1, "€0 – €50.000": 2,
        "€50.000 – €200.000": 3, "€200.000 – €500.000": 4,
        "Más de €500.000": 5
    }.get(get("Q2.4"), 0)

    score += {
        "Menos de 5%": 0, "5% – 10%": 1, "10% – 20%": 2,
        "20% – 35%": 3, "Más de 35%": 4
    }.get(get("Q2.5"), 0)

    # ───── Bloque 3 · Objetivos y tolerancia ────────────────────────
    score += {
        "Preservación del capital": 0, "Ahorro para una compra específica": 1,
        "Construcción de ahorros para la jubilación": 2,
        "Generación de ingresos regulares": 3,
        "Maximizar el crecimiento del capital": 4
    }.get(get("Q3.1"), 0)

    score += {
        "Menos de 1 año": 0, "1 – 3 años": 1, "3 – 5 años": 2,
        "5 – 10 años": 3, "Más de 10 años": 4
    }.get(get("Q3.2"), 0)

    score += {
        "Prefiero no asumir riesgos, aunque signifique menores rendimientos": 0,
        "Estoy dispuesto a asumir riesgos limitados por rendimientos modestos": 2,
        "Acepto riesgos moderados por rendimientos potencialmente mayores": 4,
        "Busco altos rendimientos y acepto posibles pérdidas significativas": 5
    }.get(get("Q3.3"), 0)

    score += {
        "Menos de €1.000": 0, "€1.000 – €5.000": 1,
        "€5.000 – €20.000": 2, "€20.000 – €100.000": 3,
        "Más de €100.000": 4
    }.get(get("Q3.4"), 0)

    freq_aport = get("Q3.5")          # lo devolvemos al usuario
    score += {
        "Inversión única": 1, "Irregular / cuando sea posible": 0,
        "Mensual": 2, "Trimestral": 2, "Anual": 2
    }.get(freq_aport, 0)

    score += {
        "Menos de €100": 1, "€100 – €500": 2,
        "€500 – €2.000": 3, "€2.000 – €10.000": 4,
        "Más de €10.000": 5
    }.get(get("Q3.6"), 0)

    # ───── Asignación de perfil (umbrales orientativos) ─────────────
    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {
        "score":         int(score),
        "risk_profile":  profile,
        "Frecuencia":    freq_aport,
    }

# ─────────────────────────────────────────────────────────────────────
#  Risk‑scoring – English v4 (for Streamlit)
# ─────────────────────────────────────────────────────────────────────
def score_risk_profile_EN_stm_v2(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute score & risk profile for the English questionnaire.

    Parameters
    ----------
    answers : dict
        {"Q1.1": <answer>, ...}  — keys must match the short IDs.

    Returns
    -------
    dict
        {"score": int, "risk_profile": str, "Frequency": str | None}
    """
    # safe getter
    g = lambda k, d=None: answers.get(k, d)

    score = 0

    # ───── Block 1 · Experience / Knowledge ──────────────────────────
    score += {
        "No formal education": 0, "High school diploma": 1,
        "Bachelor's degree": 2, "Master's degree or higher": 3
    }.get(g("Q1.1"), 0)

    score += {
        "None": 0, "Basic": 1, "Intermediate": 2,
        "Advanced": 3, "Formal academic background in finance or economics": 4
    }.get(g("Q1.2"), 0)

    score += {
        "Unemployed": 1, "Student": 1, "Employed part-time": 2,
        "Self-employed": 2, "Employed full-time": 3, "Retired": 2
    }.get(g("Q1.3"), 0)

    score += 3 if g("Q1.4") == "Yes" else 0

    score += min(len(g("Q1.5", [])), 5)   # multiselect

    if g("Q1.6") == "Yes":
        score += {"Low": 1, "Normal": 2, "High": 3}.get(g("Q1.6.1"), 0)

        score += {
            "Less than €5,000": 1, "€5,000 – €25,000": 2,
            "€25,000 – €100,000": 3, "More than €100,000": 4
        }.get(g("Q1.6.2.1"), 0)

        score += {
            "Less than 5": 0, "5–20": 1, "21–50": 2, "More than 50": 3
        }.get(g("Q1.6.2.2"), 0)

    # ───── Block 2 · Financial situation ─────────────────────────────
    score += {
        "Less than €1,000": 1, "€1,000 – €2,000": 2,
        "€2,000 – €3,500": 3, "€3,500 – €5,000": 4,
        "More than €5,000": 5
    }.get(g("Q2.1"), 0)

    score += {
        "More than €5,000": 0, "€3,500 – €5,000": 1,
        "€2,000 – €3,500": 2, "€1,000 – €2,000": 3,
        "Less than €1,000": 4
    }.get(g("Q2.2"), 0)

    score += {
        "Less than €5,000": 1, "€5,000 – €25,000": 2,
        "€25,000 – €100,000": 3, "More than €100,000": 4
    }.get(g("Q2.3"), 0)

    score += {
        "Negative or close to zero": 1, "€0 – €50,000": 2,
        "€50,000 – €200,000": 3, "€200,000 – €500,000": 4,
        "More than €500,000": 5
    }.get(g("Q2.4"), 0)

    score += {
        "Less than 5%": 0, "5% – 10%": 1, "10% – 20%": 2,
        "20% – 35%": 3, "More than 35%": 4
    }.get(g("Q2.5"), 0)

    # ───── Block 3 · Objectives & Risk tolerance ─────────────────────
    score += {
        "Capital preservation": 0,
        "Saving for a specific purchase": 1,
        "Building retirement savings": 2,
        "Generating regular income": 3,
        "Maximizing capital growth": 4
    }.get(g("Q3.1"), 0)

    score += {
        "Less than 1 year": 0, "1 – 3 years": 1,
        "3 – 5 years": 2, "5 – 10 years": 3,
        "More than 10 years": 4
    }.get(g("Q3.2"), 0)

    score += {
        "I prefer no risk, even if it means lower returns": 0,
        "I’m willing to take limited risks for modest returns": 2,
        "I accept moderate risk for potentially higher returns": 4,
        "I seek high returns and accept the possibility of significant losses": 5
    }.get(g("Q3.3"), 0)

    score += {
        "Less than €1,000": 0, "€1,000 – €5,000": 1,
        "€5,000 – €20,000": 2, "€20,000 – €100,000": 3,
        "More than €100,000": 4
    }.get(g("Q3.4"), 0)

    freq = g("Q3.5")
    score += {
        "One-time investment": 1, "Irregular / when possible": 0,
        "Monthly": 2, "Quarterly": 2, "Annually": 2
    }.get(freq, 0)

    score += {
        "Less than €100": 1, "€100 – €500": 2,
        "€500 – €2,000": 3, "€2,000 – €10,000": 4,
        "More than €10,000": 5
    }.get(g("Q3.6"), 0)

    # ───── Map score → risk profile ──────────────────────────────────
    if score <= 10:
        profile = "Defensive"
    elif score <= 20:
        profile = "Conservative"
    elif score <= 30:
        profile = "Cautious"
    elif score <= 40:
        profile = "Equilibrated"
    elif score <= 50:
        profile = "Decided"
    elif score <= 60:
        profile = "Daring"
    else:
        profile = "Aggressive"

    return {
        "score":        int(score),
        "risk_profile": profile,
        "Frequency":    freq,
    }

































