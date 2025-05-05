````markdown
# 🤖 Streamlit Robo-Advisor – Risk Profiling & Investment Simulator

This is an interactive web application that helps users determine their investment **risk profile** and simulate the expected evolution of their portfolio based on historical performance.

It is built entirely in **Python** using **Streamlit**, and is designed for easy use by investors with different levels of financial knowledge.

---

## 🧠 Features

- 🧮 **Risk Profile Questionnaire** (7 profiles: Defensive to Aggressive)
- 🧾 **Personalized result summary** with scores and investment focus
- 📊 **Investment simulator** with:
  - Initial capital and periodic contributions
  - Simulated growth using backtesting logic
- 📄 **PDF export** (without charts for now)
- 🌍 **Language support**: English and Spanish

---

## 🚀 How to use the app

You can try the app online:

👉 [Click here to open the app](https://your-app-link.streamlit.app) ← (replace after deploy)

Or run it locally:

```bash
git clone https://github.com/Mario162003/streamlit-roboadvisor.git
cd streamlit-roboadvisor
pip install -r requirements.txt
streamlit run app.py
````

---

## 📁 Project structure

```bash
streamlit_roboadvisor/
│
├── app.py                 ← Main Streamlit script
├── requirements.txt       ← Project dependencies
├── modules/               ← Helper modules (scoring, portfolio, etc.)
└── README.md              ← You're reading it
```

---

## 🛠️ Built with

* Python 3.11
* Streamlit
* Plotly / Altair
* YFinance
* Scikit-learn
* PDFKit + Jinja2

```

---

## ✅ Ahora sí: veamos el paso siguiente

### 2. 🌐 Ve a [https://streamlit.io/cloud](https://streamlit.io/cloud)

Haz lo siguiente:

1. Inicia sesión con tu cuenta de GitHub
2. Haz clic en **"New app"**
3. En el desplegable, selecciona tu nuevo repo:  
   `Mario162003/streamlit-roboadvisor`
4. Asegúrate de rellenar estos campos:

| Campo              | Valor            |
|--------------------|------------------|
| **Branch**         | `main`           |
| **Main file path** | `app.py`         |

5. Haz clic en **Deploy**

En unos segundos tendrás una app pública con una URL como:

```

[https://mario162003-streamlit-roboadvisor.streamlit.app](https://mario162003-streamlit-roboadvisor.streamlit.app)

```
