# app.py
# -*- coding: utf-8 -*-
"""
Agente Económico (Streamlit + Groq/Ollama + utilidades locales)
Autor: Tu_Nombre
Licencia: MIT
---------------------------------------------------------------
• Frontend: Streamlit
• LLM: Groq (Llama 3.x) o local vía Ollama (opcional)
• Herramientas gratuitas: sin dependencias de pago obligatorias
• Repositorio: súbelo a GitHub tal cual (app.py, requirements.txt, README opcional)
---------------------------------------------------------------
Cómo ejecutar localmente
1) Python 3.10+ recomendado.
2) (Opcional) Crear y activar un entorno virtual.
3) Instala dependencias:  pip install -r requirements.txt
4a) Si usas Groq: export GROQ_API_KEY="tu_api_key"
    (crea cuenta en https://console.groq.com/; hay plan gratuito)
4b) Si usas Ollama local:  instala Ollama y ejecuta:  ollama run llama3.1:8b-instruct
5) Inicia la app:  streamlit run app.py
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------- Opcional: Groq (nube, gratuito) ----------
try:
    from groq import Groq  # SDK oficial
except Exception:
    Groq = None  # Permite correr sin Groq instalado

# --------- Opcional: LangChain + Ollama (local) ------
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None


# =========================
#   Configuración Streamlit
# =========================
st.set_page_config(page_title="Agente Económico — Streamlit", page_icon="📈", layout="wide")
st.title("📈 Agente Económico (LLM + Streamlit)")

with st.expander("ℹ️ Instrucciones rápidas", expanded=False):
    st.markdown("""
**Objetivo:** Analizar tendencias económicas, explicar conceptos financieros y simular escenarios de mercado.
- **LLM gratuito**: usa **Groq** (Llama 3.x) con tu `GROQ_API_KEY` o **Ollama** local.
- **Pestañas**:
  1. **Chat** – conversación con el agente.
  2. **Análisis** – carga CSV y genera tendencias / señales.
  3. **Simulador** – escenarios de inflación, CAGR y series sintéticas.
    """)

# =========================
#  Estado de Sesión
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "provider_ok" not in st.session_state:
    st.session_state.provider_ok = False


# =========================
#   Sidebar (Configuración)
# =========================
st.sidebar.header("⚙️ Configuración")

provider = st.sidebar.selectbox(
    "Proveedor del modelo",
    ["Groq (nube)", "Ollama (local)"],
    help="Usa Groq (recomendado) o un modelo compatible con Ollama en tu equipo."
)

temperature = st.sidebar.slider("Temperatura (creatividad)", 0.0, 1.2, 0.2, 0.1)
system_prompt = st.sidebar.text_area(
    "🔧 'Lenguaje Pro' (System Prompt)",
    value=(
        "Eres **EcoMentor**, un agente económico hispanohablante, claro y conciso. "
        "Respondes con datos, fórmulas y ejemplos. Cuando expliques un concepto, incluye "
        "una mini definición, una fórmula (si aplica) y un ejemplo numérico breve. "
        "Si el usuario pide proyecciones, aclara que son aproximaciones educativas."
    ),
    height=180,
    help="Personaliza el comportamiento experto del agente."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Credenciales")
groq_key = st.sidebar.text_input("GROQ_API_KEY", type="password", value=os.getenv("GROQ_API_KEY", ""))
ollama_model = st.sidebar.text_input("Modelo Ollama", value="llama3.1:8b-instruct")

st.sidebar.markdown("---")
st.sidebar.caption("Hecho para cumplir los requisitos: Streamlit + modelos gratuitos (Groq/Ollama) + utilidades de análisis.")

# =========================
#   Cliente LLM
# =========================
@dataclass
class LLMConfig:
    provider: str
    temperature: float
    system_prompt: str
    groq_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    ollama_model: str = "llama3.1:8b-instruct"


def make_client(cfg: LLMConfig):
    if cfg.provider == "groq":
        if Groq is None:
            raise RuntimeError("El paquete 'groq' no está instalado.")
        if not cfg.groq_key:
            raise RuntimeError("Falta GROQ_API_KEY.")
        client = Groq(api_key=cfg.groq_key)

        def _call_llm(messages: List[Dict[str, str]]) -> str:
            resp = client.chat.completions.create(
                model=cfg.groq_model,
                temperature=cfg.temperature,
                messages=messages,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""

        return _call_llm

    elif cfg.provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("LangChain Community no está instalado o faltan extras de Ollama.")
        chat = ChatOllama(model=cfg.ollama_model, temperature=cfg.temperature)

        def _call_llm(messages: List[Dict[str, str]]) -> str:
            sys = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            last_user = [m["content"] for m in messages if m["role"] == "user"]
            prompt = (sys + "\n\nUsuario:\n" + (last_user[-1] if last_user else "")).strip()
            out = chat.invoke(prompt)
            return out.content if hasattr(out, "content") else str(out)

        return _call_llm

    else:
        raise ValueError("Proveedor inválido")


cfg = LLMConfig(
    provider="groq" if provider.startswith("Groq") else "ollama",
    temperature=float(temperature),
    system_prompt=system_prompt,
    groq_key=groq_key,
    ollama_model=ollama_model,
)

call_llm = None
try:
    call_llm = make_client(cfg)
    st.session_state.provider_ok = True
except Exception as e:
    st.session_state.provider_ok = False
    st.sidebar.error(f"No se pudo inicializar el modelo: {e}")

# =========================
#   Utilidades Económicas
# =========================
def inflacion_acumulada(tasas_mensuales: List[float]) -> float:
    factor = 1.0
    for t in tasas_mensuales:
        factor *= (1.0 + t/100.0)
    return (factor - 1.0) * 100.0


def cagr(valor_inicial: float, valor_final: float, años: float) -> float:
    if valor_inicial <= 0 or valor_final <= 0 or años <= 0:
        return float("nan")
    return ((valor_final / valor_inicial) ** (1.0 / años) - 1.0) * 100.0


def serie_sintetica(precio0=100.0, dias=252, drift=0.05, volatilidad=0.2, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0/252.0
    shocks = rng.normal((drift - 0.5*volatilidad**2)*dt, volatilidad*np.sqrt(dt), size=dias)
    precios = precio0 * np.exp(np.cumsum(shocks))
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=dias)
    return pd.DataFrame({"precio": precios}, index=fechas)


def analizar_tendencias(df: pd.DataFrame, col="precio", ventana_corta=20, ventana_larga=50) -> pd.DataFrame:
    out = df.copy()
    out["media_corta"] = out[col].rolling(ventana_corta).mean()
    out["media_larga"] = out[col].rolling(ventana_larga).mean()
    out["senal"] = np.where(out["media_corta"] > out["media_larga"], 1, -1)
    out["ret_diario"] = out[col].pct_change()
    out["ret_estrategia"] = out["ret_diario"] * out["senal"].shift(1)
    return out


def plot_series(df: pd.DataFrame, col="precio"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df[col], label=col)
    if "media_corta" in df.columns:
        ax.plot(df.index, df["media_corta"], label="media_corta")
    if "media_larga" in df.columns:
        ax.plot(df.index, df["media_larga"], label="media_larga")
    ax.set_title("Serie y medias móviles")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)


# =========================
#   Tabs principales
# =========================
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Análisis", "🧪 Simulador"])

# ------------- Tab 1: Chat ---------------
with tab1:
    st.subheader("Chat con el agente")
    if st.session_state.provider_ok and call_llm is not None:
        base_messages = [{"role": "system", "content": cfg.system_prompt}]
        for m in st.session_state.messages:
            base_messages.append(m)

        user_input = st.chat_input("Escribe tu pregunta económica aquí…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            chat_messages = base_messages + [{"role": "user", "content": user_input}]
            with st.spinner("Pensando…"):
                try:
                    answer = call_llm(chat_messages)
                except Exception as e:
                    answer = f"Error al llamar al modelo: {e}"
            st.session_state.messages.append({"role": "assistant", "content": answer})

        for m in st.session_state.messages:
            if m["role"] == "user":
                with st.chat_message("user"):
                    st.write(m["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(m["content"])
    else:
        st.warning("Configura y verifica el proveedor del modelo en el sidebar.")

# ------------- Tab 2: Análisis ---------------
with tab2:
    st.subheader("Cargar datos y obtener señales")
    uploaded = st.file_uploader("Sube un CSV con una columna 'precio' o 'close'", type=["csv"])
    ventana_corta = st.number_input("Ventana corta (días)", 5, 120, 20, 1)
    ventana_larga = st.number_input("Ventana larga (días)", 10, 300, 50, 1)

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            col = "precio" if "precio" in df.columns else ("close" if "close" in df.columns else df.columns[1])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            else:
                df.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
            df = df[[col]].rename(columns={col: "precio"})
        except Exception as e:
            st.error(f"Error al leer CSV: {e}")

    if df is not None:
        result = analizar_tendencias(df, "precio", int(ventana_corta), int(ventana_larga))
        st.dataframe(result.tail(10))
        plot_series(result, "precio")

        car = (1.0 + result["ret_estrategia"].fillna(0)).prod() - 1.0
        st.metric("Rentabilidad estrategia (periodo cargado)", f"{car*100:.2f}%")

# ------------- Tab 3: Simulador ---------------
with tab3:
    st.subheader("Escenarios rápidos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Inflación acumulada**")
        tasas_text = st.text_input("Tasas mensuales (%, separadas por coma)", "0.5,0.6,0.4,0.7,0.5,0.4,0.3,0.6,0.5,0.4,0.3,0.5")
        try:
            tasas = [float(t.strip()) for t in tasas_text.split(",") if t.strip()]
            inflac = inflacion_acumulada(tasas)
            st.write(f"**Inflación acumulada:** {inflac:.2f}%")
        except Exception:
            st.write("Ingresa valores numéricos válidos.")

        st.markdown("---")
        st.markdown("**CAGR**")
        vi = st.number_input("Valor inicial", 0.01, 1e12, 1000.0, step=100.0, format="%.2f")
        vf = st.number_input("Valor final", 0.01, 1e12, 1500.0, step=100.0, format="%.2f")
        anios = st.number_input("Años", 0.1, 100.0, 3.0, step=0.1)
        st.write(f"**CAGR:** {cagr(vi, vf, anios):.2f}%")

    with col2:
        st.markdown("**Serie sintética y medias**")
        dias = st.slider("Días hábiles", 60, 756, 252, 6)
        drift = st.slider("Drift anual", -0.5, 0.8, 0.05, 0.01)
        vol = st.slider("Volatilidad anual", 0.01, 1.2, 0.2, 0.01)
        df_syn = serie_sintetica(dias=int(dias), drift=float(drift), volatilidad=float(vol))
        res = analizar_tendencias(df_syn)
        plot_series(res)

# Footer
st.markdown("---")
st.caption("© 2025 — Agente Económico (demo educativa). No constituye asesoría financiera.")
