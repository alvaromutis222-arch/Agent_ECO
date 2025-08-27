# 📈 Agente Económico con LLM + Streamlit

Este proyecto implementa un **Agente Económico** que utiliza **modelos de lenguaje (LLM)** gratuitos (Groq u Ollama) y una interfaz web desarrollada con **Streamlit**.  

El agente es capaz de:
- Analizar tendencias económicas.
- Explicar conceptos financieros con definiciones, fórmulas y ejemplos.
- Simular escenarios de mercado (inflación, CAGR, series sintéticas).
- Interactuar en un chat especializado en economía.

---

## 🚀 Tecnologías utilizadas
- **Streamlit** → interfaz web.
- **Groq (Llama 3.x)** → LLM en la nube (API gratuita con plan básico).
- **Ollama (local)** → alternativa para correr modelos LLaMA en tu máquina.
- **LangChain Community** → integración con Ollama.
- **Pandas / Numpy / Matplotlib** → análisis y visualización de datos.

---

## 📂 Estructura del proyecto
```
Agent_ECO/
├── app.py             # Aplicación principal en Streamlit
├── requirements.txt   # Dependencias necesarias
└── README.md          # Documentación del proyecto
```

---

## ⚙️ Instalación y uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/alvaromutis222/Agent_ECO.git
cd Agent_ECO
```

### 2. Crear entorno virtual (opcional pero recomendado)

#### 🔹 Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 🔹 Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales
Si usas **Groq**:
```bash
# Linux / Mac
export GROQ_API_KEY="tu_api_key"

# Windows (PowerShell)
$Env:GROQ_API_KEY="tu_api_key"
```

Si usas **Ollama**:
- Instala [Ollama](https://ollama.ai).
- Descarga el modelo:
```bash
ollama run llama3.1:8b-instruct
```

### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en: [http://localhost:8501](http://localhost:8501)

---

## 🧩 Funcionalidades principales
1. **💬 Chat**: conversación interactiva con el agente económico.
2. **📊 Análisis**: carga de CSV con precios y detección de tendencias.
3. **🧪 Simulador**: cálculo de inflación acumulada, CAGR y generación de series sintéticas.
4. **🎤 Guion de sustentación**: guía rápida para tu presentación (≤5 min).

---

## 🌐 Despliegue en Streamlit Community Cloud
1. Sube tu repositorio a GitHub.
2. Entra en [Streamlit Cloud](https://share.streamlit.io/).
3. Conecta tu repositorio y selecciona `app.py` como archivo principal.
4. Agrega tu `GROQ_API_KEY` en **Secrets**:
   ```
   GROQ_API_KEY="tu_api_key_real"
   ```

---

## 📌 Nota importante
Este proyecto tiene fines **educativos y demostrativos**.  
⚠️ **No constituye asesoría financiera profesional.**

---

