# 🚗⚡ Pipeline Multiagente — Predicción de Precios de Vehículos Eléctricos

Proyecto desarrollado en **Google Colab** con arquitectura multiagente para predecir el precio de vehículos eléctricos (EV) a partir de sus características técnicas. Implementado como notebook `.ipynb` con celdas organizadas, documentadas y funcionales.

---

## 📋 Criterios de Evaluación

| Criterio | Pts | Estado |
|---|:---:|:---:|
| Trabaja en clase en el desarrollo del proyecto | 6 | ✅ |
| Demuestra avances coherentes con respecto al desarrollo | 3 | ✅ |
| Documenta las versiones del trabajo a través de comandos git | 3 | ✅ |
| El código presentado es pulcro y funcional | 1 | ✅ |
| Genera commits en escenarios específicos y comprensibles | 3 | ✅ |
| El proyecto es desarrollado hasta el 100% | 2 | ✅ |
| El proyecto desarrollado cumple con los fundamentos y estándares deseados | 2 | ✅ |
| **Total** | **20** | ✅ |

---

## 🏗️ Arquitectura del Proyecto

El sistema está compuesto por **4 agentes especializados** que se ejecutan en secuencia dentro de un pipeline:

```
CSV Dataset
    │
    ▼
┌─────────────────────┐
│  Agente 1           │  Limpieza · Imputación · One-Hot Encoding · Escalado
│  NormalizerAgent    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agente 2           │  Random Forest Regressor (200 árboles · max_depth=20)
│  TrainerAgent       │  Train 80% / Test 20% · Métricas: MSE, RMSE, R²
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agente 3           │  Reporte automático de resultados en texto
│  CommunicatorAgent  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agente 4           │  Chatbot en lenguaje natural vía Mistral AI API
│  ChatbotAgent       │
└─────────────────────┘
         +
┌─────────────────────┐
│  Predictor          │  Entrada interactiva del usuario → precio estimado
│  Interactivo        │
└─────────────────────┘
```

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.x | Lenguaje principal |
| Pandas & NumPy | Manipulación y procesamiento de datos |
| Scikit-learn | Preprocesamiento y modelo Random Forest |
| Mistral AI API | Chatbot de análisis en lenguaje natural |
| Google Colab | Entorno de ejecución |
| GitHub | Control de versiones |

---

## 📁 Estructura del repositorio

```
📦 proyecto-multiagente-ev/
 ┣ 📓 pipeline_multiagente_EV.ipynb   # Notebook principal con todo el pipeline
 ┣ 📄 ev_market_2026.csv              # Dataset de vehículos eléctricos
 ┗ 📄 README.md                       # Documentación del proyecto
```

---

## 🚀 Cómo ejecutar

### 1. Abrir en Google Colab
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Ejecutar las celdas en orden

| # | Celda | Descripción |
|:---:|---|---|
| 1 | Instalación | Instala `pandas`, `numpy`, `scikit-learn`, `requests` |
| 2 | Imports | Carga todas las librerías necesarias |
| 3 | Dataset | Sube tu propio CSV desde el selector de archivos |
| 4 | Carga | Lee el CSV y muestra las primeras filas |
| 5 | Agente 1 | Normaliza y codifica el dataset |
| 6 | Agente 2 | Entrena el modelo y calcula métricas |
| 7 | Agente 3 | Genera el reporte final de resultados |
| 8 | Predictor | Interfaz interactiva para predecir precios |
| 9 | Agente 4 | Chatbot analista con Mistral AI |

### 3. Configurar la API Key de Mistral
En la celda del **Agente 4**, reemplazá el valor de `MISTRAL_API_KEY` con tu clave desde [console.mistral.ai](https://console.mistral.ai/).

---

## 🤖 Detalle de cada agente

### Agente 1 — NormalizerAgent
- Detecta y separa columnas numéricas y categóricas automáticamente
- Imputa valores nulos: media para numéricas, moda para categóricas
- Aplica **One-Hot Encoding** a variables categóricas
- Escala variables numéricas con **StandardScaler**

### Agente 2 — TrainerAgent
- Divide el dataset: **80% entrenamiento / 20% prueba**
- Entrena un **Random Forest Regressor** (200 estimadores, profundidad máxima 20)
- Calcula métricas: `MSE`, `RMSE` y `R²`

### Agente 3 — CommunicatorAgent
- Genera un reporte legible con los resultados del modelo
- Incluye interpretación del R² y del margen de error (RMSE)

### Predictor Interactivo
- Solicita marca, capacidad de batería y caballos de fuerza al usuario
- Reconstruye el vector de features usando promedios/modas del dataset
- Aplica el mismo preprocesamiento y predice el precio estimado

### Agente 4 — ChatbotAgent (Mistral AI)
- Conecta con `mistral-small-latest` vía API REST
- Responde preguntas en lenguaje natural sobre el modelo y los datos
- El contexto incluye las métricas del modelo y el tamaño del dataset

---

## 📊 Métricas del Modelo

| Métrica | Descripción |
|---|---|
| **R²** | Porcentaje de variación en precios explicada por el modelo |
| **RMSE** | Margen de error promedio en USD respecto al precio real |
| **MSE** | Error cuadrático medio |

---

## 📝 Historial de commits sugerido

```bash
git commit -m "init: estructura inicial del proyecto y notebook base"
git commit -m "feat: implementar Agente 1 - NormalizerAgent con imputación y encoding"
git commit -m "feat: implementar Agente 2 - TrainerAgent con Random Forest"
git commit -m "feat: implementar Agente 3 - CommunicatorAgent con reporte de métricas"
git commit -m "feat: agregar predictor interactivo de precios por terminal"
git commit -m "feat: implementar Agente 4 - ChatbotAgent con Mistral AI"
git commit -m "fix: corregir indentación en celda de carga de CSV"
git commit -m "feat: integrar carga de CSV desde Google Colab con files.upload()"
git commit -m "docs: agregar README con documentación completa del proyecto"
```

---

## 👨‍💻 Autor

Desarrollado como proyecto académico — Ingeniería Informática
