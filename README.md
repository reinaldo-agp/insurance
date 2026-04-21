# 🏥 Health Insurance Cost Prediction

Modelo de Machine Learning para predecir el costo de seguros de salud individuales a partir de variables demográficas y de estilo de vida. Incluye análisis exploratorio completo, ingeniería de features y comparación de modelos.

## 🎯 Objetivo del proyecto

Construir un modelo predictivo capaz de estimar el costo de un seguro de salud basándose en características del asegurado, identificando los factores que más influyen en el precio final.

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Análisis de datos | Pandas · NumPy |
| Visualizaciones | Matplotlib · Seaborn |
| Machine Learning | Scikit-learn |
| Entorno | Jupyter Notebook |
| Lenguaje | Python 3.10+ |

## 📋 Variables del dataset

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `age` | Edad del asegurado | Numérica |
| `sex` | Género | Categórica |
| `bmi` | Índice de masa corporal | Numérica |
| `children` | Número de hijos/dependientes | Numérica |
| `smoker` | ¿Es fumador? | Categórica |
| `region` | Región de residencia (EE.UU.) | Categórica |
| `charges` | Costo del seguro (target) | Numérica |

## 🔬 Pipeline del proyecto

```
1. Análisis exploratorio (EDA)
   - Distribución de variables
   - Correlaciones y relaciones con el target
   - Detección de outliers
        ↓
2. Preprocesamiento
   - Encoding de variables categóricas
   - Escalado de features numéricas
   - Tratamiento de outliers
        ↓
3. Modelado y comparación
   - Regresión Lineal (baseline)
   - Random Forest Regressor
   - Gradient Boosting
        ↓
4. Evaluación
   - MSE · RMSE · R² Score
   - Feature Importance
```

## 📊 Resultados

> Ver el notebook completo para métricas detalladas y análisis de importancia de variables.

**Hallazgo principal:** El tabaquismo (`smoker`) es el factor con mayor peso en el costo del seguro, seguido por la edad y el BMI.

## 🧠 Conceptos clave demostrados

- Pipeline completo de Machine Learning con Scikit-learn
- Análisis de importancia de variables (Feature Importance)
- Comparación y selección de modelos
- Validación cruzada y métricas de regresión

## 🔭 Posibles extensiones

- App interactiva con Streamlit para predicción en tiempo real
- Optimización de hiperparámetros con GridSearchCV
- Análisis de equidad (fairness) del modelo por género y región

## 👤 Autor

Reinaldo Guerrero — Data Scientist Jr.
[LinkedIn](https://www.linkedin.com/in/reinaldo-guerrero-payares) · [GitHub](https://github.com/reinaldo-agp)

## 📄 Licencia

Licencia MIT — siéntete libre de usar y adaptar este proyecto.
