# modeloInteligenciaArtificial

Este proyecto implementa un modelo de red neuronal artificial para la clasificación de hongos comestibles y venenosos, utilizando el conjunto de datos `mushrooms.csv`. Se incluye el flujo completo: desde el procesamiento de datos, definición y entrenamiento del modelo, hasta la evaluación de resultados y visualización gráfica.


# Estructura del Proyecto

A continuación, se detallan las principales carpetas y archivos incluidos:

# `imagenes/`
Contiene visualizaciones generadas durante el entrenamiento de los modelos. Estas imágenes incluyen curvas de pérdida (`loss`) correspondientes a distintos optimizadores (SGD, Adam, RMSprop, Lamb) y regularizadores (MaxPenalty, VarianceSuppression), lo que permite comparar su desempeño visualmente.

# `Informe/`
Incluye todos los recursos gráficos utilizados en la elaboración del informe final. Contiene imágenes como la distribución de la variable objetivo y matrices de correlación (`cramers_v_matrix`). Algunas subcarpetas contienen configuraciones específicas (por ejemplo, valores de dropout).

# `pruebas con modelos/`
Aquí se encuentran los resultados obtenidos al probar diferentes configuraciones del modelo, como combinaciones de optimizadores y funciones de pérdida.

# `tablas de contingencia/`
Carpeta dedicada exclusivamente al almacenamiento de las tablas de contingencia en distintos experimentos.


## 🚀 Cómo ejecutar el proyecto

1. Clona este repositorio y entra en la carpeta del proyecto.
2. Instala los requerimientos (si aplica).
3. Ejecuta `main.py` para correr el modelo completo.
4. Revisa las carpetas `imagenes` y `pruebas con modelos` para visualizar los resultados.


# Notas finales

Este proyecto fue desarrollado como parte de una investigación académica. Está diseñado para facilitar la experimentación con diferentes configuraciones de redes neuronales en un problema de clasificación binaria, y fomentar el análisis crítico del desempeño mediante visualización y métricas detalladas.

