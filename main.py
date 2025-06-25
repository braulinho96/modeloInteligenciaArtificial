from procesamiento_datos import carga_procesar_datos
from definir_modelo import construir_modelo, VarianceSuppression, MaxPenaltyRegularizer
from entrenamiento import entrenar_modelo
from evaluacion import evaluar_modelo
from keras.optimizers import SGD, RMSprop, Adam, Lamb
import pandas as pd
import matplotlib.pyplot as plt
import random 
import numpy as np 
import tensorflow as tf 

# Configuración de la semilla para reproducibilidad
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Cargar y preprocesar los datos de Telco Customer Churn
X_train, X_test, y_train, y_test = carga_procesar_datos('mushrooms.csv')

# Definir los regularizadores y sus parámetros
lambdaRegularizador = 0.001
regularizadoresPropios = {
    'VarianceSuppression': VarianceSuppression(lmbd=lambdaRegularizador),
    'MaxPenaltyRegularizer': MaxPenaltyRegularizer(lmbd=lambdaRegularizador)
}

# Lista para almacenar los resultados
results = []
numero_caracteristicas = X_train.shape[1]
print(f"Número de características: {numero_caracteristicas}")

# Entrenar y evaluar modelos
for nombre_reg, regularizador in regularizadoresPropios.items():

    # Definir los algoritmos de optimización a utilizar y sus parámetros de aprendizaje, se dejan en esta seccion ya que se encontró un error al definirlos fuera del bucle
    algoritmos_optimizacion = {
        'SGD': SGD(learning_rate=0.001),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001),
        'Lamb': Lamb(learning_rate=0.001)  
        }
    
    # Se crea una figura para graficar la precisión de validación para cada combinación de regularizador y optimizador
    plt.figure(figsize=(10, 6))
    
    for nombre_opt, optimizador in algoritmos_optimizacion.items():
        print(f'\nEntrenando con optimizador: {nombre_opt} y regularización: {nombre_reg}')
        # Construir el modelo con regularización
        model = construir_modelo(input_dim=numero_caracteristicas, regularizer_instance = regularizador)
        
        # Entrenar el modelo con los datos de entrenamiento y validacion, luego evaluar con los datos de prueba y almacenar los resultados
        history, tiempo_entrenamiento = entrenar_modelo(model, X_train, y_train, optimizador, X_test, y_test, )
        resultados_evaluacion = evaluar_modelo(model, X_test, y_test)
        res = {
            'Regularizador': nombre_reg,
            'Algoritmo optimizacion': nombre_opt,
            'Train Loss': history.history['loss'][-1],
            'Validation Loss': history.history['val_loss'][-1],
            'Test Loss': resultados_evaluacion['loss_test'], 
            'Test Accuracy': resultados_evaluacion['accuracy_test'],
            'Precision': resultados_evaluacion['Precision'],
            'Recall': resultados_evaluacion['Recall'],
            'F1_score': resultados_evaluacion['F1_score'],
            'ROC_AUC': resultados_evaluacion['roc_auc'],
            'Training Time (s)': tiempo_entrenamiento
        }

        # Retroalimentación de los resultados
        print(f"{nombre_reg} + {nombre_opt} → Acc: {res['Test Accuracy']:.4f}, F1: {res['F1_score']:.4f}, AUC: {res['ROC_AUC']:.4f}")
        results.append(res)

        # Grafica la precisión de validación para cada regularizador en el grafico
        plt.plot(history.history['val_accuracy'], label= "{} + {}".format(nombre_reg, nombre_opt))

        # Gráfico de pérdida de entrenamiento y validación de cada regularizador y optimizador
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss', drawstyle='default' )
        plt.plot(history.history['val_loss'], label='Validation Loss', drawstyle='default')
        plt.title(f'Loss de Entrenamiento y Validación - {nombre_reg} + {nombre_opt}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_filename = f'imagenes/loss_{nombre_reg}_{nombre_opt}.png'
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

    # Finalizar la gráfica de precisión de validación para el regularizador actual con cada optimizador
    plt.title(f'Precisión de Validación - {nombre_reg}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión de Validación')
    plt.legend()
    plt.grid(True)
    nombre_archivo = f'{nombre_reg}.png'
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()

# Mostrar los resultados en una tabla 
results_df = pd.DataFrame(results)
print("\n--- Tabla de Resultados Numéricos ---")
print(results_df.round(4).to_string())

# Graficar el tiempo de entrenamiento por combinación de modelo
results_df['Combinación'] = results_df['Algoritmo optimizacion'] + " (" + results_df['Regularizador'] + ")"
combinaciones = results_df['Combinación']
tiempos = results_df['Training Time (s)']
plt.figure(figsize=(12, 6))
bars = plt.bar(combinaciones, tiempos, color='skyblue')
plt.title('Tiempo de Entrenamiento por Combinación de Modelo', fontsize=14, pad=20)
plt.xlabel('Optimizador + Regularización', fontsize=12)
plt.ylabel('Tiempo (segundos)', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    altura = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, altura, f'{altura:.2f}s', ha='center', va='bottom')
plt.tight_layout()
plt.savefig("tiempo_entrenamiento.png", dpi=300)
plt.show()

# Mostrar resumen de tiempos de entrenamiento 
print("\n--- Resumen de Tiempos de Entrenamiento ---")
mejor_tiempo = results_df['Training Time (s)'].min()
mejor_modelo = results_df.loc[results_df['Training Time (s)'] == mejor_tiempo, 'Combinación'].iloc[0]
print(f"Modelo más rápido: {mejor_modelo} - Tiempo: {mejor_tiempo:.2f} segundos")
