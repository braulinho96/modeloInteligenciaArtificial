from procesamiento_datos import load_and_preprocess_data
from definir_modelo import construir_modelo, SmoothedAbs, RoundedSquare, EntropyLikeRegularizer, VarianceSuppression, TangentRegularizer                                        
from entrenamiento import train_model
from evaluacion import evaluate_model
from keras.optimizers import SGD, RMSprop, Adam, Lamb
import pandas as pd

import matplotlib.pyplot as plt

import random 
import numpy as np 
import tensorflow as tf 

SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Cargar y preprocesar los datos de Telco Customer Churn
X_train, X_test, y_train, y_test = load_and_preprocess_data('mushrooms.csv')

# Definir los regularizadores y sus parámetros
lambdaRegularizador = 0.01
regularizadoresPropios = {
    #'SmoothedAbs': SmoothedAbs(lmbd=lambdaRegularizador),
    'RoundedSquare': RoundedSquare(lmbd=lambdaRegularizador)
    #'EntropyLike': EntropyLikeRegularizer(lmbd=lambdaRegularizador),
    #'Tangent': TangentRegularizer(lmbd=lambdaRegularizador), 
    #'VarianceSuppression': VarianceSuppression(lmbd=lambdaRegularizador)
}

# Lista para almacenar los resultados
results = []
numero_caracteristicas = X_train.shape[1]

# Entrenar y evaluar modelos
for nombre_reg, regularizador in regularizadoresPropios.items():
    # Definir los algoritmos de optimización a utilizar y sus parámetros de aprendizaje
    algoritmos_optimizacion = {
        'SGD': SGD(learning_rate=0.001)
        #'RMSprop': RMSprop(learning_rate=0.001),
        #'Adam': Adam(learning_rate=0.001),
        #'Lamb': Lamb(learning_rate=0.001)  
        }
    
    # Se crea una figura para graficar la precisión de validación para cada combinación de regularizador y optimizador
    plt.figure(figsize=(10, 6))
    
    for nombre_opt, optimizador in algoritmos_optimizacion.items():
        print(f'\nEntrenando con optimizador: {nombre_opt} y regularización: {nombre_reg}')
        # Construir el modelo con regularización
        model = construir_modelo(input_dim=numero_caracteristicas, regularizer_instance = regularizador)
        
        # Entrenar el modelo con los datos de entrenamiento y validación
        history, tiempo_entrenamiento = train_model(model, X_train, y_train, optimizador, X_test, y_test, )
        
        # Imprimir el resumen del modelo
        model.summary()
        
        # Evaluar el modelo
        resultados_evaluacion = evaluate_model(model, X_test, y_test)
        
        # Crear un diccionario con los resultados
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

        # Línea crítica que proporciona feedback inmediato del rendimiento del modelo
        print(f"{nombre_reg} + {nombre_opt} → Acc: {res['Test Accuracy']:.4f}, F1: {res['F1_score']:.4f}, AUC: {res['ROC_AUC']:.4f}")
        results.append(res)

        # Grafica la precisión de validación para cada regularizador en el grafico
        plt.plot(history.history['val_accuracy'], label= "{} + {}".format(nombre_reg, nombre_opt))

        # Gráfico de pérdida de entrenamiento y validación
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss', drawstyle='default' )
        plt.plot(history.history['val_loss'], label='Validation Loss', drawstyle='default')
        plt.title(f'Loss de Entrenamiento y Validación - {nombre_reg} + {nombre_opt}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Guarda la imagen dentro de una carpeta imagen
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

# Convertir los resultados a un DataFrame de pandas para una mejor visualización
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
