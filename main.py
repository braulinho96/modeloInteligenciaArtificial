from procesamiento_datos import load_and_preprocess_data
from definir_modelo import construir_modelo, RegularizadorL1, RegularizadorL2
from entrenamiento import train_model
from evaluacion import evaluate_model
from keras.optimizers import SGD, RMSprop, Adam, Lamb
from keras.regularizers import l1, l2
import pandas as pd
import time
# Debido a un error de CUDA (asociado a la GPU), solo se utiliza le CPU para los ajustes del modelo.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cargar y preprocesar los datos de Telco Customer Churn
X_train, X_test, y_train, y_test = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Definir funciones de penalización integradas de Keras, para comparar con las personalizadas
# L1: Penalizacion de Lasso: penaliza la suma de los valores absolutos de los pesos, provoca que # algunos pesos se vuelvan exactamente cero, 
#   lo que puede ser útil para la selección de características y eliminación de ruido. Elimina características, 
#   pero puede provocar que el modelo no aprenda bien si se usa con demasiada fuerza.

# L2: Penalizacion de Ridge: penaliza la suma de los cuadrados de los pesos, provoca que los pesos se vuelvan pequeños, pero no exactamente cero,
#   lo que puede ser útil para evitar el sobreajuste y mejorar la generalización del modelo. No elimina características, pero las reduce.
regularizador = {
    'L1': l1(0.001),
    'L2': l2(0.001)
}

# Definir regularizadores personalizados
# Se define un lambda para la regularización, que controla la fuerza de la penalización
lambdaRegularizador = 0.001
regularizadoresPropios = {
    'L1': RegularizadorL1(lmbd=lambdaRegularizador),
    'L2': RegularizadorL2(lmbd=lambdaRegularizador)
}

# Lista para almacenar los resultados
results = []
numero_caracteristicas = X_train.shape[1]

# Entrenar y evaluar modelos
for nombre_reg, regularizador in regularizadoresPropios.items():
    # Entrenadores de optimizadores --> Se dejan dentro del ciclo dado que si se definen fuera, aparece un error de tipo numpy()
    #   learning_rate: controla la magnitud de los ajustes realizados en los pesos del modelo durante el entrenamiento. 
    #       Mayor learning_rate puede acelerar el entrenamiento, pero si es demasiado alto, puede provocar que el modelo no converja.
    
    #   SGD: Stochastic Gradient Descent, es un optimizador básico que actualiza los pesos del modelo utilizando un solo ejemplo de entrenamiento a la vez.
    
    #   RMSprop: Root Mean Square Propagation, es un optimizador adaptativo que ajusta la tasa de aprendizaje para cada parámetro,
    #       teniendo en cuenta la magnitud de los gradientes recientes. Es útil para problemas con gradientes ruidosos.
    
    #   Adam: Adaptive Moment Estimation, es un optimizador adaptativo que combina las ventajas de RMSprop y AdaGrad,
    #       ajustando la tasa de aprendizaje para cada parámetro y utilizando momentos para mejorar la convergencia.
    
    #   Lamb: Layer-wise Adaptive Moments optimizer for Batch training, es un optimizador adaptativo que ajusta la tasa de aprendizaje para cada parámetro,
    #       teniendo en cuenta la magnitud de los gradientes recientes y la normalización de los pesos. Es útil para problemas con gradientes ruidosos.
    
    algoritmos_optimizacion = {
        'SGD': SGD(learning_rate=0.001),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001),
        'Lamb': Lamb(learning_rate=0.001)  
        }
    
    
    for nombre_opt, optimizador in algoritmos_optimizacion.items():
        print(f'\nEntrenando con optimizador: {nombre_opt} y regularización: {nombre_reg}')
        # Construir el modelo con regularización
        model = construir_modelo(input_dim=numero_caracteristicas, regularizer_instance = regularizador)
        
        # Entrenar el modelo con los datos de entrenamiento y validación
        model.compile(optimizer=optimizador, loss='binary_crossentropy', metrics=['accuracy'])
        start_time = time.time()
        history = model.fit(X_train, y_train, 
                            epochs=50, 
                            batch_size=32,
                            validation_data=(X_test, y_test),
                            verbose=0)
        end_time = time.time()
        
        # Obtener la pérdida final del entrenamiento y validación
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Evaluar el modelo
        resultados_evaluacion = evaluate_model(model, X_test, y_test)
        
        # Capturar los resultados
        results.append({
            'Regularizador': nombre_reg,
            'Algoritmo optimizacion': nombre_opt,
            'Train Loss': final_train_loss,
            'Validation Loss': final_val_loss,
            'Test Loss': resultados_evaluacion['loss_test'], 
            'Test Accuracy': resultados_evaluacion['accuracy_test'],
            'Precision': resultados_evaluacion['Precision'], 
            'Recall': resultados_evaluacion['Recall'], 
            'F1_score': resultados_evaluacion['F1_score'], 
            'ROC_AUC': resultados_evaluacion['roc_auc'], 
            'Training Time (s)': end_time - start_time
        })

# Convertir los resultados a un DataFrame de pandas para una mejor visualización
results_df = pd.DataFrame(results)
print("\n--- Tabla de Resultados Numéricos ---")
print(results_df.round(4).to_string())
