import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.optimizers import SGD, RMSprop, Adam, Lamb
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Tu función de carga y preprocesamiento de datos ---
def load_and_preprocess_data(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Eliminar columnas innecesarias
    # Asegúrate de que 'customerID' exista en tu dataset o ajusta esta línea
    if 'customerID' in df.columns:
        df.drop(['customerID'], axis=1, inplace=True)

    # Convertir 'TotalCharges' a numérico
    # Asegúrate de que 'TotalCharges' exista en tu dataset o ajusta esta línea
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True) # Elimina filas con NaN después de la conversión

    # Codificar variables categóricas
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Churn': # Asumiendo 'Churn' es tu variable objetivo original
            df[column] = LabelEncoder().fit_transform(df[column])

    # Codificar variable objetivo
    # Asegúrate de que 'Churn' exista en tu dataset y tenga los valores 'No'/'Yes'
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    else:
        raise ValueError("La columna 'Churn' no se encontró en el dataset. Asegúrate de que el nombre de la variable objetivo sea correcto.")


    # Dividir en características y objetivo
    # Asumiendo 'Churn' es la columna objetivo después de la codificación
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Escalar características
    # Es importante escalar DESPUÉS de dividir en conjuntos de entrenamiento y prueba
    # para evitar fuga de datos. Tu función lo hace antes, lo ajustaremos ligeramente
    # en el script principal para hacerlo correctamente.
    # Por ahora, la función solo retorna X, y para que el script principal lo divida y escale.
    return X, y

# --- Script Principal del Proyecto ---

# 1. Carga y Preprocesamiento del Dataset
# Reemplaza 'tu_dataset.csv' por la ruta real de tu archivo de datos
# Este dataset parece ser el de telecomunicaciones 'Telco Customer Churn' de Kaggle.
FILEPATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' # Asegúrate de que este archivo exista

try:
    X_full, y_full = load_and_preprocess_data(FILEPATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en '{FILEPATH}'. Asegúrate de que el dataset esté en la ruta correcta.")
    exit()
except ValueError as e:
    print(f"Error en el preprocesamiento: {e}")
    exit()

# Dividir en conjuntos de entrenamiento, validación y prueba
# Realizaremos el escalado DESPUÉS de la división para evitar data leakage
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Escalar las características después de la división
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train_raw)
X_val_processed = scaler.transform(X_val_raw)
X_test_processed = scaler.transform(X_test_raw)


# Determinar el número de características de entrada
input_shape = X_train_processed.shape[1]

# --- 2. Implementación de Modelos con Keras y Regularización ---

def build_model(input_shape, regularizer=None, regularization_strength=0.001):
    """
    Construye un modelo de red neuronal multicapa con opción de regularización.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid') # Para clasificación binaria (Churn)
    ])
    return model

# Funciones de penalidad a probar
regularizers_dict = {
    'L1': regularizers.l1(0.01),
    'L2': regularizers.l2(0.01)
 }

results = []

for reg_name, regularizer_fn in regularizers_dict.items():
    optimizers = {
        'SGD': SGD(learning_rate=0.001),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001),
        'Lamb': Lamb(learning_rate=0.001)
    }
    for opt_name, optimizer_fn in optimizers.items():
        print(f"\n--- Entrenando con Regularización: {reg_name}, Optimizador: {opt_name} ---")
        
        keras.backend.clear_session()
        
        model = build_model(input_shape, regularizer=regularizer_fn)
        model.compile(optimizer=optimizer_fn,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        start_time = time.time()
        history = model.fit(X_train_processed, y_train,
                            epochs=50, # Puedes ajustar las épocas
                            batch_size=32,
                            validation_data=(X_val_processed, y_val),
                            verbose=0) # Poner 1 para ver el progreso del entrenamiento
        end_time = time.time()
        training_time = end_time - start_time

        # Evaluar en el conjunto de test
        loss_test, accuracy_test = model.evaluate(X_test_processed, y_test, verbose=0)

        # Capturar la última métrica de la función objetivo (loss)
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        results.append({
            'Regularizer': reg_name,
            'Optimizer': opt_name,
            'Train Loss': final_train_loss,
            'Validation Loss': final_val_loss,
            'Test Loss': loss_test,
            'Test Accuracy': accuracy_test,
            'Training Time (s)': training_time
        })

# --- 3. Resultados Numéricos y Análisis ---
results_df = pd.DataFrame(results)
print("\n--- Tabla de Resultados Numéricos ---")
print(results_df.round(4).to_string())
