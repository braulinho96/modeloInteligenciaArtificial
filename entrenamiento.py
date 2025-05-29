import time

"""
    Entrena el modelo con los datos de entrenamiento y validación.
    Args:
        model: Modelo de Keras a entrenar.
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        optimizer: Optimizador a utilizar para el entrenamiento.
        X_test: Datos de validación.
        y_test: Etiquetas de validación.
    Returns:
        history: Historial del entrenamiento.
        tiempo_entrenamiento: Tiempo total de entrenamiento en segundos.
    """
def train_model(model, X_train, y_train, optimizer, X_test, y_test):
    # Compila y entrena el modelo con los datos de entrenamiento y validación
    # Compile es una funcion de Keras que prepara el modelo para el entrenamiento.
    #   optimizer: Algoritmo de optimización a utilizar (SGD, RMSprop, Adam, Lamb).
    #   loss: Función de pérdida a minimizar durante el entrenamiento (binary_crossentropy para clasificación binaria).
    #   metrics: Métricas a evaluar durante el entrenamiento y la validación (accuracy para clasificación).
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=0)
    end_time = time.time()
    return [history, end_time - start_time]
