import time
from keras.callbacks import EarlyStopping


# Compila y entrena el modelo con los datos de entrenamiento y validación
    #   optimizer: Algoritmo de optimización a utilizar (SGD, RMSprop, Adam, Lamb).
    #   loss: Función de pérdida a minimizar durante el entrenamiento (binary_crossentropy para clasificación binaria).
    #   metrics: Métricas a evaluar durante el entrenamiento y la validación (accuracy para clasificación).
def entrenar_modelo(model, X_train, y_train, optimizer, X_test, y_test):
    early_stop = EarlyStopping(
        monitor='val_loss',       # qué métrica observar 
        patience=5,               # cuántas épocas sin mejora esperar antes de detener
        restore_best_weights=True  # restaurar los mejores pesos del modelo
    )
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(X_train, y_train, 
                        epochs=50,  
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=0,
                        callbacks=[early_stop]) 
    end_time = time.time()
    return [history, end_time - start_time]
