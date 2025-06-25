from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 

# Evaluación del modelo de clasificación binaria
# Esta función evalúa el modelo utilizando los datos de prueba y calcula métricas como precisión, recall, F1-score y AUC-ROC.

def evaluar_modelo(model, X_test, y_test):
    # Evaluamos el modelo con los datos de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Obtenemos la pérdida y precisión del modelo en el conjunto de prueba
    y_pred_prob = model.predict(X_test)                         
    y_pred = (y_pred_prob > 0.5).astype(int)                    

    # Predecimos las probabilidades y convertimos a etiquetas binarias
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    resultados = {
        'loss_test': loss,
        'accuracy_test': accuracy,
        'Precision': prec,
        'Recall': rec,
        'F1_score': f1,
        'roc_auc': auc
    }
    return resultados
    