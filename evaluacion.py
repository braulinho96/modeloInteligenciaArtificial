from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

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
    