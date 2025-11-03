from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, test_data):
    y_pred = model.predict(test_data)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    print(classification_report(test_data.classes, y_pred_classes))
    print(confusion_matrix(test_data.classes, y_pred_classes))
