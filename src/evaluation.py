import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    model,
    test_generator,
    model_name,
    average="macro",
):
    test_generator.reset()

    probabilities = model.predict(
        test_generator,
        verbose=0,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels = test_generator.classes

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average=average,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average=average,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average=average,
        zero_division=0,
    )

    result = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
    }

    return pd.DataFrame([result])