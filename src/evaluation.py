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
    test_dataset,
    model_name,
    average="macro",
):
    probabilities = model.predict(
        test_dataset,
        verbose=0,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels = np.concatenate(
        [
            np.argmax(labels.numpy(), axis=1)
            for _, labels in test_dataset
        ]
    )

    result = {
        "Model": model_name,
        "Accuracy": round(
            accuracy_score(true_labels, predicted_labels),
            4,
        ),
        "Precision": round(
            precision_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            ),
            4,
        ),
        "Recall": round(
            recall_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            ),
            4,
        ),
        "F1-score": round(
            f1_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            ),
            4,
        ),
    }

    return pd.DataFrame([result])
