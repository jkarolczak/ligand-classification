from pickle import dump
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray, n_neighbors: int = 5, n_jobs: int = -1) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - n_neighbors: Number of neighbors to use for classification.

    Returns:
    - knn: Trained KNN classifier.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    knn.fit(X_train, y_train)

    return knn

def save_knn_model(model: KNeighborsClassifier, filename: str) -> None:
    """
    Save the trained KNN model to a file.

    Parameters:
    - knn: Trained KNN classifier.
    - filename: Path to save the model.
    """
    with open(filename, "wb") as f:
        dump(model, f, protocol=5)

def read_knn_model(filename: str) -> KNeighborsClassifier:
    """
    Load a KNN model from a file.

    Parameters:
    - filename: Path to the saved model.

    Returns:
    - knn: Loaded KNN classifier.
    """
    with open(filename, "rb") as f:
        knn = dump(f)
    return knn

def knn_predict(model: KNeighborsClassifier, example: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """
    Predict labels for test data using the trained KNN model.

    Parameters:
    - knn: Trained KNN classifier.
    - X_test: Test data features.

    Returns:
    - df: DataFrame containing predictions and distances.
    """
    predictions, distances = model.kneighbors(example, return_distance=True)
    predictions = predictions.flatten()
    distances = distances.flatten()
    df = pd.DataFrame({
        'predictions': predictions,
        'distances': distances
    })
    return df