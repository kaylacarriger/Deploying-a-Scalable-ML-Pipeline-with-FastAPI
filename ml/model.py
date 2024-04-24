import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.
    """
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

def compute_model_metrics(y_true: pd.Series, preds: pd.Series) -> tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y_true : pd.Series
        Known labels, binarized.
    preds : pd.Series
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_true, preds, beta=1, zero_division=1)
    precision = precision_score(y_true, preds, zero_division=1)
    recall = recall_score(y_true, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model: RandomForestClassifier, X: pd.DataFrame) -> pd.Series:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : pd.DataFrame
        Data used for prediction.
    Returns
    -------
    preds : pd.Series
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path: str):
    """ Serializes model to a file.

    Inputs
    ------
    model : object
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    """ Loads pickle file from `path` and returns it."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : sklearn.base.BaseEstimator
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # Process the slice data
    X_slice, y_slice, _, _ = process_data(
        data=data[data[column_name] == slice_value],
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    
    # Get predictions
    preds = inference(model, X_slice)
    
    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    
    return precision, recall, fbeta

