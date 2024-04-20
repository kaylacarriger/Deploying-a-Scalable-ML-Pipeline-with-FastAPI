import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

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

def performance_on_categorical_slice(df: pd.DataFrame, feature: str, y: pd.Series, preds: pd.Series) -> pd.DataFrame:
    """ Computes the model metrics on a slice of the data specified by a column name.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label.
    feature : str
        Column containing the sliced feature.
    y : pd.Series
        Known labels, binarized.
    preds : pd.Series
        Predicted labels, binarized.
    Returns
    -------
    perf_df : pd.DataFrame
        Dataframe containing performance metrics for each slice.
    """
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, columns=['feature', 'n_samples', 'precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature] == option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # Reorder columns in the performance dataframe
    perf_df.reset_index(inplace=True)
    perf_df.rename(columns={'index': 'feature value'}, inplace=True)

    return perf_df
