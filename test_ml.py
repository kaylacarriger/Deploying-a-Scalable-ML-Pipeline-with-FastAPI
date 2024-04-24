import os
import pickle
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import inference, compute_model_metrics
from ml.data import process_data

"""
Fixtures
"""
@pytest.fixture(scope="module")
def data():
    """
    Fixture - Load data from file
    """
    datapath = "./data/census.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    """
    Fixture - Path to data file
    """
    return "./data/census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - Return categorical features
    """
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - Return cleaned train dataset
    """
    train, test = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(train, categorical_features=features, label="salary", training=True)
    return X_train, y_train


"""
Test methods
"""


def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    df = pd.read_csv(path)
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_features(data, features):
    """
    Test that categorical features are in dataset
    """
    assert sorted(set(data.columns).intersection(features)) == sorted(features)


def test_is_fitted_model(train_dataset):
    """
    Test saved model is fitted
    """
    X_train, y_train = train_dataset
    savepath = "./model/model.pkl"
    model = pickle.load(open(savepath, 'rb'))
    assert model.predict(X_train).shape[0] > 0


def test_compute_model_metrics(train_dataset):
    """
    Test performance metrics function
    """
    X_train, y_train = train_dataset
    savepath = "./model/model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)
        precision, recall, fbeta = compute_model_metrics(y_train, preds)
        assert precision is not None
        assert recall is not None
        assert fbeta is not None
    else:
        pytest.fail("File not found.")



