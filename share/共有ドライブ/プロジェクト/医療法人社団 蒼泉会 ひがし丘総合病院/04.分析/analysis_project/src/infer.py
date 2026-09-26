from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator


def run_inference(model: BaseEstimator, X: pd.DataFrame):
    return model.predict(X)
