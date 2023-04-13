import inspect
import re
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
    LabelEncoder,
    Binarizer,
)

@dataclass
class OmniPipe(BaseEstimator, TransformerMixin):
    steps: list = field(default_factory = list)

    def __post_init__(self):
        self.steps = [cache_decorator(step) for step in self.steps]

    def fit(self, X, y=None):
        X_copy = X.copy()
        for step in self.steps:
            if callable(step):
                X_copy = step(X_copy)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for step in self.steps:
            if callable(step):
                X_copy = step(X_copy)
        return X_copy
    


SUPPORTED_CLASSES = {
    'OrdinalEncoder': OrdinalEncoder,
    'OneHotEncoder': OneHotEncoder,
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'MaxAbsScaler': MaxAbsScaler,
    'RobustScaler': RobustScaler,
    'PowerTransformer': PowerTransformer,
    'QuantileTransformer': QuantileTransformer,
    'KBinsDiscretizer': KBinsDiscretizer,
    'LabelEncoder': LabelEncoder,
    'Binarizer': Binarizer,
}

def cache_decorator(func):
    cache = {}
    func_code = inspect.getsource(func)

    def get_object(class_name):
        if class_name not in cache:
            cache[class_name] = SUPPORTED_CLASSES[class_name]()
        return cache[class_name]

    def wrapped_function(X):
        X_transformed = X
        for class_name in SUPPORTED_CLASSES:
            if class_name in func_code:
                obj = get_object(class_name)
                X_transformed = obj.fit_transform(X_transformed)
        X_transformed = func(X_transformed)
        return X_transformed

    return wrapped_function