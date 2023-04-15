from dataclasses import dataclass, field
from typing import Callable, Optional, List, Union, Dict
import pandas as pd
from sklearn.base import TransformerMixin
import inspect
import pickle

@dataclass
class SequentialTransformer(TransformerMixin):
    steps: List[Callable] = field(default_factory=list)
    params: Dict[Callable, dict] = field(default_factory=dict)

    @staticmethod
    def _apply_step(step: Callable, params: dict, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[pd.DataFrame, pd.Series]:
        step_signature = inspect.signature(step)

        if 'y' not in step_signature.parameters:
            return step(X, **params)
        
        if y is None:
            raise ValueError("The step function expects a 'y' argument, but 'y' is not provided. Please provide a valid 'y' argument or modify the step function to work without it.")
        
        return step(X, y.copy(), **params)

    def transform(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> Union[pd.DataFrame, pd.Series]:
        """Applies a series of preselected transformation steps to the input DataFrame X.

        Args:
            X (pd.DataFrame): DataFrame of feature vectors
            y (Optional[Union[pd.DataFrame, pd.Series]], optional): Outcome vectors. Defaults to None.

        Raises:
            ValueError: if preprocessing steps have not been initialized. See @add_step
            ValueError: if non callable object has been initialized

        Returns:
            Union[pd.DataFrame, pd.Series]: Transformed DataFrame (X)
        """
        if not self.steps:
            raise ValueError("The 'steps' list is empty. Please provide a list of callable objects (functions or methods).")

        X_copy = X.copy()
        for step in self.steps:
            if not callable(step):
                raise ValueError("Expected a callable object (function/method) in 'steps' list, but encountered a non-callable object.")
            
            step_params = self.params.get(step, {})
            X_copy = self._apply_step(step, step_params, X_copy, y)

        return X_copy

    def _add(self, step: Callable, params: dict = None) -> None:
        self.steps.append(step)

        if params:
            self.params[step] = params

    def save(self, path: str) -> None:
        """Saves transformer in permanent memory

        Args:
            path (str): path and file name with extension 
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

def add_step(pipe: SequentialTransformer, params: dict = None) -> Callable:
    def wrapper(func: Callable) -> Callable:
        pipe.add(func)
        return func
    return wrapper
    



