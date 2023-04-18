from dataclasses import dataclass, field
from typing import Callable, Optional, List, Union, Dict, TypeVar, Any
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import inspect
import pickle

@dataclass
class SequentialTransformer(BaseEstimator, TransformerMixin):
    steps: List[Callable] = field(default_factory=list)
    params: Dict[Callable, dict] = field(default_factory=dict)

    @staticmethod
    def _apply_step(step: Callable, params: dict, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input 'X' must be a pandas DataFrame.")
        
        return step(X, **params)
    

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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
            X_copy = self._apply_step(step, step_params, X_copy)

        return X_copy
    
    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None):
        """No-op."""
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> Union[pd.DataFrame, pd.Series]:
        return self.transform(X)

    def _add(self, step: Callable, params: Optional[dict] = None) -> None:
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

T = TypeVar("T", bound=Callable[..., Any])
def add_step(pipe: SequentialTransformer) -> Callable[[T], T]:
    """Wraps a function to automatically register it as a processing step in the provided pipeline, 
       along with any default parameters for that function

    Args:
        pipe (SequentialTransformer): The pipeline object to which the processing step should be added.

    Returns:
        Callable: A wrapper function that accepts a function as its argument and registers it as a processing step in the pipe.
    """
    def wrapper(func: T) -> T:
        # Find predefined parameters
        sig = inspect.signature(func)
        params = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default != inspect.Parameter.empty
        }

        pipe._add(func, params)
        return func
    return wrapper
    



