


![pipe-logo](https://user-images.githubusercontent.com/84877088/232358047-f8545063-5053-4a9e-a24e-e9c266283f5d.png)




<div align="center">
<a href="https://github.com/sk8997/pipeline-optimizer/actions/workflows/tests.yml">
  <img src="https://github.com/sk8997/pipeline-optimizer/actions/workflows/tests.yml/badge.svg" alt="Tests" />
</a>
<a href="https://codecov.io/gh/sk8997/pipeline-optimizer" > 
 <img src="https://codecov.io/gh/sk8997/pipeline-optimizer/branch/main/graph/badge.svg?token=BCWYCTXZPA"/> 
 </a>
</div>

Pipeline Optimizer is a Python library that aims to simplify and automate the machine learning pipeline, from preprocessing and testing to deployment. By providing a reusable infrastructure, the library allows you to manage custom preprocessing functions and reuse them effortlessly during the deployment of your project. This is particularly useful when dealing with a large number of custom functions.

The library currently features a single class called `SequentialTransformer` which allows you to add custom preprocessing functions using a simple decorator. This class also integrates with scikit-learn's `TransformerMixin`, making it compatible with the widely-used scikit-learn library.

# Installation

```bash
pip install pipeline_optimizer
```

# SequentialTransformer

`SequentialTransformer` is a class that stores a list of preprocessing steps and applies them sequentially to input data. You can easily add a custom preprocessing function to its memory using the `@add_step` decorator. The class also provides methods to transform the input data, save the transformer to disk, and load it for later use.


Here's a quick demonstration of how to use the `SequentialTransformer` class:

# Step 1: Import necessary libraries

```python 

import pandas as pd
from pipeline_optimizer import SequentialTransformer, add_step

```


# Step 2: Load your dataset

```python

data = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [5, 4, 3, 2, 1],
    "C": [10, 20, 30, 40, 50]
})

labels = pd.Series([0, 1, 0, 1, 1])

```

# Step 3: Define preprocessing functions and add them to the pipeline

```python

pipe = SequentialTransformer()

@add_step(pipe)
def drop_column(df: pd.DataFrame, col: str = "B") -> pd.DataFrame:
    return df.drop(columns=[col])

@add_step(pipe)
def multiply(df: pd.DataFrame, col: str = "A", multiplier: float = 2) -> pd.DataFrame:
    df[col] = df[col] * multiplier
    return df

```

# Step 4: Transform the input data
After applying the preprocessing functions, the SequentialTransformer will drop column "B" and multiply column "A" by 2.

```python

transformed_data = pipe.transform(data)
print(transformed_data)

```

Output:

```

    A   C
0   2  10
1   4  20
2   6  30
3   8  40
4  10  50

```

# 

Step 5: Save the transformer object

```python
pipe.save("transformer.pkl")
```

# Step 6: Load the saved transformer and apply it to deployment data
You can load the saved transformer using the pickle module and apply it to new deployment data to preprocess it.

```python 

import pickle

# Load the saved transformer
with open("transformer.pkl", "rb") as f:
    loaded_pipe = pickle.load(f)

# Deployment data
deployment_data = pd.DataFrame({
    "A": [6],
    "B": [3],
    "C": [60]
})

# Transform the deployment data using the loaded transformer
transformed_deployment_data = loaded_pipe.transform(deployment_data)
print(transformed_deployment_data)

```

Output:

```
   A   C
0  12  60

```

# Integration with scikit-learn Pipeline

A noteworthy feature of the `SequentialTransformer` is that it can be seamlessly integrated with scikit-learn's `Pipeline` class. This further simplifies the preprocessing and deployment processes, enabling you to create an end-to-end machine learning pipeline that combines custom preprocessing steps with scikit-learn estimators.

By incorporating the `SequentialTransformer` into an sklearn `Pipeline`, you can benefit from the full range of features provided by scikit-learn, such as cross-validation, grid search, and model evaluation.

Here's a quick example of how to integrate initialized `SequentialTransformer` with an sklearn `Pipeline`:

```python

pipe = SequentialTransformer()

@add_step(pipe)
def drop_column(df: pd.DataFrame, col: str = "B") -> pd.DataFrame:
    return df.drop(columns=[col])

@add_step(pipe)
def multiply(df: pd.DataFrame, col: str = "A", multiplier: float = 2) -> pd.DataFrame:
    df[col] = df[col] * multiplier
    return df

# Create an sklearn pipeline with the custom SequentialTransformer and a Linear Discriminant Analysis
pipeline = Pipeline([
    ("preprocessor", pipe),  # Ensure the SequentialTransformer has been initialized and steps have been added
    ("lda", LinearDiscriminantAnalysis())
])

# Fit the pipeline 
pipeline.fit_transform(X, y)


```


# Comparison with scikit-learn

When working with custom preprocessing functions using the scikit-learn library, you would typically define a custom class that inherits from `TransformerMixin` and implement `fit` and `transform` methods for each function. This can be time-consuming and may lead to code duplication.

Alternatively, you can use scikit-learn's `FunctionTransformer` to create transformers from user-defined functions. However, using `FunctionTransformer` can become unwieldy when you have many preprocessing functions, as you need to create an instance of `FunctionTransformer` for each function and manage them individually.

Here's an example of how you would use `FunctionTransformer` to accomplish the same preprocessing steps as in the previous example:

```python

from sklearn.preprocessing import FunctionTransformer

# Define the preprocessing functions
def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=[col])

def multiply(df: pd.DataFrame, col: str, multiplier: float) -> pd.DataFrame:
    df[col] = df[col] * multiplier
    return df

# Create FunctionTransformer instances for each function
drop_column_transformer = FunctionTransformer(drop_column, kw_args={"col": "B"})
multiply_transformer = FunctionTransformer(multiply, kw_args={"col": "A", "multiplier": 2})

# Apply the preprocessing functions to the toy dataset
data_dropped = drop_column_transformer.transform(data)
data_transformed = multiply_transformer.transform(data_dropped)

```

As you can see, using `FunctionTransformer` requires creating separate instances for each preprocessing function and managing them individually. This approach can become cumbersome when dealing with a large number of custom functions. In contrast, the `SequentialTransformer` class in the Pipeline Optimizer library provides a more streamlined and efficient way to manage and apply multiple preprocessing functions.

With the Pipeline Optimizer library, you can easily define preprocessing functions and add them to the `SequentialTransformer` pipeline using the `@add_step` decorator. This approach is more concise and allows you to reuse your preprocessing functions across different projects effortlessly.
