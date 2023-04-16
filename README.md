# Pipeline Optimizer

[![Tests and Coverage](https://github.com/sk8997/pipeline-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/sk8997/pipeline-optimizer/actions/workflows/tests.yml)
<a href="https://codecov.io/gh/sk8997/pipeline-optimizer" > 
 <img src="https://codecov.io/gh/sk8997/pipeline-optimizer/branch/main/graph/badge.svg?token=BCWYCTXZPA"/> 
 </a>


Pipeline Optimizer is a Python library that aims to simplify and automate the machine learning pipeline, from preprocessing and testing to deployment. By providing a reusable infrastructure, the library allows you to manage custom preprocessing functions and reuse them effortlessly during the deployment of your project. This is particularly useful when dealing with a large number of custom functions.

The library currently features a single class called `SequentialTransformer` which allows you to add custom preprocessing functions using a simple decorator. This class also integrates with scikit-learn's `TransformerMixin`, making it compatible with the widely-used scikit-learn library.

# SequentialTransformer

`SequentialTransformer` is a class that stores a list of preprocessing steps and applies them sequentially to input data. You can easily add a custom preprocessing function to its memory using the `@add_step` decorator. The class also provides methods to transform the input data, save the transformer to disk, and load it for later use.


Here's a quick demonstration of how to use the `SequentialTransformer` class:

# Step 1: Import necessary libraries

```python 

import pandas as pd
from pipeline_optimizer import SequentialTransformer, add_step

```

3
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
def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.drop(columns=[col])

@add_step(pipe)
def multiply(df: pd.DataFrame, col: str, multiplier: float) -> pd.DataFrame:
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

Step 5: Save and load the transformer

```python
pipe.save("transformer.pkl")
```

# Comparison with scikit-learn

When working with custom preprocessing functions using the scikit-learn library, you would typically define a custom class that inherits from `TransformerMixin` and implement `fit` and `transform` methods for each function. This can be time-consuming and may lead to code duplication.

With the Pipeline Optimizer library, you can easily define preprocessing functions and add them to the `SequentialTransformer` pipeline using the `@add_step` decorator. This approach is more concise and allows you to reuse your preprocessing functions across different projects effortlessly.
