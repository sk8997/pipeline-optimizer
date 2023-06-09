from pipeline_optimizer.transformers import SequentialTransformer, add_step
import pandas as pd
import pytest
import pickle

# Test 1: Test that a ValueError is raised when transform called with empty transformer.
def test_empty_transformer():
    
    transformer = SequentialTransformer()

    
    data = {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    df = pd.DataFrame(data)

    
    with pytest.raises(ValueError, match="The 'steps' list is empty. Please provide a list of callable objects"):
        transformer.transform(df)

# Test 2: Test that a ValueError is raised when a non-callable object is added to the `steps` list.
def test_invalid_step():
    transformer = SequentialTransformer(steps=[1, 2, 3])
    data = {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Expected a callable object"):
        transformer.transform(df)


# Test 3: Test that the `add_step` decorator properly adds a function and its default parameters to the pipeline.
def test_add_step_decorator():
    transformer = SequentialTransformer()

    @add_step(transformer)
    def example_step(X: pd.DataFrame, param: int = 2) -> pd.DataFrame:
        return X * param

    assert len(transformer.steps) == 1
    assert transformer.steps[0] == example_step
    assert transformer.params[example_step] == {'param': 2}



# Test 4: Test that the `_apply_step` method works correctly when the step function does not have a 'y' parameter.
def test_apply_step_without_y():
    transformer = SequentialTransformer()
    data = {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    df = pd.DataFrame(data)

    def step_without_y(X: pd.DataFrame) -> pd.DataFrame:
        return X * 2

    transformed_df = transformer._apply_step(step_without_y, {}, df)

    assert transformed_df.equals(df * 2)


# Test 5: Test that the `transform` method applies multiple steps in the correct order.
def test_transform_with_multiple_steps():
    transformer = SequentialTransformer()

    @add_step(transformer)
    def step1(X: pd.DataFrame) -> pd.DataFrame:
        return X * 2

    @add_step(transformer)
    def step2(X: pd.DataFrame) -> pd.DataFrame:
        return X + 1

    data = {'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]}
    df = pd.DataFrame(data)

    transformed_df = transformer.transform(df)

    expected_transformed_df = (df * 2) + 1

    assert transformed_df.equals(expected_transformed_df)

def sample_step(X, factor=2):
    return X * factor


# Test 6: Test saving and loading transformer
def test_sequential_transformer_save(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    st = SequentialTransformer(steps=[sample_step], params={sample_step: {"factor": 3}})
    file_path = tmp_path / "transformer.pkl"
    st.save(str(file_path))

    with open(file_path, "rb") as f:
        loaded_st = pickle.load(f)

    transformed = loaded_st.transform(df)

    expected = pd.DataFrame({"A": [3, 6, 9], "B": [12, 15, 18]})
    pd.testing.assert_frame_equal(transformed, expected)