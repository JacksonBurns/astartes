# Transitioning from `sklearn` to `astartes`
## Step 1. Installation
`astartes` has been designed to rely on (1) as few packages as possible and (2) packages which are already likely to be installed in a Machine Learning (ML) Python workflow (i.e. Numpy and Sklearn). Because of this, `astartes` should be compatible with your _existing_ workflow such as a conda environment.

To install `astartes` for general ML use (the sampling of arbitrary vectors): __`pip install astartes`__

For users in cheminformatics, `astartes` has an optional add-on that includes featurization as part of the sampling. To install, type __`pip install 'astartes[molecules]'`__. With this extra install, `astartes` uses  [`AIMSim`](https://vlachosgroup.github.io/AIMSim/README.html) to encode SMILES strings as feature vectors. The SMILES strings are parsed into molecular graphs using RDKit and then sampled with a single function call: `train_test_split_molecules`.
 - If your workflow already has a featurization scheme in place (i.e. you already have a vector representation of your chemical of interest), you can directly use `train_test_split` (though we invite you to explore the many molecular descriptors made available through AIMSim).

## Step 2. Changing the `import` Statement
In one of the first few lines of your Python script, you have the line `from sklearn.model_selection import train_test_split`. To switch to using `astartes` change this line to `from astartes import train_test_split`.

That's it! You are now using `astartes`.

If you were just calling `train_test_split(X, y)`, your script should now work in the exact same way as `sklearn` with no changes required. 

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
)
```
_becomes_
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
)
```
But we encourage you to try out a non-random sampler (see below)!

## Step 3. Specifying a Non-Random Sampler
By default (for interoperability), `astartes` will use a random sampler to produce train/test splits - but the real value of `astartes` is in the algorithmic sampling algorithms it implements. Check out the [README for a complete list of available algorithms](https://github.com/JacksonBurns/astartes#implemented-sampling-algorithms) and how to call and customize them.

If you existing call to `train_test_split` looks like this:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)
```
and you want to try out using Kennard Stone sampling, switch it to this:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    sampler="kennard_stone",
)
```
That's it!

## Step 4. Passing Keyword Arguments

All of the arguments to the `sklearn`'s `train_test_split` can still be passed to `astartes`' `train_test_split`:
```python
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X,
    y,
    labels,
    train_size = 0.75,
    test_size = 0.25,
    sampler = "kmeans",
    hopts = {"n_clusters": 4},
)
```

Some samplers have tunable hyperparameters that allow you to more finely control their behavior. To do this with Sphere Exclusion, for example, switch your call to this:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    sampler="sphere_exclusion",
    hopts={"distance_cutoff":0.15},
)
```

## Step 5. Useful `astartes` Features

### `return_indices`: Improve Code Clarity
When providing `X`, `y`, and `labels` it can become cumbersome to unpack all of arrays, and there are cirumstances where the indices of the train/test data can be useful (for example, if `y` or `labels` are large, memory-intense objects). By default, `astartes` will return the arrays themselves, but it can also return just the indices for the user to manipulate according to their needs:
```python
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X,
    y,
    labels,
    return_indices = False,
)
```
_could instead be_
```python
indices_train, indices_test = train_test_split(
    X,
    y,
    labels,
    return_indices = True,
)
```
### `train_val_test_split`: More Rigorous ML
Behind the scenes, `train_test_split` is actually just a one-line function that calls the real workhorse of `astartes` - `train_val_test_split`:
```python
def train_test_split(
    X: np.array,
    ...
    return_indices: bool = False,
):
    return train_val_test_split(
        X, y, labels, train_size, 0, test_size, sampler, hopts, return_indices
    )
```
The function call to `train_val_test_split` is identical to `train_test_split` and supports all the same samplers and hyperparameters, except for one additional keyword argument `val_size`:
```python
def train_val_test_split(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: str = "random",
    hopts: dict = {},
    return_indices: bool = False,
):
```
When called, this will return _three_ arrays from `X`, `y`, and `labels` (or three arrays of indices, if `return_indices=True`) rather than the usual two, according to the values given for `train_size`, `val_size`, and `test_size` in the function call.
```python
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X,
    y,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
)
```
For truly rigorous ML modeling, the validation set should be used for hyperparameter tuning and the test set held out until the _very final_ change has been made to the model to get a true sense of its performance. For better or for worse, this is _not_ the current standard for ML modeling, but the authors believe it should be.

### Custom Warnings: `ImperfectSplittingWarning` and `NormalizationWarning`
In the event that your requested train/validation/test split is not mathematically possible given the dimensions of the input data (i.e. you request 50/25/25 but have 101 data points), `astartes` will warn you during runtime that it has occured. `sklearn` simply moves on quietly, and while this is fine _most_ of the time, the authors felt it prudent to warn the user.
When entering a train/validation/test split, `astartes` will check that it is normalized and make it so if not, warning the user during runtime. This will hopefully help prevent head-scratching hours of debugging.
