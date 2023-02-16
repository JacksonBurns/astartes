<h1 align="center">astartes</h1> 
<h3 align="center">Train:Test Algorithmic Sampling for Molecules, Images, and Arbitrary Arrays</h3>

<p align="center">  
  <img alt="astarteslogo" src="https://github.com/JacksonBurns/astartes/blob/main/astartes_logo.png">
</p> 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/astartes?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/astartes">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/astartes">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/astartes">
  <img alt="Test Status" src="https://github.com/JacksonBurns/astartes/actions/workflows/run_tests.yml/badge.svg?branch=main&event=schedule">
</p>

## Installing `astartes`
We reccomend installing `astartes` within a virtual environment, using either `venv` or `conda` (or other tools) to simplify dependency management.

`astartes` is availble on `PyPI` and can be installed using `pip`:

 - To include the featurization options for chemical data, use `pip install astartes[molecules]`.
 - To install only the sampling algorithms, use `pip install astartes` (this install will have fewer depdencies and may be more readily compatible in environments with existing workflows).

## Using `astartes`
`astartes` is designed as a drop-in replacement for `sklearn`'s `train_test_split` function. To switch to `astartes`, change `from sklearn.model_selection import train_test_split` to 'from astartes import train_test_split`.

By default, `astartes` will use a random splitting approach identical to that which is implemented in `sklearn`, and a variety of deterministic sampling approaches can be used by specifying one additional argument ot the function:

```python
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  sampler = 'kennard_stone',  # any of the supported samplers
)
```

There are two broad categories of sampling algorithms implemented in `astartes`: supervised (requires labeled data) and unsupervised. All can be accessed via `train_test_split`, but supervised algorithms require an additional argument `labels` to be specified:

```python
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  sampler = 'time_split',  # any of the supported samplers
)
```

Here is a list of all implement sampling algorithms:

| Sampler Name | Usage String | Type | Reference | Notes |
|:---:|---|---|---|---|
| Random | 'random' | Interpolative | [sklearn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) | This sampler is a direct passthrough to sklearn's `train_test_split`. |
| Scaffold | 'scaffold' | Extrapolative | [`chemprop`'s `scaffold_split`](https://github.com/chemprop/chemprop/blob/959176dd0c6475bdca259b4ce71bab9b0a71ba4e/chemprop/data/scaffold.py#L53) | This sampler is  |
| Sphere Exclusion | 'sphere_exclusion' | Extrapolative | _custom implementation_ | Variation on Sphere Exclusion for arbitrary-valued vectors |

### Using the `astartes.molecules` Subpackage
After installing with `pip install astartes[molecules]` one can import the new train/test splitting function like this: `from astartes.molecules import train_test_split_molecules`

The usage of this function is identical to `train_test_split` but with the addition of new arguments to control how the molecules are featurized:

```python
train_test_split_molecules(
    smiles=smiles,
    y=y,
    test_size=0.2,
    train_size=0.8,
    fingerprint="daylight_fingerprint",
    fprints_hopts={
        "minPath": 2,
        "maxPath": 5,
        "fpSize": 200,
        "bitsPerHash": 4,
        "useHs": 1,
        "tgtDensity": 0.4,
        "minSize": 64,
    },
    splitter="random",
    hopts={
        "random_state": 42,
        "shuffle": True,
    },
)
```

Configuraiton options for the featurization scheme can be found in the documentation for AIMSim.

## Online Documentation
[Click here to read the documentation](https://JacksonBurns.github.io/astartes/)

## Background

### Rational Splitting Algorithms
While much machine learning is done with a random choice between training/test/validation data, an alternative is the use of so-called "rational" splitting algorithms. These approaches use some similarity-based algorithm to divide data into sets. Some of these algorithms include Kennard-Stone, minimal test set dissimilarity, and sphere exclusion algorithms [as discussed by Tropsha et. al](https://pubs.acs.org/doi/pdf/10.1021/ci300338w) as well as the DUPLEX, OptiSim, D-optimal, as discussed in [Applied Chemoinformatics: Achievements and Future Opportunities](https://www.wiley.com/en-us/Applied+Chemoinformatics%3A+Achievements+and+Future+Opportunities-p-9783527806546). Some clustering-based splitting techniques have also been introduced, such as [DBSCAN](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1016.890&rep=rep1&type=pdf).

## Sampling Algorithms
 - Random
 - Kennard-Stone (KS)
 - Minimal Test Set Dissimilarity
 - Sphere Exclusion
 - DUPLEX
 - OptiSim
 - D-Optimal
 - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
 - KMEANS Split
 - SPXY
 - RBM
 - Time Split


## Development
To install the most updated release of `astartes` for development purposes, use `pip install -e --target=. asartes[molecules]` or clone this repository. Pull requests are welcome!

### Adding New Samplers
Adding a new sampler should extend the `abstract_sampler.py` abstract base class.

It can be as simple as a passthrough to a another `train_test_split`, or it can be an original implementation that results in X and y being split into two lists. Take a look at `astartes/samplers/random_split.py` for a basic example!

### Adding New Featurization Schemes
All of the sampling methods implemented in `astartes` accept arbitrary arrays of numbers and return the sampled groups -- if you have an existing featurization scheme (i.e. take an arbitrary input and turn it into an array of numbers), we would be thrilled to include it in `astartes`.

Adding a new interface should take on this format:

```python
from extended_train_test_split import train_test_split

def train_test_split_INTERFACE(
    INTERFACE_input,
    INTERFACE_ARGS,
    y: np.array = None,
    test_size: float = 0.25,
    train_size: float = 0.75,
    splitter: str = 'random',
    hopts: dict = {},
    INTERFACE_hopts: dict = {},
):
    # turn the INTERFACE_input into an input X
    # based on INTERFACE ARGS where INTERFACE_hopts
    # specifies additional behavior
    X = []
    
    # call train test split with this input
    return train_test_split(
        X,
        y=y,
        test_size=test_size,
        train_size=train_size,
        splitter=splitter,
        hopts=hopts,
    )
```

## JORS Branch
`paper.tex` is stored in a separate branch aptly named `jors-paper`. To push changes from the `main` branch into the `jors-paper` branch, run the `Update JORS Branch` workflow.

