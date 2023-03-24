<h1 align="center">astartes</h1> 
<h3 align="center">Train:Validation:Test Algorithmic Sampling for Molecules and Arbitrary Arrays</h3>

<p align="center">  
  <img alt="astarteslogo" src="https://github.com/JacksonBurns/astartes/blob/main/astartes_logo.png">
</p> 
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/astartes?style=social">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/astartes">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/astartes">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/astartes?style=plastic">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/astartes">
  <img alt="Test Status" src="https://github.com/JacksonBurns/astartes/actions/workflows/run_tests.yml/badge.svg?branch=main&event=schedule">
</p>

## Installing `astartes`
We recommend installing `astartes` within a virtual environment, using either `venv` or `conda` (or other tools) to simplify dependency management. Python versions 3.7, 3.8, 3.9, 3.10, and 3.11 are supported on all platforms.

`astartes` is available on `PyPI` and can be installed using `pip`:

 - To include the featurization options for chemical data, use `pip install astartes[molecules]`.
 - To install only the sampling algorithms, use `pip install astartes` (this install will have fewer dependencies and may be more readily compatible in environments with existing workflows).

__Note for Windows Powershell or MacOS Catalina or newer__: On these systems the command line will complain about square brackets, so you will need to double quote the `molecules` command (i.e. `pip install "astartes[molecules]"`)

## Using `astartes`
`astartes` is designed as a drop-in replacement for `sklearn`'s `train_test_split` function. To switch to `astartes`, change `from sklearn.model_selection import train_test_split` to `from astartes import train_test_split`.

By default, `astartes` will use a random splitting approach identical to that which is implemented in `sklearn`, and a variety of deterministic sampling approaches can be used by specifying one additional argument ot the function:

```python
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  sampler = 'kennard_stone',  # any of the supported samplers
)
```

### Example Notebooks

Click the badges in the table below to be taken to a live, interactive demo of `astartes`:

| Demo | Link |
|:---:|---|
| Using `train_val_test_split` with the `sklearn` example datasets | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JacksonBurns/astartes/main?labpath=examples%2Ftrain_val_test_split_example.ipynb) |
| Cheminformatics sample set partitioning with `astartes` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JacksonBurns/astartes/main?labpath=examples%2FRDB7_barrier_prediction_example.ipynb) |

### Rational Splitting Algorithms
While much machine learning is done with a random choice between training/test/validation data, an alternative is the use of so-called "rational" splitting algorithms. These approaches use some similarity-based algorithm to divide data into sets. Some of these algorithms include Kennard-Stone, minimal test set dissimilarity, and sphere exclusion algorithms [as discussed by Tropsha et. al](https://pubs.acs.org/doi/pdf/10.1021/ci300338w) as well as the OptiSim as discussed in [Applied Chemoinformatics: Achievements and Future Opportunities](https://www.wiley.com/en-us/Applied+Chemoinformatics%3A+Achievements+and+Future+Opportunities-p-9783527806546). Some clustering-based splitting techniques have also been introduced, such as [DBSCAN](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1016.890&rep=rep1&type=pdf).

There are two broad categories of sampling algorithms implemented in `astartes`: extrapolative and interpolative. The former will force your model to predict on out-of-sample data, effectively asking a 'harder question' than interpolative sampling. See the table below for all of the sampling approaches currently implemented in `astartes`, as well as the hyperparameters that each algorithm accepts (which are passed in with `hopts`) and a helpful reference for understanding how the hyperparameters work. Note that `random_state` is defined as a keyword argument in `train_test_split` itself, even though these algorithms will use the `random_state` in their own work. Do not provide a `random_state` in the `hopts` dictionary - it will be overwritten by the `random_state` you provide for `train_test_split` (or the default if none is provided).

#### Implemented Sampling Algorithms

| Sampler Name | Usage String | Type | Hyperparameters | Reference | Notes |
|:---:|---|---|---|---|---|
| Random | 'random' | Interpolative | `shuffle` | [`sklearn train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) | This sampler is a direct passthrough to `sklearn`'s `train_test_split`, though it does not currently reproduce splits identically. |
| Kennard-Stone | 'kennard_stone' | Interpolative | _none_ | [yu9824's `kennard_stone`](https://github.com/yu9824/kennard_stone) | Fully deterministic, no hyperparameters accepted. |
| Sample set Partitioning based on join X-Y distances (SPXY) | 'spxy' | Interpolative | `distance_metric` | Saldhana et. al [original paper](https://www.sciencedirect.com/science/article/abs/pii/S003991400500192X) | Extension of Kennard Stone that also includes the response when sampling distances. |
| Scaffold | 'scaffold' | Extrapolative | `explicit_hydrogens`, `include_chirality` | [Bemis-Murcko Scaffold](https://pubs.acs.org/doi/full/10.1021/jm9602928) as implemented in RDKit | This sampler requires SMILES strings as input (use the `molecules` subpackage) |
| Sphere Exclusion | 'sphere_exclusion' | Extrapolative | `metric`, `distance_cutoff` | _custom implementation_ | Variation on Sphere Exclusion for arbitrary-valued vectors. |
| Optimizable K-Dissimilarity Selection (OptiSim) | 'optisim' | Extrapolative | `n_clusters`, `max_subsample_size`, `distance_cutoff` | _custom implementation_ | Variation on [OptiSim](https://pubs.acs.org/doi/10.1021/ci025662h) for arbitrary-valued vectors. |
| K-Means | 'kmeans' | Extrapolative | `n_clusters`, `n_init` | [`sklearn KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | Passthrough to `sklearn`'s `KMeans`. |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | 'dbscan' | Extrapolative | `eps`, `min_samples`, `algorithm`, `metric`, `leaf_size` | [`sklearn DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) | Passthrough to `sklearn`'s `DBSCAN`. |
| Minimum Test Set Dissimilarity (MTSD) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| Restricted Boltzmann Machine (RBM) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| Kohonen Self-Organizing Map (SOM) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| SPlit Method | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |

### Using the `astartes.molecules` Subpackage
After installing with `pip install astartes[molecules]` one can import the new train/test splitting function like this: `from astartes.molecules import train_test_split_molecules`

The usage of this function is identical to `train_test_split` but with the addition of new arguments to control how the molecules are featurized:

```python
train_test_split_molecules(
    molecules=smiles,
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
    sampler="random",
    random_state=42,
    hopts={
        "shuffle": True,
    },
)
```

To see a complete example of using `train_test_split_molecules` with actual chemical data, take a look in the `examples` directory.

Configuration options for the featurization scheme can be found in the documentation for [`AIMSim`](https://vlachosgroup.github.io/AIMSim/README.html#currently-implemented-fingerprints) though most of the critical configuration options are shown above.

## Online Documentation
[The online documentation](https://JacksonBurns.github.io/astartes/) contains everything you see in this README with an additional tutorial for [moving from `train_test_split` in `sklearn` to `astartes`](https://jacksonburns.github.io/astartes/sklearn_to_astartes.html).

## Contributing & Developer Notes
Pull Requests, Bug Reports, and all Contributions are welcome! Please use the appropriate issue or pull request template when making a contribution.

We make use of [the GitHub Discussions page](https://github.com/JacksonBurns/astartes/discussions) to go over potential features to add. Please feel free to stop by if you are looking for something to develop or have an idea for a useful feature!

When submitting a PR, please mark your PR with the "PR Ready for Review" label when you are finished making changes so that the GitHub actions bots can work their magic!

### Developer Install

To contribute to the `astartes` source code, start by cloning the repository (i.e. `git clone git@github.com:JacksonBurns/astartes.git`) and then inside the repository run `pip install -e .[molecules,dev]`. This will set you up with all the required dependencies to run `astartes` and conform to our formatting standards (`black` and `isort`), which you can configure to run automatically in vscode [like this](https://marcobelo.medium.com/setting-up-python-black-on-visual-studio-code-5318eba4cd00#:~:text=Go%20to%20settings%20in%20your,%E2%80%9D%20and%20select%20%E2%80%9Cblack%E2%80%9D.).

__Note for Windows Powershell or MacOS Catalina or newer__: On these systems the command line will complain about square brackets, so you will need to double quote the `molecules` command (i.e. `pip install -e ".[molecules,dev]"`)

### Unit Testing
All of the tests in `astartes` are written using the built-in python `unittest` module (to allow running without `pytest`) but we _highly_ recommend using `pytest`. To execute the tests from the `astartes` repository, simply type `pytest` after running the developer install (or alternately, `pytest -v` for a more helpful output).

### Adding New Samplers
Adding a new sampler should extend the `abstract_sampler.py` abstract base class.

It can be as simple as a passthrough to a another `train_test_split`, or it can be an original implementation that results in X and y being split into two lists. Take a look at `astartes/samplers/random_split.py` for a basic example!

After the sampler has been implemented, add it to `__init__.py` in in `astartes/samplers` and it will automatically be unit tested. Additional unit tests to verify that hyperparameters can be properly passed, etc. are also recommended.

For historical reasons, and as a guide for any developers who would like add new samplers, below is a running list of samplers which have been _considered_ for addition to `asartes` but ultimately not added for various reasons.

#### Not Implemented Sampling Algorithms

| Sampler Name | Reasoning | Relevant Link(s) |
|:---:|---|---|
| D-Optimal | Requires _a-priori_ knowledge of the test and train size which does not fit in the `astartes` framework (samplers are all agnostic to the size of the sets) and it is questionable if the use of the Fischer information matrix is actually meaningful in the context of sampling existing data rather than tuning for ideal data. | The [Wikipedia article for optimal design](https://en.wikipedia.org/wiki/Optimal_design#:~:text=Of%20course%2C%20fixing%20the%20number%20of%20experimental%20runs%20a%20priori%20would%20be%20impractical.) does a good job explaining why this is difficult, and points at some potential alternatives. |
| Duplex | Requires knowing test and train size before execution, and can only partition data into two sets which would make it incompatible with `train_val_test_split`. | This [implementation in R](https://search.r-project.org/CRAN/refmans/prospectr/html/duplex.html#:~:text=The%20DUPLEX%20algorithm%20is%20similar,that%20are%20the%20farthest%20apart.) includes helpful references and a reference implementation. |

### Adding New Featurization Schemes
All of the sampling methods implemented in `astartes` accept arbitrary arrays of numbers and return the sampled groups (with the exception of `Scaffold.py`). If you have an existing featurization scheme (i.e. take an arbitrary input and turn it into an array of numbers), we would be thrilled to include it in `astartes`.

Adding a new interface should take on this format:

```python
from astartes import train_test_split

def train_test_split_INTERFACE(
    INTERFACE_input,
    INTERFACE_ARGS,
    y: np.array = None,
    labels: np.array = None,
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
        labels=labels,
        test_size=test_size,
        train_size=train_size,
        splitter=splitter,
        hopts=hopts,
    )
```

If possible, we would like to also add an example Jupyter Notebook with any new interface to demonstrate to new users how it functions. See our other examples in the `examples` directory.

Contact [@JacksonBurns](https://github.com/JacksonBurns) if you need assistance adding an existing workflow to `astartes`. If this featurization scheme requires additional dependencies to function, we may add it as an additional _extra_ package in the same way that `molecules` in installed.

## JOSS Branch

`astartes` corresponding JOSS paper is stored in this repository on a separate branch. You can find `paper.md` on the aptly named `joss-paper` branch. 

_Note for Maintainers_: To push changes from the `main` branch into the `joss-paper` branch, run the `Update JOSS Branch` workflow.

