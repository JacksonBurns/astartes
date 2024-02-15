<h1 align="center">astartes</h1> 
<h2 align="center"><em>(as-tar-tees)</em></h2>
<h3 align="center">Train:Validation:Test Algorithmic Sampling for Molecules and Arbitrary Arrays</h3>

<p align="center">  
  <img alt="astarteslogo" src="https://raw.githubusercontent.com/JacksonBurns/astartes/main/astartes_logo.png">
</p> 
<div align="center">
  <table>
    <caption><p style="font-weight:bold">Status Badges</p></caption>
    <tr>
      <th>Usage</th>
      <th>Continuous Integration</th>
      <th>Release</th>
    </tr>
    <tr>
      <td><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/astartes?style=plastic"></td>
      <td><img alt="Reproduce Paper" src="https://github.com/JacksonBurns/astartes/actions/workflows/reproduce_paper.yml/badge.svg?branch=main&event=schedule"></td>
      <td><a href="https://github.com/pyOpenSci/software-review/issues/120"><img src="https://tinyurl.com/y22nb8up" alt="pyOpenSci approved" /></a></td>
    </tr>
    <tr>
      <td><img alt="PyPI - License" src="https://img.shields.io/github/license/JacksonBurns/astartes"></td>
      <td><img alt="Test Status" src="https://github.com/JacksonBurns/astartes/actions/workflows/run_tests.yml/badge.svg?branch=main&event=schedule"></td>
      <td><a href="https://doi.org/10.5281/zenodo.8147205"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8147205.svg" alt="DOI"></a></td>
    </tr>
    <tr>
      <td><img alt="PyPI - Total Downloads" src="https://static.pepy.tech/personalized-badge/astartes?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Lifetime%20Downloads"></td>
      <td><a alt="Documentation Status"><img src="https://github.com/JacksonBurns/astartes/actions/workflows/gen_docs.yml/badge.svg"></td>
      <td><img alt="PyPI" src="https://img.shields.io/pypi/v/astartes"> <img alt="conda-forge version" src="https://img.shields.io/conda/vn/conda-forge/astartes.svg"></td>
    </tr>
    <tr>
      <td><img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/JacksonBurns/astartes?style=social"></td>
      <td><a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active – The project has reached a stable, usable state and is being actively developed." /></a></td>
      <td><a href="https://joss.theoj.org/papers/8a9cfc71d6f75410b06510a646d5f783"><img src="https://joss.theoj.org/papers/8a9cfc71d6f75410b06510a646d5f783/status.svg"></a></td>
    </tr>
  </table>
</div>


## Online Documentation
Follow [this link](https://JacksonBurns.github.io/astartes/) for a nicely-rendered version of this README along with additional tutorials for [moving from train_test_split in sklearn to astartes](https://jacksonburns.github.io/astartes/sklearn_to_astartes.html).
Keep reading for a installation guide and links to tutorials!

## Installing `astartes`
We recommend installing `astartes` within a virtual environment, using either `venv` or `conda` (or other tools) to simplify dependency management. Python versions 3.8, 3.9, 3.10, 3.11, and 3.12 are supported on all platforms.

> **Warning**
> Windows (PowerShell) and MacOS Catalina or newer (zsh) require double quotes around text using the `'[]'` characters (i.e. `pip install "astartes[molecules]"`).

### `pip`
`astartes` is available on `PyPI` and can be installed using `pip`:

 - To include the featurization options for chemical data, use `pip install astartes[molecules]`.
 - To install only the sampling algorithms, use `pip install astartes` (this install will have fewer dependencies and may be more readily compatible in environments with existing workflows).

### `conda`
`astartes` package is also available on `conda-forge` with this command: `conda install -c conda-forge astartes`.
To install `astartes` with support for featurizing molecules, use: `conda install -c conda-forge astartes aimsim`.
This will download the base `astartes` package as well as `aimsim`, which is the backend used for molecular featurization.

### Source
To install `astartes` from source for development, see the [Contributing & Developer Notes](#contributing--developer-notes) section.

## Statement of Need
Machine learning has sparked an explosion of progress in chemical kinetics, materials science, and many other fields as researchers use data-driven methods to accelerate steps in traditional workflows within some acceptable error tolerance. 
To facilitate adoption of these models, there are two important tasks to consider:
1. use a validation set when selecting the optimal hyperparameter for the model and separately use a held-out test set to measure performance on unseen data.
2. evaluate model performance on both interpolative and extrapolative tasks so future users are informed of any potential limitations.

`astartes` addresses both of these points by implementing an `sklearn`-compatible `train_val_test_split` function.
Additional technical detail is provided below as well as in our companion paper in the Journal of Open Source Software: [Machine Learning Validation via Rational Dataset Sampling with astartes](https://joss.theoj.org/papers/10.21105/joss.05996).
For a demo-based explainer using machine learning on a fast food menu, see the `astartes` Reproducible Notebook published at the United States Research Software Engineers Conference at [this page](https://jacksonburns.github.io/use-rse-23-astartes/split_comparisons.html).

### Target Audience
`astartes` is generally applicable to machine learning involving both discovery and inference _and_ model validation.
There are specific functions in `astartes` for applications in cheminformatics (`astartes.molecules`) but the methods implemented are general to all numerical data.

## Quick Start
`astartes` is designed as a drop-in replacement for `sklearn`'s `train_test_split` function (see the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)). To switch to `astartes`, change `from sklearn.model_selection import train_test_split` to `from astartes import train_test_split`.

Like `sklearn`, `astartes` accepts any iterable object as `X`, `y`, and `labels`.
Each will be converted to a `numpy` array for internal operations, and returned as a `numpy` array with limited exceptions: if `X` is a `pandas` `DataFrame`, `y` is a `Series`, or `labels` is a `Series`, `astartes` will cast it back to its original type including its index and column names.

> **Note**
> The developers recommend passing `X`, `y`, and `labels` as `numpy` arrays and handling the conversion to and from other types explicitly on your own. Behind-the-scenes type casting can lead to unexpected behavior!

By default, `astartes` will split data randomly. Additionally, a variety of algorithmic sampling approaches can be used by specifying the `sampler` argument to the function (see the [Table of Implemented Samplers](#implemented-sampling-algorithms) for a complete list of options and their corresponding references):

```python
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
  X,  # preferably numpy arrays, but astartes will cast it for you
  y,
  sampler = 'kennard_stone',  # any of the supported samplers
)
```

> **Note**
> Extrapolation sampling algorithms will return an additional set of arrays (the cluster labels) which will result in a `ValueError: too many values to unpack` if not called properly. See the [`split_comparisons` Google colab demo](https://colab.research.google.com/github/JacksonBurns/astartes/blob/main/examples/split_comparisons/split_comparisons.ipynb) for a full explanation.

That's all you need to get started with `astartes`!
The next sections include more examples and some demo notebooks you can try in your browser.

### Example Notebooks

Click the badges in the table below to be taken to a live, interactive demo of `astartes`:

| Demo | Topic | Link |
|:---:|---|---|
| Comparing Sampling Algorithms with Fast Food | Visual representations of how different samplers affect data partitioning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacksonBurns/astartes/blob/main/examples/split_comparisons/split_comparisons.ipynb) |
| Using `train_val_test_split` with the `sklearn` example datasets | Demonstrating how witholding a test set with `train_val_test_split` can impact performance | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacksonBurns/astartes/blob/main/examples/train_val_test_split_sklearn_example/train_val_test_split_example.ipynb) |
| Cheminformatics sample set partitioning with `astartes` | Extrapolation vs. Interpolation impact on cheminformatics model accuracy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacksonBurns/astartes/blob/main/examples/barrier_prediction_with_RDB7/RDB7_barrier_prediction_example.ipynb) |
| Comparing partitioning approaches for alkanes | Visualizing how sampler impact model performance with simple chemicals | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacksonBurns/astartes/blob/main/examples/mlpds_2023_astartes_demonstration/mlpds_2023_demo.ipynb) |

To execute these notebooks locally, clone this repository (i.e. `git clone https://github.com/JacksonBurns/astartes.git`), navigate to the `astartes` directory, run `pip install .[demos]`, then open and run the notebooks in your preferred editor.
You do _not_ need to execute the cells prefixed with `%%capture` - they are only present for compatibility with Google Colab.

### Withhold Testing Data with `train_val_test_split`
For rigorous ML research, it is critical to withhold some data during training to use a `test` set.
The model should _never_ see this data during training (unlike the validation set) so that we can get an accurate measurement of its performance.

With `astartes` performing this three-way data split is readily available with `train_val_test_split`:
```python
from astartes import train_val_test_split

X_train, X_val, X_test = train_val_test_split(X, sampler = 'sphere_exclusion')
```
You can now train your model with `X_train`, optimize your model with `X_val`, and measure its performance with `X_test`.

### Evaluate the Impact of Splitting Algorithms on Regression Models
For data with many features it can be difficult to visualize how different sampling algorithms change the distribution of data into training, validation, and testing like we do in some of the demo notebooks.
To aid in analyzing the impact of the algorithms, `astartes` provides `generate_regression_results_dict`.
This function allows users to quickly evaluate the impact of different splitting techniques on any `sklearn`-compatible model's performance.
All results are stored in a nested dictionary (`{sampler:{metric:{split:score}}}`) format and can be displayed in a neatly formatted table using the optional `print_results` argument.

```python
from sklearn.svm import LinearSVR

from astartes.utils import generate_regression_results_dict as grrd

sklearn_model = LinearSVR()
results_dict = grrd(
    sklearn_model,
    X,
    y,
    print_results=True,
)

         Train       Val      Test
----  --------  --------  --------
MAE   1.41522   3.13435   2.17091
RMSE  2.03062   3.73721   2.40041
R2    0.90745   0.80787   0.78412

```

Additional metrics can be passed to `generate_regression_results_dict` via the `additional_metrics` argument, which should be a dictionary mapping the name of the metric (as a `string`) to the function itself, like this:

```python
from sklearn.metrics import mean_absolute_percentage_error

add_met = {"mape": mean_absolute_percentage_error}

grrd(sklearn_model, X, y, additional_metric=add_met)
```

See the docstring for `generate_regression_results_dict` (with `help(generate_regression_results_dict)`) for more information.

### Using `astartes` with Categorical Data
Any of the implemented sampling algorithms whose hyperparameters allow specifying the `metric` or `distance_metric` (effectively `1-metric`) can be co-opted to work with categorical data.
Simply encode the data in a format compatible with the `sklearn` metric of choice and then call `astartes` with that metric specified:
```python
from sklearn.metrics import jaccard_score

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  sampler='kennard_stone',
  hopts={"metric": jaccard_score},
)
```

Other samplers which do not allow specifying a categorical distance metric did not provide a method for doing so in their original inception, though it is possible that they can be adapted for this application.
If you are interested in adding support for categorical metrics to an existing sampler, consider opening a [Feature Request](https://github.com/JacksonBurns/astartes/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=%5BFEATURE%5D%3A+)!

### Access Sampling Algorithms Directly
The sampling algorithms implemented in `astartes` can also be directly accessed and run if it is more useful for your applications.
In the below example, we import the Kennard Stone sampler, use it to partition a simple array, and then retrieve a sample.
```python
from astartes.samplers.interpolation import KennardStone

kennard_stone = KennardStone([[1, 2], [3, 4], [5, 6]])
first_2_samples = kennard_stone.get_sample_idxs(2)
```
All samplers in `astartes` implement a `_sample()` method that is called by the constructor (i.e. greedily) and either a `get_sampler_idxs` or `get_cluster_idxs` for interpolative and extrapolative samplers, respectively.
For more detail on the implementaiton and design of samplers in `astartes`, see the [Developer Notes](#contributing--developer-notes) section.

## Theory and Application of `astartes`
This section of the README details some of the theory behind why the algorithms implemented in `astartes` are important and some motivating examples.
For a comprehensive walkthrough of the theory and implementation of `astartes`, follow [this link](https://github.com/JacksonBurns/astartes/raw/joss-paper/Burns-Spiekermann-Bhattacharjee_astartes.pdf) to read the companion paper (freely available and hosted here on GitHub).

> **Note**
> We reference open-access publications wherever possible. For articles locked behind a paywall (denoted with :small_blue_diamond:), we instead suggest reading [this Wikipedia page](https://en.wikipedia.org/wiki/Sci-Hub) and absolutely __not__ attempting to bypass the paywall.

### Rational Splitting Algorithms
While much machine learning is done with a random choice between training/validation/test data, an alternative is the use of so-called "rational" splitting algorithms.
These approaches use some similarity-based algorithm to divide data into sets.
Some of these algorithms include Kennard-Stone ([Kennard & Stone](https://www.tandfonline.com/doi/abs/10.1080/00401706.1969.10490666) :small_blue_diamond:), Sphere Exclusion ([Tropsha et. al](https://pubs.acs.org/doi/pdf/10.1021/ci300338w) :small_blue_diamond:),as well as the OptiSim as discussed in [Applied Chemoinformatics: Achievements and Future Opportunities](https://www.wiley.com/en-us/Applied+Chemoinformatics%3A+Achievements+and+Future+Opportunities-p-9783527806546) :small_blue_diamond:.
Some clustering-based splitting techniques have also been incorporated, such as [DBSCAN](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1016.890&rep=rep1&type=pdf).

There are two broad categories of sampling algorithms implemented in `astartes`: extrapolative and interpolative.
The former will force your model to predict on out-of-sample data, which creates a more challenging task than interpolative sampling.
See the table below for all of the sampling approaches currently implemented in `astartes`, as well as the hyperparameters that each algorithm accepts (which are passed in with `hopts`) and a helpful reference for understanding how the hyperparameters work.
Note that `random_state` is defined as a keyword argument in `train_test_split` itself, even though these algorithms will use the `random_state` in their own work.
Do not provide a `random_state` in the `hopts` dictionary - it will be overwritten by the `random_state` you provide for `train_test_split` (or the default if none is provided).

#### Implemented Sampling Algorithms

| Sampler Name | Usage String | Type | Hyperparameters | Reference | Notes |
|:---:|---|---|---|---|---|
| Random | 'random' | Interpolative | `shuffle` | [sklearn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) Documentation | This sampler is a direct passthrough to `sklearn`'s `train_test_split`. |
| Kennard-Stone | 'kennard_stone' | Interpolative | `metric` | Original Paper by [Kennard & Stone](https://www.tandfonline.com/doi/abs/10.1080/00401706.1969.10490666) :small_blue_diamond: | Euclidian distance is used by default, as described in the original paper. |
| Sample set Partitioning based on joint X-Y distances (SPXY) | 'spxy' | Interpolative | `distance_metric` | Saldhana et. al [original paper](https://www.sciencedirect.com/science/article/abs/pii/S003991400500192X) :small_blue_diamond: | Extension of Kennard Stone that also includes the response when sampling distances. |
| Mahalanobis Distance Kennard Stone (MDKS) | 'spxy' _(MDKS is derived from SPXY)_ | Interpolative | _none, see Notes_ | Saptoro et. al [original paper](https://espace.curtin.edu.au/bitstream/handle/20.500.11937/45101/217844_70585_PUB-SE-DCE-FM-71008.pdf?sequence=2&isAllowed=y) | MDKS is SPXY using Mahalanobis distance and can be called by using SPXY with `distance_metric="mahalanobis"` |
| Scaffold | 'scaffold' | Extrapolative | `include_chirality` | [Bemis-Murcko Scaffold](https://pubs.acs.org/doi/full/10.1021/jm9602928) :small_blue_diamond: as implemented in RDKit | This sampler requires SMILES strings as input (use the `molecules` subpackage) |
| Molecular Weight| 'molecular_weight' | Extrapolative | _none_ | ~ | Sorts molecules by molecular weight as calculated by RDKit |
| Sphere Exclusion | 'sphere_exclusion' | Extrapolative | `metric`, `distance_cutoff` | _custom implementation_ | Variation on Sphere Exclusion for arbitrary-valued vectors. |
| Time Based | 'time_based' | Extrapolative | _none_ | Papers using Time based splitting: [Chen et al.](https://pubs.acs.org/doi/full/10.1021/ci200615h) :small_blue_diamond:, [Sheridan, R. P](https://pubs.acs.org/doi/full/10.1021/ci400084k) :small_blue_diamond:, [Feinberg et al.](https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.9b02187) :small_blue_diamond:, [Struble et al.](https://pubs.rsc.org/en/content/articlehtml/2020/re/d0re00071j) | This sampler requires `labels` to be an iterable of either date or datetime objects. |
| Target Property | 'target_property' | Extrapolative | `descending` | ~ | Sorts data by regression target y |
| Optimizable K-Dissimilarity Selection (OptiSim) | 'optisim' | Extrapolative | `n_clusters`, `max_subsample_size`, `distance_cutoff` | _custom implementation_ | Variation on [OptiSim](https://pubs.acs.org/doi/10.1021/ci025662h) for arbitrary-valued vectors. |
| K-Means | 'kmeans' | Extrapolative | `n_clusters`, `n_init` | [`sklearn KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) | Passthrough to `sklearn`'s `KMeans`. |
| Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | 'dbscan' | Extrapolative | `eps`, `min_samples`, `algorithm`, `metric`, `leaf_size` | [`sklearn DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) Documentation| Passthrough to `sklearn`'s `DBSCAN`. |
| Minimum Test Set Dissimilarity (MTSD) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| Restricted Boltzmann Machine (RBM) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| Kohonen Self-Organizing Map (SOM) | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |
| SPlit Method | ~ | ~ | _upcoming in_ `astartes` _v1.x_ | ~ | ~ |

### Domain-Specific Applications
Below are some field specific applications of `astartes`. Interested in adding a new sampling algorithm or featurization approach? See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

#### Chemical Data and the `astartes.molecules` Subpackage
Machine Learning is enormously useful in chemistry-related fields due to the high-dimensional feature space of chemical data.
To properly apply ML to chemical data for inference _or_ discovery, it is important to know a model's accuracy under the two domains.
To simplify the process of partitioning chemical data, `astartes` implements a pre-built featurizer for common chemistry data formats.
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

To see a complete example of using `train_test_split_molecules` with actual chemical data, take a look in the `examples` directory and the brief [companion paper](https://github.com/JacksonBurns/astartes/raw/joss-paper/Burns-Spiekermann-Bhattacharjee_astartes.pdf).

Configuration options for the featurization scheme can be found in the documentation for [AIMSim](https://vlachosgroup.github.io/AIMSim/README.html#currently-implemented-fingerprints) though most of the critical configuration options are shown above.

## Reproducibility
`astartes` aims to be completely reproducible across different platforms, Python versions, and dependency configurations - any version of `astartes` v1.x should result in the _exact_ same splits, always.
To that end, the default behavior of `astartes` is to use `42` as the random seed and _always_ set it.
Running `astartes` with the default settings will always produce the exact same results.
We have verified this behavior on Debian Ubuntu, Windows, and Intel Macs from Python versions 3.7 through 3.11 (with appropriate dependencies for each version).

### Known Reproducibility Limitations
Inevitably external dependencies of `astartes` will introduce backwards-incompatible changes.
We continually run regression tests to catch these, and will list all _known_ limitations here:
 - `sklearn` v1.3.0 introduced backwards-incompatible changes in the `KMeans` sampler that changed how the random initialization affects the results, even given the same random seed. Different version of `sklearn` will affect the performance of `astartes` and we recommend including the exact version of `scikit-learn` and `astartes` used, when applicable.

> **Note**
> We are limited in our ability to test on M1 Macs, but from our limited manual testing we achieve perfect reproducbility in all cases _except occasionally_ with `KMeans` on Apple silicon.
`astartes` is still consistent between runs on the same platform in all cases, and other samplers are not impacted by this apparent bug.

## How to Cite
If you use `astartes` in your work please follow the link below to our (Open Access!) paper in the Journal of Open Source Software or use the "Cite this repository" button on GitHub.

[Machine Learning Validation via Rational Dataset Sampling with astartes](https://joss.theoj.org/papers/10.21105/joss.05996)

## Contributing & Developer Notes
See [CONTRIBUTING.md](./CONTRIBUTING.md) for instructions on installing `astartes` for development, making a contribution, and general guidance on the design of `astartes`.

