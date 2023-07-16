## Contributing & Developer Notes
Pull Requests, Bug Reports, and all Contributions are welcome, encouraged, and appreciated!
Please use the appropriate [issue](https://github.com/JacksonBurns/astartes/issues/new/choose) or [pull request](https://github.com/JacksonBurns/astartes/compare) template when making a contribution to help the maintainers get it merged quickly.

We make use of [the GitHub Discussions page](https://github.com/JacksonBurns/astartes/discussions) to go over potential features to add.
Please feel free to stop by if you are looking for something to develop or have an idea for a useful feature!

When submitting a PR, please mark your PR with the "PR Ready for Review" label when you are finished making changes so that the GitHub actions bots can work their magic!

### Developer Install

To contribute to the `astartes` source code, start by forking and then cloning the repository (i.e. `git clone git@github.com:YourUsername/astartes.git`) and then inside the repository run `pip install -e .[dev]`. This will set you up with all the required dependencies to run `astartes` and conform to our formatting standards (`black` and `isort`), which you can configure to run automatically in VSCode [like this](https://marcobelo.medium.com/setting-up-python-black-on-visual-studio-code-5318eba4cd00#:~:text=Go%20to%20settings%20in%20your,%E2%80%9D%20and%20select%20%E2%80%9Cblack%E2%80%9D.).

> **Note**
> Windows Powershell and MacOS Catalina or newer may complain about square brackets, so you will need to double quote the `molecules` command (i.e. `pip install "astartes[dev]"`)

### Version Checking

`astartes` uses `pyproject.toml` to specify all metadata _except_ the version, which is specified in `astartes/__init__.py` (via `__version__`) for backwards compatibility with Python 3.7.
To check which version of `astartes` you have installed, you can run `python -c "import astartes; print(astartes.__version__)"` on Python 3.7 or `python -c "from importlib.metadata import version; version('astartes')" on Python 3.8 or newer.

### Testing
All of the tests in `astartes` are written using the built-in python `unittest` module (to allow running without `pytest`) but we _highly_ recommend using `pytest`.
To execute the tests from the `astartes` repository, simply type `pytest` after running the developer install (or alternately, `pytest -v` for a more helpful output).
On GitHub, we use actions to run the tests on every Pull Request and on a nightly basis (look in `.github/workflows` for more information).
These tests include unit tests, functional tests, and regression tests.

### Adding New Samplers
Adding a new sampler should extend the `abstract_sampler.py` abstract base class.
Each subclass should override the `_sample` method with its own algorithm for data partitioning, and the constructor (`__init__.py`) perform any data validation.

All samplers in `astartes` are classified as one of two types: extrapolative or interpolative.
Extrapolative samplers work by clustering data into groups (which are then partitioned into train/validation/test to enforce extrapolation) whereas interpolative samplers provide an exact _order_ in which samples should be moved into the training set.

When actually implemented, this means that extrapolative samplers should set the `self._samples_clusters` attribute and interpolative samplers should set the `self._samples_idxs` attribute.

New samplers can be as simple as a passthrough to another `train_test_split`, or it can be an original implementation that results in X and y being split into two lists. Take a look at `astartes/samplers/interpolation/random_split.py` for a basic example!

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

### The `train_val_test_split` Function
`train_val_test_split` is the workhorse function of `astartes`.
It is responsible for instantiating the sampling algorithm, partitioning the data into training, validation, and testing, and then returning the requested results while also keeping an eye on data types.
Under the hood, `train_test_split` is just calling `train_val_test_split` with `val_size` set to `0.0`.
For more information on how it works, check out the inline documentation in `astartes/main.py`.

### Development Philosophy

The developers of `astartes` prioritize (1) reproducibility, (2) flexibility, and (3) maintainability.
 1. All versions of `astartes` `1.x` should produce the same results across all platforms, so we have thorough unit and regression testing run on a continuous basis.
 2. We specify as _few dependencies as possible_ with the _loosest possible_ dependency requirements, which allows integrating `astartes` with other tools more easily.
  - Depdencies which introduce a lot of requirements and/or specific versions of requirements are shuffled into the `extras_require` to avoid weighing down the main package.
  - Compatibility with all versions of modern Python is achieved by avoiding specifying version numbers tightly and regression testing across all versions.
 3. We follow DRY (Don't Repeat Yourself) principles to avoid code duplication and decrease maintainence burden, have near-perfect test coverage, and enforce consistent formatting style in the source code.
  - Inline comments are _critical_ for maintainability - at the time of writing, `astartes` has 1 comment line for every 2 lines of source code.

## JOSS Branch

`astartes` corresponding JOSS paper is stored in this repository on a separate branch. You can find `paper.md` on the aptly named `joss-paper` branch. 

_Note for Maintainers_: To push changes from the `main` branch into the `joss-paper` branch, run the `Update JOSS Branch` workflow.