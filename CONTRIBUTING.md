## Contributing & Developer Notes
Pull Requests, Bug Reports, and all Contributions are welcome! Please use the appropriate issue or pull request template when making a contribution.

We make use of [the GitHub Discussions page](https://github.com/JacksonBurns/astartes/discussions) to go over potential features to add. Please feel free to stop by if you are looking for something to develop or have an idea for a useful feature!

When submitting a PR, please mark your PR with the "PR Ready for Review" label when you are finished making changes so that the GitHub actions bots can work their magic!

### Developer Install

To contribute to the `astartes` source code, start by cloning the repository (i.e. `git clone git@github.com:JacksonBurns/astartes.git`) and then inside the repository run `pip install -e .[molecules,dev]`. This will set you up with all the required dependencies to run `astartes` and conform to our formatting standards (`black` and `isort`), which you can configure to run automatically in vscode [like this](https://marcobelo.medium.com/setting-up-python-black-on-visual-studio-code-5318eba4cd00#:~:text=Go%20to%20settings%20in%20your,%E2%80%9D%20and%20select%20%E2%80%9Cblack%E2%80%9D.).

> **Note**
> Windows Powershell and MacOS Catalina or newer may complain about square brackets, so you will need to double quote the `molecules` command (i.e. `pip install "astartes[molecules,dev]"`)

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