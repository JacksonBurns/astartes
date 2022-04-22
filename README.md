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
</p>

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

## Extending Functionality
Adding a new sampler should extend the `sampler.py` abstract base class.

It can be as simple as a passthrough to a another `train_test_split`, or it can be an original implementation that results in X and y being split into two lists.

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

## JOSS Branch
`paper.md` is stored in a separate branch aptly named `joss-paper`. To push changes from the `main` branch into the `joss-paper` branch, run the `Update JOSS Branch` workflow.

