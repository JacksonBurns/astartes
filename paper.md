---
title: 'Machine Learning Validation via Rational Dataset Sampling with `astartes`'
tags:
  - Python
  - machine learning
  - sampling
  - interpolation
  - extrapolation
  - extrapolation
  - data splits
authors:
  - name: Jackson W. Burns
    orcid: 0000-0002-0657-9426
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2"
  - name: Kevin A. Spiekermann
    orcid: 0000-0002-9484-9253
    equal-contrib: true
    affiliation: 2
  - name: Himaghna Bhattacharjee
    orcid: 0000-0002-6598-3939
    affiliation: 3
  - name: Dionisios G. Vlachos
    orcid: 0000-0002-6795-8403
    affiliation: 3
  - name: William H. Green
    orcid: 0000-0003-2603-9694
    affiliation: 2
affiliations:
 - name: Center for Computational Science and Engineering, Massachusetts Institute of Technology
   index: 1
 - name: Department of Chemical Engineering, Massachusetts Institute of Technology, United States
   index: 2
 - name: Department of Chemical and Biomolecular Engineering, University of Delaware, United States
   index: 3
date: 3 April 2023
bibliography: paper.bib

---

# Summary

Machine Learning (ML) has become an increasingly popular tool to accelerate traditional workflows.
Critical to the use of ML is the process of splitting datasets into training and testing subsets used to develop and evaluate models, respectively.
It is common practice in the literature to assign these subsets randomly, which is both fast and efficient.
However, this only measures a model's capacity to interpolate, which is unrealistic in certain applications and can result in misleading test set accuracy.
To address this issue, we report `astartes`, an open-source Python package that implements many existing similarity- and distance-based algorithms to partition data into more challenging training, validation, and testing splits.
This publication focuses on use-cases within cheminformatics, but `astartes` operates on arbitrary vector inputs and the principals and workflow are generalizable to any ML field.
`astartes` is available via the Python package manager `pip` and is publicly hosted on GitHub ([github.com/JacksonBurns/astartes](https://github.com/JacksonBurns/astartes)).

# Statement of Need

Machine learning has sparked an explosion of progress in chemical kinetics [@komp2022progress; @spiekermann2022fast], drug discovery [@yang2019concepts; @bannigan2021machine], materials science [@wei2019machine], and energy storage [@jha2023learning] as researchers use data-driven methods to accelerate steps in traditional workflows within some acceptable error tolerance.
To facilitate adoption of these models, researchers must critically think about several topics, such as comparing model performance to relevant baseline, operating on user-friendly inputs, and reporting performance on both interpolative and extrapolative tasks <!-- cite Kevin's comment article-->. 
`astartes` aims to make it straightforward for machine learning scientists and researchers to focus on two of the most critical points: rigorous hyperparameter optimization and accurate performance evaluation.

First, `astartes` key function `train_val_test_split` returns splits for training, validation, and testing sets using an sklearn-like interface. This partitioning is crucial since best practices in data science dictate that, in order to minimize the risk of hyperparameter overfitting, one must only optimize hyperparameters with a validation set and use a held-out test set to accurately measure performance on unseen data [@ramsundar2019deep; @geron2019hands; @lakshmanan2020machine; @huyen2022designing; @wang2020machine]. 
Unfortunately, many papers only mention training and testing sets but do not mention validation sets, implying that they optimize the hyperparameters to the test set, which is blatant data leakage that leads to overly optimistic results [@li2020predicting; @van2022physics; @ismail2022successes; @liu2023predict]. 
For researchers who are interested in critical evaluation of model performance or non-random sampling but not yet ready to make the transition to using validation sets, `astartes` also implements an sklearn-compatible `train_test_split` function.


Second, it is crucial to evaluate model performance in both interpolation and extrapolation settings so future users are informed of any potential limitations.
Although random splits are frequently used in the literature, this simply measures interpolation performance.
However, given the vastness of chemical space [@ruddigkeit_GDB-17_2012] and its often unsmooth nature (e.g. activity cliffs), it seems unlikely that users will want to be restricted to exclusively operate in an interpolation regime.
Thus, to encourage adoption of these models, it is crucial to measure performance on more challenging splits as well.
The general workflow is: (1) Convert each molecule into a vector representation (2) Cluster the molecules based on similarity (3) Train the model on some clusters then evaluate performance on unseen clusters that should be dissimilar to the clusters used for training.
Although measuring performance on chemically dissimilar compounds/clusters is not a new concept [@meredig2018can; @durdy2022random; @stuyver2022quantum; @tricarico2022construction; @terrones2023low], there are a myriad of choices for the first two steps; our software incorporates many popular representations and similarity metrics to give users freedom to easily explore which combination is suitable for their needs.

# Example Use-Case in Cheminformatics

To demonstrate the difference in performance between interpolation and extrapolation, we apply `astartes` to two relevant cheminformatics datasets and their corresponding tasks.
For each dataset, a typical interpolative split is generated using a Random sampler and two unique extrapolative splits are generated for comparison. The first uses the cheminformatics-specific Bemis-Murcko scaffold [@bemis1996properties] as calculated by RDKit [@landrum2006rdkit]. <!-- Scaffold splits are a better measure of generalizability compared to random splits [@yang2019analyzing; @wang2020machine; @heid2021machine; @guan2021regio; @artrith2021best; @greenman2022multi]. -->
The second uses the more general-purpose K-means clustering based on the Euclidean distance of Morgan (ECFP4) fingerprints using 2048 bit hashing and radius of 2 [@morgan1965generation; @rogers2010extended]. For each split, we create 5 different folds (by changing the random seed). The values in Table QM9 and Table RDB7 correspond to the mean $\pm$ one standard deviation calculated across folds.

First is property prediction with QM9 [@ramakrishnan2014quantum], a dataset containing $\sim$133,000 small organic molecules and 12 relevant chemical properties for each. We train a multi-task model to predict all properties, with the arithmetic mean of all predictions tabulated below. <!-- the actual properties are: "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298", "h298", "g298" with units of https://schnetpack.readthedocs.io/en/stable/_modules/schnetpack/datasets/qm9.html -->
Second is a single-task model to predict a reaction's barrier height using the RDB7 dataset [@spiekermann2022high; @spiekermann_zenodo_database]. This reaction database contains a diverse set of $\sim$12,000 organic reactions relevant to the field of chemical kinetics.
Models were generated using a modified version of Chemprop [@yang2019analyzing] to train a deep message passing neural network to predict the regression targets of interest. 
We use the hyperparameters reported by ref. [@spiekermann2022fast] as implemented in the `barrier_prediction` branch publicly available on [GitHub](https://github.com/kspieks/chemprop/tree/barrier_prediction) [@spiekermann_forked_chemprop]. <!-- We use the `barrier_prediction` branch from a forked version of Chemprop [@yang2019analyzing; @spiekermann_forked_chemprop] to train a deep message passing neural network using the hyperparameters reported by ref. [@spiekermann2022fast]. -->
The QM9 dataset and RDB7 datasets were organized into 100 and 10 clusters, respectively.
The results tabulated below show an expected trend that average model performance is worse (MAE and RMSE are higher) when faced with more challenging extrapolation tasks.

### Average testing errors for predicting the 12 regression targets from QM9 [@ramakrishnan2014quantum].

| Split     | MAE              | RMSE            |
|-----------|------------------|-----------------|
| random    | 2.02 $\pm$ 0.06  | 3.63 $\pm$ 0.21 |
| scaffold  | 2.XX $\pm$ 0.XX  | 3.XX $\pm$ 0.XX |
| K-means   | 2.XX  $\pm$ 0.XX | 4.XX $\pm$ 0.XX |


###  Testing errors in kcal/mol for predicting a reaction's barrier height from RDB7 [@spiekermann2022high}.

| Split     | MAE             | RMSE            |
|-----------|-----------------|-----------------|
| random    | 3.94 $\pm$ 0.03 | 6.89 $\pm$ 0.24 |
| scaffold  | 4.YY $\pm$ 0.YY | 7.YY $\pm$ 0.YY |
| K-means   | 5.YY $\pm$ 1.YY | 7.YY $\pm$ 2.YY |

In each case, the performance of the model is diminished when training and testing using extrapolative sampling algorithms.
This is to be expected, since this effectively asks the model a 'harder' question when testing.
This harder question, though, is potentially more respresentatve of real world tasks.
After successful model training, the user would expect to use their model to make predictions on new, real-world data.
In the case that this new data is not within the space of the training data (which is likely not something being verified), errors could be significant.
It is critical then to know what to expect when predicting on out-of-sample data, which could indicate hyperparameter overfitting and a potential remidiation pathway for the developer.

Note that the scaffold errors presented above are higher than what is reported in the original study [@spiekermann2022fast] for two reasons: pretraining on the B97-D3 or $\omega$B97X-D3 datasets was done in the earlier study [@spiekermann2022fast] but neglected here for simplicity, and co-training with the reaction enthalpy was left off for similar reasons. <!--, which often improves model performance and is not an unexpected observation given that a reaction’s enthalpy is often correlated to its barrier height (e.g. Evans-Polanyi relationships [@evans1938inertia]). -->
<!-- If suitable pretraining data is available, transfer learning is an established technique to improve model performance [@pan2010survey]. -->
<!-- Second, we do not use ensembling here; however, this is another established method to improve model predictions [@yang2019analyzing; @dietterich2000ensemble]. -->
 <!-- Bell-Evans- Polanyi (BEP)-type correlations.14,20,21  from https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.2c01502 -->

# Related Software and Code Availability

In the machine learning space, `astartes` functions as a drop-in replacement for the ubiquitous `train_test_split` from scikit-learn [@scikit-learn].
Transitioning existing code to use this new methodology is as simple as running `pip install astartes`, modifying an `import` statement at the top of the file, and then specifying an additional keyword parameter.
`astartes` has been especially designed to allow for maximum interoperability with other packages, using few depdencies, supporting all platforms, and continuous integration testing for Python 3.7 through 3.11.
Specific tutorials on this transition are provided in the online documentation for `astartes`, which is available on [GitHub](https://jacksonburns.github.io/astartes/sklearn_to_astartes.html).

Here is an example workflow using `train_test_split` taken from the `scikit-learn` documentation [@scikit-learn]:
```python
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

To switch to using `astartes`, `from sklearn.model_selection import train_test_split` becomes `from astartes import train_test_split` and the call to split the data is nearly identical and simple in the extensions that it provides:

```python
import numpy as np
from astartes import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, sampler="kmeans") 
```

With this small change, an extrapoative sampler based on k-means clustering will be used.

Inside cheminformatics, `astartes` makes use of all molecular featurization options implemented in `AIMSim` [@aimsim_cpc], which includes those from virtually all popular descriptor generation tools used in the cheminformatics field.

The codebase itself has a clearly defined contribution guideline and thorough, easily accesible documentation. The functionality is checked nightly via GitHub actions Constant Integration testing. Test coverage currently sits at >99%, and all pull requests are automatically subject to a coverage check and merged only if they cover all existing and new lines added.


# Acknowledgements
The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing computing resources that have contributed to the research results reported within this paper [@reuther2018interactive].

The authors thank all users who participated in beta testing and release candidate testing throughout the development of `astartes`.

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0023112.

<!-- The below section and text are required by Jackson Burns' funding agency, the DOE CSGF. -->
# Disclaimer
This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<!--
# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->
# References