# Auto-ViML

![banner](logo.png)

[![Downloads](https://pepy.tech/badge/autoviml/week)](https://pepy.tech/project/autoviml/week)
[![Downloads](https://pepy.tech/badge/autoviml/month)](https://pepy.tech/project/autoviml/month)
[![Downloads](https://pepy.tech/badge/autoviml)](https://pepy.tech/project/autoviml)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoviml.svg?logo=python&logoColor=white)](https://pypi.org/project/autoviml)
[![PyPI Version](https://img.shields.io/pypi/v/autoviml.svg?logo=pypi&logoColor=white)](https://pypi.org/project/autoviml)
[![PyPI License](https://img.shields.io/pypi/l/autoviml.svg)](https://github.com/AutoViML/Auto_ViML/blob/master/LICENSE)

Automatically Build Variant Interpretable ML models fast!
Auto_ViML is pronounced "auto vimal". Read this [medium article to learn how to use Auto_ViML](https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

Auto_ViML was designed for building High Performance Interpretable Models with the least variables.
The "V" in Auto_ViML stands for Variant because it tries Multiple Models and Multiple Features to find the best performing model for any dataset. The "i" in Auto_ViML stands for "Interpretable" since it selects the fewest features to build a simpler, more interpretable model. This is key.

Auto_ViML is built using Scikit-Learn, Numpy, Pandas and Matplotlib. It should run
on any Python 2 or Python 3 Anaconda installations. You won't have to import any special
Libraries other than "CatBoost" and "SHAP" library for interpretability.
But if you don't have these Auto_ViML will skip it and show you the regular feature importances.

## Install

**Prerequsites:**

- [Anaconda](https://docs.anaconda.com/anaconda/install/)

To clone Auto_ViML, it is better to create a new environment, and install the required dependencies:

To install from PyPi:

```
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
pip install autoviml
```

To install from source:

```
cd <AutoVIML_Destination>
git clone git@github.com:AutoViML/Auto_ViML.git
# or download and unzip https://github.com/AutoViML/Auto_ViML/archive/master.zip
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
cd Auto_ViML
pip install -r requirements.txt
```

## Usage

In the same directory, open a Jupyter Notebook and use this line to import the .py file:

```
from autoviml.Auto_ViML import Auto_ViML
```

Load a data set (any CSV or text file) into a Pandas dataframe and split it into Train and Test dataframes. If you don't have a test dataframe, you can simple assign the test variable below to '' (empty string):

```
model, features, trainm, testm = Auto_ViML(
    train,
    target,
    test,
    sample_submission,
    hyper_param="GS",
    feature_reduction=True,
    scoring_parameter="weighted-f1",
    KMeans_Featurizer=False,
    Boosting_Flag=False,
    Binning_Flag=False,
    Add_Poly=False,
    Stacking_Flag=False,
    Imbalanced_Flag=False,
    verbose=0,
)
```

Finally, it writes your submission file to disk in the current directory called `mysubmission.csv`.
This submission file is ready for you to show it clients or submit it to competitions.
If no submission file was given, but as long as you give it a test file name, it will create a submission file for you named `mySubmission.csv`.
Auto_ViML works on any Multi-Class, Multi-Label Data Set. So you can have many target labels.
You don't have to tell Auto_ViML whether it is a Regression or Classification problem.

### Additional Notes

**Suggestions for a Scoring Metric:**
If you have Binary Class and Multi-Class in a Single Label, Choose Accuracy. It will do very well. If you want something better, try roc_auc even for Multi-Class which works.
You can try F1 or Weighted F1 if you want something complex or for Multi-Class.

Note that For Imbalanced Classes (<=5% classes), it automatically adds Class Weights.

Also note that it handles Multi-Label automatically so you can send Train data with multiple Labels (Targets) and it will automatically predict for each Label.

Finally this is Meant to Be a Fast Algorithm, so use it for just quick POCs.
This is Not Meant for Production Problems. It produces great models but it is not Perfect!

## API

**Arguments**

- `train`: could be a datapath+filename or a dataframe. It will detect which is which and load it.
- `test`: could be a datapath+filename or a dataframe. If you don't have any, just leave it as "".
- `submission`: must be a datapath+filename. If you don't have any, just leave it as empty string.
- `target`: name of the target variable in the data set.
- `sep`: if you have a spearator in the file such as "," or "\t" mention it here. Default is ",".
- `scoring_parameter`: if you want your own scoring parameter such as "f1" give it here. If not, it will assume the appropriate scoring param for the problem and it will build the model.
- `hyper_param`: Tuning options are GridSearch ('GS') and RandomizedSearch ('RS'). Default is 'GS'.
- `feature_reduction`: Default = 'True' but it can be set to False if you don't want automatic feature_reduction since in Image data sets like digits and MNIST, you get better results when you don't reduce features automatically. You can always try both and see.
- `KMeans_Featurizer`
  - `True`: Adds a cluster label to features based on KMeans. Use for Linear.
  - `False (default)` For Random Forests or XGB models, leave it False since it may overfit.
- `Boosting Flag`: you have 4 possible choices (default is False):
  - `None` This will build a Linear Model
  - `False` This will build a Random Forest or Extra Trees model (also known as Bagging)
  - `True` This will build an XGBoost model
  - `CatBoost` This will build a CatBoost model (provided you have CatBoost installed)
- `Add_Poly`: Default is 0. It has 2 additional settings:
  - `1` Add interaction variables only such as x1*x2, x2*x3,...x9\*10 etc.
  - `2` Add Interactions and Squared variables such as x1**2, x2**2, etc.
- `Stacking_Flag`: Default is False. If set to True, it will add an additional feature which is derived from predictions of another model. This is used in some cases but may result in overfitting. So be careful turning this flag "on".
- `Binning_Flag`: Default is False. It set to True, it will convert the top numeric variables into binned variables through a technique known as "Entropy" binning. This is very helpful for certain datasets (especially hard to build models).
- `Imbalanced_Flag`: Default is False. If set to True, it will downsample the "Majority Class" in an imbalanced dataset and make the "Rare" class at least 5% of the data set. This the ideal threshold in my mind to make a model learn. Do it for Highly Imbalanced data.
- `verbose`: This has 3 possible states:
  - `0` limited output. Great for running this silently and getting fast results.
  - `1` more charts. Great for knowing how results were and making changes to flags in input.
  - `2` lots of charts and output. Great for reproducing what Auto_ViML does on your own.

**Return values**

- `model`: It will return your trained model
- `features`: the fewest number of features in your model to make it perform well
- `train_modified`: this is the modified train dataframe after removing and adding features
- `test_modified`: this is the modified test dataframe with the same transformations as train

## Maintainers

* [@AutoViML](https://github.com/AutoViML)
* [@morenoh149](https://github.com/morenoh149)
* [@hironroy](https://github.com/hironroy)

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

## License

Apache License 2.0 © 2020 Ram Seshadri

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.
