# Auto-ViML

![banner](logo.png)<br>
Automatically Build Various Interpretable ML models fast!<br>
[![Downloads](https://pepy.tech/badge/autoviml/week)](https://pepy.tech/project/autoviml/week)
[![Downloads](https://pepy.tech/badge/autoviml/month)](https://pepy.tech/project/autoviml/month)
[![Downloads](https://pepy.tech/badge/autoviml)](https://pepy.tech/project/autoviml)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoviml.svg?logo=python&logoColor=white)](https://pypi.org/project/autoviml)
[![PyPI Version](https://img.shields.io/pypi/v/autoviml.svg?logo=pypi&logoColor=white)](https://pypi.org/project/autoviml)
[![PyPI License](https://img.shields.io/pypi/l/autoviml.svg)](https://github.com/AutoViML/Auto_ViML/blob/master/LICENSE)

Auto_ViML is pronounced as "auto vimal" (autovimal logo created by Sanket Ghanmare).

## Update (Jan 2025)
<ol>
<li><b>Auto_ViML is now upgraded to version 0.2 </b>which means it now runs on Python 3.12 or greater and also pandas 2.0 - this is a huge upgrade to those working in Colabs, Kaggle and other latest kernels. Please make sure you check the `requirements.txt` file to know which versions are recommended.</li>
</ol>

## Update (March 2023)
<ol>
<li>Auto_ViML has a new flag to speed up processing using GPU's. Just set the `GPU_flag`=`True` on Colab and other environments. But don't forget to set the runtime type to be "GPU" while running on Colab. Otherwise you will get an error.</li>
</ol>

## Update (May 2022)
<ul>
<li><b>Auto_ViML as of version 0.1.710 uses a very high performance library called `imbalanced_ensemble` for Imbalanced dataset problems.</b> It will produce a 5-10% boost in your balanced_accuracy based on my experience with  many datasets I have tried.</li>
</ul>
<br>
<p>In addition, new features in this version are:<br>
<ul>
<li>SULOV -> Uses the SULOV algorithm for removing highly correlated features automatically.</li>
<li>Auto_NLP -> AutoViML automatically detects Text variables and does NLP processing using Auto_NLP</li>
<li>Date Time -> It automatically detects  date time variables and generates new features</li>
<li>`imbalanced_ensemble` library -> Uses imbalanced_ensemble library for imbalanced data. Just set Imbalanced_Flag = True in arguments</li>
<li>Feature Selection -> We use the same algorithm that featurewiz library uses: SULOV and Recursive XGBoost to select best features fast. See below.</li>
</ul>

## Table of Contents
<ul>
<li><a href="#background">Background</a></li>
<li><a href="#install">Install</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#tips-for-using-auto_viml">Tips for using Auto_ViML</a></li>
<li><a href="#api">API</a></li>
<li><a href="#maintainers">Maintainers</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
</ul>

## Background
<p>Read this <a href="https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46">Medium article to learn how to use Auto_ViML effectively.</a>

<p>Auto_ViML was designed for building High Performance Interpretable Models with the fewest variables needed.
The "V" in Auto_ViML stands for Variant because it tries multiple models with multiple features to find you the best performing model for your dataset. The "i" in Auto_ViML stands for "interpretable" since Auto_ViML selects the least number of features necessary to build a simpler, more interpretable model. In most cases, Auto_ViML builds models with 20%-99% fewer features than a similar performing model with all included features (this is based on my trials. Your experience may vary).<br>
<p>
Auto_ViML is every Data Scientist's model accelerator tool that:
<ol>
<li><b>Helps you with data cleaning</b>: you can send in your entire dataframe as is and Auto_ViML will suggest changes to help with missing values, formatting variables, adding variables, etc. It loves dirty data. The dirtier the better!</li>
<li><b>Performs Feature Selection</b>: Auto_ViML selects variables automatically by default. This is very helpful when you have hundreds if not thousands of variables since it can readily identify which of those are important variables vs which are unnecessary. You can turn this off as well (see API).</li>

![xgboost](sulov_xgboost.png)
<li><b>Removes highly correlated features automatically</b>. If two variables are highly correlated in your dataset, which one should you remove and which one should you keep? The decision is not as easy as it looks. Auto_ViML uses the SULOV algorithm to remove highly correlated features. You can understand SULOV from this picture more intuitively.</li>

![sulov](SULOV.jpg)

<li><b>Generates performance results graphically</b>. Just set verbose = 1 (or) 2 instead of 0 (silent). You will get higher quality of insights as you increase verbosity. </li>
<li><b>Handles text, date-time, structs (lists, dictionaries), numeric, boolean, factor and categorical</b> variables all in one model using one straight process.</li>
</ol>
Auto_ViML is built using scikit-learn, numpy, pandas and matplotlib. It should run on most Python 3 Anaconda installations. You won't have to import any special libraries other than "XGBoost", "Imbalanced-Learn", "CatBoost", and "featuretools" library. We use "SHAP" library for interpretability. <br>But if you don't have these libraries, Auto_ViML will install those for you automatically.

## Install

**Prerequsites:**

- [Anaconda](https://docs.anaconda.com/anaconda/install/)

To clone Auto_ViML, it is better to create a new environment, and install the required dependencies:

To install from PyPi:
<p>
<code>$ pip install autoviml --upgrade --ignore-installed</code><br />


```
pip install git+https://github.com/AutoViML/Auto_ViML.git
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

## Tips for using Auto_ViML:
1. `scoring_parameter`: For Classification problems and imbalanced classes, choose scoring_parameter="balanced_accuracy". It works better.
2. `Imbalanced_Flag`: For Imbalanced Classes (<5% samples in rare class), choose "Imbalanced_Flag"=True. You can also set this flag to True for Regression problems where the target variable might have skewed distributions.
3. `target`: For Multi-Label dataset, the target input target variable can be sent in as a list of variables.
4. `Boosting_Flag`: It is recommended that you first set Boosting_Flag=None to get a Linear model. Once you understand that, then you can try to set Boosting_Flag=False to get a Random Forest model. Finally, try Boosting_Flag=True to get an XGBoost model. This is the order that we recommend in order to use Auto_ViML. Finally try Boosting_Flag="CatBoost" to get a complex but high performing model.
5. `Binning_Flag`: Binning_Flag=True improves a CatBoost model since it adds to the list of categorical vars in data
6. `KMeans_featurizer`: KMeans_featurizer=True works well in NLP and CatBoost models since it creates cluster variables
7. `Add_Poly`: Add_Poly=3 improves certain models where there is date-time or categorical and numeric variables
8. `feature_reduction`: feature_reduction=True is the default and works best. But when you have <10 features in data, set it to False
9. `Stacking_Flag`: Do not set Stacking_Flag=True with Linear models since your results may not look great.
10. `Stacking_Flag`: Use Stacking_Flag=True only for complex models and as a last step with Boosting_Flag=True or CatBoost
11. `hyper_param`: Leave hyper_param ="RS" as input since it runs faster than GridSearchCV and gives better results unless you have a small data set and can afford to spend time on hyper tuning.
12. `KMeans_Featurizer`: KMeans_Featurizer=True does not work well for small data sets. Use it for data sets > 10,000 rows.
13. `Final thoughts`: Finally Auto_ViML is meant to be a baseline or challenger solution to your data set. So use it for making quick models that you can compare against or in Hackathons. It is not meant for production!

## API

**Arguments**

- `train`: could be a datapath+filename or a dataframe. It will detect which is which and load it.
- `test`: could be a datapath+filename or a dataframe. If you don't have any, just leave it as "".
- `submission`: must be a datapath+filename. If you don't have any, just leave it as empty string.
- `target`: name of the target variable in the data set.
- `sep`: if you have a spearator in the file such as "," or "\t" mention it here. Default is ",".
- `scoring_parameter`: if you want your own scoring parameter such as "f1" give it here. If not, it will assume the appropriate scoring param for the problem and it will build the model.
- `hyper_param`: Tuning options are GridSearch ('GS') and RandomizedSearch ('RS'). Default is 'RS'.
- `feature_reduction`: Default = 'True' but it can be set to False if you don't want automatic feature_reduction since in Image data sets like digits and MNIST, you get better results when you don't reduce features automatically. You can always try both and see.
- `KMeans_Featurizer`
  - `True`: Adds a cluster label to features based on KMeans. Use for Linear.
  - `False (default)` For Random Forests or XGB models, leave it False since it may overfit.
- `Boosting Flag`: you have 4 possible choices (default is False):
  - `None` This will build a Linear Model
  - `False` This will build a Random Forest or Extra Trees model (also known as Bagging)
  - `True` This will build an XGBoost model
  - `CatBoost` This will build a CatBoost model (provided you have CatBoost installed)
- `Add_Poly`: Default is 0 which means do-nothing. But it has three interesting settings:
  - `1` Add interaction variables only such as x1*x2, x2*x3,...x9\*10 etc.
  - `2` Add Interactions and Squared variables such as x1**2, x2**2, etc.
  - `3` Adds both Interactions and Squared variables such as x1*x2, x1**2,x2*x3, x2**2, etc.
- `Stacking_Flag`: Default is False. If set to True, it will add an additional feature which is derived from predictions of another model. This is used in some cases but may result in overfitting. So be careful turning this flag "on".
- `Binning_Flag`: Default is False. It set to True, it will convert the top numeric variables into binned variables through a technique known as "Entropy" binning. This is very helpful for certain datasets (especially hard to build models).
- `Imbalanced_Flag`: Default is False. Uses imbalanced_ensemble library for imbalanced data. Just set Imbalanced_Flag = True in arguments
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
