##############################################################################
#Copyright 2019 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
################################################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from numpy import inf
################################################################################
import pdb
######## Import some multi-output models #######################################
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.multioutput import RegressorChain, ClassifierChain
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (10, 6)
from sklearn.metrics import classification_report, confusion_matrix
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from autoviml.QuickML_Stacking import QuickML_Stacking
from autoviml.Transform_KM_Features import Transform_KM_Features
from autoviml.QuickML_Ensembling import QuickML_Ensembling
from autoviml.Auto_NLP import Auto_NLP

import sys
##################################################################################
def find_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    counts = OrderedDict(Counter(classes))
    total = sum(counts.values())
    if verbose >= 1:
        print('       Class  -> Counts -> Percent')
        sorted_keys = sorted(counts.keys())
        for cls in sorted_keys:
            print("%12s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
###############################################################################
def return_factorized_dict(ls):
    """
    ######  Factorize any list of values in a data frame using this neat function
    if your data has any NaN's it automatically marks it as -1 and returns that for NaN's
    Returns a dictionary mapping previous values with new values.
    """
    factos = pd.unique(pd.factorize(ls)[0])
    categs = pd.unique(pd.factorize(ls)[1])
    if -1 in factos:
        categs = np.insert(categs,np.where(factos==-1)[0][0],np.nan)
    return dict(zip(categs,factos))
#############################################################################################
from sklearn.metrics import confusion_matrix
def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score
#############################################################################################
import os
def check_if_GPU_exists():
    GPU_exists = False
    try:
        from tensorflow.python.client import device_lib
        dev_list = device_lib.list_local_devices()
        print('Number of GPUs = %d' %len(dev_list))
        for i in range(len(dev_list)):
            if 'GPU' == dev_list[i].device_type:
                GPU_exists = True
                print('%s available' %dev_list[i].device_type)
    except:
        print('')
    if not GPU_exists:
        try:
            os.environ['NVIDIA_VISIBLE_DEVICES']
            print('    GPU active on this device')
            return True
        except:
            print('    No GPU active on this device')
            return False
    else:
        return True
#############################################################################################
def modify_array_to_integer(y_pred, negative_flag=False):
    """
    Simiple utility to plug all negative values with 0 and turn them into integer
    """
    y_pred = y_pred.astype(int)
    if negative_flag:
        y_pred[y_pred<0] = 0
    return y_pred
##########################################################################
def analyze_problem_type(train, target, verbose=0) :
    target = copy.deepcopy(target)
    cat_limit = 30 ### this determines the number of categories to name integers as classification ##
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    if isinstance(target, str):
        target = [target]
    if len(target) == 1:
        targ = target[0]
    else:
        targ = target[0]
    ####  This is where you detect what kind of problem it is #################
    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Binary_Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    if verbose <= 1:
        print('''\n################ %s VISUALIZATION Started #####################''' %model_class)
    return model_class
###################################################################################
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
def split_train_into_two(train_data, target, randomstate, modeltype):
    preds = [x for x in list(train_data) if x not in target]
    if modeltype.endswith('Classification'):
        if len(target) == 1:
            X = train_data[preds]
            y = train_data[target]
            if train_data.shape[0] <= 1000:
                sss = StratifiedShuffleSplit(n_splits=2, test_size=0.10, random_state=randomstate)
            else:
                sss = StratifiedShuffleSplit(n_splits=2, test_size=0.20, random_state=randomstate)
            trainindex, testindex = sss.split(X, y)
            traindf = train_data.iloc[trainindex[0]]
            testdf = train_data.iloc[trainindex[1]]
        else:
            print('Splitting a multi-label dataset into train and validation')
            mlb = MultiLabelBinarizer()
            X = train_data[preds]
            Y = pd.DataFrame(mlb.fit_transform(train_data[target].values), columns=mlb.classes_, index=train_data.index)
            traindf, testdf = train_test_split(train_data, test_size=0.2,shuffle=True, stratify=Y, random_state=99)
    else:
        indices = np.random.permutation(train_data.shape[0])
        if len(indices) <= 1000:
            train_rows = int(0.9*train_data.shape[0])
        else:
            train_rows = int(0.8*train_data.shape[0])
        training_idx, test_idx = indices[:train_rows], indices[train_rows:]
        traindf, testdf = train_data.iloc[training_idx], train_data.iloc[test_idx]
    return traindf, testdf
#######################################################################################
def fill_missing_values_object_or_number(start_train, start_test, fill_num, col,str_flag=True):
    """
    ####  This is the easiest way to fill missing values using an object in both train and test
    #### This takes care of some categories that are present in train and not in test
    ###     and vice versa
    """
    start_train = copy.deepcopy(start_train)
    start_test = copy.deepcopy(start_test)
    missing_flag = False
    new_missing_col = ''
    if start_train[col].isnull().sum() > 0:
        missing_flag = True
        new_missing_col = col + '_Missing_Flag'
        start_train[new_missing_col] = 0
        start_train.loc[start_train[col].isnull(),new_missing_col]=1
        start_train[col] = start_train[col].fillna(fill_num)
    if type(start_test) != str:
        if missing_flag:
            start_test[new_missing_col] = 0
        if start_test[col].isnull().sum() > 0:
            start_test.loc[start_test[col].isnull(),new_missing_col]=1
            start_test[col] = start_test[col].fillna(fill_num)
    return start_train, start_test, missing_flag, new_missing_col
#####################################################################################
def convert_train_test_cat_col_to_numeric(start_train, start_test, col,str_flag=True):
    """
    ####  This is the easiest way to label encode object variables in both train and test
    #### This takes care of some categories that are present in train and not in test
    ###     and vice versa
    """
    start_train = copy.deepcopy(start_train)
    start_test = copy.deepcopy(start_test)
    missing_flag = False
    new_missing_col = ''
    if start_train[col].isnull().sum() > 0:
        missing_flag = True
        if str_flag:
            new_missing_col = col + '_Missing_Flag'
            start_train[new_missing_col] = 0
            start_train.loc[start_train[col].isnull(),new_missing_col]=1
            start_train[col] = start_train[col].fillna("NA", inplace=False).astype(str)
        else:
            new_missing_col = col + '_Missing_Flag'
            start_train[new_missing_col] = 0
            start_train.loc[start_train[col].isnull(),new_missing_col]=1
            start_train[col] = start_train[col].fillna("NA", inplace=False).astype('category')
    if len(start_train[col].apply(type).value_counts()) > 1:
        print('    Alert! Mixed Data Types in Train data set %s column with %d data types. Fixing it...' %(
                                       col, len(start_train[col].apply(type).value_counts())))
        train_categs = start_train[col].value_counts().index.tolist()
    else:
        train_categs = np.unique(start_train[col]).tolist()
    if not isinstance(start_test,str) :
        if start_test[col].isnull().sum() > 0:
            #### IN some rare cases, Test data has missing values while Train data doesn.t
            #### This section is to take care of such rare cases. We need to create a missing col
            ####  We need to create that missing flag column in both train and test in that case
            if not missing_flag:
                missing_flag = True
                new_missing_col = col + '_Missing_Flag'
                start_train[new_missing_col] = 0
            #####  THis is to take care of Missing_Flag in start_test data set!!
            start_test[new_missing_col] = 0
            start_test.loc[start_test[col].isnull(),new_missing_col]=1
            if str_flag:
                start_test[col] = start_test[col].fillna("NA", inplace=False).astype(str)
            else:
                start_test[col] = start_test[col].fillna("NA", inplace=False).astype('category')
        else:
            #### In some rare cases, there is missing values in train but not in test data!
            #### In those cases, we need to create a new_missing_col in test data in addition to train
            start_test[new_missing_col] = 0
        if len(start_test[col].apply(type).value_counts()) > 1:
            print('    Alert! Mixed Data Types in Test data set %s column with %d data types. Fixing it...' %(
                                           col, len(start_test[col].apply(type).value_counts())))
            test_categs = start_test[col].value_counts().index.tolist()
            test_categs = [x if isinstance(x,str)  else str(x) for x in test_categs]
            start_test[col] = start_test[col].astype(str).values
        else:
            test_categs = np.unique(start_test[col]).tolist()
    if not isinstance(start_test,str) :
        categs_all = np.unique( train_categs + test_categs).tolist()
        dict_all =  return_factorized_dict(categs_all)
    else:
        dict_all = return_factorized_dict(train_categs)
    start_train[col] = start_train[col].map(dict_all)
    if not isinstance(start_test,str) :
        start_test[col] = start_test[col].map(dict_all)
    return start_train, start_test, missing_flag, new_missing_col
#############################################################################################################
def flatten_list(list_of_lists):
    final_ls = []
    for each_item in list_of_lists:
        if isinstance(each_item,list):
            final_ls += each_item
        else:
            final_ls.append(each_item)
    return final_ls
#############################################################################################################
from pandas.api.types import is_datetime64_any_dtype as is_datetime
#############################################################################################################
def Auto_ViML(train, target, test='',sample_submission='',hyper_param='RS', feature_reduction=True,
            scoring_parameter='logloss', Boosting_Flag=None, KMeans_Featurizer=False,
            Add_Poly=0, Stacking_Flag=False, Binning_Flag=False,
              Imbalanced_Flag=False, verbose=0):
    """
    #########################################################################################################
    #############       This is not an Officially Supported Google Product!         #########################
    #########################################################################################################
    ####       Automatically Build Variant Interpretable Machine Learning Models (Auto_ViML)           ######
    ####                                Developed by Ramadurai Seshadri                                ######
    #####   GPU UPGRADE!! Now with Auto_NLP. Best Version to Download or Upgrade.  July 24,2020         ######
    ######          Auto_VIMAL with Auto_NLP combines structured data with NLP for Predictions.       #######
    #########################################################################################################
    #Copyright 2019 Google LLC                                                                        #######
    #                                                                                                 #######
    #Licensed under the Apache License, Version 2.0 (the "License");                                  #######
    #you may not use this file except in compliance with the License.                                 #######
    #You may obtain a copy of the License at                                                          #######
    #                                                                                                 #######
    #    https://www.apache.org/licenses/LICENSE-2.0                                                  #######
    #                                                                                                 #######
    #Unless required by applicable law or agreed to in writing, software                              #######
    #distributed under the License is distributed on an "AS IS" BASIS,                                #######
    #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         #######
    #See the License for the specific language governing permissions and                              #######
    #limitations under the License.                                                                   #######
    #########################################################################################################
    ####   Auto_ViML was designed for building a High Performance Interpretable Model With Fewest Vars.   ###
    ####   The "V" in Auto_ViML stands for Variant because it tries Multiple Models and Multiple Features ###
    ####   to find the Best Performing Model for any data set.The "i" in Auto_ViML stands " Interpretable"###
    ####   since it selects the fewest Features to build a simpler, more interpretable model. This is key. ##
    ####   Auto_ViML is built mostly using Scikit-Learn, Numpy, Pandas and Matplotlib. Hence it should run ##
    ####   on any Python 2 or Python 3 Anaconda installations. You won't have to import any special      ####
    ####   Libraries other than "SHAP" library for SHAP values which provides more interpretability.    #####
    ####   But if you don't have it, Auto_ViML will skip it and show you the regular feature importances. ###
    #########################################################################################################
    ####   INPUTS:                                                                                        ###
    #########################################################################################################
    ####   train: could be a datapath+filename or a dataframe. It will detect which is which and load it.####
    ####   test: could be a datapath+filename or a dataframe. If you don't have any, just leave it as "". ###
    ####   submission: must be a datapath+filename. If you don't have any, just leave it as empty string.####
    ####   target: name of the target variable in the data set.                                          ####
    ####   sep: if you have a spearator in the file such as "," or "\t" mention it here. Default is ",". ####
    ####   scoring_parameter: if you want your own scoring parameter such as "f1" give it here. If not, #####
    ####       it will assume the appropriate scoring param for the problem and it will build the model.#####
    ####   hyper_param: Tuning options are GridSearch ('GS'), RandomizedSearch ('RS')and now HyperOpt ('HO')#
    ####        Default setting is 'GS'. Auto_ViML with HyperOpt is approximately 3X Faster than Auto_ViML###
    ####   feature_reduction: Default = 'True' but it can be set to False if you don't want automatic    ####
    ####         feature_reduction since in Image data sets like digits and MNIST, you get better       #####
    ####         results when you don't reduce features automatically. You can always try both and see. #####
    ####   KMeans_Featurizer = True: Adds a cluster label to features based on KMeans. Use for Linear.  #####
    ####         False (default) = For Random Forests or XGB models, leave it False since it may overfit.####
    ####   Boosting Flag: you have 3 possible choices (default is False):                               #####
    ####    None = This will build a Linear Model                                                       #####
    ####    False = This will build a Random Forest or Extra Trees model (also known as Bagging)        #####
    ####    True = This will build an XGBoost model                                                     #####
    ####   Add_Poly: Default is 0. It has 2 additional settings:                                        #####
    ####    1 = Add interaction variables only such as x1*x2, x2*x3,...x9*10 etc.                       #####
    ####    2 = Add Interactions and Squared variables such as x1**2, x2**2, etc.                       #####
    ####   Stacking_Flag: Default is False. If set to True, it will add an additional feature which     #####
    ####         is derived from predictions of another model. This is used in some cases but may result#####
    ####         in overfitting. So be careful turning this flag "on".                                  #####
    ####   Binning_Flag: Default is False. It set to True, it will convert the top numeric variables    #####
    ####         into binned variables through a technique known as "Entropy" binning. This is very     #####
    ####         helpful for certain datasets (especially hard to build models).                        #####
    ####   Imbalanced_Flag: Default is False. If set to True, it will downsample the "Majority Class"   #####
    ####         in an imbalanced dataset and make the "Rare" class at least 5% of the data set. This   #####
    ####         the ideal threshold in my mind to make a model learn. Do it for Highly Imbalanced data.#####
    ####   verbose: This has 3 possible states:                                                         #####
    ####    0 = limited output. Great for running this silently and getting fast results.               #####
    ####    1 = more charts. Great for knowing how results were and making changes to flags in input.   #####
    ####    2 = lots of charts and output. Great for reproducing what Auto_ViML does on your own.       #####
    #########################################################################################################
    ####   OUTPUTS:                                                                                     #####
    #########################################################################################################
    ####   model: It will return your trained model                                                     #####
    ####   features: the fewest number of features in your model to make it perform well                #####
    ####   train_modified: this is the modified train dataframe after removing and adding features      #####
    ####   test_modified: this is the modified test dataframe with the same transformations as train    #####
    #################               A D D I T I O N A L    N O T E S                              ###########
    ####   Finally, it writes your submission file to disk in the current directory called "mysubmission.csv"
    ####   This submission file is ready for you to show it clients or submit it to competitions.       #####
    ####   If no submission file was given but as long as you give it a test file name, it will create  #####
    ####   a submission file for you named "mySubmission.csv".                                          #####
    ####   Auto_ViML works on any Multi-Class, Multi-Label Data Set. So you can have many target labels #####
    ####   You don't have to tell Auto_ViML whether it is a Regression or Classification problem.       #####
    ####   Suggestions for a Scoring Metric:                                                            #####
    ####   If you have Binary Class and Multi-Class in a Single Label, Choose Accuracy. It will        ######
    ####   do very well. If you want something better, try roc_auc even for Multi-Class which works.   ######
    ####   You can try F1 or Weighted F1 if you want something complex or for Multi-Class.             ######
    ####   Note that For Imbalanced Classes (<=5% classes), it automatically adds Class Weights.       ######
    ####   Also, Note that it handles Multi-Label automatically so you can send Train data             ######
    ####   with multiple Labels (Targets) and it will automatically predict for each Label.            ######
    ####   Finally this is Meant to Be a Fast Algorithm, so use it for just quick POCs                 ######
    ####   This is Not Meant for Production Problems. It produces great models but it is not Perfect!  ######
    ######################### HELP OTHERS! PLEASE CONTRIBUTE! OPEN A PULL REQUEST! ##########################
    #########################################################################################################
    """
    np.random.seed(42)
    #####   These copies are to make sure that the originals are not destroyed ####
    CPU_count = os.cpu_count()
    test = copy.deepcopy(test)
    orig_train = copy.deepcopy(train)
    orig_test = copy.deepcopy(test)
    train_index = train.index
    if not isinstance(test, str):
        test_index = test.index
    start_test = copy.deepcopy(orig_test)
    #######    These are Global Settings. If you change them here, it will ripple across the whole code ###
    corr_limit = 0.70   #### This decides what the cut-off for defining highly correlated vars to remove is.
    scaling = 'MinMax' ### This decides whether to use MinMax scaling or Standard Scaling ("Std").
    first_flag = 0  ## This is just a setting to detect which is
    seed= 99  ### this maintains repeatability of the whole ML pipeline here ###
    subsample=0.7 #### Leave this low so the models generalize better. Increase it if you want overfit models
    col_sub_sample = 0.7   ### Leave this low for the same reason above
    poly_degree = 2  ### this create 2-degree polynomial variables in Add_Poly. Increase if you want more degrees
    booster = 'gbtree'   ### this is the booster for XGBoost. The other option is "Linear".
    n_splits = 5  ### This controls the number of splits for Cross Validation. Increasing will take longer time.
    matplotlib_flag = True #(default) This is for drawing SHAP values. If this is False, initJS is used.
    early_stopping = 20 #### Early stopping rounds for XGBoost ######
    encoded = '_Label_Encoded' ### This is the tag we add to feature names in the end to indicate they are label encoded
    catboost_limit = 0.4 #### The catboost_limit represents the percentage of num vars in data. ANy lower, CatBoost is used.
    cat_code_limit = 100 #### If the number of dummy variables to create in a data set exceeds this, CatBoost is the default Algorithm used
    one_hot_size = 500 #### This determines the max length of one_hot_max_size parameter of CatBoost algrithm
    Alpha_min = -3 #### The lowest value of Alpha in LOGSPACE that is used in CatBoost
    Alpha_max = 2 #### The highest value of Alpha in LOGSPACE that is used in Lasso or Ridge Regression
    #Cs = np.linspace(0.0001,0.9,5).tolist()+np.linspace(1,100,10).tolist()
    C_min = 10e-5  ### The lowest value of C used in Logistic Regression
    C_max = 100  ### The highest value of C used in Logistic Regression
    #Cs = np.linspace(C_min,C_max,100)
    Cs = np.logspace(-4,3,40) ### The list of values of C used in Logistic Regression
    tolerance = 0.001 #### This tolerance is needed to speed up Logistic Regression. Otherwise, SAGA takes too long!!
    #### 'lbfgs' is the fastest one but doesnt provide accurate results. Newton-CG is slower but accurate!
    #### SAGA is extremely slow. Even slower than Newton-CG. Liblinear is the fastest and as accurate as Newton-CG!
    solvers = ['liblinear'] ### Other solvers for Logistic Regression model: ['newton-cg','lbfgs','saga','liblinear']
    solver = 'liblinear'  ### lbfgs is the next fastest solver after liblinear. Useful for Multi-class problems!
    penalties = ['l2','l1'] ### This is to determine the penalties for LogisticRegression
    n_steps = 6 ### number of estimator steps between 100 and max_estims
    max_depth = 8 ##### This limits the max_depth used in decision trees and other classifiers
    max_features = 20 #### maximum number of features in a random forest model or extra trees model
    warm_start = True ### This is to set the warm_start flag for the ExtraTrees models
    bootstrap = True #### Set this flag to control whether to bootstrap variables or not.
    n_repeats = 3 #### This is for repeated KFold and StratifiedKFold - this changes the folds every time
    Bins = 30 ### This is for plotting probabilities in a histogram. For small data sets, 30 is enough.
    top_nlp_features = 300 ### This sets a limit on the number of features added by each NLP transformer!
    removed_features_threshold = 5 #### This triggers the Truncated_SVD if number of removed features from XGB exceeds this!
    calibrator_flag = False  ### In Multi-class data sets, a CalibratedClassifier works better than regular classifiers!
    max_class_length = 1 ### It turns out the number of classes is directly correlated to Estimated Time. Hence this!
    perform_scaling_flag = True #### For linear models, scaling is a must but for other models, its not necessary but okay.
    xgb_booster = 'dart'  ### you can set this to either of dart, gbtree or gblinear
    max_neighbors = 100 #### this is max number of neighbors used in k_neighbors for RandomizedSearchCV algo
    n_jobs = -1 #### In case of KNN, having n_jobs=-1 gives an error on Windows environments. Hence avoid.
    chain_flag = False
    print('##############  D A T A   S E T  A N A L Y S I S  #######################')
    ##########  I F   CATBOOST  IS REQUESTED, THEN CHECK IF IT IS INSTALLED #######################
    if hyper_param is not None:
        if hyper_param.lower() == 'chain':
            chain_flag = True
    if isinstance(Boosting_Flag,str):
        if Boosting_Flag.lower() == 'catboost':
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
                ### You don't want to do scaling on Catboost since integers can become float and it will error in training
                perform_scaling_flag = False
            except:
                print('Not installed CatBoost on this machine. pip install catboost before trying Auto_ViML')
    #### Similarly for Random Forests Model, it takes too long with Grid Search, so MAKE IT RandomizedSearch!
    if not Boosting_Flag:  ### there is also a chance Boosting_Flag is None - This is to eliminate that chance!
        if orig_train.shape[0] >= 10000:
            hyper_param = 'RS'
            print('ALERT! Changing hyperparameter search to RS. Otherwise, Random Forests will take too long for 10,000+ rows')
    elif Boosting_Flag: ### there is also a chance Boosting_Flag is None - This is to eliminate that chance!
        if not isinstance(Boosting_Flag, str):
            if orig_train.shape[0] >= 10000:
                hyper_param = 'RS'
                print('ALERT! Changing hyperparameter search to RS. Otherwise XGBoost will take too long for 10,000+ rows.')
    ###########    T H I S   I S  W H E R E   H Y P E R O P T    P A R A M S  A R E   S E T #########
    if hyper_param == 'HO':
        ########### HyperOpt related objective functions are defined here #################
        from hyperopt import Trials
        from autoviml.custom_scores_HO import gini_meae, gini_sklearn, rmse
        from autoviml.custom_scores_HO import gini_msle, gini_mae, gini_mse, gini_rmse
        from autoviml.custom_scores_HO import gini_accuracy, gini_bal_accuracy, gini_roc
        from autoviml.custom_scores_HO import gini_precision, gini_average_precision, gini_weighted_precision
        from autoviml.custom_scores_HO import gini_macro_precision, gini_micro_precision
        from autoviml.custom_scores_HO import gini_samples_precision, gini_f1, gini_weighted_f1
        from autoviml.custom_scores_HO import gini_macro_f1, gini_micro_f1, gini_samples_f1,f2_measure
        from autoviml.custom_scores_HO import gini_log_loss, gini_recall, gini_weighted_recall
        from autoviml.custom_scores_HO import gini_samples_recall, gini_macro_recall, gini_micro_recall
    else:
        from autoviml.custom_scores import gini_meae, gini_sklearn, rmse
        from autoviml.custom_scores import gini_msle, gini_mae, gini_mse, gini_rmse
        from autoviml.custom_scores import gini_accuracy, gini_bal_accuracy, gini_roc
        from autoviml.custom_scores import gini_precision, gini_average_precision, gini_weighted_precision
        from autoviml.custom_scores import gini_macro_precision, gini_micro_precision
        from autoviml.custom_scores import gini_samples_precision, gini_f1, gini_weighted_f1
        from autoviml.custom_scores import gini_macro_f1, gini_micro_f1, gini_samples_f1,f2_measure
        from autoviml.custom_scores import gini_log_loss, gini_recall, gini_weighted_recall
        from autoviml.custom_scores import gini_samples_recall, gini_macro_recall, gini_micro_recall
    ###### If hyper_param = 'GS', it takes a LOOOONG TIME with "SAGA" solver for LogisticRegression.
    ####   Hence to speed it up you need to change the tolerance threshold to something bigger
    if hyper_param == 'GS':
            tolerance = 0.01 #### This tolerance is bigger to speed up Logistic Regression. Otherwise, SAGA takes too long!!
    if chain_flag:
        hyper_param = 'Chain' ### In some cases, Hyper Param gets reset above to RS. Hence changing it back here.
    ### Make target into a list so that we can uniformly process the target label
    if not isinstance(target, list):
        target = [target]
        model_label = 'Single_Label'
    elif isinstance(target, list):
        if len(target)==1:
            model_label = 'Single_Label'
        elif len(target) > 1:
            model_label = 'Multi_Label'
    else:
        print('Target variable is neither a string nor a list. Please check input and try again!')
        return
    ##########   This is where some more default parameters are set up ######
    data_dimension = orig_train.shape[0]*orig_train.shape[1]  ### number of cells in the entire data set .
    if data_dimension > 1000000:
        ### if data dimension exceeds 1 million, then reduce no of params
        if len(target) == 1:
            no_iter=30
        else:
            no_iter=10
        early_stopping = 10
        test_size = 0.20
        max_iter = 10000
        Bins = 100
        top_nlp_features = 300
        if isinstance(Boosting_Flag,str):
            if Boosting_Flag.lower() == 'catboost':
                max_estims = 5000
            else:
                max_estims = 400
        else:
            max_estims = 400
    else:
        if orig_train.shape[0] <= 1000:
            if len(target) == 1:
                no_iter=20
            else:
                no_iter = 10
            test_size = 0.1
            max_iter = 4000
            top_nlp_features = 250
            if isinstance(Boosting_Flag,str):
                if Boosting_Flag.lower() == 'catboost':
                    max_estims = 3000
                else:
                    max_estims = 300
            else:
                max_estims = 300
            early_stopping = 4
        else:
            if len(target) == 1:
                no_iter=15
            else:
                no_iter=10
            test_size = 0.15
            max_iter = 7000
            top_nlp_features = 200
            if isinstance(Boosting_Flag,str):
                if Boosting_Flag.lower() == 'catboost':
                    max_estims = 4000
                else:
                    max_estims = 350
            else:
                max_estims = 350
            early_stopping = 6
    #### The warnings from Sklearn are so annoying that I have to shut it off ####
    import warnings
    warnings.filterwarnings("ignore")
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    ### First_Flag is merely a flag for the first time you want to set values of variables
    if scaling == 'MinMax':
        SS = MinMaxScaler()
    elif scaling == 'Std':
        SS = StandardScaler()
    else:
        SS = MinMaxScaler()
    ##### This is where we run the Traditional models to compare them to XGB #####
    start_time = time.time()
    ####################################################################################
    ##### Set up your Target Labels and Classes Properly Here #### Label Encoding #####
    #### This is for Classification Problems Only where you do Label Encoding of Target
    mldict = lambda: defaultdict(mldict)
    label_dict = mldict()
    first_time = True
    print('Training Set Shape = {}'.format(orig_train.shape))
    print('    Training Set Memory Usage = {:.2f} MB'.format(orig_train.memory_usage().sum() / 1024**2))
    if not isinstance(orig_test,str):
        print('Test Set Shape = {}'.format(orig_test.shape))
        print('    Test Set Memory Usage = {:.2f} MB'.format(orig_test.memory_usage().sum() / 1024**2))
    print('%s Target: %s' %(model_label,target))
    ###### Now analyze what problem we have here ####
    try:
        modeltype = analyze_problem_type(train, target,verbose)
    except:
        print('Cannot find the Target variable in data set. Please check input and try again')
        return
    #####################  TARGET VARIABLE TRANSFORMATION: Single_Label or Multi_Label ###################
    ##### Transforming each target variable column in a single label or multi-label problem is important. ####
    #################    You can use label_binarize here but I have decided to use Label Encoding instead ####
    ### This is for Classification Problems Only ########
    if modeltype != 'Regression':
        print('Random shuffling the data set before training')
        start_train = orig_train.sample(frac=1.0, random_state=seed)
    else:
        print('No shuffling of data set before training...')
        start_train = copy.deepcopy(orig_train)
    ####    Choosing Hyper Parameter Tuning Options here ##############################################
    if hyper_param == 'GS':
        print('    Using GridSearchCV for Hyper Parameter Tuning. This is slow. Switch to RS for faster tuning...')
    elif hyper_param == 'RS':
        print('    Using RandomizedSearchCV for Hyper Parameter Tuning. This is 3X faster than GridSearchCV...')
    elif hyper_param == 'HO':
        print('    Using HyperOpt which is approximately 3X Faster than GridSearchCV but results vary...')
    elif hyper_param == 'Chain':
        print('    Using MultiOutput Chain Regressor model - make sure your target columns are in correct order...')
    elif hyper_param is None:
        print('    Using CatBoost which is very fast and hence not using GridSearchCV or RandomizedSearchCV...')
    ##################   L A B E L    TARGET    T R A N S F O R M A T I O N     #################################
    for each_target in target:
        #### Make sure you don't move these 2 lines: they need to be reset for every target!
        ####   HyperOpt will not do Trials beyond max_evals - so only if you reset here, it will do it again.
        if hyper_param == 'HO':
            params_dict = {}
            bayes_trials = Trials()
        ############    THIS IS WHERE OTHER DEFAULT PARAMS ARE SET ###############
        c_params = dict()
        r_params = dict()
        if orig_train[each_target].isnull().sum() > 0:
            ### If there are Null values in target columns, drop those rows that contain nulls ##
            start_train = start_train.dropna(subset=[each_target],axis=0)
            print('Dropping rows in target column: %s that contain null values...' %each_target)
        if modeltype == 'Regression':
            if len(target) == 1:
                ### This is only for Single_Label Problems ###############
                scv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            else:
                ##### Stratified K Fold does not work in Multi-Label problems. Gives error.
                scv = n_splits
            eval_metric = 'rmse'
            objective = 'reg:squarederror'
            model_class = 'Regression'
            #### In Regression problems, max_class_length is artificially set to one.
            #### It turns out that Estimated Time is correlated to number of classes in data set. Hence we use this!
            max_class_length = 1
        else:
            if len(np.unique(train[each_target])) == 2:
                model_class = 'Binary-Class'
            elif len(np.unique(train[each_target])) > 2:
                model_class = 'Multi-Class'
                ##### If multi-class happens, then you absolutely need to do SMOTE. Otherwise, you don't get good results!
                #### Unfortunately SMOTE blows up when the data set is large -> so better to turn it off!
                print('ALERT! Setting Imbalanced_Flag to True in Auto_ViML for Multi_Classification problems improves results!')
                #Imbalanced_Flag = True
            else:
                print('Target label %s has less than 2 classes. Stopping' %each_target)
                return
            if model_label == 'Single_Label':
                rare_class_orig = find_rare_class(orig_train[each_target].values,verbose=1)
                scv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            else:
                #### Don't print these if there are too many labels - not worth cluttering the screen
                rare_class_orig = find_rare_class(orig_train[each_target].values,verbose=0)
                scv = n_splits ##### Stratified K Fold does not work in Multi-Label problems. Gives error.
            ### Perfrom Label Transformation only for Classification Problems ####
            classes = np.unique(orig_train[each_target])
            if first_time:
                first_time = False
            if len(classes) > 2:
                ##### If Boosting_Flag = True, change it to False here since Multi-Class XGB is VERY SLOW!
                max_class_length = len(classes)
                if Boosting_Flag:
                    print('CAUTION: In Multi-Class Boosting (2+ classes), TRAINING WILL TAKE A LOT OF TIME!')
                objective = 'multi:softmax'
                eval_metric = "mlogloss"
            else:
                max_class_length = 2
                eval_metric="logloss"
                objective = 'binary:logistic'
            ### Do Label Encoding when the Target Classes in each Label are Strings or Multi Class ###
            if type(start_train[each_target].values[0])==str or str(start_train[each_target].dtype
                            )=='category' or sorted(np.unique(start_train[each_target].values))[0] != 0:
                ### if the class is a string or if it has more than 2 classes, then use Factorizer!
                label_dict[each_target]['values'] = start_train[each_target].values
                #### Factorizer is the easiest way to convert target in train and predictions in test
                #### This takes care of some classes that are present in train and not in predictions
                ### and vice versa. Hence it is better than Label Encoders which breaks when above happens.
                train_targ_categs = list(start_train[each_target].value_counts().index)
                if len(train_targ_categs) == 2:
                    majority_class = [x for x in train_targ_categs if x != rare_class_orig]
                    dict_targ_all = {majority_class[0]: 0, rare_class_orig: 1}
                else:
                    dict_targ_all = return_factorized_dict(train_targ_categs)
                start_train[each_target] = start_train[each_target].map(dict_targ_all)
                label_dict[each_target]['dictionary'] = copy.deepcopy(dict_targ_all)
                label_dict[each_target]['transformer'] = dict([(v,k) for (k,v) in dict_targ_all.items()])
                label_dict[each_target]['classes'] = copy.deepcopy(train_targ_categs)
                class_nums = list(dict_targ_all.values())
                label_dict[each_target]['class_nums'] = copy.deepcopy(class_nums)
                print('String or Multi Class target: %s transformed as follows: %s' %(each_target,dict_targ_all))
                rare_class = find_rare_class(start_train[each_target].values)
            else:
                ### Since the each_target here is already numeric, you don't have to modify it
                start_train[each_target] = start_train[each_target].astype(int).values
                rare_class = find_rare_class(start_train[each_target].values)
                label_dict[each_target]['values'] = start_train[each_target].values
                label_dict[each_target]['classes'] = np.unique(start_train[each_target].values)
                class_nums = np.unique(start_train[each_target].values)
                label_dict[each_target]['class_nums'] = copy.deepcopy(class_nums)
                label_dict[each_target]['transformer'] = []
                label_dict[each_target]['dictionary'] = dict(zip(classes,classes))
                if model_label == 'Single_Label':
                    print('    Target %s is already numeric. No transformation done.' %each_target)
            if model_label == 'Single_Label':
                if rare_class != 1:
                    print('Alert! Rare Class is not 1 but %s in this data set' %rare_class)
    ###########################################################################################
    ####  This is where we start doing the iterative hyper tuning parameters #####
    params_dict = defaultdict(list)
    accu_mean = []
    error_rate = []
    ######  This is where we do the training and hyper parameter tuning ########
    orig_preds = [x for x in list(orig_train) if x not in target]
    multilabel_count = 0 #### This counts the number of times multi-labels  have jaccard metrics
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(orig_train[orig_preds], verbose)
    #####       Classify Columns   ################
    id_cols = var_df['id_vars']
    nlp_columns = var_df['nlp_vars']
    date_cols = var_df['date_vars']
    del_cols = var_df['cols_delete']
    factor_cols = var_df['factor_vars']
    numvars = var_df['continuous_vars']+var_df['int_vars']
    cat_vars = var_df['string_bool_vars']+var_df['discrete_string_vars']+var_df[
                            'cat_vars']+var_df['factor_vars']+var_df['num_bool_vars']
    num_bool_vars = var_df['num_bool_vars']
    #######################################################################################
    preds = [x for x in orig_preds if x not in id_cols+del_cols+date_cols+target]
    if len(id_cols+del_cols+date_cols)== 0:
        print('    No variables removed since no ID or low-information variables found in data set')
    else:
        print('    %d variables removed since they were ID or low-information variables'
                                %len(id_cols+del_cols+date_cols))
        if verbose >= 1:
            print('        %s' %(id_cols+del_cols+date_cols))
    ################## This is where real code begins ###################################################
    GPU_exists = check_if_GPU_exists()
    ###### This is where we set the CPU and GPU parameters for XGBoost
    param = {}
    if Boosting_Flag:
        if isinstance(Boosting_Flag,str):
            if Boosting_Flag.lower() == 'catboost':
                model_name = 'CatBoost'
                hyper_param = None
            else:
                model_name = 'XGBoost'
        else:
            model_name = 'XGBoost'
    elif Boosting_Flag is None:
        model_name = 'Linear'
    else:
        model_name = 'Forests'
    #####   Set the Scoring Parameters here based on each model and preferences of user ##############
    cpu_params = {}
    if model_name == 'XGBoost':
        ##### WE should keep CPU params as backup in case GPU fails!
        cpu_params['nthread'] = -1
        cpu_params['tree_method'] = 'hist'
        cpu_params['grow_policy'] = 'depthwise'
        cpu_params['max_depth'] = max_depth
        cpu_params['max_leaves'] = 0
        cpu_params['verbosity'] = 0
        cpu_params['gpu_id'] = 0
        cpu_params['updater'] = 'grow_colmaker'
        cpu_params['predictor'] = 'cpu_predictor'
        cpu_params['num_parallel_tree'] = 1
        if GPU_exists:
            param['nthread'] = -1
            param['tree_method'] = 'gpu_hist'
            param['grow_policy'] = 'depthwise'
            param['max_depth'] = max_depth
            param['max_leaves'] = 0
            param['verbosity'] = 0
            param['gpu_id'] = 0
            param['updater'] = 'grow_gpu_hist' #'prune'
            param['predictor'] = 'gpu_predictor'
            param['num_parallel_tree'] = 1
        else:
            param = copy.deepcopy(cpu_params)
        validation_metric = copy.deepcopy(scoring_parameter)
    elif model_name.lower() == 'catboost':
        if model_class == 'Binary-Class':
            catboost_scoring = 'Accuracy'
            validation_metric = 'Accuracy'
            loss_function='Logloss'
        elif model_class == 'Multi-Class':
            catboost_scoring = 'AUC'
            validation_metric = 'AUC:type=Mu'
            loss_function='MultiClass'
        else:
            loss_function = 'RMSE'
            validation_metric = 'RMSE'
            catboost_scoring = 'RMSE'
    else:
        validation_metric = copy.deepcopy(scoring_parameter)
    ##########  D A T A    P R E P R O C E S S I N G     H E R E ##########################
    print('#############     D A T A    P R E P A R A T I O N   AND C L E A N I N G     #############')
    missing_cols = []
    if not isinstance(orig_test,str):
        missing_cols = set(train.columns)-set(test.columns).union(set(target))
        if  len(missing_cols) >= 1:
            ### If there are columns in Train that are not in Test, it is a major problem. Resolve it now.
            if len(missing_cols) == 1:
                print('There is one %s column in Train that is not in Test. Fixing that now through Iterative Imputer...' %missing_cols)
            else:
                print('There are %d missing columns: %s in Train that are not in Test. Fixing that now through Iterative Imputer...' %(len(missing_cols),missing_cols))
            missing_cols = list(missing_cols)
    ################ Now start filling missing values for each column but when it comes to missing_cols, use Iterative Imputer
    if start_train.isnull().sum().sum() > 0:
        print('Filling missing values with "missing" placeholder and adding a column for missing_flags')
    else:
        print('No Missing Values in train data set')
    ### This is where you fill missing values with either "missing" or a unique value not in that column such as -1
    copy_preds = copy.deepcopy(preds)
    missing_flag_cols = []
    if len(copy_preds) > 0:
        dict_train = {}
        for f in copy_preds:
            if f in missing_cols:
                continue
            missing_flag = False
            if f in nlp_columns:
                #### YOu have to do missing values for NLP columns. Otherwise leave them as is for Auto_NLP later ##############
                fill_num = 'missing'
                start_train, start_test,missing_flag,new_missing_col = fill_missing_values_object_or_number(
                                                start_train, start_test,fill_num,f,True)
                if missing_flag:
                    cat_vars.append(new_missing_col)
                    num_bool_vars.append(new_missing_col)
                    preds.append(new_missing_col)
                    missing_flag_cols.append(new_missing_col)
            elif start_train[f].dtype == object:
                ####  This is the easiest way to label encode object variables in both train and test
                #### This takes care of some categories that are present in train and not in test
                ###     and vice versa
                start_train, start_test,missing_flag,new_missing_col = convert_train_test_cat_col_to_numeric(start_train, start_test,f,True)
                if missing_flag:
                    cat_vars.append(new_missing_col)
                    num_bool_vars.append(new_missing_col)
                    preds.append(new_missing_col)
                    missing_flag_cols.append(new_missing_col)
            elif start_train[f].dtype in [np.int64, np.int32, np.int16, np.int8]:
                ### if there are integer variables, don't scale them. Leave them as is.
                fill_num = int(start_train[f].min() - 1)
                start_train, start_test,missing_flag,new_missing_col = fill_missing_values_object_or_number(
                                                start_train, start_test,fill_num,f,True)
                start_train[f] = start_train[f].astype(int)
                if type(orig_test) != str:
                    start_test[f] = start_test[f].astype(int)
                if missing_flag:
                    cat_vars.append(new_missing_col)
                    num_bool_vars.append(new_missing_col)
                    preds.append(new_missing_col)
                    missing_flag_cols.append(new_missing_col)
            elif is_datetime(start_train[f]):
                ### if it is a date time variable then you need to fill any missing values with ffill method
                if start_train[f].isnull().sum() > 0:
                    missing_flag = True
                    new_missing_col = f + '_Missing_Flag'
                    start_train[new_missing_col] = 0
                    start_train.loc[start_train[f].isnull(),new_missing_col]=1
                    start_train[f] = start_train[f].fillna(method='ffill')
                if type(orig_test) != str:
                    if missing_flag:
                        start_test[new_missing_col] = 0
                    if start_test[f].isnull().sum() > 0:
                        start_test.loc[start_test[f].isnull(),new_missing_col]=1
                        start_test[f] = start_test[f].fillna(method='ffill')
            elif f in factor_cols:
                start_train, start_test,missing_flag,new_missing_col = convert_train_test_cat_col_to_numeric(start_train, start_test,f,False)
                if missing_flag:
                    cat_vars.append(new_missing_col)
                    num_bool_vars.append(new_missing_col)
                    preds.append(new_missing_col)
                    missing_flag_cols.append(new_missing_col)
            else:
                ### for all numeric variables, fill missing values with 1 less than min.
                fill_num = start_train[f].min() - 1
                if start_train[f].isnull().sum() > 0:
                    missing_flag = True
                    new_missing_col = f + '_Missing_Flag'
                    start_train[new_missing_col] = 0
                    start_train.loc[start_train[f].isnull(),new_missing_col]=1
                    start_train[f] = start_train[f].fillna(fill_num)
                if type(orig_test) != str:
                    if missing_flag:
                        start_test[new_missing_col] = 0
                    if start_test[f].isnull().sum() > 0:
                        start_test.loc[start_test[f].isnull(),new_missing_col]=1
                        start_test[f] = start_test[f].fillna(fill_num)
                if missing_flag:
                    cat_vars.append(new_missing_col)
                    num_bool_vars.append(new_missing_col)
                    preds.append(new_missing_col)
                    missing_flag_cols.append(new_missing_col)
        ###########################################################################################
        ### This is where you fill the missing column initially with nan and then with imputed values
        if len(missing_cols) >= 1:
            try:
                ### Check if they have the latest version of Sklearn 0.23.1 or later, if not, this will blow up.
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                if not isinstance(orig_test, str):
                    for eachcol in missing_cols:
                        start_test[eachcol] = np.nan
                imp = IterativeImputer(max_iter=10, random_state=0)
                imp.fit(start_train[preds])
                if not isinstance(orig_test, str):
                    print('    Total rows in Test with missing values before Iterative Imputation =', start_test[preds].isnull().sum().sum())
                    start_test[preds] = imp.transform(start_test[preds])
                    print('    Total rows in Test with missing values after Iterative Imputation =', start_test[preds].isnull().sum().sum())
            except:
                print('    Error. Sklearn Iterative Imputer is not installed on this machine. Continuing...')
        ###########################################################################################
        if orig_train.isnull().sum().sum() > 0:
            ### If there are missing values in remaining features print it here ####
            top5 = orig_train.isnull().sum().sort_values(ascending=False).index.tolist()[:5]
            print('    Columns with most missing values: %s' %(
                                            [x for x in top5 if orig_train[x].isnull().sum()>0]))
            print('    and their missing value totals: %s' %([orig_train[x].isnull().sum() for x in
                                                             top5 if orig_train[x].isnull().sum()>0]))
            if start_train[copy_preds].isnull().sum().sum() == 0:
                print('Completed missing value Imputation. No more missing values in train.')
                if verbose >= 1:
                    print('    %d new missing value columns added: %s' %(len(missing_flag_cols),missing_flag_cols))
            else:
                print('Error: Unable to complete missing value imputation in train. Exiting...')
                return
        ####################################################################################
        if type(orig_test) != str:
            if start_test[copy_preds].isnull().sum().sum() > 0:
                print('    Test data still has some missing values. Fix it. Exiting...')
                return
            else:
                print('    Test data has no missing values. Continuing...')
    else:
        print('    Could not find any variables in your data set. Please check your dataset and try again')
        return
    ###########################################################################################
    print('    Completed Label Encoding and Filling of Missing Values for Train and Test Data')
    ### This is a minor test to make sure that Boolean vars are Integers if they are Numeric!
    if len(num_bool_vars) > 0:
        ### Just make sure that numeric Boolean vars are set as Integer type -> otherwise CatBoost will blow up
        for each_bool_num in var_df['num_bool_vars']:
            start_train[each_bool_num] = start_train[each_bool_num].astype(int)
            if type(start_test) != str:
                start_test[each_bool_num] = start_test[each_bool_num].astype(int)
    ######################################################################################
    #########   Set your Refit Criterion here - if you want to maximize Precision or Recall do it here ##
    if modeltype == 'Regression':
        if scoring_parameter in ['log_loss', 'neg_mean_squared_error','mean_squared_error']:
            refit_metric = 'rmse'
        else:
            refit_metric = 'mae'
    else:
        ##### Sometimes the target is not integer in Classification problems. This is to just make sure.
        start_train[target] = start_train[target].astype(int)
        if scoring_parameter in ['precision', 'precision_score','average_precision']:
            refit_metric = 'precision'
        elif scoring_parameter in ['logloss', 'log_loss']:
            refit_metric = 'log_loss'
        elif scoring_parameter in ['recall', 'recall_score']:
            refit_metric = 'recall'
        elif scoring_parameter in ['f1', 'f1_score','f1_weighted']:
            refit_metric = 'f1'
        elif scoring_parameter in ['accuracy', 'balanced_accuracy','balanced-accuracy']:
            refit_metric = 'balanced_accuracy'
        else:
            refit_metric = 'balanced_accuracy'
    print('%s problem: hyperparameters are being optimized for %s' %(modeltype,refit_metric))
    orig_numvars = copy.deepcopy(numvars)
    #### You need to save a copy of this red_preds so you can later on create a start_train
    orig_red_preds = copy.deepcopy(preds)
    #######    ALERT! #############################################################################
    ####   We use the first target variable to select variables even in multi-label Settings
    ###    The reason is sometimes, you can have ~200 labels in multi-labels and going thru selecting vars
    ####   for each label might take time. Hence to simplify we use each_target as the first column.
    each_target = target[0]
    ### Reduced Preds are now free of correlated variables and hence can be used for Poly adds
    rem_vars = left_subtract(orig_red_preds,numvars)
    red_preds = rem_vars + numvars
    ###########################################################################################
    ########   D E F I N I N G   N E W  T R A I N and N E W   T E S T here #########################
    ####  This is where we set the orig train data set with multiple labels to the new start_train
    ####     start_train has the new features added or reduced with the multi targets in one cycle
    ###      That way, we start each train with one target, and then reset it with multi target
    #############################################################################################
    train = start_train[target+red_preds]
    if type(orig_test) != str:
        test = start_test[red_preds]
    if Add_Poly >= 1:
        print('########   A D D I N G  P O L Y N O M I A L   &   I N T E R A C T I O N    V A R S  #####')
        if Add_Poly == 1:
            print('Adding only Interaction Variables. This may result in Overfitting!')
        elif Add_Poly == 2:
            print('Adding only Squared Variables. This may result in Overfitting!')
        elif Add_Poly == 3:
            print('Adding Both Interaction and Squared Variables. This may result in Overfitting!')
        ## Since the data is already scaled, we set scaling to None here ##
        ### For train data we have to set the fit_flag to True   ####
        if len(numvars) > 1:
            #### train_red contains reduced numeric variables with original and substituted poly/intxn variables
            train_sel, lm, train_red,md,fin_xvars,feature_xvar_dict = add_poly_vars_select(train,numvars,
                                        each_target,modeltype,poly_degree,Add_Poly,md='',
                                                            corr_limit=corr_limit, scaling='None',
                                                            fit_flag=True,verbose=verbose)
            #### train_red contains reduced numeric variables with original and substituted poly/intxn variables
            if len(left_subtract(train_sel,numvars)) > 0:
                #### This means that new intxn and poly vars were added. In that case, you can use them as is
                ####  Since these vars were alread tested for correlation, there should be no high correlation!
                ###   SO you can take train_sel as the new list of numeric vars (numvars) going forward!
                addl_vars = left_subtract(train_sel,numvars)
                #numvars = list(set(numvars).intersection(set(train_sel)))
                ##### Print the additional Interxn and Poly variables here #######
                if verbose >= 1:
                    print('    Intxn and Poly Vars are: %s' %addl_vars)
                train = train_red[train_sel].join(train[rem_vars+target])
                red_preds = [x for x in list(train) if x not in target]
                if type(test) != str:
                    ######### Add Polynomial and Interaction variables to Test ################
                    ## Since the data is already scaled, we set scaling to None here ##
                    ### For Test data we have to set the fit_flag to False   ####
                    _, _, test_x_df,_,_,_ = add_poly_vars_select(test,numvars,each_target,
                                                          modeltype,poly_degree,Add_Poly,md,
                                                          corr_limit, scaling='None', fit_flag=False,
                                                           verbose=verbose)
                    ### we need to convert x_vars into text_vars in test_x_df using feature_xvar_dict
                    rev_feat_x_vars_dict = dict([(v,k) for (k,v) in feature_xvar_dict.items()])
                    if len(addl_vars) > 0:
                        addl_vars_x = [rev_feat_x_vars_dict[x] for x in addl_vars]
                        if len(addl_vars) == 1:
                            ### you need to do a special case for Series since otherwise it would blow up
                            test[addl_vars[0]] = test_x_df[addl_vars_x].values
                        else:
                            test[addl_vars] = test_x_df[addl_vars_x]
                    #### Now we should change train_sel to subst_vars since that is the new list of vars going forward
                    copy_preds = copy.deepcopy(numvars)
                    ####### If the original test column's dtype was integer, put it back as integer ###
                    for each_pred in copy_preds:
                        if orig_test[each_pred].dtype == int:
                            test[each_pred] = test[each_pred].astype(int)
                    ###### Now reset the numvars to be the new set of float variables ##########
                    numvars = train.select_dtypes(include=[np.float64,np.float32,np.float16]).columns.tolist()
            else:
                ####  NO new variables were added. so we can skip the rest of the stuff now ###
                #### This means the train_sel is the new set of numeric features selected by add_poly algorithm
                red_preds = train_sel+rem_vars
                print('    No new variable was added by polynomial features...')
        else:
            print('Adding Polynomial vars ignored since no numeric vars in data')
            train_sel = copy.deepcopy(numvars)
    else:
        ### if there are no Polynomial vars, then all numeric variables are selected
        train_sel = copy.deepcopy(numvars)
    ################  A U T O   N L P  P R O C E S S I N G   B E G I N S    H E R E !!! ####
    if len(nlp_columns) > 0:
            for nlp_column in nlp_columns:
                try:
                    train1, test1, best_nlp_transformer,max_features_limit = Auto_NLP(nlp_column,
                                                train, test, each_target, refit_metric,
                                                modeltype, top_nlp_features, verbose,
                                                build_model=False)
                    ####### Make sure you include the above new columns created in the predictor variables!
                    red_preds = [x for x in list(train1) if x not in target]
                    train = train1[red_preds+target]
                    if not isinstance(orig_test, str):
                        test = test1[red_preds]
                    ################  A U T O   N L P  P R O C E S S I N G   E N D S    H E R E !!! ####
                except:
                    print('Auto_NLP error. Continuing without NLP processing')
                    train.drop(nlp_column, axis=1, inplace=True)
                    if not isinstance(orig_test, str):
                        test.drop(nlp_column, axis=1, inplace=True)
                    red_preds = [x for x in list(train) if x not in target]
    #########   A D D   D A T E  T I M E    F E A T U R E S     H E R E ####################
    if len(date_cols) > 0:
        #### Do this only if date time columns exist in your data set!
        for date_col in date_cols:
            print('Processing %s column for date time features....' %date_col)
            date_df_train = create_time_series_features(orig_train, date_col)
            date_col_adds_train = left_subtract(date_df_train.columns.tolist(),date_col)
            print('    Adding %d columns from date-time column %s in train' %(len(date_col_adds_train),date_col))
            if not isinstance(orig_test, str):
                print('        Adding time series features for test data...')
                date_df_test = create_time_series_features(orig_test, date_col)
                date_col_adds_test = left_subtract(date_df_test.columns.tolist(),date_col)
            else:
                date_col_adds_test = []
            if len(date_col_adds_test) == 0:
                ### if both create_time_series_features return the same variables, then good
                date_col_adds = copy.deepcopy(date_col_adds_train)
                if date_col_adds != []:
                    train = train.join(date_df_train)
                    ### Now time to remove the date time column from all further processing ##
                    #train.drop(date_col,axis=1,inplace=True)
                    if not isinstance(orig_test, str):
                        if date_col_adds != []:
                            test = test.join(date_df_test)
                            ### Now time to remove the date time column from all further processing ##
                            #test.drop(date_col,axis=1,inplace=True)
            else:
                if len(left_subtract(date_col_adds_train,date_col_adds_test)) != 0 or len(left_subtract(date_col_adds_test,date_col_adds_train)) :
                    ### if the number of cols created in train data is not same as test data, then
                    ###  it is a problem. It can happen due to bad data. Just select the common ones
                    ###  in that case. THat way, it would not blow up later.
                    print('Error: Due to differences in Train and Test %s, getting different columns created. Continuing...' %date_col)
                    date_col_adds = list(set(date_col_adds_train).intersection(set(date_col_adds_test)))
                else:
                    ### if both create_time_series_features return the same variables, then good
                    date_col_adds = copy.deepcopy(date_col_adds_train)
                    if date_col_adds != []:
                        train = train.join(date_df_train)
                        ### Now time to remove the date time column from all further processing ##
                        #train.drop(date_col,axis=1,inplace=True)
                        if not isinstance(orig_test, str):
                            if date_col_adds != []:
                                test = test.join(date_df_test)
                                ### Now time to remove the date time column from all further processing ##
                                #test.drop(date_col,axis=1,inplace=True)
        #########     CREATING TIME FEATURES IS COMPLETED   #############################
    ######  We have to detect float variables again since we have created new variables using Auto_NLP!!
    red_preds = [x for x in list(train) if x not in target]
    ### we add new date time variables to the list of numeric float variables to see if they have any
    ### high correlations to each other. In that case, they can be reduced by the feature reduction scheme.
    #train_sel = train_sel + date_col_adds
    #rem_cat_vars = left_subtract(red_preds, train_sel)
    num_vars = train.select_dtypes(include=[np.float64,np.float32,np.float16]).columns.tolist()
    #########     SELECT IMPORTANT FEATURES HERE   #############################
    if feature_reduction:
        print('############# R E M O V I N G   H I G H L Y  C O R R E L A T E D    V A R S #################')
        ### Make sure you remove variables that are highly correlated within data set first
        try:
            train_sel = remove_variables_using_fast_correlation(train,red_preds, modeltype,
                                    each_target, corr_limit, verbose)
        except:
            #### if for some reason, the above blows up due to memory error, then try this
            #### Dropping highly correlated Features fast using simple linear correlation ###
            remove_list = remove_highly_correlated_vars_fast(train[num_vars],corr_limit)
            train_sel = left_subtract(num_vars, remove_list)
        num_vars = train[train_sel].select_dtypes(include=[np.float64,np.float32,np.float16]).columns.tolist()
        print('Splitting features into float and categorical (integer) variables:')
        print('    (%d) float variables ...' %len(num_vars))
        rem_vars = left_subtract(list(train), num_vars+target)
        important_features,num_vars, imp_cats = find_top_features_xgb(train,train_sel,num_vars,
                                                     each_target,
                                                 modeltype,corr_limit,verbose)
    else:
        important_features = copy.deepcopy(red_preds)
        num_vars = copy.deepcopy(numvars)
        ####  we need to set the rem_vars in case there is no feature reduction #######
        imp_cats = left_subtract(important_features,num_vars)
    #####################################################################################
    if len(important_features) == 0:
        print('No important features found. Using all input features...')
        important_features = copy.deepcopy(red_preds)
        num_vars = copy.deepcopy(numvars)
        ####  we need to set the rem_vars in case there is no feature reduction #######
        imp_cats = left_subtract(important_features,num_vars)
    ### Training an XGBoost model to find important features
    train = train[important_features+target]
    ######################################################################
    if type(orig_test) != str:
        test = test[important_features]
        #######You must convert category variables into integers ###############
        integer_cats = test[imp_cats].select_dtypes(include=[np.int64,np.int32,np.int16,np.int8,int]).columns.tolist()
        non_integer_cats = left_subtract(imp_cats, integer_cats)
        for important_cat in non_integer_cats:
            test[important_cat] = test[important_cat].astype(int)
    ##############          F E A T U R E   E N G I N E E R I N G  S T A R T S  N O W    ##############
    ######    From here on we do some Feature Engg using Target Variable with Data Leakage ############
    ###   To avoid Model Leakage, we will now split the Data into Train and CV so that Held Out Data
    ##     is Pure and is unadulterated by learning from its own Target. This is known as Data Leakage.
    ###################################################################################################
    print('    Performing limited feature engineering for binning, add_poly and KMeans_Featurizer flags  ...')
    ################     I  M  P  O  R  T  A  N  T  ##################################################
    ### The reason we don't use train_test_split is because we want only a partial train entropy binned
    ### If we use the whole of Train for entropy binning then there will be data leakage and our
    ### cross validation test scores will not be so accurate. So don't change the next 5 lines here!
    ################     I  M  P  O  R  T  A  N  T  ##################################################
    part_train, part_cv = split_train_into_two(train, target, randomstate=99, modeltype=modeltype)
    X_train, X_cv = part_train[important_features], part_cv[important_features]
    if model_label == 'Single_Label':
        y_train, y_cv = part_train[each_target], part_cv[each_target]
    else:
        y_train, y_cv = part_train[target], part_cv[target]
    print('    Train CV Split completed with', "TRAIN rows = ", part_train.shape[0], ", CV rows = ", part_cv.shape[0])
    ################   IMPORTANT ENTROPY  BINNING FIRST TIME   #####################################
    ############   Add Entropy Binning of Continuous Variables Here ##############################
    num_vars = train[important_features].select_dtypes(include=[np.float64,np.float32,np.float16]).columns.tolist()
    saved_important_features = copy.deepcopy(important_features)  ### these are original features without '_bin' added
    #### saved_num_vars is an important variable: it contains the orig_num_vars before they were binned
    saved_num_vars = copy.deepcopy(num_vars)  ### these are original numeric features without '_bin' added
    if model_label == 'Single_Label':
        ###############    BINNING   FIRST    TIME ##################################################
        if Binning_Flag and len(saved_num_vars) > 0:
            #### Do binning only when there are numeric features ####
            #### When we Bin the first time, we set the entropy_binning flag to False so
            ####    no numeric variables are removed. But next time, we will remove them later!
            part_train, num_vars, important_features, part_cv = add_entropy_binning(part_train,
                                            each_target, saved_num_vars,
                                            saved_important_features, part_cv,
                                            modeltype, entropy_binning=False,verbose=verbose)
            #### In saved_num_vars we send in all the continuous_vars but we bin only the top few vars.
            ###  Those that are binned are removed from saved_num_vars and the remaining become num_vars
            ### Our job is to find the names of those original numeric variables which were binned.
            ### orig_num_vars contains original num vars. num_vars contains binned versions of those vars.
            ### Those binned variables have now become categorical vars and must be added to imp_cats.
            ### you get the name of the original vars which were binned here in this orig_num_vars variable!
            orig_num_vars = left_subtract(saved_num_vars,num_vars)
            #### you need to know the name of the binner variables. This is where you get it!
            binned_num_vars = left_subtract(num_vars,saved_num_vars)
            imp_cats += binned_num_vars
            #### Also note that important_features does not contain orig_num_vars which have been erased.
        else:
            print('    Binning_Flag set to False or there are no float vars in data set to be binned')
        #######################   KMEANS   FIRST   TIME     ############################
        ### Now we add another Feature tied to KMeans clustering using Predictor and Target variables ###
        if KMeans_Featurizer and len(saved_num_vars) > 0:
            ### DO KMeans Featurizer only if there are numeric features in the data set!
            km_label = each_target+'_KMeans_Clusters'
            print('    Adding one Feature named %s based on KMeans_Featurizer_Flag=True...' %km_label)
            #### Make the number of clusters as the same as log10 of number of rows in Train
            num_clusters = int(np.round(max(2,np.log10(train.shape[0]))))
            #### Make the number of clusters as the same as log10 of number of rows in Train
            train_clusters, cv_clusters = Transform_KM_Features(part_train[
                                important_features], part_train[each_target],
                                part_cv[important_features], num_clusters)
            #### Since this is returning the each_target in X_train, we need to drop it here ###
            print('    Used KMeans to naturally cluster Train predictor variables into %d clusters' %num_clusters)
            part_train[km_label] = train_clusters
            part_cv[km_label] = cv_clusters
            #X_train.drop(each_target,axis=1,inplace=True)
            imp_cats.append(km_label)
            for imp_cat in imp_cats:
                part_train[imp_cat] = part_train[imp_cat].astype(int)
                part_cv[imp_cat] = part_cv[imp_cat].astype(int)
            ####### The features are checked again once we add the cluster feature ####
            important_features.append(km_label)
        else:
            print('    KMeans_Featurizer set to False or there are no float variables in data')
            km_label = ''
        #######################   STACKING   FIRST   TIME     ############################
        ######### This is where you do Stacking of Multi Model Results into One Column ###
        if Stacking_Flag:
            try:
                #### In order to join, you need X_train to be a Pandas Series here ##
                print('Alert! Stacking can produce Highly Overfit models on Training Data...')
                ### In order to avoid overfitting, we are going to learn from a small sample of data
                ### That is why we are using X_train to train on and using it to predict on X_cv!
                addcol, stacks1 = QuickML_Stacking(part_train[important_features],part_train[
                                    each_target],part_train[important_features],
                              modeltype, Boosting_Flag, scoring_parameter,verbose)
                addcol, stacks2 = QuickML_Stacking(part_train[important_features],part_train[
                                    each_target],part_cv[important_features],
                              modeltype, Boosting_Flag, scoring_parameter,verbose)
                part_train = part_train.join(pd.DataFrame(stacks1,index=part_train.index,
                                                  columns=addcol))
                ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                part_cv = part_cv.join(pd.DataFrame(stacks2,index=part_cv.index,
                                                  columns=addcol))
                print('    Adding %d Stacking feature(s) to training data' %len(addcol))
                ######  We make sure that we remove any new features that are highly correlated ! #####
                #addcol = remove_variables_using_fast_correlation(X_train,addcol,modeltype,corr_limit,verbose)
                important_features += addcol
            except:
                print('    Error in Stacking first time. Continuing without Stacking for this data set...')
                Stacking_Flag = False
    ###############################################################################
    #### part train contains the unscaled original train. It also contains binned and orig_num_vars!
    #### DO NOT DO TOUCH part_train and part_cv -> we need it to recrate train later!
    ####################### Now do Feature Scaling Here #################################
    #part_train_scaled, part_cv_scaled = perform_scaling_numeric_vars(part_train, important_features,
    #                                        part_cv, model_name, SS)
    #### if important_features by mistake has duplicates they must be removed!
    important_features = find_remove_duplicates(important_features)
    if model_label == 'Single_Label':
        if not perform_scaling_flag:
            print('Skipping %s scaling since perform_scaling flag is set to False ' %scaling)
            X_train = part_train[important_features]
            X_cv = part_cv[important_features]
        else:
            print('Performing %s scaling of train and validation data' %scaling)
            SS.fit(part_train[important_features])
            X_train = pd.DataFrame(SS.transform(part_train[important_features]),index=part_train.index,
                                        columns=important_features)
            X_cv = pd.DataFrame(SS.transform(part_cv[important_features]),index=part_cv.index,
                                        columns=important_features)
            perform_scaling_flag = True
    else:
        if not perform_scaling_flag:
            print('Skipping %s scaling since perform_scaling flag is set to False ' %scaling)
            X_train = part_train[important_features]
            X_cv = part_cv[important_features]
        else:
            ###### Performing Scaling helps SVM models ######################
            print('Performing %s scaling of train and validation data' %scaling)
            SS.fit(part_train[important_features])
            X_train = pd.DataFrame(SS.transform(part_train[important_features]),index=part_train.index,
                                        columns=important_features)
            X_cv = pd.DataFrame(SS.transform(part_cv[important_features]),index=part_cv.index,
                                        columns=important_features)
            perform_scaling_flag =  True
    #### Remember that the next 2 lines are crucial: if X and y are dataframes, then predict_proba
    ###     will return  dataframes or series. Otherwise it will return Numpy array's.
    ##      Be consistent when using dataframes with XGB. That's the best way to keep feature names!
    print('############### %s M O D E L   B U I L D I N G  B E G I N S  ####################' %model_name)
    print('Rows in Train data set = %d' %X_train.shape[0])
    print('  Features in Train data set = %d' %X_train.shape[1])
    print('    Rows in held-out data set = %d' %X_cv.shape[0])
    data_dim = X_train.shape[0]*X_train.shape[1]*len(target)
    ###### You need to change the string used in hyper-params based on model selected ####
    estimator_string = 'estimator__'
    if hyper_param is not None:
        if hyper_param.lower() == 'chain':
            estimator_string = 'base_estimator__'
    ###   Setting up the Estimators for Single Label and Multi Label targets only
    if modeltype == 'Regression':
        metrics_list = ['neg_mean_absolute_error' ,'neg_mean_squared_error',
         'neg_mean_squared_log_error','neg_median_absolute_error']
        eval_metric = "rmse"
        if scoring_parameter == 'neg_mean_absolute_error' or scoring_parameter =='mae':
            meae_scorer = make_scorer(gini_meae, greater_is_better=False)
            scorer = meae_scorer
        elif scoring_parameter == 'neg_mean_squared_error' or scoring_parameter =='mse':
            mse_scorer = make_scorer(gini_mse, greater_is_better=False)
            scorer = mse_scorer
        elif scoring_parameter == 'neg_mean_squared_log_error' or scoring_parameter == 'log_error':
            msle_scorer = make_scorer(gini_msle, greater_is_better=False)
            print('    Log Error is not recommended since predicted values might be negative and error')
            rmse_scorer = make_scorer(gini_rmse, greater_is_better=False)
            scorer = rmse_scorer
        elif scoring_parameter == 'neg_median_absolute_error' or scoring_parameter == 'median_error':
            mae_scorer = make_scorer(gini_mae, greater_is_better=False)
            scorer = mae_scorer
        elif scoring_parameter =='rmse' or scoring_parameter == 'root_mean_squared_error':
            rmse_scorer = make_scorer(gini_rmse, greater_is_better=False)
            scorer = rmse_scorer
        else:
            scoring_parameter = 'rmse'
            rmse_scorer = make_scorer(gini_rmse, greater_is_better=False)
            scorer = rmse_scorer
        ####      HYPER PARAMETERS FOR REGRESSION TUNING ARE SETUP HERE      ###
        if hyper_param == 'GS':
            r_params = {
                    "Forests": {
                            "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                            "max_depth": [3, 5, max_depth],
                            #"criterion" : ['mse','mae'],
                            },
                    "Linear": {
                        #'alpha': np.logspace(-5,3),
                        'epsilon': [0, 0.01, 0.06, 0.1, 0.3, 0.5, 0.8],
                        'C': [0.01, 0.1,1,1.5, 10,100]
                            },
                    "XGBoost": {
                                    'learning_rate': [0.1,0.3,0.5],
                                    'gamma': np.linspace(0, 32,7).astype(int),
                                    "max_depth": [3, 5, max_depth],
                            },
                    "CatBoost": {
                            'learning_rate': np.logspace(Alpha_min,Alpha_max,40),
                            },
                    "Multi_Label": {
                            "Linear": {
                            #estimator_string+'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
                            estimator_string+'n_neighbors': [1,3,5,7,11]
                                    },
                            "Forests": {
                            #### No estimator string for Random Forests since it is already Multi-Label
                            'n_estimators':  np.linspace(100, max_estims, n_steps, dtype = "int"),
                            },
                            "XGBoost": {
                            estimator_string+'n_estimators':  np.linspace(100, max_estims, n_steps, dtype = "int"),
                            estimator_string+'learning_rate': np.linspace(0.1,0.5,5),
                            estimator_string+'gamma': np.linspace(0, 32,7).astype(int),
                            },
                            "CatBoost": {
                            }
                        }
                    }
        else:
            import scipy as sp
            r_params = {
                    "Forests": {
                           'n_estimators': sp.stats.randint(100,max_estims),
                            "max_depth": sp.stats.randint(3, max_depth),
                            "min_samples_leaf": sp.stats.randint(2, 20),
                            #"criterion" : ['mse','mae'],
                            },
                    "Linear": {
                            #'alpha': sp.stats.uniform(scale=1000),
                            'epsilon': sp.stats.uniform(scale=1),
                            'C': sp.stats.uniform(scale=100)
                            },
                    "XGBoost":  {
                            'learning_rate': sp.stats.uniform(scale=1),
                            'gamma': sp.stats.randint(0, 32),
                           'n_estimators': sp.stats.randint(100,max_estims),
                            "max_depth": sp.stats.randint(3, max_depth),
                                },
                    "CatBoost": {
                            'learning_rate': np.logspace(Alpha_min,Alpha_max,40),
                            },
                    "Multi_Label": {
                        "Linear": {
                            #estimator_string+'C': sp.stats.uniform(scale=1)
                            estimator_string+'n_neighbors': sp.stats.randint(1,max_neighbors)
                                },
                        "Forests": {
                            #estimator_string+'__n_estimators': sp.stats.randint(100,max_estims)
                            estimator_string+'n_estimators': sp.stats.randint(100,max_estims)
                                },
                        "XGBoost": {
                            estimator_string+'learning_rate': sp.stats.uniform(scale=1),
                           estimator_string+'n_estimators': sp.stats.randint(100,max_estims),
                           estimator_string+'gamma': sp.stats.randint(0, 32),
                                },
                        "CatBoost": {
                                }
                            }
                        }
        if Boosting_Flag:
            if model_name.lower() == 'catboost':
                xgbm = CatBoostRegressor(verbose=1,iterations=max_estims,random_state=99,
                        one_hot_max_size=one_hot_size,
                        loss_function=loss_function, eval_metric=catboost_scoring,
                        subsample=0.7,bootstrap_type='Bernoulli',
                        metric_period = 500,
                       early_stopping_rounds=250,boosting_type='Plain')
            else:
                objective = 'reg:squarederror'
                if model_label == 'Multi_Label':
                    ##### For multi-label, it is better to keep the XGBoost simple ##########
                    xgbm = XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=200,
                        n_jobs=-1, nthread=-1, objective=objective, random_state=1,
                        seed=1)
                else:
                    ##### For Single-label, it is better to keep the XGBoost complex ##########
                    xgbm = XGBRegressor( n_estimators=100,subsample=subsample,objective=objective,
                                            booster='gbtree', learning_rate=0.1,nthread=-1,
                                            colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                             gamma=1, seed=1,n_jobs=-1,random_state=1)
                    xgbm.set_params(**param)
        elif Boosting_Flag is None:
            #xgbm = Lasso(max_iter=max_iter,random_state=seed)
            #xgbm = Lasso(max_iter=max_iter,random_state=seed)
            if model_label == 'Single_Label':
                xgbm = LinearSVR(epsilon=0.0, tol=0.001, C=1.0,max_iter=max_iter, random_state=seed)
            else:
                xgbm = KNeighborsRegressor(n_neighbors=5, weights = 'distance') ### don't use n_jobs=-1 gives error!!
                #### n_jobs = None in case of KNN since it gives error if you use -1 anywhere in the process
                n_jobs = None
        else:
            xgbm = RandomForestRegressor(
                            **{
                            'bootstrap': bootstrap, 'n_jobs': -1, 'warm_start': warm_start,
                            'random_state':seed,'min_samples_leaf':2,
                            'max_features': "sqrt"
                            })
    else:
        #### This is for Binary Classification ##############################
        classes = label_dict[each_target]['classes']
        metrics_list = ['accuracy_score','roc_auc_score','logloss', 'precision','recall','f1']
        # Create regularization hyperparameter distribution with 50 C values ####
        if hyper_param == 'GS':
            c_params['XGBoost'] = {
                                    'learning_rate': np.linspace(0.1,0.5,5),
                                    'gamma': np.linspace(0, 32,7).astype(int),
                                    "max_depth": [3, 5, max_depth],
                                }
            c_params["CatBoost"] = {
                            'learning_rate': np.logspace(Alpha_min,Alpha_max,40),
                            }
            if Imbalanced_Flag:
                c_params['Linear'] = {
                                'C': Cs,
                            'solver' : solvers,
                            'penalty' : penalties,
                            'class_weight':[None, 'balanced'],
                                }
            else:
                c_params['Linear'] = {
                                'C': Cs,
                            'class_weight':[None, 'balanced'],
                            'penalty' : penalties,
                                'solver' : solvers,
                                    }
            c_params["Forests"] = {
                ##### I have selected these to avoid Overfitting which is a problem for small data sets
                            "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                "max_depth": [3, 5, max_depth],
                                #'max_features': [1,2,5, max_features],
                                #"criterion":['gini','entropy'],
                                        }
            c_params["Multi_Label"] = {
                        estimator_string+'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
                        }
        else:
            import scipy as sp
            c_params['XGBoost'] = {
                    'learning_rate': sp.stats.uniform(scale=1),
                    'gamma': sp.stats.randint(0, 32),
                   'n_estimators': sp.stats.randint(100,max_estims),
                    "max_depth": sp.stats.randint(3, max_depth),
                  }
            c_params["CatBoost"] = {
                                'learning_rate': sp.stats.uniform(scale=1),
                            }
            if Imbalanced_Flag:
                c_params['Linear'] = {
                                'C': sp.stats.uniform.rvs(loc=0.01,scale=10,size=1000),
                            'solver' : solvers,
                            'penalty' : penalties,
                            #'class_weight':[None, 'balanced'],
                                }
            else:
                c_params['Linear'] = {
                                'C': sp.stats.uniform.rvs(loc=0.01,scale=10,size=1000),
                                'penalty' : penalties,
                                'solver' : solvers,
                                }
            c_params["Forests"] = {
                ##### I have selected these to avoid Overfitting which is a problem for small data sets
                               'n_estimators': sp.stats.randint(100,max_estims),
                                "max_depth": sp.stats.randint(3, max_depth),
                                "min_samples_leaf": sp.stats.randint(2, 20),
                                #"criterion":['gini','entropy'],
                                #'max_features': ['log', "sqrt"] ,
                                #'class_weight':[None,'balanced']
                                        }
            c_params["Multi_Label"] = {
                        ### this is the best params for logistic regression
                        estimator_string+'C': sp.stats.uniform.rvs(loc=0.01,scale=10,size=1000)
                        }
        # Create regularization hyperparameter distribution using uniform distribution
        if len(classes) == 2:
            objective = 'binary:logistic'
            if scoring_parameter == 'accuracy' or scoring_parameter == 'accuracy_score':
                accuracy_scorer = make_scorer(gini_accuracy, greater_is_better=True, needs_proba=False)
                scorer =accuracy_scorer
            elif scoring_parameter == 'gini':
                gini_scorer = make_scorer(gini_sklearn, greater_is_better=False, needs_proba=True)
                scorer =gini_scorer
            elif scoring_parameter == 'auc' or scoring_parameter == 'roc_auc' or scoring_parameter == 'roc_auc_score':
                roc_scorer = make_scorer(gini_roc, greater_is_better=True, needs_threshold=True)
                scorer =roc_scorer
            elif scoring_parameter in ['log_loss','logloss','neg_log_loss','log-loss']:
                scoring_parameter = 'neg_log_loss'
                logloss_scorer = make_scorer(gini_log_loss, greater_is_better=False, needs_proba=False)
                scorer =logloss_scorer
            elif scoring_parameter=='balanced_accuracy' or scoring_parameter=='balanced-accuracy' or scoring_parameter=='average_accuracy':
                bal_accuracy_scorer = make_scorer(gini_bal_accuracy, greater_is_better=True,
                                                                  needs_proba=False)
                scorer = bal_accuracy_scorer
            elif scoring_parameter == 'precision' or scoring_parameter == 'precision_score':
                precision_scorer = make_scorer(gini_precision, greater_is_better=True, needs_proba=False,
                    pos_label=rare_class)
                scorer =precision_scorer
            elif scoring_parameter == 'recall' or scoring_parameter == 'recall_score':
                recall_scorer = make_scorer(gini_recall, greater_is_better=True, needs_proba=False,
                        pos_label=rare_class)
                scorer =recall_scorer
            elif scoring_parameter == 'f1' or scoring_parameter == 'f1_score':
                f1_scorer = make_scorer(gini_f1, greater_is_better=True, needs_proba=False,
                        pos_label=rare_class)
                scorer =f1_scorer
            elif scoring_parameter == 'f2' or scoring_parameter == 'f2_score':
                f2_scorer = make_scorer(f2_measure, greater_is_better=True, needs_proba=False)
                scorer =f2_scorer
            else:
                logloss_scorer = make_scorer(gini_log_loss, greater_is_better=False, needs_proba=False)
                scorer =logloss_scorer
                #f1_scorer = make_scorer(gini_f1, greater_is_better=True, needs_proba=False,
                #        pos_label=rare_class)
                #scorer = f1_scorer
            ### DO NOT USE NUM CLASS WITH BINARY CLASSIFICATION ######
            if Boosting_Flag:
                if model_name.lower() == 'catboost':
                    xgbm =  CatBoostClassifier(verbose=1,iterations=max_estims,
                        random_state=99,one_hot_max_size=one_hot_size,
                        loss_function=loss_function, eval_metric=catboost_scoring,
                        subsample=subsample,bootstrap_type='Bernoulli',
                        metric_period = 500,
                       early_stopping_rounds=250,boosting_type='Plain')
                else:
                    xgbm = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                        max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=200,
                        n_jobs=n_jobs, nthread=n_jobs, objective=objective,
                        random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                        seed=1)
                    xgbm.set_params(**param)
            elif Boosting_Flag is None:
                #### I have set the Verbose to be False here since it produces too much output ###
                #xgbm = LogisticRegression(random_state=seed,verbose=False,solver=solver,
                #                            fit_intercept=True, tol=tolerance,
                #                         warm_start=warm_start, max_iter=max_iter)
                xgbm = LogisticRegression(random_state=seed,verbose=False,n_jobs=-1,tol=0.01,
                                             warm_start=False, max_iter=max_iter)
            else:
                xgbm = RandomForestClassifier(
                            **{
                            'bootstrap': bootstrap, 'n_jobs': -1, 'warm_start': warm_start,
                            'random_state':seed,'min_samples_leaf':2,'oob_score':True,
                            'max_features': "sqrt", 'n_estimators': 200
                            })
        else:
            #####   This is for MULTI Classification ##########################
            objective = 'multi:softmax'
            eval_metric = "mlogloss"
            if scoring_parameter == 'gini':
                gini_scorer = make_scorer(gini_sklearn, greater_is_better=False, needs_proba=True)
                scorer = gini_scorer
            elif scoring_parameter=='balanced_accuracy' or scoring_parameter=='balanced-accuracy' or scoring_parameter=='average_accuracy':
                bal_accuracy_scorer = make_scorer(gini_bal_accuracy, greater_is_better=True,
                                                                  needs_proba=False)
                scorer = bal_accuracy_scorer
            elif scoring_parameter == 'roc_auc' or scoring_parameter == 'roc_auc_score':
                roc_auc_scorer = make_scorer(gini_sklearn, greater_is_better=False, needs_proba=True)
                scorer = roc_auc_scorer
            elif scoring_parameter == 'average_precision' or scoring_parameter == 'mean_precision':
                average_precision_scorer = make_scorer(gini_average_precision,
                                                       greater_is_better=True, needs_proba=True)
                scorer = average_precision_scorer
            elif scoring_parameter == 'samples_precision':
                samples_precision_scorer = make_scorer(gini_samples_precision,
                                                       greater_is_better=True, needs_proba=True)
                scorer = samples_precision_scorer
            elif scoring_parameter == 'weighted_precision' or scoring_parameter == 'weighted-precision':
                weighted_precision_scorer = make_scorer(gini_weighted_precision,
                                                        greater_is_better=True, needs_proba=True)
                scorer = weighted_precision_scorer
            elif scoring_parameter == 'macro_precision':
                macro_precision_scorer = make_scorer(gini_macro_precision,
                                                     greater_is_better=True, needs_proba=True)
                scorer = macro_precision_scorer
            elif scoring_parameter == 'micro_precision':
                scorer = micro_precision_scorer
                micro_precision_scorer = make_scorer(gini_micro_precision,
                                                     greater_is_better=True, needs_proba=True)

            elif scoring_parameter == 'samples_recall':
                samples_recall_scorer = make_scorer(gini_samples_recall, greater_is_better=True, needs_proba=True)
                scorer = samples_recall_scorer
            elif scoring_parameter == 'weighted_recall' or scoring_parameter == 'weighted-recall':
                weighted_recall_scorer = make_scorer(gini_weighted_recall,
                                                     greater_is_better=True, needs_proba=True)
                scorer = weighted_recall_scorer
            elif scoring_parameter == 'macro_recall':
                macro_recall_scorer = make_scorer(gini_macro_recall,
                                                  greater_is_better=True, needs_proba=True)
                scorer = macro_recall_scorer
            elif scoring_parameter == 'micro_recall':
                micro_recall_scorer = make_scorer(gini_micro_recall, greater_is_better=True, needs_proba=True)
                scorer = micro_recall_scorer

            elif scoring_parameter == 'samples_f1':
                samples_f1_scorer = make_scorer(gini_samples_f1,
                                                greater_is_better=True, needs_proba=True)
                scorer = samples_f1_scorer
            elif scoring_parameter == 'weighted_f1' or scoring_parameter == 'weighted-f1':
                weighted_f1_scorer = make_scorer(gini_weighted_f1,
                                                 greater_is_better=True, needs_proba=True)
                scorer = weighted_f1_scorer
            elif scoring_parameter == 'macro_f1':
                macro_f1_scorer = make_scorer(gini_macro_f1,
                                              greater_is_better=True, needs_proba=True)
                scorer = macro_f1_scorer
            elif scoring_parameter == 'micro_f1':
                micro_f1_scorer = make_scorer(gini_micro_f1,
                                              greater_is_better=True, needs_proba=True)
                scorer = micro_f1_scorer
            else:
                weighted_f1_scorer = make_scorer(gini_weighted_f1,
                                                 greater_is_better=True, needs_proba=True)
                scorer = weighted_f1_scorer
            import scipy as sp
            if hyper_param == 'GS':
                c_params['XGBoost'] = {
                                'learning_rate': np.linspace(0.1,0.5,5),
                                'gamma': np.linspace(0, 32,7).astype(int),
                                "n_estimators": np.linspace(100, max_estims, 3).astype(int),
                            }
                c_params["CatBoost"] = {
                                'learning_rate': np.logspace(Alpha_min,Alpha_max,40),
                                }
                if Imbalanced_Flag:
                    c_params['Linear'] = {
                                'C': Cs,
                            'class_weight':[None, 'balanced'],
                            }
                else:
                    c_params['Linear'] = {
                                'C': Cs,
                                    }
                c_params["Forests"] = {
                ##### I have selected these to avoid Overfitting which is a problem for small data sets
                        "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                            "max_depth": [3, 5, max_depth],
                            #"criterion":['gini','entropy'],
                                    }
                c_params["Multi_Label"] = {
                            'estimator__C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
                            }
            else:
                import scipy as sp
                c_params['XGBoost'] = {
                        'learning_rate': sp.stats.uniform(scale=1),
                        'gamma': sp.stats.randint(0, 32),
                       'n_estimators': sp.stats.randint(100, max_estims),
                      }
                c_params['CatBoost'] = {
                        'learning_rate': sp.stats.uniform(scale=1),
                        }
                if Imbalanced_Flag:
                    c_params['Linear'] = {
                            'C': sp.stats.uniform(scale=100),
                            'class_weight':[None, 'balanced'],
                            }
                else:
                    c_params['Linear'] = {
                            'C': sp.stats.uniform(scale=100),
                                    }
                c_params["Forests"] = {
                #####   I have set these to avoid OverFitting which is a problem for small data sets ###
                           'n_estimators': sp.stats.randint(100,max_estims),
                            "max_depth": sp.stats.randint(3, max_depth),
                            "min_samples_leaf": sp.stats.randint(2, 20),
                            #"criterion":['gini','entropy'],
                            #'class_weight':[None,'balanced']
                                        }
                c_params["Multi_Label"] = {
                            ### this is the best params for logistic regression
                            estimator_string+'C': sp.stats.uniform.rvs(loc=0.01,scale=10,size=1000)
                        }
            ###################################################################
            if Boosting_Flag:
                # Create regularization hyperparameter distribution using uniform distribution
                if model_name.lower() == 'catboost':
                    xgbm =  CatBoostClassifier(verbose=1,iterations=max_estims,
                            random_state=99,one_hot_max_size=one_hot_size,
                            loss_function=loss_function, eval_metric=catboost_scoring,
                            subsample=subsample,bootstrap_type='Bernoulli',
                            metric_period = 500,
                           early_stopping_rounds=25,boosting_type='Plain')
                else:
                    #xgbm = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                    #            colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                    #    max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=200,
                    #    n_jobs=-1, nthread=-1, objective=objective,
                    #    random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                    #    num_class= len(classes),
                    #    seed=1)
                    #xgbm.set_params(**param)
                    if model_label == 'Single_Label':
                        #### Found XGB too slow for multi-class - hence better to use One Vs Rest against it
                        xgbm = OneVsRestClassifier(estimator=XGBClassifier(booster='gbtree',
                                                n_estimators=200,random_state=99,nthread=n_jobs,
                                                n_jobs=n_jobs)
                                                    )
                    else:
                        ### This will be automatically wrapped around by a MultiOutputClassifier later ###
                        XGBClassifier(booster='gbtree',
                                                n_estimators=200,random_state=99,nthread=n_jobs,
                                                n_jobs=n_jobs)
                    if hyper_param == 'GS':
                        c_params = {
                                    "XGBoost": {
                                        }
                                    }
                    else:
                        c_params = {
                                    "XGBoost": {
                                    }
                                }
            elif Boosting_Flag is None:
                #### I have set the Verbose to be False here since it produces too much output ###
                xgbm = LogisticRegression(random_state=seed,verbose=False,n_jobs=-1,solver=solver,
                                            fit_intercept=True, tol=tolerance, multi_class='ovr',
                                          max_iter=10000, warm_start=False,
                                          )
            else:
                xgbm = RandomForestClassifier(bootstrap=bootstrap, oob_score=True,warm_start=warm_start,
                                        n_estimators=200,max_depth=None,
                                        min_samples_leaf=2,max_features='auto',
                                      random_state=seed,n_jobs=-1)
    #### NEXT LINE IS NEEDED TO GET MODELS ABOVE INTO A WRAPPER
    ###  WRAPPER IS THIS MULTI_LABEL FORMAT. SO LEAVE IT ALONE - AS-IS !!!
    ####### Now we add another wrapper for Multi-Output Problems ##############
    #### DON'T MODIFY THE NEXT LINE!!! IT MAY APPEAR STUPID BUT IT IS NEEDED TO GET MODELS ABOVE
    ###  INTO THIS MULTI_LABEL FORMAT. SO LEAVE IT ALONE - AS-IS !!!
    if model_label != 'Single_Label':
        #### For Multi-Label Problems = you have to change the model to a multi-output model
        if chain_flag:
            if modeltype == 'Regression':
                ### Regressor Chain a stacking algorithm - it predicts one target and
                ###  uses it to predict other targets in a daisy chain. Neat!
                xgbm = RegressorChain(xgbm, order= np.arange(len(target)).tolist(),
                            random_state=99)
            else:
                xgbm = ClassifierChain(xgbm, order= np.arange(len(target)).tolist(),
                            random_state=99)
        else:
            if modeltype == 'Regression':
                ### This seems to work best for XGBRegressor and CatBoostRegressor only ####
                xgbm = MultiOutputRegressor(xgbm, n_jobs=n_jobs)
            else:
                if Boosting_Flag is None:
                    ##### This is for Multi-Label: Both Binary and Multi-class Classification ######################
                    xgbm = OneVsRestClassifier(estimator=xgbm)
                elif not Boosting_Flag:
                    ### RandomForest is a special model: we can use the model as is: RFR is Multi Output! ###
                        print('    Using %s model for multi-output processing... ' %model_name)
                else:
                    ##### This is for Multi-Label: Both Binary and Multi-class Classification ######################
                    xgbm = OneVsRestClassifier(estimator=xgbm)
        print('Using Multi-Label %s model' %(str(xgbm).split("(")[0]))
    ######   Now do RandomizedSearchCV  using # Early-stopping ################
    if modeltype == 'Regression':
        if model_label == 'Single_Label':
            #scoreFunction = {"mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error"}
            #### I have set the Verbose to be False here since it produces too much output ###
            if hyper_param == 'GS':
                #### I have set the Verbose to be False here since it produces too much output ###
                gs = GridSearchCV(xgbm,param_grid=r_params[model_name],
                                               scoring = scorer,
                                               cv = scv,
                                               refit = refit_metric,
                                               return_train_score = True,
                                               n_jobs=n_jobs,
                                                verbose=0)
            elif hyper_param == 'RS':
                gs = RandomizedSearchCV(xgbm,
                                               param_distributions = r_params[model_name],
                                               n_iter = no_iter,
                                               scoring = scorer,
                                               refit = refit_metric,
                                               return_train_score = True,
                                               random_state = seed,
                                               cv = scv,
                                               n_jobs=n_jobs,
                                                verbose = 0)
            else:
                #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
                gs = copy.deepcopy(xgbm)
        else:
            #### For Multilabel Regressors you need to do some hyper parameter tuning but not too much!  ####
            #### Better to reduce the number of iterations and no hyper-param searching since it takes too long #########
            #scorer = 'neg_mean_squared_error'
            no_iter = 5
            if hyper_param == 'GS':
                #### I have set the Verbose to be False here since it produces too much output ###
                gs = GridSearchCV(xgbm,param_grid=r_params[model_label][model_name],
                                               scoring = scorer,
                                               n_jobs=n_jobs,
                                               cv = scv,
                                               refit = refit_metric,
                                               return_train_score = True,
                                                verbose=0)
            elif hyper_param == 'RS':
                gs = RandomizedSearchCV(xgbm,
                                               param_distributions = r_params[model_label][model_name],
                                               n_iter = no_iter,
                                               scoring = scorer,
                                               refit = refit_metric,
                                               return_train_score = True,
                                               random_state = seed,
                                               cv = scv,
                                               n_jobs=n_jobs,
                                                verbose = 0)
            else:
                #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
                gs = copy.deepcopy(xgbm)
    else:
        ####### This is for Classification Problems #############
        if model_label == 'Single_Label':
            if hyper_param == 'GS':
                #### I have set the Verbose to be False here since it produces too much output ###
                gs = GridSearchCV(xgbm,param_grid=c_params[model_name],
                                               scoring = scorer,
                                               return_train_score = True,
                                               n_jobs=n_jobs,
                                               refit = refit_metric,
                                               cv = scv,
                                                verbose=0)
            elif hyper_param == 'RS':
                #### I have set the Verbose to be False here since it produces too much output ###
                gs = RandomizedSearchCV(xgbm,
                                               param_distributions = c_params[model_name],
                                               n_iter = no_iter,
                                               scoring = scorer,
                                               refit = refit_metric,
                                               return_train_score = True,
                                               random_state = seed,
                                               n_jobs=n_jobs,
                                               cv = scv,
                                                verbose = 0)
            else:
                #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
                gs = copy.deepcopy(xgbm)
        else:
            ## For multi-label problems, you can't do GridSearch or RandomizedSearch since it will take too long
            gs = copy.deepcopy(xgbm)
    #trains and optimizes the model #########
    eval_set = [(X_train,y_train),(X_cv,y_cv)]
    print('Finding Best Model and Hyper Parameters for %s model...' %model_name)
    ##### Here is where we put the part_train and part_cv together ###########
    if modeltype != 'Regression':
        ### Do this only for Binary Classes and Multi-Classes, both are okay
        baseline_accu = 1-(train[each_target].value_counts(1).sort_values())[rare_class]
        print('    Baseline Accuracy Needed for Model = %0.2f%%' %(baseline_accu*100))
    print('    CPU Count = %s in this device' %CPU_count)
    if modeltype == 'Regression':
        if Boosting_Flag:
            if model_name.lower() == 'catboost':
                data_dim = data_dim*one_hot_size/len(preds)
                print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(3000000.*CPU_count)))
            else:
                print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(50000.*CPU_count)))
        elif Boosting_Flag is None:
            print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(80000.*CPU_count)))
        else:
            print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(40000.*CPU_count)))
    else:
        ##### This is for classification which differs for GridSearch vs RandomizedSearch ##############
        if hyper_param == 'GS':
            if Boosting_Flag:
                if model_name.lower() == 'catboost':
                    data_dim = data_dim*one_hot_size/len(preds)
                    print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(300000.*CPU_count)))
                else:
                    print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*(max_class_length**2)*len(target)/(100000.*CPU_count)))
            elif Boosting_Flag is None:
                #### A Linear model is usually the fastest ###########
                print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(50000.*CPU_count)))
            else:
                print('Using %s Model, Estimated Training time = %0.3f mins' %(model_name,data_dim*max_class_length*len(target)/(16000.*CPU_count)))
        else:
            if Boosting_Flag:
                if model_name.lower() == 'catboost':
                    data_dim = data_dim*one_hot_size/len(preds)
                    print('Using %s Model, Estimated Training time = %0.2f mins' %(model_name,data_dim*max_class_length*len(target)/(3000000.*CPU_count)))
                else:
                    print('Using %s Model, Estimated Training time = %0.2f mins' %(model_name,data_dim*(max_class_length)*len(target)**2/(40000.*CPU_count)))
            elif Boosting_Flag is None:
                print('Using %s Model, Estimated Training time = %0.2f mins' %(model_name,data_dim*max_class_length*len(target)/(100000.*CPU_count)))
            else:
                print('Using %s Model, Estimated Training time = %0.2f mins' %(model_name,data_dim*max_class_length*len(target)/(25000.*CPU_count)))
    ##### Since we are using Multiple Models each with its own quirks, we have to make sure it is done this way
    ##### ############      TRAINING MODEL FIRST TIME WITH X_TRAIN AND TESTING ON X_CV ############
    model_start_time = time.time()
    ############################################################################
    #####   BE VERY CAREFUL ABOUT MODIFYING THIS SECTION JUST BECAUSE ##########
    ####    IT APPEARS LIKE DUPLICATED CODE!!  IT IS NOT !!        #############
    ############################################################################
    ##### This is where you start training the model for FIRST TIME  ############
    if model_label == 'Single_Label':
        if Imbalanced_Flag:
            try:
                print('##################  Imbalanced Model Training  ############################')
                print('Imbalanced Training using SMOTE Rare Class Oversampling method...')
                if modeltype == 'Regression':
                    ### Since rare class is needed in the next step, just setting the rare_class as artificially 0.
                    rare_class = 0
                ### Since it is a single label problem, we use each_target instead of target
                ###   Also do not use the GridSearchCV model GS here since SMOTE cannot work with it
                ### use the base model trained with SMOTE.
                model = training_with_SMOTE(X_train,y_train,each_target,eval_set, xgbm,
                                       Boosting_Flag, eval_metric,
                                       modeltype, model_name, training=True,
                                       minority_class=rare_class,imp_cats=imp_cats,
                                       calibrator_flag=calibrator_flag,
                                       GPU_exists=GPU_exists, params = cpu_params,
                                       model_label=model_label, verbose=verbose)
            except:
                print('Training model first time with SMOTE erroring. Continuing...')
                Imbalanced_Flag =  False
                best_score = 0
    #########   IMPORTANT    IMPORTANT #####################################
    ##   Though this next step looks like it is a Coding Mistake, don't change it!
    ##   This is for case when Imbalanced with Classification succeeds,
    ##   this next step is skipped ##
    #########################################################################
    ### The model blows up when you use average-precision in Multi-Classes -> needs checking!
    if not Imbalanced_Flag:
        #### This is for both regular Regression and regular Classification Model Training. It is not a Mistake #############
        ### In case Imbalanced training fails, this method is also tried. That's why we test the Flag here!!  #############
        try:
            model = copy.deepcopy(gs)
            if model_label == 'Single_Label':
                if Boosting_Flag:
                    if model_name == 'XGBoost':
                        try:
                            if  modeltype == 'Multi_Classification':
                                model.fit(X_train,y_train)
                            else:
                                #### Set the Verbose to 0 since we don't want too much output ##
                                model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                                        eval_metric=eval_metric,eval_set=eval_set,verbose=0)
                        except:
                            #### On Colab, even though GPU exists, many people don't turn it on.
                            ####  In that case, XGBoost blows up when gpu_predictor is used.
                            ####  This is to turn it back to cpu_predictor in case GPU errors!
                            if GPU_exists:
                                print('Error: GPU exists but it is not turned on. Using CPU for predictions...')
                                model.estimator.set_params(**cpu_params)
                                if  modeltype == 'Multi_Classification':
                                    model.fit(X_train,y_train)
                                else:
                                    model.fit(X_train,y_train, early_stopping_rounds=early_stopping,
                                        eval_metric=eval_metric,eval_set=eval_set,verbose=False)
                            else:
                                model.fit(X_train,y_train)
                    else:
                        ##### This is for Catboost ###############
                        try:
                            model.fit(X_train, y_train, cat_features=imp_cats,
                                        eval_set=(X_cv,y_cv), use_best_model=True, plot=True)
                        except:
                            model.fit(X_train, y_train, cat_features=imp_cats, use_best_model=False, plot=False)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
        except:
            print('Training regular model first time is Erroring: Check if your Input is correct...')
            return
    ###########   FIRST TIME MODEL TRAINING COMPLETED ##########################
    ##   TRAINING OF MODELS COMPLETED. NOW GET METRICS on CV DATA ###############
    print('    Actual training time (in seconds): %0.0f' %(time.time()-model_start_time))
    print('###########  %s  M O D E L   R E S U L T S #################' %model_label)
    try:
        if model_label == 'Single_Label':
            if hyper_param == 'RS' or hyper_param == 'GS':
                best_score = model.best_score_
                validation_metric = copy.deepcopy(scoring_parameter)
            elif hyper_param is None:
                val_keys = list(model.best_score_.keys())
                if 'validation' in val_keys:
                    validation_metric = list(model.best_score_['validation'].keys())[0]
                    best_score = model.best_score_['validation'][validation_metric]
                else:
                    validation_metric = list(model.best_score_['learn'].keys())[0]
                    best_score = model.best_score_['learn'][validation_metric]
    except:
        print('Error: Not able to print validation metrics. Continuing...')
        best_score = 0
    if model_label == 'Single_Label':
        if modeltype != 'Regression':
            ############## This is for Classification Only !! ########################
            if scoring_parameter in ['logloss','neg_log_loss','log_loss','log-loss','']:
                print('{}-fold Cross Validation {} = {}'.format(n_splits, 'logloss', best_score))
            elif scoring_parameter in ['accuracy','balanced-accuracy','balanced_accuracy','roc_auc','roc-auc',
                                       'f1','precision','recall','average-precision','average_precision',
                                      'weighted_f1','weighted-f1','AUC']:
                print('%d-fold Cross Validation  %s = %0.1f%%' %(n_splits,scoring_parameter, best_score*100))
            else:
                print('%d-fold Cross Validation  %s = %0.1f' %(n_splits,validation_metric, best_score))
        else:
            ######### This is for Regression only ###############
            if best_score < 0:
                best_score = best_score*-1
            if scoring_parameter == '':
                print('%d-fold Cross Validation %s Score = %0.4f' %(n_splits,'RMSE', best_score))
            else:
                print('%d-fold Cross Validation %s Score = %0.4f' %(n_splits,validation_metric, best_score))
        #### We now need to set the Best Parameters, Fit the Model on Full X_train and Predict on X_cv
        ### Find what the order of best params are and set the same as the original model ###
        if hyper_param == 'RS' or hyper_param == 'GS':
            if not Imbalanced_Flag:
                best_params= model.best_params_
                print('    Best Parameters for Model = %s' %model.best_params_)
        elif hyper_param is None:
            #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
            #### CatBoost does not need too many iterations. Just make sure you set the iterations low after the first time!
            if not Imbalanced_Flag:
                if model.get_best_iteration() == 0:
                    ### In some small data sets, the number of iterations becomes zero, hence we set it as a default number
                    best_params = dict(zip(['iterations','learning_rate'],[1000,model.get_all_params()['learning_rate']]))
                else:
                    best_params = dict(zip(['iterations','learning_rate'],[model.get_best_iteration(),model.get_all_params()['learning_rate']]))
                print('    %s Best Parameters for Model: Iterations = %s, learning_rate = %0.2f' %(
                                    model_name, model.get_best_iteration(), model.get_all_params()['learning_rate']))
            elif Imbalanced_Flag and modeltype == 'Regression':
                ## this means regression resampler is being used
                if model.get_best_iteration() == 0:
                    ### In some small data sets, the number of iterations becomes zero, hence we set it as a default number
                    best_params = dict(zip(['iterations','learning_rate'],[1000,model.get_all_params()['learning_rate']]))
                else:
                    best_params = dict(zip(['iterations','learning_rate'],[model.get_best_iteration(),model.get_all_params()['learning_rate']]))
                print('    %s Best Parameters for Model: Iterations = %s, learning_rate = %0.2f' %(
                                    model_name, model.get_best_iteration(), model.get_all_params()['learning_rate']))
            else:
                best_params = {}
        if hyper_param == 'RS' or hyper_param == 'GS':
            #### In the case of CatBoost, we don't do any Hyper Parameter tuning #########
            if not Imbalanced_Flag:
                gs = copy.deepcopy(model)
                model = gs.best_estimator_
            else:
                print('No cross validation done when Imbalanced_Flag set to True...')
            if model_label == 'Single_Label' and modeltype == 'Multi_Classification':
                try:
                    if X_cv.shape[0] <= 1000:
                        # THis works well for small data sets and is similar to parametric
                        method=  'sigmoid' # 'isotonic' # #
                    else:
                        # THis works well for large data sets and is non-parametric
                        method=  'isotonic'
                    from sklearn.calibration import CalibratedClassifierCV
                    model = CalibratedClassifierCV(model, method=method, cv=n_splits)
                    if not perform_scaling_flag:
                        X_ful = X_train.append(X_cv)
                        y_ful = y_train.append(y_cv)
                    else:
                        X_ful = np.r_[X_train, X_cv]
                        y_ful = np.r_[y_train, y_cv]
                    model.fit(X_ful, y_ful)
                    print('Using a Calibrated Classifier in this Multi_Classification dataset to improve results...')
                    calibrator_flag = True
                except:
                    calibrator_flag = False
            elif model_label == 'Multi_Label' and modeltype == 'Binary_Classification':
                print('    No GridSearch or RandomizedSearch for %s %s since it takes too long' %(
                                    model_label,modeltype))
        #if model_name.lower() == 'catboost':
        #    print('Best Model selected and its parameters are:\n    %s' %model.get_all_params())
        #else:
        #    print('Best Model selected and its attributes are:\n    %s' %model)
    else:
        ######## If this is a multi-label model, try to see if it went throgh GS or RS, if not, use it as is
        try:
            gs = copy.deepcopy(model)
            model = gs.best_estimator_
        except:
            gs = copy.deepcopy(model)
        print('    Using Multi-Output version of %s' %model)
    ##############   These are settings for making sure predictions as same type as target ################
    ### Make sure you set this flag as False so that when ensembling is completed, this flag is True ##
    performed_ensembling = False
    modify_targets_flag = False
    if model_label == 'Single_Label' and modeltype != 'Regression':
        ############   This is for Single Label Classification Problems only ######
        m_thresh = 0.5
        y_proba = model.predict_proba(X_cv)
        y_pred = model.predict(X_cv)
        if len(classes) <= 2:
            print('Finding Best Threshold for Highest F1 Score...')
            precision, recall, thresholds = precision_recall_curve(y_cv, y_proba[:,rare_class])
            #precision, recall, thresholds = precision_recall_curve(y_cv, y_proba[:,1])
            try:
                f1 = (2*precision*recall)/(precision+recall)
                f1 = np.nan_to_num(f1)
                m_idx = np.argmax(f1)
                m_thresh = thresholds[m_idx]
                best_f1 = f1[m_idx]
            except:
                best_f1 = f1_score(y_cv, y_pred)
                m_thresh = 0.5
            # retrieve just the probabilities for the positive class
            pos_probs = y_proba[:, rare_class]
            if verbose >= 1:
                # create a histogram of the predicted probabilities for the Rare Class since it will help decide threshold
                plt.figure(figsize=(6,6))
                plt.hist(pos_probs, bins=Bins, color='g')
                plt.title("Model's Predictive Probability Histogram for Rare Class=%s with suggested threshold in red" %rare_class_orig)
                plt.axvline(x=m_thresh, color='r', linestyle='--')
                plt.show();
            print("    Using threshold=0.5. However, %0.3f provides better F1=%0.2f for rare class..." %(m_thresh,best_f1))
            ###y_pred = (y_proba[:,rare_class]>=m_thresh).astype(int)
            predicted = copy.deepcopy(y_proba)
            predicted [:,0] = (predicted [:,0] >= (1-m_thresh)).astype('int')
            predicted [:,1] = (predicted [:,1] > m_thresh).astype('int')
            if m_thresh != 0.5:
                y_pred = predicted[:,rare_class]
        else:
            #### This is where you predict for Multi-class classification models #####
            y_pred = model.predict(X_cv)
    else:
        #### This is where you predict for Multi-Output Multi-Label models #####
        y_pred = model.predict(X_cv)
    ###   This is where you print out the First Model's Results ########
    print('########################################################')
    ##### This next step is very important since some models give series, others give arrays. Very painful!
    if isinstance(y_pred,pd.Series) or isinstance(y_pred,pd.DataFrame):
        y_pred = y_pred.values
    if isinstance(y_cv,pd.Series) or isinstance(y_cv,pd.DataFrame):
        y_cv = y_cv.values
    print('%s Model Prediction Results on Held Out CV Data Set:' %model_name)
    ###### Make sure that you change the model's predictions to be the same type as original target ####
    int_types = [np.int8, np.int16, np.int32, np.int64]
    targets_negative_flag = len(y_cv<0) == 0
    if y_cv.dtype in int_types:
        y_pred = modify_array_to_integer(y_pred, targets_negative_flag)
        modify_targets_flag = True
    #############    N O W   P R I N T   A L L  R E S U L T S ####################
    if model_label == 'Single_Label':
        if modeltype == 'Regression':
            rmsle_calculated_m = rmse(y_cv, y_pred)
            print_regression_model_stats(y_cv, y_pred,target,plot_name=model_name)
        else:
            if model_name == 'Forests':
                if calibrator_flag:
                    print('    OOB Score = %0.3f' %model.base_estimator.oob_score_)
                else:
                    print('    OOB Score = %0.3f' %model.oob_score_)
            rmsle_calculated_m = balanced_accuracy_score(y_cv,y_pred)
            if len(classes) == 2:
                print('    Regular Accuracy Score = %0.1f%%' %(accuracy_score(y_cv,y_pred)*100))
                y_probas = model.predict_proba(X_cv)
                rmsle_calculated_m = print_classification_model_stats(y_cv, y_probas, m_thresh)
            else:
                ###### Use a nice classification matrix printing module here #########
                y_probas = model.predict_proba(X_cv)
                print_classification_metrics(y_cv, y_probas)
                print(classification_report(y_cv,y_pred))
                print(confusion_matrix(y_cv, y_pred))
    else:
        if modeltype == 'Regression':
            #### This is for Multi-Label Regression ################################
            print_regression_model_stats(y_cv, y_pred,target,plot_name=model_name)
        else:
            y_probas = model.predict_proba(X_cv)
            #### This is for Multi-Label Classification ################################
            print_classification_metrics(y_cv, y_pred, False)
            print(classification_report(y_cv, y_pred ))
    ######      SET BEST PARAMETERS HERE ######
    ### Find what the order of best params are and set the same as the original model ###
    ## This is where we set the best parameters from training to the model ####
    if model_label == 'Single_Label':
        #### You do ensembling or stacking only for Single_Label problems ############
        if modeltype == 'Regression':
            if not Stacking_Flag:
                print('################# E N S E M B L E  M O D E L  ##################')
                try:
                    cols = []
                    subm = pd.DataFrame()
                    #### This is for Ensembling  Only #####
                    models_list, cv_ensembles = QuickML_Ensembling(X_train, y_train, X_cv, y_cv,
                                              modeltype=modeltype, Boosting_Flag=Boosting_Flag,
                                               scoring='', verbose=verbose)
                    models_list.append(model_name)
                    for models, each in zip(models_list, range(len(models_list))):
                        new_col = each_target+'_'+models+'_predictions'
                        if each+1 == len(models_list):
                            subm[new_col] = y_pred
                        else:
                            subm[new_col] = cv_ensembles[:,each]
                        cols.append(new_col)
                    if len(cols) == 5:
                        print('    Displaying results of weighted average ensemble of %d regressors' %len(cols))
                        ensem_pred = subm[cols[-1]]*0.5+0.125*(subm[cols[0]]+subm[
                                        cols[1]]+subm[cols[2]]+subm[cols[3]])
                    else:
                        print('    Calculating regular average ensemble of %d regressors' %len(cols))
                        ensem_pred = (subm[cols].mean(axis=1))
                    print('#############################################################################')
                    performed_ensembling = True
                    #### Since we have a new ensembled y_pred, make sure it is an array before printing it!
                    if isinstance(ensem_pred,pd.Series) or isinstance(ensem_pred,pd.DataFrame):
                        print_regression_model_stats(y_cv, ensem_pred.values, target,plot_name='Ensemble')
                    else:
                        print_regression_model_stats(y_cv, ensem_pred, target, plot_name='Ensemble')
                except:
                    print('Could not complete Ensembling predictions on held out data due to Error')
        else:
            ##  This is for Classification Problems Only #
            ### Find what the order of best params are and set the same as the original model ###
            ## This is where we set the best parameters from training to the model ####
            if not Stacking_Flag:
                print('################# E N S E M B L E  M O D E L  ##################')
                #### We do Ensembling only if the Stacking_Flag is False. Otherwise, we don't!
                try:
                    classes = label_dict[each_target]['classes']
                    cols = []
                    subm = pd.DataFrame()
                    #### This is for Ensembling  Only #####
                    if len(classes) == 2:
                        models_list, cv_ensembles = QuickML_Ensembling(X_train, y_train, X_cv, y_cv,
                                                  modeltype='Binary_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=verbose)
                    else:
                        models_list, cv_ensembles = QuickML_Ensembling(X_train, y_train, X_cv, y_cv,
                                                  modeltype='Multi_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=verbose)
                    models_list.append(model_name)
                    for models, each in zip(models_list, range(len(models_list))):
                        new_col = each_target+'_'+models+'_predictions'
                        if each+1 == len(models_list):
                            subm[new_col] = y_pred
                        else:
                            subm[new_col] = cv_ensembles[:,each]
                        cols.append(new_col)
                    print('#############################################################################')
                    if len(cols) == 5:
                        print('    Displaying results of weighted average ensemble of %d classifiers' %len(cols))
                        ensem_pred = np.round(subm[cols[-1]]*0.5+0.125*(subm[cols[0]]+subm[
                                        cols[1]]+subm[cols[2]]+subm[cols[3]])).astype(int)
                    else:
                        print('    Calculating regular average ensemble of %d classifiers' %len(cols))
                        ensem_pred = (subm[cols].mean(axis=1)).astype(int)
                    performed_ensembling = True
                    ##### This next step is very important since some models give series, others give arrays. Very painful!
                    if isinstance(ensem_pred,pd.Series):
                        ensem_pred = ensem_pred.values
                except:
                    print('Could not complete Ensembling predictions on held out data due to Error')
            else:
                print('No Ensembling of models done since Stacking_Flag = True ')
            if verbose >= 1:
                if len(classes) == 2:
                    plot_classification_results(model,X_cv, y_cv, y_pred, classes, class_nums, each_target )
                else:
                    try:
                        Draw_ROC_MC_ML(model, X_cv, y_cv, each_target, model_name, verbose)
                        Draw_MC_ML_PR_ROC_Curves(model,X_cv,y_cv)
                    except:
                        print('Could not plot PR and ROC curves. Continuing...')
            #### In case there are special scoring_parameter requests, you can print it here!
            if scoring_parameter == 'roc_auc' or scoring_parameter == 'auc':
                if len(classes) == 2:
                    print('    ROC AUC Score = %0.1f%%' %(roc_auc_score(y_cv, y_proba[:,rare_class])*100))
                else:
                    print('    No ROC AUC score for multi-class problems')
        if not Stacking_Flag and performed_ensembling:
            if modeltype == 'Regression':
                rmsle_calculated_f = rmse(y_cv, ensem_pred)
                print('After multiple models, Ensemble Model Results:')
                print('    RMSE Score = %0.5f' %(rmsle_calculated_f,))
                print('#############################################################################')
                if rmsle_calculated_f < rmsle_calculated_m:
                    print('Ensembling Models is better than Single Model for this data set.')
                    error_rate.append(rmsle_calculated_f)
                else:
                    print('Single Model is better than Ensembling Models for this data set.')
                    error_rate.append(rmsle_calculated_m)
            else:
                rmsle_calculated_f = balanced_accuracy_score(y_cv,ensem_pred)
                print('After multiple models, Ensemble Model Results:')
                rare_pct = y_cv[y_cv==rare_class].shape[0]/y_cv.shape[0]
                #print('    Balanced Accuracy Score = %0.3f%%' %(
                #        rmsle_calculated_f*100))
                print_classification_metrics(y_cv, ensem_pred, False)
                print(classification_report(y_cv,ensem_pred))
                print(confusion_matrix(y_cv,ensem_pred))
                print('#############################################################################')
                if rmsle_calculated_f > rmsle_calculated_m:
                    print('Ensembling Models is better than Single Model for this data set.')
                    error_rate.append(rmsle_calculated_f)
                else:
                    print('Single Model is better than Ensembling Models for this data set.')
                    error_rate.append(rmsle_calculated_m)
        if verbose >= 1:
            if Boosting_Flag:
                try:
                    if model_name.lower() == 'catboost':
                        plot_xgb_metrics(model,catboost_scoring,eval_set,modeltype,'%s Results' %each_target,
                                        model_name)
                    else:
                        plot_xgb_metrics(gs.best_estimator_,eval_metric,eval_set,modeltype,'%s Results' %each_target,
                                        model_name)
                except:
                    print('Could not plot Model Evaluation Results Metrics')
            else:
                try:
                    plot_RS_params(gs.cv_results_, scoring_parameter, each_target)
                except:
                    print('Could not plot Cross Validation Parameters')
        print('    Time taken for this Target (in seconds) = %0.0f' %(time.time()-start_time))
    else:
        #### You can use Jaccard Similarity for Multi-Label Problems ###########
        if scoring_parameter == 'jaccard':
            accu_all = jaccard_singlelabel(y_cv, y_pred)
            print('        Mean Jaccard Similarity  = {:,.1f}%'.format(
                                    accu_all*100))
            ## This is for multi-label problems ##
            if multilabel_count == 0:
                zipped = copy.deepcopy(y_pred)
                multilabel_count += 1
            else:
                zipped = zip(zipped,y_pred)
                multilabel_count += 1
        elif scoring_parameter == 'basket_recall':
            if multilabel_count == 0:
                zipped = copy.deepcopy(y_pred)
                multilabel_count += 1
            else:
                zipped = zip(zipped,y_pred)
                multilabel_count += 1
        print('Training model on complete Train data and Predicting using given Test Data...')
    ################        I M P O R T A N T: C O M B I N I N G  D A T A ######################
    #### This is Second time: we combine train and CV into Train and Test Sets #################
    train = part_train.append(part_cv)
    important_features = [x for x in list(train) if x not in target]
    ############################################################################################
    if model_label == 'Single_Label':
        ###### Now that we have used partial data to make stacking predictors, we can remove them from consideration!
        if Stacking_Flag:
            important_features = left_subtract(important_features, addcol)
            try:
                train.drop(addcol,axis=1, inplace=True)
            except:
                pass
        ###### Similarly we will have to create KMeans_Clusters again using full Train data!
        if KMeans_Featurizer:
            important_features = left_subtract(important_features, km_label)
            try:
                train.drop(km_label,axis=1, inplace=True)
            except:
                pass
        ########################## BINNING SECOND TIME  ###############################
        new_num_vars = train[important_features].select_dtypes(include=[np.float64,np.float32,np.float16]).columns.tolist()
        ## Now we re-use the saved_num_vars which contained a list of num_vars for binning now!
        ###### Once again we do Entropy Binning on the Full Train Data Set !!
        ########################## BINNING SECOND TIME  ###############################
        if Binning_Flag and len(saved_num_vars) > 0:
            ### when you bin the second time, you have to send in important_features with original
            ### numeric variables so that it works on binning only those. Otherwise it will fail.
            #important_features = left_subtract(important_features,binned_num_vars)
            #train = train[important_features+[each_target]]
            ### Do Entropy Binning only if there are numeric variables in the data set! #####
            #### When we Bin the second first time, we set the entropy_binning flag to True so
            ####    that all numeric variables that are binned are removed. This way, only bins remain.
            train, num_vars, important_features, test = add_entropy_binning(train, each_target,
                                                  saved_num_vars, important_features, test,
                                                  modeltype,  entropy_binning=True,verbose=verbose)
            #### In saved_num_vars we send in all the continuous_vars but we bin only the top few vars.
            ###  Those that are binned are removed from saved_num_vars and the remaining become num_vars
            ### Our job is to find the names of those original numeric variables which were binned.
            ### orig_num_vars contains original num vars. num_vars contains binned versions of those vars.
            ### Those binned variables have now become categorical vars and must be added to imp_cats.
            #### Also note that important_features does not contain orig_num_vars which have been erased.
        else:
            print('    Binning_Flag set to False or there are no float vars in data set to be binned')
        ### Now we add another Feature tied to KMeans clustering using Predictor and Target variables ###
        #######################   KMEANS   SECOND   TIME     ############################
        if KMeans_Featurizer and len(saved_num_vars) > 0:
            #### Perform KMeans Featurizer only if there are numeric variables in data set! #########
            print('Adding one feature named %s using KMeans_Featurizer...' %km_label)
            #### Make the number of clusters as the same as log10 of number of rows in Train
            if isinstance(test, str):
                #### In some cases, test data may not be available. In that case, just send in train as test
                train_cluster, _ = Transform_KM_Features(train[important_features], train[
                                    each_target], train[important_features], num_clusters)
            else:
                #### If test data is available, send in test data to get KM clusters for test ###
                train_cluster, test_cluster = Transform_KM_Features(train[important_features], train[
                                    each_target], test[important_features], num_clusters)
            #### Now make sure that the cat features are either string or integers ######
            print('    Used KMeans to naturally cluster Train predictor variables into %d clusters' %num_clusters)
            train[km_label] = train_cluster
            if not isinstance(test, str):
                test[km_label] = test_cluster
            #X_train.drop(each_target,axis=1,inplace=True)
            for imp_cat in imp_cats:
                train[imp_cat] = train[imp_cat].astype(int)
                if not isinstance(test, str):
                    test[imp_cat] = test[imp_cat].astype(int)
            saved_num_vars.append(km_label) ### You need to add it to this variable list for Scaling later!
            important_features.append(km_label)
        ########################## STACKING SECOND TIME  ###############################
        ######### This is where you do Stacking of Multi Model Results into One Column ###
        if Stacking_Flag:
            try:
                #### In order to join, you need X_train to be a Pandas Series here ##
                print('CAUTION: Stacking can produce Highly Overfit models on Training Data...')
                ### In order to avoid overfitting, we are going to learn from a small sample of data
                ### That is why we are using X_cv to train on and using it to predict on X_train!
                addcol, stacks1 = QuickML_Stacking(train[important_features],train[each_target],'',
                              modeltype, Boosting_Flag, scoring_parameter,verbose)
                ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                #### The reason we add the word "Partial_Train" is to show that these Stacking results are from Partial Train data!
                addcols = copy.deepcopy(addcol)
                ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                train = train.join(pd.DataFrame(stacks1,index=train.index,
                                                  columns=addcols))
                ##### Leaving multiple columns for Stacking is best! Do not do the average of predictions!
                print('    Adding %d Stacking feature(s) to training data' %len(addcols))
                if not isinstance(orig_test, str):
                    ### In order to avoid overfitting, we are going to learn from a small sample of data
                    ### That is why we are using X_train to train on and using it to predict on X_test
                    _, stacks2 = QuickML_Stacking(train[important_features],train[each_target],test[important_features],
                              modeltype, Boosting_Flag, scoring_parameter,verbose)
                    ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                    test = test.join(pd.DataFrame(stacks2,index=test.index,
                                                      columns=addcols))
                    ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                    #test = test.join(pd.DataFrame(stacks2.mean(axis=1).round().astype(int),
                    #                             columns=[addcol],index=test.index))
                ######  We make sure that we remove too many features that are highly correlated ! #####
                #addcol = remove_variables_using_fast_correlation(train,addcol,modeltype,corr_limit,verbose)
                important_features += addcols
                saved_num_vars.append(addcol) ### You need to add it for binning later!
            except:
                print('Error in Stacking second time. Continuing...')
    ############################################################################################
    if len(important_features) == 0:
        print('No important features found. Using all input features...')
        important_features = copy.deepcopy(saved_important_features)
        #important_features = copy.deepcopy(red_preds)
    ############################################################################################
    if model_name.lower() == 'catboost':
        if model_label == 'Single_Label':
            print('    Setting best params for CatBoost model from Initial State since you cannot change params to a fitted Catboost model ')
            model = xgbm.set_params(**best_params)
        print('    Number of Categorical and Integer variables used in CatBoost training = %d' %len(imp_cats))
    #### Perform Scaling of Train data a second time using FULL TRAIN data set this time !
    #### important_features keeps track of all variables that we need to ensure they are scaled!
    if not perform_scaling_flag:
        print('No %s scaling performed since scaling flag is set to false' %scaling)
        X = train[important_features]
        if not isinstance(orig_test, str):
            X_test = test[important_features]
    else:
        print('Performing %s scaling of train and test data...' %scaling)
        X = train[important_features]
        SS.fit(X)
        X = pd.DataFrame(SS.transform(X),index=orig_train.index,
                                        columns=important_features)
        if not isinstance(orig_test, str):
            X_test = pd.DataFrame(SS.transform(test[important_features]),index=orig_test.index,
                                        columns=important_features)
    print('#####   T R A I N I N G   M O D E L   O N    F U L L   T R A I N  D A T A  #############')
    ### The next 2 lines are crucial: if X and y are dataframes, then next 2 should be df's
    ###   They should not be df.values since they will become numpy arrays and XGB will error.
    trainm = train[important_features+target]
    red_preds = copy.deepcopy(important_features)
    if model_label == 'Single_Label':
        y = trainm[each_target]
    else:
        y = trainm[target]
    eval_set = [()]
    ##### ############      TRAINING MODEL SECOND TIME WITH FULL_TRAIN AND PREDICTING ON TEST ############
    model_start_time = time.time()
    if model_label == 'Single_Label':
        if Imbalanced_Flag:
            try:
                print('##################  Imbalanced Model Training  ############################')
                print('Imbalanced Training using SMOTE Rare Class Oversampling method...')
                ### Since it is a single label problem, we use each_target instead of target
                model = training_with_SMOTE(X,y, each_target, eval_set, model,
                                  Boosting_Flag, eval_metric,modeltype, model_name,
                                  training=False, minority_class=rare_class,
                                  imp_cats=imp_cats, calibrator_flag=calibrator_flag,
                                  GPU_exists=GPU_exists, params=cpu_params,
                                  model_label=model_label, verbose=verbose)
            except:
                print('Error in training Imbalanced model second time. Trying regular model..')
                Imbalanced_Flag =  False
    #########   IMPORTANT    IMPORTANT #####################################
    #### This is not a mistake! In case, Imbalanced Training Fails
    ####   or if there is no flag, we must still train the model!!
    ####   DO NOT TOUCH THIS NEXT LINE THINKING IT IS A MISTAKE!
    if not Imbalanced_Flag:
        ## This is where we test if the previous training with SMOTE went smoothly. If not, we  try with a different method here.
        #### If downsampling model failed, it will just be an empty string, so you can try regular model ###
        if model_label == 'Single_Label':
            if calibrator_flag:
                model.fit(X, y)
            else:
                if Boosting_Flag:
                        #### Set the Verbose to 0 since we don't want too much output ##
                    if model_name == 'XGBoost':
                        #### Set the Verbose to 0 since we don't want too much output ##
                        model.fit(X, y)
                    else:
                        model.fit(X, y, cat_features=imp_cats, plot=False)
                else:
                        model.fit(X, y)
    else:
        ### This is for Multi-Label Training #########
        model.fit(X, y)
    print('     Actual Training time taken in seconds = %0.0f' %(time.time()-model_start_time))
    print('Training of models completed. Now starting predictions on test data...')
    #### new_cols is to keep track of new prediction columns we are creating #####
    new_cols = []
    if not isinstance(orig_test, str):
        ### If there is a test data frame, then let us predict on it #######
        ### The next 3 lines are crucial: if X and y are dataframes, then next 2 should be df's
        ###   They should not be df.values since they will become numpy arrays and XGB will error.
        try:
            #### We need the id columns to carry over into the predictions ####
            testm = orig_test[id_cols].join(pd.DataFrame(X_test, index=orig_test.index, columns=red_preds))
        except:
            ### if for some reason id columns are not available, then do without it
            testm = pd.DataFrame(X_test, index=orig_test.index, columns=red_preds)
    else:
        print('Since there is no Test file, we will do a final prediction on Train itself as return it as testm dataframe')
        orig_index = orig_train.index
        trainm = train.reindex(index = orig_index)
        if not perform_scaling_flag:
            X_test = trainm[important_features]
        else:
            X_test = pd.DataFrame(SS.transform(trainm[important_features]),
                    index=orig_index, columns=important_features)
        testm = orig_train[id_cols].join(X_test)
    ############  This is where we start predictions on test data if available #############
    try:
        y_pred = model.predict(X_test)
    except:
        if (X_train.dtypes == X_test.dtypes).values.all():
            print('Model erroring on predictions. Check if test data has any differences from train data')
        else:
            erroring_column = X_train.columns[~(X_train.dtypes == X_test.dtypes).values][0]
            print('Model erroring since %s has a different dtype in test compared to train. Fix it and re-run.' %erroring_column)
            return model, important_features, trainm, testm
    ##### This next step is very important since some models give series, others give arrays. Very painful!
    if isinstance(y_pred,pd.Series) or isinstance(y_pred,pd.DataFrame) :
        y_pred = y_pred.values
    ####### This is where you modify predictions on test data to be same as original data type #####
    if modify_targets_flag:
        y_pred = modify_array_to_integer(y_pred, targets_negative_flag)
    if modeltype == 'Regression':
        ########   This is for Regression Problems Only ###########
        ######  If Stacking_ Flag is False, then we do Ensembling #######
        if not Stacking_Flag and not model_label == 'Multi_Label':
            try:
                new_cols = []
                subm = pd.DataFrame()
                #### This is for Ensembling  Only #####
                #### In Test data verbose is set to zero since no results can be obtained!
                models_list, ensembles = QuickML_Ensembling(X, y, X_test, '',
                                          modeltype=modeltype, Boosting_Flag=Boosting_Flag,
                                           scoring='', verbose=0)
                models_list.append(model_name)
                for models, each in zip(models_list, range(len(models_list))):
                    new_col = each_target+'_'+models+'_predictions'
                    if each+1 == len(models_list):
                        subm[new_col] = y_pred
                        testm[new_col] = y_pred
                    else:
                        subm[new_col] = ensembles[:,each]
                        testm[new_col] = ensembles[:,each]
                    new_cols.append(new_col)
                ### After this, y_pred is a Series from now on. You need y_pred.values  ####
                if len(new_cols) == 5:
                    print('    Calculating weighted average ensemble of %d regressors' %len(new_cols))
                    ensem_pred = subm[new_cols[-1]]*0.5+0.125*(subm[new_cols[0]]+subm[
                                    new_cols[1]]+subm[new_cols[2]]+subm[new_cols[3]])
                else:
                    print('    Calculating regular average ensemble of %d regressors' %len(cols))
                    ensem_pred = (subm[new_cols].mean(axis=1))
                ##### This next step is very important since some models give series, others give arrays. Very painful!
                if isinstance(ensem_pred,pd.Series) or isinstance(ensem_pred,pd.DataFrame):
                    ensem_pred = ensem_pred.values
                new_col = each_target+'_Ensembled_predictions'
                testm[new_col] = ensem_pred
                new_cols.append(new_col)
                print('Completed Ensemble predictions on held out data')
            except:
                print('Could not complete Ensembling predictions on held out data due to Error')
        elif Stacking_Flag and not model_label == 'Multi_Label':
            stack_cols, stacksfinal = QuickML_Stacking(X, y, X_test,
                          modeltype, Boosting_Flag,
                          scoring_parameter,verbose=verbose)
            new_col = each_target+'_Stacked_'+stack_cols[0].split("_")[0]+'_predictions'
            if len(stack_cols) == 1:
                testm[new_col] = stacksfinal
            else:
                #### Just average the predictions from each stacked model into a final pred
                testm[new_col] = stacksfinal.mean(axis=1)
        ###### This applies to Single_Label as well as Multi_Label problems ##########
        if model_label == 'Single_Label':
            if not isinstance(sample_submission, str):
                try:
                    sample_submission[each_target] = y_pred
                except:
                    ### if for some reason, the sample submission file has a different shape than y_pred,
                    ###  this may cause an error and you can just skip that step!
                    print('Error: Sample submission file is incorrect. Please check it. continuing...')
            #### If there is a test file, it probably doesn't have target, so add predictions to it!
            testm[each_target+'_predictions'] = y_pred
        else:
            if not isinstance(sample_submission, str):
                for predict_col, i in zip(target, range(len(target))):
                    try:
                        sample_submission[predict_col] = y_pred[:,i]
                    except:
                        ### if for some reason, the sample submission file has a different shape than y_pred,
                        ###  this may cause an error and you can just skip that step!
                        print('Error: Sample submission file is incorrect. Please check it. continuing...')
        #### If there is a test file, it probably doesn't have target, so add predictions to it!
            for predict_col, i in zip(target, range(len(target))):
                testm[predict_col+'_predictions'] = y_pred[:,i]
    else:
        if model_label == 'Single_Label':
            #### Only certain multi-output classifiers provide probas. For now LinearSVC can't. Hence skip this...
            proba_cols = []
            ########   This is for both Binary and Multi Classification Problems ###########
            y_proba = model.predict_proba(X_test)
            predicted = copy.deepcopy(y_proba)
            if len(classes) <= 2:
                predicted [:,0] = (predicted [:,0] >= (1-m_thresh)).astype('int')
                predicted [:,1] = (predicted [:,1] > m_thresh).astype('int')
                if predicted[:,rare_class].mean()==0 or predicted[:,rare_class].mean()==1:
                    ### If the model is predicting all 0's or all 1's, you need to use a regular threshold
                    m_thresh = 0.5
                    print('    Making test Data predictions using regular Threshold = %0.3f' %m_thresh)
                else:
                    ### If the model is good with the modified threshold, then you use the modified threshold!
                    print('    Making test Data predictions using modified Threshold = %0.3f' %m_thresh)
                    y_pred = predicted[:,rare_class]
            ### You will need to create probabilities for each class here  ####
            for each_class in classes:
                if isinstance(each_class, str):
                    proba_col = each_target+'_proba_'+each_class
                else:
                    proba_col = each_target+'_proba_'+str(each_class)
                count = int(label_dict[each_target]['dictionary'][each_class])
                testm[proba_col] = y_proba[:,count]
                proba_cols.append(proba_col)
        ##### Now take each target in the list of targets and apply reverse transformation on predicted labels ####
        for each_target, i in zip(target, range(len(target))):
            if len(label_dict[each_target]['transformer']) == 0:
                #########  NO   T R A N S F O R M E R   L O G I C  B E G I N S    H E R E ! #####################
                ### if there is no transformer, then leave the predicted classes as is
                classes = label_dict[each_target]['classes']
                ##### If there is no transformer, you can just predict the classes as is and save it here ###
                predict_cols = each_target+'_predictions'
                if model_label == 'Single_Label':
                    testm[predict_cols] = y_pred
                else:
                    testm[predict_cols] = y_pred[:,i]
                ######  If Stacking_Flag is False, then we do Ensembling #######
                if not Stacking_Flag and model_label == 'Single_Label':
                    ### Ensembling is not done when the model name is CatBoost ####
                    new_cols = []
                    subm = pd.DataFrame()
                    #### This is for Ensembling  Only #####
                    #### In Test data verbose is set to zero since no results can be obtained!
                    if len(classes) == 2:
                        models_list, ensembles = QuickML_Ensembling(X, y, X_test, '',
                                                  modeltype='Binary_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=0)
                    else:
                        models_list, ensembles = QuickML_Ensembling(X, y, X_test, '',
                                                  modeltype='Multi_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=0)
                    models_list.append(model_name)
                    for models, each in zip(models_list, range(len(models_list))):
                        new_col = each_target+'_'+models+'_predictions'
                        if each+1 == len(models_list):
                            subm[new_col] = y_pred
                            testm[new_col] = y_pred
                        else:
                            subm[new_col] = ensembles[:,each]
                            testm[new_col] = ensembles[:,each]
                        new_cols.append(new_col)
                    ####### now combine the predictions into one ensembled prediction #######
                    new_col = each_target+'_Ensembled_predictions'
                    if len(new_cols) == 5:
                        print('    Calculating weighted average ensemble of %d classifiers' %len(new_cols))
                        ensem_pred = np.round(subm[new_cols[-1]]*0.5+0.125*(subm[new_cols[0]]+subm[
                                        new_cols[1]]+subm[new_cols[2]]+subm[new_cols[3]])).astype(int)
                    else:
                        print('    Calculating average ensemble of %d classifiers' %len(new_cols))
                        ensem_pred = (subm[new_cols].mean(axis=1)).astype(int)
                elif Stacking_Flag and model_label == 'Single_Label':
                    stack_cols, stacksfinal = QuickML_Stacking(X, y, X_test,
                                       modeltype, Boosting_Flag,scoring_parameter,verbose)
                    new_col = each_target+'_Stacked_'+stack_cols[0].split("_")[0]+'_predictions'
                    ensem_pred = np.argmax(stacksfinal,axis=1)
                if model_label == 'Single_Label':
                    ##### This next step is very important since some models give series, others give arrays. Very painful!
                    if isinstance(ensem_pred,pd.Series):
                        ensem_pred = ensem_pred.values
                    testm[new_col] = ensem_pred
                    new_cols.append(new_col)
                if not isinstance(sample_submission, str):
                    try:
                        if model_label == 'Single_Label':
                            sample_submission[each_target] = y_pred
                        else:
                            sample_submission[each_target] = y_pred[:,i]
                    except:
                        print('Error: Sample submission file is incorrect. Please check it. continuing...')
            else:
                #########    T R A N S F O R M E R   L O G I C  B E G I N S    H E R E ! #####################
                ### if there is a transformer, then you must convert the predicted classes to orig classes
                classes = label_dict[each_target]['classes']
                dic = label_dict[each_target]['dictionary']
                transformer = label_dict[each_target]['transformer']
                class_nums = label_dict[each_target]['class_nums']
                ##### If there is a transformer, you must convert predictions to original classes
                ### In some cases such as CatBoost, the output for multi-class is 2-D with shape (,1)
                try:
                    assert y_pred.shape[1]
                    y_pred = y_pred.ravel()
                except:
                    pass
                ###### Once you get the above right, you can do the rest easy ######
                testm[each_target+'_predictions'] = pd.Series(y_pred).map(transformer).values
                for each_class in classes:
                    if isinstance(each_class, str):
                        proba_col = each_target+'_proba_'+each_class
                    else:
                        proba_col = each_target+'_proba_'+str(each_class)
                    count = label_dict[each_target]['dictionary'][each_class]
                    testm[proba_col] = y_proba[:,count]
                    proba_cols.append(proba_col)
                ######  If Stacking_ Flag is False, then we do Ensembling #######
                if not Stacking_Flag:
                    subm = pd.DataFrame()
                    #### This is for Ensembling  Only #####
                    if len(classes) == 2:
                        models_list, ensembles = QuickML_Ensembling(X, y, X_test, '',
                                                  modeltype='Binary_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=verbose)
                    else:
                        models_list, ensembles = QuickML_Ensembling(X, y, X_test, '',
                                                  modeltype='Multi_Classification', Boosting_Flag=Boosting_Flag,
                                                   scoring='', verbose=verbose)
                    models_list.append(model_name)
                    for models, each in zip(models_list, range(len(models_list))):
                        new_col = each_target+'_'+models+'_predictions'
                        if each+1 == len(models_list):
                            subm[new_col] = y_pred
                            testm[new_col] = pd.Series(y_pred).map(transformer).values
                        else:
                            subm[new_col] = ensembles[:,each]
                            testm[new_col] = pd.Series(ensembles[:,each]).map(transformer).values
                        new_cols.append(new_col)
                    ### After this, y_pred is a Series from now on. You need y_pred.values  ####
                    if len(cols) == 5:
                        print('    Calculating weighted average ensemble of %d classifiers' %len(new_cols))
                        ensem_pred = np.round(subm[new_cols[-1]]*0.5+0.125*(subm[new_cols[0]]+subm[
                                        new_cols[1]]+subm[new_cols[2]]+subm[new_cols[3]])).astype(int)
                    else:
                        print('    Calculating regular average ensemble of %d classifiers' %len(new_cols))
                        ensem_pred = (subm[new_cols].mean(axis=1)).astype(int)
                    print('########################################################')
                    ##### This next step is very important since some models give series, others give arrays. Very painful!
                    if isinstance(ensem_pred,pd.Series):
                        ensem_pred = ensem_pred.values
                    print('Completed Ensemble predictions on held out data')
                    new_col = each_target+'_Ensembled_predictions'
                else:
                    stack_cols, stacksfinal = QuickML_Stacking(X, y, X_test,
                                       modeltype, Boosting_Flag,scoring_parameter,verbose)
                    new_col = each_target+'_Stacked_'+stack_cols[0].split("_")[0]+'_predictions'
                    ensem_pred = np.argmax(stacksfinal,axis=1)
                    print('########################################################')
                    print('Completed Stacked predictions on held out data')
                testm[new_col] = pd.Series(ensem_pred).map(transformer).values
                new_cols.append(new_col)
                if not isinstance(sample_submission, str):
                    try:
                        sample_submission[each_target] = pd.Series(y_pred).map(transformer).values
                    except:
                        print('Error: Sample submission file is incorrect. Please check it. continuing...')
    #####################   P L O T   F E A T U R E   I M P O R T A N C E S   H E R E ###################
    if model_label == 'Single_Label':
        if calibrator_flag:
            plot_model = model.base_estimator
        else:
            plot_model = copy.deepcopy(model)
        try:
            ####### This is where plot feature_importances_ for the models ######
            if Boosting_Flag is None:
                    if modeltype == 'Regression':
                        ### If you use LinearSVR  then use all absolute values, so you can plot them in decreasing order of importance
                        imp_features_df = pd.DataFrame(abs(plot_model.coef_),
                                            columns=['Feature Importances'],index=important_features).sort_values(
                                            'Feature Importances',ascending=False)
                    else:
                        ### If you don't use absolute values, you won't get the right set of features in order. Make sure!
                        imp_features_df = pd.DataFrame(abs(plot_model.coef_[0]),
                                            columns=['Feature Importances'],index=important_features).sort_values(
                                            'Feature Importances',ascending=False)
            else:
                if model_name.lower() == 'xgboost':
                    #####  SHAP requires this step: XGBoost models must have been "predicted"
                    _ = plot_model.predict(X_test)
                    ### It is possible that in some cases, XGBoost has fewer features than what was sent in.
                    ### In those cases, we need to identify and know which features in XGBoost are in and which are out
                    #### In that case, we need to find those features and then do a feature importance
                    dictf = plot_model.get_booster().get_score(importance_type='gain')
                    if len(left_subtract(plot_model.get_booster().feature_names,important_features)) > 0:
                        #### If feature names from XGBoost and important_features are not same,you must transform dictf like this!
                        dicta = dict(zip(plot_model.get_booster().feature_names,important_features))
                        featdict = dict([(x,dicta[x]) for x in dictf.keys()])
                        featdict2 = dict([(dicta[x],dictf[x]) for x in featdict.keys()])
                        imp_features_df =  pd.DataFrame(featdict2.values(),index=featdict2.keys(),
                                        columns = ['Feature Importances']).sort_values('Feature Importances',
                                        ascending=False)
                    else:
                        #### If the feature names from XGBoost and the important_features are same,
                        ###     you can plot dictf immediately!
                        imp_features_df =  pd.DataFrame(dictf.values(),index=dictf.keys(),
                                        columns = ['Feature Importances']).sort_values('Feature Importances',
                                        ascending=False)
                elif model_name == 'Forests':
                    imp_features_df = pd.DataFrame(plot_model.feature_importances_, columns=['Feature Importances'],
                         index=important_features).sort_values('Feature Importances',
                         ascending=False)
                elif model_name.lower() == 'catboost':
                    from catboost import Pool
                    try:
                        imp_features_df = pd.DataFrame(plot_model.get_feature_importance(
                                    Pool(X_cv, label=y_cv,cat_features=imp_cats)),
                                    columns=['Feature Importances'],
                                    index=important_features).sort_values(
                                    'Feature Importances',ascending=False)
                    except:
                        imp_features_df = pd.DataFrame(plot_model.get_feature_importance(),
                                        columns=['Feature Importances'],
                                        index=important_features).sort_values(
                                        'Feature Importances',ascending=False)
            ### Now draw the feature importances using the data frame above!
            height_size = 5
            width_size = 10
            color_string = 'byrcmgkbyrcmgkbyrcmgkbyrcmgk'
            print('    Plotting Feature Importances to explain the output of model')
            imp_features_df[:15].plot(kind='barh',title='Feature Importances for predicting %s' %each_target,
                                 figsize=(width_size, height_size), color=color_string);
        except:
            print('Could not draw feature importance plot due to an error')
        ###########   D R A W  SHAP  VALUES USING TREE BASED MODELS. THE REST WILL NOT GET SHAP ############
        if verbose >= 2:
            print('Trying to plot SHAP values if SHAP is installed in this machine...')
            X_plot = X_cv[:1000]
            y_plot = y_cv[:1000]
            try:
                if model_name.lower() == 'catboost':
                    if verbose > 0:
                        import shap
                        from catboost import Pool
                        shap.initjs()
                        try:
                            shap_values = plot_model.get_feature_importance(Pool(X_plot, label=y_plot,
                                            cat_features=imp_cats),type="ShapValues")
                        except:
                            shap_values = plot_model.get_feature_importance(type="ShapValues")
                        shap_df = pd.DataFrame(np.c_[X_plot.values,y_plot],columns=[list(X_plot)+[each_target]])
                        if modeltype == 'Multi_Classification':
                            for each_i in range(len(classes)):
                                ### This is needed for Catboost models but it is very cumbersome!
                                ### You need to cycle through multiple values of classes from 0 to n_classes-1.
                                ### There is no way to force it in an Ax => so you are stuck printing multiple charts
                                plt.figure()
                                shap.summary_plot(shap_values[:,each_i,:], shap_df, plot_type="violin")
                        else:
                            plt.figure()
                            shap.summary_plot(shap_values, shap_df, plot_type="violin")
                else:
                    X_plot = X_test[:1000]
                    import shap
                    shap.initjs()
                    #### This works well for RFC and XGBoost for multiclass problems #####
                    #### This plots a violin plot that is different from the bar chart above!
                    #### This does not work for CatBoost so try something else!
                    if model_name.lower() == 'linear':
                        explainer = shap.LinearExplainer(plot_model, X_plot, feature_dependence="independent")
                        shap_values = explainer.shap_values(X_plot)
                        plt.figure()
                        shap.summary_plot(shap_values, X_plot, plot_type="bar")
                        if modeltype != 'Regression':
                            plt.figure()
                            shap.summary_plot(shap_values, X_plot)
                    elif model_name.lower() == 'forests':
                        #### This works well for RFC and XGBoost for multiclass problems #####
                        ### It works for both binary and multi-class problems ########
                        ### However, it does NOT work for CatBoost models!
                        explainer = shap.TreeExplainer(plot_model)
                        shap_values = explainer.shap_values(X_plot)
                        plt.figure()
                        shap.summary_plot(shap_values, X_plot, plot_type="bar")
                        ### There is no single violin plot for Random Forests in SHAP
                        #### It actually has multiple outputs so you can loop through it for each class
                        if modeltype != 'Regression':
                            for each_i in range(len(classes)):
                                plt.figure()
                                shap.summary_plot(shap_values[each_i], X_plot)
                    elif model_name.lower() == 'xgboost':
                        #### This works well for RFC and XGBoost for multiclass problems #####
                        ### It works for both binary and multi-class problems ########
                        ### However, it does NOT work for CatBoost models!
                        explainer = shap.TreeExplainer(plot_model)
                        shap_values = explainer.shap_values(X_plot)
                        plt.figure()
                        shap.summary_plot(shap_values, X_plot, plot_type="bar")
                        if modeltype != 'Regression':
                            plt.figure()
                            shap.summary_plot(shap_values, X_plot)
            except:
                print('Could not plot SHAP values since SHAP is not installed or could not import SHAP in this machine')
    else:
        ############ No Plotting feature importances for Multi-Label problems ##############
        print('    No feature importance plots available for Multi-Label problems')
    print('############### P R E D I C T I O N  O N  T E S T   C O M P L E T E D  #################')
    print('    Time taken thus far (in seconds) = %0.0f' %(time.time()-start_time))
    ##  Write the test and submission files to disk ###
    print('Writing Output files to disk...')
    #############################################################################################
    if not isinstance(testm, str):
        try:
            write_file_to_folder(testm, each_target, each_target+'_'+modeltype+'_'+'test_modified.csv')
            #####   D R A W   K D E  P L O T S   FOR PROBABILITY OF PREDICTIONS - very useful! #########
            if modeltype != 'Regression':
                if verbose >= 2:
                    testm[proba_cols].plot(kind='kde',figsize=(10,6),
                                    title='Predictive Probability Density Chart with suggested threshold in red')
                    plt.axvline(x=m_thresh, color='r', linestyle='--');
        except:
            print('    Error: Not able to save test modified file. Skipping...')
    #############################################################################################
    if isinstance(sample_submission, str):
        try:
            if len(id_cols) > 0:
                sample_submission = testm[id_cols]
            else:
                sample_submission = pd.DataFrame(index=testm.index,columns=[])
            if len(target) == 1:
                    sample_submission[each_target] = testm[each_target+'_predictions']
            else:
                predict_cols = [x+'_predictions' for x in target]
                for predict_col, target_col in zip(predict_cols, target):
                    sample_submission[target_col] = testm[predict_col]
        except:
            print('Error: Could not create sample submission file from Test data')
    try:
        write_file_to_folder(sample_submission, each_target, each_target+'_'+modeltype+'_'+'submission.csv')
    except:
        print('    Error: Not able to save submission file. Skipping...')
    #############################################################################################
    try:
        #### Bring trainm back to its original index ###################
        orig_index = orig_train.index
        trainm = train.reindex(index = orig_index)
        write_file_to_folder(trainm, each_target, each_target+'_'+modeltype+'_'+'train_modified.csv')
    except:
        print('    Error: Not able to save train modified file. Skipping...')
    ### In case of multi-label models, we will reset the start train and test dataframes to contain new features created
    start_train = start_train[target].join(start_train[orig_red_preds])
    if not isinstance(orig_test, str):
        start_test = start_test[orig_red_preds]
    #### Once each target cycle is over, reset the red_preds to the orig_red_preds so we can start over
    #red_preds = copy.deepcopy(orig_red_preds)
    if model_label != 'Single_Label':
        #### Perform Final Multi-Label Operations here since all Labels are finished by now ###
        #### Don't change the target here to each_target since this is for multi-label situations only ###
        if (scoring_parameter == 'basket_recall' or scoring_parameter == 'jaccard') and modeltype != 'Regression':
            y_preds = np.array(list(zipped))
            _,_,_,y_actuals = train_test_split(train[red_preds], train[target].values,
                                        test_size=test_size, random_state=seed)
            print('Shape of Actuals: %s and Preds: %s' %(y_actuals.shape[0], y_preds.shape[0]))
            if y_actuals.shape[0] == y_preds.shape[0]:
                if scoring_parameter == 'basket_recall' and len(target) > 1:
                    accu_all = basket_recall(y_actuals, y_preds).mean()
                    print('        Mean Basket Recall = {:,.1f}%'.format(
                                                                    accu_all*100))
                elif scoring_parameter == 'jaccard' and len(target) > 1:
                ##  This shows similarity in multi-label situations ####
                    accu_all = jaccard_multilabel(y_actuals, y_preds)
                    print('        Mean Jaccard Similarity  = %s' %(
                                            accu_all))
        ##   END OF SHOWING METRICS LABEL IN A MULTI LABEL DATA SET ! WHEW ! ###################
    print('###############  C O M P L E T E D  ################')
    print('Time Taken in mins = %0.1f for the Entire Process' %((time.time()-start_time)/60))
    #return model, imp_features_df.index.tolist(), trainm, testm
    return model, important_features, trainm, testm
###############################################################################
def plot_SHAP_values(m,X,modeltype,Boosting_Flag=False,matplotlib_flag=False,verbose=0):
    import shap
    # load JS visualization code to notebook
    if not matplotlib_flag:
        shap.initjs();
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X)
    if not Boosting_Flag is None:
        if Boosting_Flag:
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            if verbose > 0 and modeltype != 'Multi_Classification':
                shap.summary_plot(shap_values, X, plot_type="violin");
        if verbose >= 1:
            shap.summary_plot(shap_values, X, plot_type="bar");
    else:
        shap.summary_plot(shap_values, X, plot_type="bar");
################################################################################
################      Find top features using XGB     ###################
################################################################################
from xgboost import XGBClassifier, XGBRegressor
def find_top_features_xgb(train,preds,numvars,target,modeltype,corr_limit,verbose=0):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    """
    ######################   I M P O R T A N T ##############################################
    ###### This top_num decides how many top_n features XGB selects in each iteration.
    ####  There a total of 5 iterations. Hence 5x10 means maximum 50 featues will be selected.
    #####  If there are more than 50 variables, then maximum 5*25 = 125 variables will be selected
    if len(preds) <= 50:
        top_num = 10
    else:
        top_num = 25
    ######################   I M P O R T A N T ##############################################
    #### If there are more than 30 categorical variables in a data set, it is worth reducing features.
    ####  Otherwise. XGBoost is pretty good at finding the best features whether cat or numeric !
    n_splits = 5
    max_depth = 8
    max_cats = 5
    ######################   I M P O R T A N T ##############################################
    train = copy.deepcopy(train)
    preds = copy.deepcopy(preds)
    numvars = copy.deepcopy(numvars)
    subsample =  0.7
    col_sub_sample = 0.7
    train = copy.deepcopy(train)
    start_time = time.time()
    test_size = 0.2
    seed = 1
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits)
    rem_vars = left_subtract(preds,numvars)
    catvars = copy.deepcopy(rem_vars)
    ############   I  M P O R T A N T ! I M P O R T A N T ! ######################
    ##### I am removing the Cat Vars selection using Linear Methods since they fail so often.
    ##### Linear methods such as Chi2 or Mutual Information Score are not great
    ####  for feature selection since they can't handle large data and provide
    ####  inconsistent results for large data sets. Hence I am using XGBoost alone.
    ####  Also, another method of using Spearman Correlation for CatVars with 100's
    ####  of variables is extremely slow. Also, it's effectivenes is not very clear.
    ####  Hence for now, I am not going to use Spearman method.
    ##############################################################################
    #if len(catvars) > max_cats:
    #    start_time = time.time()
    #    important_cats = remove_variables_using_fast_correlation(train,catvars,modeltype,
    #                         corr_limit,verbose)
    #    if verbose >= 1:
    #        print('Time taken for reducing highly correlated Categorical vars was %0.0f seconds' %(time.time()-start_time))
    #else:
    important_cats = copy.deepcopy(catvars)
    print('    (%d) categorical vars...' %(len(catvars)))
    final_list = copy.deepcopy(numvars)
    if  isinstance(final_list,np.ndarray):
        final_list = final_list.tolist()
    preds = final_list+important_cats
    #######You must convert category variables into integers ###############
    for important_cat in important_cats:
        if str(train[important_cat].dtype) == 'category':
            train[important_cat] = train[important_cat].astype(int)
    ########    Drop Missing value rows since XGB for some reason  #########
    ########    can't handle missing values in early stopping rounds #######
    train.dropna(axis=0,subset=preds+[target],inplace=True)
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    print('############## F E A T U R E   S E L E C T I O N    BY   X G B O O S T    ####################')
    important_features = []
    if modeltype == 'Regression':
        objective = 'reg:squarederror'
        model_xgb = XGBRegressor( n_estimators=100,subsample=subsample,objective=objective,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                 seed=1,random_state=1,n_jobs=-1, nthread=-1)
        eval_metric = 'rmse'
    else:
        #### This is for Classifiers only
        classes = np.unique(train[target].values)
        if len(classes) == 2:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=200,
                n_jobs=-1, nthread=-1, objective='binary:logistic',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                seed=1)
            eval_metric = 'logloss'
        else:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=max_depth, min_child_weight=1, missing=-999, n_estimators=200,
                n_jobs=-1, nthread=-1, objective='multi:softmax',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5,
                seed=1)
            eval_metric = 'mlogloss'
    ####   This is where you start to Iterate on Finding Important Features ################
    save_xgb = copy.deepcopy(model_xgb)
    train_p = train[preds]
    if train_p.shape[1] < 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    print('Current number of predictors = %d ' %(train_p.shape[1],))
    print('    Finding Important Features using Boosted Trees algorithm...')
    try:
        for i in range(0,train_p.shape[1],iter_limit):
            new_xgb = copy.deepcopy(save_xgb)
            print('        using %d variables...' %(train_p.shape[1]-i))
            if train_p.shape[1]-i < iter_limit:
                X = train_p.iloc[:,i:]
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                try:
                    eval_set = [(X_train,y_train),(X_cv,y_cv)]
                    model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                        eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                except:
                    new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=eval_set,
                                        eval_metric=eval_metric,verbose=False)
                    print('XGB has a bug in version xgboost 1.02 for feature importances. Try to install version 0.90 or 1.10 - continuing...')
                    important_features += pd.Series(new_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                important_features = list(OrderedDict.fromkeys(important_features))
            else:
                X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
                #### Split here into train and test #####
                if modeltype == 'Regression':
                    train_part = int((1-test_size)*X.shape[0])
                    X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
                else:
                    X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                                test_size=test_size, random_state=seed)
                eval_set = [(X_train,y_train),(X_cv,y_cv)]
                try:
                    model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                except:
                    new_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                                  eval_set=eval_set,eval_metric=eval_metric,verbose=False)
                    important_features += pd.Series(model_xgb.get_booster().get_score(
                                importance_type='gain')).sort_values(ascending=False)[:top_num].index.tolist()
                important_features = list(OrderedDict.fromkeys(important_features))
    except:
        print('Finding top features using XGB is crashing. Continuing with all predictors...')
        important_features = copy.deepcopy(preds)
        return important_features, [], []
    important_features = list(OrderedDict.fromkeys(important_features))
    print('Found %d important features' %len(important_features))
    #print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    numvars = [x for x in numvars if x in important_features]
    important_cats = [x for x in important_cats if x in important_features]
    return important_features, numvars, important_cats
################################################################################
def basket_recall(label, pred):
    """
    This tests the recall of a given basket of items in a label by the second basket, pred.
    It compares the 2 baskets (arrays or lists) named as label and pred, and finds common items
    between the two. Then it divides that length by the total number of items in the label basket
    to come up with a basket recall score. This score may be useful in recommendation problems
    where you are interested in finding how many items in a basket (labels) that your
    predictions (pred) basket got correct. The order of the items in the baskets does not matter.
    """
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(pred, list):
        pred = np.array(pred)
    if len(label) > 1:
        jacc_arr = []
        for row1,row2,count in zip(label,pred, range(len(label))):
            intersection = len(np.intersect1d(row1,row2))
            union = len(row1)
            jacc = float(intersection / union)
            if count == 0:
                jacc_arr = copy.deepcopy(jacc)
            else:
                jacc_arr = np.r_[jacc_arr,jacc]
        return jacc_arr
    else:
        intersection = len(list(set(list1).intersection(set(list2))))
        union = (len(list1) + len(list2)) - intersection
        jacc_arr = float(intersection / union)
    return jacc_arr
################################################################################
def jaccard_singlelabel(label, pred):
    """
    This compares 2 baskets (could be lists or arrays): label and pred, and finds common items
    between the two. Then it divides that number by either rows or columns to return %.
        ### Jaccard_Columnwise = this means you have multi-labels and you want it summed columnwise
        ###   This tells you what is the average accuracy for each column in multi-label target
        ###   It will return as many totals as the number of columns in your multi-label target.
        ###   To get a percentage, you will have to divide it by the number of rows in the data set.
        ###   This percentage gives you the % of rows in each label you got correctly=%Each_Label Accuracy
        ###   This will give you as many percentages as there are labels in your multi-label target.
        ### Jaccard_Row-wise = this means you have combos but where order matters and you want it compared row-wise
        ###   This tells you how many labels in each row did you get right. THat's accuracy by row.
        ###   It will return as many totals as the number of rows in your data set.
        ###   To get a percentage, you will have to divide it by the number of labels in the data set.
        ###   This percentage gives you the % of labels in each row you got correctly=%Combined_Label_Accuracy
        ###   This will give you a single percentage number for the whole data set
    """
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(pred, list):
        pred = np.array(pred)
    try:
        ### This is for Multi-Label Problems ##### Returns 2 results: one number and
        ###    the second is an array with as many items as number of labels in target
        jacc_each_label = np.sum(label==pred,axis=0)/label.shape[0]
        return  jacc_each_label
    except:
        return 0
################################################################################
def jaccard_multilabel(label, pred):
    """
    This compares 2 baskets (could be lists or arrays): label and pred, and finds common items
    between the two. Then it divides that number by either rows or columns to return %.
        ### Jaccard_Columnwise = this means you have multi-labels and you want it summed columnwise
        ###   This tells you what is the average accuracy for each column in multi-label target
        ###   It will return as many totals as the number of columns in your multi-label target.
        ###   To get a percentage, you will have to divide it by the number of rows in the data set.
        ###   This percentage gives you the % of rows in each label you got correctly=%Each_Label Accuracy
        ###   This will give you as many percentages as there are labels in your multi-label target.
        ### Jaccard_Row-wise = this means you have combos but where order matters and you want it compared row-wise
        ###   This tells you how many labels in each row did you get right. THat's accuracy by row.
        ###   It will return as many totals as the number of rows in your data set.
        ###   To get a percentage, you will have to divide it by the number of labels in the data set.
        ###   This percentage gives you the % of labels in each row you got correctly=%Combined_Label_Accuracy
        ###   This will give you a single percentage number for the whole data set
    """
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(pred, list):
        pred = np.array(pred)
    ### This is for Multi-Label Problems ##### Returns 2 results: one number and
    ###    the second is an array with as many items as number of labels in target
    try:
        jacc_data_set = np.sum(label==pred,axis=1).sum()/label.shape[1]
        return jacc_data_set
    except:
        return 0
################################################################################
def plot_RS_params(cv_results, score, mname):
    """
    ####### This plots the GridSearchCV Results sent in ############
    """
    df = pd.DataFrame(cv_results)
    params = [x for x in list(df) if x.startswith('param_')]
    traincols = ['mean_train_score' ]
    testcols = ['mean_test_score' ]
    cols = traincols+testcols
    ncols = 2
    noplots = len(params)
    if noplots%ncols == 0:
        rows = noplots/ncols
    else:
        rows = (noplots/ncols)+1
    height_size = 5
    width_size = 15
    fig = plt.figure(figsize=(width_size,rows*height_size))
    fig.suptitle('Training and Validation: Hyper Parameter Tuning for target=%s' %mname, fontsize=20,y=1.01)
    #### If the values are negative, convert them to positive ############
    if len(df.loc[df[cols[0]]<0]) > 0:
        df[cols] = df[cols]*-1
    for each_param, count in zip(params, range(noplots)):
        plt.subplot(rows,ncols,count+1)
        ax1 = plt.gca()
        if df[each_param].dtype != object:
            df[[each_param]+cols].groupby(each_param).mean().plot(kind='line',
                            title='%s for %s' %(each_param,mname),ax=ax1)
        else:
            try:
                df[each_param] = pd.to_numeric(df[each_param])
                df[[each_param]+cols].groupby(each_param).mean().plot(kind='line',
                            title='%s for %s' %(each_param,mname), ax=ax1)
            except:
                df[[each_param]+cols].groupby(each_param).mean().plot(kind='bar',stacked=False,
                            title='%s for %s' %(each_param,mname), ax=ax1)
    #### This is to plot the test_mean_score against params to see how it increases
    for each_param in params:
        #### This is to find which parameters are non string and convert them to strings
        if df[each_param].dtype!=object:
              df[each_param] = df[each_param].astype(str)
    try:
        df['combined_parameters'] = df[params].apply(lambda x: '__'.join(x), axis=1 )
    except:
        df['combined_parameters'] = df[params].apply(lambda x: '__'.join(x.map(str)), axis=1 )
    if len(params) == 1:
        df['combined_parameters'] = copy.deepcopy(df[params])
    else:
        df[['combined_parameters']+cols].groupby('combined_parameters').mean().sort_values(
            cols[1]).plot(figsize=(width_size,height_size),kind='line',subplots=False,
                            title='Combined Parameters: %s scores for %s' %(score,mname))
    plt.xticks(rotation=45)
    plt.show();
    return df
################################################################################
def plot_xgb_metrics(model,eval_metric,eval_set,modeltype,model_label='',model_name=""):
    height_size = 5
    width_size = 10
    if model_name.lower() == 'catboost':
        results = model.get_evals_result()
    else:
        results = model.evals_result()
    res_keys = list(results.keys())
    eval_metric = list(results[res_keys[0]].keys())
    if isinstance(eval_metric, list):
        # plot log loss
        eval_metric = eval_metric[0]
    # plot metrics now
    fig, ax = plt.subplots(figsize=(width_size, height_size))
    epochs = len(results[res_keys[0]][eval_metric])
    x_axis = range(0, epochs)
    if model_name.lower() == 'catboost':
        ax.plot(x_axis, results[res_keys[0]][eval_metric], label='%s' %res_keys[0])
    else:
        ax.plot(x_axis, results[res_keys[0]][eval_metric], label='%s' %res_keys[0])
    epochs = len(results[res_keys[-1]][eval_metric])
    x_axis = range(0, epochs)
    ax.plot(x_axis, results[res_keys[-1]][eval_metric], label='%s' %res_keys[-1])
    ax.legend()
    plt.ylabel(eval_metric)
    plt.title('%s Train and Validation Metrics across Epochs (Early Stopping in effect)' %model_label)
    plt.show();
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
def classify_columns(df_preds, verbose=0):
    """
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    max_nlp_char_size = 30
    max_cols_to_print = 30
    print('############## C L A S S I F Y I N G  V A R I A B L E S  ####################')
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 15
    float_limit = 8 #### Make this limit low so that float variables below this limit become cat vars ###
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                   ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete
    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    ##### This is complicated logic - do not touch #############################
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) >= cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) >= cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            elif len(train[col].value_counts()) < cat_limit and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if train[col].map(lambda x: len(x) if type(x)==str else 0).mean() >= max_nlp_char_size:
                    var_df.loc[var_df['index']==col,'nlp_strings'] = 1
                else:
                    var_df.loc[var_df['index']==col,'discrete_strings'] = 1
    ##### This is complicated logic - do not touch #############################
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    var_df['date_time'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    ########  V E R Y    I M P O R T A N T   ###################################################
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose == 2:
        marthas_columns(df_preds,verbose=1)
        print("    Numeric Columns: %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])
    ##### now collect all the column types and column names into a single dictionary to return!
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
#################################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        if verbose==1:
            print('Data Set columns info:')
            for col in data.columns:
                print('* %s: %d nulls, %d unique vals, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')
##################################################################################
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import pdb
def print_classification_metrics(y_test, y_probs, proba_flag=True):
    """
    #######  Send in the actual_values and prediction_probabilities for binary classes
    This will return back metrics and print them all in a neat format
    """
    y_test = copy.deepcopy(y_test)
    multi_label_flag = False
    multi_class_flag = False
    try:
        if y_test.shape[1] > 0:
            multi_label_flag = True
    except:
        pass
    ######## This is where you start printing metrics ###############
    if len(np.unique(y_test)) > 2:
        multi_class_flag = True
        print('Multi Class Model Metrics Report')
        print('#####################################################')
    else:
        print('Binary Class Model Metrics Report')
        print('#####################################################')
    #### for some cases, you won't get proba, so check the proba_flag
    if proba_flag:
        if multi_label_flag:
            y_preds = (y_probs>0.5).astype(int)
        else:
            y_preds = y_probs.argmax(axis=1)
    else:
        y_preds = copy.deepcopy(y_probs)
    if not multi_class_flag  and not multi_label_flag:
        # Calculate comparison metrics for Binary classification results.
        accuracy = metrics.accuracy_score(y_test, y_preds)
        balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_preds)
        precision = metrics.precision_score(y_test, y_preds)
        f1_score = metrics.f1_score(y_test, y_preds)
        recall = metrics.recall_score(y_test, y_preds)
        if type(np.mean((y_test==y_preds))) == pd.Series:
            print('    Accuracy          = %0.1f%%' %(np.mean(accuracy)*100))
        else:
            print('    Accuracy          = %0.1f%%' %(accuracy*100))
        print('    Balanced Accuracy = %0.1f%%' %(balanced_accuracy*100))
        print('    Precision         = %0.1f%%' %(precision*100))
        if proba_flag:
            average_precision = np.mean(metrics.precision_score(y_test, y_preds, average=None))
        else:
            average_precision = metrics.precision_score(y_test, y_preds, average='macro')
        print('    Average Precision = %0.1f%%' %(average_precision*100))
        print('    Recall            = %0.1f%%' %(recall*100))
        print('    F1 Score          = %0.1f%%' %(f1_score*100))
        if proba_flag:
            fpr, tpr, threshold = metrics.roc_curve(y_test, y_probs[:,1])
            roc_auc = metrics.auc(fpr, tpr)
            print('    ROC AUC           = %0.1f%%' %(roc_auc*100))
        else:
            roc_auc = 0
        print('#####################################################')
        return [accuracy, balanced_accuracy, precision, average_precision, f1_score, recall, roc_auc]
    else:
        # Calculate comparison metrics for Multi-Class classification results.
        accuracy = np.mean((y_test==y_preds))
        if multi_label_flag:
            balanced_accuracy = np.mean(metrics.recall_score(y_test, y_preds, average=None))
            precision = metrics.precision_score(y_test, y_preds, average=None)
            average_precision = metrics.precision_score(y_test, y_preds, average='macro')
            f1_score = metrics.f1_score(y_test, y_preds, average=None)
            recall = metrics.recall_score(y_test, y_preds, average=None)
        else:
            balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_preds)
            precision = metrics.precision_score(y_test, y_preds, average = None)
            average_precision = metrics.precision_score(y_test, y_preds,average='macro')
            f1_score = metrics.f1_score(y_test, y_preds, average = None)
            recall = metrics.recall_score(y_test, y_preds, average = None)
        if type(np.mean((y_test==y_preds))) == pd.Series:
            print('    Accuracy          = %0.1f%%' %(np.mean(accuracy)*100))
        else:
            print('    Accuracy          = %0.1f%%' %(accuracy*100))
        print('    Balanced Accuracy (average recall) = %0.1f%%' %(balanced_accuracy*100))
        print('    Average Precision (macro) = %0.1f%%' %(average_precision*100))
        ### these are basically one for each class #####
        print('    Precisions by class:')
        for precisions in precision:
            print('    %0.1f%%  ' %(precisions*100),end="")
        print('\n    Recall Scores by class:')
        for recalls in recall:
            print('    %0.1f%%  ' %(recalls*100), end="")
        print('\n    F1 Scores by class:')
        for f1_scores in f1_score:
            print('    %0.1f%%  ' %(f1_scores*100),end="")
        # Return list of metrics to be added to a Dataframe to compare models.
        print('\n#####################################################')
        return [accuracy, balanced_accuracy, precision, average_precision, f1_score, recall, 0]
##################################################################################################
##################################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
from collections import Counter
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
################################################################################
from collections import defaultdict
from collections import OrderedDict
import time
def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict
##################################################################################
def remove_variables_using_fast_correlation(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0):
    """
    #### THIS METHOD IS KNOWN AS THE SULA METHOD in HONOR OF my mother SULOCHANA SESHADRI #######
    S U L A : Simple Uncorrelated Linear Algorithm will get a set of uncorrelated vars easily.
    SULA is a highly efficient method that removes variables that are highly correlated but
    breaks the logjam that occurs when deciding which of the 2 correlated variables to remove.
    SULA uses the MIS (mutual_info_score) to decide to keep the higher scored variable.
    This method enables an extremely fast and highly effective method that keeps the variables
    you want while discarding lesser highr correlated variables in less than a minute, even on a laptop.
    You need to send in a dataframe with a list of numeric variables and define a threshold - that's all.
    This threshold will define the "high Correlation" mark - you can set it as anything over + or - 0.70.
    You can change this limit. If two variables have absolute correlation higher than limit, they
    will be red-lined, and using a series of knockout rounds, one of them will get knocked out:
    To decide which variables to keep in knockout rounds, mutual information score is used.
    MIS returns a ranked list of correlated variables: when we select one to keep, we knock out those
    that were correlated to it. This way we knock out correlated variables from each round.
    In the end, SULA gives us the least correlated variables that have the best mutual information score!
    ##############  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY #####
    """
    df = copy.deepcopy(df)
    print('Removing highly correlated variables using SULA method among (%d) numeric variables' %len(numvars))
    correlation_dataframe = df[numvars].corr().astype(np.float16)
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = [k for (k,v) in corr_pair_count_dict]
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('    No numeric vars removed since none have high correlation with each other in this data...')
    else:
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        else:
            sel_function = mutual_info_classif
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        fs.fit(df[corr_list].astype(np.float16), df[target])
        mutual_info = dict(zip(corr_list,fs.scores_))
        #### The first variable in list has the highest correlation to the target variable ###
        sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
        #####   Now we select the final list of correlated variables ###########
        selected_corr_list = []
        #### select each variable by the highest mutual info and see what vars are correlated to it
        for each_corr_name in sorted_by_mutual_info:
            ### add the selected var to the selected_corr_list
            selected_corr_list.append(each_corr_name)
            for each_remove in corr_pair_dict[each_corr_name]:
                #### Now remove each variable that is highly correlated to the selected variable
                if each_remove in sorted_by_mutual_info:
                    sorted_by_mutual_info.remove(each_remove)
        ##### Now we combine the uncorrelated list to the selected correlated list above
        rem_col_list = left_subtract(list(correlation_dataframe),corr_list)
        final_list = rem_col_list + selected_corr_list
        removed_cols = left_subtract(numvars, final_list)
        if len(removed_cols) > 0:
            print('    Removing (%d) highly correlated variables:' %(len(removed_cols)))
            if len(removed_cols) <= 30:
                print('    %s' %removed_cols)
            if len(final_list) <= 30:
                print('    Following (%d) vars selected: %s' %(len(final_list),final_list))
    return final_list
###############################################################################################
def count_freq_in_list(lst):
    """
    This counts the frequency of items in a list but MAINTAINS the order of appearance of items.
    This order is very important when you are doing certain functions. Hence this function!
    """
    temp=np.unique(lst)
    result = []
    for i in temp:
        result.append((i,lst.count(i)))
    return result

def find_corr_vars(correlation_dataframe,corr_limit = 0.70):
    """
    This returns a dictionary of counts of each variable and how many vars it is correlated to in the dataframe
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_items = lambda dic: [x for x in dic.items()]
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    correlated_pair_dict = dict(correlated_pair)
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    #corr_pair_count_dict = Counter(flat_corr_pair_list)
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = list(set(flatten(flatten_items(correlated_pair_dict))))
    rem_col_list = left_subtract(list(correlation_dataframe),list(OrderedDict.fromkeys(flat_corr_pair_list)))
    return corr_pair_count_dict, rem_col_list, corr_list, correlated_pair_dict
################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

def add_poly_vars_select(data,numvars,targetvar,modeltype,poly_degree=2,Add_Poly=2,md='',
                            corr_limit=0.70,scaling=True, fit_flag=False, verbose=0):
    """
    #### This adds Polynomial and Interaction Variables of any Size to a data set and returns the best vars
    among those poly and interaction variables. Notice you will get a list of variables as well as the modified
    data set with the old and new (added) variables. Very Convenient when you want to do quick testing.
    There are 3 settings for Add_Poly flag: 0 => No poly or intxn variables added. 1=> Only intxn vars
    added. 2=> only polynomial degree (squared) vars added. 3=> both squared and intxn vars added.
    If Fit_Flag=True, then it is assumed that it is Training data and hence variables are selected
    If Fit_Flag=False, then it is assumed it is Test data and no variables are chosen but we keep training ones.
    """
    data = copy.deepcopy(data)
    numvars = copy.deepcopy(numvars)
    tolerance = 0.01
    orig_data_index = data.index
    if modeltype == 'Regression':
        lm = Lasso(alpha=0.001, max_iter=2000,
                 fit_intercept=True, normalize=False)
    else:
        if modeltype == 'Binary_Classification':
            lm = LogisticRegression(C=0.01,fit_intercept=True,tol=tolerance,
                                max_iter=2000,solver='liblinear',n_jobs=-1,
                              penalty='l2',dual=False, random_state=0)
        else:
            lm = LogisticRegression(multi_class='ovr', max_iter=10000) ## logistic seems best among all
    predictors = copy.deepcopy(numvars)
    #### number of original features in data set ####
    n_orig_features = len(predictors)
    selected = []
    X_data = data[predictors]
    #######  Initial Model with all input variables ######
    if scaling == 'Standard':
        XS = StandardScaler().fit_transform(X_data)
    elif scaling == 'MinMax':
        XS = MinMaxScaler().fit_transform(X_data)
    ###  or you can use Centering which simply subtracts the Mean:
    elif scaling == 'Centering':
        XS = (X_data-X_data.mean())
    else:
        XS = copy.deepcopy(X_data)
    #XS.columns=predictors
    X = copy.deepcopy(XS)
    ################## Fit the model after scaling data ###############
    if fit_flag:
        Y = data[targetvar]
        print('Building Inital Model with given variables...')
        print_model_metrics(modeltype,lm,X,Y,False)
    ######## Here is where the Interaction variable selection begins ############
    #print('Adding Polynomial %s-degree and Interaction variables...' %poly_degree)
    if Add_Poly == 1:
        ### If it is 1, add only Interaction variables.
        poly = PolynomialFeatures(degree=poly_degree, include_bias = False, interaction_only=True)
    elif Add_Poly == 2:
        #### If it is 2 or 3 add both Squared and Interaction variables. We will remove interaction
        ###       variables later in this program. For now include both!
        poly = PolynomialFeatures(degree=poly_degree, include_bias = False, interaction_only=False)
    elif Add_Poly == 3:
        #### If it is 2 or 3 add both Squared and Interaction variables. We will remove interaction
        ###       variables later in this program. For now include both!
        poly = PolynomialFeatures(degree=poly_degree, include_bias = False, interaction_only=False)
    if fit_flag:
        md = poly.fit(X) #### This defines the Polynomial feature extraction
    try:
        XP = md.transform(X) #### This transforms X into a Polynomial Order
    except MemoryError:
        return predictors, '', X, md, [], dict()
    #################################################################################
    #####   CONVERT X-VARIABLES FROM POLY AND INTERACTION INTO ORIGINAL VARIABLES ###
    #################################################################################
    xnames = md.get_feature_names() ### xnames contains all x-only, Poly and Intxn variables in x-format
    if len(xnames) > 300:
        max_iter = 5000
    else:
        max_iter = 2000
    ###### BUILDING COMPARISON MODEL WITH INTERACTION VARIABLES ########################
    start_time = time.time()
    if modeltype == 'Regression':
        lm = Lasso(alpha=0.001, max_iter=max_iter,
                 fit_intercept=True, normalize=False)
    else:
        if modeltype == 'Binary_Classification':
            lm = LogisticRegression(C=0.01,fit_intercept=True, tol=tolerance,
                            max_iter=max_iter,solver='liblinear',n_jobs=-1,
                          penalty='l2',dual=False, random_state=0)
        else:
            lm = LogisticRegression(multi_class='ovr', max_iter=10000) ## logistic seems best among all
    ########### Here starts the conversion of X variables into Text feature variable names #####################
    XP1 = pd.DataFrame(XP,index=orig_data_index, columns=xnames) ## XP1 has all the Xvars:incl orig+poly+intxn vars
    x_vars = xnames[:n_orig_features]  ### x_vars contain x_variables such as 'x1'
    #### Feature_xvar_dict will map the X_vars, Squared_vars and Intxn_vars back to  Text vars in one variable
    feature_xvar_dict = dict(zip(x_vars,predictors)) ###
    if fit_flag:
        #### If there is fitting to be done, then you must do this ###
        if Add_Poly == 1: #### This adds only Interaction variables => no Polynomials!
            sq_vars = []    ### sq_vars contain only x-squared variables such as 'x^2'
            intxn_vars = left_subtract(xnames, sq_vars+x_vars) ### intxn_vars contain interaction vars such as 'x1 x2'
        elif Add_Poly == 2: #### This adds only Polynomial variables => no Interactions!
            sq_vars = [x for x in xnames if '^2' in x]    ### sq_vars contain only x-squared variables such as 'x^2'
            intxn_vars = [] ### intxn_vars contain interaction vars such as 'x1 x2'
        elif Add_Poly == 3: #### This adds Both Interaction and Polynomial variables => Best of Both worlds!
            sq_vars = [x for x in xnames if '^2' in x]    ### sq_vars contain only x-squared variables such as 'x^2'
            intxn_vars = left_subtract(xnames, sq_vars+x_vars) ### intxn_vars contain interaction vars such as 'x1 x2'
        ####  It is now time to cut down the original x_variables to just squared variables and originals here ####
        dict_vars = dict(zip(predictors,x_vars)) ### this is a dictionary mapping original variables and their x-variables
        reverse_dict_vars = dict([(y,x) for (x,y) in dict_vars.items()]) ### this maps the x-vars to original vars
        ##### Now let's convert Interaction x_variables into their corresponding text variables
        intxn_text_vars = []
        for each_item in intxn_vars:
            if len(each_item.split(" ")) == 1:
                intxn_text_vars.append(reverse_dict_vars[each_item])
                feature_xvar_dict[each_item] = reverse_dict_vars[each_item]
            elif len(each_item.split(" ")) == 2:
                two_items_list = each_item.split(" ")
                full_intxn_name = reverse_dict_vars[two_items_list[0]] +" "+ reverse_dict_vars[two_items_list[1]]
                intxn_text_vars.append(full_intxn_name)
                feature_xvar_dict[each_item] = full_intxn_name
            else:
                pass
        ##### Now let's convert Squared x_variables into their corresponding text variables
        sq_text_vars = []
        for each_sq_item in sq_vars:
            if len(each_sq_item.split("^")) == 2:
                two_item_list = each_sq_item.split("^")
                full_sq_name = reverse_dict_vars[two_item_list[0]] +"^2"
                sq_text_vars.append(full_sq_name)
                feature_xvar_dict[each_sq_item] = full_sq_name
            else:
                pass
        #### Now we need to combine the x_vars, Squared_vars and the Intxn_vars together as Text vars in one variable
        full_x_vars = x_vars + sq_vars + intxn_vars
        text_vars = predictors + sq_text_vars + intxn_text_vars #### text_vars now contains all the text version of x-variables
        if len(text_vars) == len(full_x_vars):
            print('Successfully transformed x-variables into text-variables after Polynomial transformations')
        else:
            print('Error: Not able to transform x-variables into text-variables. Continuing without Poly vars...')
            return predictors, lm, XP1, md, x_vars,feature_xvar_dict
        feature_textvar_dict  = dict([(y,x) for (x,y) in feature_xvar_dict.items()])
        #### Now Build a Data Frame containing containing additional Poly and Intxn variables in x-format here ####
        new_addx_vars = sq_vars+intxn_vars
        if len(new_addx_vars) == 0:
            print('Error: There are no squared or interaction vars to add. Continuing without Poly vars...')
            return predictors, lm, XP1, md, x_vars,feature_xvar_dict
        ####  We define 2 data frames: one for removing highly correlated vars and other for Lasso selection
        XP2 = XP1[new_addx_vars].join(Y)
        XP1X = XP1[new_addx_vars]
        new_addtext_vars = [feature_xvar_dict[x] for x in new_addx_vars]
        XP2.columns = new_addtext_vars+[targetvar]
        XP1X.columns = new_addtext_vars
        ###################################################################################
        ####    FAST FEATURE REDUCTION USING L1 REGULARIZATION FOR LARGE DATA SETS   ######
        ####   Use LassoCV or LogisticRegressionCV to Reduce Variables using Regularization
        ###################################################################################
        print('Building Comparison Model with only Poly and Interaction variables...')
        lm_p, _ = print_model_metrics(modeltype,lm,XP1X,Y,True)
        print('    Time Taken: %0.0f (in seconds)' %(time.time()-start_time))
        ####  We need to build a dataframe to hold coefficients from the model for each variable ###
        dfx = pd.DataFrame([new_addx_vars, new_addtext_vars])
        df = dfx.T
        df.columns=['X Names','Interaction Variable Names']
        ##########  N O W   B U I L D  T H E  M O D E L ##################################
        if modeltype == 'Regression':
            df['Coefficient Values'] = lm_p.coef_
        else:
            df['Coefficient Values'] = lm_p.coef_[0]
        #### This part selects the top 10% of interaction variables created using linear modeling.
        ###   It is very important to leave the next line as is since > symbol is better than >=
        lim = abs(df['Coefficient Values']).quantile(0.9)
        df90 = df[abs(df['Coefficient Values'])>lim].sort_values('Coefficient Values',ascending=False)
        if df['Coefficient Values'].sum()==0.0 or df90['Coefficient Values'].sum()==0.0:
            ### There is no coefficient that is greater than 0 here. So reject all variables!
            sel_x_vars = []
            print('Zero Interaction and Polynomial variable(s) selected...')
        elif df['Coefficient Values'].shape[0] == df90['Coefficient Values'].shape[0]:
            ### There is no coefficient that is greater than 0 here. So reject all variables!
            sel_x_vars = []
            print('Zero Interaction and Polynomial variable(s) selected...')
        else:
            #### there is some coefficients at least that are non-zero and hence can be trusted!
            if verbose >= 1:
                df90.sort_values('Coefficient Values',ascending=False)[:10].plot(
                    kind='bar',x='Interaction Variable Names',y='Coefficient Values',
                    title='Top 10% Variable Interactions and their Coefficients ')
            elif verbose == 2:
                df90.sort_values('Coefficient Values',ascending=False).plot(
                    kind='bar',x='Interaction Variable Names',y='Coefficient Values',
                        title='All Variable Interactions and their Coefficients',figsize=(20,10))
            interactions = df90['Interaction Variable Names'].values.tolist()
            sel_x_vars = df90["X Names"].values.tolist()
        ####### Now we have to put the full dataframe together with selected variables and original predictors!
        #### DO NOT CHANGE THE NEXT LINE EVEN THOUGH IT MIGHT BE TEMPTING TO DO SO! It is correct!
        final_x_vars = x_vars + sel_x_vars
        sel_text_vars = [feature_xvar_dict[x] for x in sel_x_vars]
        print('Initially adding %d variable(s) due to Add_Poly = %s setting' %(len(sel_text_vars),Add_Poly))
        if verbose > 1:
            if len(sel_text_vars) <= 30:
                print('    Added variables: %s' %sel_text_vars)
        final_text_vars = [feature_xvar_dict[x] for x in final_x_vars]
        ##########################################################################################
        final_df = md.transform(data[predictors])
        xnames = md.get_feature_names()
        final_df = pd.DataFrame(final_df,index=orig_data_index, columns=xnames)
        final_df = final_df[final_x_vars]
        final_df.columns = final_text_vars
        copy_preds = copy.deepcopy(predictors)
        ####### If the original column's dtype was integer, put it back as integer ###
        for each_pred in copy_preds:
            if data[each_pred].dtype == int:
                final_df[each_pred] = final_df[each_pred].astype(int)
        final_df[targetvar] = data[targetvar].values
        #####################################################
        return final_text_vars, lm_p, final_df, md, final_x_vars,feature_xvar_dict
    else:
        #### Just return the transformed dataframe with all orig+poly+intxn vars
        final_df = md.transform(data[predictors])
        xnames = md.get_feature_names()
        final_df = pd.DataFrame(final_df,index=orig_data_index, columns=xnames)
        return predictors, lm, final_df, md, xnames, ""
##################################################################################
## Import sklearn
from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold
def print_model_metrics(modeltype,reg,X,y,fit_flag=False,verbose=1):
    ### If fit_flag is set to True, then you must return a fitted model ###
    ###   Else you must return the cv_scores.mean only ####
    n_splits = 5
    if verbose>0:
        print("Model Report :")
        print('    Number of Variables = ',X.shape[1])
    if modeltype == 'Regression':
        scv = KFold(n_splits=n_splits)
        #performs cross validation
        cv_scores = cross_val_score(reg,X,y.values.ravel(),cv=scv,scoring='neg_mean_squared_error')
        cv_scores = np.sqrt(np.abs(cv_scores))
        if verbose:
            print("    CV RMSE Score : %.4g +/- %.4g | Min = %.4g | Max = %.4g" %(
                np.mean(cv_scores),np.std(cv_scores),
                 np.min(cv_scores),np.max(cv_scores)))
    else:
        scv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        ### use F1 weighted since it works on multiple classes as well as binary ##
        cv_scores = cross_val_score(reg,X,y.values.ravel(),cv=scv,scoring='f1_weighted')
        if verbose>0:
            print("    CV Weighted F1 Score : %.4g +/- %.4g | Min = %.4g | Max = %.4g" % (
                np.mean(cv_scores),np.std(cv_scores),
                 np.min(cv_scores),np.max(cv_scores)))
    if fit_flag:
        return reg.fit(X,y),  cv_scores.mean()
    else:
        return  cv_scores.mean()
##############################################################################################
def select_best_variables(md, reg, df,names,n_orig_features,Add_Poly,verbose=0):
    """
    This program selects the top 10% of interaction variables created using linear modeling.
    """
    df90 = df[abs(df['Coefficient Values'])>=df['Coefficient Values'].quantile(0.90)]
    if verbose >= 1:
        df90.plot(
            kind='bar',x='Interaction Variable Names',y='Coefficient Values',
            title='Top 10% Variable Interactions and their Coefficients ')
    if verbose >= 1:
        df.plot(kind='bar',x='Interaction Variable Names',y='Coefficient Values',
                title='All Variable Interactions and their Coefficients',figsize=(20,10))
    interactions = df90['Interaction Variable Names'].values.tolist()
    print('%d Interaction and Polynomial variable(s) selected...\n' %len(interactions))
    finalvars = names + interactions
    #finalvars_index = [df[df['Interaction Variable Names']==x].index[0] for x in finalvars]
    return finalvars
##############################################################################################
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd
def Draw_ROC_MC_ML(model, X_test, y_true, target, model_name, verbose=0):
    figsize = (10, 6)
    y_proba = model.predict_proba(X_test)
    predicted = copy.deepcopy(y_proba)
    threshold = 0.50
    y_pred = model.predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    iterations = 0
    rare_class = find_rare_class(y_true)
    if isinstance(target,str):
        target = [target]
    for targ in target:
        if len(left_subtract(np.unique(y_true),np.unique(y_pred))) == 0:
            classes = list(range(len(np.unique(y_true))))
        else:
            classes = list(range(len(np.unique(y_pred))))
        n_classes = len(classes)
        if model_name.lower() == 'catboost':
            try:
                y_pred = (y_proba>0.5).astype(int).dot(classes)
            except:
                pass
        if n_classes == 2:
            ### Always set rare_class to 1 when there are only 2 classes for drawing ROC Curve!
            y_test = copy.deepcopy(y_true)
            predicted [:,0] = (predicted [:,0] >= (1-m_thresh)).astype('int')
            predicted [:,1] = (predicted [:,1] > threshold).astype('int')
            y_pred = predicted[:,rare_class]
        else:
            y_test = label_binarize(y_true, classes)
        try:
            if n_classes > 2:
                if y_test.shape[1] == y_proba.shape[1]:
                    classes = list(range(y_test.shape[1]))
                else:
                    if y_proba.shape[1] > y_test.shape[1]:
                        classes = list(range(y_proba.shape[1]))
                    else:
                        classes = list(range(y_test.shape[1]))
        except:
            pass
        if n_classes == 2:
            iterations = 1
        else:
            iterations = copy.deepcopy(n_classes)
        #####################   This is where you calculate ROC_AUC for all classes #########
        for i in range(iterations):
            if n_classes==2:
                fpr[rare_class], tpr[rare_class], _ = roc_curve(y_test.ravel(), y_proba[:,rare_class])
                roc_auc[rare_class] = auc(fpr[rare_class], tpr[rare_class])
            else:
                fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_proba[:,i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        ### Compute Micro Average which is a row-by-row score ###########
        try:
            if n_classes == 2:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
            else:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),label_binarize(y_pred, classes).ravel())
        except:
            if y_test.ravel().shape[0] != y_pred.ravel().shape[0]:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:,:n_classes].ravel(), y_proba[:,:n_classes].ravel())
            else:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba[:,rare_class].ravel())
        try:
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except:
            pass
        # micro-average ROC curve and ROC area has been computed
        labels = []
        if n_classes <= 2:
            ##############################################################################
            # Plot of a ROC curve for one class
            plt.figure(figsize=figsize)
            plt.plot(fpr[rare_class], tpr[rare_class], label='ROC curve (AUC = %0.5f)' % roc_auc[rare_class])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Predictions on held out data for target: %s' %targ)
            plt.legend(loc="lower right")
            plt.show();
        else:
        # Compute macro-average ROC curve and ROC area
            fig, ax = plt.subplots(figsize=figsize)
            colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
            for i in range(iterations):

                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for j in range(n_classes):
                    mean_tpr += interp(all_fpr, fpr[j], tpr[j])

                # Finally average it and compute AUC
                mean_tpr /= n_classes

                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                # Plot all ROC curves
                lw = 2
                color = next(colors)
                ax.plot(fpr[i], tpr[i], color=color, lw=lw)
                labels.append('ROC curve of class %d (area = %0.2f)' %(i, roc_auc[i]))
            try:
                labels.append('micro-average ROC curve (area = {0:0.2f})'
                               ''.format(roc_auc["micro"]))
                ax.plot(fpr["micro"], tpr["micro"],
                         color='deeppink', linestyle=':', linewidth=4)
            except:
                pass
            try:
                labels.append('macro-average ROC curve (area = {0:0.2f})'
                               ''.format(roc_auc["macro"]))
                ax.plot(fpr["macro"], tpr["macro"],
                         color='navy', linestyle=':', linewidth=4)
            except:
                pass
            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve for Predictions on held out data for target: %s' %targ)
            plt.legend(labels,loc="lower right")
            plt.show();

###################################################################################
# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
###################################################################################
def split_data_new(trainfile, testfile, target,sep, modeltype='Regression', randomstate=0):
    """
    Split your file or data frame into 2 or 3 splits. Stratified by Class automatically.
    Additionally, it will strip out non-numeric cols in your data if you need it.
    """
    randomstate = 99
    ####### Codes_list contains multiple encoders that are used to open a CSV file
    codes_list = ['utf-8','iso-8859-1','cp1252','latin1']
    seed = 99
    if type(trainfile) == str:
        if trainfile != '':
            #### First load the Train file if it is available ##########
            for codex in codes_list:
                try:
                    traindf = pd.read_csv(trainfile,sep=sep,encoding=codex,index_col=None)
                    break
                except:
                    continue
            print('Train data given. Shape = %s' %(traindf.shape,))
            print('Target Rows Train = %d' %len(traindf[traindf[target]==1]))
        else:
            print('Trainfile is not found. Check your input or file destination')
            return
        if  type(testfile) == str:
            if testfile != '':
                print('Train file and Test file given.')
                ###### If testfile is available, you need to combine train and test datasets
                print('Train DataFrame and Testfile given.')
                for codex in codes_list:
                    try:
                        testdf = pd.read_csv(testfile,sep=sep,encoding=codex,index_col=None)
                        break
                    except:
                        continue
                print('Test data given. Shape = %s' %(testdf.shape,))
            else:
                print('Testfile is not given. Continuing')
                testdf = ''
        elif type(testfile) == pd.DataFrame:
            print('Train file and Test dataframe given.')
            testdf = testfile[:]
        else:
            print('Testfile is unknown type.')
            testdf = ''
    elif type(trainfile) == pd.DataFrame:
        traindf = trainfile[:]
        if  type(testfile) == str:
            if testfile != '':
                print('Train DataFrame and Test file given.')
                ###### If testfile is available, you need to combine train and test datasets
                print('Train DataFrame and Testfile given.')
                for codex in codes_list:
                    try:
                        testdf = pd.read_csv(testfile,sep=sep,encoding=codex,index_col=None)
                        break
                    except:
                        continue
                print('Train Shape:%s Test Shape: %s' %(traindf.shape,testdf.shape))
            else:
                print('No testfile. Train Shape:%s' %(traindf.shape,))
                testdf = ''
        elif type(testfile) == pd.DataFrame:
            print('Train DataFrame and Test DataFrame given.')
            testdf = testfile[:]
    else:
        print('No Train or Test file provided. End...')
        return
    ######## This is where the splitting begins #################
    if target == '':
        print('No target variable provided. Please provide target variable')
        return traindf, None, None
    else:
        y = traindf[target]
    if modeltype == 'Classification':
        if traindf.shape[0] <= 1000:
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.30, random_state=randomstate)
        else:
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.20, random_state=randomstate)
    else:
        indices = np.random.permutation(traindf.shape[0])
        train_rows = int(0.8*traindf.shape[0])
    predictors = [x for x in traindf if x not in target]
    ##### select only string columns from the train data
    cols = copy.deepcopy(predictors)
    #cols = select_num_columns(traindf[predictors]).columns.tolist()
    traindf = traindf[cols+[target]]
    X = traindf[cols]
    if type(testdf) == str:
        #### If no test DataFrame is given, then split the train data into train, cv and test data #######
        if modeltype == 'Classification':
            trainindex, testindex = sss.split(X, y)
            tradata = traindf.iloc[trainindex[0]]
            tsdata = traindf.iloc[trainindex[1]]
        else:
            training_idx, test_idx = indices[:train_rows], indices[train_rows:]
            tradata, tsdata = traindf.iloc[training_idx], traindf.iloc[test_idx]
        #####################    CROSS VALIDATION PORTION  #######################
        X = tsdata[cols]
        y = tsdata[target]
        if modeltype == 'Classification':
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.40, random_state=randomstate)
            cvindex, ts_index = sss.split(X, y)
            cvdata = tsdata.iloc[cvindex[0]]
            tsdata = tsdata.iloc[cvindex[1]]
        else:
            indices = np.random.permutation(tsdata.shape[0])
            train_rows = int(0.4*tsdata.shape[0])
            training_idx, test_idx = indices[:train_rows], indices[train_rows:]
            cvdata, tsdata = tsdata.iloc[training_idx], tsdata.iloc[test_idx]
        return tradata, cvdata, tsdata, cols
    else:
        ####### If Test DataFrame is given then keep that as tsdata ########
        if modeltype == 'Classification':
            trainindex, testindex = sss.split(X, y)
            tradata = traindf.iloc[trainindex[0]]
            cvdata = traindf.iloc[trainindex[1]]
            try:
                if len(testdf[target].value_counts() >= 2):
                    print('    Number of Target Rows in Test Data: %d' %len(testdf[testdf[target]==1]))
                    tsdata = testdf[:]
                else:
                    print('    Test data has single class. Overwriting predictions on Test Target column.')
                    predictors = [x for x in list(traindf) if x not in target]
                    tsdata = testdf[:]
            except:
                print('    No Target column in Test data')
                testdf[target] = 0
                tsdata = testdf[cols+[target]]
        else:
            indices = np.random.permutation(traindf.shape[0])
            train_rows = int(0.8*traindf.shape[0])
            training_idx, test_idx = indices[:train_rows], indices[train_rows:]
            tradata, cvdata = traindf.iloc[training_idx], traindf.iloc[test_idx]
            tsdata = testdf[:]
        return tradata, cvdata, tsdata, cols
############################################################################
def filling_missing_values_simple(train, test, cats, nums):
    #### if you want to fill missing values using mean, median mode, go ahead!
    ####  I don't recommend But some people seem to like it. #####
    for varm in cats:
        if train[varm].isnull().sum() > 0:
            fillnum = train[varm].mode()[0]
            train[varm].fillna(fillnum, inplace=True)
            if not isinstance(test, str):
                if test[varm].isnull().sum() > 0:
                    test[varm].fillna(fillnum, inplace=True)
    for num in nums:
        if train[varm].isnull().sum() > 0:
            fillnum = train[num].mean()
            train[num].fillna(fillnum, inplace=True)
            if not isinstance(test, str):
                if test[varm].isnull().sum() > 0:
                    test[num].fillna(fillnum, inplace=True)
    ### This is a simple log transform for very large numeric values => highly recommended!
    for num in nums:
        mask = train[num]==0
        fillnum = 10e-5
        train.ix[mask,num] = fillnum
        train[num+'_log'] = np.log10(train[num]).fillna(0)
        if not isinstance(test, str):
            mask = test[num]==0
            test.ix[mask,num] = fillnum
            test[num+'_log'] = np.log10(test[num]).fillna(0)
    return train, test
############################################################
import os
def write_file_to_folder(df, each_target, base_filename, verbose=1):
    if isinstance(each_target, str):
        dir_name = copy.deepcopy(each_target)
    else:
        dir_name = str(each_target)
    filename = os.path.join(dir_name, base_filename)
    if verbose >= 1:
        if os.name == 'nt':
            print('    Saving predictions to .\%s' %filename)
        else:
            print('    Saving predictions to ./%s' %filename)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        df.to_csv(filename,index=False)
##############################################################
def create_ts_features(df, tscol):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.
    """
    df = copy.deepcopy(df)
    dt_adds = []
    try:
        df[tscol+'_hour'] = df[tscol].dt.hour.astype(int)
        df[tscol+'_minute'] = df[tscol].dt.minute.astype(int)
        dt_adds.append(tscol+'_hour')
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df[tscol].dt.dayofweek.astype(int)
        dt_adds.append(tscol+'_dayofweek')
        df[tscol+'_quarter'] = df[tscol].dt.quarter.astype(int)
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df[tscol].dt.month.astype(int)
        dt_adds.append(tscol+'_month')
        df[tscol+'_year'] = df[tscol].dt.year.astype(int)
        dt_adds.append(tscol+'_year')
        today = date.today()
        df[tscol+'_age_in_years'] = today.year - df[tscol].dt.year.astype(int)
        dt_adds.append(tscol+'_age_in_years')
        df[tscol+'_dayofyear'] = df[tscol].dt.dayofyear.astype(int)
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df[tscol].dt.day.astype(int)
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df[tscol].dt.weekofyear.astype(int)
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 5) | (df[tscol+'_dayofweek'] == 6)
        df[tscol+'_weekend'] = 0
        df.loc[weekends, tscol+'_weekend'] = 1
        df[tscol+'_weekend'] = df[tscol+'_weekend'].astype(int)
        dt_adds.append(tscol+'_weekend')
    except:
        print('    Error in creating date time derived features. Continuing...')
    df = df[dt_adds].fillna(0).astype(int)
    return df
################################################################
from dateutil.relativedelta import relativedelta
from datetime import date
##### This is a little utility that computes age from year ####
def compute_age(year_string):
    today = date.today()
    age = relativedelta(today, year_string)
    return age.years
#################################################################
def create_time_series_features(dtf, ts_column):
    """
    This creates between 8 and 10 date time features for each date variable. The number of features
    depends on whether it is just a year variable or a year+month+day and whether it has hours and mins+secs.
    So this can create all these features using just the date time column that you send in.
    It returns the entire dataframe with added variables as output.
    """
    dtf = copy.deepcopy(dtf)
    #### If for some reason ts_column is just a number, make sure it is a string so it does not blow up and concatenated
    if not isinstance(ts_column,str):
        ts_column = str(ts_column)
    try:
        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        if dtf[ts_column].isnull().sum() > 0:
            missing_flag = True
            new_missing_col = ts_column + '_Missing_Flag'
            dtf[new_missing_col] = 0
            dtf.loc[dtf[ts_column].isnull(),new_missing_col]=1
            dtf[ts_column] = dtf[ts_column].fillna(method='ffill')
        if dtf[ts_column].dtype in [np.float64,np.float32,np.float16]:
            dtf[ts_column] = dtf[ts_column].astype(int)
        ### if we have already found that it was a date time var, then leave it as it is. Thats good enough!
        date_items = dtf[ts_column].apply(str).apply(len).values
        #### In some extreme cases,
        if all(date_items[0] == item for item in date_items):
            if date_items[0] == 4:
                ### If it is just a year variable alone, you should leave it as just a year!
                age_col = ts_column+'_age_in_years'
                dtf[age_col] = dtf[ts_column].map(lambda x: pd.to_datetime(x,format='%Y')).apply(compute_age).values
                return dtf[[ts_column,age_col]]
            else:
                ### if it is not a year alone, then convert it into a date time variable
                dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        else:
            dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        dtf = create_ts_features(dtf,ts_column)
    except:
        print('Error in Processing %s column for date time features. Continuing...' %ts_column)
    return dtf
########################################
from collections import Counter
import time
########################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
####################################################################################
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, SMOTENC
from collections import Counter
import copy
import pdb
# Regression_Resampler is a neat library that returns a numpy list of classes for each
#  corresponding sample. from target data. It also automatically merges classes if they are small
from reg_resampler import resampler
def training_with_SMOTE(X_df,y_df,target,eval_set,model_input,Boosting_Flag,eval_metric,
                                        modeltype, model_name,
                                     training=True,minority_class=1,imp_cats=[],
                                     calibrator_flag=False,
                                     GPU_exists=False, params={},
                                     model_label='Single_Label', verbose=0):
    """
    #########    OVERSAMPLING OF MINORITY CLASS AND TRAINING  SIZES  ###############
    ####  SMOTE (Synthetic Minority Oversampling Technique) works well with imbalanced classes.
    ####  This is a great way to train a model in such cases with repeated samples of rare-class
    ####  along with batches of majority class and get the model to differentiate between 2 classes.
    ####  This can also be used for Regression problems by using the numeric target values converted
    ####  to cluster labels using KMeans. You can then use the cluster labels to add more samples.
    #########    MAKE SURE YOU TUNE THE DEFAULTS GIVEN HERE!  ###############
    """
    X_df = copy.deepcopy(X_df)
    y_df = copy.deepcopy(y_df)
    model_copy = copy.deepcopy(model_input)
    model = copy.deepcopy(model_copy)
    start_time = time.time()
    ##### Now we will start using Resampling Library to make data more balanced ###
    if modeltype == 'Regression':
        print("Using Regression Resampler library to make skewed target variable more balanced...")
    else:
        print("Using SMOTE's over-sampling techniques to make the %d classes balanced..." %len(np.unique(y_df)))
    try:
        #######################################################################
        # Initialize the resampler object
        rs = resampler()
        df_train = X_df.join(y_df)
        # Generate classes
        if modeltype == 'Regression':
            num_samples = max(10, int(df_train.shape[0]/1000))
            #### Get the cluster labels for the data  - transform y_df into labels. Then send it to SMOTE.
            num_clusters = int(np.round(max(2,np.log10(y_df.shape[0]))))
            try:
                Y_classes = rs.fit(df_train, target=target, bins=num_clusters,
                                min_n_samples=num_samples, verbose=2)
            except:
                print('Min number of samples %s is too high. Resetting it to 1 and continuing...' %num_samples)
                num_samples = 1
                Y_classes = rs.fit(df_train, target=target, bins=num_clusters,
                                min_n_samples=num_samples, verbose=2)
        else:
            # Create the actual target variable if this is Classification
            Y_classes = df_train[target]
        #### Select the right SMOTE algorithm based on whether there are any imp_cats in data
        cat_vars_index = [i for i, j in enumerate(X_df.columns) if j in imp_cats]
        ### get the number of K_Neighbors using this calculation
        rare_samples = pd.Series(Counter(Y_classes)).min()
        #print('    Number of Rare Class samples = %d' %rare_samples)
        rare_pct = rare_samples/Y_classes.shape[0]
        smallest_kn =  rare_samples - 1
        if smallest_kn > 5:
            smallest_kn = 5
        elif smallest_kn > 2 and smallest_kn < 5:
            smallest_kn = 2
        else :
            smallest_kn = 1
        #print('    Number of K Neighbors selected for SMOTE = %d' %smallest_kn)
        #print('    Pct of Rare Class in data = %0.2f%%' %(rare_pct*100))
        if rare_pct > 0.10:
            print('    This is not an Imbalanced data set. No need to use SMOTE but continuing...')
    except:
        print('    SMOTE is erroring. Continuing without SMOTE...')
        return model
        #######################################################################
    ### now train the model on full train data #################################
    class_weighted_rows = get_class_distribution(Y_classes)
    if len(imp_cats) > 0:
        # Your favourite oversampler = SMOTENC is better than SMOTE in most cases!
        smote = SMOTENC(categorical_features=cat_vars_index,
                    sampling_strategy=class_weighted_rows,
                    random_state=42, k_neighbors=smallest_kn, n_jobs=-1)
    else:
        # Your favourite oversampler = SMOTE is better than SMOTEENN in most cases!
        smote = SMOTE(random_state=27, sampling_strategy=class_weighted_rows, n_jobs=-1)
    try:
        if modeltype == 'Regression':
            #### Now we must re-sample the data using reg resampler to train full model
            X_df_res, y_df_res = rs.resample(smote, df_train, Y_classes)
        else:
            #### For classification problems simply using the original data will do
            X_df_res, y_df_res = smote.fit_resample(X_df, y_df)
    except:
        print('Regression-resampler is erroring. Continuing...')
        model = model_training_smote(model, X_df, y_df, eval_set, eval_metric,
                             params, imp_cats, modeltype,
                            calibrator_flag, model_name, Boosting_Flag, model_label,
                             GPU_exists)
        print('    Model Training time taken = %0.0f seconds' %(time.time()-start_time))
        return model
    print('Training model now on resampled train data: %s. This will take time...' %(X_df_res.shape,))
    # Fit the model using x_train and y_train full data sent in
    model = model_training_smote(model, X_df_res, y_df_res, eval_set, eval_metric,
                         params, imp_cats, modeltype,
                        calibrator_flag, model_name, Boosting_Flag, model_label,
                         GPU_exists)
    print('    Resampled data Training time taken = %0.0f seconds' %(time.time()-start_time))
    #Now make predictions on held out data
    if training:
        x_test = eval_set[0][0]
        y_test = eval_set[0][1]
        # Check the score
        score = model.score(x_test,y_test)
        print('    Resampled model score on Held out Validation data = %s' %abs(score))
    model_str = str(model).split("(")[0]
    print('##################  Completed Imbalanced Training using %s ################' %model_str)
    if not training:
        if verbose >= 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(12,5))
            if modeltype == 'Regression':
                y_df.plot(kind='hist',title='Dist. of target before Regression Resampler', ax=ax1, color='r')
                y_df_res.plot(kind='hist',title='Dist. of target after Regression Resampler', ax=ax2, color='b')
            else:
                y_df.value_counts(1).plot(kind='bar',title='Distribution of classes before SMOTE', ax=ax1, color='r')
                y_df_res.value_counts(1).plot(kind='bar',title='Distribution of classes after SMOTE', ax=ax2, color='b')
    return model
##############################################################################################
def get_class_distribution(y_input):
    classes = np.unique(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_input), y=y_input)
    class_weights[(class_weights<1)]=1
    class_weights = class_weights.astype(int)
    class_rows = class_weights*[xp[x] for x in classes]
    class_weighted_rows = dict(zip(classes,class_rows))
    return class_weighted_rows
######################################################################################
def model_training_smote(model, x_train, y_train, eval_set, eval_metric,
                         params, imp_cats, modeltype,
                        calibrator_flag, model_name, Boosting_Flag, model_label,
                         GPU_exists):
    early_stopping = 5
    if model_label == 'Single_Label':
        if Boosting_Flag:
            if model_name == 'XGBoost':
                try:
                    if calibrator_flag:
                        model.fit(x_train, y_train)
                    else:
                        if eval_set==[()] or modeltype == 'Multi_Classification':
                            model.fit(x_train, y_train)
                        else:
                            model.fit(x_train, y_train,early_stopping_rounds=early_stopping,
                                eval_metric=eval_metric,eval_set=eval_set,verbose=False)
                except:
                    #### On Colab, even though GPU exists, many people don't turn it on.
                    ####  In that case, XGBoost blows up when gpu_predictor is used.
                    ####  This is to turn it back to cpu_predictor in case GPU errors!
                    if GPU_exists:
                        print('Warning: GPU exists but it is not turned on. Using CPU for predictions...')
                        if calibrator_flag:
                            model.base_estimator.set_params(**params)
                            model.fit(x_train, y_train)
                        else:
                            model.estimator.set_params(**params)
                            model.fit(x_train, y_train)
                    else:
                        if eval_set==[()] or modeltype == 'Multi_Classification':
                            model.fit(x_train, y_train)
                        else:
                            model.fit(x_train, y_train, early_stopping_rounds=early_stopping,
                                eval_metric=eval_metric,eval_set=eval_set,verbose=False)
            else:
                early_stopping = 20
                if eval_set == [()]:
                    model.fit(x_train, y_train, cat_features=imp_cats,plot=False, verbose=0)
                else:
                    model.fit(x_train, y_train, cat_features=imp_cats, verbose=0,
                                eval_set=eval_set, use_best_model=True, plot=False)
        else:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    return model
#################################################################################
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
import copy
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
#############################################################################################
def multi_f1(truth, predictions):
    return f1_score(truth, predictions,average=None)
def multi_precision(truth, predictions):
    return precision_score(truth, predictions,average=None)
##############################################################################################
def Draw_MC_ML_PR_ROC_Curves(classifier,X_test,y_test):
    """
    ========================================================================================
    Precision-Recall Curves: Extension of Original Version in SKLearn's Documentation Page:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    ========================================================================================
    """
    figsize = (10, 6)
    ###############################################################################
    # In binary classification settings
    # Compute the average precision score for Binary Classes
    ###############################################################################
    y_pred = classifier.predict(X_test)
    if len(left_subtract(np.unique(y_test),np.unique(y_pred))) == 0:
        classes = list(range(len(np.unique(y_test))))
    else:
        classes = list(range(len(np.unique(y_pred))))
    #classes = list(range(len(np.unique(y_test))))
    n_classes = len(classes)
    if n_classes == 2:
        try:
            y_score = classifier.decision_function(X_test)
        except:
            y_score = classifier.predict_proba(X_test)
        try:
            average_precision = average_precision_score(y_test, y_score)
        except:
            average_precision = multi_precision(y_test, classifier.predict(X_test)).mean()
        print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))
        f_scores = multi_f1(y_test,classifier.predict(X_test))
        print('Macro F1 score, averaged over all classes: {0:0.2f}'
              .format(f_scores.mean()))
        ###############################################################################
        # Plot the Precision-Recall curve
        # ................................
        plt.figure(figsize=figsize)
        try:
            ### This works for Logistic Regression and other Linear Models ####
            precision, recall, _ = precision_recall_curve(y_test, y_score)
        except:
            ### This works for Non Linear Models such as Forests and XGBoost #####
            precision, recall, _ = precision_recall_curve(y_test, y_score[:,1])

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='g', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='g', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Avg.Precision={0:0.2f}, Avg F1={1:0.2f}'.format(
                  average_precision,f_scores.mean()))
        plt.show();
        ###############################################################################
    else:
        # In multi-label settings
        # ------------------------
        #
        # Create multi-label data, fit, and predict
        # ...........................................
        #
        # We create a multi-label dataset, to illustrate the precision-recall in
        # multi-label settings

        # Use label_binarize to be multi-label like settings
        Y = label_binarize(y_test, classes)
        n_classes = Y.shape[1]
        Y_test = copy.deepcopy(Y)
        try:
            y_score = classifier.decision_function(X_test)
        except:
            y_score = classifier.predict_proba(X_test)

        ###############################################################################
        # The average precision score in multi-label settings
        # ....................................................

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test[
                        :,:n_classes].ravel(),y_score[:,:n_classes].ravel())
        average_precision["micro"] = average_precision_score(Y_test[
                        :,:n_classes], y_score[:,:n_classes],average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))

        ###############################################################################
        # Plot Precision-Recall curve for each class and iso-f1 curves
        # Plot the micro-averaged Precision-Recall curve
        ###############################################################################
        # .............................................................
        #
        # setup plot details
        colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')

        plt.figure(figsize=figsize)
        f_scores = multi_f1(y_test,classifier.predict(X_test))
        print('Macro F1 score, averaged over all classes: {0:0.2f}'
              .format(f_scores.mean()))
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=1)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Micro Avg Precision-Recall curve with iso-f1 curves')
        plt.legend(lines, labels, loc='lower left', prop=dict(size=10))
        plt.show();
######################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
def draw_confusion_maxtrix(y_test,y_pred, model_name='Model',ax=''):
    """
    This plots a beautiful confusion matrix based on input: ground truths and predictions
    """
    #Confusion Matrix
    '''Plotting CONFUSION MATRIX'''
    import seaborn as sns
    sns.set_style('darkgrid')

    '''Display'''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    pd.options.display.float_format = '{:,.2f}'.format

    #Get the confusion matrix and put it into a df
    from sklearn.metrics import confusion_matrix, f1_score

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,
                         index = np.unique(y_test).tolist(),
                         columns = np.unique(y_test).tolist(),
                        )

    sns.heatmap(cm_df,
                center=0,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),
                annot=True,
                fmt='g',
               ax=ax)

    ax.set_title(' %s \nF1 Score(avg = micro): %0.2f \nF1 Score(avg = macro): %0.2f' %(
        model_name,f1_score(y_test, y_pred, average='micro'),f1_score(y_test, y_pred, average='macro')),
              fontsize = 13)
    ax.set_ylabel('True label', fontsize = 13)
    ax.set_xlabel('Predicted label', fontsize = 13)
######################################################################################
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def plot_classification_results(m, X_true, y_true, y_pred, labels, target_names, each_target):
    try:
        fig, axes = plt.subplots(2,2,figsize=(15,15))
        plot_roc_curve(m, X_true, y_true, ax=axes[0,1])
        axes[0,1].set_title('ROC AUC Curve: %s' %each_target)
        plot_precision_recall_curve(m, X_true, y_true, ax=axes[1,0])
        axes[1,0].set_title('PR AUC Curve %s' %each_target)
        y_pred = m.predict(X_true)
        draw_confusion_maxtrix(y_true, y_pred, 'Confusion Matrix', ax=axes[0,0])
        try:
            clf_report = classification_report(y_true,
                                               y_pred,
                                               labels=labels,
                                               target_names=target_names,
                                               output_dict=True)
        except:
            clf_report = classification_report(y_true,y_pred,labels=target_names,
                target_names=labels,output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True,ax=axes[1,1],fmt='0.2f');
        axes[1,1].set_title('Classification Report: %s' %each_target)
    except:
        print('Error: could not plot classification results. Continuing...')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
def print_classification_model_stats(y_true, predicted, m_thresh=0.5):
    """
    This prints classification metrics in a nice format only for binary classes
    """
    # Use this to Test Classification Problems Only ####
    try:
        rare_class = find_rare_class(y_true)
        reg_acc = [0,0]
        for i,threshold in zip(range(2), [0.5, m_thresh]):
            if threshold != 0.5:
                predicted [:,0] = (predicted [:,0] >= (1-threshold)).astype('int')
                predicted [:,1] = (predicted [:,1] >= threshold).astype('int')
                y_pred = predicted[:,rare_class]
            else:
                y_pred = predicted.argmax(axis=1)
            print('Balanced Accuracy = %0.2f%% with Threshold = %0.2f' %(
                100*balanced_accuracy_score(y_true, y_pred),threshold))
            print('Confusion Matrix:')
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            reg_acc[i] = balanced_accuracy_score(y_true, y_pred)
        print('#####################################################################')
        if  reg_acc[0] >= reg_acc[1] :
            print('Regular threshold = %0.2f is better' %0.5)
            return reg_acc[0]
        else:
            print('Modified threshold = %0.2f is better' %m_thresh)
            return reg_acc[1]
    except:
        print('Error: printing classification model metrics. Continuing...')
        return 0
#####################################################################
#####     REGRESSION CHARTS AND METRICS ARE PRINTED PLOTTED HERE
#####################################################################
import time
def plot_regression_scatters(df, df2, num_vars, kind='scatter', plot_name=''):
    """
    Great way to plot continuous variables fast. Just sent them in and it will take care of the rest!
    """
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    num_vars_len = len(num_vars)
    col = 2
    start_time = time.time()
    row = len(num_vars)
    fig, ax = plt.subplots(row, col)
    if col < 2:
        fig.set_size_inches(min(15,8),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    else:
        fig.set_size_inches(min(col*10,20),row*5)
        fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    fig.suptitle('Regression Metrics Plots for %s Model' %plot_name, fontsize=20)
    counter = 0
    if row == 1:
        ax = ax.reshape(-1,1).T
    for k in np.arange(row):
        row_color = next(colors)
        for l in np.arange(col):
            try:
                if col==1:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:,k]
                        y = df2[:,k]
                    ax1 = ax[k][l]
                    lineStart = x.min()
                    lineEnd = x.max()
                    ax1.scatter(x, y, color=row_color)
                    ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=row_color)
                    ax1.set_xlabel('Actuals')
                    ax1.set_ylabel('Predicted')
                    ax1.set_title('Predicted vs Actuals Plot for Target = %s' %num_vars[k])
                else:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:,k]
                        y = df2[:,k]
                    lineStart = x.min()
                    lineEnd = x.max()
                    if l == 0:
                        ax1 = ax[k][l]
                        ax1.scatter(x, y,  color = row_color)
                        ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = row_color)
                        ax1.set_xlabel('Actuals')
                        ax1.set_ylabel('Predicted')
                        ax1.set_title('Predicted vs Actuals Plot for Target = %s' %num_vars[k])
                    else:
                        ax1 = ax[k][l]
                        ax1.hist((x-y), density=True,color = row_color)
                        ax1.axvline(color='k')
                        ax1.set_title('Residuals Plot for Target = %s' %num_vars[k])
            except:
                if col == 1:
                    counter += 1
                else:
                    ax[k][l].set_title('No Predicted vs Actuals Plot for plot as %s is not numeric' %num_vars[k])
                    counter += 1
    print('Regression Plots completed in %0.3f seconds' %(time.time()-start_time))
################################################################################
def print_regression_model_stats(actuals, predicted, targets='', plot_name=''):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    if isinstance(actuals,pd.Series) or isinstance(actuals,pd.DataFrame):
        actuals = actuals.values
    if isinstance(predicted,pd.Series) or isinstance(predicted,pd.DataFrame):
        predicted = predicted.values
    if len(actuals) != len(predicted):
        print('Error: Number of actuals and predicted dont match. Continuing...')
    if targets == "":
        try:
            ### This is for Multi_Label Problems ###
            assert actuals.shape[1]
            multi_label = True
            if isinstance(actuals,pd.Series):
                cols = [actuals.name]
            elif isinstance(actuals,pd.DataFrame):
                cols = actuals.columns.tolist()
            else:
                cols = ['target_'+str(i) for i in range(actuals.shape[1])]
        except:
            #### THis is for Single Label problems #####
            multi_label = False
            if isinstance(actuals,pd.Series):
                cols = [actuals.name]
            elif isinstance(actuals,pd.DataFrame):
                cols = actuals.columns.tolist()
            else:
                cols = ['target_1']
    else:
        cols = copy.deepcopy(targets)
        if isinstance(targets, str):
            cols = [targets]
        if len(cols) == 1:
            multi_label = False
        else:
            multi_label = True
    try:
        plot_regression_scatters(actuals,predicted,cols,plot_name=plot_name)
    except:
        print('Could not draw regression plot but continuing...')
    if multi_label:
        for i in range(actuals.shape[1]):
            actuals_x = actuals[:,i]
            try:
                predicted_x = predicted[:,i]
            except:
                predicted_x = predicted[:]
            print('Regression Metrics for Target=%s' %cols[i])
            mae, mae_asp, rmse_asp = print_regression_metrics(actuals_x, predicted_x)
    else:
        mae, mae_asp, rmse_asp = print_regression_metrics(actuals, predicted)
    return mae, mae_asp, rmse_asp
################################################################################
def print_regression_metrics(actuals, predicted):
    mae = mean_absolute_error(actuals, predicted)
    mae_asp = (mean_absolute_error(actuals, predicted)/actuals.std())*100
    rmse_asp = (np.sqrt(mean_squared_error(actuals,predicted))/actuals.std())*100
    rmse = print_rmse(actuals, predicted)
    _ = print_mape(actuals, predicted)
    mape = print_mape(actuals, predicted)
    print('    MAE = %0.4f' %mae)
    print("    MAPE = %0.0f%%" %(mape))
    print('    RMSE = %0.4f' %rmse)
    print('    MAE as %% std dev of Actuals = %0.1f%%' %(mae/abs(actuals).std()*100))
    # Normalized RMSE print('RMSE = {:,.Of}'.format(rmse))
    print('    Normalized RMSE (%% of MinMax of Actuals) = %0.0f%%' %(100*rmse/abs(actuals.max()-actuals.min())))
    print('    Normalized RMSE (%% of Std Dev of Actuals) = %0.0f%%' %(100*rmse/actuals.std()))
    return mae, mae_asp, rmse_asp
################################################################################
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev
################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(abs(perc_err))
################################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import cycle
import matplotlib.pyplot as plt
def plot_dfplot(dfplot,plot_title=""):
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    plt.figure(figsize=figsize)
    actuals = dfplot['Actuals']
    predicted =  dfplot['Predicted']
    lineStart = actuals.min()
    lineEnd = actuals.max()
    plt.scatter(actuals, predicted, color = next(colors), alpha=0.5,label='Predictions')
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Predicted vs Actuals for Target = %s' %plot_title, fontsize=12)
    plt.show();
###############################################################################
from collections import defaultdict
def return_list_matching_keys(dicto,list_keys):
    '''Return a list of values matching keys in a dictionary
    from a list of keys. Returns Values in the Same order as the List Keys.
    '''
    results = []
    for each in list_keys:
        results.append(dicto[each])
    return results
##################################################################################
def remove_highly_correlated_vars_fast(df, corr_limit=0.70):
    """
    This is a simple method to remove highly correlated features fast using Pearson's Correlation.
    Use this only for float and integer variables. It will automatically select those only.
    It can be used for very large data sets where featurewiz has trouble with memory
    """
    # Creating correlation matrix
    cor_matrix = df.corr().abs().astype(np.float16)
    # Selecting upper triangle of correlation matrix
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),
                                  k=1).astype(np.bool))
    # Finding index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_limit)]
    print();
    print('Highly correlated columns to remove: %s' %to_drop)
    return to_drop
#####################################################################################
def add_entropy_binning(temp_train, targ, num_vars, important_features, temp_test,
                       modeltype, entropy_binning,verbose=0):
    """
        ######   This is where we do ENTROPY BINNING OF CONTINUOUS VARS ###########
        #### It is best to do Binning on ONLY on the top most variables from Important_Features!
        #### Make sure that the Top 2-10 vars are all CONTINUOUS VARS! Otherwise Binning is Waste!
        #### This method ensures you get the Best Results by generalizing on the top numeric vars!
    """
    temp_train = copy.deepcopy(temp_train)
    temp_test = copy.deepcopy(temp_test)
    max_depth = 10
    seed = 99
    num_vars = copy.deepcopy(num_vars)
    continuous_vars = copy.deepcopy(num_vars)
    important_features = copy.deepcopy(important_features)
    print('Determining which of %d continuous variables should be Entropy Binned...' %len(continuous_vars))
    if len(continuous_vars) > 0 and len(continuous_vars) <= 2:
        max_depth =  2
        continuous_vars = continuous_vars[:]
    elif len(continuous_vars) > 2 and len(continuous_vars) <= 5:
        max_depth = len(continuous_vars) - 2
        continuous_vars = continuous_vars[:2]
    elif len(continuous_vars) > 5 and len(continuous_vars) <= 10:
        max_depth = 5
        continuous_vars = continuous_vars[:5]
    elif len(continuous_vars) > 10 and len(continuous_vars) <= 50:
        max_depth = max_depth
        continuous_vars = continuous_vars[:10]
    else:
        max_depth = max_depth
        continuous_vars = continuous_vars[:50]
    new_bincols = []
    ###   This is an Awesome Entropy Based Binning Method for Continuous Variables ###########
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    if modeltype == 'Regression':
        clf = DecisionTreeRegressor(criterion='mse',min_samples_leaf=2,
                                    max_depth=max_depth,
                                    random_state=seed)
    else:
        clf = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2,
                                         max_depth=max_depth,
                                         random_state=seed)
    entropy_threshold = []
    ####### This is where we bin each variable through a method known as Entropy Binning ##############
    for each_num in continuous_vars:
        try:
            clf.fit(temp_train[each_num].values.reshape(-1,1),temp_train[targ].values)
            entropy_threshold = clf.tree_.threshold[clf.tree_.threshold>-2]
            entropy_threshold = np.sort(entropy_threshold)
            if isinstance(each_num, str):
                bincol = each_num+'_bin'
                temp_train[bincol] = np.digitize(temp_train[each_num].values, entropy_threshold)
            else:
                bincol = 'bin_'+str(each_num)
                temp_train[bincol] = np.digitize(temp_train[each_num].values, entropy_threshold)
            #### We Drop the original continuous variable after you have created the bin when Flag is true
            ### We Don't drop these original numeric vars since they will be used later for full train binning
            if type(temp_test) != str:
                if isinstance(each_num, str):
                    bincol = each_num+'_bin'
                    temp_test[bincol] = np.digitize(temp_test[each_num].values, entropy_threshold)
                else:
                    bincol = 'bin_'+str(each_num)
                    temp_test[bincol] = np.digitize(temp_test[each_num].values, entropy_threshold)
                #### We Drop the original continuous variable after you have created the bin when Flag is true
                ### We Don't drop these original numeric vars since they will be used later for full train binning
            if entropy_binning:
                ### In the second time, we don't repeat adding binned vars since they have already been added!
                #### we also make sure that the orig num vars which have now been binned are removed!
                temp_train.drop(each_num,axis=1,inplace=True)
                if type(temp_test) != str:
                    temp_test.drop(each_num,axis=1,inplace=True)
            else:
                #### In the first time, we add binned vars to  important_features  ###
                ### In the second time, we don't repeat that since they have already been added!
                important_features.append(bincol)
            num_vars.append(bincol)
            important_features.remove(each_num)
            #### Drop these original continuous variable from further consideration that's all! ###
            num_vars.remove(each_num)
            new_bincols.append(bincol)
        except:
            print('Error in %s during Entropy Binning' %each_num)
    print('    Selected and binned only top %s continuous variables.' %(len(new_bincols)))
    if verbose and len(new_bincols) <= 30:
        print('    %s' %new_bincols)
    return temp_train, num_vars, important_features, temp_test
###########################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.1.677'
print("""Imported Auto_ViML version: %s. Call using:
             m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission='',
                            scoring_parameter='', KMeans_Featurizer=False,
                            hyper_param='RS',feature_reduction=True,
                             Boosting_Flag='CatBoost', Binning_Flag=False,
                            Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False,
                            verbose=1)
            """ %version_number)
print('The new Auto_ViML can solve multi-label, multi-output problems. Check if your version is >= 0.1.669')
print('To get the latest version, perform "pip install autoviml --no-cache-dir --ignore-installed"')
###########################################################################################
