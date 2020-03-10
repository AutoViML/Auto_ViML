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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import pdb
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import classification_report, confusion_matrix
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from autoviml.Transform_KM_Features import Transform_KM_Features
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor

from autoviml.custom_scores import accu, rmse, gini_sklearn, gini_meae
from autoviml.custom_scores import gini_msle, gini_mae, gini_mse, gini_rmse
from autoviml.custom_scores import gini_accuracy, gini_bal_accuracy, gini_roc

from autoviml.custom_scores import gini_precision, gini_average_precision, gini_weighted_precision
from autoviml.custom_scores import gini_macro_precision, gini_micro_precision
from autoviml.custom_scores import gini_samples_precision, gini_f1, gini_weighted_f1
from autoviml.custom_scores import gini_macro_f1, gini_micro_f1, gini_samples_f1
from autoviml.custom_scores import gini_log_loss, gini_recall, gini_weighted_recall
from autoviml.custom_scores import gini_samples_recall, gini_macro_recall, gini_micro_recall
from autoviml.QuickML_Stacking import QuickML_Stacking
from autoviml.Transform_KM_Features import Transform_KM_Features
from autoviml.QuickML_Ensembling import QuickML_Ensembling

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
        print(' Class  -> Counts -> Percent')
        for cls in counts.keys():
            print("%6s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
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
def analyze_problem_type(train, targ,verbose=0):
    """
    This module analyzes a Target Variable and finds out whether it is a
    Regression or Classification type problem
    """
    if train[targ].dtype != 'int64' and train[targ].dtype != float :
        if len(train[targ].unique()) == 2:
            if verbose >= 1:
                print('"\n ################### Binary-Class ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 15:
                model_class = 'Multi_Classification'
                if verbose >= 1:
                    print('"\n ################### Multi-Class ######################''')
    elif train[targ].dtype == 'int64' or train[targ].dtype == float :
        if len(train[targ].unique()) == 2:
            if verbose >= 1:
                print('"\n ################### Binary-Class ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 15:
                model_class = 'Multi_Classification'
                if verbose >= 1:
                    print('"\n ################### Multi-Class ######################''')
        else:
            model_class = 'Regression'
            if verbose >= 1:
                print('"\n ################### Regression  ######################''')
    elif train[targ].dtype == object:
            if len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 2:
                model_class = 'Binary_Classification'
                if verbose >= 1:
                    print('"\n ################### Binary-Class  ##################### " ')
            else:
                model_class = 'Multi_Classification'
                if verbose >= 1:
                    print('"\n ################### Multi-Class  ######################''')
    elif train[targ].dtype == bool:
                model_class = 'Binary_Classification'
                if verbose >= 1:
                    print('"\n ################### Binary-Class  ######################''')
    elif train[targ].dtype == 'int64':
        if len(train[targ].unique()) == 2:
            if verbose >= 1:
                print('"\n ################### Binary-Class  ##################### " ')
            model_class = 'Binary_Classification'
        elif len(train[targ].unique()) > 1 and len(train[targ].unique()) <= 25:
                model_class = 'Multi_Classification'
                if verbose >= 1:
                    print('"\n ################### Multi-Class  ######################''')
        else:
            model_class = 'Regression'
            if verbose >= 1:
                print('"\n ################### Regression  ######################''')
    else :
        if verbose >= 1:
            print('\n ###################### REGRESSION  #####################')
        model_class = 'Regression'
    return model_class
#############################################################################################################
def Auto_ViML(train, target, test='',sample_submission='',hyper_param='GS', feature_reduction=True,
            scoring_parameter='logloss', Boosting_Flag=None, KMeans_Featurizer=False,
            Add_Poly=0, Stacking_Flag=False, Binning_Flag=False,
              Imbalanced_Flag=False, verbose=0):
    """
    #########################################################################################################
    #############       This is not an Officially Supported Google Product!         #########################
    #########################################################################################################
    ####       Automatically Build Variant Interpretable Machine Learning Models (Auto_ViML)           ######
    ####                                Developed by Ramadurai Seshadri                                ######
    ######                               Version 1.0452                                              ########
    #####    STABLE VERSION WITH CATBOOST for categorical heavy data sets.  Dec 5,2019              #########
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
    ####   Libraries other than "CatBoost" and "SHAP" library for SHAP values for interpretability.     #####
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
    ####   hyper_param: Tuning options are GridSearch ('GS') and RandomizedSearch ('RS'). Default is 'GS'.###
    ####   feature_reduction: Default = 'True' but it can be set to False if you don't want automatic    ####
    ####         feature_reduction since in Image data sets like digits and MNIST, you get better       #####
    ####         results when you don't reduce features automatically. You can always try both and see. #####
    ####   KMeans_Featurizer = True: Adds a cluster label to features based on KMeans. Use for Linear.  #####
    ####         False (default) = For Random Forests or XGB models, leave it False since it may overfit.####
    ####   Boosting Flag: you have 4 possible choices (default is False):                               #####
    ####    None = This will build a Linear Model                                                       #####
    ####    False = This will build a Random Forest or Extra Trees model (also known as Bagging)        #####
    ####    True = This will build an XGBoost model                                                     #####
    ####    CatBoost = THis will build a CatBoost model (provided you have CatBoost installed)          #####
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
    #####   These copies are to make sure that the originals are not destroyed #############################
    test = copy.deepcopy(test)
    orig_train = copy.deepcopy(train)
    orig_test = copy.deepcopy(test)
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
    early_stopping = 4 #### Early stopping rounds for XGBoost ######
    encoded = '_Label_Encoded' ### This is the tag we add to feature names in the end to indicate they are label encoded
    catboost_limit = 0.4 #### The catboost_limit represents the percentage of num vars in data. ANy lower, CatBoost is used.
    cat_code_limit = 100 #### If the number of dummy variables to create in a data set exceeds this, CatBoost is the default Algorithm used
    one_hot_size = 100 #### This determines the max length of one_hot_max_size parameter of CatBoost algrithm
    n_steps = 3 ### number of estimator steps between 100 and max_estims
    class colour:
       PURPLE = '\033[95m'
       CYAN = '\033[96m'
       DARKCYAN = '\033[36m'
       BLUE = '\033[94m'
       GREEN = '\033[92m'
       YELLOW = '\033[93m'
       RED = '\033[91m'
       BOLD = '\033[1m'
       UNDERLINE = '\033[4m'
       END = '\033[0m'
    ##########   This is where some more default parameters are set up ###########################################
    data_dimension = orig_train.shape[0]*orig_train.shape[1]  ### number of cells in the entire data set .
    if data_dimension > 1000000:
        ### if data dimension exceeds 1 million, then reduce no of params
        no_iter=30
        early_stopping = 5
        test_size = 0.15
        max_iter = 4000
        max_estims = 400
    else:
        if orig_train.shape[0] <= 1000:
            no_iter=20
            test_size = 0.1
            max_iter = 2000
            max_estims = 200
        else:
            no_iter=30
            test_size = 0.1
            max_iter = 3000
            max_estims = 350
        early_stopping = 4
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
    ### Make target into a list so that we can uniformly process the target label
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
    ##### This is where we run the Traditional models to compare them to XGB #####
    start_time = time.time()
    ####################################################################################
    ##### Set up your Target Labels and Classes Properly Here #### Label Encoding #####
    #### This is for Classification Problems Only where you do Label Encoding of Target
    mldict = lambda: defaultdict(mldict)
    label_dict = mldict()
    first_time = True
    print('Train (Size: %d,%d) has %s with target: %s' %(train.shape[0],train.shape[1],model_label,target))
    ###### Now analyze what problem we have here ####
    try:
        modeltype = analyze_problem_type(train, target[0],verbose)
    except:
        print('Cannot find the Target variable in data set. Please check input and try again')
        return
    for each_target in target:
        c_params = dict()
        r_params = dict()
        if modeltype == 'Regression':
            scv = KFold(n_splits=n_splits, random_state=seed)
            eval_metric = 'rmse'
            objective = 'reg:linear'
            model_class = 'Regression'
            start_train = copy.deepcopy(orig_train)
        else:
            if len(np.unique(train[each_target])) == 2:
                model_class = 'Binary-Class'
            elif len(np.unique(train[each_target])) > 2:
                model_class = 'Multi-Class'
            else:
                print('Target label %s has less than 2 classes. Stopping' %each_target)
                return
            ### This is for Classification Problems Only ########
            print('Shuffling the data set before training')
            orig_train = orig_train.sample(frac=1.0, random_state=seed)
            start_train = copy.deepcopy(orig_train)
            scv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        if modeltype != 'Regression':
            rare_class = find_rare_class(orig_train[each_target].values,verbose=1)
            ### Perfrom Label Transformation only for Classification Problems ####
            classes = np.unique(orig_train[each_target])
            if first_time:
                print('Selecting %d-Class Classifier...' %len(classes))
                if hyper_param == 'GS':
                    print('    Using GridSearchCV for Hyper Parameter tuning...')
                else:
                    print('    Using RandomizedSearchCV for Hyper Parameter Tuning. This will take time...')
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
            if type(orig_train[each_target].values[0])==str or str(orig_train[each_target].dtype
                            )=='category' or sorted(np.unique(orig_train[each_target].values))[0] != 0:
                ### if the class is a string or if it has more than 2 classes, then use Factorizer!
                label_dict[each_target]['values'] = orig_train[each_target].values
                #### Factorizer is the easiest way to convert target in train and predictions in test
                #### This takes care of some classes that are present in train and not in predictions
                ### and vice versa. Hence it is better than Label Encoders which breaks when above happens.
                train_targ_categs = list(orig_train[each_target].value_counts().index)
                rare_class = find_rare_class(orig_train[each_target])
                if len(train_targ_categs) == 2:
                    majority_class = [x for x in train_targ_categs if x != rare_class]
                    dict_targ_all = {majority_class[0]: 0, rare_class: 1}
                else:
                    dict_targ_all = return_factorized_dict(train_targ_categs)
                start_train[each_target] = orig_train[each_target].map(dict_targ_all)
                label_dict[each_target]['dictionary'] = copy.deepcopy(dict_targ_all)
                label_dict[each_target]['transformer'] = dict([(v,k) for (k,v) in dict_targ_all.items()])
                label_dict[each_target]['classes'] = copy.deepcopy(train_targ_categs)
                label_dict[each_target]['class_nums'] = list(dict_targ_all.values())
                print('String or Multi Class target: %s transformed as follows: %s' %(each_target,dict_targ_all))
                rare_class = find_rare_class(start_train[each_target].values)
            else:
                ### Since the each_target here is already numeric, you don't have to modify it
                rare_class = find_rare_class(orig_train[each_target].values)
                label_dict[each_target]['values'] = orig_train[each_target].values
                label_dict[each_target]['classes'] = np.unique(orig_train[each_target].values)
                label_dict[each_target]['class_nums'] = np.unique(orig_train[each_target].values)
                label_dict[each_target]['transformer'] = []
                label_dict[each_target]['dictionary'] = dict(zip(classes,classes))
                print('    Target %s is already numeric. No transformation done.' %each_target)
    ###########################################################################################
    if orig_train.isnull().sum().sum() > 0:
        ### If there are missing values print it here ####
        top5 = orig_train.isnull().sum().sort_values(ascending=False).index.tolist()[:5]
        print('    Top columns in Train with missing values: %s' %(
                                        [x for x in top5 if orig_train[x].isnull().sum()>0]))
        print('         and their missing value totals: %s' %([orig_train[x].isnull().sum() for x in
                                                         top5 if orig_train[x].isnull().sum()>0]))
    ###########################################################################################
    ####  This is where we start doing the iterative hyper tuning parameters #####
    params_dict = defaultdict(list)
    accu_mean = []
    error_rate = []
    ######  This is where we do the training and hyper parameter tuning ########
    orig_preds = [x for x in list(orig_train) if x not in target]
    count = 0
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(orig_train[orig_preds], verbose)
    #####       Classify Columns   ########################################
    id_cols = var_df['id_vars']
    string_cols = var_df['nlp_vars']+var_df['date_vars']
    del_cols = var_df['cols_delete']
    factor_cols = var_df['factor_vars']
    continuous_vars = var_df['continuous_vars']+var_df['int_vars']+var_df['num_bool_vars']
    cat_vars = var_df['string_bool_vars']+var_df['discrete_string_vars']+var_df['cat_vars']+var_df['factor_vars']
    #######################################################################################
    preds = [x for x in orig_preds if x not in id_cols+del_cols+string_cols+target]
    if len(id_cols+del_cols+string_cols)== 0:
        print('    No variables removed since no ID or low-information variables found in data set')
    else:
        print('    %d variables removed since they were some ID or low-information variables'
                                %len(id_cols+del_cols+string_cols))
    ################## This is where we set the models ########################
    percent_num_vars = len(continuous_vars)/len(preds)
    if Boosting_Flag:
        if Boosting_Flag == 'CatBoost' or Boosting_Flag == 'Catboost' or Boosting_Flag == 'catboost':
            model_name = 'CatBoost'
            hyper_param = None
        elif percent_num_vars <= catboost_limit:
            ### If the percentage of num vars in data set is less than 40%, Use Cat Boost whiich is fast
            ####  when there are very few numeric vars in a data set. This limit is known as CatBoost _Limit
            model_name = 'CatBoost'
            hyper_param = None
        else:
            model_name = 'XGBoost'
    elif Boosting_Flag is None:
        model_name = 'Linear'
    else:
        model_name = 'Forests'
    if len(cat_vars) > 0:
        if len(factor_cols) > 0:
            cat_var_categs_list = []
            for cat_var in factor_cols:
                cat_var_categs_list.append(start_train[cat_var].value_counts().index.tolist())
            model_name = 'CatBoost'
            Boosting_Flag = 'CatBoost'
            one_hot_size = len(max(cat_var_categs_list, key = lambda i: len(i)))
            if one_hot_size == 0:
                one_hot_size = 100
            print('Changing model to CatBoost since No. of dummy variables to create in data set exceeds %d' %cat_code_limit)
        elif Boosting_Flag is None:
            ####### Test here if the number of Categories is very large whether you should use CatBoost or Linear models
            cat_var_categs_list = []
            for cat_var in cat_vars:
                cat_var_categs_list.append(start_train[cat_var].value_counts().index.tolist())
            if len(sum(cat_var_categs_list, [])) > cat_code_limit and model_name is None:
                ### If data set has too many categorical variables resulting in 100 or more dummy variables, then swtitch to CatBoost
                ### A Linear Model such as Logistic Regression will struggle with 100 or more dummy variables, esp if sample size is large.
                ### However CatBoost is very fast in such Category-heavy data sets and hence is a better choice for such data sets.
                model_name = 'CatBoost'
                Boosting_Flag = 'CatBoost'
                one_hot_size = len(max(cat_var_categs_list, key = lambda i: len(i)))
                if one_hot_size == 0:
                    one_hot_size = 100
                print('Changing model to CatBoost since No. of dummy variables to create in data set exceeds %d' %cat_code_limit)
    #####   Set the Scoring Parameters here based on each model and preferences of user ##############
    if model_name == 'CatBoost':
        if model_class == 'Binary-Class':
            catboost_scoring = 'Accuracy'
        elif model_class == 'Multi-Class':
            catboost_scoring = 'AUC'
        else:
            catboost_scoring = 'RMSE'
    ######  Fill Missing Values, Scale Data and Classify Variables Here ###
    cat_vars_encoded = []
    numvars = []
    if len(preds) > 0:
        dict_train = {}
        for col in preds:
            if start_train[col].dtype == object:
                col_le = col + encoded
                if model_name == 'CatBoost':
                    ######  If Model is CatBoost, it handles Category Variables automatically except NaN.
                    #####  So we will fill NaN values in Category Variables with the word "missing"
                    if orig_train[col].isnull().sum() > 0:
                        start_train[col] = start_train[col].fillna("NA")
                    start_train[col_le] = start_train[col].astype('category')
                    if type(orig_test) != str:
                        if orig_test[col].isnull().sum() > 0:
                            start_test[col] = start_test[col].fillna("NA")
                        start_test[col_le] = start_test[col].astype('category')
                else:
                    #####  For All other models except CatBoost, we need special methods to handle Cat variables
                    ####  This is the easiest way to label encode object variables in both train and test
                    #### This takes care of some categories that are present in train and not in test
                    ###     and vice versa
                    train_categs = list(pd.unique(start_train[col].values))
                    if type(orig_test) != str:
                        test_categs = list(pd.unique(start_test[col].values))
                        categs_all = train_categs+test_categs
                    else:
                        categs_all = copy.deepcopy(train_categs)
                    dict_all =  return_factorized_dict(categs_all)
                    try:
                        ### First try Ordinal Encoder which works most times. But it has a problem...
                        ### when Test has diff categories than train, it blows up. So you need exception handling.
                        oe = LabelEncoder()
                        if Boosting_Flag is None:
                            start_train[col_le] = oe.fit_transform(start_train[col].values.reshape(-1,1))
                            if type(orig_test) != str:
                                start_test[col_le] = oe.transform(start_test[col].values.reshape(-1,1))
                        else:
                            start_train[col] = oe.fit_transform(start_train[col].values.reshape(-1,1))
                            if type(orig_test) != str:
                                start_test[col] = oe.transform(start_test[col].values.reshape(-1,1))
                    except:
                        #### In case the above transform errors, you lose the original values.
                        ####   Hence this next statement restores the original values it lost.
                        start_train[col] = orig_train[col].values
                        ### This is where we handle all exceptions by combining all categories from train and test
                        ### Then we apply factorize using these categories. This works on transforming train and test.
                        if Boosting_Flag is None:
                            start_train[col_le] = start_train[col].map(dict_all)
                            if type(orig_test) != str:
                                #### In case the above transform errors, you lose the original values.
                                ####   Hence this next statement restores the original values it lost.
                                start_test[col_le] = orig_test[col].values
                                start_test[col_le] = start_test[col].map(dict_all)
                        else:
                            start_train[col] = start_train[col].map(dict_all)
                            if type(orig_test) != str:
                                #### In case the above transform errors, you lose the original values.
                                ####   Hence this next statement restores the original values it lost.
                                start_test[col] = orig_test[col].values
                                start_test[col] = start_test[col].map(dict_all)
                cat_vars.append(col)
                cat_vars_encoded.append(col_le)
            elif start_train[col].dtype == int:
                numvars.append(col)
                ### if there are integer variables, don't scale them. Leave them as is.
                fill_num = start_train[col].min() - 1
                if start_train[col].isnull().sum() > 0:
                    start_train[col] = start_train[col].fillna(fill_num)
                if type(orig_test) != str:
                    if start_test[col].isnull().sum() > 0:
                        start_test[col] = start_test[col].fillna(fill_num)
                pass
            elif col in factor_cols:
                if orig_train[col].isnull().sum() > 0:
                    start_train[col] = start_train[col].fillna("NA")
                if type(orig_test) != str:
                    if orig_test[col].isnull().sum() > 0:
                        start_test[col] = start_test[col].fillna("NA")
            else:
                numvars.append(col)
                ### for all numeric variables, fill missing values with 1 less than min.
                fill_num = start_train[col].min() - 1
                if start_train[col].isnull().sum() > 0:
                    start_train[col] = start_train[col].fillna(fill_num)
                if type(orig_test) != str:
                    if start_test[col].isnull().sum() > 0:
                        start_test[col] = start_test[col].fillna(fill_num)
                ##### DO SCALING ON TRAIN HERE ############
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        start_train[col] = SS.fit_transform(start_train[col].values.reshape(-1,1))
                    except:
                        start_train.loc[start_train[col]==np.inf,col]=0
                        start_train[col] = SS.fit_transform(start_train[col].values.reshape(-1,1))
                ##### DO SCALING ON TEST HERE ############
                if type(orig_test) != str:
                    try:
                        start_test[col] = SS.transform(start_test[col].values.reshape(-1,1))
                    except:
                        start_test.loc[start_test[col]==np.inf,col]=0
                        start_test[col] = SS.fit_transform(start_test[col].values.reshape(-1,1))
        ################   THE BELOW STEP IS VERY IMPORTANT #############################################
        print('Completed Label Encoding, Missing Value Imputing and Scaling of data without errors.')
        if start_train[preds].isnull().sum().sum() == 0:
            print('    No Missing values in Train')
        if type(orig_test) != str:
            if start_test[preds].isnull().sum().sum() > 0:
                print('    Test data still has some missing values. Continuing...')
            else:
                print('    Test data has no missing values')
        else:
            print('Train data set still has some missing values. Stopped. Fix it')
            return
    else:
        print('    No predictors in your data set. Please check your data and try again')
        return
    ### Make sure you remove variables that are highly correlated within data set first ############
    rem_vars = left_subtract(preds,numvars)
    if len(numvars) > 0 and feature_reduction:
        numvars = remove_variables_using_fast_correlation(start_train[preds],numvars,corr_limit,verbose)
    ### Reduced Preds are now free of correlated variables and hence can be used for Poly adds
    red_preds = rem_vars + numvars
    #### You need to save a copy of this red_preds so you can later on create a start_train
    ####     with it after each_target cycle is completed. Very important!
    orig_red_preds = copy.deepcopy(red_preds)
    for each_target in target:
        print('\nData Ready for Modeling with %s as Target...' %each_target)
        ########   D E F I N I N G   N E W  T R A I N and N E W   T E S T here #########################
        ####  This is where we set the orig train data set with multiple labels to the new start_train
        ####     start_train has the new features added or reduced with the multi targets in one cycle
        ###      That way, we start each train with one target, and then reset it with multi target
        train = start_train[[each_target]+red_preds]
        if type(orig_test) != str:
            test = start_test[red_preds]
            print('    Feature Reduction begins: currently %d predictors' %(
                                                    len(red_preds)))
        ###### Add Polynomial Variables and Interaction Variables to Train ######
        if Add_Poly >= 1:
            print('\nCAUTION: Adding 2nd degree Polynomial & Interaction Variables may result in Overfitting!')
            ## Since the data is already scaled, we set scaling to None here ##
            ### For train data we have to set the fit_flag to True   ####
            if len(numvars) > 1:
                train_sel, lm, train_red,md,fin_xvars = add_poly_vars_select(train,numvars,
                                            each_target,modeltype,poly_degree,Add_Poly,md='',
                                                                scaling='None',
                                                                fit_flag=True,verbose=verbose)
                if len(train_sel) > len(numvars):
                    #### Do the next step only if new variables were added #############
                    train = train_red.join(train[rem_vars+[each_target]])
                    red_preds = list(OrderedDict.fromkeys(train_sel+rem_vars))
                    if type(test) != str:
                    ######### Add Polynomial and Interaction variables to Test ################
                    ## Since the data is already scaled, we set scaling to None here ##
                    ### For Test data we have to set the fit_flag to False   ####
                        _, _, test_x_vars,_,_ = add_poly_vars_select(test,numvars,each_target,
                                                              modeltype,poly_degree,Add_Poly,md,
                                                              scaling='None', fit_flag=False,
                                                               verbose=verbose)
                        test_red = test_x_vars[fin_xvars]
                        #### test_red contain xvars with orig and poly/intxn variables
                        ###  we need to convert it into orig text variables
                        test_red.columns = train_sel
                        test = test_red[train_sel].join(test[rem_vars])
                else:
                    ####  NO new variables were added. so we can skip the rest of the stuff now ###
                    print('    No new variable was added by polynomial features...')
            else:
                print('\nAdding Polynomial vars ignored since no numeric vars in data')
                train_sel = copy.deepcopy(numvars)
        else:
            ### if there are no Polynomial vars, then all numeric variables are selected
            train_sel = copy.deepcopy(numvars)
        #########     SELECT IMPORTANT FEATURES HERE   #############################
        if feature_reduction:
            if len(numvars) == 0:
                ####  In case there are no numeric values in data, then use Chi-Square function to reduce features
                #### Heuristic: max number of features should be 50% of total features ####
                max_feats = int(0.5*len(red_preds))
                ### Since SelectKBest does not accept any negative values in Chi-Sq function, check for neg values ##
                ###  Only in case of missing values we have earlier substituted -1 so that needs to be replaced with
                ####   the max value of each column plus 1 to make sure there are no negative values in data frame.##
                if model_name != 'CatBoost':
                    for col_cat in red_preds:
                        train[col_cat] = np.where(train[col_cat]<0, train[col_cat].max()+1, train[col_cat])
                    if modeltype == 'Regression':
                        sel_function = mutual_info_regression
                    else:
                        #sel_function = mutual_info_classif
                        sel_function = chi2
                    fs = SelectKBest(score_func=sel_function, k=max_feats)
                    fs.fit(train[red_preds], train[each_target])
                    cols_index = fs.get_support(indices=True)
                    important_features = np.array(red_preds)[cols_index]
                    important_features= important_features.tolist()
                    num_vars = copy.deepcopy(numvars)
                    print('Selected %d features using Linear feature selection methods' %len(important_features))
                else:
                    print('No numeric features in data set. Hence CatBoost used: no feature reduction needed.')
                    important_features = copy.deepcopy(red_preds)
                    num_vars = copy.deepcopy(numvars)
            else:
                if model_name != 'CatBoost':
                    important_features,num_vars = find_top_features_xgb(train,red_preds,train_sel,
                                                         each_target,
                                                     modeltype,corr_limit,verbose)
                    print('Selected %d features using Boosted feature selection methods' %len(important_features))
                else:
                    print('In CatBoost, no feature reduction is done. All %d features are used in model' %len(red_preds))
                    important_features = copy.deepcopy(red_preds)
                    num_vars = copy.deepcopy(numvars)
        else:
            print('No feature reduction done. All %d features to be used in model' %len(red_preds))
            important_features = copy.deepcopy(red_preds)
            num_vars = copy.deepcopy(numvars)
        ###################################################################################################
        if len(important_features) == 0:
            print('No important features found. Using all input features...')
            important_features = copy.deepcopy(red_preds)
        if verbose and len(important_features) <= 30:
            print('    Features list: %s' %important_features)
        #############  C R E A T I N G  D U M M Y   V A R I A B L E S FOR LINEAR MODELS ONLY  ############
        if Boosting_Flag is None and len(cat_vars) > 0:
            if not feature_reduction:
                LE_features = copy.deepcopy(cat_vars_encoded)
            else:
                LE_features = [x for x in important_features if x.endswith(encoded) ]
            LE_features = [x[:-14] for x in LE_features]
            dummy_vars = [x for x in LE_features if x in cat_vars]
            print('    Creating %d  dummy variables from %d categorical features for Linear Models' %(len(sum(cat_var_categs_list, [])),len(dummy_vars)))
            all_other_vars = [x for x in LE_features if x not in dummy_vars]
            orig_train['SOURCE_DATA_SPLIT'] = 'Train'
            if type(orig_test) != str:
                orig_test['SOURCE_DATA_SPLIT'] = 'Test'
                dummy_df = orig_train[dummy_vars+['SOURCE_DATA_SPLIT']].append(orig_test[dummy_vars+['SOURCE_DATA_SPLIT']])
            else:
                dummy_df = orig_train[dummy_vars+['SOURCE_DATA_SPLIT']]
            dummy_df = pd.get_dummies(dummy_df)
            dummy_train = dummy_df.loc[dummy_df['SOURCE_DATA_SPLIT_Train']==1]
            if type(orig_test) != str:
                dummy_test = dummy_df.loc[dummy_df['SOURCE_DATA_SPLIT_Test']==1]
                dummy_train.drop(['SOURCE_DATA_SPLIT_Test','SOURCE_DATA_SPLIT_Train'],axis=1,inplace=True)
                dummy_test.drop(['SOURCE_DATA_SPLIT_Test','SOURCE_DATA_SPLIT_Train'],axis=1,inplace=True)
            else:
                dummy_train.drop(['SOURCE_DATA_SPLIT_Train'],axis=1,inplace=True)
            train = train[[each_target]+all_other_vars].join(dummy_train)
            if type(orig_test) != str:
                test = test[all_other_vars].join(dummy_test)
            #### IF YOU HAVE DONE ONE-HOT-ENCODING, YOU WILL HAVE ADDITIONAL COLUMNS SO RE-DO PREDS HERE #####
            new_preds = [x for x in list(train) if x not in [each_target]]
            preds = [x for x in new_preds if x not in id_cols+del_cols+string_cols]
            important_features = copy.deepcopy(preds)
        else:
            ########  ADDING T A R G E T  V A R  TO   R E S T  OF  T R A I N FOR Non-Linear MODELS  #######
            train = train[important_features+[each_target]]
            if type(orig_test) != str:
                test = test[important_features]
        ##############          F E A T U R E   E N G I N E E R I N G  S T A R T S  N O W    ##############
        ######    From here on we do some Feature Engg using Target Variable with Data Leakage ############
        ###   To avoid Model Leakage, we will now split the Data into Train and CV so that Held Out Data
        ##     is Pure and is unadulterated by learning from its own Target. This is known as Data Leakage.
        ###################################################################################################
        print('Starting Feature Engineering now...')
        X = train[important_features]
        y = train[each_target]
        train_num = int((1-test_size)*train.shape[0])
        part_train = train[:train_num]
        part_cv = train[train_num:]
        ############   Add Entropy Binning of Continuous Variables Here ##############################
        saved_important_features = copy.deepcopy(important_features)  ### these are original features without '_bin' added
        saved_num_vars = copy.deepcopy(num_vars)  ### these are original numeric features without '_bin' added
        if len(saved_num_vars) > 0:
            #### Do binning only when there are numeric features ####
            part_train, num_vars, important_features, part_cv = add_entropy_binning(part_train, each_target, num_vars,
                                                                 important_features, part_cv, modeltype,Binning_Flag)
        ### Now we add another Feature tied to KMeans clustering using Predictor and Target variables ###
        X_train, X_cv, y_train, y_cv = part_train[important_features],part_cv[important_features
                                                   ],train[each_target][:train_num],train[each_target][train_num:]
        if KMeans_Featurizer and len(num_vars) > 0:
            ### DO KMeans Featurizer only if there are numeric features in the data set!
            print('    Adding one Feature named "cluster" using KMeans_Featurizer...')
            if modeltype != 'Regression':
                ### If it is Classification, you can specify the number of clusters same as classes
                X_train, X_cv = Transform_KM_Features(X_train, y_train, X_cv, len(classes))
            else:
                ### If it is Regression, you don't have to specify the number of clusters
                X_train, X_cv = Transform_KM_Features(X_train, y_train, X_cv)
            #### Since this is returning the each_target in X_train, we need to drop it here ###
            X_train.drop(each_target,axis=1,inplace=True)
        important_features = [x for x in list(X_train) if x not in [each_target]]
        ######### This is where you do Stacking of Multi Model Results into One Column ###
        if Stacking_Flag:
            #### In order to join, you need X_train to be a Pandas Series here ##
            print('    Adding Features using Stacking models...')
            print('        Caution: Stacking can produce Highly Overfit models on Training Data...')
            ### In order to avoid overfitting, we are going to learn from a small sample of data
            ### That is why we are using X_cv to train on and using it to predict on X_train!
            addcol, stacks1 = QuickML_Stacking(X_train,y_train,X_train,
                          modeltype, Boosting_Flag, scoring_parameter,verbose)
            addcol, stacks2 = QuickML_Stacking(X_train,y_train,X_cv,
                          modeltype, Boosting_Flag, scoring_parameter,verbose)
            ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
            X_train = X_train.join(pd.DataFrame(stacks1,index=X_train.index,
                                              columns=addcol))
            ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
            X_cv = X_cv.join(pd.DataFrame(stacks2,index=X_cv.index,
                                              columns=addcol))
            print('    Added %d Stacking feature(s) to training data' %len(addcol))
            ######  We make sure that we remove any new features that are highly correlated ! #####
            #addcol = remove_variables_using_fast_correlation(X_train,addcol,corr_limit,verbose)
            important_features += addcol
        #### This is where we divide train and test into Train and CV Test Sets #################
        #### Remember that the next 2 lines are crucial: if X and y are dataframes, then predict_proba
        ###     will have to also predict on dataframes. So don't confuse values with df's.
        ##      Be consistent with XGB. That's the best way.
        cat_vars_encoded = [x for x in cat_vars_encoded if x in X_train.columns]
        print('Number of Rows in Train data set = %d' %X_train.shape[0])
        print('    Number of Features in Train data set = %d' %X_train.shape[1])
        data_dim = X_train.shape[0]*X_train.shape[1]
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
            ####      HYPER PARAMETERS FOR TUNING ARE SETUP HERE      ###
            if hyper_param == 'GS':
                r_params = {
                        "Forests": {
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                "max_depth": [2, 5, 10],
                                #"criterion" : ['mse','mae'],
                                },
                        "Linear": {
                            'alpha': np.logspace(-3,3),
                                },
                        "XGBoost": {
                                'max_depth': [2,5,10],
                                'gamma': [0,1,2,4,8,16,32],
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                },
                        "CatBoost": {
                                'learning_rate': [0.01,  0.05,  0.1],
                                },
                        }
            else:
                r_params = {
                        "Forests": {
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                "max_depth": [2, 5, 10],
                                #"min_samples_leaf": np.linspace(2, 50, 20, dtype = "int"),
                                #"criterion" : ['mse','mae'],
                                },
                        "Linear": {
                            'alpha': np.logspace(-3,3),
                                },
                        "XGBoost": {
                                'max_depth': [2,5,10],
                                'gamma': [0,1,2,4,8,16,32],
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                },
                        "CatBoost": {
                                'learning_rate': [0.01,  0.05,  0.1],
                                },
                        }
            if Boosting_Flag:
                if model_name == 'CatBoost':
                    xgbm = CatBoostRegressor(verbose=0,n_estimators=max_estims,random_state=99,
                            one_hot_max_size=one_hot_size,
                            loss_function='RMSE', eval_metric='RMSE',
                            subsample=0.7,bootstrap_type='Bernoulli',
                           early_stopping_rounds=25,boosting_type='Plain')
                else:
                    xgbm = XGBRegressor(seed=seed,n_jobs=-1,random_state=seed,subsample=subsample,
                                         colsample_bytree=col_sub_sample,
                                        objective=objective)
            elif Boosting_Flag is None:
                xgbm = Lasso(max_iter=max_iter,random_state=seed)
            else:
                xgbm = ExtraTreesRegressor(
                                **{
                                'bootstrap': True, 'n_jobs': -1, 'warm_start': False,
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
                                            'max_depth': [2,5,10],
                                            'gamma': [0,1,2,4,8,16,32],
                                'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                    "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                    }
                c_params["CatBoost"] = {
                                'learning_rate': [0.01,  0.05,  0.1],
                                }
                if not Imbalanced_Flag:
                    c_params['Linear'] = {
                                'C': np.linspace(0.01,100,100),
                                'solver' :[ 'lbfgs','saga', 'liblinear','newton-cg']
                                    }
                else:
                    c_params['Linear'] = {
                                'C': np.linspace(0.01,100,100),
                                'solver' :[ 'lbfgs' ],#'saga', 'liblinear','newton-cg'
                                'class_weight':[None,'balanced'],
                                    }
                c_params["Forests"] = {
                    ##### I have selected these to avoid Overfitting which is a problem for small data sets
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    "criterion":['gini','entropy'],
                                            }
            else:
                c_params["CatBoost"] = {
                                'learning_rate': [0.01,  0.05,  0.1],
                                }
                c_params['XGBoost'] = {
                                                'max_depth': [2,5,10],
                                                'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                                'gamma': [0,1,2,4,8,16,32],
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                    }
                C = np.linspace(0.01,100,100)
                if not Imbalanced_Flag:
                    c_params['Linear'] = {
                            'C': C,
                                'solver' :[ 'lbfgs','saga', 'lbfgs', 'liblinear','newton-cg']
                                    }
                else:
                    c_params['Linear'] = {
                            'C': C,
                                'solver' :[ 'lbfgs'],#'saga', 'lbfgs', 'liblinear','newton-cg'
                                'class_weight':[None,'balanced'],
                                    }
                if not Imbalanced_Flag:
                    c_params["Forests"] = {
                    ##### I have selected these to avoid Overfitting which is a problem for small data sets
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    #"min_samples_leaf": np.linspace(2, 50, 20, dtype = "int"),
                                    "criterion":['gini','entropy'],
                                    #'max_features': ['log', "sqrt"] ,
                                            }
                else:
                    c_params["Forests"] = {
                    ##### I have selected these to avoid Overfitting which is a problem for small data sets
                                "n_estimators" : np.linspace(100, max_estims, n_steps, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    #"min_samples_leaf": np.linspace(2, 50, 20, dtype = "int"),
                                    "criterion":['gini','entropy'],
                                    'class_weight':[None,'balanced']
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
                elif scoring_parameter == 'log_loss' or scoring_parameter == 'logloss':
                    scoring_parameter = 'neg_log_loss'
                    logloss_scorer = make_scorer(gini_log_loss, greater_is_better=False, needs_proba=False)
                    scorer =logloss_scorer
                elif scoring_parameter=='balanced_accuracy' or scoring_parameter=='balanced-accuracy' or scoring_parameter=='average_accuracy':
                    try:
                        bal_accuracy_scorer = make_scorer(gini_bal_accuracy, greater_is_better=True,
                                                                      needs_proba=False)
                    except:
                        bal_accuracy_scorer = make_scorer(gini_accuracy, greater_is_better=True,
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
                else:
                    f1_scorer = make_scorer(gini_f1, greater_is_better=True, needs_proba=False,
                            pos_label=rare_class)
                    scorer = f1_scorer
                ### DO NOT USE NUM CLASS WITH BINARY CLASSIFICATION ######
                if Boosting_Flag:
                    if model_name == 'CatBoost':
                        xgbm =  CatBoostClassifier(verbose=0,n_estimators=max_estims,
                            random_state=99,one_hot_max_size=one_hot_size,
                            loss_function='Logloss', eval_metric='AUC',
                            subsample=0.7,bootstrap_type='Bernoulli',
                           early_stopping_rounds=25,boosting_type='Plain')
                    else:
                        xgbm = XGBClassifier(seed=seed,n_jobs=-1, random_state=seed,subsample=subsample,
                                         colsample_bytree=col_sub_sample,
                                     objective=objective)
                elif Boosting_Flag is None:
                    #### I have set the Verbose to be False here since it produces too much output ###
                    xgbm = LogisticRegression(random_state=seed,verbose=False,n_jobs=-1,tol=0.01,
                                             warm_start=False, max_iter=max_iter)
                else:
                    xgbm = ExtraTreesClassifier(
                                **{
                                'bootstrap': True, 'n_jobs': -1, 'warm_start': False,
                                'random_state':seed,'min_samples_leaf':2,'oob_score':True,
                                'max_features': "sqrt"
                                })
            else:
                #####   This is for MULTI Classification ##########################
                objective = 'multi:softmax'
                eval_metric = "mlogloss"
                if scoring_parameter == 'gini':
                    gini_scorer = make_scorer(gini_sklearn, greater_is_better=False, needs_proba=True)
                    scorer = gini_scorer
                elif scoring_parameter=='balanced_accuracy' or scoring_parameter=='balanced-accuracy' or scoring_parameter=='average_accuracy':
                    try:
                        bal_accuracy_scorer = make_scorer(gini_bal_accuracy, greater_is_better=True,
                                                                      needs_proba=False)
                    except:
                        bal_accuracy_scorer = make_scorer(gini_accuracy, greater_is_better=True,
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
                if Boosting_Flag:
                    # Create regularization hyperparameter distribution using uniform distribution
                    if hyper_param == 'GS':
                        c_params['XGBoost'] = {
                                            'max_depth': [2,5,10],
                                            'gamma': [0,1,2,4,8,16,32],
                                        'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                "n_estimators" : np.linspace(100, max_estims, 4, dtype = "int"),
                                    }
                    else:
                        c_params['XGBoost'] = {
                                            'learning_rate': [0.10, 0.2, 0.30, 0.4, 0.5],
                                                'max_depth': [2,5,10],
                                                'gamma': [0,1,2,4,8,16,32],
                                "n_estimators" : np.linspace(100, max_estims, 4, dtype = "int"),
                                    }
                    if model_name == 'CatBoost':
                        xgbm =  CatBoostClassifier(verbose=0,n_estimators=max_estims,
                                random_state=99,one_hot_max_size=one_hot_size,
                                loss_function='MultiClass', eval_metric='AUC',
                                subsample=0.7,bootstrap_type='Bernoulli',
                               early_stopping_rounds=25,boosting_type='Plain')
                    else:
                        xgbm = XGBClassifier(seed=seed,n_jobs=-1, random_state=seed,subsample=subsample,
                                         colsample_bytree=col_sub_sample,
                                         num_class= len(classes),
                                     objective=objective)
                elif Boosting_Flag is None:
                    if hyper_param == 'GS':
                        c_params['Linear'] = {
                            'C': np.linspace(0.01,1000,50),
                            'solver' :[ 'lbfgs'],# 'saga', 'newton-cg','lbfgs', 'liblinear',
                            'multi_class': ['ovr','multinomial'],
                                    }
                    else:
                        # Create regularization hyperparameter distribution with 50 C values ####
                        C = np.linspace(0.01,1000,50)
                        c_params['Linear'] = {
                                            'C': C,
                                            'solver' :['lbfgs'],# newton-cg,'lbfgs', 'saga'],
                                            'multi_class': ['ovr','multinomial'],
                                        }
                    #### I have set the Verbose to be False here since it produces too much output ###
                    xgbm = LogisticRegression(random_state=seed,verbose=False,n_jobs=-1,
                                              max_iter=max_iter, warm_start=False,
                                              )
                else:
                    if hyper_param == 'GS':
                        c_params["Forests"] = {
                        ##### I have selected these to avoid Overfitting which is a problem for small data sets
                                "n_estimators" : np.linspace(100, max_estims, 4, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    "criterion":['gini','entropy'],
                                            }
                    if not Imbalanced_Flag:
                        c_params["Forests"] = {
                        #####   I have set these to avoid OverFitting which is a problem for small data sets ###
                                "n_estimators" : np.linspace(100, max_estims, 4, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    #"min_samples_leaf": np.linspace(2, 50, 20, dtype = "int"),
                                    "criterion":['gini','entropy'],
                                                }
                    else:
                        c_params["Forests"] = {
                        #####   I have set these to avoid OverFitting which is a problem for small data sets ###
                                "n_estimators" : np.linspace(100, max_estims, 4, dtype = "int"),
                                    "max_depth": [2, 5, 10],
                                    #"min_samples_leaf": np.linspace(2, 50, 20, dtype = "int"),
                                    "criterion":['gini','entropy'],
                                    'class_weight':[None,'balanced']
                                                }
                    xgbm = ExtraTreesClassifier(bootstrap=True, oob_score=True,warm_start=False,
                                            n_estimators=100,max_depth=3,
                                            min_samples_leaf=2,max_features='auto',
                                          random_state=seed,n_jobs=-1)
        ######   Now do RandomizedSearchCV  using # Early-stopping ################
        if modeltype == 'Regression':
            #scoreFunction = {"mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error"}
            #### I have set the Verbose to be False here since it produces too much output ###
            if hyper_param == 'GS':
                #### I have set the Verbose to be False here since it produces too much output ###
                model = GridSearchCV(xgbm,param_grid=r_params[model_name],
                                               scoring = scorer,
                                               n_jobs=-1,
                                               cv = scv,
                                               return_train_score=True,
                                                verbose=0)
            elif hyper_param == 'RS':
                model = RandomizedSearchCV(xgbm,
                                               param_distributions = r_params[model_name],
                                               n_iter = no_iter,
                                               scoring = scorer,
                                               refit = "rmse",
                                               return_train_score = True,
                                               random_state = seed,
                                               cv = scv,
                                               n_jobs=-1,
                                                verbose = 0)
            else:
                #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
                model = copy.deepcopy(xgbm)
        else:
            if hyper_param == 'GS':
                #### I have set the Verbose to be False here since it produces too much output ###
                model = GridSearchCV(xgbm,param_grid=c_params[model_name],
                                               scoring = scorer,
                                               n_jobs=-1,
                                               cv = scv,
                                               return_train_score = True,
                                                 verbose=0)
            elif hyper_param == 'RS':
                #### I have set the Verbose to be False here since it produces too much output ###
                model = RandomizedSearchCV(xgbm,
                                               param_distributions = c_params[model_name],
                                               n_iter = no_iter,
                                               scoring = scorer,
                                               refit = "precision",
                                               return_train_score = True,
                                               random_state = seed,
                                               n_jobs=-1,
                                               cv = scv,
                                                verbose = 0)
            else:
                #### CatBoost does not need Hyper Parameter tuning => it's great out of the box!
                model = copy.deepcopy(xgbm)
        #trains and optimizes the model
        eval_set = [(X_cv, y_cv)]
        print('Finding Best Model and Hyper Parameters for Target: %s...' %each_target)
        ##### Here is where we put the part_train and part_cv together ###########
        if modeltype != 'Regression':
            ### Do this only for Binary Classes and Multi-Classes, both are okay
            baseline_accu = 1-(train[each_target].value_counts(1).sort_values())[rare_class]
            print('    Baseline Accuracy Needed for Model = %0.2f%%' %(baseline_accu*100))
        print()
        if modeltype == 'Regression':
            if Boosting_Flag:
                if model_name == 'CatBoost':
                    data_dim = data_dim*one_hot_size/len(preds)
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/200000.))
                else:
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/20000.))
            elif Boosting_Flag is None:
                print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/50000.))
            else:
                print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/40000.))
        else:
            if hyper_param == 'GS':
                if Boosting_Flag:
                    if model_name == 'CatBoost':
                        data_dim = data_dim*one_hot_size/len(preds)
                        print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/300000.))
                    else:
                        print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/30000.))
                elif Boosting_Flag is None:
                    #### A Linear model is usually the fastest ###########
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/50000.))
                else:
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/10000.))
            else:
                if Boosting_Flag:
                    if model_name == 'CatBoost':
                        data_dim = data_dim*one_hot_size/len(preds)
                        print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/300000.))
                    else:
                        print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/30000.))
                elif Boosting_Flag is None:
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/50000.))
                else:
                    print('Using %s Model, Estimated Training time = %0.1f mins' %(model_name,data_dim/20000.))
        ##### Since we are using Multiple Models each with its own quirks, we have to make sure it is done this way
        print('    Number of Rows in held-out CV set = %d' %X_cv.shape[0])
        ##### ############      TRAINING MODEL FIRST TIME WITH X_TRAIN AND TESTING ON X_CV ############
        model_start_time = time.time()
        if modeltype != 'Regression':
            if Imbalanced_Flag:
                ######   This is for Imbalanced Classification tasks ##############
                try:
                    print('\nImbalanced Class Training using Majority Class Downsampling method...')
                    #### The d_model is the downsampled model Trained on downsampled data sets. ####
                    d_model = downsampling_with_model_training(X_train,y_train,eval_set,model,
                                           Boosting_Flag, eval_metric,
                                           modeltype,no_training=False,
                                           minority_class=rare_class, verbose=verbose)
                    if not isinstance(d_model, str):
                        #### If d_model succeeds, it will be used to get the best score and can become model again ##
                        if hyper_param == 'RS' or hyper_param == 'GS':
                            best_score = d_model.best_score_
                        else:
                            if model_class == 'Binary-Class':
                                best_score = model.best_score_['validation'][catboost_scoring]
                            else:
                                best_score = model.best_score_['validation']['AUC:type=Mu']
                        model = copy.deepcopy(d_model)
                    else:
                        #### If d_model failed, it will just be an empty string, so you try the regular model ###
                        print('Error in training Imbalanced model first time. Trying regular model..')
                        Imbalanced_Flag = False
                        if Boosting_Flag:
                            if model_name == 'XGBoost':
                                #### Set the Verbose to 0 since we don't want too much output ##
                                model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                                    eval_metric=eval_metric,eval_set=eval_set,verbose=0)
                            else:
                                try:
                                    model.fit(X_train, y_train, cat_features=cat_vars,eval_set=(X_cv,y_cv))
                                except:
                                    model.fit(X_train, y_train, cat_features=cat_vars)
                        else:
                                model.fit(X_train, y_train)
                        if hyper_param == 'RS' or hyper_param == 'GS':
                            best_score = model.best_score_
                        else:
                            if model_class == 'Binary-Class':
                                best_score = model.best_score_['validation'][catboost_scoring]
                            else:
                                best_score = model.best_score_['validation']['AUC:type=Mu']
                except:
                    print('Error in training Imbalanced model first time. Trying regular model..')
                    Imbalanced_Flag = False
            else:
                ######   This is for Regular Classification tasks ##############
                try:
                    if Boosting_Flag:
                        if model_name == 'XGBoost':
                            #### Set the Verbose to 0 since we don't want too much output ##
                            model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                                    eval_metric=eval_metric,eval_set=eval_set,verbose=0)
                        else:
                            try:
                                model.fit(X_train, y_train, cat_features=cat_vars,eval_set=(X_cv,y_cv))
                            except:
                                model.fit(X_train, y_train, cat_features=cat_vars)
                    else:
                        model.fit(X_train, y_train)
                    if hyper_param == 'RS' or hyper_param == 'GS':
                        best_score = model.best_score_
                    else:
                        if model_class == 'Binary-Class':
                            best_score = model.best_score_['validation'][catboost_scoring]
                        else:
                            best_score = model.best_score_['validation']['AUC:type=Mu']
                except:
                    print('Training regular model first time is Erroring: Check if your Input is correct...')
                    return
        else:
            ########### This is for Regression Model Training ###################
            try:
                if Boosting_Flag:
                    if model_name == 'XGBoost':
                        #### Set the Verbose to 0 since we don't want too much output ##
                        model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                            eval_metric=eval_metric,eval_set=eval_set,verbose=0)
                    else:
                        try:
                            model.fit(X_train, y_train, cat_features=cat_vars,eval_set=(X_cv,y_cv))
                        except:
                            model.fit(X_train, y_train, cat_features=cat_vars)
                else:
                    model.fit(X_train, y_train)
                if hyper_param == 'RS' or hyper_param == 'GS':
                    best_score = model.best_score_
                else:
                    best_score = model.best_score_['validation'][catboost_scoring]
            except:
                print('Training regular model is Erroring: Check if your Input Data is in correct Format...')
                return
        print('Model Training time taken in seconds = %0.0f' %(time.time()-model_start_time))
        ##   TRAINING OF MODELS COMPLETED. NOW GET METRICS on CV DATA ################
        if modeltype != 'Regression':
            if scoring_parameter == 'logloss' or scoring_parameter == 'neg_log_loss' :
                print('    Hyper Tuned %s = %0.4f' %(scoring_parameter, best_score))
            elif scoring_parameter == '':
                print('    Hyper Tuned  %s = %0.1f%%' %('Accuracy', best_score*100))
            else:
                print('    Hyper Tuned  %s = %0.1f%%' %(scoring_parameter, best_score*100))
        else:
            if scoring_parameter == '':
                print('    Hyper Tuned %s Score = %0.4f' %('RMSE', best_score))
            else:
                print('    Hyper Tuned %s Score = %0.4f' %(scoring_parameter, best_score))
        #### We now need to set the Best Parameters, Fit the Model on Full X_train and Predict on X_cv
        ### Find what the order of best params are and set the same as the original model ###
        if hyper_param == 'RS' or hyper_param == 'GS':
            print('Model Best Parameters = %s' %model.best_params_)
        else:
           print('%s Model Best Parameters = %s' %(model_name, model.get_all_params()))
        gs = copy.deepcopy(model)
        if hyper_param == 'RS' or hyper_param == 'GS':
            model = gs.best_estimator_
        ### Make sure you set this flag as False so that when ensembling is completed, this flag is True ##
        performed_ensembling = False
        if modeltype != 'Regression':
            m_thresh = 0.5
            y_proba = model.predict_proba(X_cv)
            y_pred = model.predict(X_cv)
            if len(classes) <= 2:
                print('Searching for other Thresholds for Higher F1 Score...')
                ###precision, recall, thresholds = precision_recall_curve(y_cv, y_proba[:,rare_class])
                precision, recall, thresholds = precision_recall_curve(y_cv, y_proba[:,1])
                try:
                    f1 = (2*precision*recall)/(precision+recall)
                    f1 = np.nan_to_num(f1)
                    m_idx = np.argmax(f1)
                    m_thresh = thresholds[m_idx]
                    best_f1 = f1[m_idx]
                except:
                    best_f1 = f1_score(y_cv, y_pred)
                    m_thresh = 0.5
                print("    Using threshold=0.5. However %0.2f alternatively provides F1 = %0.2f in CV data..." %(m_thresh,best_f1))
                ###y_pred = (y_proba[:,rare_class]>=m_thresh).astype(int)
                y_pred = (y_proba[:,1]>0.5).astype(int)
            else:
                y_proba = model.predict_proba(X_cv)
                y_pred = model.predict(X_cv)
        else:
            y_pred = model.predict(X_cv)
        ###   This is where you print out the First Model's Results ########
        print('%s Model Results on Held Out CV Data Set:' %model_name)
        if modeltype == 'Regression':
            rmsle_calculated_m = rmse(y_cv.values, y_pred)
            print_regression_model_stats(y_cv.values, y_pred,'%s Model: Predicted vs Actual'%model_name)
        else:
            if model_name == 'Forests':
                print('    OOB Score = %0.3f' %model.oob_score_)
            rmsle_calculated_m = (y_cv.values==y_pred).astype(int).sum(axis=0)/(y_cv.shape[0])
            print('    Regular Accuracy Score = %0.1f%%' %(rmsle_calculated_m*100))
            rmsle_calculated_m = balanced_accuracy_score(y_cv,y_pred)
            print('    Balanced Accuracy Score = %0.1f%%' %(rmsle_calculated_m*100))
            rare_pct = y_cv[y_cv==rare_class].shape[0]/y_cv.shape[0]
            print(classification_report(y_cv,y_pred))
            print(confusion_matrix(y_cv, y_pred))
        ######      SET BEST PARAMETERS HERE ######
        ### Find what the order of best params are and set the same as the original model ###
        ## This is where we set the best parameters from training to the model ####
        if modeltype == 'Regression':
            if not Stacking_Flag:
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
                    y_pred = subm[cols].mean(axis=1)
                    print('########################################################')
                    print('Completed Ensemble predictions on held out data')
                    performed_ensembling = True
                    print_regression_model_stats(y_cv.values, y_pred.values,'Ensemble Model: Model Predicted vs Actual')
                except:
                    print('########################################################################')
                    print('Could not complete Ensembling predictions on held out data due to Error')
        else:
            ##  This is for Classification Problems Only #
            ### Find what the order of best params are and set the same as the original model ###
            ## This is where we set the best parameters from training to the model ####
            if not Stacking_Flag:
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
                    if len(classes) <= 2:
                        y_pred = (subm[cols].mean(axis=1)>0.5).astype(int)
                    else:
                        y_pred = (subm[cols].mean(axis=1)).astype(int)
                    print('########################################################')
                    print('Completed Ensemble predictions on held out data')
                    performed_ensembling = True
                except:
                    print('########################################################################')
                    print('Could not complete Ensembling predictions on held out data due to Error')
            else:
                print('########################################################################')
                print('No Ensembling of models done since Stacking_Flag = True ')
            if verbose >= 1:
                try:
                    Draw_ROC_MC_ML(y_cv, y_proba,y_pred, each_target, model_name, verbose)
                except:
                    print('    Error: Could not draw ROC AUC curve for multi-class data')
                try:
                    Draw_MC_ML_PR_ROC_Curves(model,X_cv,y_cv)
                except:
                    print('    Error: Could not draw PR ROC AUC curve for multi-class data')
            #### In case there are special scoring_parameter requests, you can print it here!
            if scoring_parameter == 'roc_auc' or scoring_parameter == 'auc':
                if len(classes) == 2:
                    print('    ROC AUC Score = %0.1f%%' %(roc_auc_score(y_cv, y_proba[:,rare_class])*100))
                else:
                    print('    No ROC AUC score for multi-class problems')
            elif scoring_parameter == 'jaccard':
                accu_all = jaccard_singlelabel(y_cv, y_pred)
                print('        Mean Jaccard Similarity  = {:,.1f}%'.format(
                                        accu_all*100))
                ## This is for multi-label problems ##
                if count == 0:
                    zipped = copy.deepcopy(y_pred)
                    count += 1
                else:
                    zipped = zip(zipped,y_pred)
                    count += 1
            elif scoring_parameter == 'basket_recall':
                if count == 0:
                    zipped = copy.deepcopy(y_pred)
                    count += 1
                else:
                    zipped = zip(zipped,y_pred)
                    count += 1
        if not Stacking_Flag and performed_ensembling:
            if modeltype == 'Regression':
                rmsle_calculated_f = rmse(y_cv, y_pred)
                print('After multiple models, Ensemble Model Results:')
                print('    RMSE Score = %0.1f' %(rmsle_calculated_f,))
                if rmsle_calculated_f < rmsle_calculated_m:
                    print('\nEnsembling Models is better than Single Model for this data set.')
                    error_rate.append(rmsle_calculated_f)
                else:
                    print('\nSingle Model is better than Ensembling Models for this data set.')
                    error_rate.append(rmsle_calculated_m)
            else:
                rmsle_calculated_f = balanced_accuracy_score(y_cv,y_pred)
                print('After multiple models, Ensemble Model Results:')
                rare_pct = y_cv[y_cv==rare_class].shape[0]/y_cv.shape[0]
                print('    Balanced Accuracy Score = %0.3f%%' %(
                        rmsle_calculated_f*100))
                print(classification_report(y_cv.values,y_pred.values))
                print(confusion_matrix(y_cv.values,y_pred.values))
                if rmsle_calculated_f > rmsle_calculated_m:
                    print('\nEnsembling Models is better than Single Model for this data set.')
                    error_rate.append(rmsle_calculated_f)
                else:
                    print('\nSingle Model is better than Ensembling Models for this data set.')
                    error_rate.append(rmsle_calculated_m)
        try:
            if verbose >= 2:
                plot_RS_params(gs.cv_results_, scoring_parameter, each_target)
        except:
            print('Could not plot Cross Validation Parameters')
        try:
            if Boosting_Flag and verbose >= 1:
                #### This plot is only for XGBoost ############
                plot_xgb_metrics(model,eval_metric,eval_set,modeltype,'%s Results' %each_target)
        except:
            print('Could not plot Model Evaluation Results Metrics')
        print('    Time taken for this Target (in seconds) = %0.0f' %(time.time()-start_time))
        if Boosting_Flag is None:
            if modeltype == 'Regression':
                imp = pd.DataFrame(model.coef_[:len(important_features)].T, index=important_features)
            else:
                imp = pd.DataFrame(model.coef_[0][:len(important_features)].T, index=important_features)
            imp_features_df = imp.sort_values(0,ascending=False)
        else:
            imp_features_df = pd.DataFrame(model.feature_importances_, columns=['Feature Weightings'],
                         index=important_features).sort_values('Feature Weightings',
                         ascending=False)
        if verbose >= 1:
            try:
                if len(X_train) >10000:
                    SHAP_model = copy.deepcopy(model)
                    SHAP_model.fit(X_train[:10000],y_train[:10000])
                    #### This is to make sure that SHAP values plotting doesn't take too long ############
                    plot_SHAP_values(SHAP_model,
                                     pd.DataFrame(X_train[:10000].T,columns=important_features),
                                     Boosting_Flag, matplotlib_flag)
                    print('Plotting SHAP (first 10,000) values to explain the output of model')
                else:
                    plot_SHAP_values(model,pd.DataFrame(X_train,columns=important_features),
                                     Boosting_Flag, matplotlib_flag)
                    print('Plotting SHAP (SHapley Additive exPlanations) values to explain the output of model')
            except:
                height_size = 5
                width_size = 10
                print('Plotting Feature Importances to explain the output of model')
                imp_features_df[:10].plot(kind='barh',title='Feature Importances for predicting %s' %each_target,
                                 figsize=(width_size, height_size))
        print()
        print('###################################################################################')
        print('Training model on complete Train data and Predicting using given Test Data...')
        ###############################################################################################################
        ###### Once again we do Entropy Binning on the Full Train Data Set !!
        if len(num_vars) > 0:
            ### Do Entropy Binning only if there are numeric variables in the data set! #####
            train, num_vars, important_features, test = add_entropy_binning(train, each_target,
                                                  saved_num_vars, saved_important_features, test, modeltype,Binning_Flag)
        ### Now we add another Feature tied to KMeans clustering using Predictor and Target variables ###
        if KMeans_Featurizer and len(num_vars) > 0:
            #### Perform KMeans Featurizer only if there are numeric variables in data set! #########
            print('    Adding one Feature named "cluster" using KMeans_Featurizer...')
            if isinstance(test, str):
                if modeltype != 'Regression':
                    train, _ = Transform_KM_Features(train[important_features], train[each_target], train[important_features], len(classes))
                else:
                    train, _ = Transform_KM_Features(train[important_features], train[each_target], train[important_features])
            else:
                if modeltype != 'Regression':
                    train, test = Transform_KM_Features(train[important_features], train[each_target], test[important_features], len(classes))
                else:
                    train, test = Transform_KM_Features(train[important_features], train[each_target], test[important_features])
        important_features = [x for x in list(train) if x not in [each_target]]
        ######### This is where you do Stacking of Multi Model Results into One Column ###
        if Stacking_Flag:
            #### In order to join, you need X_train to be a Pandas Series here ##
            from autoviml.QuickML_Stacking import QuickML_Stacking
            print('CAUTION: Stacking can produce Highly Overfit models on Training Data...')
            ### In order to avoid overfitting, we are going to learn from a small sample of data
            ### That is why we are using X_cv to train on and using it to predict on X_train!
            addcol, stacks1 = QuickML_Stacking(train[important_features],train[each_target],'',
                          modeltype, Boosting_Flag, scoring_parameter,verbose)
            ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
            train = train.join(pd.DataFrame(stacks1,index=train.index,
                                              columns=addcol))
            ##### Leaving multiple columns for Stacking is best! Do not do the average of predictions!
            print('    Adding %d Stacking feature(s) to training data' %len(addcol))
            if not isinstance(orig_test, str):
                ### In order to avoid overfitting, we are going to learn from a small sample of data
                ### That is why we are using X_train to train on and using it to predict on X_test
                from autoviml.QuickML_Stacking import QuickML_Stacking
                _, stacks2 = QuickML_Stacking(train[important_features],train[each_target],test[important_features],
                          modeltype, Boosting_Flag, scoring_parameter,verbose)
                ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                test = test.join(pd.DataFrame(stacks2,index=test.index,
                                                  columns=addcol))
                ##### Adding multiple columns for Stacking is best! Do not do the average of predictions!
                #test = test.join(pd.DataFrame(stacks2.mean(axis=1).round().astype(int),
                #                             columns=[addcol],index=test.index))
            #important_features.append(addcol)
            ######  We make sure that we remove too many features that are highly correlated ! #####
            #addcol = remove_variables_using_fast_correlation(train,addcol,corr_limit,verbose)
            important_features += addcol
        #####################################################################################################
        feature_reduction = False #### Let's not do any more feature reduction after the first time!
        if feature_reduction:
            #########     SELECT IMPORTANT FEATURES FROM THE NEWLY ADDED FEATURES   #############################
            if model_name != 'CatBoost':
                #### Heuristic: Leave the features as is since you are not adding any categorical features ####
                max_feats = int(1.0*len(important_features))
                fs = SelectKBest(score_func=chi2, k=max_feats)
                fs.fit(train[important_features], train[each_target])
                cols_index = fs.get_support(indices=True)
                important_features = np.array(important_features)[cols_index]
                important_features= important_features.tolist()
            else:
                important_features,num_vars = find_top_features_xgb(train,important_features,num_vars,
                                                         each_target,
                                                     modeltype,corr_limit,verbose)
            print('Selected %d features using feature reduction' %len(important_features))
            if verbose and len(important_features) <= 30:
                print('    Features list: %s' %important_features)
        #####################################################################################
        if len(important_features) == 0:
            print('No important features found. Using all input features...')
            important_features = copy.deepcopy(saved_important_features)
            #important_features = copy.deepcopy(red_preds)
        #### This is Second time: divide train and test into Train and Test Sets #################
        ### The next 2 lines are crucial: if X and y are dataframes, then next 2 should be df's
        ###   They should not be df.values since they will become numpy arrays and XGB will error.
        trainm = train[important_features+[each_target]]
        red_preds = copy.deepcopy(important_features)
        if Boosting_Flag:
            X = trainm[red_preds]
            y = trainm[each_target]
            if modeltype == 'Regression':
                train_part = int((1-test_size)*X.shape[0])
                X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
            else:
                X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                            test_size=test_size, random_state=seed)
            eval_set = [(X_cv,y_cv)]
        else:
            X_train = trainm[red_preds]
            y_train = trainm[each_target]
            eval_set = [()]
        ##### ############      TRAINING MODEL SECOND TIME WITH FULL_TRAIN AND PREDICTING ON TEST ############
        model_start_time = time.time()
        if modeltype != 'Regression':
            if Imbalanced_Flag:
                try:
                    print('\nImbalanced Class Training using Majority Class Downsampling method...')
                    d_model = downsampling_with_model_training(X_train,y_train, eval_set, model,
                                      Boosting_Flag, eval_metric,modeltype,no_training=False,
                                      minority_class=rare_class,verbose=verbose)
                    if not isinstance(d_model, str):
                        #### If d_model succeeds,copy it so it can become a regular model again ##
                        model = copy.deepcopy(d_model)
                    else:
                        #### If d_model failed, it will just be an empty string, so you can try regular model ###
                        print('Error in training Imbalanced model second time. Trying regular model..')
                        Imbalanced_Flag = False
                        if Boosting_Flag:
                            if model_name == 'XGBoost':
                                #### Set the Verbose to 0 since we don't want too much output ##
                                model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                                        eval_metric=eval_metric,silent=True)
                            else:
                                model.fit(X_train, y_train, cat_features=imp_cats)
                                #model.fit(X_train, y_train, cat_features=cat_vars_encoded,
                                #    early_stopping_rounds=early_stopping)
                        else:
                                model.fit(X_train, y_train)
                except:
                    print('Error in training Imbalanced model second time. Trying regular model..')
                    Imbalanced_Flag = False
                    if Boosting_Flag:
                        if model_name == 'XGBoost':
                            #### Set the Verbose to 0 since we don't want too much output ##
                            model.fit(X_train, y_train, early_stopping_rounds=early_stopping,
                                    eval_metric=eval_metric,silent=True)
                        else:
                            model.fit(X_train, y_train, cat_features=imp_cats)
                            #model.fit(X_train, y_train, cat_features=cat_vars_encoded,
                            #    early_stopping_rounds=early_stopping )
                    else:
                            model.fit(X_train, y_train)
            else:
                try:
                    if Boosting_Flag:
                        if model_name == 'XGBoost':
                            ### Since second time we don't have X_cv, we remove it
                                model.fit(X_train, y_train,
                                    eval_metric=eval_metric,verbose=0)
                        else:
                            model.fit(X_train, y_train, cat_features=imp_cats)
                            #model.fit(X_train, y_train, cat_features=cat_vars_encoded,
                            #    early_stopping_rounds=early_stopping)
                    else:
                            model.fit(X_train, y_train)
                except:
                    print('Training regular model second time erroring: Check if Input is correct...')
                    return
        else:
            try:
                if Boosting_Flag:
                    if model_name == 'XGBoost':
                        model.fit(X_train, y_train,
                            eval_metric=eval_metric,verbose=verbose)
                    else:
                        model.fit(X_train, y_train, cat_features=imp_cats)
                        #model.fit(X_train, y_train, cat_features=cat_vars_encoded,
                        #        early_stopping_rounds=early_stopping)
                else:
                        model.fit(X_train, y_train)
            except:
                print('Training model second time is Erroring: Check if Input is correct...')
                return
        print('Model Training time taken in seconds = %0.0f' %(time.time()-model_start_time))
        ##   TRAINING OF MODELS COMPLETED. NOW START PREDICTIONS ON TEST DATA   ################
        if modeltype != 'Regression' and len(classes) <= 2:
            y_proba = model.predict_proba(X_cv[important_features])
            y_pred = (y_proba[:,1]>0.5).astype(int)
            bal1 = balanced_accuracy_score(y_cv, y_pred)
            y_pred = (y_proba[:,1]>m_thresh).astype(int)
            bal2 = balanced_accuracy_score(y_cv, y_pred)
            if bal1 >= bal2 :
                m_thresh = 0.5
            if m_thresh != 0.5:
                print('After comparing thresholds, alternative threshold=%0.2f for data set' %m_thresh)
        #### new_cols is to keep track of new prediction columns we are creating #####
        new_cols = []
        if not isinstance(orig_test, str):
            ### If there is a test data frame, then let us predict on it #######
            ### The next 3 lines are crucial: if X and y are dataframes, then next 2 should be df's
            ###   They should not be df.values since they will become numpy arrays and XGB will error.
            testm = test[red_preds]
            X_test = copy.deepcopy(testm)
            y_pred = model.predict(X_test)
            if modeltype == 'Regression':
                ########   This is for Regression Problems Only ###########
                ######  If Stacking_ Flag is False, then we do Ensembling #######
                if not Stacking_Flag and model_name != 'CatBoost':
                    try:
                        subm = pd.DataFrame()
                        #### This is for Ensembling  Only #####
                        #### In Test data verbose is set to zero since no results can be obtained!
                        models_list, ensembles = QuickML_Ensembling(X_train, y_train, X_test, '',
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
                        y_pred = subm[new_cols].mean(axis=1)
                        new_col = each_target+'_Ensembled_predictions'
                        testm[new_col] = y_pred.values
                        new_cols.append(new_col)
                        print('########################################################')
                        print('Completed Ensemble predictions on held out data')
                        if not isinstance(sample_submission, str):
                            sample_submission[each_target] = y_pred
                    except:
                        print('########################################################################')
                        print('Could not complete Ensembling predictions on held out data due to Error')
                else:
                    if not isinstance(sample_submission, str):
                        sample_submission[each_target] = y_pred
            else:
                ########   This is for both Binary and Multi Classification Problems ###########
                y_proba = model.predict_proba(X_test)
                if len(classes) <= 2:
                    print('Test Data predictions using Threshold = 0.5')
                    y_pred = (y_proba[:,1]>0.5).astype(int)
                if len(label_dict[each_target]['transformer']) == 0:
                    ### if there is no transformer, then leave the predicted classes as is
                    classes = label_dict[each_target]['classes']
                    ######  If Stacking_Flag is False, then we do Ensembling #######
                    if not Stacking_Flag and model_name != 'CatBoost':
                        subm = pd.DataFrame()
                       #### This is for Ensembling  Only #####
                        #### In Test data verbose is set to zero since no results can be obtained!
                        if len(classes) == 2:
                            models_list, ensembles = QuickML_Ensembling(X_train, y_train, X_test, '',
                                                      modeltype='Binary_Classification', Boosting_Flag=Boosting_Flag,
                                                       scoring='', verbose=0)
                        else:
                            models_list, ensembles = QuickML_Ensembling(X_train, y_train, X_test, '',
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
                        ### After this, y_pred is a Series from now on. You need y_pred.values  ####
                        if len(classes) <= 2:
                            y_pred = (subm[cols].mean(axis=1)>0.5).astype(int)
                        else:
                            y_pred = (subm[cols].mean(axis=1)).astype(int)
                    for each_class in classes:
                        try:
                            new_col = each_target+'_proba_'+str(each_class)
                            count = int(label_dict[each_target]['dictionary'][each_class])
                            testm[new_col] = y_proba[:,count]
                            new_cols.append(new_col)
                        except:
                            new_col = each_target+'_proba_'+each_class
                            count = int(label_dict[each_target]['dictionary'][each_class])
                            testm[new_col] = y_proba[:,count]
                            new_cols.append(new_col)
                    if not Stacking_Flag:
                        new_col = each_target+'_Ensembled_predictions'
                        try:
                            testm[new_col] = y_pred.values
                        except:
                            testm[new_col] = y_pred
                    else:
                        new_col = each_target+'_Stacked_predictions'
                        testm[new_col] = y_pred
                    new_cols.append(new_col)
                else:
                    ### if there is a transformer, then convert the predicted classes to orig classes
                    classes = label_dict[each_target]['classes']
                    dic = label_dict[each_target]['dictionary']
                    transformer = label_dict[each_target]['transformer']
                    class_nums = label_dict[each_target]['class_nums']
                    for each_class in classes:
                        try:
                            new_col = each_target+'_proba_'+str(each_class)
                            count = label_dict[each_target]['dictionary'][each_class]
                            testm[new_col] = y_proba[:,count]
                            new_cols.append(new_col)
                        except:
                            new_col = each_target+'_proba_'+each_class
                            count = label_dict[each_target]['dictionary'][each_class]
                            testm[new_col] = y_proba[:,count]
                            new_cols.append(new_col)
                    ######  If Stacking_ Flag is False, then we do Ensembling #######
                    if not Stacking_Flag and model_name != 'CatBoost':
                        subm = pd.DataFrame()
                        #### This is for Ensembling  Only #####
                        if len(classes) == 2:
                            models_list, ensembles = QuickML_Ensembling(X_train, y_train, X_test, '',
                                                      modeltype='Binary_Classification', Boosting_Flag=Boosting_Flag,
                                                       scoring='', verbose=verbose)
                        else:
                            models_list, ensembles = QuickML_Ensembling(X_train, y_train, X_test, '',
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
                        if len(classes) <= 2:
                            y_pred = (subm[cols].mean(axis=1)>0.5).astype(int)
                        else:
                            y_pred = (subm[cols].mean(axis=1)).astype(int)
                        print('########################################################')
                        print('Completed Ensemble predictions on held out data')
                        new_col = each_target+'_Ensembled_predictions'
                    else:
                        print('########################################################')
                        print('Completed Stacked predictions on held out data')
                        new_col = each_target+'_Stacked_predictions'
                    try:
                        testm[new_col] = pd.Series(y_pred).map(transformer).values
                    except:
                        testm[new_col] = pd.Series(y_pred.ravel()).map(transformer).values
                    new_cols.append(new_col)
                    if not isinstance(sample_submission, str):
                        try:
                            sample_submission[each_target] = pd.Series(y_pred.values).map(transformer).values
                        except:
                            sample_submission[each_target] = pd.Series(y_pred).map(transformer).values
            ##  Write the test and submission files to disk ###
            if not isinstance(testm, str):
                try:
                    write_file_to_folder(testm, each_target, each_target+'_'+modeltype+'_'+'test_modified.csv')
                except:
                    print('Error: Not able to write test file to disk. Skipping...')
            if not isinstance(sample_submission, str):
                print('    Saving sample_submission...')
                sample_submission = pd.concat([sample_submission,testm[new_cols]], axis=1)
                if modeltype != 'Regression':
                    probacols = [x for x in new_cols if 'proba' in x]
                    if verbose >= 2:
                        sample_submission[probacols].plot(kind='kde',figsize=(10,6),
                                                     title='Predictive Probability Density Chart')
                try:
                    write_file_to_folder(sample_submission, each_target, each_target+'_'+modeltype+'_'+'submission.csv')
                except:
                    print('Error: Not able to write submission file to disk. Skipping...')
            else:
                sample_submission = pd.concat([orig_test,testm[new_cols]], axis=1)
                if modeltype != 'Regression':
                    probacols = [x for x in new_cols if 'proba' in x]
                    if verbose >= 1:
                        sample_submission[probacols].plot(kind='kde',figsize=(10,6),
                                                     title='Predictive Probability Density Chart ')
                try:
                    write_file_to_folder(sample_submission, each_target, each_target+'_'+modeltype+'_'+'submission.csv')
                except:
                    print('Error: Not able to write submission file to disk. Skipping...')
            try:
                write_file_to_folder(trainm, each_target, each_target+'_'+modeltype+'_'+'train_modified.csv')
            except:
                print('Error: Not able to write train modified file to disk. Skipping...')
        else:
            ##### If there is no Test file, then do a final prediction on Train itself ###
            testm = trainm[red_preds]
            X_test = copy.deepcopy(testm)
            if modeltype == 'Regression':
                y_pred = model.predict(X_test)
                trainm[each_target+'_predictions'] = y_pred
            else:
                y_proba = model.predict_proba(X_test)
                #y_proba = model.predict_proba(trainm[red_preds].values)
                if type(label_dict[each_target]['transformer']) == str:
                    ### if there is no transformer, then leave the predicted classes as is
                    y_pred = y_proba.argmax(axis=1)
                    ### After this, y_pred is an array from now on. No .values needed ####
                    trainm[each_target+'_predictions'] = y_pred
                else:
                    dic = label_dict[each_target]['dictionary']
                    transformer = label_dict[each_target]['transformer']
                    ### To transform to original classes you need the dictionary, than map column and fill
                    ## new classes with some "unknown value"
                    try:
                        y_pred = pd.Series(y_proba.argmax(axis=1)).map(transformer)
                    except:
                        y_pred = np.zeros(y_proba.shape[0])
                        print('Could not transform test predictions to original classes')
                    ### After this, y_pred is an array from now on. No .values needed ####
                    trainm[each_target+'_predictions'] = y_pred
            try:
                write_file_to_folder(trainm, each_target, each_target+'_'+modeltype+'_'+'train_modified.csv')
            except:
                print('Error: Not able to write train modified file to disk. Skipping...')
            testm = copy.deepcopy(trainm)
            #### We do Ensembling only if there is a Test file. Otherwise, we don't!
            print('    No Ensembling of models done since there is no Test file given.')
        ### In case of multi-label models, we will reset the start train and test dataframes to contain new features created
        start_train = start_train[target].join(start_train[orig_red_preds])
        if not isinstance(orig_test, str):
            start_test = start_test[orig_red_preds]
        #### Once each target cycle is over, reset the red_preds to the orig_red_preds so we can start over
        red_preds = copy.deepcopy(orig_red_preds)
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
    ##   END OF ONE LABEL IN A MULTI LABEL DATA SET ! WHEW ! ###################
    print('Time Taken in mins = %0.1f for the Entire Process' %((time.time()-start_time)/60))
    #return model, imp_features_df.index.tolist(), trainm, testm
    return model, important_features, trainm, testm
###############################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
def plot_SHAP_values(m,X,Boosting_Flag=False,matplotlib_flag=False):
    import shap
    # load JS visualization code to notebook
    if not matplotlib_flag:
        shap.initjs()
    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X)
    if Boosting_Flag:
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        shap.summary_plot(shap_values, X, plot_type="violin")
    else:
        shap.summary_plot(shap_values, X, plot_type="bar")

################      Find top features using XGB     ###################
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor


def find_top_features_xgb(train,preds,numvars,target,modeltype,corr_limit,verbose=0):
    """
    This is a fast utility that uses XGB to find top features. You
    It returns a list of important features.
    Since it is XGB, you dont have to restrict the input to just numeric vars.
    You can send in all kinds of vars and it will take care of transforming it. Sweet!
    """
    subsample =  0.5
    col_sub_sample = 0.5
    train = copy.deepcopy(train)
    start_time = time.time()
    test_size = 0.2
    seed = 1
    n_splits = 5
    early_stopping = 5
    ####### All the default parameters are set up now #########
    kf = KFold(n_splits=n_splits,random_state= 33)
    rem_vars = left_subtract(preds,numvars)
    if len(numvars) > 0:
        final_list = remove_variables_using_fast_correlation(train,numvars,corr_limit,verbose)
    else:
        final_list = numvars[:]
    print('    Adding %s categorical variables to reduced numeric variables  of %d' %(
                            len(rem_vars),len(final_list)))
    preds = final_list+rem_vars
    ########    Drop Missing value rows since XGB for some reason  #########
    ########    can't handle missing values in early stopping rounds #######
    train.dropna(axis=0,subset=preds+[target],inplace=True)
    ########   Dont move this train and y definition anywhere else ########
    y = train[target]
    ######################################################################
    important_features = []
    if modeltype == 'Regression':
        model_xgb = XGBRegressor(objective='reg:linear', n_estimators=100,subsample=subsample,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                 seed=1,n_jobs=-1,random_state=1)
        eval_metric = 'rmse'
    else:
        #### This is for Classifiers only
        classes = np.unique(train[target].values)
        if len(classes) == 2:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                colsample_bytree=col_sub_sample,gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=5, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='binary:logistic',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=1,
                seed=1, silent=True)
            eval_metric = 'logloss'
        else:
            model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', subsample=subsample,
                        colsample_bytree=col_sub_sample, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=5, min_child_weight=1, missing=-999, n_estimators=100,
                n_jobs=-1, nthread=None, objective='multi:softmax',
                random_state=1, reg_alpha=0.5, reg_lambda=0.5, scale_pos_weight=1,
                seed=1, silent=True)
            eval_metric = 'mlogloss'
    ####   This is where you start to Iterate on Finding Important Features ################
    train_p = train[preds]
    if train_p.shape[1] < 10:
        iter_limit = 2
    else:
        iter_limit = int(train_p.shape[1]/5+0.5)
    print('Selected No. of variables = %d ' %(train_p.shape[1],))
    print('Finding Important Features...')
    for i in range(0,train_p.shape[1],iter_limit):
        if verbose >= 1:
            print('        in %d variables' %(train_p.shape[1]-i))
        if train_p.shape[1]-i < iter_limit:
            X = train_p.iloc[:,i:]
            if modeltype == 'Regression':
                train_part = int((1-test_size)*X.shape[0])
                X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
            else:
                X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                            test_size=test_size, random_state=seed)
            try:
                model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,eval_set=[(X_cv,y_cv)],
                                    eval_metric=eval_metric,verbose=False)
            except:
                print('Finding top features erroring. Missing values probably. Continuing...')
                return important_features, numvars
            try:
                [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_
                        ),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'
                    }).sort_values(by='importance',ascending=False)[:25]['column'])]
            except:
                print('Finding top features erroring. No feature_importances probably. Continuing...')
                important_features = copy.deepcopy(preds)
                return important_features, numvars
        else:
            X = train_p[list(train_p.columns.values)[i:train_p.shape[1]]]
            #### Split here into train and test #####
            if modeltype == 'Regression':
                train_part = int((1-test_size)*X.shape[0])
                X_train, X_cv, y_train, y_cv = X[:train_part],X[train_part:],y[:train_part],y[train_part:]
            else:
                X_train, X_cv, y_train, y_cv = train_test_split(X, y,
                                                            test_size=test_size, random_state=seed)
            model_xgb.fit(X_train,y_train,early_stopping_rounds=early_stopping,
                          eval_set=[(X_cv,y_cv)],eval_metric=eval_metric,verbose=False)
            try:
                [important_features.append(x) for x in list(pd.concat([pd.Series(model_xgb.feature_importances_
                        ),pd.Series(list(X_train.columns.values))],axis=1).rename(columns={0:'importance',1:'column'
                    }).sort_values(by='importance',ascending=False)[:25]['column'])]
                important_features = list(OrderedDict.fromkeys(important_features))
            except:
                print('Finding top features erroring. Multi Label possibly. Continuing...')
                important_features = copy.deepcopy(preds)
                return important_features, numvars
    important_features = list(OrderedDict.fromkeys(important_features))
    print('    Found %d important features' %len(important_features))
    #print('    Time taken (in seconds) = %0.0f' %(time.time()-start_time))
    numvars = [x for x in numvars if x in important_features]
    return important_features, numvars
###############################################
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
#################################################
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
                df[[each_param]+cols].groupby(each_param).mean().plot(kind='bar',stacked=True,
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
    plt.show();
    return df
################################################################################
def plot_xgb_metrics(model,eval_metric,eval_set,modeltype,model_label=''):
    height_size = 5
    width_size = 10
    if modeltype == 'Regression':
        results = model.evals_result()
        epochs = len(results['validation_0'][eval_metric])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = plt.subplots(figsize=(width_size, height_size))
        if len(eval_set) == 1:
            ax.plot(x_axis, results['validation_0'][eval_metric], label='Cross Validation')
        else:
            ax.plot(x_axis, results['validation_0'][eval_metric], label='Train')
            ax.plot(x_axis, results['validation_1'][eval_metric], label='Cross Validation')
        ax.legend()
        plt.ylabel('RMSE')
        plt.title('%s Train and CV Model Performance RMSE vs Epochs' %model_label)
        plt.show();
    else:
        # retrieve performance metrics
        # plot classification error
        results = model.evals_result()
        epochs = len(results['validation_0'][eval_metric])
        x_axis = range(0, epochs)
        # plot log loss
        if isinstance(eval_metric, list):
            eval_metric = 'logloss'
        fig, ax = plt.subplots(figsize=(width_size, height_size))
        if len(eval_set) == 1:
            ax.plot(x_axis, results['validation_0'][eval_metric], label='Cross Validation')
        else:
            ax.plot(x_axis, results['validation_0'][eval_metric], label='Train')
            ax.plot(x_axis, results['validation_1'][eval_metric], label='Cross Validation')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('%s Validation Metrics: Classification Error vs Epochs' %model_label)
        plt.show();
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
def classify_columns(df_preds, verbose=0):
    """
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 15
    def add(a,b):
        return a+b
    train = df_preds[:]
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
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [
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
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= 50 and len(train[col].value_counts()
                        ) < len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) < len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
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
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in ['int8','int16',
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
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= cat_limit and len(train[col].value_counts()) != len(train):
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
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    if verbose >= 1:
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
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        print('    Missing columns = %s' %set(list(train))-set(flat_list))
    return sum_all_cols
#################################################################################
from collections import OrderedDict
def remove_variables_using_fast_correlation(df,numvars,corr_limit = 0.70,verbose=0):
    """
    Removes variables that are highly correlated using a pair-wise
    high-correlation knockout method. It is highly efficient and hence can work on thousands
    of variables in less than a minute, even on a laptop. Only send in a list of numeric
    variables, otherwise, it will blow-up!
    Correlation = 0.70 This is the highest correlation that any two variables can have.
    Above this, and one of them gets knocked out: this is decided in the shootout stage
    after the initial round of cutoffs for pair-wise correlations...It returns a list of
    clean variables that are uncorrelated (atleast in a pair-wise sense).
    """
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_items = lambda dic: [x for x in dic.items()]
    flatten_keys = lambda dic: [x for x in dic.keys()]
    flatten_values = lambda dic: [x for x in dic.values()]
    start_time = time.time()
    print('Number of numeric variables = %d' %len(numvars))
    corr_pair_count_dict, rem_col_list, temp_corr_list,correlated_pair_dict  = find_corr_vars(df[numvars].corr())
    temp_dict = Counter(flatten(flatten_items(correlated_pair_dict)))
    temp_corr_list = []
    for name, count in temp_dict.items():
        if count >= 2:
            temp_corr_list.append(name)
    temp_uncorr_list = []
    for name, count in temp_dict.items():
        if count == 1:
            temp_uncorr_list.append(name)
    ### Do another correlation test to remove those that are correlated to each other ####
    corr_pair_count_dict2, rem_col_list2 , temp_corr_list2, correlated_pair_dict2 = find_corr_vars(
                            df[rem_col_list+temp_uncorr_list].corr(),corr_limit)
    final_dict = Counter(flatten(flatten_items(correlated_pair_dict2)))
    #### Make sure that these lists are sorted and compared. Otherwise, you will get False compares.
    if temp_corr_list2.sort() == temp_uncorr_list.sort():
        ### if what you sent in, you got back the same, then you now need to pick just one:
        ###   either keys or values of this correlated_pair_dictionary. Which one to pick?
        ###   Here we select the one which has the least overall correlation to rem_col_list
        ####  The reason we choose overall mean rather than absolute mean is the same reason in finance
        ####   A portfolio that has lower overall mean is better than  a portfolio with higher correlation
        corr_keys_mean = df[rem_col_list+flatten_keys(correlated_pair_dict2)].corr().mean().mean()
        corr_values_mean = df[rem_col_list+flatten_values(correlated_pair_dict2)].corr().mean().mean()
        if corr_keys_mean <= corr_values_mean:
            final_uncorr_list = flatten_keys(correlated_pair_dict2)
        else:
            final_uncorr_list = flatten_values(correlated_pair_dict2)
    else:
        final_corr_list = []
        for name, count in final_dict.items():
            if count >= 2:
                final_corr_list.append(name)
        final_uncorr_list = []
        for name, count in final_dict.items():
            if count == 1:
                final_uncorr_list.append(name)
    ####  Once we have chosen a few from the highest corr list, we add them to the highest uncorr list#####
    selected = copy.deepcopy(final_uncorr_list)
    #####  Now we have reduced the list of vars and these are ready to be used ####
    final_list = list(OrderedDict.fromkeys(selected + rem_col_list))
    if int(len(numvars)-len(final_list)) == 0:
        print('    No variables were removed since no highly correlated variables found in data')
    else:
        print('    Number of variables removed due to high correlation = %d ' %(len(numvars)-len(final_list)))
    if verbose == 2:
        if len(left_subtract(numvars, final_list)) > 0:
            print('    List of variables removed: %s' %(left_subtract(numvars, final_list)))
    return final_list

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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import re


def add_poly_vars_select(data,numvars,targetvar,modeltype,poly_degree=2,Add_Poly=2,md='',
                         scaling=True, fit_flag=False, verbose=0):
    """
    #### This adds Polynomial and Interaction Variables of any Size to a data set and returns the best vars
    among those poly and interaction variables. Notice you will get a list of variables as well as the modified
    data set with the old and new (added) variables. Very Convenient when you want to do quick testing.
    There are 3 settings for Add_Poly flag: 0 => No poly or intxn variables added. 1=> Only intxn vars
    added. 2=> only polynomial degree (squared) vars added. 3=> both squared and intxn vars added.
    If Fit_Flag=True, then it is assumed that there are predictors and labels and hence a model is fitted.
    If Fit_Flag=False, then it is assumed it is Test data and hence does not contain labels. So no model
    is fitted and no variables are chosen.
    """
    orig_data_index = data.index
    if modeltype == 'Regression':
        lm = LassoCV(alphas=np.linspace(0.01,100),n_jobs=-1,max_iter=2000,
                 fit_intercept=True, normalize=False)
    else:
        lm = LogisticRegressionCV(Cs=[0.01,0.1,1,2,5,10,20],fit_intercept=True,
                            max_iter=1000,solver='saga',n_jobs=-1,
                          penalty='l2',dual=False, random_state=0)
    predictors = copy.deepcopy(numvars)
    #### number of original features in data set ####
    n_orig_features = len(predictors)
    selected = []
    X_data = data[predictors]
    #######  Initial Model with all input variables ######
    if fit_flag:
        Y = data[targetvar]
        print('Building Inital Model with given variables...')
        print_model_metrics(modeltype,lm,X_data,Y,False)
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
    ######## Here is where the Interaction variable selection begins ############
    #print('Adding Polynomial %s-degree and Interaction variables...' %poly_degree)
    if Add_Poly == 1:
        ### If it is 1, add only Interaction variables.
        poly = PolynomialFeatures(degree=poly_degree, include_bias = False, interaction_only=True)
    elif Add_Poly >= 2:
        #### If it is 2 or 3 add both Squared and Interaction variables. We will remove interaction
        ###       variables later in this program. For now include both!
        poly = PolynomialFeatures(degree=poly_degree, include_bias = False, interaction_only=False)
    if fit_flag:
        md = poly.fit(X) #### This defines the Polynomial feature extraction
    try:
        XP = md.transform(X) #### This transforms X into a Polynomial Order
    except MemoryError:
        return predictors, lm, X, md, []
    ###########   FAST FEATURE REDUCTION USING LASSO FOR LARGE DATA SETS ########
    snames = md.get_feature_names() ### snames contains all the Poly and Intxn variables including x variables
    XP1 = pd.DataFrame(XP,index=orig_data_index, columns=snames) ## XP1 has all the Xvars:incl orig+poly+intxn vars
    if not fit_flag:
        #### Just return the transformed dataframe with all x_vars incl orig+poly+intxn vars
        return predictors, lm, XP1, md, snames
        #############################################################################
    else:
        #### If there is fitting to be done, then you must do this ###
        lst = snames[:n_orig_features]  ### lst contains x_variables corresponding to the original number
        ####  It is now time to cut down the original x_variables to just squared variables and originals here ####
        if Add_Poly == 2:
            ### X_vars contains only x versions of original and squared variables only ####
            X_vars = [x for x in snames if '^2' in x]    ### x_vars contains a short version
        else:
            #### If Add_Poly is not 2, then you just excl the orig variables and take only the Poly and Intxn vars
            X_vars = snames[n_orig_features:]
        ##### Now let's convert the x_variables into text variables which is the original variable names
        xvars = copy.deepcopy(X_vars)
        dict_vars = dict(zip(predictors,lst)) ### this is only containing the original variables
        #### Remember that the LHS and RHS of the next 2 lines should be the same: xvars. Don't change it!
        for key,val in dict_vars.items():
            #### this converts all the x_variables into Poly and Intxn of original variables ###
            xvars = [re.sub(val+r'\b',key,xvars[x]) for x in range(len(xvars))]
        #### Now Build a Data Frame containing containing x variables and their counterpart original variables here ####
        text_vars = copy.deepcopy(xvars) #### text_vars now contains the text version of x variables
        XP1X = XP1[X_vars]
        XP2 = XP1[lst+X_vars]  ### XP2 has all the Xx variables from orig to polynomial+intxn vars
        dfx = pd.DataFrame([X_vars, text_vars])
        df = dfx.T
        df.columns=['X Names','Interaction Variable Names']
        ####   Use LassoCV or LogisticRegressionCV to Reduce Number of Variables using Regularization
        print('Building Comparison Model with only Poly and Interaction variables...')
        lm_p, _ = print_model_metrics(modeltype,lm,XP1X,Y,True)
        if modeltype == 'Regression':
            df['Coefficient Values'] = lm_p.coef_
        else:
            df['Coefficient Values'] = lm_p.coef_[0]
        #### This part selects the top 10% of interaction variables created using linear modeling.
        ###   It is very important to leave the next line as is since > symbol is better than >=
        lim = abs(df['Coefficient Values']).quantile(0.9)
        df90 = df[df['Coefficient Values']>lim].sort_values('Coefficient Values',ascending=False)
        if df['Coefficient Values'].sum()==0.0 or df90['Coefficient Values'].sum()==0.0:
            ### There is no coefficient that is greater than 0 here. So reject all variables!
            sel_x_vars = []
            print('Zero Interaction and Polynomial variable(s) selected...\n')
        elif df['Coefficient Values'].shape[0] == df90['Coefficient Values'].shape[0]:
            ### There is no coefficient that is greater than 0 here. So reject all variables!
            sel_x_vars = []
            print('Zero Interaction and Polynomial variable(s) selected...\n')
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
            print('%d Interaction and Polynomial variable(s) selected...\n' %len(interactions))
            sel_x_vars = df90["X Names"].values.tolist()
        final_x_vars = lst+sel_x_vars
        New_XP = XP2[final_x_vars]
        finalvars = predictors + df[df["X Names"].isin(final_x_vars)][
                                'Interaction Variable Names'].values.tolist()
        #### New_XP will be the final dataframe that we will send with orig and poly/intxn variables
        New_XP.columns = finalvars
        return finalvars, lm_p, New_XP, md, final_x_vars

## Import sklearn
from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold


def print_model_metrics(modeltype,reg,X,y,fit_flag=False,verbose=1):
    ### If fit_flag is set to True, then you must return a fitted model ###
    ###   Else you must return the cv_scores.mean only ####
    n_splits = 5
    if verbose==1:
        print("Model Report :")
        print('    Number of Variables = ',X.shape[1])
    if modeltype == 'Regression':
        scv = KFold(n_splits=n_splits, random_state=0)
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
        if verbose:
            print("    CV Weighted F1 Score : %.4g +/- %.4g | Min = %.4g | Max = %.4g" % (
                np.mean(cv_scores),np.std(cv_scores),
                 np.min(cv_scores),np.max(cv_scores)))
    if fit_flag:
        return reg.fit(X,y),  cv_scores.mean()
    else:
        return  cv_scores.mean()


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
    return finalvars##############################################################################################
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd

get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
figsize = (10, 6)
rcParams['figure.figsize'] = figsize
##################################################################################
def Draw_ROC_MC_ML(y_true, y_proba,y_pred,target, model_name, verbose=0):
    y_proba = y_proba[:]
    y_pred = y_pred[:]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    iterations = 0
    if isinstance(target,str):
        target = [target]
    for targ in target:
        classes = list(range(len(np.unique(y_true))))
        n_classes = len(classes)
        if n_classes == 2:
            ### Always set rare_class to 1 when there are only 2 classes for drawing ROC Curve!
            rare_class = 1
        else:
            rare_class = find_rare_class(y_true)
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
                pass
            else:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba[:,rare_class].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
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
            plt.title('ROC Curve for label:%s' %targ)
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
            labels.append('micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]))
            ax.plot(fpr["micro"], tpr["micro"],
                     color='deeppink', linestyle=':', linewidth=4)
            labels.append('macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]))
            ax.plot(fpr["macro"], tpr["macro"],
                     color='navy', linestyle=':', linewidth=4)
            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve for label: %s' %targ)
            plt.legend(labels,loc="lower right")
            plt.show();

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
    predictors = [x for x in traindf if x not in [target]]
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
                    print('Number of Target Rows in Test Data: %d' %len(testdf[testdf[target]==1]))
                    tsdata = testdf[:]
                else:
                    print('Test data has single class. Overwriting predictions on Test Target column.')
                    predictors = [x for x in list(traindf) if x not in [target]]
                    tsdata = testdf[:]
            except:
                print('No Target column in Test data')
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
def simple_feature_engineering(train, test, cats, nums):
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
            fillnum = train[num].mode()[0]
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
        print('    Adding output files to current folder: ./%s' %filename)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        df.to_csv(filename,index=False)
##############################################################
# This performs Down Sampling of Majority Class to a 90/10 or 80/20 ratio
###  depending on rare_class percentage. If the rare class is less than 5%,
###   it uses 90/10 to improve the training and 80/20 otherwise.
# This performs Down Sampling of Majority Class to a 90/10 or 80/20 ratio
###  depending on rare_class percentage. If the rare class is less than 5%,
###   it uses 90/10 to improve the training and 80/20 otherwise.
from collections import Counter
import time
########################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
####################################################################################
def downsampling_with_model_training(X_df,y_df,eval_set,model,Boosting_Flag,eval_metric,modeltype,
                                     no_training=False,minority_class=1, verbose=0):
    """
    #########    DOWNSAMPLING OF MAJORITY CLASS AND TRAINING IN SMALL BATCH SIZES  ###############
    #### One of the worst things you can do with imbalanced classes is to train all at once!
    ####  Hence this is one way to train a model in such cases with repeated samples of pos-class
    ####  along with batches of neg-class and get the model to differentiate between 2 classes.
    #########    MAKE SURE YOU TUNE THE DEFAULTS GIVEN HERE!  ###############
    """
    early_stopping = 5
    df = X_df.join(y_df)
    model = copy.deepcopy(model)
    downsampled_list = []
    df_target = y_df.name
    train_preds = [x for x in list(df) if x not in [df_target]]
    ccounts = Counter(y_df)   # Get class counts
    # Identify minority and majority classes
    # Get indices of each class
    print('Rare Class = %s' %minority_class)
    train_size = df.shape[0]
    df_pos = df[y_df==minority_class]
    df_neg = df[y_df!=minority_class]
    n_minority = ccounts[minority_class]
    rare_pct = n_minority/y_df.shape[0]
    print('    Pct of Rare Class in data = %0.2f%%' %(rare_pct*100))
    #### Remember that if you increase the denominator, you get small batch sizes and too many iter
    ###  Small batch sizes do not give good results and with Class_Weights they give terrible results.
    #### Instead keep the denominator small such as 10 and do not use any class_weights at all!
    n_iter = int(min(np.ceil(0.1/rare_pct),20))
    print('    Number of iterations for training =  %d' %n_iter)
    batch_size = int(df_neg.shape[0]/n_iter)
    print('  Rare Class Batch Size = %d' %n_minority)
    print('  Majority Class Batch Size = %d' %batch_size)
    for each_iter in range(n_iter):
        start_time = time.time()
        majority_egs_idx = df_pos.index
        # We undersample the majority class so they're somewhat balanced
        # THis is where you train the model with 90/10 Pos/Neg Sample Ratio ##
        begin_row = each_iter*batch_size
        end_row = begin_row + batch_size
        train_batch = df_neg.iloc[begin_row:end_row].append(df_pos)
        rare_pct = train_batch[train_batch[df_target]==minority_class].shape[0]/train_batch.shape[0]
        if verbose >= 1:
            print('     %d. Training Batch Size = %s' %((each_iter+1),train_batch.shape[0]))
            print('        Training Batch incident rate: %0.1f%%' %(rare_pct*100))
        if not no_training:
            ####  DO NOT USE CLASS WEIGHTS! THEY ARE A BLUNT INSTRUMENT! SAMPLE WEIGHTS ARE BETTER!
            #classes = [0,1]
            #wt = compute_class_weight("balanced", classes, train_batch[df_target])
            ### If using the plain model, use this next line ####
            #model.set_params(**{'class_weight':dict(zip(classes,wt))})
            ### If using GridSearchCV use the next line
            #model.estimator.set_params(**{'class_weight':dict(zip(classes,wt))})
            try:
                X_train = train_batch[train_preds]
                y_train = train_batch[df_target]
                if Boosting_Flag:
                    model.fit(X_train,y_train, early_stopping_rounds=early_stopping,
                                eval_metric=eval_metric,eval_set=eval_set,verbose=False)
                    if verbose >= 1:
                        print('             Batch Training completed' )
                        print('        Time Taken = %0.0f (in seconds)' %(
                                        time.time()-start_time))
                else:
                    model.fit(X_train,y_train)
                    if verbose >= 1:
                        print('             Batch Training completed' )
                        print('        Time Taken = %0.0f (in seconds)' %(
                                        time.time()-start_time))
            except:
                print('Error in training batch...continuing')
                downsampled_list.append(train_batch)
                return ''
        else:
            downsampled_list.append(train_batch)
    if no_training:
        return downsampled_list
    else:
        return model
##############################################################################################
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

def Draw_MC_ML_PR_ROC_Curves(classifier,X_test,y_test):
    """
    ========================================================================================
    Precision-Recall Curves: Extension of Original Version in SKLearn's Documentation Page:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    ========================================================================================
    """
    ###############################################################################
    # In binary classification settings
    # Compute the average precision score for Binary Classes
    # --------------------------------------------------------
    ###############################################################################
    classes = list(range(len(np.unique(y_test))))
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
            precision, recall, _ = precision_recall_curve(y_test.values, y_score)
        except:
            ### This works for Non Linear Models such as Forests and XGBoost #####
            precision, recall, _ = precision_recall_curve(y_test.values, y_score[:,1])

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
        plt.title('Precision-Recall curve: Avg.Precision={0:0.2f}, Avg F1={0:0.2f}'.format(
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
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                             average="micro")
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
def print_regression_model_stats(actuals, predicted, title='Model'):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    if len(actuals) != len(predicted):
        print('Error: Number of actuals and predicted dont match. Continuing...')
    else:
        plt.figure(figsize=figsize)
        dfplot = pd.DataFrame([actuals,predicted]).T
        dfplot.columns = ['Actuals','Predictions']
        x = actuals
        y =  predicted
        lineStart = actuals.min()
        lineEnd = actuals.max()
        plt.figure()
        plt.scatter(x, y, color = next(colors), alpha=0.5,label='Predictions')
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
        plt.xlim(lineStart, lineEnd)
        plt.ylim(lineStart, lineEnd)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.title(title)
        plt.show();
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
###################################################
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
##########################################################
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
######################################
from collections import defaultdict
def return_list_matching_keys(dicto,list_keys):
    '''Return a list of values matching keys in a dictionary
    from a list of keys. Returns Values in the Same order as the List Keys.
    '''
    results = []
    for each in list_keys:
        results.append(dicto[each])
    return results
###########################################
def add_entropy_binning(temp_train, targ, num_vars, important_features, temp_test,
                       modeltype, entropy_binning):
    """
        ######   This is where we do ENTROPY BINNING OF CONTINUOUS VARS ###########
        #### It is best to do Binning on ONLY on the top most variables from Important_Features!
        #### Make sure that the Top 2-10 vars are all CONTINUOUS VARS! Otherwise Binning Waste!
        ####    That ensures you get the Best Results!
    """
    seed = 99
    continuous_vars = copy.deepcopy(num_vars)
    if entropy_binning:
        if len(continuous_vars) > 0 and len(continuous_vars) <= 2:
            max_depth =  2
            continuous_vars = continuous_vars[:]
        elif len(continuous_vars) > 2 and len(continuous_vars) <= 5:
            max_depth = len(continuous_vars) - 2
            continuous_vars = continuous_vars[:2]
            entropy_binning = True
        elif len(continuous_vars) > 5 and len(continuous_vars) <= 10:
            max_depth = 5
            continuous_vars = continuous_vars[:5]
            entropy_binning = True
        elif len(continuous_vars) > 10 and len(continuous_vars) <= 50:
            max_depth = 10
            continuous_vars = continuous_vars[:10]
            entropy_binning = True
        elif len(continuous_vars) > 50:
            max_depth = 10
            continuous_vars = continuous_vars[:50]
        print('Binning Top %d continuous variables...' %len(continuous_vars))
    else:
        print('Leaving Top %d continuous variables as is...' %len(continuous_vars))
    if entropy_binning:
        new_bincols = []
        ###   This is an Awesome Entropy Based Binning for Continuous Variables ###########
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
                #### Drop the original continuous variable after you have created the bin ###
                temp_train.drop(each_num,axis=1,inplace=True)
                if type(temp_test) != str:
                    if isinstance(each_num, str):
                        bincol = each_num+'_bin'
                        temp_test[bincol] = np.digitize(temp_test[each_num].values, entropy_threshold)
                    else:
                        bincol = 'bin_'+str(each_num)
                        temp_test[bincol] = np.digitize(temp_test[each_num].values, entropy_threshold)
                    #### Drop the original continuous variable after you have created the bin ###
                    temp_test.drop(each_num,axis=1,inplace=True)
                important_features.append(bincol)
                important_features.remove(each_num)
                num_vars.append(bincol)
                num_vars.remove(each_num)
                new_bincols.append(bincol)
            except:
                print('Error in %s during Entropy Binning' %each_num)
        print('    Binning and replacing %s numeric features.' %(len(new_bincols)))
    else:
        print('    No Entropy Binning specified')
    return temp_train, num_vars, important_features, temp_test
############################################################################
