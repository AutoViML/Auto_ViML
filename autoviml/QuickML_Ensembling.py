import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit,StratifiedKFold,KFold
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit,StratifiedKFold,KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, Ridge
from sklearn.svm import LinearSVC, SVR, LinearSVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LassoLarsCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import time
import pdb
import time
import copy
import operator
#########################################################
def QuickML_Ensembling(X_train, y_train, X_test, y_test='', modeltype='Regression', 
                       Boosting_Flag=False,
                       scoring='', verbose=0):
    """
    Quickly builds and runs multiple models for a clean data set(only numerics).
    """
    start_time = time.time()
    seed = 99
    FOLDS = 5
    model_dict = {}
    model_tuples = []
    if len(X_train) <= 100000 and X_train.shape[1] < 50:
        NUMS = 100
    else:
        try:
            X_train = X_train.sample(frac=0.30,random_state=99)
            y_train = y_train[X_train.index]
        except:
            pass
        NUMS = 200
    if modeltype == 'Regression':
        if scoring == '':
            scoring = 'neg_mean_squared_error'
        scv = ShuffleSplit(n_splits=FOLDS,random_state=seed)
        if Boosting_Flag is None:
            ## Create an ensemble model ####
            model5 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                                    random_state=seed, max_depth=1, min_samples_leaf=2
                                    ), n_estimators=NUMS, random_state=seed)
            model_tuples.append(('Adaboost',model5))
        elif not Boosting_Flag:
            model5 = LassoLarsCV(cv=scv)
            model_tuples.append(('LassoLarsCV',model5))
        else:
            model5 = LassoLarsCV(cv=scv)
            model_tuples.append(('LassoLarsCV',model5))
        if Boosting_Flag is None:
            model6 = DecisionTreeRegressor(max_depth=5,min_samples_leaf=2)
            model_tuples.append(('Decision_Tree',model6))
        elif not Boosting_Flag:
            model6 = LinearSVR()
            model_tuples.append(('Linear_SVR',model6))
        else:
            model6 = DecisionTreeRegressor(max_depth=5,min_samples_leaf=2)
            model_tuples.append(('Decision_Tree',model6))
        model7 = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                        n_estimators=NUMS,random_state=seed)
        model_tuples.append(('Bagging_Regressor',model7))
        if Boosting_Flag is None:
            #### If the Boosting_Flag is True, it means Boosting model is present. 
            ###   So choose a different kind of classifier here
            model8 = RandomForestRegressor(bootstrap = False,
                                       max_depth = 10,
                                       max_features = 'auto',
                                       min_samples_leaf = 2,
                                       n_estimators = 200,
                                       random_state=99)
            model_tuples.append(('RF_Regressor',model8))
        elif not Boosting_Flag:
            #### If the Boosting_Flag is True, it means Boosting model is present. 
            ###   So choose a different kind of classifier here
            model8 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                                    random_state=seed, max_depth=1, min_samples_leaf=2
                                    ), n_estimators=NUMS, random_state=seed)
            model_tuples.append(('Adaboost',model8))
        else:
            model8 = RandomForestRegressor(bootstrap = False,
                                       max_depth = 10,
                                       max_features = 'auto',
                                       min_samples_leaf = 2,
                                       n_estimators = 200,
                                       random_state=99)
            model_tuples.append(('RF_Regressor',model8))
    else:
        if scoring == '':
            scoring = 'accuracy'
        num_classes = len(np.unique(y_test))
        scv = StratifiedKFold(n_splits=FOLDS,random_state=seed)
        if Boosting_Flag is None:
            ## Create an ensemble model ####
            model5 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                                    random_state=seed, max_depth=1, min_samples_leaf=2
                                    ), n_estimators=NUMS, random_state=seed)
            model_tuples.append(('Adaboost',model5))
        elif not Boosting_Flag:
            model5 = LinearDiscriminantAnalysis()
            model_tuples.append(('Linear_Discriminant',model5))
        else:
            model5 = LogisticRegressionCV(Cs=[0.001,0.01,0.1,1,10,100],
                                          solver='liblinear', random_state=seed)
            model_tuples.append(('Logistic_Regression_CV',model5))
        if Boosting_Flag is None:
            model6 = DecisionTreeClassifier(max_depth=5,min_samples_leaf=2)
            model_tuples.append(('Decision_Tree',model6))
        elif not Boosting_Flag:
            model6 = LinearSVC()
            model_tuples.append(('Linear_SVC',model6))
        else:
            model6 = DecisionTreeClassifier(max_depth=5,min_samples_leaf=2)
            model_tuples.append(('Decision_Tree',model6))
        if modeltype == 'Binary_Classification':
            model7 = GaussianNB()
        else:
            model7 = MultinomialNB()
        model_tuples.append(('Naive_Bayes',model7))
        if Boosting_Flag is None:
            #### If the Boosting_Flag is True, it means Boosting model is present. 
            ###   So choose a different kind of classifier here
            model8 = RandomForestClassifier(bootstrap = False,
                                       max_depth = 10,
                                       max_features = 'auto',
                                       min_samples_leaf = 2,
                                       n_estimators = 200,
                                       random_state=99)
            model_tuples.append(('Bagging_Classifier',model8))
        elif not Boosting_Flag:
            #### If the Boosting_Flag is True, it means Boosting model is present. 
            ###   So choose a different kind of classifier here
            sgd_best_model = SGDClassifier(alpha=1e-06,
                                loss='log',
                               max_iter=1000,
                               penalty='l2',
                               learning_rate = 'constant',
                               eta0 = .1,
                               random_state = 3,
                               tol=None)
            model8 = OneVsRestClassifier(sgd_best_model)
            model_tuples.append(('One_vs_Rest_Classifier',model8))
        else:
            model8 = RandomForestClassifier(bootstrap = False,
                                       max_depth = 10,
                                       max_features = 'auto',
                                       min_samples_leaf = 2,
                                       n_estimators = 200,
                                       random_state=99)
            model_tuples.append(('Bagging_Classifier',model8))
    model_dict = dict(model_tuples)
    models, results = run_ensemble_models(model_dict, X_train, y_train, X_test, y_test, 
                                          scoring, modeltype)
    return models, results
#########################################################
from sklearn.metrics import balanced_accuracy_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
def run_ensemble_models(model_dict, X_train, y_train, X_test, y_test, scoring, modeltype):   
    start_time = time.time()
    model_name,  bac_score_list, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], [], []
    iteration = 0
    estimators = []
    estim_tuples = []
    for k,v in model_dict.items():   
        estimator_name = k
        model_name.append(k)
        if str(v).split("(")[0] == 'MultinomialNB':
            #### Multinomial models need only positive values!!
            v.fit(abs(X_train), y_train)
            y_pred = v.predict(abs(X_test))
        else:
            v.fit(X_train, y_train)
            y_pred = v.predict(X_test)
        if iteration == 0:
            stacks = copy.deepcopy(y_pred)
            iteration += 1
        else:
            stacks = np.c_[stacks,y_pred]
        if not isinstance(y_test,str):
            if modeltype == 'Regression':
                bac_score = np.sqrt(mean_squared_error(y_test, y_pred))
                estimators.append((estimator_name,v, bac_score))
                estim_tuples.append((estimator_name, bac_score))
                bac_score_list.append(bac_score)
                ac_score_list.append(mean_squared_error(y_test, y_pred))
                p_score_list.append(mean_absolute_error(y_test, y_pred))
                model_comparison_df = pd.DataFrame([model_name, bac_score_list, ac_score_list,p_score_list]).T
                model_comparison_df.columns = ['model_name', 'RMSE', 'MSE','MAE']
                model_comparison_df = model_comparison_df.sort_values(by='RMSE', ascending=True)
            else:
                bac_score = balanced_accuracy_score(y_test, y_pred)
                estimators.append((estimator_name,v, bac_score))
                estim_tuples.append((estimator_name, bac_score))
                bac_score_list.append(balanced_accuracy_score(y_test, y_pred))
                ac_score_list.append(accuracy_score(y_test, y_pred))
                p_score_list.append(precision_score(y_test, y_pred, average='macro'))
                r_score_list.append(recall_score(y_test, y_pred, average='macro'))
                f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
                model_comparison_df = pd.DataFrame([model_name, bac_score_list, ac_score_list, p_score_list,
                                                         r_score_list, f1_score_list]).T
                model_comparison_df.columns = ['model_name', 'bal_accuracy_score', 'accuracy_score',
                                         'ave_precision_score', 'ave_recall_score', 'ave_f1_score']
                model_comparison_df = model_comparison_df.sort_values(by='bal_accuracy_score', ascending=False)
    if not isinstance(y_test, str):
        data_frame = model_comparison_df.set_index('model_name').astype(float)
        plt.figure(figsize=(10,10))
        g = sns.heatmap(data_frame,annot=True,fmt='0.2f',cbar=False)
        g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 12)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 12)
        print('Time taken = %0.0f seconds' %(time.time()-start_time))
        g.set_title('QuickML Ensembling Models Results',fontsize=18)
        f1_stats = dict(estim_tuples)
        try:
            if scoring in ['logloss','rmse','mae','mape','RMSE','neg_mean_squared_error']:
                best_model_name = min(f1_stats.items(), key=operator.itemgetter(1))[0]
            else:
                best_model_name = max(f1_stats.items(), key=operator.itemgetter(1))[0]
            print('Based on trying multiple models, Best type of algorithm for this data set is %s' %best_model_name)
        except:
            print('Could not detect best algorithm type from ensembling. Continuing...')
    return model_name, stacks
#########################################################
