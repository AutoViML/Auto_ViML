import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit,StratifiedKFold,KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
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
import time
import pdb
import time
import copy
import operator
#############################################################################
def accu(results, y_cv):
    return (results==y_cv).astype(int).sum(axis=0)/(y_cv.shape[0])
def rmse(results, y_cv):
    return np.sqrt(np.mean((results - y_cv)**2, axis=0))
#############################################################################
def QuickML_Ensembling(X_train, y_train, X_test, y_test='', modeltype='Regression', Boosting_Flag=False,
                            scoring='', verbose=0):
    """
    Quickly builds and runs multiple models for a clean data set(only numerics).
    """
    start_time = time.time()
    seed = 99
    if len(X_train) <= 100000 or X_train.shape[1] < 50:
        NUMS = 100
        FOLDS = 5
    else:
        NUMS = 200
        FOLDS = 10
    ## create Voting models
    estimators = []
    if modeltype == 'Regression':
        if scoring == '':
            scoring = 'neg_mean_squared_error'
        scv = ShuffleSplit(n_splits=FOLDS,random_state=seed)
        if Boosting_Flag is None:
            model5 = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                        n_estimators=NUMS,random_state=seed)
            results1 = model5.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics1 = rmse(results1, y_test).mean()
            else:
                metrics1 = 0
            estimators.append(('Bagging1',model5, metrics1))
        else:
            model5 = LassoLarsCV(cv=scv)
            results1 = model5.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics1 = rmse(results1, y_test).mean()
            else:
                metrics1 = 0
            estimators.append(('LassoLarsCV',model5, metrics1))
        model6 = LassoCV(alphas=np.logspace(-10,-1,50), cv=scv,random_state=seed)
        results2 = model6.fit(X_train,y_train).predict(X_test)
        if not isinstance(y_test, str):
            metrics2 = rmse(results2, y_test).mean()
        else:
            metrics2 = 0
        estimators.append(('LassoCV',model6, metrics2))
        model7 = RidgeCV(alphas=np.logspace(-10,-1,50), cv=scv)
        results3 = model7.fit(X_train,y_train).predict(X_test)
        if not isinstance(y_test, str):
            metrics3 = rmse(results3, y_test).mean()
        else:
            metrics3 = 0
        estimators.append(('RidgeCV',model7, metrics3))
        ## Create an ensemble model ####
        if Boosting_Flag:
            model8 = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                        n_estimators=NUMS,random_state=seed)
            results4 = model8.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics4 = rmse(results4, y_test).mean()
            else:
                metrics4 = 0
            estimators.append(('Bagging2',model8, metrics4))
        else:
            model8 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                        min_samples_leaf=2, max_depth=1, random_state=seed),
                        n_estimators=NUMS, random_state=seed)
            results4 = model8.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics4 = rmse(results4, y_test).mean()
            else:
                metrics4 = 0
            estimators.append(('Boosting',model8, metrics4))
        estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
        estimator_names = [tuples[0] for tuples in estimators]
        estim_tuples = [(estimator_names[0], metrics1),(estimator_names[1],metrics2), (
            estimator_names[2], metrics3), (estimator_names[3], metrics4)]
        if verbose > 1:
            print('QuickML_Ensembling Model results:')
            for atuple in estim_tuples: 
                    print('    %s = %0.4f' %atuple)
    else:
        if scoring == '':
            scoring = 'accuracy'
        scv = StratifiedKFold(n_splits=FOLDS,random_state=seed)
        if Boosting_Flag is None:
            model5 = ExtraTreesClassifier(n_estimators=NUMS,min_samples_leaf=2,random_state=seed)
            results1 = model5.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics1 = accu(results1, y_test).mean()
            else:
                metrics1 = 0
            estimators.append(('Bagging',model5, metrics1))
        else:
            model5 = LogisticRegressionCV(Cs=np.linspace(0.01,100,20),cv=scv,scoring=scoring,
                                          random_state=seed)
            results1 = model5.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics1 = accu(results1, y_test).mean() 
            else:
                metrics1 = 0
            estimators.append(('Logistic Regression',model5, metrics1))
        model6 = LinearDiscriminantAnalysis()
        results2 = model6.fit(X_train,y_train).predict(X_test)
        if not isinstance(y_test, str):
            metrics2 = accu(results2, y_test).mean()
        else:
            metrics2 = 0
        estimators.append(('Linear Discriminant',model6, metrics2))
        if modeltype == 'Binary_Classification':
            float_cols = X_train.columns[(X_train.dtypes==float).values].tolist()
            int_cols = X_train.columns[(X_train.dtypes==int).values].tolist()
            if (X_train[float_cols+int_cols]<0).astype(int).sum().sum() > 0:
                model7 = DecisionTreeClassifier(max_depth=5)
            else:
                model7 = GaussianNB()
        else:
            float_cols = X_train.columns[(X_train.dtypes==float).values].tolist()
            int_cols = X_train.columns[(X_train.dtypes==int).values].tolist()
            if (X_train[float_cols+int_cols]<0).astype(int).sum().sum() > 0:
                model7 = DecisionTreeClassifier(max_depth=5)
            else:
                model7 = MultinomialNB()
        results3 = model7.fit(X_train,y_train).predict(X_test)
        if not isinstance(y_test, str):
            metrics3 = accu(results3, y_test).mean()
        else:
            metrics3 = 0
        estimators.append(('Naive Bayes',model7, metrics3))
        if Boosting_Flag:
            #### If the Boosting_Flag is True, it means Boosting model is present. So choose a Bagging here.
            model8 = ExtraTreesClassifier(n_estimators=NUMS,min_samples_leaf=2,random_state=seed)
            results4 = model8.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics4 = accu(results4, y_test).mean()
            else:
                metrics4 = 0
            estimators.append(('Bagging',model8, metrics4))
        else:
            ## Create an ensemble model ####
            model8 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                                    random_state=seed, max_depth=1, min_samples_leaf=2
                                    ), n_estimators=NUMS, random_state=seed)
            results4 = model8.fit(X_train,y_train).predict(X_test)
            if not isinstance(y_test, str):
                metrics4 = accu(results4, y_test).mean()
            else:
                metrics4 = 0
            estimators.append(('Boosting',model8, metrics4))
        estimators_list = [(tuples[0],tuples[1]) for tuples in estimators]
        estimator_names = [tuples[0] for tuples in estimators]
        estim_tuples = [(estimator_names[0], metrics1),(estimator_names[1],metrics2), (
            estimator_names[2], metrics3), (estimator_names[3], metrics4)]
        if not isinstance(y_test, str):
            if verbose > 1:
                print('QuickML_Ensembling Model results:')
                for atuple in estim_tuples: 
                    print('    %s = %0.4f' %atuple)
        else:
            if verbose >= 1:
                print('QuickML_Ensembling completed.')
    stacks = np.c_[results1,results2,results3,results4]
    f1_stats = dict(estim_tuples)
    try:
        if scoring in ['logloss','rmse','mae','mape','RMSE','neg_mean_squared_error']:
            best_model_name = min(f1_stats.items(), key=operator.itemgetter(1))[0]
        else:
            best_model_name = max(f1_stats.items(), key=operator.itemgetter(1))[0]
        if verbose > 0:
            print('Based on trying multiple models, Best type of algorithm for this data set is %s' %best_model_name)
    except:
        print('Could not detect best algorithm type from ensembling. Continuing...')
    return estimator_names, stacks
#########################################################