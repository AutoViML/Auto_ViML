import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn import metrics
from sklearn import model_selection, metrics   #Additional sklearn functions
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss
from sklearn.metrics import mean_squared_error,median_absolute_error,mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_log_error
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score   
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
#####################################################################################
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
#####################################################################################
def accu(results, y_cv):
    return (results==y_cv).astype(int).sum(axis=0)/(y_cv.shape[0])
def rmse(results, y_cv):
    return np.sqrt(np.mean((results - y_cv)**2, axis=0))
######## Defining objective functions for HyperOpt here ######################
def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

def gini_meae(truth, predictions):
    score = median_absolute_error(truth, predictions)
    return score

def gini_msle(truth, predictions):
    score = mean_squared_log_error(truth, predictions)
    return score

def gini_mae(truth, predictions):
    score = mean_absolute_error(truth, predictions)
    return score

def gini_mse(truth, predictions):
    score = mean_squared_error(truth, predictions)
    return score

def gini_rmse(truth, predictions):
    score = np.sqrt(mean_squared_error(truth, predictions))
    return score

def gini_accuracy(truth, predictions):
    return accuracy_score(truth, predictions)

def gini_bal_accuracy(truth, predictions):
    return balanced_accuracy_score(truth, predictions)

def gini_roc(truth, predictions):
    return roc_auc_score(truth, predictions)

def gini_precision(truth, predictions,pos_label=1):
    return precision_score(truth, predictions,average=None)[pos_label]

def gini_average_precision(truth, predictions):
    return average_precision_score(truth, predictions.argmax(axis=1),average='weighted')

def gini_weighted_precision(truth, predictions):
    return precision_score(truth, predictions.argmax(axis=1),average='weighted')

def gini_macro_precision(truth, predictions):
    return precision_score(truth, predictions.argmax(axis=1),average='macro')

def gini_micro_precision(truth, predictions):
    return precision_score(truth, predictions.argmax(axis=1),average='micro')

def gini_samples_precision(truth, predictions):
    return precision_score(truth, predictions.argmax(axis=1),average='samples')

def gini_f1(truth, predictions,pos_label=1):
    return f1_score(truth, predictions,average=None)[pos_label]

def gini_weighted_f1(truth, predictions):
    return f1_score(truth, predictions.argmax(axis=1),average='weighted')

def gini_macro_f1(truth, predictions):
    return f1_score(truth, predictions.argmax(axis=1),average='macro')

def gini_micro_f1(truth, predictions):
    return f1_score(truth, predictions.argmax(axis=1),average='micro')

def gini_samples_f1(truth, predictions):
    return f1_score(truth, predictions.argmax(axis=1),average='samples')

def gini_log_loss(truth, predictions):
    return log_loss(truth, predictions,normalize=True)

def gini_recall(truth, predictions,pos_label=1):
    return recall_score(truth, predictions,average=None)[pos_label]

def gini_weighted_recall(truth, predictions):
    return recall_score(truth, predictions.argmax(axis=1),average='weighted')

def gini_samples_recall(truth, predictions):
    return recall_score(truth, predictions.argmax(axis=1),average='samples')

def gini_macro_recall(truth, predictions):
    return recall_score(truth, predictions.argmax(axis=1),average='macro')

def gini_micro_recall(truth, predictions):
    return recall_score(truth, predictions.argmax(axis=1),average='micro')

# calculate f2-measure
def f2_measure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)