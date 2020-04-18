################################################################################
####                       Auto NLP version 1.1                             ####
####                      Developed by Ram Seshadri                         ####
####                        All Rights Reserved                             ####
################################################################################
#### Auto NLP applies NLP processing techniques on a dataset with one variable##
#### You cannot give a dataframe with multiple string variables as only one ####
####  is allowed. It splits the dataset into train and test and returns     ####
####  predictions on both for Classification or Regression.                 ####
################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
import pdb
import time
import pprint
import matplotlib
matplotlib.style.use('ggplot')

from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,make_scorer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

### For NLP problems
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
from nltk.corpus import stopwords
import string

#### For Classification problems
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



#### For Regression problems
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,make_scorer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection, metrics   #Additional sklearn functions
from sklearn.model_selection import GridSearchCV   #Performing grid search
from sklearn.model_selection import cross_val_predict, cross_val_score
from blagging import BlaggingClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import SVC

import copy
import pdb
from itertools import cycle
from collections import defaultdict, Counter
import operator
from scipy import interp

############################################################################
# define a function that accepts a vectorizer and calculates its accuracy
def tokenize_test_by_metric(model, X_train, X_cv, y_train, y_cv,
    target, metric, vect=None, seed=99, modeltype='Classification'):
    if vect==None:
        # use default options for CountVectorizer
        vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)
    print('Features: ', X_train_dtm.shape[1])
    print_sparse_stats(X_train_dtm)
    X_cv_dtm = vect.transform(X_cv)
    model.fit(X_train_dtm, y_train)
    y_preds = model.predict(X_cv_dtm)
    # calculate return_scoreval for score_type
    metric_val = return_scoreval(metric, y_cv, y_preds, '', modeltype)
    print('%s Metrics for %s = %0.2f' %(metric, X_train.shape,
                metric_val))
    return metric_val, model

###########  NLP functions #####################
# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    #text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]
############################################################################
from nltk.corpus import stopwords
def strip_out_special_chars(txt):
    return re.compile("[^\w']|_").sub(" ",txt)
#Simple text cleaning functions that are very fast!
import re
def remove_html(text):
    return re.sub(re.compile('<.*?>'), ' ', text)

def remove_punctuations(text):
    remove_puncs = re.sub(r'[?|!|~|@|$|%|^|&|#]', r'', text).lower()
    return re.sub(r'[.|,|)|(|\|/|+|-|{|}|]', r' ', remove_puncs)

def expand_text(text):
    return text.replace("i'll","i will").replace("i'm","i am").replace("i've","i have").replace(
    "n't"," not").replace("let's","let us").replace("'re"," are").replace("'d","did").replace(
    "'em","them").replace("y'all","you all").replace("it's","it is").replace("'"," ").replace(
    '"',' ').replace(" s "," ")
def remove_stopwords(txt):
    """
    Takes in an array, so it is very fast. But must be Vectorized!!
    1. Removes all stopwords
    """
    return " ".join([txt if txt not in stopwords.words('english') else ""])
############################################################################
def print_sparse_stats(X_dtm):
    """
    Prints the stats around a Sparse Matrix (typically) generated in NLP problems.
    """
    print ('Shape of Sparse Matrix: ', X_dtm.shape)
    print ('Amount of Non-Zero occurences: ', X_dtm.nnz)
    print ('    Density: %.2f%%' % (100.0 * X_dtm.nnz /
                                 (X_dtm.shape[0] * X_dtm.shape[1])))
############################################################################
from sklearn.feature_extraction import text 
def select_best_nlp_vectorizer(model, data, col, target, metric,
                    seed, modeltype,min_df):
    """
    ################################################################################
    #### VERY IMPORTANT: CountVectorizer can only deal with one Column at a Time!!
    ####  SO YOU MUST NOT SEND AN ENTIRE DATAFRAME AND EXPECT IT TO VECTORIZE. IT BLOWS UP!
    #### Hence we repeatedly send one NLP column after another for Vectorizing and
    ####   find the best NLP technique that yields the highest CV metric. Metrics could be:
    ####   Accuracy, AUC, F1, Precision, and Recall. Having min_df as 10% (i.e. 0.1) is
    ####   a  good idea, since it results in much cleaner and much better terms selected.
    ################################################################################
    """
    stop_words = text.ENGLISH_STOP_WORDS
    if len(data) >= 1000000:
        max_features = 1000
    elif len(data) >= 100000:
        max_features = 100
    else:
        try:
            max_features = data.shape[1]*10
        except:
            max_features =  data.shape[0]
    print('\n####################################################################################')
    print('Generating new features for NLP column = %s using NLP Transformers' %col)
    print('    However only the top %d generated features will be selected' %max_features)
    print('    Cleaning text in %s before doing transformation...' %col)
    start_time = time.time()
    ####### CLEAN THE DATA FIRST ###################################
    data[col] =  data[col].map(remove_html).map(remove_punctuations).map(expand_text)
    #### To make removing stop words fast we need to run it through a vectorizer!
    #remover = lambda txt: txt if txt not in stopwords.words('english') else ""
    #vectorized_func = np.vectorize(remover)
    #vectorized_func = np.vectorize(remove_stopwords)
    #data[col] =  data[col].map(lambda x: vectorized_func(np.array(x.split(" "))))
    data[col] = data[col].map(strip_out_special_chars)
    print('Text cleaning completed. Time taken = %d seconds' %(time.time()-start_time))
    ################################################################
    if modeltype is None or modeltype == '':
        print('Since modeltype is None, Using TFIDF vectorizer with min_df and max_features')
        tvec = TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=int(max_features*0.5), min_df=min_df)
        data_dtm =  data[col]
        data_dtm = tvec.fit_transform(data_dtm)
        print('Features: ', data_dtm.shape[1])
        print_sparse_stats(data_dtm)
        #data_dense = convert_sparse_to_dense(data_dtm)
        return tvec, data_dtm
    else:
        data_dtm =  data[col]
    ##### Then do a test train split using data and NLP column called "col" #########
    X_train,X_test,y_train,y_test = train_test_split(data[col],
                                    data[target],test_size=0.2,random_state=seed)
    best_vec = None
    all_vecs = {}
    all_models = {}
    print('\n#### First choosing the default Count Vectorizer with 1-3 ngrams and limited features')
    for min_df in sorted(np.linspace(0.10,0.01,10)):
        try:
            vect_5000 = CountVectorizer(ngram_range=(1, 3), max_features=int(max_features*0.5),
                                min_df=min_df, binary=False, stop_words='english')
            all_vecs[vect_5000], all_models[vect_5000] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                            y_test, target, metric,
                              vect_5000, seed, modeltype)
            print('Best min_df is %0.2f' %min_df)
            break
        except:
            continue
    print('\n#### Using lemmas for feature extraction, with min_df and features (WARNING: SLOW!)')
    vect_lemma = CountVectorizer(analyzer=split_into_lemmas, max_features=int(max_features*0.5), ngram_range=(1, 3), 
                    min_df=min_df, binary=True,stop_words='english')
    try:
        all_vecs[vect_lemma], all_models[vect_lemma] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                           y_test, target, metric,
                                             vect_lemma, seed, modeltype)
    except:
        print('Vectorizer with Lemma and Min_df skipped')

    print('\n# Using TFIDF vectorizer with min_df and max_features')
    if modeltype == 'Classification':
        tvec = TfidfVectorizer( max_features=int(max_features*0.5), 
                                stop_words='english', ngram_range=(1, 3), min_df=min_df, binary=True)
    else:
        tvec = TfidfVectorizer( max_features=int(max_features*0.5), 
                                stop_words='english', ngram_range=(1, 3), min_df=min_df, binary=False)
    all_vecs[tvec], all_models[tvec] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec, seed, modeltype)
    ######## Once you have built 3 different transformers it is time to compare them
    if modeltype != 'Regression':
        best_vec = pd.Series(all_vecs).idxmax()
    else:
        if modeltype.endswith('Classification'):
            best_vec = pd.Series(all_vecs).idxmin()
        else:
            print('Error: Modeltype not recognized. You must choose Classification or Regression or None')
            return
    print('\nBest NLP technique selected is: \n%s' %best_vec)
    data_dtm = best_vec.transform(data_dtm)
    return best_vec, all_models[best_vec], data_dtm, min_df

############################################################################
def return_scoreval(scoretype, y_true, y_preds, y_proba, modeltype):
    if modeltype == 'Classification':
        if scoretype == 'f1':
            try:
                scoreval = f1_score(y_true, y_preds)
            except:
                scoreval = f1_score(y_true, y_preds, average = 'micro')
        elif scoretype == 'roc_auc':
            #### ROC AUC can be computed only for Binary classifications ###
            try:
                scoreval = roc_auc_score(y_true, y_proba)
            except:
                scoreval = 0
                print('Single class present. No ROC-AUC computed')
        elif scoretype == 'precision':
            try:
                scoreval = precision_score(y_true, y_preds)
            except:
                scoreval = precision_score(y_true, y_preds, average='micro')
        elif scoretype == 'recall':
            try:
                scoreval = recall_score(y_true, y_preds)
            except:
                scoreval = recall_score(y_true, y_preds, average='micro')
        else:
            print('Scoring Type not Recognized - selecting default as F1.')
            scoretype == 'f1'
            try:
                scoreval = f1_score(y_true, y_preds)
            except:
                scoreval = f1_score(y_true, y_preds, average='micro')
    else:
        if scoretype == 'rmse':
            try:
                scoreval = np.sqrt(mean_squared_error(y_true, y_preds))
            except:
                scoreval = 0
        else:
            print('Scoring Type not Recognized.')
            scoretype == 'abs'
            scoreval = mean_absolute_error(y_true, y_preds)
    return scoreval
######### Print the % count of each class in a Target variable  #####
def class_info(classes):
    """
    Only works on Binary variables. Prints class percentages count of target variable.
    It returns the number of instances of the RARE (or minority) Class.
    """
    counts = Counter(classes)
    total = sum(counts.values())
    for cls in counts.keys():
        print("%6s: % 7d  =  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))

###################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
def select_top_features_from_vectorizer(X, vectorizer,top_n=25):
    """
    This program returns the top X features from a TFIDF or CountVectorizer on a dataset.
    You just need to send in the Vectorized data set X and the Vectorizer itself along with
    how many top features you want back. It will automatically assume you want the top 25.
    You can change the top X features to any number you want. But it must be less than the
    number of features in X. Otherwise, it will assume you want all.
    """
    features_by_gram = defaultdict(list)
    if str(vectorizer).split("(")[0] == 'CountVectorizer':
        for f, w in zip(vectorizer.get_feature_names(), vectorizer.vocabulary_):
            features_by_gram[len(f.split(' '))].append((f, w))        
    else:
        for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
            features_by_gram[len(f.split(' '))].append((f, w))
    dicti = {}
    grams_length = 0
    for gram, features in features_by_gram.items():
        top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [f[0] for f in top_features]
        print('{}-gram top features:'.format(gram), top_features)
        dicti[gram] = top_features
        grams_length +=  len(dicti[gram])
    #### Convert the Feature Array from a Sparse Matrix to a Dense Array #########
    XA = X.toarray()
    if XA.shape[1] <= grams_length:
        #### If the TFIDF array is very small, you just select the entire TFIDF array
        best_features_array = copy.deepcopy(XA)
        ls = sorted(vectorizer.vocabulary_, key=lambda x: x[0], reverse=False)
        best_df = pd.DataFrame(best_features_array,columns=ls)
    else:
        #### If the shape of the TFIDF array is huge in the thousands of terms,
        ####   then you select the top 25 terms in 1-gram and 2-gram that make sense.
        print('Transformed data...')
        iteration = 1
        ls = []
        best_features_array = XA[:]
        ##### There are instances where dicti is empty or has only 1-grams. So test to make sure!
        if len(dicti) > 0:
            for i in range(1,len(dicti)+1):
                for eachterm in dicti[i]:
                    if iteration == 1:
                        try:
                            index = vectorizer.vocabulary_[eachterm]
                            best_features_array = XA[:,index]
                            ls.append(eachterm)
                            iteration += 1
                        except:
                            pass
                    else:
                        try:
                            index = vectorizer.vocabulary_[eachterm]
                            best_features_array = np.c_[best_features_array, XA[:,index]]
                            ls.append(eachterm)
                            iteration += 1
                        except:
                            pass
        best_df = pd.DataFrame(best_features_array,columns=ls)
    print('Combined best features array shape: %s' %(best_features_array.shape,))
    return best_df

####################################################################
def Auto_NLP(nlp_column, train, test, target, score_type,
                            seed, modeltype,top_num_features=25):
    """
    ##################################################################################
    #### Auto_NLP expects both train and test to be data frames with one NLP column 
    ####  and one target.
    #### It uses the sole NLP_column to analyze, train and predict the target. The predictions
    #### train and test are returned. If no target is given, it just analyzes NLP column.
    #### VERY IMPORTANT: CountVectorizer can only deal with one NLP Column at a Time!!
    #### SO DONT SEND AN ENTIRE DATAFRAME AND EXPECT IT TO VECTORIZE. IT WILL BLOW UP!
    #### I have selected min_df to be 10% (i.e. 0.1) to select the best features from NLP.
    #### You can make it smaller to return higher number of features and vice versa.
    #### You can use top_num_features (default = 25) to control how many features to add.
    ##################################################################################
    """
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)
    start_time = time.time()
    min_df=0.1
    if type(nlp_column) == str:
        pass
    elif type(nlp_column) == list:
        nlp_column = nlp_column[0]
    else:
        print('NLP column must be either a string or a list with one column name in data frame')
        return
    if modeltype.endswith('Classification'):
        #print('Class distribution in Train:')
        #class_info(train[target])
        if len(Counter(train[target])) > 2:
            model = MultinomialNB()
        else:
            model = LogisticRegression()
        best_nlp_vect, model, train_dtm, min_df = select_best_nlp_vectorizer(model, train, nlp_column, target,
                    score_type, seed, modeltype,min_df)
    elif modeltype == 'Regression':
        if train.shape[0] < 100000:
            model = RandomForestRegressor(n_estimators=50, max_depth=6,random_state=seed)
        else:
            model = BaggingRegressor(n_estimators=200,max_samples=0.1,random_state=seed)
            best_nlp_vect, model, train_dtm, min_df = select_best_nlp_vectorizer(model, train, nlp_column, target,
                            score_type, seed, modeltype,min_df)
    else:
        #### Just complete the transform of NLP column and return the transformed data ####
        model = None
        best_nlp_vect, model, train_dtm, min_df = select_best_nlp_vectorizer(model, train, nlp_column, target,
                            score_type, seed, modeltype,min_df)
    #### Now that the Best VECTORIZER has been selected, transform Train and Test and return vectorized dataframes
    train_best = select_top_features_from_vectorizer(train_dtm, best_nlp_vect, top_num_features)
    if type(test) != str:
        #### If test data is given, then convert it into a Vectorized frame using best vectorizer
        test_dtm = best_nlp_vect.transform(test[nlp_column])
        test_best = select_top_features_from_vectorizer(test_dtm, best_nlp_vect, top_num_features)
    else:
        test_best = ''
    #### best contains the entire data rows with the top X features of a Vectorizer
    #### from an NLP analysis. This means that you can add these top NLP features to
    #### your data set and start performing your classification or regression.
    print('Completed NLP_Column_Vectorizer. Time taken = %0.1f minutes' %((time.time()-start_time)/60))
    #### train_best contains the entire data rows with the top X features of a Vectorizer
    #### from an NLP analysis. This means that you can add these top NLP features to
    #### your data set and start performing your classification or regression.
    train['source'] = 'Train'
    if type(test) != str:
        test[target] = 0
        test['source'] = 'Test'
        nlp_data = train.append(test)
    else:
        nlp_data = copy.deepcopy(train)
    #### Now let's do a combined transformation of NLP column
    nlp_data, nlp_result_columns = create_summary_of_nlp_cols(nlp_data, nlp_column)
    #### next create parts-of-speech tagging ####
    if len(nlp_data) <= 10000:
        ### VADER is accurate but very SLOOOWWW. Do not do this for Large data sets ##############
        nlp_data, pos_cols = add_sentiment(nlp_data, nlp_column)
        nlp_result_columns += pos_cols
    else:
        ### TEXTBLOB is faster but somewhat less accurate. So we do this for Large data sets ##############
        print('Using TextBlob to add sentiment scores...warning: could be slow for large data sets')
        senti_cols = [nlp_column+'_text_sentiment', nlp_column+'_senti_polarity',
                                nlp_column+'_senti_subjectivity',nlp_column+'_overall_sentiment']
        start_time2 = time.time()
        nlp_data[senti_cols[0]] = nlp_data[nlp_column].map(detect_sentiment)
        nlp_data[senti_cols[1]] = nlp_data[nlp_column].map(calculate_line_sentiment,'polarity')
        nlp_data[senti_cols[2]] = nlp_data[nlp_column].map(calculate_line_sentiment,'subjectivity')
        nlp_data[senti_cols[3]] = nlp_data[nlp_column].map(calculate_paragraph_sentiment)
        nlp_result_columns += senti_cols
        print('    Added %d columns using TextBlob Sentiment Analyzer. Time Taken = %d seconds' %(
                                    len(senti_cols), time.time()-start_time2))
    ##### Just do a fillna of all NLP columns in case there are some NA's in them ###########
    nlp_data[nlp_result_columns] = nlp_data[nlp_result_columns].apply(lambda x: x.fillna(0)
                            if x.dtype.kind in 'biufc' else x.fillna('missing'))
    print('Number of new columns created using NLP = %d' %(len(nlp_result_columns)))
    ######### Split it back into train_best and test_best ##########
    train_source = nlp_data[nlp_data['source']=='Train']
    train_source = train_source.drop('source',axis=1)
    train_best = train_best.join(train_source[nlp_result_columns])
    if type(test) == str:
        test_best = ''
    else:
        test_source = nlp_data[nlp_data['source']=='Test']
        test_source = test_source.drop([target,'source'],axis=1)
        test_best = test_best.join(test_source[nlp_result_columns])
    print('    Completed Auto_NLP. Time taken %0.1f minutes' %((time.time()-start_time)/60))
    train_full = train.drop(nlp_column, axis=1, inplace=False).join(train_best,rsuffix='_OVERLAPPING_COLUMN_DELETE')
    train_full = train_full.drop('source',axis=1)
    if type(test) != str:
        test_full = test.drop(nlp_column, axis=1, inplace=False).join(test_best,rsuffix='_OVERLAPPING_COLUMN_DELETE')
        test_full = test_full.drop('source',axis=1)
    else:
        test_full = ''
    return train_full, test_full, best_nlp_vect
##############################################################################################
def calculate_line_sentiment(text,senti_type='polarity'):
    review = TextBlob(text)
    review_totals = []
    for each_sentence in review.sentences:
        if senti_type == 'polarity':
            review_totals.append(each_sentence.sentiment.polarity)
        else:
            review_totals.append(each_sentence.sentiment.subjectivity)
    return np.mean(review_totals)
########################################################################
#### Do a sentiment analysis of whole review text rather than line by line ##
def calculate_paragraph_sentiment(text):
    return TextBlob(text).sentiment.polarity
########################################################################
########## define a function that accepts text and returns the polarity
def detect_sentiment(text):
    try:
        return TextBlob(text.decode('utf-8')).sentiment.polarity
    except:
        return TextBlob(text).sentiment.polarity

##############################################################################################
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def add_sentiment(data, nlp_column):
    """
    ############ Parts of SPeech Tagging using Spacy   ################################
    ### We will now use the text column to calculate the sentiment in each to
    ### assign an average objectivity score and positive vs. negative scores.
    ### If a word cannot be found in the dataset we can ignore it. If a
    ### text has no words that match something in our dataset, we can
    ### assign an overall neutral score of 'objectivity = 1' and 'pos_vs_neg of 0'.
    #######   This is to be done only where data sets are small <10K rows. ####
    """
    start_time = time.time()
    print('Using Vader to calculate objectivity and pos-neg-neutral scores')
    analyzer = SentimentIntensityAnalyzer()
    data[nlp_column+'_vader_neg'] = 0
    data[nlp_column+'_vader_pos'] = 0
    data[nlp_column+'_vader_neu'] = 0
    data[nlp_column+'_vader_compound'] = 0
    data[nlp_column+'_vader_neg'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['neg'])
    data[nlp_column+'_vader_pos'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['pos'])
    data[nlp_column+'_vader_neutral'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['neu'])
    data[nlp_column+'_vader_compound'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['compound'])
    cols = [nlp_column+'_vader_neg',nlp_column+'_vader_pos',nlp_column+'_vader_neu',nlp_column+'_vader_compound']
    print('    Created %d new columns using SentinmentIntensityAnalyzer. Time taken = %d seconds' %(len(cols),time.time()-start_time))
    return data, cols
######### Create new columns that provide summary stats of NLP string columns
def create_summary_of_nlp_cols(data, col,verbose=0):
    """
    Create new columns that provide summary stats of NLP string columns
    This gives us insights into the number of characters we want in our NLP column
    in order for the column to be relevant to the target.
    This can also be a Business question. It may help us in building a better predictive model.
    """
    data[col+"_num_of_words"] = data[col].apply(lambda x : len(str(x).split()))
    if verbose == 1:
        plot_nlp_column(data[col+"_num_of_words"],"words")
    data[col+"_num_of_chars"] = data[col].apply(lambda x : len(str(x)))
    if verbose == 1:
        plot_nlp_column(data[col+"_num_of_chars"],"characters")
    return data, [col+"_num_of_words",col+"_num_of_chars"]
#############################################################################
def plot_nlp_column(df_col,label_title):
    """
    We want to know the average number of words per row of text.
    So we first plot the distribution of number of words per text row.
    """
    cnt_srs = df_col.value_counts()
    plt.figure(figsize=(12,6))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Number of %s in %s' %(label_title,df_col.name),fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show();

def plot_histogram_probability(dist_train, dist_test, label_title):
    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
    plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
    plt.title('Normalised histogram of %s count in questions' %label_title, fontsize=12)
    plt.legend()
    plt.xlabel('Number of %s' %label_title, fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show();
########################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.10'
print("""Imported Auto_NLP version: %s.. Call using:
     train_nlp, test_nlp, best_nlp_transformer = Auto_NLP(nlp_column, train, test, target, score_type, seed, modeltype)""" %version_number)
########################################################################
