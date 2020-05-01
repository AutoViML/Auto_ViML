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
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import SVC

import copy
import pdb
from itertools import cycle
from collections import defaultdict, Counter
import operator
from scipy import interp
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
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
    try:
        if str(model).split("(")[0] == 'MultinomialNB':
            #### Multinomial models need only positive values!!
            model.fit(abs(X_train_dtm), y_train)
            y_preds = model.predict(abs(X_cv_dtm)) 
        else:
            model.fit(X_train_dtm, y_train)
            y_preds = model.predict(X_cv_dtm) 
    except:
        if str(model).split("(")[0] == 'MultinomialNB':
            #### Multinomial models need only positive values!!
            model.fit(abs(X_train_dtm.toarray()), y_train)
            y_preds = model.predict(abs(X_cv_dtm.toarray())) 
        else:
            model.fit(X_train_dtm.toarray(), y_train)
            y_preds = model.predict(X_cv_dtm.toarray()) 
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
import regex as re

#Expand all these terms
expand_dict = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "I would",
  "i'd've": "I would have",
  "i'll": "I will",
  "i'll've": "I will have",
  "i'm": "I am",
  "i've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(expand_dict.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return expand_dict[match.group(0)]
    return c_re.sub(replace, text)

def expand_text(text):
    expanded = [expandContractions(item, c_re=c_re) for item in text]
    return ''.join(map(str, expanded))
#######################################################################
def clean_text_using_regex(text):

    # Special characters
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"åÀ", "", text)
    text = re.sub(re.compile('<.*?>'), ' ', text)

    # Expand Text
    text = text.replace("i'll","i will").replace("i'm","i am").replace("i've","i have").replace(
    "n't"," not").replace("let's","let us").replace("'re"," are").replace("'d","did").replace(
    "'em","them").replace("y'all","you all").replace("it's","it is").replace("'"," ").replace(
    '"',' ').replace(" s "," ")
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)
    text = re.sub(r"donå«t", "do not", text)

    # Character entity references
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&amp;", "&", text)

    # Typos, slang and informal abbreviations
    text = re.sub(r"w/e", "whatever", text)
    text = re.sub(r"w/", "with", text)
    text = re.sub(r"USAgov", "USA government", text)
    text = re.sub(r"recentlu", "recently", text)
    text = re.sub(r"Ph0tos", "Photos", text)
    text = re.sub(r"amirite", "am I right", text)
    text = re.sub(r"exp0sed", "exposed", text)
    text = re.sub(r"<3", "love", text)
    text = re.sub(r"amageddon", "armageddon", text)
    text = re.sub(r"Trfc", "Traffic", text)
    text = re.sub(r"WindStorm", "Wind Storm", text)
    text = re.sub(r"lmao", "laughing my ass off", text)
    # Urls
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)

    ### remove numbers
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, ' %s ' %p)

    # ... and ..
    text = text.replace('...', ' ... ')
    if '...' not in text:
        text = text.replace('..', ' ... ')
    return text

import regex as re

def remove_punctuations(text):
    remove_puncs = re.sub(r'[?|!|~|@|$|%|^|&|#]', r'', text).lower()
    return re.sub(r'[.|,|)|(|\|/|+|-|{|}|]', r' ', remove_puncs)

def strip_out_special_chars(txt):
    return re.compile("[^\w']|_").sub(" ",txt)

def remove_stop_words(text):
    stopWords = return_stop_words()
    return text.map(lambda x: x.split(" ")).map(lambda row: " ".join([x for x in row if x not in stopWords]))

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
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
import nltk
def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
################################################################################
def return_stop_words():
    add_words = ["s", "m",'you', 'not',  'get', 'no', 'if', 'via', 'one', 'still', 'us']
    from sklearn.feature_extraction import text
    #stopWords = text.ENGLISH_STOP_WORDS.union(add_words)
    stopWords = set(stopwords.words('english')).union(add_words)
    excl =['will',"i'll",'shall',"you'll",'may',"don't","hadn't","hasn't","haven't",
           "don't","isn't",'if',"mightn't","mustn'","mightn't",'mightn',"needn't",
           'needn',"needn't",'no','not','shan',"shan't",'shouldn',"shouldn't","wasn't",
          'wasn','weren',"weren't",'won',"won't",'wouldn',"wouldn't","you'd",'you',
          "you'd","you'll","you're",'yourself','yourselves']
    stopWords = [x for x in stopWords if x not in excl]
    return stopWords
################################################################################
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
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
    nltk.download("popular")
    stopWords = return_stop_words()
    ######################################################
    #### This calculates based on the average number of words in an NLP column how many max_features
    min_df = 2
    max_df = 0.95
    ######################################################
    if len(data) >= 1000000:
        max_features = 1000
    elif len(data) >= 100000:
        max_features = 500
    else:
        max_features = int(data[col].map(len).mean()*4)
    print('    A U T O - N L P   P R O C E S S I N G  O N   N L P   C O L U M N = %s ' %col)
    print('#################################################################################')
    print('Generating new features for NLP column = %s using NLP Transformers' %col)
    print('    However min_df and max_features = %d will limit too many features from being generated' %max_features)
    ################################################################
    if modeltype is None or modeltype == '':
        print('Since modeltype is None, Using TFIDF vectorizer with min_df and max_features')
        tvec = TfidfVectorizer(ngram_range=(1,3), stop_words=stopWords, max_features=max_features, min_df=min_df,max_df=max_df)
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
    tokenizer = RegexpTokenizer(r'\b[a-z|A-Z]{2,}\b')
    print('Using low min_df = %d for all Vectorizers' %min_df)
    vect_5000 = CountVectorizer(
               ngram_range=(1, 3), max_features=max_features, max_df=max_df,
                strip_accents='unicode',
                min_df=min_df, binary=False, stop_words=None, token_pattern=r'\w{1,}')
    all_vecs[vect_5000], all_models[vect_5000] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                    y_test, target, metric,
                      vect_5000, seed, modeltype)
    print('\n#### Using Count Vectorizer with limited max_features and a low 0.001 min_df with n_gram (1-5)')
    ##########################################################################
    ##### It's BEST to use small max_features (50) and a low 0.001 min_df with n_gram (2-5).
    ######  There is no need in that case for stopwords or analyzer since the 2-grams take care of it
    #### Once you do above, there is no difference between count_vectorizer and tfidf_vectorizer
    #### Once u do above, increasing max_features from 50 to even 500 doesn't get you a higher score!
    ##########################################################################
    vect_lemma = CountVectorizer(max_df=max_df,
                                   max_features=max_features, strip_accents='unicode',
                                   ngram_range=(1, 5), token_pattern=r'\w{1,}',
                                    min_df=min_df, stop_words=stopWords,
                                   binary=True,
                                    )
    try:
        all_vecs[vect_lemma], all_models[vect_lemma] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                           y_test, target, metric,
                                             vect_lemma, seed, modeltype)
    except:
        print('Error: Using CountVectorizer')

    print('\n# Using TFIDF vectorizer with min_df=2 and very high max_features')
    ##### This is based on artificially setting 5GB as being max memory limit for the term-matrix
    max_features_high = int(250000000/X_train.shape[0])
    if modeltype != 'Regression':
        tvec = TfidfVectorizer( max_features=max_features_high,max_df=max_df, token_pattern=r'\w{1,}',
                                strip_accents='unicode',
                                stop_words=stopWords, ngram_range=(1, 3), min_df=min_df, binary=True)
    else:
        tvec = TfidfVectorizer( max_features=max_features_high, max_df=max_df, token_pattern=r'\w{1,}',
                                strip_accents='unicode',
                                stop_words=stopWords, ngram_range=(1, 3), min_df=min_df, binary=False)
    all_vecs[tvec], all_models[tvec] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec, seed, modeltype)
    max_features_limit = int(tvec.fit_transform(data_dtm).shape[1])
    print('\n# Using TFIDF vectorizer with Snowball Stemming and limited max_features')
    if modeltype != 'Regression':
        tvec2 = TfidfVectorizer( max_features=max_features, max_df=max_df,
                                token_pattern=r'\w{1,}',
                                 min_df=min_df, stop_words=None, binary=True, strip_accents='unicode',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    else:
        tvec2 = TfidfVectorizer( max_features=max_features, max_df=max_df,
                                token_pattern=r'\w{1,}',
                                 min_df=min_df, stop_words=None, binary=False, strip_accents='unicode',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    all_vecs[tvec2], all_models[tvec2] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec2, seed, modeltype)

    ######## Once you have built 4 different transformers it is time to compare them
    if modeltype.endswith('Classification'):
        best_vec = pd.Series(all_vecs).idxmax()
    else:
        if modeltype == 'Regression':
            best_vec = pd.Series(all_vecs).idxmin()
        else:
            print('Error: Modeltype not recognized. You must choose Classification or Regression or None')
            return
    print('\nBest NLP technique selected is: \n%s' %best_vec)
    data_dtm = best_vec.transform(data_dtm)
    return best_vec, all_models[best_vec], data_dtm, max_features_limit

############################################################################
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,mean_squared_error
def return_scoreval(scoretype, y_true, y_preds, y_proba, modeltype):
    if modeltype.endswith('Classification'):
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
                scoreval = balanced_accuracy_score(y_true, y_preds)
                print('Multi-class problem. Instead of ROC-AUC, Balanced Accuracy computed')
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
        elif scoretype in ['balanced_accuracy','accuracy','balanced-accuracy']:
            try:
                scoreval = balanced_accuracy_score(y_true, y_preds)
            except:
                scoreval = accuracy(y_true, y_preds)
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
        elif scoretype == 'mae':
            try:
                scoreval = np.sqrt(mean_absolute_error(y_true, y_preds))
            except:
                scoreval = 0
        else:
            print('Scoring Type not Recognized.')
            scoretype == 'mae'
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
from sklearn.decomposition import TruncatedSVD
def select_top_features_from_SVD(X, tsvd, is_train=True, top_n=100):
    """
    This program returns the top X features from a TFIDF or CountVectorizer on a dataset.
    You just need to send in the Vectorized data set X and along with a number denoting
    how many top features you want back. It will automatically assume you want the top 100.
    You can change the top X features to any number you want. But it must be less than the
    number of features in X. Otherwise, it will assume you want all.
    """
    X = copy.deepcopy(X)
    start_time = time.time()
    #### If the shape of the TFIDF array is huge in the thousands of terms,
    ####   then you select the top 25 terms in 1-gram and 2-gram that make sense.
    print('Reducing dimensions from %d term-matrix to %d dimensions using TruncatedSVD...' %(X.shape[1],top_n))
    if is_train:
        tsvd = TruncatedSVD(n_components=top_n,
                   n_iter=10,
                   random_state=3)
        tsvd = tsvd.fit(X)
    XA = tsvd.transform(X)
    print('    Reduced dimensional array shape to %s' %(XA.shape,))
    print('    Time Taken for Truncated SVD = %0.0f seconds' %(time.time()-start_time) )
    return XA, tsvd
###########################################################################################
def print_top_feature_grams(X, vectorizer, top_n = 25):
    """
    This prints the top features by each n-gram using the vectorizer that is selected as best!
    """
    X = copy.deepcopy(X)
    vectorizer = copy.deepcopy(vectorizer)
    for i in range(1,4):
        vectorizer.ngram_range=(i, i)
        XA = vectorizer.fit_transform(X)
        feature_array = vectorizer.get_feature_names()
        top_sorted_tuples = sorted(list(zip(vectorizer.get_feature_names(),
                                                     XA.sum(0).getA1())),
                                         key=lambda x: x[1], reverse=True)[:top_n]
        top_sorted = [x for (x,y) in  top_sorted_tuples]
        print("Top %d %d-gram\n: %s" %(top_n, i, top_sorted))
###########################################################################################
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
    #### Convert the Feature Array from a Sparse Matrix to a Dense Array #########
    XA = X.toarray()
    if XA.shape[1] <= grams_length:
        #### If the TFIDF array is very small, you just select the entire TFIDF array
        best_features_array = copy.deepcopy(XA)
        best_df = pd.DataFrame(best_features_array.todense(),columns=vectorizer.get_feature_names())
    else:
        #### If the shape of the TFIDF array is huge in the thousands of terms,
        ####   then you select the top 25 terms in 1-gram and 2-gram that make sense.
        print('    Transformed data...')
        iteration = 1
        ls = []
        best_features_array = XA[:]
        ##### There are instances where dicti is empty or has only 1-grams. So test to make sure!
        if len(dicti) > 0:
            for i,key in enumerate(dicti):
                for eachterm in dicti[key]:
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

#########################################################################################
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
def Auto_NLP(nlp_column, train, test, target, score_type,
                            modeltype,top_num_features=50, verbose=0):
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
    seed = 99
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)
    start_time = time.time()
    num_classes = len(np.unique(train[target].values))
    top_num_features = num_classes*top_num_features
    min_df=0.1
    max_depth = 8
    subsample =  0.5
    col_sub_sample = 0.5
    test_size = 0.2
    seed = 1
    early_stopping = 5
    train_index = train.index
    ######################################
    if isinstance(nlp_column, str):
        cols_excl_nlp_cols = [x for x in list(train) if x not in  [nlp_column]]
    else:
        cols_excl_nlp_cols = [x for x in list(train) if x not in nlp_column]
    nlp_result_columns = []
    #############   THIS IS WHERE WE START PROCESSING NLP COLUMNS #####################
    if type(nlp_column) == str:
        pass
    elif type(nlp_column) == list:
        nlp_column = nlp_column[0]
    else:
        print('NLP column must be either a string or a list with one column name in data frame')
        return
    ########################  S U M M A R Y  C O L U M N S  C R E A T I O N ######################
    #### Now let's do a combined transformation of NLP column
    train, nlp_summary_cols = create_summary_of_nlp_cols(train, nlp_column, target, is_train=True, verbose=verbose)
    nlp_result_columns += nlp_summary_cols
    print('    Added %d summary columns for counts of words and characters in each row' %len(nlp_summary_cols))
    if not isinstance(test, str):
        #### You don't want to draw the same set of charts for Test data since it would be repetitive
        #####   Hence set the verbose to 0 in this case !!!
        test, nlp_summary_cols = create_summary_of_nlp_cols(test, nlp_column, target, is_train=False, verbose=0)
    ########################  C L E AN    C O L U M N S   F I R S T ######################
    print('    Cleaning text in %s before doing transformation...' %nlp_column)
    start_time = time.time()
    ####### CLEAN THE DATA FIRST ###################################
    train[nlp_column] = train[nlp_column].map(expand_text).values
    train[nlp_column] = train[nlp_column].map(clean_text_using_regex).map(strip_out_special_chars).values
    train[nlp_column] = remove_stop_words(train[nlp_column])
    if not isinstance(test, str):
        test[nlp_column] = test[nlp_column].map(expand_text).values
        test[nlp_column] = test[nlp_column].map(clean_text_using_regex).map(strip_out_special_chars).values
        test[nlp_column] = remove_stop_words(test[nlp_column])
    print('Train and Test data Text cleaning completed. Time taken = %d seconds' %(time.time()-start_time))
    ##########################################################################################
    if modeltype.endswith('Classification'):
        #print('Class distribution in Train:')
        #class_info(train[target])
        if len(Counter(train[target])) > 2:
            model = MultinomialNB()
        else:
            model = GaussianNB()
        best_nlp_vect, model, train_dtm, max_features_limit = select_best_nlp_vectorizer(model, train, nlp_column, target,
                    score_type, seed, modeltype,min_df)
    elif modeltype == 'Regression':
        if float(xgb.__version__[0])<1:
            objective = 'reg:linear'
        else:
            objective = 'reg:squarederror'
        model = XGBRegressor( n_estimators=100,subsample=subsample,objective=objective,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                 seed=1,n_jobs=-1,random_state=1)
        ####   This is where you start to Iterate on Finding Important Features ################
        best_nlp_vect, model, train_dtm, max_features_limit = select_best_nlp_vectorizer(model, train, nlp_column, target,
                            score_type, seed, modeltype,min_df)
    else:
        #### Just complete the transform of NLP column and return the transformed data ####
        model = None
        best_nlp_vect, model, train_dtm, max_features_limit = select_best_nlp_vectorizer(model, train, nlp_column, target,
                            score_type, seed, modeltype,min_df)
    #### Now that the Best VECTORIZER has been selected, transform Train and Test and return vectorized dataframes
    #### Convert the Feature Array from a Sparse Matrix to a Dense Array #########
    print('Setting Max Features limit to NLP vectorizer as %d' %max_features_limit)
    best_nlp_vect.max_features = max_features_limit
    train_all = best_nlp_vect.fit_transform(train[nlp_column])
    if train_all.shape[1] <= top_num_features:
        #### If the TFIDF array is very small, you just select the entire TFIDF array
        train_best = pd.DataFrame(train_all.todense(),columns=best_nlp_vect.get_feature_names())
    else:
        #### For Train data, you don't have to send in an SVD. It will automatically select one and train it
        best_features_array, tsvd = select_top_features_from_SVD(train_all,'',True)
        ls = ['svd_dim_'+str(x) for x in range(best_features_array.shape[1])]
        train_best = pd.DataFrame(best_features_array,columns=ls)
    #train_best = select_top_features_from_vectorizer(train_dtm, best_nlp_vect, )
    print_top_feature_grams(train[nlp_column], best_nlp_vect)
    if type(test) != str:
        test_index = test.index
        #### If test data is given, then convert it into a Vectorized frame using best vectorizer
        test_all = best_nlp_vect.transform(test[nlp_column])
        if test_all.shape[1] <= top_num_features:
            test_best = pd.DataFrame(test_all.todense(),columns=best_nlp_vect.get_feature_names())
        else:
            best_features_array, _ = select_top_features_from_SVD(test_all,tsvd,False)
            test_best = pd.DataFrame(best_features_array,columns=ls)
    #test_best = select_top_features_from_vectorizer(test_dtm, best_nlp_vect, top_num_features)
    else:
        test_best = ''
    #### best contains the entire data rows with the top X features of a Vectorizer
    #### from an NLP analysis. This means that you can add these top NLP features to
    #### your data set and start performing your classification or regression.
    #################################################################################
    nlp_result_columns = left_subtract(list(train_best), cols_excl_nlp_cols)
    print('Completed selecting the best NLP transformer. Time taken = %0.1f minutes' %((time.time()-start_time)/60))
    train_best = train_best.set_index(train_index)
    #################################################################################
    #### train_best contains the entire data rows with the top X features of a Vectorizer
    #### from an NLP analysis. This means that you can add these top NLP features to
    #### your data set and start performing your classification or regression.
    train_best = train_best.fillna(0)
    train_nlp = train.join(train_best,rsuffix='NLP_token_added')
    train_nlp['auto_nlp_source'] = 'Train'
    if type(test) != str:
        test_best = test_best.set_index(test_index)
        test_best = test_best.fillna(0)
        test_nlp = test.join(test_best, rsuffix='NLP_token_added')
        test_nlp[target] = 0
        test_nlp['auto_nlp_source'] = 'Test'
        nlp_data = train_nlp.append(test_nlp)
    else:
        nlp_data = copy.deepcopy(train_nlp)
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
        nlp_data[senti_cols[0]] = nlp_data[nlp_column].map(detect_sentiment).fillna(0)
        nlp_data[senti_cols[1]] = nlp_data[nlp_column].map(calculate_line_sentiment,'polarity').fillna(0)
        nlp_data[senti_cols[2]] = nlp_data[nlp_column].map(calculate_line_sentiment,'subjectivity').fillna(0)
        nlp_data[senti_cols[3]] = nlp_data[nlp_column].map(calculate_paragraph_sentiment).fillna(0)
        nlp_result_columns += senti_cols
        print('    Added %d columns using TextBlob Sentiment Analyzer. Time Taken = %d seconds' %(
                                    len(senti_cols), time.time()-start_time2))
    ##### Just do a fillna of all NLP columns in case there are some NA's in them ###########
    #nlp_data[nlp_result_columns] = nlp_data[nlp_result_columns].apply(lambda x: x.fillna(0)
    #                        if x.dtype.kind in 'biufc' else x.fillna('missing'))
    ######### Split it back into train_best and test_best ##########
    train_source = nlp_data[nlp_data['auto_nlp_source']=='Train']
    train_full = train_source.drop(['auto_nlp_source',nlp_column],axis=1)
    if type(test) == str:
        test_full = ''
    else:
        test_source = nlp_data[nlp_data['auto_nlp_source']=='Test']
        test_full = test_source.drop([target,'auto_nlp_source',nlp_column],axis=1)
    print('Number of new columns created using NLP = %d' %(len(nlp_result_columns)))
    print('         A U T O   N L P  C O M P L E T E D. Time taken %0.1f minutes' %((time.time()-start_time)/60))
    print('####################################################################################')
    return train_full, test_full, best_nlp_vect, max_features_limit
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
    try:
        return TextBlob(text.decode('utf-8')).sentiment.polarity
    except:
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
                    lambda txt: analyzer.polarity_scores(txt)['neg']).fillna(0)
    data[nlp_column+'_vader_pos'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['pos']).fillna(0)
    data[nlp_column+'_vader_neutral'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['neu']).fillna(0)
    data[nlp_column+'_vader_compound'] = data[nlp_column].map(
                    lambda txt: analyzer.polarity_scores(txt)['compound']).fillna(0)
    cols = [nlp_column+'_vader_neg',nlp_column+'_vader_pos',nlp_column+'_vader_neu',nlp_column+'_vader_compound']
    print('    Created %d new columns using SentinmentIntensityAnalyzer. Time taken = %d seconds' %(len(cols),time.time()-start_time))
    return data, cols
######### Create new columns that provide summary stats of NLP string columns
def create_summary_of_nlp_cols(data, col, target, is_train=False, verbose=0):
    """
    Create new columns that provide summary stats of NLP string columns
    This gives us insights into the number of characters we want in our NLP column
    in order for the column to be relevant to the target.
    This can also be a Business question. It may help us in building a better predictive model.
    """
    cols = []
    stop_words = return_stop_words()
    # word_count
    data[col+'_word_count'] = data[col].apply(lambda x: len(str(x).split(" ")))
    cols.append(col+'_word_count')
    # unique_word_count
    data[col+'_unique_word_count'] = data[col].apply(lambda x: len(set(str(x).split(" "))))
    cols.append(col+'_unique_word_count')
    # stop_word_count
    data[col+'_stop_word_count'] = data[col].apply(lambda x: len([w for w in str(x).lower().split(" ") if w in stop_words]))
    cols.append(col+'_stop_word_count')
    # url_count
    data[col+'_url_count'] = data[col].apply(lambda x: len([w for w in str(x).lower().split(" ") if 'http' in w or 'https' in w]))
    cols.append(col+'_url_count')
    # mean_word_length
    try:
      data[col+'_mean_word_length'] = data[col].apply(lambda x: int(np.mean([len(w) for w in str(x).split(" ")])))
      cols.append(col+'_mean_word_length')
    except:
      print('Error: Cannot create word length in %s due to NaNs in data' %col)
    # char_count
    try:
      data[col+'_char_count'] = data[col].apply(lambda x: len(str(x)))
      cols.append(col+'_char_count')
    except:
      print('Error: Cannot create char count in %s due to NaNs in data' %col)
    # punctuation_count
    data[col+'_punctuation_count'] = data[col].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    cols.append(col+'_punctuation_count')
    # hashtag_count
    data[col+'_hashtag_count'] = data[col].apply(lambda x: len([c for c in str(x) if c == '#']))
    cols.append(col+'_hashtag_count')
    # mention_count
    data[col+'_mention_count'] = data[col].apply(lambda x: len([c for c in str(x) if c == '@']))
    cols.append(col+'_mention_count')
    if verbose >= 1:
        if is_train:
            plot_nlp_column(data[col+'_unique_word_count'],"Word Count")
    if verbose > 1:
        if is_train:
            plot_nlp_column(data[col+'_char_count'],"Character Count")
    if verbose >= 2:
        if is_train:
            draw_dist_plots_summary_cols(data, target, cols)
    return data, cols
#############################################################################
def plot_nlp_column(df_col,label_title):
    """
    We want to know the average number of words per row of text.
    So we first plot the distribution of number of words per text row.
    """
    color_string = 'bryclg'
    cnt_srs = df_col.value_counts()
    plt.figure(figsize=(12,6))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color_string[0])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Distribution of %s in newly created %s column' %(label_title,df_col.name),fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show();
#############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, combinations
def draw_dist_plots_summary_cols(df_train, target, summary_cols):
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgkbyr')
    target_names = np.unique(df_train[target])
    ncols =2
    nrows = int((len(summary_cols)/2)+0.50)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,nrows*6), dpi=100)
    axs = []
    for i in range(nrows):
        for j in range(ncols):
            axs.append('axes['+str(i)+','+str(j)+']')
    labels = []
    for axi, feature in enumerate(summary_cols):
        for target_name in target_names:
            label = str(target_name)
            color = next(colors)
            sns.distplot(df_train.loc[df_train[target] == target_name][feature],
                         label=label,
                     ax=eval(axs[axi]), color=color, kde_kws={'bw':1.5})
            labels.append(label)
    plt.legend(labels=labels)
    plt.show();
#############################################################################
def plot_histogram_probability(dist_train, dist_test, label_title):
    pal = 'bryclg'
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
version_number = '0.0.27'
print("""Imported Auto_NLP version: %s.. Call using:
     train_nlp, test_nlp, best_nlp_transformer, _ = Auto_NLP(
                nlp_column, train, test, target, score_type,
                modeltype,top_num_features=50, verbose=0)""" %version_number)
########################################################################
