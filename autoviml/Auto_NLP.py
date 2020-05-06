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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV

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
    target, metric, vect=None, seed=99, modeltype='Classification',verbose=0):
    if vect==None:
        # use default options for CountVectorizer
        vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)
    if verbose >= 1:
        print('Features: ', X_train_dtm.shape[1])
        print_sparse_stats(X_train_dtm)
    X_cv_dtm = vect.transform(X_cv)
    if str(model).split("(")[0] == 'MultinomialNB':
        try:
            #### Multinomial models need only positive values!!
            model.fit(abs(X_train_dtm), y_train)
            y_preds = model.predict(abs(X_cv_dtm))
            if modeltype != 'Regression':
                y_probas = model.predict_proba(abs(X_cv_dtm))
        except:
            #### Multinomial models need only positive values!!
            model.fit(abs(X_train_dtm.toarray()), y_train)
            y_preds = model.predict(abs(X_cv_dtm.toarray()))
            if modeltype != 'Regression':
                y_probas = model.predict_proba(abs(X_cv_dtm.toarray()))
    else:
        try:
            model.fit(X_train_dtm, y_train)
            y_preds = model.predict(X_cv_dtm)
            if modeltype != 'Regression':
                y_probas = model.predict_proba(X_cv_dtm)
        except:
            model.fit(X_train_dtm.toarray(), y_train)
            y_preds = model.predict(X_cv_dtm.toarray())
            if modeltype != 'Regression':
                y_probas = model.predict_proba(X_cv_dtm.toarray())
    # calculate return_scoreval for score_type
    if modeltype != 'Regression':
        metric_val = return_scoreval(metric, y_cv, y_preds, y_probas, modeltype)
    else:
        metric_val = return_scoreval(metric, y_cv, y_preds, '', modeltype)
    print('    %s Metrics for %s features = %0.4f' %(metric, X_train_dtm.shape[1],
                  metric_val))
    return metric_val, model
###########  NLP functions #####################
# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    #text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]
############################################################################
####   The below Process_Text section is Re-used with Permission from:
####  R O B   S A L G A D O    robert.salgado@gmail.com     Thank YOU!
# https://github.com/robsalgado/personal_data_science_projects/tree/master/mulitclass_text_class
#############################################################################
import itertools, string, operator, re, unicodedata, nltk
from operator import itemgetter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from bs4 import BeautifulSoup
import numpy as np
from itertools import combinations
from gensim.models import Phrases
from collections import Counter
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
############################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
def return_stop_words():
    from nltk.corpus import stopwords
    add_words = ["s", "m",'you', 'not',  'get', 'no', 'if', 'via', 'one', 'still', 'us', 'u', 'if',
                'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'for', 'on', 'it', 'got',
                'not', 'my', 'that', 'by', 'with', 'are', 'at', 'this', 'from', 'be', 'have', 'was']
    from sklearn.feature_extraction import text
    #stopWords = text.ENGLISH_STOP_WORDS.union(add_words)
    stop_words = set(set(stopwords.words('english')).union(add_words))
    excl =['will',"i'll",'shall',"you'll",'may',"don't","hadn't","hasn't","haven't",
           "don't","isn't",'if',"mightn't","mustn'","mightn't",'mightn',"needn't",
           'needn',"needn't",'no','not','shan',"shan't",'shouldn',"shouldn't","wasn't",
          'wasn','weren',"weren't",'won',"won't",'wouldn',"wouldn't","you'd",'you',
          "you'd","you'll","you're",'yourself','yourselves']
    stopWords = left_subtract(stop_words,excl)
    return sorted(stopWords)

stop_words = return_stop_words()
################################################################################

c_re = re.compile('(%s)' % '|'.join(expand_dict.keys()))

tokenizer = TweetTokenizer()
pattern = r"(?u)\b\w\w+\b"

lemmatizer = WordNetLemmatizer()

punc = list(set(string.punctuation))

def casual_tokenizer(text): #Splits words on white spaces (leaves contractions intact) and splits out trailing punctuation
    tokens = tokenizer.tokenize(text)
    return tokens

#Function to replace the nltk pos tags with the corresponding wordnet pos tag to use the wordnet lemmatizer
def get_word_net_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemma_wordnet(tagged_text):
    final = []
    for word, tag in tagged_text:
        wordnet_tag = get_word_net_pos(tag)
        if wordnet_tag is None:
            final.append(lemmatizer.lemmatize(word))
        else:
            final.append(lemmatizer.lemmatize(word, pos=wordnet_tag))
    return final

def expandContractions(text, c_re=c_re):
    def replace(match):
        return c_dict[match.group(0)]
    return c_re.sub(replace, text)

def remove_html(text):
    soup = BeautifulSoup(text, "html5lib")
    tags_del = soup.get_text()
    uni = unicodedata.normalize("NFKD", tags_del)
    bracket_del = re.sub(r'\[.*?\]', '  ', uni)
    apostrphe = re.sub('’', "'", bracket_del)
    string = apostrphe.replace('\r','  ')
    string = string.replace('\n','  ')
    extra_space = re.sub(' +',' ', string)
    return extra_space

def process_text(text):
    soup = BeautifulSoup(text, "lxml")
    tags_del = soup.get_text()
    no_html = re.sub('<[^>]*>', '', tags_del)
    tokenized = casual_tokenizer(no_html)
    lower = [item.lower() for item in tokenized]
    decontract = [expandContractions(item, c_re=c_re) for item in lower]
    tagged = nltk.pos_tag(decontract)
    lemma = lemma_wordnet(tagged)
    no_num = [re.sub('[0-9]+', '', each) for each in lemma]
    no_punc = [w for w in no_num if w not in punc]
    no_stop = [w for w in no_punc if w not in stop_words]
    return no_stop
############################################################################
####   THE ABOVE Entire Process_Text secion Re-used with Permission from:
####  R O B   S A L G A D O    robert.salgado@gmail.com Thank YOU!
#####################################################################/
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
    #text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

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

############################################################################
def print_sparse_stats(X_dtm):
    """
    Prints the stats around a Sparse Matrix (typically) generated in NLP problems.
    """
    print ('Shape of Sparse Matrix: ', X_dtm.shape)
    print ('Amount of Non-Zero occurences: ', X_dtm.nnz)
    print ('    Density: %.2f%%' % (100.0 * X_dtm.nnz /
                                 (X_dtm.shape[0] * X_dtm.shape[1])))
################################################################################
import nltk
def remove_stop_words(text):
    stopWords = return_stop_words()
    return text.map(lambda x: x.split(" ")).map(lambda row: " ".join([x for x in row if x not in stopWords]))

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
import nltk
def simple_tokenizer(text):
    # create a space between special characters
    text=re.sub("(\\W)"," \\1 ",text)

    # split based on whitespace
    return re.split("\\s+",text)

from nltk.stem import PorterStemmer

# Get the Porter stemmer
porter_stemmer=PorterStemmer()
import regex as re
def simple_preprocessor(text):

    text=text.lower()
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words

    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)
#########################################################################################
def lower_words(text):
    return text.map(lambda x: x.split(" ")).map(lambda row: " ".join([x.lower() for x in row ]))
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
    print('    Cleaning text in %s before doing transformation...' %col)
    start_time = time.time()
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
    max_features_high = int(250000000/X_train.shape[0])
    print('    However max_features limit = %d will limit numerous features from being generated' %max_features_high)
    best_vec = None
    all_vecs = {}
    all_models = {}
    if data.shape[0] < 10000:
        count_max_df = 0
        print('Trying multiple max_df values in range %s to find best max_df...' %np.linspace(0.95,0.05,5))
        for each_max_df in np.linspace(0.95,0.05,5):
            print('    max_df = %0.4f     ' %each_max_df, end='')
            vect_5000 = CountVectorizer(
                       ngram_range=(1, 3), max_features=max_features_high, max_df=each_max_df,
                        strip_accents='unicode', tokenizer=simple_tokenizer,preprocessor=simple_preprocessor,
                        min_df=min_df, binary=False, stop_words=None, token_pattern=r'\w{1,}')
            current_metric, current_model = tokenize_test_by_metric(model, X_train, X_test, y_train,
                            y_test, target, metric,
                              vect_5000, seed, modeltype,verbose=0)
            if count_max_df == 0:
                best_metric = copy.deepcopy(current_metric)
                best_model = copy.deepcopy(current_model)
            else:
                if modeltype == 'Regression' or metric in ['logloss','log_loss']:
                    if current_metric <= best_metric:
                        best_metric = copy.deepcopy(current_metric)
                        best_model = copy.deepcopy(current_model)
                    else:
                        break
                else:
                    if current_metric >= best_metric:
                        best_metric = copy.deepcopy(current_metric)
                        best_model = copy.deepcopy(current_model)
                    else:
                        break
            count_max_df += 1
        best_max_df = each_max_df + 0.20
        print('Best max_df selected to be %0.2f' %best_max_df)
    else:
        best_max_df = 0.5
        print('\n#### Optimizing Count Vectorizer with best max_df=%0.2f, 1-3 n-grams and high features...' %best_max_df)
        vect_5000 = CountVectorizer(
                   ngram_range=(1, 3), max_features=max_features_high, max_df=best_max_df,
                    strip_accents='unicode', tokenizer=simple_tokenizer,preprocessor=simple_preprocessor,
                    min_df=min_df, binary=False, stop_words=None, token_pattern=r'\w{1,}')
        best_metric, best_model = tokenize_test_by_metric(model, X_train, X_test, y_train,
                        y_test, target, metric,
                          vect_5000, seed, modeltype,verbose=0)
    #### You have to set the best max df to the recent one plus 0.05 since it breaks when the metric drops
    vect_5000.max_df = best_max_df
    all_vecs[vect_5000] = best_metric
    all_models[vect_5000] = best_model
    ##########################################################################
    ##### It's BEST to use small max_features (50) and a low 0.001 min_df with n_gram (2-5).
    ######  There is no need in that case for stopwords or analyzer since the 2-grams take care of it
    #### Once you do above, there is no difference between count_vectorizer and tfidf_vectorizer
    #### Once u do above, increasing max_features from 50 to even 500 doesn't get you a higher score!
    ##########################################################################
    print('\n#### Using Count Vectorizer with limited max_features and a min_df=%s with n_gram (1-5)' %min_df)
    vect_lemma = CountVectorizer(max_df=best_max_df,
                                   max_features=max_features_high, strip_accents='unicode',
                                   ngram_range=(1, 5), token_pattern=r'\w{1,}',
                                    min_df=min_df, stop_words=None,
                                   binary=True,
                                    )
    try:
        all_vecs[vect_lemma], all_models[vect_lemma] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                           y_test, target, metric,
                                             vect_lemma, seed, modeltype)
    except:
        print('Error: Using CountVectorizer')

    print('\n# Using TFIDF vectorizer with min_df=%s, ngram (1,3) and very high max_features' %min_df)
    ##### This is based on artificially setting 5GB as being max memory limit for the term-matrix
    if modeltype != 'Regression':
        tvec = TfidfVectorizer( max_features=max_features_high, max_df=best_max_df, token_pattern=r'\w{1,}',
                                strip_accents='unicode', sublinear_tf=True,
                                stop_words=None, ngram_range=(1, 3), min_df=min_df, binary=True)
    else:
        tvec = TfidfVectorizer( max_features=max_features_high, max_df=best_max_df, token_pattern=r'\w{1,}',
                                strip_accents='unicode', sublinear_tf=True,
                                stop_words=None, ngram_range=(1, 3), min_df=min_df, binary=False)
    all_vecs[tvec], all_models[tvec] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec, seed, modeltype)
    max_features_limit = int(tvec.fit_transform(data_dtm).shape[1])
    print('\n# Using TFIDF vectorizer with Porter Stemming, ngram (1,3) and limited max_features')
    if modeltype != 'Regression':
        tvec2 = TfidfVectorizer( max_features=max_features_high, max_df=best_max_df,
                                token_pattern=r'\w{1,}', sublinear_tf=True,
                                tokenizer=simple_tokenizer,preprocessor=simple_preprocessor,
#                                 tokenizer=tokenize_and_stem,
                                 min_df=min_df, stop_words=None, binary=True, strip_accents='unicode',
                                 use_idf=True, ngram_range=(1,3))
    else:
        tvec2 = TfidfVectorizer( max_features=max_features_high, max_df=best_max_df,
                                token_pattern=r'\w{1,}', sublinear_tf=True,
                                tokenizer=simple_tokenizer,preprocessor=simple_preprocessor,
#                                 tokenizer=tokenize_and_stem,
                                 min_df=min_df, stop_words=None, binary=False, strip_accents='unicode',
                                 use_idf=True, ngram_range=(1,3))
    all_vecs[tvec2], all_models[tvec2] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec2, seed, modeltype)

    #print('\n# Using TFIDF vectorizer with Snowball Stemming, ngram (1,3) and very high max_features')
    print('\n# Finally comparing them against a Basic Count Vectorizer with all defaults, max_features = %d and  lowercase=True' %max_features_high)
    cvect = CountVectorizer(min_df=2, lowercase=True, max_features=max_features_high)
    all_vecs[cvect], all_models[cvect] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                      cvect, seed, modeltype)
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
        tsvd.fit(X)
    XA = tsvd.transform(X)
    print('    Reduced dimensional array shape to %s' %(XA.shape,))
    print('    Time Taken for Truncated SVD = %0.0f seconds' %(time.time()-start_time) )
    return XA, tsvd
###########################################################################################
def print_top_features(train,nlp_column, best_nlp_vect, target, top_nums=200):
    """
    #### This can be done only for C L A S S I F I C A T I O N   Data Sets ####################
    This is an alternate way to look at classification tasks in NLP. Itis a simple technique
    1. First separate samples into labels belonging to each class
    1. Run a countvectorizer on each sample set.
    1. Then append the sample sets and let there be nans where the word columns don't match
    1. Fill those NaN's with 0's.
    1. Now take this combined entity and run it through a Multinomial GaussianNB
    1. See if the results are any better since the top words have different counts for different classes
    ###########################################################################################
    """
    #### Make sure that you do this for only small data sets ####################
    max_samples = min(10000,train.shape[0])
    buswo = train.sample(max_samples, random_state=99)
    classes = np.unique(buswo[target])
    orig_vect = copy.deepcopy(best_nlp_vect)
    df_names = []
    classes_copy = copy.deepcopy(classes)
    for itera in classes_copy:
        if not isinstance(itera, str):
            new_df_name = 'df_'+str(itera)
        else:
            new_df_name = 'df_'+itera
        print('%s is about class=%s of shape: ' %(new_df_name,itera), end='')
        new_df_name = buswo[(buswo[target]==itera)]
        df_names.append(new_df_name)
        print(new_df_name.shape)
    #### now we split into as many data frames as there are classes in the train data set
    df_dtms = []
    count_df = 0
    count_bus_wo = pd.DataFrame()
    for each_df, each_class in zip(df_names,classes):
        print('\nFor class = %s' %each_class)
        eachdf_index = each_df.index
        each_df[nlp_column] = lower_words(each_df[nlp_column])
        each_df[nlp_column] = each_df[nlp_column].map(expand_text).values
        each_df[nlp_column] = each_df[nlp_column].map(clean_text_using_regex).map(strip_out_special_chars).values
        each_df[nlp_column] = each_df[nlp_column].apply(process_text).map(lambda x: " ".join(x)).astype(str)
        #### This is for the 0 class - notice how the top words are different!
        #### After cleaning now you have to set the Countvectorizer ####
        cv = copy.deepcopy(best_nlp_vect)
        top_num_feats = print_top_feature_grams(each_df[nlp_column], cv, top_nums)
        #### This is an Alternative Method to get Top Num features ###########
        #top_num_feats =set([" ".join(x.split("_")) for x in word_freq_bigrams(bus_texts,int(top_nums*1/2))[0].values]+[
        #                    " ".join(x.split("_")) for x in bigram_freq(bus_texts,int(top_nums*1/3))[0].values])
        print('    Top n-grams will be used as features in the data set of size: %d' %len(top_num_feats))
        #### Once you do that set it as the vocab and get a dataframe built with those vocabs
        if len(top_num_feats) > 0:
            ### If it is zero, you want to skip setting the vocab to zero. Hence this check!
            cv.vocabulary = top_num_feats
            cv.fit(each_df[nlp_column])
            each_df_dtm = cv.transform(each_df[nlp_column])
            each_df_dtm = pd.DataFrame(each_df_dtm.toarray(),index=eachdf_index,
                                     columns=cv.get_feature_names())
            print('    Completed building data frame of shape: %s' %(each_df_dtm.shape,))
            df_dtms.append(each_df_dtm)
            if count_df == 0:
                count_bus_wo = copy.deepcopy(each_df_dtm)
            else:
                count_bus_wo = count_bus_wo.append(each_df_dtm).sort_index().fillna(0).astype(int)
            print(count_bus_wo.shape)
        count_df += 1
    print('Printed top n-grams for each class in the train data set')
    #### this is where all the columns from multiple classes are put into one set ###
    all_class_feats = count_bus_wo.columns.tolist()
    # This is where you combine dataframe with its original target to create a new data frame
    try:
        buswo_dtm = count_bus_wo.join(buswo[target])
    except:
        ### there must be another variable with the name target so try again
        buswo_dtm = (count_bus_wo.drop(target,axis=1)).join(buswo[target])
    print(buswo_dtm.shape)
    #### now return the original vectorizer with all_class_feats as the vocab
    ###  You can now use this orig_vect to fit and transform train and test datasets
    #### This will mean that both your train and test will contain the same columns
    #### buswo_dtm represents the new train data set that has been transformed by cv
    orig_vect.vocabulary = all_class_feats
    return buswo_dtm, all_class_feats
###########################################################################################
import collections
def print_top_feature_grams(X, vectorizer, top_n = 200):
    """
    This prints the top features by each n-gram using the vectorizer that is selected as best!
    """
    X = copy.deepcopy(X)
    vectorizer = copy.deepcopy(vectorizer)
    all_sorted = []
    for i in range(1,4):
        #### set min_df to be low so that you catch at least a few  of them
        try:
            if i == 1:
                top_num = int(top_n*2/3)
            elif i == 2:
                top_num = int(top_n*1/6)
            else:
                top_num = int(top_n*1/6)
            vectorizer.ngram_range = (i,i)
            XA = vectorizer.fit_transform(X)
            feature_array = vectorizer.get_feature_names()
            top_sorted_tuples = sorted(list(zip(vectorizer.get_feature_names(),
                                                         XA.sum(0).getA1())),
                                             key=lambda x: x[1], reverse=True)[:top_num]
            top_sorted = [x for (x,y) in  top_sorted_tuples]
            all_sorted += top_sorted
        except:
            pass
    ### the reason you want to do "set" after each time is because you have some duplicates.
    ### after removing spaces, you may find more duplicates, hence you do it twice
    all_sorted_set = sorted(set(all_sorted), key=all_sorted.index)
    all_sorted_final = sorted(set(remove_unicode_strings(all_sorted_set)),
                                  key=remove_unicode_strings(all_sorted_set).index)
    print("Top %d n-grams\n: %s" %(top_n, all_sorted_final))
    return all_sorted_final
###########################################################################################
def remove_unicode_strings(lst):
    try:
        ret_lst = []
        for string_var in lst:
            clean_string = "".join(["" if ord(x) > 127 else x for x in string_var])
            clean_string = " ".join(clean_string.split())
            if len(clean_string) == 0:
                pass
            else:
                ret_lst.append(clean_string)
        return ret_lst
    except:
        return lst
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
    X = copy.deepcopy(X)
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import time
import scipy as sp
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
def Auto_NLP(nlp_column, train, test, target, score_type='',
                            modeltype='Classification',
                            top_num_features=200, verbose=0,
                            build_model=True):
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
    start_time4 = time.time()
    start_time = time.time()
    print('Auto NLP processing on NLP Column: %s' %nlp_column)
    print('Shape of Train Data: %d rows' %train.shape[0])
    if not isinstance(test, str):
        print('    Shape of Test Data: %d rows' %test.shape[0])
    ########   Set the number of top NLP features that will be generated #########
    if top_num_features == 50:
      if modeltype == 'Regression':
        top_num_features = 100
      else:
        num_classes = len(np.unique(train[target].values))
        top_num_features = num_classes*top_num_features
    ###### Set Scoring parameter score_type here ###############
    if score_type == '':
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
    elif score_type in ['f1','precision','average_precision','recall','average_recall','roc_auc']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
    elif score_type in ['rmse','mae','mean_squared_error','mean_absolute_error','mean_absolute_percentage_error']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
    elif score_type in ['neg_log_loss', 'logloss','log_loss']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'neg_log_loss'
    ###### Set Defaults for cross-validation size data  here ###############
    if train.shape[0] <= 1000:
        test_size = 0.1
    elif train.shape[0] > 1000 and train.shape[0] <= 10000:
        test_size = 0.15
    else:
        test_size = 0.2
    ###### Set Other Defaults  here ###############
    min_df=0.1
    max_depth = 8
    subsample =  0.7
    col_sub_sample = 0.7
    seed = 99
    early_stopping = 5
    n_splits = 5
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
    ##########################################################################################
    ########################  C L E AN    C O L U M N S   F I R S T ######################
    print('    Cleaning text in %s before doing transformation...' %nlp_column)
    start_time1 = time.time()
    ####### CLEAN THE DATA FIRST ###################################
    train[nlp_column] = lower_words(train[nlp_column])
    train[nlp_column] = train[nlp_column].map(expand_text).values
    train[nlp_column] = train[nlp_column].map(clean_text_using_regex).map(strip_out_special_chars).values
    train[nlp_column] =  remove_stop_words(train[nlp_column])
    train[nlp_column] = train[nlp_column].apply(process_text).map(lambda x: " ".join(x)).astype(str)
    if not isinstance(test, str):
        test[nlp_column] = lower_words(test[nlp_column])
        test[nlp_column] = test[nlp_column].map(expand_text).values
        test[nlp_column] = test[nlp_column].map(clean_text_using_regex).map(strip_out_special_chars).values
        test[nlp_column] =  remove_stop_words(test[nlp_column])
        test[nlp_column] = test[nlp_column].apply(process_text).map(lambda x: " ".join(x)).astype(str)
    print('Train and Test data Text cleaning completed. Time taken = %d seconds' %(time.time()-start_time1))
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
                                 seed=1,n_jobs=-1,random_state=seed)
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
    if modeltype == 'Regression':
        print_top_feature_grams(train[nlp_column], best_nlp_vect, top_num_features)
    else:
        ### Do this only for priting top words n-grams by classes since each class may be different
        print_top_features(train, nlp_column, best_nlp_vect, target, top_num_features)
    print('Time taken so far = %0.1f minutes' %((time.time()-start_time1)/60))
    #############   THIS IS WHERE WE USE BUILD_MODEL TO DECIDE ################################
    if build_model:
        print('##################    THIS IS FOR BUILD_MODEL = TRUE           #################')
        print('Building Model and Pipeline for NLP column = %s. This will take time...' %nlp_column)
        if isinstance(best_nlp_vect, str):
            print('    Using Cross-Validation to build best model and pipeline using default Vectorizer for optimizing %s' %score_type)
            cvect = CountVectorizer(min_df=2, lowercase=True)
        else:
            print('    Using Cross-Validation to build best model and pipeline using Best Vectorizer for optimizing %s' %score_type)
            cvect = copy.deepcopy(best_nlp_vect)
        ### Split into Train and CV to test the model #####################
        X = train[nlp_column]
        y = train[target]
        #Train test split with stratified sampling for evaluation
        if modeltype == 'Regression':
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size = test_size,
                                                                random_state=seed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            shuffle = True,
                                                            stratify = y,
                                                            random_state=seed)
        ############  THIS IS WHERE THE MAIN LOGIC TO SPEED UP BOTH MODEL AND PIPELINE BEGINS! ########
        print('Transforming train and cross validation data sets into Vectorized form. This will take time...')
        start_time = time.time()
        X_train_dtm = cvect.fit_transform(X_train)
        X_test_dtm =  cvect.transform(X_test)
        print('    Time taken to transform train data into vectorized data = %0.2f seconds' %(time.time()-start_time) )
        print('    Train Vectorized data shape = %s, Cross Validation data shape = %s' %(X_train_dtm.shape, X_test.shape))
        if modeltype == 'Regression':
            model_name = 'XGB Regressor'
            scv = KFold(n_splits=n_splits, random_state=seed)
            nlp_model = XGBRegressor(learning_rate=0.1,subsample=subsample,max_depth=10,
                                colsample_bytree=col_sub_sample,reg_alpha=0.5, reg_lambda=0.5,
                                seed=1,n_jobs=-1,random_state=seed)
            low, high = 100, 400
            params = {}
            params['learning_rate'] = sp.stats.uniform(scale=1)
            params['n_estimators'] = sp.stats.randint(low,high)
        else:
            scv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            model_name = 'Multinomial Naive Bayes'
            nlp_model = MultinomialNB()
            params = {}
            params['alpha'] = sp.stats.uniform(scale=1)
        gs = RandomizedSearchCV(nlp_model,params, n_iter=10, cv=scv,
                                scoring=score_type, random_state=seed)
        gs.fit(X_train_dtm,y_train)
        y_pred = gs.predict(X_test_dtm)
        ##### Print the model results on Cross Validation data set (held out)
        if modeltype == 'Regression':
            print_regression_model_stats(y_test, y_pred,'%s Model: Predicted vs Actual for %s' %(model_name,target))
        else:
            plot_confusion_matrix(y_test, y_pred, model_name)
            plot_classification_matrix(y_test, y_pred, model_name)
        #### Now select the best estimator from the RandomizedSearchCV models
        nlp_model = gs.best_estimator_
        #### Build a pipeline with the best estimator and the best vectorizer together here!
        pipe = make_pipeline(cvect,nlp_model)
        ### Train the Pipeline on the full data set !
        print('Training Pipeline on full Train data. This will take time...')
        #####  Now AFTER TRAINING, make predictions on the given test data set!
        start_time = time.time()
        pipe.fit(X,y)
        if not isinstance(test, str):
            y_pred = pipe.predict(test[nlp_column])
        print('    Time taken to train Pipeline on full Train shape (%s) and test on (%s) = %0.2f seconds' %(
                            X.shape,test.shape,time.time()-start_time) )
        print('Time taken for Auto_NLP = %0.1f minutes' %((time.time()-start_time4)/60))
        print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
        if not isinstance(test, str):
            return train, test, pipe, y_pred
        else:
            return train, '', pipe, ''
    else:
        ##################    THIS IS FOR BUILD_MODEL = FALSE           #################
        ##################  THIS IS WHERE YOU ADD COLUMNS, SVD, SENTIMENT ETC.       #################
        train_all = best_nlp_vect.fit_transform(train[nlp_column])
        if train_all.shape[1] <= top_num_features:
            #### If the TFIDF array is very small, you just select the entire TFIDF array
            train_best = pd.DataFrame(train_all.todense(),columns=best_nlp_vect.get_feature_names())
        else:
            #### For Train data, you don't have to send in an SVD. It will automatically select one and train it
            best_features_array, tsvd = select_top_features_from_SVD(train_all,'',True)
            ls = ['svd_dim_'+str(x) for x in range(best_features_array.shape[1])]
            train_best = pd.DataFrame(best_features_array,columns=ls)
        #_ = select_top_features_from_vectorizer(train_all, best_nlp_vect, top_num_features)
        if type(test) != str:
            test_index = test.index
            #### If test data is given, then convert it into a Vectorized frame using best vectorizer
            test_all = best_nlp_vect.transform(test[nlp_column])
            if test_all.shape[1] <= top_num_features:
                best_df = pd.DataFrame(test_all.todense(),columns=best_nlp_vect.get_feature_names())
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
        ######### BUILD   MODEL   HERE   IF  BUILD_MODEL  IS   TRUE  ###########################
        train_source = nlp_data[nlp_data['auto_nlp_source']=='Train'].drop('auto_nlp_source',axis=1)
        if not isinstance(test, str):
            test_source = nlp_data[nlp_data['auto_nlp_source']=='Test'].drop('auto_nlp_source',axis=1)
        ######### Split it back into train_best and test_best ##################################
        train_full = train_source.drop([nlp_column],axis=1)
        if type(test) == str:
            test_full = ''
        else:
            test_full = test_source.drop([target,nlp_column],axis=1)
        print('Number of new columns created using NLP = %d' %(len(nlp_result_columns)))
        print('Time taken = %0.1f minutes' %((time.time()-start_time4)/60))
        print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
        return train_full, test_full, best_nlp_vect, max_features_limit
##############################################################################################
def plot_confusion_matrix(y_test,y_pred, model_name='Model'):
    """
    This plots a beautiful confusion matrix based on input: ground truths and predictions
    """
    #Confusion Matrix
    '''Plotting CONFUSION MATRIX'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    '''Display'''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    pd.options.display.float_format = '{:,.2f}'.format

    #Get the confusion matrix and put it into a df
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm,
                         index = np.unique(y_test).tolist(),
                         columns = np.unique(y_test).tolist(),
                        )

    #Plot the heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(cm_df,
                center=0,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),
                annot=True,
                fmt='g')

    plt.title(' %s \nF1 Score(avg = micro): %0.2f \nF1 Score(avg = macro): %0.2f' %(
        model_name,f1_score(y_test, y_pred, average='micro'),f1_score(y_test, y_pred, average='macro')),
              fontsize = 13)
    plt.ylabel('True label', fontsize = 13)
    plt.xlabel('Predicted label', fontsize = 13)
    plt.show();
##############################################################################################
def plot_classification_matrix(y_test, y_pred, model_name='Model'):
    """
    This plots a beautiful classification report based on 2 inputs: ground truths and predictions
    """
    # Classification Matrix
    '''Plotting CLASSIFICATION MATRIX'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    '''Display'''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    pd.options.display.float_format = '{:,.2f}'.format

    #Get the confusion matrix and put it into a df
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix,classification_report
    from sklearn.metrics import precision_score

    cm = classification_report(y_test, y_pred,output_dict=True)

    cm_df = pd.DataFrame(cm)

    #Plot the heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(cm_df,
                center=0,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),
                annot=True,
                fmt='0.2f')

    plt.title(""" %s
    \nAverage Precision Score(avg = micro): %0.2f \nAverage Precision Score(avg = macro): %0.2f""" %(
        model_name, precision_score(y_test,y_pred, average='micro'),
        precision_score(y_test, y_pred, average='macro')),
              fontsize = 13)
    plt.ylabel('True label', fontsize = 13)
    plt.xlabel('Predicted label', fontsize = 13)
    plt.show();
#################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
#####################################################################
def print_regression_model_stats(actuals, predicted, title='Model'):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    figsize = (10, 10)
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
#########################################################################
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
version_number = '0.0.31'
print("""Imported Auto_NLP version: %s.. Call using:
     train_nlp, test_nlp, nlp_pipeline, predictions = Auto_NLP(
                nlp_column, train, test, target, score_type,
                modeltype,top_num_features=50, verbose=0,
                build_model=True)""" %version_number)
########################################################################
