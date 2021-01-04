################################################################################
####                       Auto NLP for Python 3 version                    ####
####                      Developed by Ram Seshadri                         ####
####                        All Rights Reserved                             ####
################################################################################
#### Auto NLP applies NLP processing techniques on a dataset with one variable##
#### You cannot give a dataframe with multiple string variables as only one ####
####  is allowed. It splits the dataset into train and test and returns     ####
####  predictions on test for Classification or Regression problems.        ####
################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_colwidth',1000)
sns.set(style="white", color_codes=True)
import time
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
import pdb
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

### For NLP problems
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
import string

#### For Classification problems
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#### For Regression problems
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn import model_selection, metrics   #Additional sklearn functions
from sklearn.model_selection import GridSearchCV   #Performing grid search
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import copy
from itertools import cycle
from collections import Counter
############################################################################
from collections import OrderedDict, Counter
def print_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    try:
        ### test if it is a multi-label problem by seeing if the classes has multiple columns
        len(classes.columns) > 1
        ### This is a multi-label problem, hence you have to do value counts by each target name
        targets = classes.columns.tolist()
        for each_target in targets:
            print('%s value counts:\n%s' %(each_target,classes[each_target].value_counts()))
    except:
        ##### if classes has only one column, then it is a single-label problem
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
################################################################################
###########     N  L  P    F  U  N C  T  I  O   N  S       #####################
################################################################################
'''Features'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

'''Classifiers'''
from sklearn.naive_bayes import GaussianNB

'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import cycle

'''Plotting'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

'''Display'''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
################################################################################
####   The below Process_Text section is Re-used with Permission from:
####  R O B   S A L G A D O    robert.salgado@gmail.com     Thank YOU!
# https://github.com/robsalgado/personal_data_science_projects/tree/master/mulitclass_text_class
################################################################################
import nltk
import re
import string
import unicodedata
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import regex as re

#Contraction map
c_dict = {
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

c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))
##################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
################################################################################
def return_stop_words():
    from nltk.corpus import stopwords
    STOP_WORDS = ['it', "this", "that", "to", 'its', 'am', 'is', 'are', 'was', 'were', 'a',
                'an', 'the', 'and', 'or', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
                 'into','above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'over',
                  'under', 'again', 'further', 'then', 'once', 'all', 'any', 'both', 'each',
                   'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'just', 'd', 'll', 'm', 'o', 're',
                    've', 'y', 'ain', 'ma']
    add_words = ["s", "m",'you', 'not',  'get', 'no', 'via', 'one', 'still', 'us', 'u','hey','hi','oh','jeez',
                'the', 'a', 'in', 'to', 'of', 'i', 'and', 'is', 'for', 'on', 'it', 'got','aww','awww',
                'not', 'my', 'that', 'by', 'with', 'are', 'at', 'this', 'from', 'be', 'have', 'was',
                '', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\']
    #stopWords = text.ENGLISH_STOP_WORDS.union(add_words)
    stop_words = list(set(STOP_WORDS+add_words))
    excl =['will',"i'll",'shall',"you'll",'may',"don't","hadn't","hasn't","haven't",
           "don't","isn't",'if',"mightn't","mustn'","mightn't",'mightn',"needn't",
           'needn',"needn't",'no','not','shan',"shan't",'shouldn',"shouldn't","wasn't",
          'wasn','weren',"weren't",'won',"won't",'wouldn',"wouldn't","you'd",
          "you'd","you'll","you're",'yourself','yourselves']
    stopWords = left_subtract(stop_words,excl)
    return sorted(stopWords)
##################################################################################
tokenizer = TweetTokenizer()
pattern = r"(?u)\b\w\w+\b"
lemmatizer = WordNetLemmatizer()
punc = list(set(string.punctuation))+['/;','//']

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
#
# remove entire URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Remove just HTML markup language
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Convert Emojis to Text
import emoji
def convert_emojis(text):
    try:
        return emoji.demojize(text)
    except:
        return "Errorintext"

import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

# Clean even further removing non-printable text
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
import string
def remove_stopwords(tweet):
    """Removes STOP_WORDS characters"""
    stop_words = return_stop_words()
    tweet = tweet.lower()
    tweet = ' '.join([x for x in tweet.split(" ") if x not in stop_words])
    tweet = ''.join([x for x in tweet if x in string.printable])
    return tweet

# Expand Abbreviations
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
from nltk.tokenize import word_tokenize
def convert_abbrev(word):
    return abbreviations[word] if word in abbreviations.keys() else word

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def convert_abbrev_in_text(sentence):
    text = " ".join(sentence.split())
    tokens = text.split(" ")
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

def join_words(text):
    return " ".join(text)

def remove_punctuations(text):
    try:
        remove_puncs = re.sub(r'[?|!|~|@|$|%|^|&|#]', r'', text)
    except:
        return "error"
    return re.sub(r'[.|,\'|,|)|(|\|/|+|-|{|}|:|]', r' ', remove_puncs)

def process_text(text):
    text = text.split(" ")
    decontract = join_words([expandContractions(item, c_re=c_re) for item in text])
    soup = BeautifulSoup(decontract, "lxml")
    tags_del = soup.get_text()
    no_html = re.sub('<[^>]*>', '', tags_del)
    tokenized = casual_tokenizer(no_html)
    tagged = nltk.pos_tag(tokenized)
    lemma = lemma_wordnet(tagged)
    no_num = [re.sub('[0-9]+', '', each) for each in lemma]
    no_punc = join_words([remove_punctuations(w) for w in no_num ])
    no_stop = remove_stopwords(no_punc)
    return no_stop

def clean_text(df_x):
    start_time2 = time.time()
    df_x = df_x.apply(convert_emojis).apply(lambda x: x.lower()).apply(convert_abbrev_in_text)
    print('    Time Taken for Expanding emojis and abbreviations in data = %0.0f seconds' %(time.time()-start_time2) )
    start_time3 = time.time()
    df_x = df_x.apply(process_text)
    print('        Time Taken for Processing text in data = %0.0f seconds' %(time.time()-start_time3) )
    return df_x
##########################################################################################################
from nltk.stem import PorterStemmer
def stemming(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", text).split()])

# Remove Emojis
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
import re
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove punctuation marks
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
###############################################################################
####    CLEAN   TWEETS   IS   MEANT FOR SHORT  SENTENCES SUCH AS TWEETS
###############################################################################
def clean_steps(x):
    """
    Input must be one text string only. Don't send arrays or dataframes.
    Clean steps cleans one tweet at a time using following steps:
    1. removes URL
    2. Removes a very small list of stop words - about 65
    3. Removes Emojis
    """
    x = x.split(" ")
    x = [word.lower() for word in x]
    x = join_words([expandContractions(item, c_re=c_re) for item in x])
    x = remove_html(x)
    x = remove_URL(x)
    x = remove_stopwords(x)
    x = convert_emojis(x)
    x = remove_punct(x)
    x = convert_abbrev_in_text(x)
    x = stemming(x)
    x = remove_stopwords(x)
    return x

def clean_tweets(df_x):
    """
    Clean Tweets cleans an entire dataframe or array full of tweets. So input must be a df or array of texts.
    """
    df_x = df_x.apply(clean_steps)
    return df_x
################################################################################
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
################################################################################
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
    target = copy.deepcopy(target)
    targets = copy.deepcopy(target)
    #### Make sure that you do this for only small data sets ####################
    max_samples = min(10000,train.shape[0])
    buswo = train.sample(max_samples, random_state=99)
    orig_vect = copy.deepcopy(best_nlp_vect)
    df_names = []
    if not isinstance(targets,list):
        targets = [target]
    for target in targets:
        print('\nFor target = %s' %target)
        classes = np.unique(buswo[target])
        classes_copy = copy.deepcopy(classes)
        for itera in classes_copy:
            if not isinstance(itera, str):
                new_df_name = 'df_'+str(itera)
            else:
                new_df_name = 'df_'+itera
            #print('%s is about class=%s of shape: ' %(new_df_name,itera), end='')
            new_df_name = buswo[(buswo[target]==itera)]
            df_names.append(new_df_name)
            #print(new_df_name.shape)
        #### now we split into as many data frames as there are classes in the train data set
        df_dtms = []
        count_df = 0
        count_bus_wo = pd.DataFrame()
        all_sorted = []
        for each_df, each_class in zip(df_names,classes):
            print('\n    For class = %s' %each_class)
            eachdf_index = each_df.index
            cv = copy.deepcopy(best_nlp_vect)
            top_num_feats = print_top_feature_grams(each_df[nlp_column], cv, top_nums)
            #### This is an Alternative Method to get Top Num features ###########
            #top_num_feats =set([" ".join(x.split("_")) for x in word_freq_bigrams(bus_texts,int(top_nums*1/2))[0].values]+[
            #                    " ".join(x.split("_")) for x in bigram_freq(bus_texts,int(top_nums*1/3))[0].values])
            print('    Top n-grams that are most frequent in this class are: %d' %len(top_num_feats))
            #### Once you do that set it as the vocab and get a dataframe built with those vocabs
            all_sorted += top_num_feats
        all_sorted_set = sorted(set(all_sorted), key=all_sorted.index)
    return all_sorted_set
####################################################################################
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
    from sklearn.metrics import confusion_matrix, f1_score

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
    from sklearn.metrics import precision_score
    from sklearn.metrics import classification_report
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
import scipy as sp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
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
        if X.shape[1] < top_n:
            ### Sometimes there are not even 100 columns after using TFIDF, so better to cut it short.
            top_n = int(X.shape[1] - 1)
        tsvd = TruncatedSVD(n_components=top_n,
                   n_iter=10,
                   random_state=3)
        tsvd.fit(X)
    XA = tsvd.transform(X)
    print('    Reduced dimensional array shape to %s' %(XA.shape,))
    print('    Time Taken for Truncated SVD = %0.0f seconds' %(time.time()-start_time) )
    return XA, tsvd
################################################################################
# define a function that accepts a vectorizer and calculates its accuracy
################################################################################
def fit_and_predict(model, X_train, y_train, X_cv, modeltype='Classification', is_train=True):
    X_train = copy.deepcopy(X_train)
    X_cv = copy.deepcopy(X_cv)
    if is_train:
        #### They are used to fit and predict here ###
        try:
            model.fit(X_train, y_train)
            y_preds = model.predict(X_cv)
            return y_preds
        except:
            model.fit(X_train.toarray(), y_train)
            y_preds = model.predict(X_cv)
            return y_preds
    else:
        #### Just use model for predicting here
        try:
            y_preds = model.predict(X_cv)
            return y_preds
        except:
            y_preds = model.predict(X_cv.toarray())
            return y_preds
################################################################################
import copy
def reduce_dimensions_with_Truncated_SVD(each_df, each_df_dtm, is_train=True,trained_svd=''):
    """
    This is a new method to combine the top X features from Vectorizers and the top 100 dimensions from Truncated SVD.
    The idea is to have a small number of features that are the best in each class (label) to produce a very fast accurate model.
    This model outperforms many models that have 10X more features. Hence it can be used to build highly interpretable models.
    """
    import copy
    orig_each_df = copy.deepcopy(each_df)
    orig_each_df_index = orig_each_df.index
    ### Now you have to use transformed data to create a Trained SVD that will reduce dimensions to 100-dimensions
    ### You have to make sure that you send in a trained SVD and set training to False since this is each_df
    if is_train:
        each_df_dtm1, trained_svd = select_top_features_from_SVD(each_df_dtm, '', True)
        ls = ['svd_dim_'+str(x) for x in range(each_df_dtm1.shape[1])]
        each_df_dtm1 = pd.DataFrame(each_df_dtm1,columns=ls, index=orig_each_df_index)
    else:
        each_df_dtm1, _ = select_top_features_from_SVD(each_df_dtm, trained_svd, False)
        ls = ['svd_dim_'+str(x) for x in range(each_df_dtm1.shape[1])]
        each_df_dtm1 = pd.DataFrame(each_df_dtm1,columns=ls, index=orig_each_df_index)
    print('TruncatedSVD Data Frame size = %s' %(each_df_dtm1.shape,))
    return each_df_dtm1, trained_svd
###########################################################################
def transform_combine_top_feats_with_SVD(each_df_dtm, nlp_column, big_nlp_vect, new_vect,
                                            top_feats,is_train=True,trained_svd=''):
    """
    This is a new method to combine the top 300 features from Vectorizers and the top 100 dimensions from Truncated SVD.
    The idea is to have a small number of features that are the best in each class (label) to produce a very fast accurate model.
    This model outperforms many models that have 10X more features. Hence it can be used to build highly interpretable models.
    """
    import copy
    orig_each_df = copy.deepcopy(each_df)
    orig_each_df_index = orig_each_df.index
    # For each_df data, it is tricky! You need to use two vectorizers: one that is smaller and another that is bigger!
    start_time = time.time()
    if is_train:
        each_df_dtm = big_nlp_vect.fit_transform(each_df[nlp_column])
        print('Time Taken for Transforming Train %s data = %0.0f seconds' %(each_df_dtm.shape,time.time()-start_time) )
    else:
        each_df_dtm = big_nlp_vect.transform(each_df[nlp_column])
        print('Time Taken for Transforming Test data = %0.0f seconds' %(time.time()-start_time) )
    if is_train:
        small_nlp_vect = copy.deepcopy(big_nlp_vect)
        small_nlp_vect.vocabulary = top_feats
    else:
        small_nlp_vect = copy.deepcopy(new_vect)
    ### Now you have to use the bigger Vectorizer to create a Trained SVD that will reduce dimensions to 100-dimensions
    ### You have to make sure that you send in a trained SVD and set training to False since this is each_df
    if is_train:
        each_df_dtm1, trained_svd = select_top_features_from_SVD(each_df_dtm, '', True)
        ls = ['svd_dim_'+str(x) for x in range(each_df_dtm1.shape[1])]
        each_df_dtm1 = pd.DataFrame(each_df_dtm1,columns=ls, index=orig_each_df_index)
    else:
        each_df_dtm1, _ = select_top_features_from_SVD(each_df_dtm, trained_svd, False)
        ls = ['svd_dim_'+str(x) for x in range(each_df_dtm1.shape[1])]
        each_df_dtm1 = pd.DataFrame(each_df_dtm1,columns=ls, index=orig_each_df_index)
    #### You have to create another vector with smaller vocab "small_nlp_vect" vectorizer
    #### You have to make sure you just do a Transform and not a Fit!
    if is_train:
        each_df_dtm2 = small_nlp_vect.fit_transform(each_df[nlp_column])
        each_df_dtm2 = pd.DataFrame(each_df_dtm2.toarray(),index=orig_each_df_index,
                                             columns=small_nlp_vect.get_feature_names())
        #### Since the top features from each class is a pretty bad idea, I am dropping it here!
        #print('Added top %d features from Train data' %(each_df_dtm2.shape[1]))
    else:
        each_df_dtm2 = small_nlp_vect.transform(each_df[nlp_column])
        each_df_dtm2 = pd.DataFrame(each_df_dtm2.toarray(),index=orig_each_df_index,
                                             columns=small_nlp_vect.get_feature_names())
        #### Since the top features from each class is a pretty bad idea, I am dropping it here!
        #print('Added top %d features from Test data' %(each_df_dtm2.shape[1]))
    # Now you have to combine them all to get a new each_df_best dataframe
    ### Since the top features from each class is not helping improve model, it is best dropped!
    #each_df_best = each_df_dtm2.join(each_df_dtm1)
    each_df_best = copy.deepcopy(each_df_dtm1)
    print('TruncatedSVD Data Frame size = %s' %(each_df_best.shape,))
    return each_df_best, big_nlp_vect, small_nlp_vect, trained_svd
###########################################################################
def print_sparse_stats(X_dtm):
    """
    Prints the stats around a Sparse Matrix (typically) generated in NLP problems.
    """
    print ('Shape of Sparse Matrix: ', X_dtm.shape)
    print ('Amount of Non-Zero occurences: ', X_dtm.nnz)
    print ('    Density: %.2f%%' % (100.0 * X_dtm.nnz /
                                 (X_dtm.shape[0] * X_dtm.shape[1])))
################################################################################
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
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
################################################################################
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
        tvec = TfidfVectorizer(ngram_range=(1,3), stop_words=None, max_features=max_features, min_df=min_df,max_df=max_df)
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
    print('    However max_features limit = %d will limit too many features from being generated' %max_features_high)
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
                        strip_accents='unicode', tokenizer=None,preprocessor=None,
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
                        best_max_df = copy.deepcopy(each_max_df)
                    else:
                        best_max_df = each_max_df + 0.20
                        break
                else:
                    if current_metric >= best_metric:
                        best_metric = copy.deepcopy(current_metric)
                        best_model = copy.deepcopy(current_model)
                        best_max_df = copy.deepcopy(each_max_df)
                    else:
                        best_max_df = each_max_df + 0.20
                        break
            count_max_df += 1
        print('Best max_df selected to be %0.2f' %best_max_df)
    else:
        best_max_df = 0.5
        print('\n#### Optimizing Count Vectorizer with best max_df=%0.2f, 1-3 n-grams and high features...' %best_max_df)
        vect_5000 = CountVectorizer(
                   ngram_range=(1, 3), max_features=max_features_high, max_df=best_max_df,
                    strip_accents='unicode', tokenizer=None, preprocessor=None,
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
    print('\n#### Using Count Vectorizer with Latin-1 encoding, limited max_features =%d and a min_df=%s with n_gram (1-5)' %(max_features,min_df))
    vect_lemma = CountVectorizer(max_df=best_max_df,
                                   max_features=max_features, strip_accents='unicode',
                                   ngram_range=(1, 5), token_pattern=r'\w{1,}',
                                    min_df=min_df, stop_words=None, encoding='latin-1',
                                   binary=False,
                                    )
    try:
        all_vecs[vect_lemma], all_models[vect_lemma] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                           y_test, target, metric,
                                             vect_lemma, seed, modeltype)
    except:
        print('Error: Using CountVectorizer')
    print('\n# Using TFIDF vectorizer with binary=True, ngram = (1,3) and max_features=%d' %max_features_high)
    ##### This is based on artificially setting 5GB as being max memory limit for the term-matrix
    tvec = TfidfVectorizer( max_features=max_features_high, max_df=best_max_df, token_pattern=r'\w{1,}',
                                strip_accents='unicode', sublinear_tf=True, binary=True,
                                stop_words=None, ngram_range=(1, 3), min_df=min_df)
    all_vecs[tvec], all_models[tvec] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec, seed, modeltype)
    max_features_limit = int(tvec.fit_transform(data_dtm).shape[1])
    ##### This is based on using a Latin-1 vectorizer in case Spanish words are in text
    print('\n# Using TFIDF vectorizer with latin-1 encoding, binary=False, ngram (1,3) and limited max_features')
    tvec2 = TfidfVectorizer( max_features=max_features, max_df=best_max_df,
                                token_pattern=r'\w{1,}', sublinear_tf=True,
#                                tokenizer=simple_tokenizer,preprocessor=simple_preprocessor,
                                 tokenizer=None, encoding='latin-1',
                                 min_df=min_df, stop_words=None,  binary=False, strip_accents='unicode',
                                 use_idf=True, ngram_range=(1,3))
    all_vecs[tvec2], all_models[tvec2] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                        tvec2, seed, modeltype)
    #Finally Using a basic count vectorizer with all defaults while limited max features
    print('\n# Finally comparing them against a Basic Count Vectorizer with all defaults, max_features = %d and  lowercase=True' %max_features_high)
    cvect = CountVectorizer(min_df=2, lowercase=True, max_features=max_features_high, binary=False)
    try:
        all_vecs[cvect], all_models[cvect] = tokenize_test_by_metric(model, X_train, X_test, y_train,
                                      y_test, target, metric,
                                      cvect, seed, modeltype)
    except:
        print('Error: Using CountVectorizer')
    ######## Once you have built 4 different transformers it is time to compare them
    if modeltype.endswith('Classification'):
        if metric in ['log_loss','logloss']:
            best_vec = pd.Series(all_vecs).idxmin()
        else:
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
#########################################################################################
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import scipy as sp
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#########################################################################################
def Auto_NLP(nlp_column, train, test, target, score_type='',
                            modeltype='Classification',
                            top_num_features=300, verbose=0,
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
    #### You can use top_num_features (default = 200) to control how many features to add.
    ##################################################################################
    """
    import nltk
    nltk.download("popular")
    calibrator_flag = False
    import time
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
    top_num_features_limit = 300
    ###### Set Scoring parameter score_type here ###############
    if score_type == '':
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
    elif score_type in ['f1','precision','average_precision','recall','average_recall','roc_auc',
                        'balanced-accuracy','balanced_accuracy']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
    elif score_type in ['rmse','mae','mean_squared_error','mean_absolute_error','mean_absolute_percentage_error']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'balanced_accuracy_score'
    elif score_type in ['neg_log_loss', 'logloss','log_loss']:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
    else:
        if modeltype == 'Regression':
            score_type = 'neg_mean_squared_error'
        else:
            score_type = 'accuracy'
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
        print('Error: your NLP column name must be either a string or list with the column name in data frame')
        return
    ########################  S U M M A R Y  C O L U M N S  C R E A T I O N ######################
    #### Since NLP Summary Columns do more harm than good, I am not adding them to features of train
    train, nlp_summary_cols = create_summary_of_nlp_cols(train, nlp_column, target, is_train=True, verbose=verbose)
    nlp_result_columns += nlp_summary_cols
    print('    Added %d summary columns for counts of words and characters in each row' %len(nlp_summary_cols))
    if not isinstance(test, str):
        #### You don't want to draw the same set of charts for Test data since it would be repetitive
        #####   Hence set the verbose to 0 in this case !!!
        test, nlp_summary_cols = create_summary_of_nlp_cols(test, nlp_column, target, is_train=False, verbose=0)
    ########################  C L E AN    C O L U M N S   F I R S T ######################
    #if train[nlp_column].apply(len).mean() <= 1500:
    if top_num_features >= top_num_features_limit:
        tweets_flag = True
    else:
        tweets_flag = False
    if train.shape[0] >= 100000:
        print('Cleaning text in %s column. Please be patient since this is a large dataset with >100K rows...' %nlp_column )
    else:
        print('Cleaning text in Train data for %s column' %nlp_column)
    ###############################################################################################################
    #### If you are not going to use select_best_nlp_vectorizer, then you must use pipelines with a basic TFIDF vectorizer
    best_nlp_vect = TfidfVectorizer( min_df=2, sublinear_tf=True, norm='l2', analyzer='word',
                token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words=None,
                lowercase=True, binary = True, encoding = 'latin-1',
                max_features = None, max_df = 0.5
                )
    if tweets_flag:
        #print('    Text appears to be short sentences or tweets, using clean_tweets function...')
        print('    Faster text processing using clean_tweets function, since top_num_features exceeds %s' %top_num_features_limit)
        train[nlp_column] = clean_tweets(train[nlp_column])
        #train[nlp_column] = clean_text(train[nlp_column])
    else:
        #print('    Text appears to be long sentences or paragraphs, using clean_text function')
        print('    Faster text processing using clean_text function, since top_num_features is below %s' %top_num_features_limit)
        train[nlp_column] = clean_text(train[nlp_column])
    ###############################################################################################################
    print('Train data Text cleaning completed. Time taken = %d seconds' %(time.time()-start_time))
    #### Build a RandomizedSearchCV Parameters Dictionary here  #########
    params = {}
    params['tfidfvectorizer__binary'] = [True,False]
    params['tfidfvectorizer__encoding'] = ['latin-1','utf-8']
    params['tfidfvectorizer__max_df'] = sp.stats.uniform(scale=1)
    ##### Just set the printing of top features to 200 always #################
    if modeltype != 'Regression':
        ### Do this only for printing top words n-grams by classes since each class may be different
        start_time = time.time()
        top_feats = print_top_features(train,nlp_column, best_nlp_vect, target, 200)
        print('Time Taken = %0.0f seconds' %(time.time()-start_time) )
    #############   THIS IS WHERE WE USE BUILD_MODEL TO DECIDE ################################
    k_best_features = min(top_num_features,1200)
    if build_model:
        print('##################    THIS IS FOR BUILD_MODEL = TRUE           #################')
        n_iter = 30
    else:
        print('##################    THIS IS FOR BUILD_MODEL = FALSE           #################')
        n_iter = 10
    print('Building Model and Pipeline for NLP column = %s. This will take time...' %nlp_column)
    if isinstance(best_nlp_vect, str):
        cvect = CountVectorizer(min_df=2, lowercase=True)
    else:
        cvect = copy.deepcopy(best_nlp_vect)
    ### Split into Train and CV to test the model #####################
    X = train[nlp_column]
    y = train[target]
    if modeltype != 'Regression':
        print_rare_class(y,verbose=1)
    #Train test split with stratified sampling for evaluation
    if modeltype == 'Regression':
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            random_state=seed)
    else:
        if isinstance(target, list):
            X = X.sample(frac=1.0, random_state=seed)
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
    max_features_limit = best_nlp_vect.fit_transform(X_train).shape[1]
    print('    Selected the maximum number of features limit = %d' %max_features_limit)
    if modeltype == 'Regression':
        ### currently SelectKBest does not seem to work with regression data - hence it's out
        select =  SelectKBest(f_regression, k=k_best_features)
        params['selectkbest__k'] = sp.stats.randint(k_best_features,max_features_limit)
    else:
        select =  SelectKBest(chi2, k=k_best_features)
        params['selectkbest__k'] = sp.stats.randint(k_best_features,max_features_limit)
    print('Performing RandomizedSearchCV across 30 params. Optimizing for %s' %score_type)
    print('    Using train data = %s and Cross Validation data = %s' %(X_train.shape, X_test.shape,))
    ######   This is where we choose one model vs another based on problem type ##
    if modeltype == 'Regression':
        scv = KFold(n_splits=n_splits)
        model_name = 'Random Forest Regressor'
        nlp_model = RandomForestRegressor(n_estimators = 100, n_jobs=-1, random_state=seed)
        #params['randomforestregressor__max_depth'] = sp.stats.randint(2,10),
        #params['randomforestregressor__n_estimators'] = sp.stats.randint(200,500)
    else:
        if isinstance(target, list):
            scv = KFold(n_splits=n_splits)
            model_name = 'Random Forest Classifier'
            nlp_model = RandomForestClassifier(n_estimators = 100, n_jobs=-1, random_state=seed)
        else:
            scv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if top_num_features < top_num_features_limit:
                model_name = 'Multinomial NB'
                nlp_model = MultinomialNB()
                params['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
            else:
                model_name = 'Random Forest Classifier'
                nlp_model = RandomForestClassifier(random_state=seed,n_estimators=200,n_jobs=-1)
                #params['randomforestclassifier__max_depth'] = sp.stats.randint(2,10),
                #params['randomforestclassifier__n_estimators'] = sp.stats.randint(200,500)
    #### Adding a CalibratedClassifier to text classification tasks  ########################
    if modeltype != 'Regression':
        if isinstance(target, list):
            ### There is no need for CalibratedClassifierCV in Multi-Label problems
            pass
        else:
            if X_train.shape[0] <= 1000:
                # This works well for small data sets and is similar to parametric
                method=  'sigmoid' # 'isotonic' # #
            else:
                # This works well for large data sets and is non-parametric
                method=  'isotonic'
            calibrator_flag = True
            print('Using a Calibrated Classifier in this Multi_Classification dataset to improve results...')
    ################    B U I L D I N G   A   P I P E L I N E   H E R E  ######################
    if top_num_features < top_num_features_limit:
        print("""Since top_num_features = %d, %s model selected. If you need different model, increase it >= %d.""" %(
                            top_num_features,model_name,top_num_features_limit))
    else:
        print("""Since top_num_features = %d, selecting %s model. If you need different model, decrease it <%d.""" %(
                            top_num_features,model_name,top_num_features_limit))
    ### The reason we don't add a clean_text function here in pipeline is because it takes too long in online
    ### It is better to clean the data in advance and then use the pipeline here in GS mode to find best params
    from sklearn.preprocessing import FunctionTransformer
    ##### Train and test the model ##########
    try:
        pipe = make_pipeline(
            cvect,
            select,
            nlp_model)
        gs = RandomizedSearchCV(pipe, params, n_iter=n_iter, cv=scv,
                                scoring=score_type, random_state=seed)
        gs.fit(X_train,y_train)
    except:
        ### If there is an error, we will just skip parameter tuning and just take a simple model
        params = {}
        pipe = make_pipeline(
             cvect,
             FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
             select,
             nlp_model)
        gs = RandomizedSearchCV(pipe, params, n_iter=30, cv=scv,
                                scoring=score_type, random_state=seed)
        gs.fit(X_train,y_train)
    ##### Now check to see if the CalibratedClassifier can work on this data set #####
    model_string = "".join(model_name.lower().split(" "))
    #### Now select the best estimator from the RandomizedSearchCV models
    best_vect = gs.best_estimator_.named_steps['tfidfvectorizer']
    best_sel = gs.best_estimator_.named_steps['selectkbest']
    if calibrator_flag:
        best_estimator = gs.best_estimator_.named_steps[model_string]
        calib_pipe = make_pipeline(
             best_vect,
             FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
             best_sel,
             )
        best_model = CalibratedClassifierCV(best_estimator,cv=3, method='isotonic')
        best_model.fit(calib_pipe.transform(X_train), y_train)
        y_pred = best_model.predict(calib_pipe.transform(X_test))
    else:
        best_model = gs.best_estimator_.named_steps[model_string]
        y_pred = gs.predict(X_test)
    ##### Print the model results on Cross Validation data set (held out)
    print('Training completed. Time taken for training = %0.1f minutes' %((time.time()-start_time)/60))
    print('Best Params of NLP pipeline are: %s' %gs.best_params_)
    if modeltype == 'Regression':
        print_regression_model_stats(y_test, y_pred,'%s Model: Predicted vs Actual for %s' %(model_name,target))
    else:
        if isinstance(target, list):
            from sklearn.metrics import multilabel_confusion_matrix
            print('Multi Label Confusion Matrix for each label:\n%s' %multilabel_confusion_matrix(y_test.values,y_pred))
            from sklearn.metrics import classification_report
            print(classification_report(y_test.values, y_pred, target_names=target))
        else:
            plot_confusion_matrix(y_test, y_pred, model_name)
            plot_classification_matrix(y_test, y_pred, model_name)
    ### Train the Pipeline on the full data set here ###########################
    ### The reason we add clean_tweets and clean_text to the transformer here is because once GS is done, its faster
    #### UP TO NOW BOTH BUILD_MODEL PATHS ARE SAME. HERE THEY DIVERTSE A BIT ######################
    #### If there is no model, then we can just use this to create a Transformer pipeline going forward ###
    #### This Transformer pipeline will transform train and test data using the selected vectorizer and selectKbest ####
    if tweets_flag:
        transform_pipe = make_pipeline(
            FunctionTransformer(lambda x: clean_tweets(x)),
            #FunctionTransformer(lambda x: clean_text(x)),
            best_vect,
            best_sel,
            )
    else:
        transform_pipe = make_pipeline(
            FunctionTransformer(lambda x: clean_text(x)),
            best_vect,
            best_sel,
            )
    ###  This transform_pipe is used merely to transform text into the best term matrix for modeling
    #### Using the transform pipeline we will transform the train and test data sets!
    print('  Now transforming Train data to return as output...')
    trainm = transform_pipe.transform(train[nlp_column])
    sel_col_names = np.array(best_vect.get_feature_names())[transform_pipe.named_steps[
                            'selectkbest'].get_support()]
    trainm = pd.DataFrame(trainm.todense(),index=train_index,columns=sel_col_names)
    if not isinstance(test, str):
        test_index = test.index
        print('  Transforming Test data to return as output...')
        testm = transform_pipe.transform(test[nlp_column])
        testm = pd.DataFrame(testm.todense(),index=test_index,columns=sel_col_names)
    #### This best_pipe pipeline will however in addition to transforming, will also train and predict using the trained model ####
    if build_model:
        if tweets_flag:
            best_pipe = make_pipeline(
                FunctionTransformer(lambda x: clean_tweets(x)),
                #FunctionTransformer(lambda x: clean_text(x)),
                best_vect,
                best_sel,
                best_model)
        else:
            best_pipe = make_pipeline(
                FunctionTransformer(lambda x: clean_text(x)),
                best_vect,
                best_sel,
                best_model)
        print('Training best Auto_NLP Pipeline on full Train data...will be faster since best params are known')
        best_pipe.fit(X,y)
        train = train.join(trainm, rsuffix='_NLP_token_by_Auto_NLP')
        if not isinstance(test, str):
            print('    Returning best Auto_NLP pipeline to transform and make predictions on test data...')
            test = test.join(testm,rsuffix='_NLP_token_by_Auto_NLP')
            y_pred = best_pipe.predict(test[nlp_column])
            print('Training completed. Time taken for Auto_NLP = %0.1f minutes' %((time.time()-start_time4)/60))
            print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
            return train, test, best_pipe, y_pred
        else:
            print('Training completed. Time taken for Auto_NLP = %0.1f minutes' %((time.time()-start_time4)/60))
            print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
            return train, '', best_pipe, ''
    else:
        #####################################################################################
        print('##################    AFTER BEST NLP TRANSFORMER SELECTED, NOW ENRICH TEXT DATA  #####################')
        print('    Now we will start transforming NLP_column for train and test data using best vectorizer...')
        #####################################################################################
        start_time1 = time.time()
        best_nlp_vect = copy.deepcopy(best_vect)
        trainm = best_nlp_vect.transform(train[nlp_column])
        #######################################################################################
        ##################  THIS IS WHERE YOU ADD TRUNCATED SVD DIMENSIONS HERE      ##########
        #######################################################################################
        ### train_best contains the the TruncatedSVD dimensions of train data
        train_best, trained_svd = reduce_dimensions_with_Truncated_SVD(train,
                                                            trainm, is_train=True,trained_svd='')
        nlp_result_columns = left_subtract(list(train_best), cols_excl_nlp_cols)
        train_best = train_best.fillna(0)
        ### train_nlp contains the the TruncatedSVD dimensions along with original train data
        train_nlp = train.join(train_best,rsuffix='_SVD_Dim_by_Auto_NLP')
        #################################################################################
        if type(test) != str:
            testm = best_nlp_vect.transform(test[nlp_column])
            test_best, _ = reduce_dimensions_with_Truncated_SVD(test,
                                            testm, is_train=False, trained_svd=trained_svd)
            test_best = test_best.fillna(0)
            test_nlp = test.join(test_best, rsuffix='_SVD_Dim_by_Auto_NLP')
        ########################################################################
        ##### C R E A T E   C L U S T E R   L A B E L S    U S I N G   TruncatedSVD
        ########################################################################
        nlp_column_train = train[nlp_column].values
        if not isinstance(test, str):
            nlp_column_test = test[nlp_column].values
        ##### Do a clustering of word vectors from TruncatedSVD Dimensions array. It gives great results.
        #tfidf_term_array = create_tfidf_terms(nlp_column_train, best_nlp_vect,
        #                        is_train=True, max_features_limit=max_features_limit)
        tfidf_term_array = train_best.values
        print ('Creating word clusters using term matrix of size: %d for Train data set...' %len(
                                                    tfidf_term_array))
        #n_clusters = int(np.sqrt(len(tfidf_term_array['terms']))/2)
        n_clusters = int(np.sqrt(tfidf_term_array.shape[1])/2)
        if n_clusters < 2:
            n_clusters = 2
        ##### Always set verbose to 0 since usually KMEANS running is too verbose!
        km = KMeans(n_clusters=n_clusters, random_state=seed, verbose=0)
        kme, cluster_labels = return_cluster_labels(km, tfidf_term_array, n_clusters,
                                is_train=True)
        if isinstance(nlp_column, str):
            cluster_col = nlp_column + '_word_cluster_label'
        else:
            cluster_col = str(nlp_column) + '_word_cluster_label'
        train_nlp[cluster_col] = cluster_labels
        print ('    Created one new column: %s using KMeans_Clusters on NLP transformed columns...' %cluster_col)
        if not isinstance(test, str):
            ##### Do a clustering of word vectors from TruncatedSVD Dimensions array. It gives great results.
            #tfidf_term_array_test = create_tfidf_terms(nlp_column_test, best_nlp_vect,
            #                            is_train=False, max_features_limit=max_features_limit)
            tfidf_term_array_test = test_best.values
            _, cluster_labels_test = return_cluster_labels(kme, tfidf_term_array_test, n_clusters,
                                        is_train=False)
            test_nlp[cluster_col] = cluster_labels_test
            print ('    Created word clusters using NLP transformed matrix for Test data set...')
        print('    Time Taken for creating word cluster labels  = %0.0f seconds' %(time.time()-start_time1) )
        ########################################################################################
        #######  COMBINE TRAIN AND TEST INTO ONE DATA FRAME HERE BEFORE SENTIMENT ANALYSIS #####
        ########################################################################################
        train_nlp['auto_nlp_source'] = 'Train'
        if type(test) != str:
            test_nlp[target] = 0
            test_nlp['auto_nlp_source'] = 'Test'
            nlp_data = train_nlp.append(test_nlp)
        else:
            nlp_data = copy.deepcopy(train_nlp)
        ########################################################################################
        ######################## next Add SENTIMENT ANALYSIS HERE #########################
        ########################################################################################
        if len(nlp_data) <= 10000:
            ### VADER is accurate but very SLOOOWWW. Do not do this for Large data sets
            nlp_data, pos_cols = add_sentiment(nlp_data, nlp_column)
            nlp_result_columns += pos_cols
        else:
            ### TEXTBLOB is faster but somewhat less accurate. So we do this for Large data sets
            print('Since samples in data > 10000 using TextBlob, which is faster to add sentiment scores...')
            senti_cols = [nlp_column+'_text_sentiment', nlp_column+'_senti_polarity',
                                    nlp_column+'_senti_subjectivity',nlp_column+'_overall_sentiment']
            start_time2 = time.time()
            nlp_data[senti_cols[0]] = nlp_data[nlp_column].map(detect_sentiment).fillna(0)
            nlp_data[senti_cols[1]] = nlp_data[nlp_column].apply(calculate_line_sentiment,'polarity').fillna(0).values
            nlp_data[senti_cols[2]] = nlp_data[nlp_column].apply(calculate_line_sentiment,'subjectivity').fillna(0).values
            nlp_data[senti_cols[3]] = nlp_data[nlp_column].apply(calculate_paragraph_sentiment).fillna(0).values
            nlp_result_columns += senti_cols
            print('    Added %d columns using TextBlob Sentiment Analyzer. Time Taken = %d seconds' %(
                                        len(senti_cols), time.time()-start_time2))
        ########################################################################################
        #########      SPLIT DATA INTO BACK INTO TRAIN AND TEST HERE   #########################
        ########################################################################################
        train_source = nlp_data[nlp_data['auto_nlp_source']=='Train'].drop('auto_nlp_source',axis=1)
        if not isinstance(test, str):
            test_source = nlp_data[nlp_data['auto_nlp_source']=='Test'].drop('auto_nlp_source',axis=1)
        ######### Split it back into train_best and test_best ##################################
        train_full = train_source.drop([nlp_column],axis=1)
        if type(test) == str:
            test_full = ''
        else:
            test_full = test_source.drop([target,nlp_column],axis=1)
        number_of_created_columns = train_full.shape[1] - 1
        print('Number of new columns created using NLP = %d' %number_of_created_columns)
        print('Time taken for Auto_NLP to complete = %0.1f minutes' %((time.time()-start_time4)/60))
        print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
        return train_full, test_full, best_nlp_vect, max_features_limit
##############################################################################################
from sklearn.cluster import KMeans
####  These functions are for creating word cluster labels using each NLP column if it exists
def cluster_using_k_means(km, tfidf_matrix, num_cluster, is_train=True):
    print ("    Running k-means on NLP token matrix to create " + str(num_cluster) + " word clusters.")
    if is_train:
        clusters = km.fit_predict(tfidf_matrix)
    else:
        clusters = km.predict(tfidf_matrix)
    return km, clusters

def create_tfidf_terms(list_X, tfidf_vectorizer,is_train=True, max_features_limit=5000):
    tfidf_vectorizer.max_features = max_features_limit
    if is_train:
        tfidf_matrix = tfidf_vectorizer.fit_transform(list_X)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(list_X)
    return {
        'tfidf_matrix' : tfidf_matrix ,
        'terms' : tfidf_vectorizer.get_feature_names()
    }

def return_cluster_labels(km, tfid_terms, num_cluster, is_train):
    #X_terms = tfid_terms['tfidf_matrix']
    X_terms = copy.deepcopy(tfid_terms)
    km, cluster = cluster_using_k_means(km, X_terms, num_cluster, is_train)
    return km, cluster
###################################################################################
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
    from sklearn.metrics import confusion_matrix, f1_score

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
    from sklearn.metrics import precision_score
    from sklearn.metrics import classification_report
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
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
#### For Classification problems
from sklearn.naive_bayes import MultinomialNB
def NLP_select_best_model_fit_predict(X, y, test, modeltype, score_type):
    """
    ###############################################################################################
    This is a simple way to build a model for NLP and use train_best and test_best to iterate on it.
    For example: if you have two transformed NLP data sets named train_best and test_best, here's how:
    X = train_best.values
    y = train[target].values
    modeltype = 'Classification'
    score_type = 'balanced_accuracy'
    ### Just call this function as follows:
    predictions, nlp_model = NLP_select_best_model_fit_predict(X,y,test_best,modeltype,score_type)
    ###############################################################################################
    """
    X = copy.deepcopy(X)
    y = copy.deepcopy(y)
    test_size = 0.1
    seed = 99
    start_time = time.time()
    print('##################    BUILDING NLP TRANSFORMATION PIPELINE           #################')
    print('Building Model and Pipeline for NLP column = %s. This will take time...' %nlp_column)
    ### Split into Train and CV to test the model #####################
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
    print('    Train Vectorized data shape = %s, Cross Validation data shape = %s' %(X_train.shape, X_test.shape))
    if modeltype == 'Regression':
        model_name = 'XGB Regressor'
        scv = KFold(n_splits=n_splits)
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
    print('Finding best hyperparameters on Train data and testing on held out data...')
    if str(nlp_model).split("(")[0] == 'MultinomialNB':
        #### Multinomial models need only positive values!!
        X_train = abs(X_train)
        X_test = abs(X_test)
    y_pred = fit_and_predict(gs, X_train, y_train, X_test, modeltype, is_train=True)
    print('    Time taken = %0.2f seconds' %(time.time()-start_time) )

    ##### Print the model results on Cross Validation data set (held out)
    if modeltype == 'Regression':
        print_regression_model_stats(y_test, y_pred,'%s Model: Predicted vs Actual for %s' %(model_name,target))
    else:
        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_classification_matrix(y_test, y_pred, model_name)
    #### Now select the best estimator from the RandomizedSearchCV models
    nlp_model = gs.best_estimator_
    #####  Now AFTER TRAINING, make predictions on the given test data set!
    start_time2 = time.time()
    print('Training Pipeline on full Train data. This will take time...')
    if str(nlp_model).split("(")[0] == 'MultinomialNB':
        X_train = abs(X)
        if not isinstance(test, str):
            test = abs(test)
    if not isinstance(test, str):
        y_pred = fit_and_predict(gs, X, y, test, modeltype, is_train=False)
    print('    Time taken = %0.2f seconds' %(time.time()-start_time2) )
    print('Time taken for Auto_NLP = %0.1f minutes' %((time.time()-start_time)/60))
    print('#########          A U T O   N L P  C O M P L E T E D    ###############################')
    return y_pred, nlp_model
#################################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
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
    data = copy.deepcopy(data)
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
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,4))
            plot_nlp_column(data[col+'_unique_word_count'],"Word Count", ax1, 'r')
            plot_nlp_column(data[col+'_char_count'],"Character Count", ax2, 'b')
    if verbose >= 2:
        if is_train:
            draw_dist_plots_summary_cols(data, target, cols)
    return data, cols
#############################################################################
def plot_nlp_column(df_col, label_title, ax,color='r'):
    """
    We want to know the average number of words per row of text.
    So we first plot the distribution of number of words per text row.
    """
    df_col.hist(bins=30,ax=ax, color=color)
    ax.set_title(label_title);
#############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
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
version_number = '0.0.45'
print("""%s Auto_NLP version: %s.. Call using:
     train_nlp, test_nlp, nlp_pipeline, predictions = Auto_NLP(
                nlp_column, train, test, target, score_type='balanced_accuracy',
                modeltype='Classification',top_num_features=200, verbose=0,
                build_model=True)""" %(module_type, version_number))
########################################################################
