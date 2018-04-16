import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import numpy as np
from numpy import array
import random
import nltk
#nltk.download('punkt')
import re

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.util import ngrams
#nltk.download('averaged_perceptron_tagger')

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pylab as plt

class FeatEngr:
    def __init__(self):

        #baseline feature
        #self.vectorizer = CountVectorizer()

        estimators = [('bag-of-words',Pipeline([
                                                ('extract-field', FunctionTransformer(lambda x: x[0], validate = False)),
                                                ('tfidf', TfidfVectorizer(analyzer = "word",ngram_range = (2,2),max_df=0.5))
                                                ])),

                        ('type-of-trope', Pipeline([
                                                 ('extract-field', FunctionTransformer(lambda x: x[1], validate = False)),
                                                 ('tf', TfidfVectorizer())
                                                ])),

                       
                        #('name-of-page', Pipeline([
                         #                       ('extract_field', FunctionTransformer(lambda x: x[2], validate = False)), 
                          #                      ('page', TfidfVectorizer())
                           #                     ])),
                        
                        #('baseline', Pipeline([
                         #                     ('extract-field', FunctionTransformer(lambda x: x[0], validate = False)),
                          #                    ('countvec',CountVectorizer())
                           #                   ]))
                
                                                ]
                        
        featureunion = FeatureUnion(estimators)
       
        self.vectorizer = featureunion
        #self.vectorizer =  TfidfVectorizer( ngram_range = [1,2], max_df=0.5,stop_words = "english")
        #self.vectorizer =  TfidfVectorizer()
        #self.vectorizer = PageTransformer()

    def build_train_features(self, examples):
        """
        Method to take in training text features and do further feature engineering
        Most of the work in this homework will go here, or in similar functions
        :param examples: currently just a list of forum posts
        """
        return self.vectorizer.fit_transform(examples)

    def get_test_features(self, examples):
        """
        Method to take in test text features and transform the same way as train features
        :param examples: currently just a list of forum posts
        """
        return self.vectorizer.transform(examples)

    def show_top10(self):
        """
        prints the top 10 features for the positive class and the
        top 10 features for the negative class.
        """
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        top10 = np.argsort(self.logreg.coef_[0])[-30:]
        bottom10 = np.argsort(self.logreg.coef_[0])[:30]
        print("Pos: %s" % ", ".join(feature_names[top10]))
        print("Neg: %s" % " ,".join(feature_names[bottom10]))

    def train_model(self, random_state=1234):
        """
        Method to read in training data from file, and
        train Logistic Regression classifier.

        :param random_state: seed for random number generator
        """       

        # load data
        dfTrain = pd.read_csv("../data/spoilers/train.csv")
        

        #imdb = pd.DataFrame.from_csv("../data/spoilers/title.basics.tsv",sep='\t')
        #eqguide = pd.read_csv("../data/spoilers/allshows.csv")

        shape = dfTrain.shape
        #print(list(dfTrain["trope"]))
        # get training features and labels
        #print(list(dfTrain["spoiler"]).count(False))

        post = []
        print()
        #for sentence in list(dfTrain["sentence"]):
        text = word_tokenize(dfTrain["sentence"][0])

        text = pos_tag(text)
        print(text[0][1])        

        
        ####error analysis###
        mytrain, mytest = train_test_split(dfTrain, test_size=0.2, shuffle=False, random_state=1230)
        self.X_train = self.build_train_features([list(mytrain["sentence"]),list(mytrain["trope"]),list(mytrain["page"])])
        #self.X_train = self.build_train_features(list(mytrain["sentence"]))

        self.y_train = np.array(mytrain["spoiler"], dtype=int)

        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)
        self.X_test = self.get_test_features([list(mytest["sentence"]),list(mytest["trope"]),list(mytest["page"])])
        #self.X_test = self.get_test_features(list(mytest["sentence"]))

        pred = self.logreg.predict(self.X_test)

        misclassified = pd.Series(np.array(mytest['spoiler']) != pred)
        print(mytest[misclassified.values][['spoiler', 'sentence','trope']])
        #np.savetxt(r'../data/spoilers/err.txt', mytest[misclassified.values][['spoiler', 'sentence']], fmt='%d')
        ###error analysis###

        """
        ###train the model
        self.X_train = self.build_train_features(list(dfTrain["sentence"]))
        #self.X_train = self.build_train_features([list(dfTrain["sentence"]),list(dfTrain["trope"]),list(dfTrain["page"]), post])
        self.y_train = np.array(dfTrain["spoiler"], dtype=int)

       
        # train logistic regression model.  !!You MAY NOT CHANGE THIS!!
        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)

        
        #do 5-fold cross validation
        scores = cross_val_score(self.logreg, self.X_train, self.y_train, cv=5)
        print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        y_pred = cross_val_predict(self.logreg,self.X_train,self.y_train,cv=5)
        conf_mat = confusion_matrix(self.y_train,y_pred)
        print(conf_mat)
        """


    def model_predict(self):
        """
        Method to read in test data from file, make predictions
        using trained model, and dump results to file
        """

        # read in test data
        dfTest = pd.read_csv("../data/spoilers/test.csv")


        # featurize test data
        self.X_test = self.get_test_features([list(dfTest["sentence"]),list(dfTest["trope"]),list(dfTest["page"])])

        # make predictions on test data
        pred = self.logreg.predict(self.X_test)

        # dump predictions to file for submission to Kaggle
        pd.DataFrame({"spoiler": np.array(pred, dtype=bool)}).to_csv("prediction.csv", index=True, index_label="Id")


class PageTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
            
        # Initiaize matrix 
        X = np.zeros((len(examples[0]), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples[0]):            
            sentence_word = nltk.word_tokenize(x)
            page = re.sub(r'([A-Z])', r' \1', examples[1][ii])           
            page_word = nltk.word_tokenize(page)
            common_word = set(sentence_word).intersection(page_word)
            # Remove "The" from common_word
            if 'The' in common_word:
                common_word.remove('The')
                
            X[ii,:] = len(common_word)
        
        X = preprocessing.normalize(X, norm='l2')
        return csr_matrix(X)     



# Instantiate the FeatEngr class
feat = FeatEngr()

# Train your Logistic Regression classifier
feat.train_model(random_state=1230)

# Shows the top 10 features for each class
#feat.show_top10()

# Make prediction on test data and produce Kaggle submission file
#feat.model_predict()

