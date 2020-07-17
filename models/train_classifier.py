# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import matplotlib.pyplot as plt
import pickle

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, 
precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier

def load_data(database_filepath):
    """ Creates a sqlite engine, downloads cleaned dataframe from database and 
    	loads it as pandas dataframe. 
    	Defines input and output variables and creates a list of category names.
    
    Parameters:
        'database_filepath': file path to database
        
    Returns:
        'X': pandas dataframe with input variable, messages to be classified
        'Y': pandas dataframe with output variables, category columns to be 
        used as classification labels
        'category_names': list of category names to be used for classification 
        report
    """
    # create sqlite engine and download data from database, load as pandas 
    # dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
   
    # define input and output variables
    X = df.message
    Y = df.loc[:, 'related':]
    
    # define category names
    category_names = df.loc[:, 'related':]
    
    return X, Y, category_names

def tokenize(text):
    """ Function that normalizes the message text by removing punctuation 
    	characters and converts it to lowercase.
        Tokenizes the message text by splitting it into words, removes stop 
        words and reduces words to their stem and root form.
    
    Parameters:
        'text': message string to be tokenized
        
    Returns:
        'lemmed': list of tokenized, stemmed and lemmed words
    """
    # normalize text - remove punctuation characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize text - split text into words
    words = word_tokenize(text)
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed

def build_model():
    """ Defines the machine learning pipeline, the parameters to tune and 
    	creates a grid search object that includes a search function over the 
    	specified parameter values for the estimators.
    
    Parameters:
        None
        
    Returns:
        'cv': GridSearchCV object including estimators and parameters
    """
    # define ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # set parameters to tune
    parameters = {
        #'clf__estimator__algorithm': ['SAMME', 'SAMME.R'],
        #'clf__estimator__learning_rate': [0.1, 1.0],
        #'clf__estimator__n_estimators': [50, 100]
        'clf__estimator__n_estimators': [100]
    }
    
    # create grid search object
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=2)
    
    # train classifier using grid search to find best parameters
    #cv.fit(X_train, Y_train)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Makes model prediction on test set and prints model performance metrics 
    	in classification report
    
    Parameters:
        'model': Trained GridSearchCV object
        'X_test': test set for input variable, messages to be classified
        'Y_test': test set for output variables, category columns to be used as 
        classification labels
        'category_names': list of category names to be used in classification 
        report
        
    Returns:
        None
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """ Saves model as pickle file to specified file path
    
    Parameters:
        'model': Trained GridSearchCV object
        'model_filepath': file path to save model to
        
    Returns:
        None
    """
    # save model as pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()