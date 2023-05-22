import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report

import xgboost as xgb
import lightgbm as lgb
import sys

import warnings

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):

    """
    loads data from database
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_mesages', engine.connect())

    df.drop('original', axis=1, inplace=True)

    X = df['message']
    Y = df.iloc[:, 4:]

    return X,Y


def tokenize(text):

    """
    normalize, tokenize, remove stop words and lemmatize text
    
    Parameters:
    text: Text to be processed

    Returns:
    clean_tokens: Processed text
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
                  
    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # Lemmatization

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()

        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():

    """
    Build a model

    Returns:
    cv: GridSearchCV object
    """

    pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=tokenize)),
    ('clf',MultiOutputClassifier(lgb.LGBMClassifier()))])

    parameters = {

    'clf__estimator__learning_rate': [0.1,0.5]}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose = 3)

    return cv


def evaluate_model(model, X_test, Y_test):

    """

    Evaluate model

    Parameters:
    model: Model to be evaluated
    X_test: Test features
    Y_test: Test target

    """
    
    pred = model.predict(X_test)

    f1_score(Y_test, pred, average='micro')
    print(classification_report(Y_test, pred, target_names=Y_test.columns))


def save_model(model, model_filepath):

    """
    Save model

    Parameters:
    model: Model to be saved
    model_filepath: Filepath to save the model

    """
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
        evaluate_model(model, X_test, Y_test)

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