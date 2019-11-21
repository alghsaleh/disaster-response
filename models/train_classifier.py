# Essentials and Utilities
import sys
import re
import os
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Tokenization and Lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support

def load_data(database_filepath):
    """
    """

    # Create engine for SQLite database
    engine = create_engine(os.path.join('sqlite:///', database_filepath))

    # Read table into DataFrame
    df = pd.read_sql_table('Messages', con=engine)

    # Split DataFrame to features and targets
    X = df['message'].values
    y = df.drop(columns=['id', 'message', 'original', 'genre']).values

    # Retain target category names
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns.values

    return X, y, category_names


def tokenize(text, url_regex='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'):
    """
    """

    # Replace URLs with `urlplaceholder` in text
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Convert text to lowercase
    text = text.lower()

    # Normalize text by removing punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords from tokens
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize tokens
    tokens = [WordNetLemmatizer().lemmatize(i) for i in tokens]

    return tokens


def build_model():
    """
    """

    # Create and instantiate pipeline
    pipeline = Pipeline([

        # Transform estimators
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        # Predict estimator
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Create GridSearchCV object
    model = GridSearchCV(
        estimator=pipeline,
        param_grid={'clf__estimator__max_depth': [10, 50],
                    'clf__estimator__n_estimators': [50, 100]},
        cv=2, n_jobs=-1, verbose=20)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    """

    # Predict on test data using trained classifier
    y_pred = model.predict(X_test)

    # Initiate empty metrics_list
    metrics_list = []

    # Iterate through columns to get metrics
    for i in range(len(category_names)):
        precision, recall, f1_score, support = precision_recall_fscore_support(
            Y_test[:,i], y_pred[:,i],
            # Weighted average to account for labels imbalance
            average='weighted')

        # Append metrics to metrics_list
        metrics_list.append([precision, recall, f1_score])

    # Transform metrics_list to DataFrame
    metrics_df = pd.DataFrame(
        data=np.array(np.around(metrics_list,2)),
        index=category_names,
        columns=['precision', 'recall', 'f1_score']
    )

    # Display metrics_df
    print(metrics_df)


def save_model(model, model_filepath):
    """
    """

    # Export model as pickle file
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
