import json
import plotly
import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    # Figure 1: "Distribution of Messages by Genre and Relevance"
    genre_relevant = df[df['related']==1].groupby('genre').count()['message']
    genre_irrelevant = df[df['related']==0].groupby('genre').count()['message']
    genre_names = list(genre_relevant.index)

    # Figure 2: "Percentage of Relevant Messages by Top 10 Categorie"
    cat_pct = df.drop(columns=['id', 'message', 'original', 'genre', 'related']).sum() / len(df) * 100
    cat_pct = cat_pct.sort_values(ascending=False).head(10)
    cat_names = list(cat_pct.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_relevant,
                    name = 'Relevant'
                ),

                Bar(
                    x=genre_names,
                    y=genre_irrelevant,
                    name = 'Irrelevant'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Relevance',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_pct
                )
            ],

            'layout': {
                'title': 'Percentage of Relevant Messages by Top 10 Categories',
                'yaxis': {
                    'title': "Percentage (%)"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
