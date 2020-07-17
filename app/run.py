import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
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
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load('../models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    df2 = df.drop(columns='child_alone')
    
    genre_counts = df2.groupby('genre').count()['message']
    genre_names = genre_counts.index.tolist()
    genre_names = [item.capitalize() for item in genre_names]
    
    category_counts = df2.loc[:, 'related':].sum()
    category_names = category_counts.index.tolist()
    category_names = [item.replace("_", " ").capitalize() for item in category_names]
    
    corr_matrix = df2.loc[:, 'related':].corr().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                {
                    'values': genre_counts,
                    'labels': genre_names,
                    'type': 'pie',
                }
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Category'
                },
                'height': 700,
                'margin': dict(
                    b = 250
                )
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr_matrix,
                    x=category_names,
                    y=category_names,
                    colorscale=[
                        [0, '#1F77B4'],
                        [0.5, '#2CA02C'],
                        [1, '#FF7F0F']
                    ]
                )
            ],

            'layout': {
                'title': 'Correlation Matrix of Categories Occuring in the Same Message',
                'height': 1070,
                'margin': dict(
                    l = 150,
                    b = 160
                )
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
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