import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Cargar el archivo Excel
df = pd.read_excel('tweet_preprocesado.xlsx')

# Obtener los tweets y las etiquetas
X = df['Tweet preprocesado']
y = df['polaridad']

# Crear una instancia del vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Transformar los datos utilizando el vectorizador TF-IDF
X_tfidf = vectorizer.fit_transform(X)

# Crear una instancia del clasificador Random Forest
rf_classifier = RandomForestClassifier()

# Realizar validación cruzada con 5 folds
scores = cross_val_score(rf_classifier, X_tfidf, y, cv=5)

# Obtener el promedio de los scores
average_score = scores.mean()

# Configurar la aplicación Dash
app = dash.Dash(__name__)


# Cargar el modelo desde el archivo
nb_model = joblib.load('modelo_nb.pkl')

# Cargar el vectorizador TF-IDF (asegurate de que tengas un archivo vectorizador.pkl)
vectorizer = joblib.load('vectorizador.pkl')

app = dash.Dash(__name__)

# Disenio de la interfaz para Analisis de Sentimientos
sentiments_layout = html.Div([
    html.H1("Analisis de Sentimientos"),
    dcc.Textarea(id='user_text', placeholder='Ingresa tu texto', style={'width': '100%', 'height': 100}),
    html.Button('Analizar', id='analyze-button'),
    html.Div(id='result')
])

# Disenio de la interfaz para Validacion Cruzada - Random Forest (copiado del codigo anterior)
cross_validation_layout = html.Div([
    html.H1("Validacion Cruzada - Random Forest"),
    dcc.Graph(
        id='cross-validation-plot',
        figure={
            'data': [
                {'x': range(1, len(scores) + 1), 'y': scores * 100, 'type': 'bar', 'name': 'Fold'},
            ],
            'layout': {
                'xaxis': {'title': 'Categoria'},
                'yaxis': {'title': 'Precision (%)'},
                'title': 'Validacion Cruzada - Random Forest',
                'legend': {'x': 0, 'y': 1},
                'annotations': [
                    {'x': i + 1, 'y': score * 100, 'text': f'{score:.2f}', 'showarrow': True, 'arrowhead': 2}
                    for i, score in enumerate(scores)
                ],
                'shapes': [
                    {'type': 'line', 'x0': 0, 'x1': len(scores), 'y0': average_score * 100, 'y1': average_score * 100,
                     'line': {'color': 'red', 'dash': 'dash'}}
                ]
            }
        }
    )
])

# Disenio general de la aplicacion
app.layout = html.Div([
    html.Div([
        dcc.Link('Analisis de Sentimientos', href='/sentiments'),
        html.Br(),
        dcc.Link('Validacion Cruzada - Random Forest', href='/cross-validation'),
        html.Br(),
        html.Br(),
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/cross-validation':
        return cross_validation_layout
    else:
        return sentiments_layout

@app.callback(
    Output('result', 'children'),
    [Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('user_text', 'value')]
)
def analyze_sentiments(n_clicks, user_text):
    if n_clicks is not None:
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([user_text])

        # Realizar la prediccion utilizando el modelo cargado
        polarity = nb_model.predict(tweet_tfidf)

        return f"Polaridad del texto: {polarity[0]}"

if __name__ == "__main__":
    app.run_server(debug=True)