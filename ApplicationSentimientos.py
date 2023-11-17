import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo Naive Bayes desde el archivo
nb_model = joblib.load('modelo_nb.pkl')

# Cargar el modelo SVM entrenado
svm_classifier = joblib.load('modelo_svm.pkl')

# Cargar el modelo Random Forest desde el archivo
rf_model = joblib.load('modelo_svm.pkl')

# Cargar el vectorizador TF-IDF (asegúrate de que tengas un archivo vectorizador.pkl)
vectorizer = joblib.load('vectorizador.pkl')

# Inicializar la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Definir el diseño de la aplicación
app.layout = html.Div([
    # Interfaz de bienvenida
    html.Div([
        html.H1("Bienvenido al Análisis de Sentimientos"),
        html.P("Seleccione el modelo de análisis de sentimientos que desea utilizar:"),
        html.Button('Naive Bayes', id='nb-button'),
        html.Button('SVM', id='svm-button'),
        html.Button('Random Forest', id='rf-button')
    ], id='welcome-interface'),

    # Interfaz para el análisis de sentimientos con Naive Bayes
    html.Div([
        html.H1("Análisis de Sentimientos - Naive Bayes"),
        dcc.Textarea(id='nb-user_text', placeholder='Ingresa tu texto', style={'width': '100%', 'height': 100}),
        html.Button('Analizar', id='nb-analyze-button'),
        html.Div(id='nb-result')
    ], id='nb-interface', style={'display': 'none'}),  # Inicialmente oculto

    # Interfaz para el análisis de sentimientos con SVM
    html.Div([
        html.H1("Análisis de Sentimientos - SVM"),
        dcc.Textarea(id='svm-user_text', placeholder='Ingresa tu texto', style={'width': '100%', 'height': 100}),
        html.Button('Analizar', id='svm-analyze-button'),
        html.Div(id='svm-result')
    ], id='svm-interface', style={'display': 'none'}),  # Inicialmente oculto

    # Interfaz para el análisis de sentimientos con Random Forest
    html.Div([
        html.H1("Análisis de Sentimientos - Random Forest"),
        dcc.Textarea(id='rf-user_text', placeholder='Ingresa tu texto', style={'width': '100%', 'height': 100}),
        html.Button('Analizar', id='rf-analyze-button'),
        html.Div(id='rf-result')
    ], id='rf-interface', style={'display': 'none'})  # Inicialmente oculto
])

# Callback para mostrar y ocultar las interfaces según el botón seleccionado
@app.callback(
    [Output('nb-interface', 'style'),
     Output('svm-interface', 'style'),
     Output('rf-interface', 'style')],
    [Input('nb-button', 'n_clicks'),
     Input('svm-button', 'n_clicks'),
     Input('rf-button', 'n_clicks')]
)
def toggle_interfaces(nb_clicks, svm_clicks, rf_clicks):
    triggered_id = dash.callback_context.triggered_id
    if triggered_id == 'nb-button':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif triggered_id == 'svm-button':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif triggered_id == 'rf-button':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callbacks para realizar el análisis de sentimientos con los modelos seleccionados
@app.callback(
    [Output('nb-result', 'children'),
     Output('svm-result', 'children'),
     Output('rf-result', 'children')],
    [Input('nb-analyze-button', 'n_clicks'),
     Input('svm-analyze-button', 'n_clicks'),
     Input('rf-analyze-button', 'n_clicks')],
    [dash.dependencies.State('nb-user_text', 'value'),
     dash.dependencies.State('svm-user_text', 'value'),
     dash.dependencies.State('rf-user_text', 'value')]
)
def analyze_sentiments_nb(nb_clicks, svm_clicks, rf_clicks, nb_user_text, svm_user_text, rf_user_text):
    if nb_clicks is not None:
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([nb_user_text])

        # Realizar la predicción utilizando el modelo Naive Bayes cargado
        polarity = nb_model.predict(tweet_tfidf)

        return f"Polaridad del texto: {polarity[0]}", None, None

    elif svm_clicks is not None:
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([svm_user_text])

        # Realizar la predicción utilizando el modelo SVM cargado
        polarity = svm_classifier.predict(tweet_tfidf)

        return None, f"Polaridad del texto: {polarity[0]}", None

    elif rf_clicks is not None:
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([rf_user_text])

        # Realizar la predicción utilizando el modelo Random Forest cargado
        polarity = rf_model.predict(tweet_tfidf)

        return None, None, f"Polaridad del texto: {polarity[0]}"
    else:
        return None, None, None

# Iniciar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)
