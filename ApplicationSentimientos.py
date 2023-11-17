from flask import Flask, render_template, request
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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nb', methods=['GET', 'POST'])
def nb_interface():
    result = None

    if request.method == 'POST':
        user_text = request.form['user_text']
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([user_text])
        # Realizar la predicción utilizando el modelo Naive Bayes cargado
        polarity = nb_model.predict(tweet_tfidf)
        result = f"Polaridad del texto: {polarity[0]}"

    return render_template('nb_interface.html', result=result)

@app.route('/svm', methods=['GET', 'POST'])
def svm_interface():
    result = None

    if request.method == 'POST':
        user_text = request.form['user_text']
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([user_text])
        # Realizar la predicción utilizando el modelo SVM cargado
        polarity = svm_classifier.predict(tweet_tfidf)
        result = f"Polaridad del texto: {polarity[0]}"

    return render_template('svm_interface.html', result=result)

@app.route('/rf', methods=['GET', 'POST'])
def rf_interface():
    result = None

    if request.method == 'POST':
        user_text = request.form['user_text']
        # Transformar el nuevo tweet utilizando el vectorizador TF-IDF
        tweet_tfidf = vectorizer.transform([user_text])
        # Realizar la predicción utilizando el modelo Random Forest cargado
        polarity = rf_model.predict(tweet_tfidf)
        result = f"Polaridad del texto: {polarity[0]}"

    return render_template('rf_interface.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
