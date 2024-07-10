from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Define class labels for ham and spam
class_labels = {0: "ham", 1: "spam"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(f"Received message: {message}")  # Debugging line
        if message:
            data = [message]
            vect = tfidf.transform(data)
            print(f"Transformed vector shape: {vect.shape}")  # Debugging line
            prediction = model.predict(vect)
            print(f"Prediction: {prediction}")  # Debugging line
            predicted_label = class_labels.get(prediction[0], "Unknown")
            return render_template('index.html', prediction=predicted_label)
        else:
            print("No message received")
    return render_template('index.html', prediction="No prediction")

if __name__ == '__main__':
    app.run(debug=True)
