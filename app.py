from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer saved from your training script
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = ""

    if request.method == 'POST':
        user_input = request.form.get('text', '').strip()
        if user_input:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
        else:
            prediction = "Please enter some text to analyze."

    return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
