from flask import Flask, request, render_template
import pickle
import numpy as n

app = Flask(__name__)
model = pickle.load(open(
    'C:\\Users\\cemre\\Documents\\repos\\Diabetes-Prediction-Flask-App\\model.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [n.array(float_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1:
        return render_template('index.html', prediction_text='Your risk of diabetes is very high! Please go to a doctor to check.')
    else:
        return render_template('index.html', prediction_text='Your risk of diabets is low.')


if __name__ == "__main__":
    app.run(port=5000, debug=True)
