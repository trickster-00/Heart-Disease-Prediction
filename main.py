from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('heart_rf.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    int_features = [i for i in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output <= 0:
        return render_template('home.html', prediction_text = "Suffering from Heart Disease")
    else:
        return render_template('home.html', prediction_text = 'No Heart Disease Detected ')


if __name__ == "__main__":
    app.run(debug = True)