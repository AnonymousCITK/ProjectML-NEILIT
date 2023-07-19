from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model_path = 'logistic_regression_model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as f:
    loaded_lr_model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])
    std_data = loaded_scaler.transform(input_data)

    # Make predictions using the loaded model
    pred = loaded_lr_model.predict(std_data)

    # Interpret the prediction
    if pred[0] == 0:
        result_text = "The person does not have heart disease."
    else:
        result_text = "The person has heart disease."

    return redirect(url_for('result', result=result_text))
@app.route('/result')
def result():
    result_text = request.args.get('result')
    return render_template('result.html', result=result_text)
if __name__ == '__main__':
    app.run(debug=True)