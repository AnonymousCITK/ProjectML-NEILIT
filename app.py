from flask import Flask, render_template, request
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
    # input_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # # Collect form data as list
    # features = [float(request.values.get(field)) for field in input_fields]

    # # Create a NumPy array from the collected form data
    # features_array = np.array(features).reshape(1, -1)
    float_features = [float(x) for x in request.values()]
    features = [np.array(float_features)]
    input_data_reshape = features.reshape(1, -1)
    std_data = loaded_scaler.transform(input_data_reshape)

    # Make predictions using the loaded model
    pred = loaded_lr_model.predict(std_data)

    # Interpret the prediction
    if pred[0] == 0:
        result_text = "The person does not have heart disease."
    else:
        result_text = "The person has heart disease."

    return render_template('index.html', result=result_text)
if __name__ == '__main__':
    app.run(debug=True)