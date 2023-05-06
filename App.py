from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from xgboost import XGBRegressor


app = Flask(__name__)
app.secret_key = "QWERTY123"

model = joblib.load(open('xgb_modelHP.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    items = [np.array(float_features)]
    print(float_features)
    print(items)
    prediction = model.predict(items)
    # prediction = float(prediction[0])

    if prediction:
        output = np.round(prediction[0], 2)
        return render_template('index.html', prediction='rate is {}'.format(output))
    else:
        return render_template('index.html', prediction='Invalid input. Please try again.')


if __name__ == '__main__':
    app.run(port=5000)
