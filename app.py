import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__, static_folder="/home/leon/Documents/python/deployment/static")


def build_model():
    model = Sequential()
    model.add(Dense(25, input_dim=11, activation='relu'))  # Hidden 1
    model.add(Dense(10, activation='relu'))  # Hidden 2
    model.add(Dense(1))  # Output
    return model


model = build_model()
model.load_weights('my_model_weights.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[float(x) for x in request.form.values()]]
    final_features = np.array(int_features)
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Wine quality is predicted to be {}/8'.format(round(float(output), 2)))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
