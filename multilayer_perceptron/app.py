import flask
import numpy as np
from flask import Flask, jsonify, request
import json
import pickle

app = Flask(__name__)


def load_network():
    file_name = "../dist/network_file.p"
    with open(file_name, "rb") as pickled:
        data = pickle.load(pickled)
        network = data["network"]
    return network


@app.route("/predict", methods=["GET"])
def predict():
    request_json = request.get_json()
    image = np.array(request_json["input"])
    network = load_network()
    prediction = network.predict([image])[0]
    response = json.dumps({"response": int(prediction)})
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
