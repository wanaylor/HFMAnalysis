import sklearn
import numpy as np
from joblib import load
from flask import Flask, request, jsonify

app = Flask(__name__)

data_normalizers = {'combined': load('./preprocessors/scaler_combined.joblib'),
                    '10col': load('./preprocessors/scaler_10col.joblib'),
                    '14col': load('./preprocessors/scaler_14col.joblib')}

models = {'gb_10col': load('./models/gb_10cols.joblib'),
          'gb_14col': load('./models/gb_14cols.joblib'),
          'gb': load('./models/gb_combined.joblib'),
          'knn_10col': load('./models/knn_10cols.joblib'),
          'knn_14col': load('./models/knn_14cols.joblib'),
          'knn': load('./models/knn_combined.joblib'),
          'ridge_10col': load('./models/ridge_10cols.joblib'),
          'ridge_14col': load('./models/ridge_14cols.joblib'),
          'ridge': load('./models/ridge_combined.joblib'),
          }

@app.route("/hfm_failure_prediction", methods=['POST'])
def result():
    print(request.json)
    response = {}
    inference_request = dict(request.json)
    for key in inference_request.keys():
        if '_10col' in key:
            data_normalizer = data_normalizers['10col']
        elif '_14col' in key:
            data_normalizer = data_normalizers['14col']
        else:
            data_normalizer = data_normalizers['combined']
        model = models[key]
        inference_data = inference_request[key]
        inference_data = np.array([inference_data])
        scaled_inference_data = data_normalizer.transform(inference_data)
        prediction = model.predict(scaled_inference_data)
        response[key+'_prediction'] = float(prediction)
    return jsonify(response)

app.run(host='0.0.0.0', port=8080)