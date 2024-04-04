# HFMAnalysis
Exploratory analysis and model development for predicting Hot Finishing Mill failure

# Environment
There are two environment files that should allow setup of the current environment, a conda-env.txt for anaconda environments and requirements.txt for pip environments. However, the dependencies are mainly numpy, pandas, sklearn, plotly, flask and jupyter.

# Files
Most development took place in analysis.ipynb. If run in jypyter it will train the models and save them off for use in the API. It also has inline plotting of the plotly graphs as well as saving them to ./figures. There is also an analysis.py file which is just a dump of analysis.ipynb that can be run without jupyter. analysis.py has all plot showing commented out but still saves plots to ./figures. Word and PDF versions of the report are also available here.

# App
Inference is served via a standalone app serving an api endpoint. To run the app run the app.py file and it should listen on localhost:8080.

Alternatively there is a DOCKERFILE provided to build a docker container to serve the app. To build run the following 

`docker build . -t hfmanalysis:amd64 -f DOCKERFILE`

Then to run the app in the container run

`docker run -it -p 8080:8080 hfmanalysis:amd64 python3 app.py`

The api allows you to send a POST message for model inference. You can select inference from any of the models developed in the Jupyter notebook by providing a json object with the model name as the key and an array of values for the sensor data. If _10col or _14col is not specified in the key the models default to the larger model that was trained on the combined datasets. Example payload:

```
{
  "gb": [sen2_value, sen3_value, sen10_value, sen12_value, sen13_value, sen14_value, sen16_value, sen1_value, sen6_value, sen7_value, sen8_value, sen11_value, sen19_value, sen20_value],
  "knn_10col": [sen2_value, sen3_value, sen10_value, sen12_value, sen13_value, sen14_value, sen16_value],
  "ridge_14col": [sen1_value, sen6_value, sen7_value, sen8_value, sen11_value, sen19_value, sen20_value]
}
```

The full payload would look like this:

`curl -X POST -H "Content-Type: application/json" -d '{"gb": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], "gb_10col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}' http://localhost:8080/hfm_failure_prediction`
