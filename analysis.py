#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from joblib import dump
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ## Assumptions
# * no_of_days means days up or online, not how many days remain until an observed failure happened. I will assume that a failure happened the day after the highest no_of_days value for each given run. For instance day 192 was the last no_of_days for the first hfm run I will assume a failure happened on day 193 before the reading could be taken.
# * Each observation is stand alone and does not show autocorrelation with previous days readings. In short each observation is a snapshot in time
# * •	The 10 column and 14 column datasets were collected at the same time. The hfm run numbers and duration in days align between the two datasets so we can assume that the two datasets can be combined into one large dataset if necessary.

# ## First we will do some feature engineering, we are given no_of_days which mean uptime. We are trying to predict days until failure which would be MAX_DAYS_UP - no_of_days so we will add that column to our datasets

# In[30]:


dataset_10col = pd.read_csv("./hfm_10cols.csv")
dataset_14col = pd.read_csv("./hfm_14cols.csv")

dataset_10col_groups = dataset_10col.groupby("hfm_runs")
dataset_14col_groups = dataset_14col.groupby("hfm_runs")

# get the max no_of_days for each dataset to calculate days_until_failure
max_days_of_groups_10col = dataset_10col.groupby("hfm_runs")["no_of_days"].max()
max_days_of_groups_14col = dataset_14col.groupby("hfm_runs")["no_of_days"].max()

# insert dummy column for days_until_failure
dataset_10col["days_until_failure"] = np.NaN
dataset_14col["days_until_failure"] = np.NaN

for idx, row in dataset_10col.iterrows():
    max_day_of_group = max_days_of_groups_10col[int(row["hfm_runs"])]
    dataset_10col.at[idx, "days_until_failure"] = max_day_of_group - row["no_of_days"] + 1 # so we dont hit 0

for idx, row in dataset_14col.iterrows():
    max_day_of_group = max_days_of_groups_14col[int(row["hfm_runs"])]
    dataset_14col.at[idx, "days_until_failure"] = max_day_of_group - row["no_of_days"] + 1 # so we dont hit 0


# ## Next we will look at graphs of the sensors over all the runs, I have always found it very useful to start with some data viz, especially with sensor data

# ### Data viz for the 10 column dataset

# In[31]:


# Make one figure with subplots and add traces to avoid WebGL limitations
fig = make_subplots(rows=8, cols=1, subplot_titles=["sensor_2", "sensor_3", "sensor_10", "sensor_12", "sensor_13", "sensor_14", "sensor_16", "sensor_17"])
sensor_plots = []
i = 1
for sensor in ["sensor_2", "sensor_3", "sensor_10", "sensor_12", "sensor_13", "sensor_14", "sensor_16", "sensor_17"]:
    #sns.lineplot(data=dataset_10col, x="no_of_days", y=sensor, hue= "run " + dataset_10col["hfm_runs"].astype(str))
    sensor_plot = px.line(dataset_10col, x="no_of_days", y=sensor, line_group="hfm_runs", color="hfm_runs")
    sensor_plots.append(sensor_plot)
    for d in sensor_plot.data:
        fig.add_trace((go.Scatter(x=d["x"], y=d["y"], name=d['name'])), row=i, col=1)
    i += 1
    
fig.update_layout(height=4000)
#fig.show()
fig.write_html("./figures/10column.html")


# ## We see here that sensor 17 does not provide any information so we will drop it. It could be a setpoint which would explain why it wouldn't change

# In[32]:


print(f'The min and max for sensor 17 respectively are: {dataset_10col["sensor_17"].min()} {dataset_10col["sensor_17"].max()}')
dataset_10col = dataset_10col.drop(columns="sensor_17")


# ### Data viz for the 14 column dataset

# In[33]:


# Make one figure with subplots and add traces to avoid WebGL limitations
fig = make_subplots(rows=6, cols=1, subplot_titles=["sensor_1", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8"])
sensor_plots = []
i = 1
for sensor in ["sensor_1", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8"]:
    sensor_plot = px.line(dataset_14col, x="no_of_days", y=sensor, line_group="hfm_runs", color="hfm_runs")
    sensor_plots.append(sensor_plot)
    for d in sensor_plot.data:
        fig.add_trace((go.Scatter(x=d["x"], y=d["y"], name=d['name'])), row=i, col=1)
    i += 1

fig.update_layout(height=4000)
#fig.show()
fig.write_html("./figures/14column_1.html")


# In[34]:


# Make one figure with subplots and add traces to avoid WebGL limitations
fig = make_subplots(rows=6, cols=1, subplot_titles=["sensor_9", "sensor_11", "sensor_15", "sensor_18", "sensor_19", "sensor_20"])
sensor_plots = []
i = 1
for sensor in ["sensor_9", "sensor_11", "sensor_15", "sensor_18", "sensor_19", "sensor_20"]:
    sensor_plot = px.line(dataset_14col, x="no_of_days", y=sensor, line_group="hfm_runs", color="hfm_runs")
    sensor_plots.append(sensor_plot)
    for d in sensor_plot.data:
        fig.add_trace((go.Scatter(x=d["x"], y=d["y"], name=d['name'])), row=i, col=1)
    i += 1

fig.update_layout(height=4000)
#fig.show()
fig.write_html("./figures/14column_2.html")


# ## In the 14 column dataset sensors 4, 5, 9, 15, and 18 also appear to be nonfunctional or setpoints and do not provide us with any information so we will drop these

# In[16]:


print(f'The min and max for sensor 4 respectively are: {dataset_14col["sensor_4"].min()} {dataset_14col["sensor_4"].max()}')
print(f'The min and max for sensor 5 respectively are: {dataset_14col["sensor_5"].min()} {dataset_14col["sensor_5"].max()}')
print(f'The min and max for sensor 9 respectively are: {dataset_14col["sensor_9"].min()} {dataset_14col["sensor_9"].max()}')
print(f'The min and max for sensor 15 respectively are: {dataset_14col["sensor_15"].min()} {dataset_14col["sensor_15"].max()}')
print(f'The min and max for sensor 18 respectively are: {dataset_14col["sensor_18"].min()} {dataset_14col["sensor_18"].max()}')

dataset_14col = dataset_14col.drop(columns="sensor_4")
dataset_14col = dataset_14col.drop(columns="sensor_5")
dataset_14col = dataset_14col.drop(columns="sensor_9")
dataset_14col = dataset_14col.drop(columns="sensor_15")
dataset_14col = dataset_14col.drop(columns="sensor_18")


# ## Model Building

# ## Splitting into train and test sets and normalization

# In[17]:


# Scaler for data normalization
scaler_10col = MinMaxScaler()
# 10 column dataset
y_10col = dataset_10col["days_until_failure"]
X_10col = dataset_10col.drop(columns=["hfm_runs", "no_of_days", "days_until_failure"])
scaler_10col.fit(X_10col)
X_10col = scaler_10col.transform(X_10col)
dump(scaler_10col, "./preprocessors/scaler_10col.joblib")
X_train_10col, X_test_10col, y_train_10col, y_test_10col = train_test_split(X_10col, y_10col, test_size=0.2, random_state=235)

# 14 column dataset
scaler_14col = MinMaxScaler()
y_14col = dataset_14col["days_until_failure"]
X_14col = dataset_14col.drop(columns=["hfm_runs", "no_of_days", "days_until_failure"])
scaler_14col.fit(X_14col)
X_14col = scaler_14col.transform(X_14col)
dump(scaler_14col, "./preprocessors/scaler_14col.joblib")
X_train_14col, X_test_14col, y_train_14col, y_test_14col = train_test_split(X_14col, y_14col, test_size=0.2, random_state=235)

# Combining the 10 column and 14 column datasets
scaler_combined = MinMaxScaler()
y_combined = dataset_10col["days_until_failure"]
X_combined = np.concatenate([dataset_10col.drop(columns=["hfm_runs", "no_of_days", "days_until_failure"]),
                             dataset_14col.drop(columns=["hfm_runs", "no_of_days", "days_until_failure"])], axis=1)
scaler_combined.fit(X_combined)
X_combined = scaler_combined.transform(X_combined)
dump(scaler_combined, "./preprocessors/scaler_combined.joblib")
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=235)


# ### Ridge Regression

# In[11]:


# Start Ridge regression fitting and evaluation
ridge_params = {'alpha': [0.001, 0.1, 0.5, 1.0, 5, 50, 100],
                "solver": ["auto", "svd", "cholesky", "lsqr"]}
ridge_model = linear_model.Ridge()
ridge_10col_clf = GridSearchCV(ridge_model, ridge_params)
ridge_10col_clf = ridge_10col_clf.fit(X_train_10col, y_train_10col)
ridge_14col_clf = GridSearchCV(ridge_model, ridge_params)
ridge_14col_clf = ridge_14col_clf.fit(X_train_14col, y_train_14col)
ridge_combined_clf = GridSearchCV(ridge_model, ridge_params)
ridge_combined_clf = ridge_combined_clf.fit(X_train_combined, y_train_combined)


# In[12]:


print(ridge_10col_clf.best_params_)
print(ridge_14col_clf.best_params_)
print(ridge_combined_clf.best_params_)


# In[25]:


# Fit with the best parameters for the 10col dataset and evaluate metrics
ridge_10col_model = linear_model.Ridge(**ridge_10col_clf.best_params_)
ridge_10col_model.fit(X_train_10col, y_train_10col)
ridge_10col_predictions = ridge_10col_model.predict(X_test_10col)
ridge_10col_mae = mean_absolute_error(y_test_10col, ridge_10col_predictions)
ridge_10col_mape = mean_absolute_percentage_error(y_test_10col, ridge_10col_predictions)
ridge_10col_r2 = r2_score(y_test_10col, ridge_10col_predictions)

# Fit with the best parameters for the 14col dataset and evaluate metrics
ridge_14col_model = linear_model.Ridge(**ridge_14col_clf.best_params_)
ridge_14col_model.fit(X_train_14col, y_train_14col)
ridge_14col_predictions = ridge_14col_model.predict(X_test_14col)
ridge_14col_mae = mean_absolute_error(y_test_14col, ridge_14col_predictions)
ridge_14col_mape = mean_absolute_percentage_error(y_test_14col, ridge_14col_predictions)
ridge_14col_r2 = r2_score(y_test_14col, ridge_14col_predictions)

# Fit with the best parameters for the combined dataset and evaluate metrics
ridge_combined_model = linear_model.Ridge(**ridge_combined_clf.best_params_)
ridge_combined_model.fit(X_train_combined, y_train_combined)
ridge_combined_predictions = ridge_combined_model.predict(X_test_combined)
ridge_combined_mae = mean_absolute_error(y_test_combined, ridge_combined_predictions)
ridge_combined_mape = mean_absolute_percentage_error(y_test_combined, ridge_combined_predictions)
ridge_combined_r2 = r2_score(y_test_combined, ridge_combined_predictions)

print(f"The mean average error of Ridge regression on the 10 column dataset is {ridge_10col_mae}")
print(f"The mean average error of Ridge regression on the 14 column dataset is {ridge_14col_mae}")
print(f"The mean average error of Ridge regression on the combined dataset is {ridge_combined_mae}")
print(f"The mean average percentage error of Ridge regression on the 10 column dataset is {ridge_10col_mape}")
print(f"The mean average percentage error of Ridge regression on the 14 column dataset is {ridge_14col_mape}")
print(f"The mean average percentage error of Ridge regression on the combined dataset is {ridge_combined_mape}")
print(f"The R2 score for Ridge regression on the 10 column dataset is {ridge_10col_r2}")
print(f"The R2 score for Ridge regression on the 14 column dataset is {ridge_14col_r2}")
print(f"The R2 score for Ridge regression on the combined dataset is {ridge_combined_r2}")

dump(ridge_10col_model, "./models/ridge_10cols.joblib")
dump(ridge_14col_model, "./models/ridge_14cols.joblib")
dump(ridge_combined_model, "./models/ridge_combined.joblib")


# In[14]:


ridge_10col_error = pd.DataFrame(data={"TestValue": y_test_10col.values})
ridge_10col_error["predictions"] = ridge_10col_predictions
ridge_10col_error["residual"] = ridge_10col_error["TestValue"] - ridge_10col_error["predictions"]
ridge_10col_error["percent_error"] = ridge_10col_error["TestValue"] / ridge_10col_error["residual"]
fig = px.scatter(ridge_10col_error, x="TestValue", y="residual", title="ridge_10col residual vs. observed value")
#fig.show()
fig.write_html("./figures/ridge_10col_residual.html")

ridge_14col_error = pd.DataFrame(data={"TestValue": y_test_14col.values})
ridge_14col_error["predictions"] = ridge_14col_predictions
ridge_14col_error["residual"] = ridge_14col_error["TestValue"] - ridge_14col_error["predictions"]
ridge_14col_error["percent_error"] = ridge_14col_error["TestValue"] / ridge_14col_error["residual"]
fig = px.scatter(ridge_14col_error, x="TestValue", y="residual", title="ridge_14col residual vs. observed value")
#fig.show()
fig.write_html("./figures/ridge_14col_residual.html")

ridge_combined_error = pd.DataFrame(data={"TestValue": y_test_combined.values})
ridge_combined_error["predictions"] = ridge_combined_predictions
ridge_combined_error["residual"] = ridge_combined_error["TestValue"] - ridge_combined_error["predictions"]
ridge_combined_error["percent_error"] = ridge_combined_error["TestValue"] / ridge_combined_error["residual"]
fig = px.scatter(ridge_combined_error, x="TestValue", y="residual", title="ridge_combined residual vs. observed value")
#fig.show()
fig.write_html("./figures/ridge_combined_residual.html")



# ## KNN Model
# A KNN model could capture the stateful nature of the running machine. Each combination of sensor readings is a known operational state of the machine with a label indicating the number of days left until failure. At inference time the input will be used to find the nearest known operational state (by some distance metric) and that will be the prediction.

# In[15]:


# Start KNN fitting and evaluation
knn_params = {"n_neighbors": [2, 5, 10, 50, 100],
              "weights": ["uniform", "distance"],
              "algorithm": ["auto", "ball_tree", "kd_tree"], 
              "leaf_size": [5, 10, 30, 50],
              "metric": ["minkowski", "chebyshev", "cityblock"]}

knn_model = KNeighborsRegressor()
knn_10col_clf = GridSearchCV(knn_model, knn_params)
knn_10col_clf = knn_10col_clf.fit(X_train_10col, y_train_10col)
knn_14col_clf = GridSearchCV(knn_model, knn_params)
knn_14col_clf = knn_14col_clf.fit(X_train_14col, y_train_14col)
knn_combined_clf = GridSearchCV(knn_model, knn_params)
knn_combined_clf = knn_combined_clf.fit(X_train_combined, y_train_combined)


# In[16]:


print(knn_10col_clf.best_params_)
print(knn_14col_clf.best_params_)
print(knn_combined_clf.best_params_)


# In[26]:


# Fit with the best parameters for the 10col dataset and evaluate metrics
knn_10col_model = KNeighborsRegressor(**knn_10col_clf.best_params_)
knn_10col_model.fit(X_train_10col, y_train_10col)
knn_10col_predictions = knn_10col_model.predict(X_test_10col)
knn_10col_mae = mean_absolute_error(y_test_10col, knn_10col_predictions)
knn_10col_mape = mean_absolute_percentage_error(y_test_10col, knn_10col_predictions)
knn_10col_r2 = r2_score(y_test_10col, knn_10col_predictions)

# Fit with the best parameters for the 14col dataset and evaluate metrics
knn_14col_model = KNeighborsRegressor(**knn_14col_clf.best_params_)
knn_14col_model.fit(X_train_14col, y_train_14col)
knn_14col_predictions = knn_14col_model.predict(X_test_14col)
knn_14col_mae = mean_absolute_error(y_test_14col, knn_14col_predictions)
knn_14col_mape = mean_absolute_percentage_error(y_test_14col, knn_14col_predictions)
knn_14col_r2 = r2_score(y_test_14col, knn_14col_predictions)

# Fit with the best parameters for the combined dataset and evaluate metrics
knn_combined_model = KNeighborsRegressor(**knn_combined_clf.best_params_)
knn_combined_model.fit(X_train_combined, y_train_combined)
knn_combined_predictions = knn_combined_model.predict(X_test_combined)
knn_combined_mae = mean_absolute_error(y_test_combined, knn_combined_predictions)
knn_combined_mape = mean_absolute_percentage_error(y_test_combined, knn_combined_predictions)
knn_combined_r2 = r2_score(y_test_combined, knn_combined_predictions)

print(f"The mean average error of KNN on the 10 column dataset is {knn_10col_mae}")
print(f"The mean average error of KNN on the 14 column dataset is {knn_14col_mae}")
print(f"The mean average error of KNN on the combined dataset is {knn_combined_mae}")
print(f"The mean average percentage error of KNN on the 10 column dataset is {knn_10col_mape}")
print(f"The mean average percentage error of KNN on the 14 column dataset is {knn_14col_mape}")
print(f"The mean average percentage error of KNN on the combined dataset is {knn_combined_mape}")
print(f"The R2 score for  KNN on the 10 column dataset is {knn_10col_r2}")
print(f"The R2 score for  KNN on the 14 column dataset is {knn_14col_r2}")
print(f"The R2 score for  KNN on the combined dataset is {knn_combined_r2}")

dump(knn_10col_model, "./models/knn_10cols.joblib")
dump(knn_14col_model, "./models/knn_14cols.joblib")
dump(knn_combined_model, "./models/knn_combined.joblib")


# In[18]:


knn_10col_error = pd.DataFrame(data={"TestValue": y_test_10col.values})
knn_10col_error["predictions"] = knn_10col_predictions
knn_10col_error["residual"] = knn_10col_error["TestValue"] - knn_10col_error["predictions"]
knn_10col_error["percent_error"] = knn_10col_error["TestValue"] / knn_10col_error["residual"]
fig = px.scatter(knn_10col_error, x="TestValue", y="residual", title="knn_10col residual vs. observed value")
#fig.show()
fig.write_html("./figures/knn_10col_residual.html")

knn_14col_error = pd.DataFrame(data={"TestValue": y_test_14col.values})
knn_14col_error["predictions"] = knn_14col_predictions
knn_14col_error["residual"] = knn_14col_error["TestValue"] - knn_14col_error["predictions"]
knn_14col_error["percent_error"] = knn_14col_error["TestValue"] / knn_14col_error["residual"]
fig = px.scatter(knn_14col_error, x="TestValue", y="residual", title="knn_14col residual vs. observed value")
#fig.show()
fig.write_html("./figures/knn_14col_residual.html")

knn_combined_error = pd.DataFrame(data={"TestValue": y_test_combined.values})
knn_combined_error["predictions"] = knn_combined_predictions
knn_combined_error["residual"] = knn_combined_error["TestValue"] - knn_combined_error["predictions"]
knn_combined_error["percent_error"] = knn_combined_error["TestValue"] / knn_combined_error["residual"]
fig = px.scatter(knn_combined_error, x="TestValue", y="residual", title="knn_combined residual vs. observed value")
#fig.show()
fig.write_html("./figures/knn_combined_residual.html")


# ## Gradient Boosted Regression

# In[19]:


# Start Gradient Boosted fitting and evaluation
gb_params = {"max_depth": [2,4,10],
             "min_samples_leaf": [2,5,10],
             "learning_rate": [0.01, 0.05, 0.1],
             "loss": ["squared_error", "absolute_error"]}

gb_model = HistGradientBoostingRegressor()
gb_10col_clf = GridSearchCV(gb_model, gb_params)
gb_10col_clf = gb_10col_clf.fit(X_train_10col, y_train_10col)
gb_14col_clf = GridSearchCV(gb_model, gb_params)
gb_14col_clf = gb_14col_clf.fit(X_train_14col, y_train_14col)
gb_combined_clf = GridSearchCV(gb_model, gb_params)
gb_combined_clf = gb_combined_clf.fit(X_train_combined, y_train_combined)


# In[20]:


print(gb_10col_clf.best_params_)
print(gb_14col_clf.best_params_)
print(gb_combined_clf.best_params_)


# In[21]:


from sklearn.model_selection import TimeSeriesSplit


# In[27]:


# Fit with the best parameters for the 10col dataset and evaluate metrics
gb_10col_model = HistGradientBoostingRegressor(**gb_10col_clf.best_params_)
gb_10col_model.fit(X_train_10col, y_train_10col)
gb_10col_predictions = gb_10col_model.predict(X_test_10col)
gb_10col_mae = mean_absolute_error(y_test_10col, gb_10col_predictions)
gb_10col_mape = mean_absolute_percentage_error(y_test_10col, gb_10col_predictions)
gb_10col_r2 = r2_score(y_test_10col, gb_10col_predictions)

# Fit with the best parameters for the 14col dataset and evaluate metrics
gb_14col_model = HistGradientBoostingRegressor(**gb_14col_clf.best_params_)
gb_14col_model.fit(X_train_14col, y_train_14col)
gb_14col_predictions = gb_14col_model.predict(X_test_14col)
gb_14col_mae = mean_absolute_error(y_test_14col, gb_14col_predictions)
gb_14col_mape = mean_absolute_percentage_error(y_test_14col, gb_14col_predictions)
gb_14col_r2 = r2_score(y_test_14col, gb_14col_predictions)

# Fit with the best parameters for the combined dataset and evaluate metrics
gb_combined_model = HistGradientBoostingRegressor(**gb_combined_clf.best_params_)
gb_combined_model.fit(X_train_combined, y_train_combined)
gb_combined_predictions = gb_combined_model.predict(X_test_combined)
gb_combined_mae = mean_absolute_error(y_test_combined, gb_combined_predictions)
gb_combined_mape = mean_absolute_percentage_error(y_test_combined, gb_combined_predictions)
gb_combined_r2 = r2_score(y_test_combined, gb_combined_predictions)

print(f"The mean average error of Gradient Boosting Regressor on the 10 column dataset is {gb_10col_mae}")
print(f"The mean average error of Gradient Boosting Regressor on the 14 column dataset is {gb_14col_mae}")
print(f"The mean average error of Gradient Boosting Regressor on the combined dataset is {gb_combined_mae}")
print(f"The mean average percentage error of Gradient Boosting Regressor on the 10 column dataset is {gb_10col_mape}")
print(f"The mean average percentage error of Gradient Boosting Regressor on the 14 column dataset is {gb_14col_mape}")
print(f"The mean average percentage error of Gradient Boosting Regressor on the combined dataset is {gb_combined_mape}")
print(f"The R2 score for  Gradient Boosting Regressor on the 10 column dataset is {gb_10col_r2}")
print(f"The R2 score for  Gradient Boosting Regressor on the 14 column dataset is {gb_14col_r2}")
print(f"The R2 score for  Gradient Boosting Regressor on the combined dataset is {gb_combined_r2}")

dump(gb_10col_model, "./models/gb_10cols.joblib")
dump(gb_14col_model, "./models/gb_14cols.joblib")
dump(gb_combined_model, "./models/gb_combined.joblib")


# In[23]:


gb_10col_error = pd.DataFrame(data={"TestValue": y_test_10col.values})
gb_10col_error["predictions"] = gb_10col_predictions
gb_10col_error["residual"] = gb_10col_error["TestValue"] - gb_10col_error["predictions"]
gb_10col_error["percent_error"] = gb_10col_error["TestValue"] / gb_10col_error["residual"]
fig = px.scatter(gb_10col_error, x="TestValue", y="residual", title="gb_10col residual vs. observed value")
#fig.show()
fig.write_html("./figures/gb_10col_residual.html")

gb_14col_error = pd.DataFrame(data={"TestValue": y_test_14col.values})
gb_14col_error["predictions"] = gb_14col_predictions
gb_14col_error["residual"] = gb_14col_error["TestValue"] - gb_14col_error["predictions"]
gb_14col_error["percent_error"] = gb_14col_error["TestValue"] / gb_14col_error["residual"]
fig = px.scatter(gb_14col_error, x="TestValue", y="residual", title="gb_14col residual vs. observed value")
#fig.show()
fig.write_html("./figures/gb_14col_residual.html")

gb_combined_error = pd.DataFrame(data={"TestValue": y_test_combined.values})
gb_combined_error["predictions"] = gb_combined_predictions
gb_combined_error["residual"] = gb_combined_error["TestValue"] - gb_combined_error["predictions"]
gb_combined_error["percent_error"] = gb_combined_error["TestValue"] / gb_combined_error["residual"]
fig = px.scatter(gb_combined_error, x="TestValue", y="residual", title="gb_combined residual vs. observed value")
#fig.show()
fig.write_html("./figures/gb_combined_residual.html")


# 
