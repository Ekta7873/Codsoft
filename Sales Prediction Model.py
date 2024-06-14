#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


# # Read csv file
# 

# In[18]:


df = pd.read_csv("C:/Users/lenovo/Downloads/advertising.csv")
df.head(6)


# In[19]:


df.info()


# In[20]:


df.describe()


# In[21]:


df.isnull()


# # EDA(Exploratory Data Analysis)

# In[22]:


df.corr()


# In[23]:


columns = ["TV", "Radio", "Newspaper"]
labels = ["TV", "Radio", "Newspaper"]

for col, label in zip(columns, labels):
    f, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(df[col], df["Sales"])
    ax.set_xlabel(label)
    ax.set_ylabel("Sales")
    ax.set_title(f"Sales vs {label}")

plt.show()


# # Data Modeling

# In[24]:


x = df.drop("Sales", axis=1)
y = df["Sales"]

print("== Predictors (x) ==")
print(f"Size: {x.shape}")
print(x.head())
print(f"Data Type: {type(x.head())}")

print("\n== Target (y) ==")
print(f"Size: {y.shape}")
print(y.head())
print(f"Data Type: {type(y.head())}")


# In[25]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")


# In[26]:


results_columns = ["Predictor/s", "R2", "MAE", "MSE", "RMSE", "Cross-Val Mean"]
df_results = pd.DataFrame(columns=results_columns)


# In[27]:


def linear_regression_model(x_train, x_test):
    model = LinearRegression()
    
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    
    plt.figure(figsize=(11, 9))
    plt.scatter(predictions, y_test, alpha=0.7)
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Actual vs Predicted Values")
    plt.show()
    
    return {"R2": r2 * 100, "MAE": mae, "MSE": mse, "RMSE": rmse}


# # Linear regression model

# In[28]:


linreg_results = linear_regression_model(x_train, x_test)

cross_val_scores = cross_val_score(LinearRegression(), x, y, cv=10)

print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Cross-Validation Mean Score: {cross_val_scores.mean()}")

linreg_results.update({"Predictor/s": "All", "Cross-Val Mean": cross_val_scores.mean() * 100})

df_results = pd.concat([df_results, pd.DataFrame(linreg_results, index=[0])], ignore_index=True)


# In[29]:


tv_predictors = x_train["TV"].values.reshape(-1, 1)
tv_test_data = x_test["TV"].values.reshape(-1, 1)
tv_results = linear_regression_model(tv_predictors, tv_test_data)

cv_scores_tv = cross_val_score(LinearRegression(), x["TV"].values.reshape(-1, 1), y, cv=10)

print(f"Cross-Validation Scores for TV: {cv_scores_tv}")
print(f"Cross-Validation Mean Score for TV: {cv_scores_tv.mean()}")

tv_results.update({"Predictor/s": "TV", "Cross-Val Mean": cv_scores_tv.mean() * 100})

df_results = pd.concat([df_results, pd.DataFrame(tv_results, index=[0])], ignore_index=True)


# In[30]:


tv_radio_predictors_train = x_train[["TV", "Radio"]]
tv_radio_predictors_test = x_test[["TV", "Radio"]]
tv_radio_results = linear_regression_model(tv_radio_predictors_train, tv_radio_predictors_test)

cv_scores_tv_radio = cross_val_score(LinearRegression(), x[["TV", "Radio"]], y, cv=10)

print(f"Cross-Validation Scores for TV & Radio: {cv_scores_tv_radio}")
print(f"Cross-Validation Mean Score for TV & Radio: {cv_scores_tv_radio.mean()}")

tv_radio_results.update({"Predictor/s": "TV & Radio", "Cross-Val Mean": cv_scores_tv_radio.mean() * 100})

df_results = pd.concat([df_results, pd.DataFrame(tv_radio_results, index=[0])], ignore_index=True)


# In[31]:


print(df_results.columns)


# # Final Results

# In[32]:


df_results.set_index("Predictor/s", inplace = True)
df_results.head()


# In[ ]:





# In[ ]:




