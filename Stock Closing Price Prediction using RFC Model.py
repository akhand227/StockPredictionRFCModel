#!/usr/bin/env python
# coding: utf-8

# # Predicting Tomorrow's Closing Price for Any Stock

# ## Aim: To predict tomorrow's stock price using historical data. 

# ### Importing Libraries 

# In[117]:


get_ipython().system('pip install yfinance')
import yfinance as yf # YahooFinance for stock data
import pandas as pd #Pandas
import numpy as np #Numpy
from sklearn.metrics import precision_score #Precision score metrics to compare actual vs predicted values
import matplotlib.pyplot as plt


# ## Data

# ### Insert Stock Symbol to access Stock Data

# In[152]:


def search_stocks():
    symbol=input(str('Please Enter the Stock Symbol')) #Asks user for their Stock Input
    stock_symbol = yf.Ticker(symbol) #Requests the data
    stock_symbol = stock_symbol.history(period="max") #Pulls all the historical data of the stock
    stock_symbol.to_csv("stock_symbol.csv") #converting the retrieved data into csv
    stock_data=pd.read_csv('stock_symbol.csv') 
    return stock_data #returning the data 


# In[203]:


stock_data = search_stock()
stock_data


# ### Setting index as date-time

# In[204]:


stock_data.index = pd.to_datetime(stock_data.Date)
stock_data.head()


# In[205]:


#Visualising Stock Closing Price since 2007
stock_data.plot.line(y=['Close'], use_index= True, figsize=(16,5), xlabel='Year', ylabel='Price')


# ## Preprocessing Data

# In[206]:


#Removing irrelevant columns before Model Training - Dividends + Stock Splits
del stock_data['Dividends']
del stock_data['Stock Splits']
stock_data.head()


# ### Creating Target Column

# Target column = 1 if tomorrow's Closing Price (Tomorrow Column) is greater than today's closing price else 0. (Close Column)

# In[207]:


stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
stock_data


# In[208]:


stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
stock_data.head()


# ## Training the Model

# In[209]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=250, min_samples_split=100, random_state=1)

train = stock_data.iloc[:-820] #80% of data for training approx.
test = stock_data.iloc[-820:] #20% of data for testing approx.

predictors = ["Close", "Volume", "Open", "High", "Low"] #Valid Columns for training rfc model
model.fit(train[predictors], train["Target"])


# ## Making Prediction from the RFC Model

# In[210]:


#Predicting From the Trained Model
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index, name="Predictions")
preds, type(test.index)


# ### Comparing Target (Actual) vs Predictions generated from rfc model

# In[211]:


combined = pd.concat([test["Target"], preds], axis=1)
combined


# ### Bring the above steps altogether

# In[212]:


def generate_predictions(data, model, predictors, start=820, step=150):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy() #splitting data into training data
        test = data.iloc[i:(i+step)].copy() #splitting data inot testing data
        model.fit(train[predictors], train["Target"]) #model fitting
        preds = model.predict(test[predictors]) #making predictions on the test data
        preds = pd.Series(preds, index=test.index, name="Predictions") 
        combined = pd.concat([test["Target"], preds], axis=1) #pd df for target (actual) vs predicted outcome
        all_predictions.append(combined)
    
    return pd.concat(all_predictions) # returns list of target (actual outcome) vs predictions from rfc model


# In[213]:


predictions = generate_predictions(stock_data, model, predictors)
predictions


# ### Comparison for difference in predicted vs actual outcome

# In[214]:


predictions["Predictions"].value_counts(),predictions["Target"].value_counts()


# This shows bad performance.For more clarity, we can conduct precision scoring on this: 

# In[215]:


precision_score(predictions["Target"], predictions["Predictions"])


# ## Conclusion

# The precision score is quite low. This goes onto show the our rfc model does not perform well in predicting next day's price just from historical data. To resolve this, we can: 
# 1. Scrape more relevant data that realistically affects the stock prices. (Eg: Stock news)
# 2. Perform hyperparameter tuning
