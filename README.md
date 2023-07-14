# Stock Prediction using RandomForestClassifier Model

## AIM
Predicting Directionality of Tomorrow's Closing Price for Any Stock through Historical Data (RandomForestClassifier Model)

## Data
Data can be customized by the user by running `search_stocks()` function and providing the stock symbol. For eg: AMZN for Amazon. The data is sourced from yfinance library.

### Evaluation
**Trained on the following models:**
 - k-nearest-neighbors
 - RandomForestClassifier
 - LogisticRegression

**Model**
 - RandomForestClassifier was used to predict whether the stock price will rise(1) or not (0). 

At or above 80% accuracy we consider our model as successful. 
For evaluation check: credit-card-default-payment-prediction.ipynb 

### Data Features
This dataset used was scraped directly from yfinance. After training and testing the model it is our clear we need more features in our dataset. The features used to train our sotkc price data:

<li>Volume: Total Trading volume of stocks<br>
<li>Open: Stocks opening price for that specific day. <br>
<li>Close: Stocks closing price for that specific day. <br>
<li>Low: The lowest price the stock has hit over the years. <br> 
<li>High: The highest price the stock has reached over the years. <br> 
<li>Target: 0 = closing prices tomorrow is lower than closing price tomorrow; 1 = closing price tomorrow > closing price today <br> 


### Used Libraries and Pre-Defined Metrics/Models
```
!pip install yfinance
import yfinance as yf 
import pandas as pd #Pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
 ```
 ### Conclusion
 
The precision score is quite low. This goes onto show that rfc model does not perform well in predicting the next day's price just from historical data and with our limited data features. To resolve this, we can:
<li>1. Scrape more relevant data that realistically affects the stock prices. (Eg: Stock news)</li>
<li>2. Perform hyperparameter tuning</li>
