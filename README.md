# Trading System and Sentiment Analysis
Mehnoosh Askari, Chris Bryant, Raniah Jeanlys, Robert Leonhardt, Will Turner
# Methodology
Team decided to create an automated paper trading system using sentiment analysis for two biopharmaceutical companies ( Pfizer and Moderna). The objective is to prove based on chosen models the stock prices and sentiment are correlated. In order, to prove our hypothesis we used  classification (Linear Discriminate) and linear regression model . For linear discriminate, we wanted the model to predict whether the current days close price was going to go up or down based on the previous days news sentiment and stock information. The features: for this model are as follow: stock prices (open, high, low, volume), sentiment (compound, positive, negative, and neutral). Also target : encoded on the close column. 
For linear regression model, We used a years worth of data to train our model. We wanted the model to predict the next days close price and see if there was correlation between sentiment and stock prices. We used the same futures as other model, but the target for this model is close column. 
**All datasets were used from various sources:**
- Data Source: Google News API
- Live Data Source: News API
- Stock Pirces: Alpaca Trade API
- Sentiment Analysis: NLTK (Vader Sentiment)
**Libraries:**
- Flask
- Plotly
- Bootstrap
## Final Analysis
Our primary goal was to create a model to make a predictions for stock price(Pfizer and Moderna). We used trained model on live data and news sentiments and stock price to make the predications. We were successful to create that program to achieve our goal. 

View a live demo at [http://trading-sentiment.ml](http://trading-sentiment.ml)