import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import nltk
import requests
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
from datetime import timedelta
import time
from model import train_model, make_predictions
import warnings
import json
import dotenv
import os

# TODO: Sell all shares of security in sell function
# Make it so it doesn't loop orders

warnings.filterwarnings('ignore')

analyzer = SentimentIntensityAnalyzer()

key_id = os.getenv("alpaca_key")
secret_key = os.getenv("alpaca_secret_key")

api = tradeapi.REST(
    base_url = 'https://paper-api.alpaca.markets',
    key_id = key_id,
    secret_key = secret_key
)

# returns sentiment of news
def get_news(company, from_date):
    news_key = os.getenv("news_api")
    company_news = requests.get(f"http://newsapi.org/v2/everything?q={company}&from={from_date}&to={from_date}&language=en&sortBy=publishedAt&apiKey={news_key}").json()
    company_news = pd.DataFrame(data=company_news)
    company_df = pd.DataFrame.from_dict(company_news["articles"])
    company_sentiments = []

    for article in company_news["articles"]:
        try:
            text = article["content"]
            date = article["publishedAt"][:10]
            sentiment = analyzer.polarity_scores(text)
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]
            
            company_sentiments.append({
                "text": text,
                "date": date,
                "compound": compound,
                "positive": pos,
                "negative": neg,
                "neutral": neu
                
            })
            
        except AttributeError:
            pass
        
    # Create DataFrame
    company_df = pd.DataFrame(company_sentiments)

    # Reorder DataFrame columns
    cols = ["date", "text", "compound", "positive", "negative", "neutral"]
    company_df = company_df[cols]
    company_df = company_df.drop(["text"], axis=1)
    company_df['date'] = pd.to_datetime(company_df['date'])
    return company_df


# returns models in a list
# one for pfizer, one for moderna
def get_models(companies):
    model_list = []
    for company in companies:
        model_list.append(train_model(company))
    return model_list

# passes yesterdays news and stock price as test data to make_predictions() to generate signal
def generate_signal(news, bars, model):
    bars = bars.dropna()
    bars.columns = ['t', 'o', 'h', 'l', 'c', 'v']
    # encodes close as price either going up or down
    bars["c"][bars["c"] < 0] = 0
    bars["c"][bars["c"] > 0] = 1

    # get mean of all news articles for the day
    news.set_index('date', inplace=True)
    news = news.resample('D').mean()
    # if day is monday or sunday
    # sets stock price index to be yesterday's date
    # so that bars and news can be concatenated
    if datetime.datetime.today().weekday() in [0, 6]: # 0 is monday sunday is 6
        bars.iloc[0,0] = (datetime.datetime.now()  - datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
        bars.reset_index(inplace=True)
        bars['t'] = pd.to_datetime(bars['t']).dt.date
        bars.drop(['index'], axis=1, inplace=True)
    
    # concatenates bars & news and passes to make_predictions for signal
    bars.set_index('t', inplace=True)
    df = pd.concat([news,bars], axis=1)
    df = df.dropna()
    df = df.drop(['c'], axis=1)
    predictions = make_predictions(model, df)
    print(predictions)
    return predictions


def submit_order(symbol):
    api.submit_order(
        symbol=symbol,
        qty=100,
        side='buy',
        type='market',
        time_in_force='gtc',
    )

# sells all shares of a stock
def submit_sell(symbol):
    # gets amount of shares owned of a symbol & places sell order
    order = api.get_position(symbol=symbol)
    qty = order.qty
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='day'
    )

# checks if we already have a position for the stock
def check_positions(symbol):
    try:
        position = api.get_position(symbol)
    except Exception as e:
        print(f'Error {e}')
        return "No position"
    return position


while True:
    symbols = ["PFE", "MRNA"]
    companies = ["pfizer", "moderna"]
    # returns models as list (index same as companies)
    models = get_models(companies)

    # only operates in market hours
    timestamp_now = datetime.datetime.now()
    date_now = timestamp_now.date()
    time_now = timestamp_now.time()
    yesterday = datetime.datetime.now() - datetime.timedelta(days = 1)
    yesterday = yesterday.strftime('%Y-%m-%d')
    if time_now < datetime.time(3,59,00):
        time.sleep(5)
        continue
    if time_now > datetime.time(23,00,00):
        break
    time.sleep(15)
    index_counter = 0
    for symbol in symbols:
        print(f'*' * 10)
        # gets news from yesterday
        news = get_news(companies[index_counter], str(yesterday))
        model = models[index_counter]

        # gets stock data on current company
        bars = api.get_barset(symbols=[symbol], timeframe='1D', limit=5)[symbol]._raw
        bars = pd.DataFrame(data=bars)
        bars["t"] = pd.to_datetime(bars['t'], unit='s').dt.date

        signal = generate_signal(news, bars, model)
        position = check_positions(symbol)
        # if we have signal of 1, we buy 100 shares of stock
        # otherwise we sell all our shares of that stock
        if signal == 1.0 and position == "No position":
            submit_order(symbol)
            print(f'Placing order to buy 100 shares of {symbol}')
        elif signal == 0.0 and not position == "No position":
            submit_sell(symbol)
            print(f'Placing sell order to sell 100 shares of {symbol}')

        index_counter += 1
