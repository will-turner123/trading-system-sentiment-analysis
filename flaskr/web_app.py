from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_core_components as dcc
import dash
import json
import plotly
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from pandas_datareader import data
from datetime import datetime
import logging
import alpaca_trade_api as tradeapi
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

app = Flask(__name__)

key_id = "PKF0OWCNN3UCX2LSBR78"
secret_key = "G8QkmBXCXharATCugZeovzgKyUbPoPVp2cd2ken6"

# get most recent stock data
def get_stock_data(symbol):
    start_date = "2020-01-01"
    end_date = datetime.now()
    stock_data = data.DataReader(symbol, 'yahoo', start_date, end_date)
    return stock_data

def return_sentiment_by_month(company):
    sentiment_df = pd.read_csv(f'../data/{company}_sentiment_analysis.csv', index_col='date', infer_datetime_format=True, parse_dates=True)
    sentiment_df = sentiment_df.drop(['positive','negative','neutral'], axis=1)
    sentiment_df = sentiment_df.resample('M').mean()
    return sentiment_df

def return_news_by_month(company):
    news_df = pd.read_csv(f'../data/{company}_year.csv', index_col='date', infer_datetime_format=True, parse_dates=True)
    news_df = news_df.groupby(news_df.index.month).count()
    return news_df

# linear regression function
def return_linear_regression(symbol):
    file_path = f'../data/{symbol}_sentiment_analysis.csv'
    df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True, index_col='date')
    df = df.resample('D').mean()

    df2 = pd.read_csv(f'data/{symbol}_prices.csv', index_col="t", infer_datetime_format=True, parse_dates=True)

    df = pd.concat([df,df2], axis=1)
    df = df.dropna()

    x = df.drop(["c", "h", "l", "v"], axis=1)
    y = df["c"]

    split = int(0.7 * len(x))
    x_train = x[: split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = y_test.to_frame()
    results["Predicted_results"] = predictions

    mse = mean_squared_error(results['c'], results['Predicted_results'])
    r2 = r2_score(results['c'], results['Predicted_results'])

    return results, mse, r2

def up_or_down(value):
    if float(value) >= 0:
        return "<i class='fas fa-chevron-up' style='color:#66ff5f'></i>"
    else:
        return "<i class='fas fa-chevron-down' style='color:#FF5F66'></i>"


def get_news(from_date):
    # modified get news function which returns dataframe with title, 
    analyzer = SentimentIntensityAnalyzer()
    news_key = "5e3ba89b67f849e6bab8e1ec1f9d8d8e"
    company_news = requests.get(f"http://newsapi.org/v2/everything?q=(pfizer)OR(moderna)&from={from_date}&to={from_date}&language=en&sortBy=publishedAt&apiKey={news_key}").json()
    company_news = pd.DataFrame(data=company_news)
    company_df = pd.DataFrame.from_dict(company_news["articles"])
    company_sentiments = []

    for article in company_news["articles"]:
        try:
            text = article["content"]
            title = article["title"]
            url = article["url"]
            source_name = article["source"]["name"]
            date = article["publishedAt"][:10]
            sentiment = analyzer.polarity_scores(text)
            compound = sentiment["compound"]
            pos = sentiment["pos"]
            neu = sentiment["neu"]
            neg = sentiment["neg"]
            
            company_sentiments.append({
                "text": text,
                "date": date,
                "title": title,
                "link": url,
                "source_name": source_name,
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
    cols = ["date", "text", "compound", "positive", "negative", "neutral", "title", "link", "source_name"]
    company_df = company_df[cols]
    company_df = company_df.drop(["text"], axis=1)
    company_df['date'] = pd.to_datetime(company_df['date'])
    return company_df

def return_smile(sentiment):
    sentiment = float(sentiment)
    if sentiment == 0.0:
        sentiment_text = f'<h6 style="color:#fff85f" data-placement="top" title="{str(sentiment)}"><i class="far fa-meh"></i></h6>'
    elif sentiment > 0:
        sentiment_text = f'<h6 style="color:#66ff5f" data-placement="top" title="{str(sentiment)}"><i class="far fa-smile"></i></h6>'
    elif sentiment < 0:
        sentiment_text = f'<h6 style="color:#FF5F66" data-placement="top" title="{str(sentiment)}"><i class="far fa-frown"></i></h6>'
    return sentiment_text
    



@app.route('/trade')
def trade():
    api = tradeapi.REST(
        base_url = 'https://paper-api.alpaca.markets',
        key_id = key_id,
        secret_key = secret_key
    )
    portfolio_history = api.get_portfolio_history(period='30D').df
    
    equity = portfolio_history.equity[-1]
    profit_loss = up_or_down(portfolio_history.profit_loss[-1]) + " $" + str(portfolio_history.profit_loss[-1])
    profit_loss_pct = round(float(portfolio_history.profit_loss_pct[-1]) * 100, 2)
    
    account = api.get_account()
    cash = account.cash
    # profit loss icon logic cuz im too lazy to use javascript
    if profit_loss_pct >= 0:
        profit_loss_pct_span = f'<span style="color:#66ff5f"><i class="fas fa-chevron-up"></i></span> {profit_loss_pct}%'
    else:
        profit_loss_pct_span = f'<span style="color:#FF5F66"><i class="fas fa-chevron-down"></i></span> {profit_loss_pct}%'
      
    # this is probably a very bad way of doing this
    # TODO: clean up this spaghetti
    orders = api.list_orders()
    order_dict = {}
    for order in orders:
        order_dict[order.id] = {'submitted_at': order.created_at, 'type': order.type, 
        'symbol': order.symbol, 'status': order.status, 'side': order.side, 'shares': order.qty}
    row_list = []
    for id in order_dict: # stock order shares status
        order_dict[id]['submitted_at'] = pd.to_datetime(order_dict[id]['submitted_at']).strftime('%Y-%m-%d %H:%M')
        row = f'<tr><th scope="row">{order_dict[id]["submitted_at"]}</th><td>{order_dict[id]["symbol"]}</td><td>{order_dict[id]["type"]}</td><td>{order_dict[id]["shares"]}</td><td>{order_dict[id]["status"]}</td><tr>'
        row_list.append(row)
    order_table = ' '.join(row_list)

    positions = api.list_positions()
    positions_dict = {}
    position_counter = 0
    for position in positions:
        positions_dict[position_counter] = {'symbol': position.symbol, 'shares': position.qty, 'unrealized_pl': position.unrealized_pl, 
        'entry_price': position.avg_entry_price, 'current_price': position.current_price, 'change_today': round(float(position.change_today) * 100, 2), 'market_value': position.market_value}
        position_counter += 1
    positions_row_list = []
    for id in positions_dict:
        row = f'<tr><th scope="row">{positions_dict[id]["symbol"]}</th><td>{positions_dict[id]["shares"]}</td>'
        row += f'<td>{up_or_down(positions_dict[id]["unrealized_pl"])} ${positions_dict[id]["unrealized_pl"]}</td>'
        row += f'<td>${positions_dict[id]["entry_price"]}</td><td>${positions_dict[id]["current_price"]}</td>'
        row += f'<td>{up_or_down(positions_dict[id]["change_today"])} {positions_dict[id]["change_today"]}%</td>'
        row += f'<td>${positions_dict[id]["market_value"]}</td>'
        row += f'</tr>'
        positions_row_list.append(row)

    # news table stuff
    news = get_news("2021-01-11")
    news_rows = [] # date title symbol src sentiment
    row = ""
    for index, news_row in news.iterrows():
        row = f'<tr><th scope="row">{pd.to_datetime(news_row.date).strftime("%Y-%m-%d")}</th>'
        row += f'<td><a target="_blank" style="color:white;text-decoration:underline;" href="{news_row.link}">{news_row.title}</a></td>'
        row += f'<td class="text-center"{return_smile(news_row.compound)}</td>'
        row += f'<td>{news_row.source_name}</td>'
        news_rows.append(row)
    sentiment_table_body = ' '.join(news_rows)
    position_table_body = ' '.join(positions_row_list)
    plot_bg = "#121729"
    plot_primary = "#2D92FE"
    plot_secondary = "#FF5F66"
    primary_text = "#FFFFFF"
    secondary_text = "#4d8af0"
    graphs = [
        dict(
            data=[
                dict(
                    x=portfolio_history.index,
                    y=portfolio_history['equity'],
                    type='scatter',
                    line=dict(color=plot_primary, width=1),
                    name=f'Equity'
                ),
            ],
            layout=dict(
                title='Equity',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=primary_text)
            ),
            config=dict(responsive=True)
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('trade.html',
                            graphJSON=graphJSON,
                            ids=ids,
                            equity=equity,
                            profit_loss=profit_loss,
                            profit_loss_pct=profit_loss_pct,
                            profit_loss_pct_span=profit_loss_pct_span,
                            cash=cash,
                            order_table_body=order_table,
                            position_table_body=position_table_body,
                            sentiment_table_body=sentiment_table_body)


@app.route('/')
def index():
    pfizer_results, pfizer_mse, pfizer_r2 = return_linear_regression('pfizer')
    moderna_results, moderna_mse, moderna_r2 = return_linear_regression('moderna')
    pfizer_stock = get_stock_data("PFE")
#    pfizer_ohlc = [pfizer_stock.Open.tolist(), pfizer_stock.High.tolist(), pfizer_stock.Low.tolist(), pfizer_stock.Close.tolist()]
    moderna_stock = get_stock_data("MRNA")
    
    pfizer_monthly_sentiment = return_sentiment_by_month('pfizer')
    moderna_monthly_sentiment = return_sentiment_by_month('moderna')

    pfizer_monthly_news = return_news_by_month('pfizer')
    moderna_monthly_news = return_news_by_month('moderna')


    # plot styling
    plot_bg = "#121729"
    plot_primary = "#2D92FE"
    plot_secondary = "#FF5F66"
    primary_text = "#FFFFFF"
    secondary_text = "#4d8af0"
    graphs = [
        dict(
            data=[
                dict(
                    x=pfizer_stock.index,
                    y=pfizer_stock.Close,
                    type='scatter',
                    line=dict(color=plot_primary, width=1),
                    name=f'Closing Price'
                ),
            ],
            layout=dict(
                title='Pfizer Stock Price',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=primary_text)
            ),
            config=dict(responsive=True)
        ),

        dict(
            data=[
                dict(
                    x=pfizer_monthly_news.index,
                    y=pfizer_monthly_news.text,
                    type='bar'
                ),
            ],
            layout=dict(
                title='Pfizer News Per Month',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=primary_text)
            )
        ),

        dict(
            data=[
                dict(
                    x=pfizer_monthly_sentiment.index,  # Can use the pandas data structures directly
                    y=pfizer_monthly_sentiment.compound
                )
            ],
            layout=dict(
                title='Sentiment Over Time',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=primary_text)
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base.html',
                           ids=ids,
                           graphJSON=graphJSON, 
                           pfizer_linear_x=pfizer_results.index.astype(str).tolist(),
                           moderna_stock_x=moderna_stock.index.astype(str).tolist(),
                           moderna_stock_y=moderna_stock['Close'].tolist(), 
                           pfizer_linear_y1=pfizer_results.c.tolist(),
                           pfizer_linear_y2=pfizer_results.Predicted_results.tolist(),
                           pfizer_mse=pfizer_mse,
                           pfizer_r2=pfizer_r2,
                           moderna_linear_x=moderna_results.index.astype(str).tolist(),
                           moderna_linear_y1=moderna_results.c.tolist(),
                           moderna_linear_y2=moderna_results.Predicted_results.tolist(),
                           moderna_sentiment_x=moderna_monthly_sentiment.index.astype(str).tolist(),
                           moderna_sentiment_y=moderna_monthly_sentiment.compound.tolist(),
                           moderna_news_x=moderna_monthly_news.index.astype(str).tolist(),
                           moderna_news_y=moderna_monthly_news.text.tolist(),
                           moderna_mse=moderna_mse,
                           moderna_r2=moderna_r2
                           ) 




if __name__ == '__main__':
    app.run(debug=True)