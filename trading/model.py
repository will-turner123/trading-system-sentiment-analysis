from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


def train_model(company):
    df = pd.read_csv(f'data/{company}_sentiment_analysis.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
    df = df.resample('D').mean()
    df2 = pd.read_csv(f'data/{company}_prices.csv', index_col="t", infer_datetime_format=True, parse_dates=True)
    df2["c"] = df2["c"].pct_change()
    df2 = df2.dropna()
    df2["c"][df2["c"] < 0] = 0
    df2["c"][df2["c"] > 0] = 1
    df = pd.concat([df,df2], axis =1)
    df = df.dropna()

    X = df.drop(["c"], axis=1)
    y = df["c"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 50)
    model = LinearDiscriminantAnalysis().fit(x_train, y_train)
    return model


def make_predictions(model, x_test):
    predictions = model.predict(x_test)
    return predictions[-1]