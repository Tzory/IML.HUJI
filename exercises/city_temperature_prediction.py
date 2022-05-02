import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
import datetime as dt
from IMLearn.utils import split_train_test


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    file = pd.read_csv(filename, parse_dates=["Date"])
    data = pd.DataFrame(file)
    data = data.reset_index()
    data = data.drop(data[(data.Temp < -70)].index)
    data.dropna(inplace=True)
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == 'Israel']

    avg_temp = np.ones(365)
    for i in range(365):
        df_days = df_israel.loc[df_israel['DayOfYear'] == i]
        avg_temp[i] = df_days['Temp'].mean()

    fig = px.scatter(x=df_israel['DayOfYear'], y=df_israel['Temp'],
                     title="average daily temperature",
                     labels=dict(x="Day of year", y="Temp"),
                     color=df_israel['Year'].astype(str))
    fig.show()

    fig2 = px.bar(df_israel.groupby('Month')['Temp'].agg(['std']))
    fig2.show()

    # # Question 3 - Exploring differences between countries
    temp = (df.groupby(['Month', 'Country'])['Temp'].agg(['std', 'mean']))
    temp = temp.reset_index()
    fig3 = px.line(x=temp['Month'], y=temp['mean'], color=temp['Country'],
                   error_y=temp["std"],
                   title="Average monthly temperature",
                   labels=dict(x="Month", y="mean temperature"))
    fig3.show()

    # # Question 4 - Fitting model for different values of `k`
    y_is = df_israel['Temp']
    X_is = df_israel.drop(columns=["Temp"])
    loss = np.ones(10)
    [X_train, y_train, X_test, y_test] = split_train_test(X_is, y_is)

    k = range(1, 11)
    for i in range(10):
        Poly = PolynomialFitting(i + 1)
        Poly.fit(X_train['DayOfYear'], y_train)
        loss[i] = round(Poly.loss(X_test['DayOfYear'], y_test), 2)
        print("k =", i, ": ", loss[i])

    fig4 = px.bar(x=k, y=loss, title='loss of model k degree polynomial',
                  labels=dict(x="k", y="loss"))
    fig4.show()


    # # Question 5 - Evaluating fitted model on different countries
    Poly = PolynomialFitting(k=5)
    Poly.fit(X_is['DayOfYear'], y_is)

    df_SA = df[df['Country'] == 'South Africa']
    df_JDN = df[df['Country'] == 'Jordan']
    df_NTL = df[df['Country'] == 'The Netherlands']
    loss_by_country = np.ones(4)
    loss_by_country[0] = Poly.loss(df_SA['DayOfYear'], df_SA['Temp'])
    loss_by_country[1] = Poly.loss(df_JDN['DayOfYear'], df_JDN['Temp'])
    loss_by_country[2] = Poly.loss(df_NTL['DayOfYear'], df_NTL['Temp'])
    loss_by_country[3] = Poly.loss(X_is['DayOfYear'], y_is)

    fig5 = px.bar(x=['South Africa', 'Jordan', 'The Netherlands' ,'Israel'],
                  y=loss_by_country,
                    title='loss of the model on other countries',
                    labels=dict(x="countries", y="loss"))
    fig5.show()


