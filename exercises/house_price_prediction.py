import math

import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import mean_square_error
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from utils import *



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    data = data.drop(data[(data.price < 0) | (data.yr_built < 1)].index)    # validation of condition
    data.dropna(inplace=True)
    price = data["price"] # split y vector from matrix
    data = pd.concat([data, pd.get_dummies(data.zipcode, drop_first=True)], axis=1) # translate categorical variable
    return data.drop(columns=["price", "zipcode", "date", "id"]), price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    feature = X['grade']
    p = (np.cov(feature, y)[0][1] / (np.std(feature, ddof=1) * np.std(y, ddof=1)))

    fig = px.scatter(x=feature, y=y,
                     title=f"feature with good correlation: {p}",
                     labels=dict(x="grade the house has", y="price of the house"))
    fig.show()

    feature = X['yr_built']
    p2 = (np.cov(feature, y) / (np.std(feature, ddof=1) * np.std(y, ddof=1)))[0][1]
    fig2 = px.scatter(x=feature, y=y,
                     title=f"feature with bad correlation: {p2}",
                     labels=dict(x="what year house was built at", y="price of the house"))
    fig2.show()


    pio.write_image(fig, output_path + '/' + "grade.png", format='png')
    pio.write_image(fig2, output_path + '/' + "yr_built.png", format='png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, y_true = load_data("../datasets/house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, y_true, output_path="../")

    # Question 3 - Split samples into training- and testing sets.
    [X_train, y_train, X_test, y_test] = split_train_test(df, y_true)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    LR = LinearRegression()
    p = np.linspace(10, 100, 91)/100
    mse = np.ones(91)
    std = np.ones(91)

    for i, p_ in enumerate(p):
        mse_p = np.ones(10)
        for j in range(10):
            p_sample_X = X_train.sample(frac=p_)
            LR.fit(p_sample_X.to_numpy(), y_train[p_sample_X.index].to_numpy())

            y_pred = LR.predict(X_test.to_numpy())
            mse_p[j] = mean_square_error(y_test.to_numpy(),y_pred.flatten())
        mse[i] = mse_p.mean()
        std[i] = mse_p.std(ddof=1)
    fig = px.scatter(x=p, y=mse,
                     title="Loss of model on increasing percentages of training data",
                     labels=dict(x="p% of training data", y="loss"))
    fig.add_scatter(x=p, y=mse+2*std)
    fig.add_scatter(x=p, y=mse-2*std)
    fig.show()

