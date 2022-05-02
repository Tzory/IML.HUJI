from IMLearn.learners.classifiers import *
import numpy as np
from typing import Tuple
import plotly.io as pio
pio.templates.default = "simple_white"
from IMLearn.metrics import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi



def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
        ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
        Parameters
        ----------
        filename: str
            Path to .npy data file
        Returns
        -------
        X: ndarray of shape (n_samples, 2)
            Design matrix to be used
        y: ndarray of shape (n_samples,)
            Class vector specifying for each sample its class
        """
        data = np.load(filename)
        return data[:, :2], data[:, 2].astype(int)



def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """


    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(p : Perceptron):
            loss = p._loss(X, y)
            losses.append(loss)

        model = Perceptron(callback=callback)
        model.fit(X,y)

        # Plot figure
        fig1 = go.Figure(data=go.Scatter(x=np.linspace(0, len(losses), len(losses)),
                                         y=losses, mode='lines'))
        fig1.update_layout(xaxis_title="iteration number",
                    yaxis_title="errors rate",
                    title= f'loss as a function of the iterations in {n} data')

        fig1.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set

        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_pred = gnb.predict(X)



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        fig1 = make_subplots(rows=1, cols=2, subplot_titles=
        ((f + f'Gaussian Naive Bayes, accuracy: {accuracy(y, gnb_pred)}'),
         (f + f'LDA prediction, accuracy: {accuracy(y, lda_pred)}')))
        fig1.add_trace(go.Scatter(
            x=X.T[0], y=X.T[1], mode='markers',
            marker=go.scatter.Marker(color=lda_pred, symbol=y)),row=1, col=2)


        fig1.add_trace(go.Scatter(
            x=X.T[0], y=X.T[1], mode='markers',
            marker=go.scatter.Marker(color=gnb_pred, symbol=y)),row=1, col=1)

        fig1.add_trace(go.Scatter(
            x=lda.mu_.T[0], y=lda.mu_.T[1], mode='markers',
            marker=go.scatter.Marker(color='blue', symbol='x', size=15)),row=1, col=2)

        fig1.add_trace(go.Scatter(
            x=gnb.mu_.T[0], y=gnb.mu_.T[1], mode='markers',
            marker=go.scatter.Marker(color='blue', symbol='x', size=15)),row=1, col=1)

        colors = ['green', 'red', 'blue']

        for i in range(3):
            fig1.add_trace(go.Scatter(x=[1], y=[1], mode='markers', visible="legendonly",
                                      name=f'prediction of label {i}',
                                      marker=dict(color=colors[i])))


        fig1.add_trace(go.Scatter(x=[1], y=[1], mode='markers', visible="legendonly",
                                  name='mean of gaussian',
                                  marker=dict(color='black',symbol='x', size=15)))


        for i, j in enumerate(lda.classes_):
            fig1.add_trace(go.Scatter(get_ellipse(lda.mu_[i], lda.cov_)), row=1, col=2)
            fig1.add_trace(go.Scatter(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))), row=1, col=1)



        fig1.show()



if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
