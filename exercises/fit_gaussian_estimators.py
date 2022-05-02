from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as ex

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1

    samples = np.random.normal(mu, var, 1000)
    A = UnivariateGaussian()

    print("(", A.fit(samples).mu_, ",", A.fit(samples).var_, ")")

    # # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    for m in ms:
        A.fit(samples[:m])
        estimated_mean.append(abs(A.mu_ - mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines',
                          name=r'$\widehat\mu$')], layout=go.Layout(
        title=r"$\text{Absolute distance between the estimated- and true value of the expectation}$",
        xaxis_title="$m\\text{ - number of samples}$",
        yaxis_title="r$\hat\mu$",
        height=400)).show()

    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=samples, y=A.pdf(samples), mode='markers',
                          name=r'$\widehat\mu$')], layout=go.Layout
    (title=r"$\text{Samples drom normal distribution and their PDF}$",
     xaxis_title="samples values",
     yaxis_title="pdf",
     height=300)).show()


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=1000)
    B = MultivariateGaussian()
    B.fit(samples)

    print(B.mu_, "\n", B.cov_)

    # Question 5 - Likelihood evaluation
    ret = np.ones((200, 200))
    f1 = f3 = x = np.linspace(-10, 10, 200)
    for i in range(200):
        for j in range(200):
            mu_new = np.array([f1[i],0,f3[j],0])
            ret[i][j] = B.log_likelihood(mu_new, sigma, samples)
    fig = ex.imshow(ret,x=x, y=x, title="log likelihood of mu = [f1,0,f3,0]",
                    labels=dict(x="f3", y="f1", color="log likelihood"),
                                color_continuous_scale = "tempo")
    fig.show()



    # Question 6 - Maximum likelihood
    vmax = ret.max()
    f1v = round(f1[np.where(ret == vmax)[0][0]], 3)
    f3v = round(f3[np.where(ret == vmax)[1][0]], 3)
    print("f1 = ", f1v, ", f3 = ", f3v)



if __name__ == '__main__':

    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

