import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

import emcee
import corner

if __name__ == "__main__":
    fname = "./age_metallicity.txt"
    fname2 = "./star_age.txt"

    # Read data from tables
    df = pd.read_table(fname, delim_whitespace=True)
    df2 = pd.read_table(fname2, delim_whitespace=True, comment="#")
    table = pd.merge(df, df2, on="star")

    # Select x and y data
    y = table["age"]
    yerr = table["sigma_age"]

    names = ["[C/Fe]", "[Y/Fe]", "[Mg/Fe]"]
    ndim = len(names)
    nstars = table.shape[0]

    A = np.zeros((nstars, ndim + 1))
    A[:, 0] = 1  # intercept
    for i in range(1, ndim + 1):
        A[:, i] = table[names[i]].values  # linear fit in n dimensions

    # Fit linear regression
    # TODO: Use Gaussian Process instead of Linear Fit?
    res = least_squares(lambda x: (A.dot(x) - y) / yerr, x0=np.ones(nstars))
    coef = res.x
    cov = np.linalg.pinv(res.jac.T.dot(res.jac))  # covariance = inverse(Hessian)

    age = lambda met: coef[0] + coef[1:].dot(met)

    # Fit line
    # coef, cov = np.polynomial.polynomial.polyfit(x, y, deg=1, w=1 / yerr)
    # print(coef)

    # def model(theta, x):
    #     return np.polyval(theta, x)

    # def lnprior(theta):
    #     a, b = theta
    #     return 0

    # def lnpost(theta, x, y, yerr):
    #     prior = lnprior(theta)
    #     like = -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)
    #     return prior + like

    # nwalkers, ndim = 10, 2
    # p0 = [coef + np.random.randn(ndim) for i in range(nwalkers)]
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(x, y, yerr))
    # sampler.run_mcmc(p0, 1000)
    # chain = sampler.flatchain
    # coef2 = np.percentile(chain, 50, axis=0)
    # cov2 = np.percentile(chain, (16, 84), axis=0)
    # cov2 = np.diff(cov2, axis=0) ** 2

    # corner.corner(chain)
    # plt.show()

    # Plot
    plt.errorbar(A[:, 1], y, yerr=yerr, fmt=".")
    plt.plot(A[:, 1], age(A[:, 1:]))
    plt.xlabel(names[1 - 1])
    plt.ylabel("Age [Gyr]")
    plt.show()
