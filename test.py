import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, minimize, LinearConstraint
from scipy.stats import levy, skellam, gennorm, norm
from scipy.special import gamma

import astropy.stats as apstats
import corner
import emcee

from joblib import Parallel, delayed
from sklearn.linear_model import Lasso

from src.sme.sme import SME_Struct
from src.sme.abund import Abund

# param = p[0], loc=p[1], scale=p[2]
stat_func = lambda x, *p: p[3] * gennorm.pdf(x, p[0], loc=p[1], scale=p[2])

fname = "./sme.npy"

sme = SME_Struct.load(fname)

jac = sme.pder[sme.mob != 0]
H = jac.T.dot(jac)


def load_sme(sme):
    # Only use pixels that have been used by SME for the fit
    r = (sme.sob - sme.smod)[sme.mob != 0]
    der = sme.pder[sme.mob != 0]
    par = sme.pfree[-1]
    norm = np.abs(np.median(der, axis=0))  # TODO ???
    names = sme.pname
    return r, der, par, norm, names


def toy_problem():
    sigma_stat = 0.01  # Photon Noise, constant in whole range
    sigma_syst = 0.5  # Systematic Line data uncertainty, only in Section 2
    npoints = 10000
    nsection = npoints // 2
    res = np.zeros(npoints)
    der = np.zeros((npoints, 2))

    # Section 1 "Continuum":
    res[:nsection] = np.random.normal(scale=1.01, size=nsection)
    der[:nsection, 0] = 1
    der[:nsection, 1] = 0.01

    # Section 2 "Line":
    res[nsection:] = np.random.normal(scale=1.5, size=nsection)
    der[nsection:, 0] = 1
    der[nsection:, 1] = 0.5

    par = np.array([0, 0])
    norm = np.array([1, 1])
    names = ["A", "B"]
    return res, der, par, norm, names


def sum_is_1(x):
    """ Sum of x - 1, constrain for minimization """
    return np.sum(x) - 1


def random_fraction(n):
    res = np.empty(n)
    k = 0
    for i in range(n):
        res[i] = np.random.rand() * (1 - k)
        k += res[i]
    np.random.shuffle(res)
    return res


def mini(x, A):
    return np.sum((A * x) ** 2)


def jac(x, A):
    return 2 * A ** 2 * x


def hess(x, A):
    return np.diag(2 * A ** 2)


def optionB():
    # tst = [
    #     minimize(
    #         mini,
    #         x0=random_fraction(nparam),
    #         bounds=bounds,
    #         constraints=constraint,
    #         args=(A[0],),
    #         jac=jac,
    #         tol=np.finfo(float).eps,
    #     )
    #     for _ in range(100)
    # ]

    # For each point find the smallest vector of parameter changes
    # while still reproducing the residual (constraint)

    A = r[:, None] / (der * norm)

    # Bound Constrained problem
    # Bounds: x fraction between 0 and 1
    # Constraint: total x == 1
    bounds = [(0, 1) for _ in range(nparam)]
    constraint = {"type": "eq", "fun": sum_is_1}
    alt_constraint = LinearConstraint(np.ones(nparam), 1, 1)

    tst = Parallel(n_jobs=6)(
        delayed(minimize)(
            mini,
            x0=random_fraction(nparam),
            bounds=bounds,
            constraints=constraint,
            args=(A[i],),
            jac=jac,
            tol=np.finfo(float).eps,
        )
        for i in range(npixel)
    )
    tst = [res.x for res in tst]
    tst = np.array(tst)

    np.save("tst.dat", tst)


def optionC():
    # Step 1: How many points in each batch give a good fit?
    # too many points will properly represent the data, resulting in low variation
    # and low scatter/error on the parameters
    # too few data points will overfit the data, creating huge scatter/error
    # Therefore pick the lowest number of points that gives a chisq of at least 1
    # Question: Is there any statistical reasoning for this ad-hoc argument
    # print("NPoints  ChiSq")
    # for i in range(1, 2 * nparam):
    #     nsamples = r.size // nparam
    #     npoints = nparam + i
    #     idx = np.random.choice(r.size, size=(nsamples, npoints), replace=True)
    #     chisq = np.empty(nsamples)
    #     for j in range(nsamples):
    #         res = np.linalg.lstsq(der[idx[j], :], r[idx[j]], rcond=None)
    #         chisq[j] = np.sum(res[1] ** 2) / npoints
    #     print(npoints, np.median(chisq))
    #     if np.median(chisq) > 1:
    #         break

    # Step 2: Pick some number of sections with npoints each
    # Solve the leastsq problem for each batch
    # There will be some outliers, but those will be thrown away down the line
    npoints = nparam
    nsections = r.size  # * npoints
    idx = np.random.choice(r.size, size=(nsections, npoints), replace=True)
    y = r[idx]
    x = der[idx, :]
    tst = np.zeros((nsections, nparam))
    chisq = np.zeros(nsections)

    for i in range(nsections):
        res = np.linalg.lstsq(x[i], y[i], rcond=None)
        chisq[i] = np.sum(res[1] ** 2) / npoints
        tst[i] = res[0]

    print("Final Chisq:", np.median(chisq))
    np.save("tstC.dat", tst)


r, der, par, normal, names = toy_problem()
# r, der, par, normal, names = load_sme(sme)

nparam = len(par)
npixel = len(r)

# Option A: Classic SME, only one parameter is used to explain the full residual
tst = r[:, None] / np.mean(der, axis=0)[None, :]
tst /= np.sum(np.abs(der), axis=1)[:, None]
weights = np.abs(der) / np.sum(np.abs(der), axis=1)[:, None]

# Option B: Residual is the smallest linear combination that explains the residual
# optionB()
# tst = np.load("tst.dat.npy")
# tst *= r[:, None] / der

# Option C: Use 2*nparam (random) points to find solution to residuals, and examine the scatter
# optionC()
# tst = np.load("tstC.dat.npy")

ncol = 2
nrow = nparam // ncol
nrow += 1 if nparam % ncol != 0 else 0
sigma = [0 for _ in range(nparam)]
popts = [0 for _ in range(nparam)]

for i in range(nparam):
    tmp = tst[:, i]
    tmp = np.nan_to_num(tmp)

    # Only use non masked lines
    std = np.std(tmp)
    bounds = np.percentile(tmp, (2.5, 16, 50, 84, 97.5))
    ci = bounds[2] - bounds[1], bounds[3] - bounds[2]
    ci95 = bounds[2] - bounds[0], bounds[4] - bounds[2]

    # Discard outliers
    # tmp = tmp[(tmp <= bounds[4]) & (tmp >= bounds[0])]
    std = np.std(tmp)
    sigma[i] = std

    h, x = apstats.histogram(tmp, bins=1000, range=(bounds[0], bounds[4]))
    h, x = np.histogram(tmp, bins=x, weights=weights[:, i])

    threshold = max(h) / 10
    bounds[0] = x[:-1][h >= threshold][0]
    bounds[-1] = x[1:][h >= threshold][-1]

    h, x = apstats.histogram(tmp, bins=100, range=(bounds[0], bounds[4]))
    h, x = np.histogram(tmp, bins=x, weights=weights[:, i])

    stat_x = x[:-1] + np.diff(x) / 2
    # h = np.clip(h, 1, None)
    # stat_h = np.log10(h)

    step = x[1] - x[0]
    A = np.sum(h) * step
    height = 1.2 * A / (np.sqrt(2 * np.pi) * np.abs(std))

    mu0, mu1 = 0.5 * ci[0] ** 2, 0.5 * ci[1] ** 2
    popt, pcov = curve_fit(
        lambda x, *p: p[0] * norm.pdf(x, *p[1:]), stat_x, h, p0=(max(h), 0, step)
    )
    std = popt[2]
    # popt, pcov = curve_fit(
    #     stat_func, stat_x, stat_h, p0=(0.5, bounds[2], std, bounds[-1] * 2)
    # )
    # popts[i] = popt
    # a = popt[0] * popt[3] / (2 * popt[2] * gamma(1 / popt[0]))
    # var = popt[2] ** 2 / 2
    # sigma = np.sqrt(var)

    print(names[i])
    print("68 %% C.I. %.3g , %.3g" % ci)
    print("95 %% C.I. %.3g , %.3g" % ci95)
    print("Std %.3g" % std)

    plt.subplot("%i%i%i" % (nrow, ncol, i + 1))
    a = plt.hist(tmp, bins=x, label="Hist", weights=weights[:, i])
    height = max(a[0]) * 1.1
    # plt.plot(x, gauss(x, A, 0, ci[0]))
    # plt.plot(x, gauss(x, A, 0, std), label="Gauss")
    # plt.plot(x, gauss(x, A/2, 0, std) + gauss(x, A/2, 0, ci[0]))
    # plt.plot(x, 10 ** stat_func(x, *popt))
    x = np.linspace(bounds[0], bounds[-1], 100)
    plt.plot(x, popt[0] * norm.pdf(x, *popt[1:]), label="Gauss")
    # plt.plot(x, stat_func(x, 2, *popt[1:]) * 100)

    plt.vlines(bounds[[0, 4]], 0, height, colors="r", label="95% C.I.")
    plt.vlines(
        bounds[[1, 3]], 0, height, colors="k", linestyles="dashed", label="68% C.I."
    )
    plt.vlines((-std, std), 0, height, colors="g", linestyles="dashed", label="Std")

    plt.title("%s = %.3g, std = %.3g" % (names[i], par[i], std))
    plt.ylim([0, height])
    # plt.yscale("log")
    plt.xlim([1.1 * bounds[0], 1.1 * bounds[4]])
    plt.legend(loc="best")

plt.show()
input("Wait")

# Determine Age
solar = Abund(0, "asplund2009")

y = sme.abund.get_pattern("sme")["Y"] - solar["Y"]
sy = sigma[4]
fe = sme.abund.get_pattern("sme")["Fe"] - solar["Fe"]
sfe = 0

a = 0.146
sa = 0.011
b = -33
sb = 2

# Classic error propagation
x = y - fe
sx = np.sqrt((sy) ** 2 + (sfe) ** 2)

age = (x - a) / b
sage = np.sqrt(
    (sx * (1 / b)) ** 2 + (sa * (-1 / b)) ** 2 + (sb * (x - a) / b ** 2) ** 2
)

# Bayesian error estimation
def lnprior(f):
    if np.any(f < 0):
        return -np.inf
    return 0
    # ga = ((ab - a) / sa) ** 2
    # gb = ((bb - b) / sb) ** 2
    # return stat_func(x, *popts[4])
    # return -0.5 * (ga + gb)


def lnlike(x, f):
    model = a * x + b
    return -0.5 * (f - model) ** 2


def lnpost(f, x):
    prior = lnprior(f)
    if prior == -np.inf:
        return prior

    like = lnlike(x, f)

    return prior + like


p0 = [age + np.random.randn(1) * 1e-4 for i in range(10)]
sampler = emcee.EnsembleSampler(10, 1, lnpost, args=(x,))
sampler.run_mcmc(p0, 1000)

chain = sampler.flatchain
age = np.percentile(chain, 0.5)
sage = np.percentile(chain, (16, 84))
corner.corner(chain)
plt.show()

print("Age: %.3g +- %.3g Gyr" % (age, sage))

input("Wait")
