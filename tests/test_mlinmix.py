import numpy as np
import linmix
from astropy.table import Table


def generate_test_data():
    alpha = 1.0
    beta = np.array([2.0, 3.0])
    sigsqr = 4.0

    # GMM with 3 components for xi
    mu1 = np.array([0., 0.])
    mu2 = np.array([1., 2.])
    mu3 = np.array([-1., 1.])
    var1 = np.diag([2.0, 2.0])
    var2 = np.diag([2.0, 3.0])
    var3 = np.diag([3.0, 2.0])

    xi = np.random.multivariate_normal(mu1, var1, size=10)
    xi = np.concatenate([xi, np.random.multivariate_normal(mu2, var2, size=20)])
    xi = np.concatenate([xi, np.random.multivariate_normal(mu3, var3, size=30)])
    eta = np.random.normal(loc=alpha+np.dot(xi, beta), scale=np.sqrt(sigsqr))

    # Mix in some measurement uncertainties:
    x1var = 0.25 * np.sin(np.arange(len(eta))) + 0.5
    x2var = 0.25 * np.cos(np.arange(len(eta)))**3 + 0.5
    xvar = np.array([np.diag([x1v, x2v]) for x1v, x2v in zip(x1var, x2var)])
    yvar = 0.25 * np.cos(np.arange(len(eta)))**2 + 0.5

    x = np.array([np.random.multivariate_normal(xi[i], xvar[i])
                  for i in range(len(eta))])
    x1 = x[:, 0]
    x2 = x[:, 1]
    y = np.random.normal(loc=eta, scale=np.sqrt(yvar))

    out = Table([x1, x2, y, x1var, x2var, yvar], names=['x1', 'x2', 'y', 'x1var', 'x2var', 'yvar'])
    import astropy.io.ascii as ascii
    ascii.write(out, 'test_mlinmix.dat')

    return x, y, xvar, yvar


def run():
    import astropy.io.ascii as ascii
    try:
        a = ascii.read('test_mlinmix.dat')
    except:
        generate_test_data()
        a = ascii.read('test_mlinmix.dat')

    x = np.transpose(np.vstack([a['x1'], a['x2']]))
    xvar = np.array([np.diag([x1v, x2v]) for x1v, x2v in zip(a['x1var'], a['x2var'])])

    lm = linmix.MLinMix(x, a['y'], xvar, a['yvar'])
    lm.run_mcmc()


if __name__ == '__main__':
    run()
