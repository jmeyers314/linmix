import numpy as np
import linmix
from astropy.table import Table

def generate_test_data():
    alpha = 4.0
    beta = 3.0
    sigsqr = 0.5

    # GMM with 3 components for xi
    xi = np.random.normal(loc=1.0, scale=1.0, size=9)
    xi = np.concatenate([xi, np.random.normal(loc=2.0, scale=1.5, size=20)])
    xi = np.concatenate([xi, np.random.normal(loc=3.0, scale=0.5, size=30)])
    eta = np.random.normal(loc=alpha+beta*xi, scale=np.sqrt(sigsqr))

    # Let's mix in some weird measurement uncertainties:
    xsig = 0.25 * np.sin(np.arange(len(xi))) + 0.5
    ysig = 0.25 * np.cos(np.arange(len(eta)))**2 + 0.5
    x = np.random.normal(loc=xi, scale=xsig)
    y = np.random.normal(loc=eta, scale=ysig)

    out = Table([x, y, xsig, ysig], names=['x', 'y', 'xsig', 'ysig'])
    import astropy.io.ascii as ascii
    ascii.write(out, 'test.dat')


if __name__ == '__main__':
    import astropy.io.ascii as ascii
    try:
        a = ascii.read('test.dat')
    except:
        generate_test_data()
        a = ascii.read('test.dat')

    lm = linmix.LinMix(a['x'], a['y'], a['xsig'], a['ysig'])
    lm.run_mcmc()
    ascii.write(lm.chain, 'test.pyout')