import numpy as np
import triangle
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt

pyout = ascii.read('test.pyout')
idlout = ascii.read('test.idlout')

fig, axarr = plt.subplots(9, 9, figsize=(10, 10))
triangle.corner(np.array([pyout['alpha'], pyout['beta'], pyout['sigsqr'], 
                          pyout['mu0'], pyout['usqr'], pyout['wsqr'],
                          pyout['ximean'], pyout['xisig'], pyout['corr']]).T, 
                labels=[r"$\alpha$", r"$\beta$", r"$\sigma^2$", 
                        r"$\mu_0$", r"$u^2$", r"$w^2$",
                        r"$\bar{\xi}$", r"$\sigma_\xi$", r"$\rho_{\xi\eta}$"],
                extents=[0.99]*9, plot_datapoints=False,
                fig=fig)
triangle.corner(np.array([idlout['alpha'], idlout['beta'], idlout['sigsqr'], 
                          idlout['mu00'], idlout['usqr'], idlout['wsqr'],
                          idlout['ximean'], idlout['xisig'], idlout['corr']]).T, 
                extents=[0.99]*9, plot_datapoints=False,
                fig=fig, color='r')
fig.subplots_adjust(bottom=0.065, left=0.07)
plt.show()
