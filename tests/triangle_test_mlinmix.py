import numpy as np
import triangle
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt

pyout = ascii.read('test_mlinmix.pyout')
idlout = ascii.read('test_mlinmix.idlout')

fig, axarr = plt.subplots(4, 4, figsize=(10, 10))
triangle.corner(np.array([pyout['alpha'], pyout['beta0'], pyout['beta1'],
                          pyout['sigsqr']]).T,
                labels=[r"$\alpha$", r"$\beta_0$", r"$\beta_1$", r"$\sigma^2$"],
                extents=[0.99]*4, plot_datapoints=False,
                fig=fig)
triangle.corner(np.array([idlout['alpha'], idlout['beta0'], idlout['beta1'],
                          idlout['sigsqr']]).T,
                extents=[0.99]*4, plot_datapoints=False,
                fig=fig, color='r')
# fig.subplots_adjust(bottom=0.065, left=0.07)
plt.show()
