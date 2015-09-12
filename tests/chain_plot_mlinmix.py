import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt

pyout = ascii.read('test_mlinmix.pyout')
idlout = ascii.read('test_mlinmix.idlout')

fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
axarr[0,0].plot(pyout['alpha'])
axarr[0,0].set_ylabel('alpha')

axarr[0,1].plot(pyout['beta0'])
axarr[0,1].set_ylabel('beta0')

axarr[1,0].plot(pyout['beta1'])
axarr[1,0].set_ylabel('beta1')

axarr[1,1].plot(pyout['sigsqr'])
axarr[1,1].set_ylabel('sigsqr')


fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
axarr[0,0].plot(idlout['alpha'])
axarr[0,0].set_ylabel('alpha')

axarr[0,1].plot(idlout['beta0'])
axarr[0,1].set_ylabel('beta0')

axarr[1,0].plot(idlout['beta1'])
axarr[1,0].set_ylabel('beta1')

axarr[1,1].plot(idlout['sigsqr'])
axarr[1,1].set_ylabel('sigsqr')

plt.show()
