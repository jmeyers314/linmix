import numpy as np
import triangle
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt

pyout = ascii.read('test.pyout')
idlout = ascii.read('test.idlout')

fig, axarr = plt.subplots(4, 2, figsize=(10, 10))
axarr[0,0].plot(pyout['alpha'])
axarr[0,0].set_ylabel('alpha')

axarr[0,1].plot(pyout['beta'])
axarr[0,1].set_ylabel('beta')

axarr[1,0].plot(pyout['sigsqr'])
axarr[1,0].set_ylabel('sigsqr')

axarr[1,1].plot(pyout['mu0'])
axarr[1,1].set_ylabel('mu0')

axarr[2,0].plot(pyout['usqr'])
axarr[2,0].set_ylabel('usqr')

axarr[2,1].plot(pyout['wsqr'])
axarr[2,1].set_ylabel('wsqr')

axarr[3,0].plot(pyout['ximean'])
axarr[3,0].set_ylabel('ximean')

axarr[3,1].plot(pyout['xisig'])
axarr[3,1].set_ylabel('xisig')


fig, axarr = plt.subplots(4, 2, figsize=(10, 10))
axarr[0,0].plot(idlout['alpha'])
axarr[0,0].set_ylabel('alpha')

axarr[0,1].plot(idlout['beta'])
axarr[0,1].set_ylabel('beta')

axarr[1,0].plot(idlout['sigsqr'])
axarr[1,0].set_ylabel('sigsqr')

axarr[1,1].plot(idlout['mu00'])
axarr[1,1].set_ylabel('mu00')

axarr[2,0].plot(idlout['usqr'])
axarr[2,0].set_ylabel('usqr')

axarr[2,1].plot(idlout['wsqr'])
axarr[2,1].set_ylabel('wsqr')

axarr[3,0].plot(idlout['ximean'])
axarr[3,0].set_ylabel('ximean')

axarr[3,1].plot(idlout['xisig'])
axarr[3,1].set_ylabel('xisig')

plt.show()