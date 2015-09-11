""" linmix -- A hierarchical Bayesian approach to linear regression with error in both X and Y.
"""

import numpy as np


def randomwish(dof, S):
    dim = S.shape[0]
    x = np.random.multivariate_normal(np.zeros(dim), S, dof)
    return np.dot(x.T, x)


class Chain(object):
    def __init__(self, x, y, xsig, ysig, xycov, delta, K, nchains):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

        if xsig is None:
            self.xsig = np.zeros_like(self.x)
            xycov = np.zeros_like(self.x)
        else:
            self.xsig = np.array(xsig, dtype=float)
        if ysig is None:
            self.ysig = np.zeros_like(self.y)
            xycov = np.zeros_like(self.y)
        else:
            self.ysig = np.array(ysig, dtype=float)
        self.wxerr = (self.xsig != 0.0)
        self.wyerr = (self.ysig != 0.0)
        self.werrs = werrs = self.wxerr & self.wyerr

        if xycov is None:
            self.xycov = np.zeros_like(self.x)
        else:
            self.xycov = np.array(xycov, dtype=float)

        self.xycorr = np.zeros_like(self.xycov)
        self.xycorr[werrs] = self.xycov[werrs] / (self.xsig[werrs] * self.ysig[werrs])

        self.N = len(self.x)
        self.K = K
        self.nchains = nchains

        self.xvar = self.xsig**2
        self.yvar = self.ysig**2

        if delta is None:
            self.delta = np.ones((self.N), dtype=bool)
        else:
            self.delta = np.array(delta, dtype=bool)

        self.initialized = False

    def initial_guess(self):  # Step 1
        # For convenience
        x = self.x
        y = self.y
        xycov = self.xycov
        xvar = self.xvar
        yvar = self.yvar
        N = self.N
        K = self.K

        # Use BCES estimator for initial guess of theta = {alpha, beta, sigsqr}
        self.beta = ((np.cov(x, y, ddof=1)[1, 0] - np.mean(xycov))
                     / (np.var(x, ddof=1) - np.mean(xvar)))
        self.alpha = np.mean(y) - self.beta * np.mean(x)
        self.sigsqr = np.var(y, ddof=1) - np.mean(yvar) - self.beta * (np.cov(x, y, ddof=1)[1, 0]
                                                                       - np.mean(xycov))
        self.sigsqr = np.max([self.sigsqr,
                              0.05 * np.var(y - self.alpha - self.beta * x, ddof=1)])

        self.mu0 = np.median(x)
        self.wsqr = np.var(x, ddof=1) - np.median(xvar)
        self.wsqr = np.max([self.wsqr, 0.01*np.var(x, ddof=1)])

        # Now get an MCMC value dispersed around above values
        X = np.ones((N, 2), dtype=float)
        X[:, 1] = x
        Sigma = np.linalg.inv(np.dot(X.T, X)) * self.sigsqr
        coef = np.random.multivariate_normal([0, 0], Sigma)
        chisqr = np.random.chisquare(self.nchains)
        self.alpha += coef[0] * np.sqrt(1.0/chisqr)
        self.beta += coef[1] * np.sqrt(1.0/chisqr)
        self.sigsqr *= 0.5 * N / np.random.chisquare(0.5*N)

        # Now get the values for the mixture parameters, first do prior params
        self.mu0min = min(x)
        self.mu0max = max(x)

        mu0g = np.nan
        while not (mu0g > self.mu0min) & (mu0g < self.mu0max):
            mu0g = self.mu0 + (np.random.normal(scale=np.sqrt(np.var(x, ddof=1) / N)) /
                               np.sqrt(self.nchains/np.random.chisquare(self.nchains)))
        self.mu0 = mu0g

        # wsqr is the global scale
        self.wsqr *= 0.5 * N / np.random.chisquare(0.5 * N)

        self.usqrmax = 1.5 * np.var(x, ddof=1)
        self.usqr = 0.5 * np.var(x, ddof=1)

        self.tausqr = 0.5 * self.wsqr * self.nchains / np.random.chisquare(self.nchains, size=K)

        self.mu = self.mu0 + np.random.normal(scale=np.sqrt(self.wsqr), size=K)

        # get initial group proportions and group labels

        pig = np.zeros(self.K, dtype=float)
        if K == 1:
            self.G = np.ones(N, dtype=int)
            self.pi = np.array([1], dtype=float)
        else:
            self.G = np.zeros((N, K), dtype=int)
            for i in xrange(N):
                minind = np.argmin(abs(x[i] - self.mu))
                pig[minind] += 1
                self.G[i, minind] = 1
            self.pi = np.random.dirichlet(pig+1)

        self.eta = y.copy()
        self.y_ul = y.copy()
        self.xi = x.copy()

        self.cens = np.nonzero(np.logical_not(self.delta))[0]

        self.initialized = True

    def update_cens_y(self):  # Step 2
        todo = self.cens[:]
        while len(todo) > 0:
            self.y[todo] = np.random.normal(loc=self.eta[todo],
                                            scale=np.sqrt(self.yvar[todo]),
                                            size=len(todo))
            todo = np.nonzero(np.logical_not(self.delta) & (self.y > self.y_ul))[0]

    def update_xi(self):  # Step 3
        wxerr = self.wxerr
        wyerr = self.wyerr

        # Eqn (58)
        sigma_xihat_ik_sqr = 1.0/(1.0/(self.xvar * (1.0 - self.xycorr**2))[:, np.newaxis]
                                  + self.beta**2 / self.sigsqr
                                  + 1.0/self.tausqr)
        # Eqn (57)
        sigma_xihat_i_sqr = np.sum(self.G * sigma_xihat_ik_sqr, axis=1)
        # Eqn (56)
        xihat_xy_i = self.x.copy()
        xihat_xy_i[wyerr] += (self.xycov / self.yvar * (self.eta - self.y))[wyerr]
        # Eqn (55)
        xihat_ik = (sigma_xihat_i_sqr[:, np.newaxis]
                    * ((xihat_xy_i/self.xvar
                        * (1.0 - self.xycorr**2))[:, np.newaxis]
                       + self.beta*(self.eta[:, np.newaxis] - self.alpha)/self.sigsqr
                       + self.mu/self.tausqr))
        # Eqn (54)
        xihat_i = np.sum(self.G * xihat_ik, axis=1)
        # Eqn (53)
        self.xi[wxerr] = np.random.normal(loc=xihat_i[wxerr],
                                          scale=np.sqrt(sigma_xihat_i_sqr[wxerr]))

    def update_eta(self):  # Step 4
        wxerr = self.wxerr
        wyerr = self.wyerr

        etaxyvar = self.yvar * (1.0 - self.xycorr**2)
        etaxy = self.y.copy()
        etaxy[wxerr] += (self.xycov / self.xvar * (self.xi - self.x))[wxerr]

        # Eqn (68)
        sigma_etahat_i_sqr = 1.0/(1.0/etaxyvar + 1.0/self.sigsqr)
        # Eqn (67)
        etahat_i = (sigma_etahat_i_sqr * (etaxy / etaxyvar
                    + (self.alpha + self.beta * self.xi) / self.sigsqr))
        # Eqn (66)
        self.eta[wyerr] = np.random.normal(loc=etahat_i[wyerr],
                                           scale=np.sqrt(sigma_etahat_i_sqr[wyerr]))

    def update_G(self):  # Step 5
        # Eqn (74)
        piNp = self.pi * (1.0/np.sqrt(2.0*np.pi*self.tausqr)
                          * np.exp(-0.5 * (self.xi[:, np.newaxis] - self.mu)**2 / self.tausqr))
        q_ki = piNp / np.sum(piNp, axis=1)[:, np.newaxis]
        # Eqn (73)
        for i in xrange(self.N):
            self.G[i] = np.random.multinomial(1, q_ki[i])

    def update_alpha_beta(self):  # Step 6
        X = np.ones((self.N, 2), dtype=float)
        X[:, 1] = self.xi
        # Eqn (77)
        XTXinv = np.linalg.inv(np.dot(X.T, X))
        Sigma_chat = XTXinv * self.sigsqr
        # Eqn (76)
        chat = np.dot(np.dot(XTXinv, X.T), self.eta)
        # Eqn (75)
        self.alpha, self.beta = np.random.multivariate_normal(chat, Sigma_chat)

    def update_sigsqr(self):  # Step 7
        # Eqn (80)
        ssqr = 1.0/(self.N-2) * np.sum((self.eta - self.alpha - self.beta * self.xi)**2)
        # Eqn (79)
        nu = self.N - 2
        # Eqn (78)
        self.sigsqr = nu * ssqr / np.random.chisquare(nu)

    def update_pi(self):  # Step 8
        # Eqn (82)
        self.nk = np.sum(self.G, axis=0)
        # Eqn (81)
        self.pi = np.random.dirichlet(self.nk+1)

    def update_mu(self):  # Step 9
        Gsum = np.sum(self.G * self.xi[:, np.newaxis], axis=0)
        for k in xrange(self.K):
            if self.nk[k] != 0:
                # Eqn (86)
                Sigma_muhat_k = 1.0/(1.0/self.usqr + self.nk[k]/self.tausqr[k])
                # Eqn (85)
                xibar_k = 1.0/self.nk[k] * Gsum[k]
                # Eqn (84)
                muhat_k = Sigma_muhat_k * (self.mu0/self.usqr + self.nk[k]/self.tausqr[k]*xibar_k)
                # Eqn (83)
                self.mu[k] = np.random.normal(loc=muhat_k, scale=np.sqrt(Sigma_muhat_k))
            else:
                self.mu[k] = np.random.normal(loc=self.mu0, scale=np.sqrt(self.usqr))

    def update_tausqr(self):  # Step 10
        # Eqn (88)
        nu_k = self.nk + 1
        # Eqn (89)
        tk_sqr = 1.0/nu_k * (self.wsqr + np.sum(self.G*(self.xi[:, np.newaxis]-self.mu)**2, axis=0))
        # Eqn (87)
        self.tausqr = tk_sqr * nu_k / np.random.chisquare(nu_k, size=self.K)

    def update_mu0(self):  # Step 11
        # Eqn (94)
        mubar = np.mean(self.mu)
        # Eqn (93)
        self.mu0 = np.random.normal(loc=mubar, scale=np.sqrt(self.usqr/self.K))

    def update_usqr(self):  # Step 12
        # Eqn (96)
        nu_u = self.K + 1
        # Eqn (97)
        usqrhat = 1.0/nu_u * (self.wsqr + np.sum((self.mu - self.mu0)**2))
        usqr = np.inf
        while not usqr <= self.usqrmax:
            usqr = usqrhat * nu_u / np.random.chisquare(nu_u)
        self.usqr = usqr

    def update_wsqr(self):  # Step 13
        # Eqn (102)
        a = 0.5 * (self.K + 3)
        # Eqn (103)
        b = 0.5 * (1.0/self.usqr + np.sum(1.0/self.tausqr))
        # Eqn (101)
        self.wsqr = np.random.gamma(a, 1.0/b)

    def initialize_chain(self, chain_length):
        self.chain_dtype = [('alpha', float),
                            ('beta', float),
                            ('sigsqr', float),
                            ('pi', (float, self.K)),
                            ('mu', (float, self.K)),
                            ('tausqr', (float, self.K)),
                            ('mu0', float),
                            ('usqr', float),
                            ('wsqr', float),
                            ('ximean', float),
                            ('xisig', float),
                            ('corr', float)]
        self.chain = np.empty((chain_length,), dtype=self.chain_dtype)
        self.ichain = 0

    def extend_chain(self, length):
        extension = np.empty((length), dtype=self.chain_dtype)
        self.chain = np.hstack((self.chain, extension))

    def update_chain(self):
        self.chain['alpha'][self.ichain] = self.alpha
        self.chain['beta'][self.ichain] = self.beta
        self.chain['sigsqr'][self.ichain] = self.sigsqr
        self.chain['pi'][self.ichain] = self.pi
        self.chain['mu'][self.ichain] = self.mu
        self.chain['tausqr'][self.ichain] = self.tausqr
        self.chain['mu0'][self.ichain] = self.mu0
        self.chain['usqr'][self.ichain] = self.usqr
        self.chain['wsqr'][self.ichain] = self.wsqr
        ximean = np.sum(self.pi * self.mu)
        self.chain['ximean'][self.ichain] = ximean
        xisig = np.sqrt(np.sum(self.pi * (self.tausqr + self.mu**2)) - ximean**2)
        self.chain['xisig'][self.ichain] = xisig
        self.chain['corr'][self.ichain] = self.beta * xisig / np.sqrt(self.beta**2 * xisig**2
                                                                      + self.sigsqr)
        self.ichain += 1

    def step(self, niter):
        for i in xrange(niter):
            self.update_cens_y()
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            self.update_xi()
            self.update_eta()
            np.seterr(**old_settings)
            self.update_G()
            self.update_alpha_beta()
            self.update_sigsqr()
            self.update_pi()
            self.update_mu()
            self.update_tausqr()
            self.update_mu0()
            self.update_usqr()
            self.update_wsqr()
            self.update_chain()


class MultiChain(object):
    def __init__(self, x, y, xvar, yvar, xycov, delta, K, nchains):
        shape = x.shape
        N = self.N = shape[0]
        Np = self.Np = shape[1]

        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

        if xvar is None:
            self.xvar = np.zeros((N, Np, Np), dtype=float)
            xycov = np.zeros((N, Np), dtype=float)
        else:
            self.xvar = np.array(xvar, dtype=float)
            # Convert diagonals-only input vector input matrix (for each sample)
            if self.xvar.ndim == 2:
                tmp = np.zeros((N, Np, Np), dtype=float)
                for i in range(N):
                    tmp[i, ...] = np.diag(self.xvar[i, ...])
                self.xvar = tmp
        if yvar is None:
            self.yvar = np.zeros_like(self.y)
            xycov = np.zeros((N, Np), dtype=float)
        else:
            self.yvar = np.array(yvar, dtype=float)

        # self.wxerr = (self.xsig != 0.0)
        # self.wyerr = (self.ysig != 0.0)
        # self.werrs = werrs = self.wxerr & self.wyerr

        if xycov is None:
            self.xycov = np.zeros((N, Np), dtype=float)
        else:
            self.xycov = np.array(xycov, dtype=float)

        self.K = K
        self.nchains = nchains

        if delta is None:
            self.delta = np.ones((N), dtype=bool)
        else:
            self.delta = np.array(delta, dtype=bool)

        self.xyvar = np.zeros((N, Np+1, Np+1), dtype=float)
        self.xyvar[:, 0, 0] = self.yvar
        self.xyvar[:, 1:, 0] = self.xycov
        self.xyvar[:, 0, 1:] = self.xycov
        self.xyvar[:, 1:, 1:] = self.xvar

        self.initialized = False

    def initial_guess(self):  # Step 1
        diag = np.arange(self.Np + 1) * (self.Np + 2)

        # Moment correction method to estimate regression coefficients and intrinsic dispersion
        Xmat = np.ones((self.N), dtype=float)
        Xmat = np.hstack([Xmat[:, np.newaxis], self.x])
        denom = np.dot(Xmat.T, Xmat)
        Vcoef = denom
        denom[1:, 1:] -= np.median(self.xvar, axis=0)

        denom_diag = np.diag(denom[1:, 1:])
        denom_diag = np.maximum(denom_diag, 0.025 * np.diag(Vcoef[1:, 1:]))
        denom.flat[diag[1:]] = denom_diag

        numer = np.dot(self.y, Xmat) - np.concatenate([[0.0], np.median(self.xycov, axis=0)])
        coef = np.linalg.solve(denom, numer)

        alpha = coef[0]
        beta = coef[1:]

        sigsqr = np.var(self.y, ddof=1) - np.mean(self.yvar)

        sigsqr -= np.dot(beta, np.dot(np.cov(self.x.T) - np.median(self.xvar, axis=0), beta.T))

        sigsqr = np.max(sigsqr, 0.05 * np.var(self.y - alpha - np.dot(self.x, beta)))

        # Randomly disperse starting values for (alpha, beta) from a
        # multivariate Students-t distribution with 4 degrees of freedom

        Vcoef = np.linalg.inv(Vcoef) * sigsqr * 4
        coef = np.random.multivariate_normal(mean=np.zeros((self.Np+1,), dtype=float), cov=Vcoef)
        chisqr = np.random.chisquare(4)

        self.alpha = alpha + coef[0] * np.sqrt(4.0 / chisqr)
        self.beta = beta + coef[1:] * np.sqrt(4.0 / chisqr)

        # Draw sigsqr from an Inverse scaled chi-square density
        self.sigsqr = sigsqr * self.N/2.0 / np.random.chisquare(self.N/2.0)

        # Randomly choose K data pints, set these to the group means
        self.mu = self.x[np.random.permutation(self.N)[0:self.K]].T

        if self.K > 1:
            # get distance of data points to each centroid; classify to closest centroid
            dist = np.zeros((self.N, self.K), dtype=float)
            for k in range(self.K):
                dist[:, k] = np.sum((self.x - self.mu[:,k])**2, axis=1)
            self.G = np.argmin(dist, axis=1)
        else:
            self.G = np.ones((self.N,), dtype=int)
        self.nk = np.empty(self.K, dtype=int)
        for k in range(self.K):
            self.nk[k] = np.sum(self.G == k)

        # Next get initial guesses for Pi and T
        self.pi = np.empty(self.K, dtype=float)
        self.T = np.empty((self.Np, self.Np, self.K), dtype=float)
        for k in range(self.K):
            gk = self.G == k
            nk = np.sum(gk)
            if nk > self.Np:
                self.pi[k] = float(nk) / self.N
                self.T[:, :, k] = np.cov(self.x[gk, :].T)
            else:
                self.pi[k] = float(max(1, nk)) / self.N
                self.T[:, :, k] = np.cov(self.x.T)
        # Make sure Pi sums to unity
        self.pi /= np.sum(self.pi)

        # Now get initial guesses for prior parameters
        self.mu0 = self.mu[:, 0] if self.K == 1 else np.mean(self.mu, axis=1)
        Smat = np.cov(self.x.T)
        self.U = randomwish(self.N, Smat / self.N)
        self.W = randomwish(self.N, Smat / self.N)

        self.xyvar_inv = np.empty_like(self.xyvar)
        for i in range(self.N):
            self.xyvar_inv[i, :, :] = np.linalg.inv(self.xyvar[i, :, :])

        # Get starting values for eta
        self.eta = self.y.copy()
        self.xi = self.x.copy()
        self.y_ul = self.y.copy()

        # Initialize some matrix inverses
        self.Tk_inv = np.empty((self.Np, self.Np, self.K), dtype=float)
        for k in range(self.K):
            self.Tk_inv[:, :, k] = np.linalg.inv(self.T[:, :, k])
        self.U_inv = np.linalg.inv(self.U)

        self.cens = np.nonzero(np.logical_not(self.delta))[0]

    def update_cens_y(self):  # Step 2
        todo = self.cens[:]
        while len(todo) > 0:
            self.y[todo] = np.random.normal(loc=self.eta[todo],
                                            scale=np.sqrt(self.yvar[todo]),
                                            size=len(todo))
            todo = np.nonzero(np.logical_not(self.delta) & (self.y > self.y_ul))[0]

    def update_xi(self):  # Step 3
        xstar = np.empty((self.N, self.Np), dtype=float)
        for j in range(self.Np):
            # The "leave-one-out" subscripts
            if j == 0:
                inactive = np.arange(self.Np-1)+1
            elif j == self.Np - 1:
                inactive = np.arange(self.Np-1)
            else:
                inactive = np.hstack([np.arange(j), np.arange(self.Np-j-1)+j+1])

            # Part of Eqn (62)
            xstar[:, j] = self.x[:, j]
            xstar[:, inactive] = self.x[:, inactive] - self.xi[:, inactive]

            # Part of Eqn (62)
            zstar = np.hstack([(self.y - self.eta)[:, np.newaxis], xstar])

            # Term in Eqn (61) numerator
            zmu = np.sum(self.xyvar_inv[:, :, j+1] * zstar, axis=1)

            # For Eqn (63)
            mustar = np.empty((self.N, self.Np), dtype=float)

            # Do one Gaussian at a time
            for k in range(self.K):
                gk = np.nonzero(self.G == k)[0]
                if len(gk) > 0:
                    # Eqn (63)
                    mustar[gk, j] = self.mu[j, k]
                    for l in range(self.Np-2):
                        mustar[gk, inactive[l]] = (self.mu[inactive[l], k]
                                                   - self.xi[gk, inactive[l]])
                    # Term in Eqn (61) numerator
                    mmu = np.dot(self.Tk_inv[:, j, k], mustar[gk, :].T)

                    # Term in Eqn (61) numerator
                    etamu = (self.eta[gk] - self.alpha
                             - np.dot(self.beta[inactive], self.xi[np.ix_(gk, inactive)].T))

                    # Eqn (65)
                    xihvar = 1. / (self.xyvar_inv[gk, j+1, j+1] + self.Tk_inv[j, j, k]
                                   + self.beta[j]**2 / self.sigsqr)
                    # Eqn (61)
                    xihat = xihvar * (zmu[gk] + mmu + self.beta[j] * etamu / self.sigsqr)

                    # Eqn (59)
                    self.xi[gk, j] = xihat + np.sqrt(xihvar) * np.random.normal(size=len(gk))

    def update_eta(self):  # Step 4
        # Eqn (72)
        zstar = np.hstack([self.y[:, np.newaxis], self.x-self.xi])

        # Term in Eqn (70) numerator
        zmu = np.sum(self.xyvar_inv[:, :, 0] * zstar, axis=1)

        # Term in Eqn (70) numerator
        ximu = self.alpha + np.dot(self.beta, self.xi.T) / self.sigsqr

        # Denominator in Eqn (70)
        etahvar = 1. / (self.xyvar_inv[:, 0, 0] + 1./self.sigsqr)

        # Eqn (70)
        etahat = etahvar * (zmu + ximu)

        # Eqn (69)
        self.eta = etahat + np.sqrt(etahvar) * np.random.normal(size=self.N)

    def update_G(self):  # Step 5
        gamma = np.empty((self.N, self.K), dtype=float)
        for k in range(self.K):
            xicent = self.xi - np.outer(np.ones(self.N), self.mu[:, k])
            # Eqn (74)
            gamma[:, k] = (
                self.pi[k] / ((2.0 * np.pi)**(self.Np/2.0) * np.linalg.det(self.T[:, :, k]))
                * np.exp(-0.5 * np.sum(xicent * np.dot(xicent, self.Tk_inv[:, :, k]), axis=1)))
        norm = np.sum(gamma, axis=1)
        for j in range(self.N):
            gamma0 = gamma[j, :]/norm[j]
            # Eqn (73)
            Gjk = np.random.multinomial(1, gamma0)
            self.G[j] = np.nonzero(Gjk)[0]

    def update_alpha_beta(self):  # Step 6
        X = np.ones((self.N, self.Np+1), dtype=float)
        X[:, 1:] = self.xi
        # Eqn (77)
        XTXinv = np.linalg.inv(np.dot(X.T, X))
        Sigma_chat = XTXinv * self.sigsqr
        # Eqn (76)
        chat = np.dot(np.dot(XTXinv, X.T), self.eta)
        # Eqn (75)
        x = np.random.multivariate_normal(chat, Sigma_chat)
        self.alpha = x[0]
        self.beta = x[1:]

    def update_sigsqr(self):  # Step 7
        # Eqn (80)
        ssqr = 1.0/(self.N-2) * np.sum((self.eta - self.alpha - np.dot(self.xi, self.beta))**2)
        # Eqn (79)
        nu = self.N - 2
        # Eqn (78)
        self.sigsqr = nu * ssqr / np.random.chisquare(nu)

    def update_pi(self):  # Step 8
        # Eqn (82)
        for k in range(self.K):
            self.nk[k] = np.sum(self.G == k)
        # Eqn (81)
        self.pi = np.random.dirichlet(self.nk+1)

    def update_mu(self):  # Step 9
        for k in xrange(self.K):
            if self.nk[k] > 0:
                gk = (self.G == k)
                # Eqn (86)
                Sigma_muhat_k = np.linalg.inv(self.U_inv + self.nk[k] * self.Tk_inv[:, :, k])
                # Eqn (85)
                xibar_k = np.mean(self.xi[gk], axis=0)
                # Eqn (84)
                muhat_k = np.dot(Sigma_muhat_k, (np.dot(self.mu0, self.U_inv)
                                                 + self.nk[k] * np.dot(self.Tk_inv[:, :, k],
                                                                       xibar_k)))
                # Eqn (83)
                self.mu[:, k] = np.random.multivariate_normal(mean=muhat_k, cov=Sigma_muhat_k)
            else:
                self.mu[:, k] = np.random.normal(mean=self.mu0, cov=self.U)

    def update_T(self):  # Step 10
        for k in xrange(self.K):
            if self.nk[k] > 0:
                gk = (self.G == k)
                xicent = self.xi[gk, :] - np.outer(np.ones(self.nk[k]), self.mu[:, k])
                Smat = np.linalg.inv(self.W + np.dot(xicent.T, xicent))
            else:
                Smat = np.linalg.inv(W)
            Tmat = randomwish(self.nk[k] + self.Np, Smat)
            self.Tk_inv[:, :, k] = Tmat
            self.T[:, :, k] = np.linalg.inv(Tmat)

    def update_mu0(self):  # Step 11
        # Eqn (94)
        mubar = np.mean(self.mu, axis=1)
        # Eqn (93)
        self.mu0 = np.random.multivariate_normal(mean=mubar, cov=self.U/self.K)

    # def update_usqr(self):  # Step 12
    #     # Eqn (96)
    #     nu_u = self.K + 1
    #     # Eqn (97)
    #     usqrhat = 1.0/nu_u * (self.wsqr + np.sum((self.mu - self.mu0)**2))
    #     usqr = np.inf
    #     while not usqr <= self.usqrmax:
    #         usqr = usqrhat * nu_u / np.random.chisquare(nu_u)
    #     self.usqr = usqr
    #
    # def update_wsqr(self):  # Step 13
    #     # Eqn (102)
    #     a = 0.5 * (self.K + 3)
    #     # Eqn (103)
    #     b = 0.5 * (1.0/self.usqr + np.sum(1.0/self.tausqr))
    #     # Eqn (101)
    #     self.wsqr = np.random.gamma(a, 1.0/b)
    #

    def initialize_chain(self, chain_length):
        self.chain_dtype = [('alpha', float),
                            ('beta', (float, self.Np)),
                            ('sigsqr', float),
                            ('pi', (float, self.K)),
                            ('mu', (float, (self.Np, self.K))),
                            ('T', (float, (self.Np, self.Np, self.K))),
                            ('mu0', (float, self.Np)),
                            ('U', (float, (self.Np, self.Np))),
                            ('W', (float, (self.Np, self.Np)))]
        self.chain = np.empty((chain_length,), dtype=self.chain_dtype)
        self.ichain = 0

    def extend_chain(self, length):
        extension = np.empty((length), dtype=self.chain_dtype)
        self.chain = np.hstack((self.chain, extension))

    def update_chain(self):
        self.chain['alpha'][self.ichain] = self.alpha
        self.chain['beta'][self.ichain] = self.beta
        self.chain['sigsqr'][self.ichain] = self.sigsqr
        self.chain['pi'][self.ichain] = self.pi
        self.chain['mu'][self.ichain] = self.mu
        self.chain['T'][self.ichain] = self.T
        self.chain['mu0'][self.ichain] = self.mu0
        self.chain['U'][self.ichain] = self.U
        self.chain['W'][self.ichain] = self.W
        self.ichain += 1

    def step(self, niter):
        for i in xrange(niter):
            self.update_cens_y()
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            self.update_xi()
            self.update_eta()
            np.seterr(**old_settings)
            self.update_G()
            self.update_alpha_beta()
            self.update_sigsqr()
            self.update_pi()
            self.update_mu()
            self.update_T()
            self.update_mu0()
            # self.update_U()
            # self.update_W()
            self.update_chain()


class LinMix(object):
    """ A class to perform linear regression of `y` on `x` when there are measurement errors in
    both variables.  The regression assumes:

    eta = alpha + beta * xi + epsilon

    x = xi + xerr

    y = eta + yerr

    Here, `alpha` and `beta` are the regression coefficients, `epsilon` is the intrinsic random
    scatter about the regression, `xerr` is the measurement error in `x`, and `yerr` is the
    measurement error in `y`.  `epsilon` is assumed to be normally-distributed with mean zero and
    variance `sigsqr`.  `xerr` and `yerr` are assumed to be normally-distributed with means equal
    to zero, variances `xsig`^2 and `ysig`^2, respectively, and covariance `xycov`. The
    distribution of `xi` is modelled as a mixture of normals, with group proportions `pi`, means
    `mu`, and variances `tausqr`.

    Args:
        x(array_like): The observed independent variable.
        y(array_like): The observed dependent variable.
        xsig(array_like): 1-sigma measurement errors in x.
        ysig(array_like): 1-sigma measurement errors in y.
        xycov(array_like): Covariance between the measurement errors in x and y.
        delta(array_like): Array indicating whether a data point is censored (i.e., not detected),
            or not.  If delta[i] == 1, then the ith source is detected.  If delta[i] == 0, then
            the ith source is not detected and y[i] will be interpreted as an upper limit.  Note
            that if there are censored data points, then the maximum-likelihood estimate
            (alpha, beta, sigsqr) is not valid.  By default, all data points are assumed to be
            detected.
        K(int): The number of Gaussians to use in the mixture model for the distribution of xi.
        nchains(int): The number of Monte Carlo Markov Chains to instantiate.

    Attributes:
        nchains(int): The number of instantiated MCMCs.
        chain(numpy recarray): The concatenated MCMCs themselves.  Actually, only the concatenation
            of the last half of each chain is stored here after convergence is reached.  The
            recarray has the following columns:
                - alpha(float): The regression intercept.
                - beta(float): The regression slope.
                - sigsqr(float): The regression intrinsic scatter.
                - pi(array_like): The mixture model component fractions.
                - mu(array_like): The mixture model component means.
                - tausqr(array_like): The mixture model component variances.
                - mu0(float): The hyperparameter describing the prior variance of the distribution
                    of mixture means.
                - usqr(float): The hyperparameter describing the prior variance of the distribution
                    of mixture variances.
                - wsqr(float): The hyperparameter describing the typical scale for the prior on
                    `usqr` and `tausqr`.
                - ximean(float): The mean of the distribution for the independent latent variable
                    `xi`.
                - xisig(float): The standard deviation of the distribution for the independent
                    latent variable `xi`.
                - corr(float): The linear correlation coefficient between the latent dependent and
                    independent variables `xi` and `eta`.
    """
    def __init__(self, x, y, xsig=None, ysig=None, xycov=None, delta=None, K=3, nchains=4):
        self.nchains = nchains
        self.chains = [Chain(x, y, xsig, ysig, xycov, delta, K, self.nchains)
                       for i in xrange(self.nchains)]

    def _get_psi(self):
        c0 = self.chains[0]
        ndraw = c0.ichain/2
        psi = np.empty((ndraw, self.nchains, 6), dtype=float)
        psi[:, :, 0] = np.vstack([c.chain['alpha'][0:ndraw] for c in self.chains]).T
        beta = np.vstack([c.chain['beta'][0:ndraw] for c in self.chains]).T
        psi[:, :, 1] = beta
        sigsqr = np.vstack([c.chain['sigsqr'][0:ndraw] for c in self.chains]).T
        psi[:, :, 2] = np.log(sigsqr)
        ximean = np.vstack([np.sum(c.chain['pi'][0:ndraw] * c.chain['mu'][0:ndraw], axis=1)
                            for c in self.chains]).T
        psi[:, :, 3] = ximean
        xivar = np.vstack([np.sum(c.chain['pi'][0:ndraw] * (c.chain['tausqr'][0:ndraw] +
                                                            c.chain['mu'][0:ndraw]**2),
                                  axis=1)
                           for c in self.chains]).T - ximean**2
        psi[:, :, 4] = xivar
        psi[:, :, 5] = np.arctanh(beta * np.sqrt(xivar / (beta**2 * xivar + sigsqr)))
        return psi

    def _get_Rhat(self):
        psi = self._get_psi()
        ndraw = psi.shape[0]
        psibarj = np.sum(psi, axis=0)/ndraw
        psibar = np.mean(psibarj, axis=0)
        sjsqr = np.sum((psi-psibarj)**2 / (ndraw-1.0), axis=(0, 1))
        Bvar = ndraw / (self.nchains-1.0) * np.sum((psibarj-psibar)**2, axis=0)
        Wvar = sjsqr / self.nchains
        varplus = (1.0 - 1.0 / ndraw) * Wvar + Bvar / ndraw
        Rhat = np.sqrt(varplus / Wvar)
        return Rhat

    def run_mcmc(self, miniter=5000, maxiter=100000, silent=False):
        """ Run the Markov Chain Monte Carlo for the LinMix object.

        Bayesian inference is employed, and a Markov chain containing random draws from the
        posterior is developed.  Convergence of the MCMC to the posterior is monitored using the
        potential scale reduction factor (RHAT, Gelman et al. 2004). In general, when RHAT < 1.1
        then approximate convergence is reached.  After convergence is reached, the second halves
        of all chains are concatenated and stored in the `.chain` attribute as a numpy recarray.

        Args:
            miniter(int): The minimum number of iterations to use.
            maxiter(int): The maximum number of iterations to use.
            silent(bool): If true, then suppress updates during sampling.
        """
        checkiter = 100
        for c in self.chains:
            c.initial_guess()
            c.initialize_chain(miniter)
        for i in xrange(0, miniter, checkiter):
            for c in self.chains:
                c.step(checkiter)
            Rhat = self._get_Rhat()

            if not silent:
                print
                print "Iteration: ", i+checkiter
                print ("Rhat values for alpha, beta, log(sigma^2)"
                       ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
                print Rhat

        i += checkiter
        while not np.all(Rhat < 1.1) and (i < maxiter):
            for c in self.chains:
                c.extend_chain(checkiter)
                c.step(checkiter)
            Rhat = self._get_Rhat()
            if not silent:
                print
                print "Iteration: ", i+checkiter
                print ("Rhat values for alpha, beta, log(sigma^2)"
                       ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
                print Rhat
                i += checkiter

        # Throw away first half of each chain
        self.chain = np.hstack([c.chain[0:i/2] for c in self.chains])


class MLinMix(object):
    """ A class to perform linear regression of `y` on `{x}` when there are measurement errors in
    both variables.  The regression assumes:

    eta = alpha + beta # xi + epsilon

    x = xi + xerr

    y = eta + yerr

    Here, `alpha` and `beta` are the regression coefficients, `epsilon` is the intrinsic random
    scatter about the regression, `xerr` is the measurement error in `x`, and `yerr` is the
    measurement error in `y`.  `epsilon` is assumed to be normally-distributed with mean zero and
    variance `sigsqr`.  `xerr` and `yerr` are assumed to be normally-distributed with means equal
    to zero, variances `xsig`^2 and `ysig`^2, respectively, and covariance `xycov`. The
    distribution of `xi` is modelled as a mixture of normals, with group proportions `pi`, means
    `mu`, and variances `tausqr`.

    Args:
        x(array_like): The observed independent variable.
        y(array_like): The observed dependent variable.
        xsig(array_like): 1-sigma measurement errors in x.
        ysig(array_like): 1-sigma measurement errors in y.
        xycov(array_like): Covariance between the measurement errors in x and y.
        delta(array_like): Array indicating whether a data point is censored (i.e., not detected),
            or not.  If delta[i] == 1, then the ith source is detected.  If delta[i] == 0, then
            the ith source is not detected and y[i] will be interpreted as an upper limit.  Note
            that if there are censored data points, then the maximum-likelihood estimate
            (alpha, beta, sigsqr) is not valid.  By default, all data points are assumed to be
            detected.
        K(int): The number of Gaussians to use in the mixture model for the distribution of xi.
        nchains(int): The number of Monte Carlo Markov Chains to instantiate.

    Attributes:
        nchains(int): The number of instantiated MCMCs.
        chain(numpy recarray): The concatenated MCMCs themselves.  Actually, only the concatenation
            of the last half of each chain is stored here after convergence is reached.  The
            recarray has the following columns:
                - alpha(float): The regression intercept.
                - beta(float): The regression slope.
                - sigsqr(float): The regression intrinsic scatter.
                - pi(array_like): The mixture model component fractions.
                - mu(array_like): The mixture model component means.
                - tausqr(array_like): The mixture model component variances.
                - mu0(float): The hyperparameter describing the prior variance of the distribution
                    of mixture means.
                - usqr(float): The hyperparameter describing the prior variance of the distribution
                    of mixture variances.
                - wsqr(float): The hyperparameter describing the typical scale for the prior on
                    `usqr` and `tausqr`.
                - ximean(float): The mean of the distribution for the independent latent variable
                    `xi`.
                - xisig(float): The standard deviation of the distribution for the independent
                    latent variable `xi`.
                - corr(float): The linear correlation coefficient between the latent dependent and
                    independent variables `xi` and `eta`.
    """
    def __init__(self, x, y, xvar=None, yvar=None, xycov=None, delta=None, K=3, nchains=4):
        self.nchains = nchains
        self.chains = [MultiChain(x, y, xvar, yvar, xycov, delta, K, self.nchains)
                       for i in xrange(self.nchains)]

    def run_mcmc(self, miniter=5000, maxiter=100000, silent=False):
        """ Run the Markov Chain Monte Carlo for the MLinMix object.

        Bayesian inference is employed, and a Markov chain containing random draws from the
        posterior is developed.  Convergence of the MCMC to the posterior is monitored using the
        potential scale reduction factor (RHAT, Gelman et al. 2004). In general, when RHAT < 1.1
        then approximate convergence is reached.  After convergence is reached, the second halves
        of all chains are concatenated and stored in the `.chain` attribute as a numpy recarray.

        Args:
            miniter(int): The minimum number of iterations to use.
            maxiter(int): The maximum number of iterations to use.
            silent(bool): If true, then suppress updates during sampling.
        """
        # checkiter = 100
        checkiter = 1
        for c in self.chains:
            c.initial_guess()
            c.initialize_chain(miniter)
        # for i in xrange(0, miniter, checkiter):

        for i in xrange(0, 1, checkiter):
            for c in self.chains:
                c.step(checkiter)
        #     Rhat = self._get_Rhat()
        #
        #     if not silent:
        #         print
        #         print "Iteration: ", i+checkiter
        #         print ("Rhat values for alpha, beta, log(sigma^2)"
        #                ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
        #         print Rhat
        #
        # i += checkiter
        # while not np.all(Rhat < 1.1) and (i < maxiter):
        #     for c in self.chains:
        #         c.extend_chain(checkiter)
        #         c.step(checkiter)
        #     Rhat = self._get_Rhat()
        #     if not silent:
        #         print
        #         print "Iteration: ", i+checkiter
        #         print ("Rhat values for alpha, beta, log(sigma^2)"
        #                ", mean(xi), log(var(xi)), atanh(corr(xi, eta)):")
        #         print Rhat
        #         i += checkiter
        #
        # # Throw away first half of each chain
        # self.chain = np.hstack([c.chain[0:i/2] for c in self.chains])
