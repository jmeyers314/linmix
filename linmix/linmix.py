import numpy as np
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar

class LinMix(object):
    def __init__(self, x, y, xsig, ysig, xycov=None, K=3):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.xsig = np.array(xsig, dtype=float)
        self.ysig = np.array(ysig, dtype=float)
        self.N = len(self.x)
        self.K = K
        if xycov is None:
            self.xycov = np.zeros_like(self.x)
        else:
            self.xycov = np.array(xycov, dtype=float)
        self.xycorr = self.xycov / (self.xsig * self.ysig)
        self.xvar = self.xsig**2
        self.yvar = self.ysig**2

    def initial_guess(self): # Step 1
        # For convenience
        x = self.x
        y = self.y
        xsig = self.xsig
        ysig = self.ysig
        xycov = self.xycov
        xycorr = self.xycorr
        xvar = self.xvar
        yvar = self.yvar
        N = self.N
        K = self.K

        # Use BCES estimator for initial guess of theta = {alpha, beta, sigsqr}
        self.beta = ((np.cov(x, y, ddof=1)[1,0] - np.mean(xycov))
                     / (np.var(x, ddof=1) - np.mean(xvar)))
        self.alpha = np.mean(y) - self.beta * np.mean(x)
        self.sigsqr = np.var(y, ddof=1) - np.mean(yvar) - self.beta * (np.cov(x, y, ddof=1)[1,0]
                                                                       - np.mean(xycov))
        self.sigsqr = np.max([self.sigsqr,
                              0.05 * np.var(y - self.alpha - self.beta * x, ddof=1)])

        self.mu0 = np.median(x)
        self.wsqr = np.var(x, ddof=1) - np.median(xvar)
        self.wsqr = np.max([self.wsqr, 0.01*np.var(x, ddof=1)])

        # Now get an MCMC value dispersed around above values
        X = np.ones((N, 2), dtype=float)
        X[:,1] = x
        Sigma = np.linalg.inv(np.dot(X.T, X)) * self.sigsqr
        coef = np.random.multivariate_normal([0, 0], Sigma)
        chisqr = np.random.chisquare(1)
        self.alpha += coef[0] * np.sqrt(1.0/chisqr)
        self.beta += coef[1] * np.sqrt(1.0/chisqr)
        self.sigsqr *= 0.5 * N / np.random.chisquare(0.5*N)

        # Now get the values for the mixture parameters, first do prior params
        self.mu0min = min(x)
        self.mu0max = max(x)

        mu0g = np.nan
        while not (mu0g > self.mu0min) & (mu0g < self.mu0max):
            mu0g = self.mu0 + (np.random.normal(scale=np.sqrt(np.var(x, ddof=1) / N)) /
                               np.sqrt(1.0/np.random.chisquare(1.0)))
        self.mu0 = mu0g

        # wsqr is the global scale
        self.wsqr *= 0.5 * N / np.random.chisquare(0.5 * N)

        self.usqrmax = 1.5 * np.var(x, ddof=1)
        self.usqr = 0.5 * np.var(x, ddof=1)

        self.tausqr = 0.5 * self.wsqr / np.random.chisquare(1.0, size=K)

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
                self.G[i,minind] = 1
            self.pi = np.random.dirichlet(pig+1)

        self.eta = y
        self.xi = x

    def update_xi(self): # Step 3
        # Eqn (58)
        sigma_xihat_ik_sqr = 1.0/(1.0/(self.xvar * (1.0 - self.xycorr**2))[:,np.newaxis]
                                  + self.beta**2 / self.sigsqr
                                  + 1.0/self.tausqr)
        # Eqn (57)
        sigma_xihat_i_sqr = np.sum(self.G * sigma_xihat_ik_sqr, axis=1)
        # Eqn (56)
        xihat_xy_i = self.x + self.xycov / self.yvar * (self.eta - self.y)
        # Eqn (55)
        xihat_ik = (sigma_xihat_i_sqr[:,np.newaxis]
                    * ((xihat_xy_i/self.xvar * (1.0 - self.xycorr**2))[:,np.newaxis]
                       + self.beta*(self.eta[:,np.newaxis] - self.alpha)/self.sigsqr
                       + self.mu/self.tausqr))
        # Eqn (54)
        xihat_i = np.sum(self.G * xihat_ik, axis=1)
        # Eqn (53)
        self.xi = np.random.normal(loc=xihat_i, scale=np.sqrt(sigma_xihat_i_sqr))

    def update_eta(self): # Step 4
        # Eqn (68)
        sigma_etahat_i_sqr = 1.0/(1.0/(self.yvar*(1.0 - self.xycorr**2)) + 1.0/self.sigsqr)
        # Eqn (67)
        etahat_i = (sigma_etahat_i_sqr
                    * ((self.y + self.xycov * (self.xi - self.x)/self.xvar)
                       / (self.yvar * (1.0 - self.xycorr**2))
                       + (self.alpha + self.beta * self.xi) / self.sigsqr))
        # Eqn (66)
        self.eta = np.random.normal(loc=etahat_i, scale=np.sqrt(sigma_etahat_i_sqr))

    def update_G(self): # Step 5
        # Eqn (74)
        piNp = self.pi * (1.0/np.sqrt(2.0*np.pi*self.tausqr)
                          * np.exp(-0.5 * (self.xi[:,np.newaxis] - self.mu)**2 / self.tausqr))
        q_ki = piNp / np.sum(piNp, axis=1)[:,np.newaxis]
        # Eqn (73)
        for i in xrange(self.N):
            self.G[i] = np.random.multinomial(1, q_ki[i])

    def update_alpha_beta(self): # Step 6
        X = np.ones((self.N, 2), dtype=float)
        X[:,1] = self.xi
        # Eqn (77)
        XTXinv = np.linalg.inv(np.dot(X.T, X))
        Sigma_chat = XTXinv * self.sigsqr
        # Eqn (76)
        chat = np.dot(np.dot(XTXinv, X.T), self.eta)
        # Eqn (75)
        self.alpha, self.beta = np.random.multivariate_normal(chat, Sigma_chat)

    def update_sigsqr(self): # Step 7
        # Eqn (80)
        ssqr = 1.0/(self.N-2) * np.sum((self.eta - self.alpha - self.beta * self.xi)**2)
        # Eqn (79)
        nu = self.N - 2
        # Eqn (78)
        self.sigsqr = nu * ssqr / np.random.chisquare(nu)

    def update_pi(self): # Step 8
        # Eqn (82)
        self.nk = np.sum(self.G, axis=0)
        # Eqn (81)
        self.pi = np.random.dirichlet(self.nk+1)

    def update_mu(self): # Step 9
        Gsum = np.sum(self.G * self.xi[:,np.newaxis], axis=0)
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

    def update_tausqr(self): # Step 10
        # Eqn (88)
        nu_k = self.nk + 1
        # Eqn (89)
        tk_sqr = 1.0/nu_k * (self.wsqr + np.sum(self.G*(self.xi[:,np.newaxis]-self.mu)**2, axis=0))
        # Eqn (87)
        self.tausqr = tk_sqr * nu_k / np.random.chisquare(nu_k, size=self.K)

    def update_mu0(self): # Step 11
        # Eqn (94)
        mubar = np.mean(self.mu)
        # Eqn (93)
        self.mu0 = np.random.normal(loc=mubar, scale=np.sqrt(self.usqr/self.K))

    def update_usqr(self): # Step 12
        # Eqn (96)
        nu_u = self.K + 1
        # Eqn (97)
        usqrhat = 1.0/nu_u * (self.wsqr + np.sum((self.mu - self.mu0)**2))
        usqr = np.inf
        while not usqr <= self.usqrmax:
            usqr = usqrhat * nu_u / np.random.chisquare(nu_u)
        self.usqr = usqr

    def update_wsqr(self): # Step 13
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

    def step(self):
        self.update_xi()
        self.update_eta()
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

    def run_mcmc(self, niter):
        self.initial_guess()
        self.initialize_chain(niter)
        with ProgressBar(niter) as bar:
            for i in xrange(niter):
                self.step()
                bar.update()

        return self.chain
