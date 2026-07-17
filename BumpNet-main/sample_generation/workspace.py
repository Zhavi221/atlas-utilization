import numpy as np
import scipy
import yaml

class Workspace:

    def __init__(self, config=None, seed=None):

        ### settings
        self.config = config
        self.two_sided = True
        self.tol = 1e-4
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        ### observable
        self.bin_edges = None
        self.bin_centers = None
        self.bin_widths = None

        ### functions
        self.hypo_sig_func = None
        self.inj_sig_func = None
        self.bkg_func = None

        ### histograms
        self.data = np.array([])
        self.sig_hist  = np.array([])
        self.bkg_hist  = np.array([])

        ### parameters
        self.mu = 0.0  #signal strength
        self.M_hypo  = 0.0  #signal hypothesis mass
        self.W_hypo  = 0.0  #signal hypothesis width
        self.W_hypo_bins = 1.0 #signal hypothesis width in bins
        self.M_inj  = 0.0  #injected signal mass
        self.W_inj  = 0.0  #injected signal width
        self.W_inj_bins = 1.0 #injected signal width in bins

        ### read config
        if self.config is not None:
            self.read_config()

        self.update()

    def read_config(self):

        ### Takes path to yaml file or a dictionary
        if isinstance(self.config, str):
            self.config = yaml.safe_load(open(self.config,'r'))

        for k,v in self.config.items():
            if hasattr(self,k):
                setattr(self,k,v)

    ### update bin quantities and update sig_hist
    def update(self):

        ### read bin edges from csv file
        if isinstance(self.bin_edges,str):
            self.bin_edges = np.genfromtxt(self.bin_edges, delimiter=',')

        ### calculate bin widths, centers based on bin edges
        if self.bin_edges is not None:
            if self.bin_widths is None:
                self.bin_widths = np.diff(self.bin_edges)
            if self.bin_centers is None:
                bin_edges = np.asarray(self.bin_edges, dtype=float)
                bin_widths = np.asarray(self.bin_widths, dtype=float)
                self.bin_centers = bin_edges[:-1] + bin_widths / 2


        ### read histogram from csv file
        if isinstance(self.bkg_hist, str):
            self.bkg_hist = np.genfromtxt(self.bkg_hist, delimiter=',')

        ### update hypo_sig_hist based on current M_hypo, W_hypo
        if self.hypo_sig_func is not None:
            self.hypo_sig_hist = self.asimov(self.hypo_sig_func,params=(self.M_hypo,self.W_hypo))

        ### update inj_sig_hist based on current M_inj, W_inj
        if self.inj_sig_func is not None:
            self.inj_sig_hist = self.asimov(self.inj_sig_func,params=(self.M_inj,self.W_inj))

    ### Takes pdf (function) and returns (normalized?) histogram where each bin is the integral of the pdf over that bin
    ### The integral is approximated by the value of the pdf at the bin center times the bin width (should be accurate for small bin widths)
    def asimov(self, pdf, params, normalize=True):

        assert callable(pdf)

        hist = pdf(self.bin_centers, *params) * self.bin_widths
        if normalize:
            hist = hist/np.sum(hist)

        return hist

    ### Takes histogram (pdf_hist) and draws new histogram by poisson fluctuating each bin
    ### By default it draws from the bkg + inj_sig histogram of itself
    def sample(self, pdf_hist=None, atleast=1):

        if pdf_hist is None:
            pdf_hist = self.bkg_hist + self.mu * self.inj_sig_hist

        pdf_hist = np.asarray(pdf_hist, dtype=np.float64)
        toy = self.rng.poisson(pdf_hist)

        return toy

    ### Negative log-likehood
    def nll(self, mu=None, Nobs=None, Nexp=None, drop_const_term=True):

        if mu is None:
            mu = self.mu

        if Nobs is None:
            Nobs = self.data

        if Nexp is None:
            Nexp = np.clip(self.bkg_hist + mu * self.hypo_sig_hist,1e-12,None) # avoid log(0)

        Nobs = np.asarray(Nobs, dtype=float)
        Nexp = np.asarray(Nexp, dtype=float)

        nll = -np.sum( Nobs * np.log(Nexp) - Nexp)

        ### This term does not matter if you are calculating differences of nll values since Nobs is same under both hypotheses
        if drop_const_term == False:
            nll += np.sum(np.log(np.math.factorial(Nobs)))

        return nll

    ### Calculate q0 (negative log-likelihood ratio) and then take square-root to get z0
    ### mu_hat is needed only to check the sign of the best-fit signal strength, in case of a two-sided test
    def calc_z(self,nll_null,nll_alt,mu_hat=None):

        if mu_hat is None:
            mu_hat = self.mu

        z0, q0 = 0, 0

        if nll_alt > nll_null:
            print(f'nll_alt is not the minimal value, so we have a problem')
        
        if mu_hat >= 0:
            q0 = -2 * (nll_alt - nll_null)
            z0 = np.sqrt(q0)
        elif self.two_sided:
            q0 =  2 * (nll_alt - nll_null)
            z0 = -1*np.sqrt(-1*q0)

        return z0

    ### Fit mu to self.data by minimizing negative log-likelihood
    def fit(self):

        # Minimization using the Brent algorithm, which is a type of improved golden-section algorithm which guarantees 
        # finding a minimum between 3 bracket points. If fit does not converge, it's because it could not find a 3rd bracket
        # point. It might be good to add more iterations to bracketing using the scipy bracket method. See here:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bracket.html
        
        brack = np.sum(np.abs(self.data - self.bkg_hist))
        res = scipy.optimize.minimize_scalar(self.nll, bracket=(0,brack), tol=self.tol, options={'maxiter':500000})

        self.mu = res.x

        if not res.success:
            print('Fit did not converge!')
            print(f'termination message:')
            print(res.message)

        return {'mu_hat': res.x, 'nll': res.fun, 'success': res.success}

    ### Hypothesis test: calculate z0 by calculating nll under null and alternative hypotheses
    ### If mu_inj is given, assume the data are bkg + mu_inj*sig
    def hypo_test(self, mu_inj=None):

        if mu_inj is not None:
            self.data = self.bkg_hist + mu_inj * self.inj_sig_hist

        self.mu = 0
        nll_null = self.nll()
        fit_result = self.fit() #self.mu gets updated
        nll_alt  = fit_result['nll']

        z0 = self.calc_z(nll_null, nll_alt)

        return {'z0': z0, 'mu_hat': self.mu,  'nll_null': nll_null, 'nll_alt': nll_alt}

    ### Find the mu_inj that gives z0 = wanted_z
    def calc_mu_for_wanted_z(self, wanted_z):

        fun = lambda mu_inj: abs(self.hypo_test(mu_inj=mu_inj)['z0'] - wanted_z)

        mu_bracks = (0, 1e6)
        mu_for_wanted_z = scipy.optimize.golden(fun, brack=mu_bracks, tol=self.tol)

        try:
            assert abs(self.hypo_test(mu_inj=mu_for_wanted_z)['z0'] - wanted_z) < 1e-2
        except:
            print(f'Could not assert optimization, difference is {abs(self.hypo_test(mu_inj=mu_for_wanted_z)["z0"] - wanted_z):.2e}. Skipping this sample.')
            return None

        return mu_for_wanted_z

    ### For each bin center, do a hypothesis test with signal centered there
    def z_scan(self):

        # Scan over M in bin centers and test significance of signal centered there
        z_scan = []
        for j, x_j in enumerate(self.bin_centers):
            self.M_hypo = x_j
            self.W_hypo = self.bin_widths[j]*self.W_hypo_bins
            self.update() ### update hypo_sig_hist
            z0 = self.hypo_test()['z0']
            
            if z0==0 or np.isnan(z0):
                print(f'mu_hat = {self.mu}')
                print(f'data at the point: {self.data[j]}')
                print(f'bkg at the point: {self.bkg_hist[j]}')

                if z0 == 0:
                    print('z0 is exactly 0')
                if np.isnan(z0):
                    print(f'z0 is nan, something has gone wrong in the calculation and this shouldnt be used. Skipping this histogram.')
                    return 'skip'
            
            z_scan.append(z0)

        return np.array(z_scan)
