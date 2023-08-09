import time
import numpy as np
from numpy.linalg import norm
from helmholtz_base import Helmholtz_Base


class State(object):
    def __init__(self, base: Helmholtz_Base):
        self.base = base
        self.subdomain_residuals = [[] for _ in range(self.base.total_domains)]
        self.full_residuals = []
        self.u_iter = []
        self.iterations = 0
        self.s1 = time.time()
        self.sim_time = 0
        self.should_terminate = False

    def residual_initial(self, preconditioned_source):
        self.init_norm = norm(np.sum(np.array(preconditioned_source), axis=0))

    def log_subdomain_residual(self, j, residual_s):
        """ Normalize subdomain residual wrt preconditioned source """
        self.subdomain_residuals[j].append( norm(residual_s)/self.init_norm )

    def log_full_residual(self, residual_f):
        """ Normalize full domain residual wrt preconditioned source """
        self.full_residuals.append( norm(np.sum(np.array(residual_f), axis=0)) / self.init_norm)

    def next(self, i, u):
        self.u_iter.append(u)
        self.iterations = i
        if self.full_residuals[i] < self.base.threshold_residual:
            print(f'Stopping. Iter {self.iterations + 1} '
                  f'residual {self.full_residuals[i]:.2e}<={self.base.threshold_residual}')
            self.should_terminate = True

    def finalize(self, u):
        self.sim_time = time.time() - self.s1
        u = self.base.Tr * u

        # collect ...
        if self.base.n_dims > 1 and self.base.max_iterations > 500:
            self.u_iter = self.u_iter[::10]
        self.u_iter = self.base.Tr.flatten() * np.array(self.u_iter)
        self.u_iter = self.u_iter[tuple((slice(None),)) + self.base.crop_to_roi]

        # do something else
        self.subdomain_residuals = np.array(self.subdomain_residuals).T
        if self.subdomain_residuals.shape[0] < self.subdomain_residuals.shape[1]:
            self.subdomain_residuals = self.subdomain_residuals.T
        self.full_residuals = np.array(self.full_residuals)

        u = u[self.base.crop_to_roi]  # Truncate u to ROI
        return u
