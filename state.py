import time
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase


class State(object):
    def __init__(self, base: HelmholtzBase):
        self.base = base
        self.init_norm = None
        self.subdomain_residuals = defaultdict(list)  # Initialize empty dict of lists
        self.full_residuals = []
        self.u_iter = []
        self.iterations = self.base.max_iterations - 1
        self.should_terminate = False
        self.start_time = time.time()
        self.sim_time = 0

    def log_subdomain_residual(self, residual_s, j):
        """ Normalize subdomain residual wrt preconditioned source """
        self.subdomain_residuals[j].append(residual_s/self.init_norm)

    def log_full_residual(self, residual_f):
        """ Normalize full domain residual wrt preconditioned source """
        self.full_residuals.append(residual_f / self.init_norm)

    def log_u_iter(self, i, u):
        """ Collect u_iter"""
        if self.base.n_dims > 1 and self.base.max_iterations > 300:
            if i % 10 == 0:
                self.u_iter.append(u)
        else:
            self.u_iter.append(u)

    def next(self, i):
        """ Check termination conditions and to proceed to next iteration or not """
        if (self.full_residuals[i] <= self.base.threshold_residual
                or self.full_residuals[i] >= self.base.divergence_limit
                or i >= self.base.max_iterations - 1):
            print(f'Residual {self.full_residuals[i]:.2e}. '
                  f'Stopping at iteration {i + 1} ')
            self.should_terminate = True
            self.iterations = i
            self.sim_time = time.time() - self.start_time
            print('Simulation done (Time {} s)'.format(np.round(self.sim_time, 2)))

    def finalize(self, u):
        """ Rescale u and u_iter, and convert residual lists to arrays """
        u = self.base.Tr * u            # rescale u
        self.u_iter = self.base.Tr.flatten() * np.array(self.u_iter)                # rescale u_iter

        # convert residuals to arrays and reshape if needed
        self.subdomain_residuals = np.array(list(map(list, self.subdomain_residuals.values())))
        if self.subdomain_residuals.shape[0] < self.subdomain_residuals.shape[1]:
            self.subdomain_residuals = self.subdomain_residuals.T
        self.full_residuals = np.array(self.full_residuals)

        return u
