import time
import numpy as np
from helmholtzbase import HelmholtzBase


class State(object):
    def __init__(self, base: HelmholtzBase):
        self.base = base
        self.init_norm = None
        self.subdomain_residuals = [[] for _ in range(self.base.total_domains)]
        self.full_residuals = []
        self.u_iter = []
        self.iterations = 0
        self.s1 = time.time()
        self.sim_time = 0
        self.should_terminate = False

    def log_subdomain_residual(self, j, residual_s):
        """ Normalize subdomain residual wrt preconditioned source """
        self.subdomain_residuals[j].append(residual_s/self.init_norm)

    def log_full_residual(self, residual_f):
        """ Normalize full domain residual wrt preconditioned source """
        self.full_residuals.append(residual_f / self.init_norm)

    def log_u_iter(self, i, u):
        """ Collect u_iter"""
        if self.base.n_dims > 1 and self.base.max_iterations > 500:
            if i % 10 == 0:
                self.u_iter.append(u)
        else:
            self.u_iter.append(u)

    def next(self, i):
        """ Decide whether to proceed to next iteration or not """
        self.iterations = i
        if self.full_residuals[i] < self.base.threshold_residual:
            print(f'Stopping. Iter {self.iterations + 1} '
                  f'residual {self.full_residuals[i]:.2e}<={self.base.threshold_residual}')
            self.should_terminate = True
            self.sim_time = time.time() - self.s1
            print('Simulation done (Time {} s)'.format(np.round(self.sim_time, 2)))

    def finalize(self, u):
        """ Rescale and truncate to ROI u and u_iter, and convert residual lists to arrays """
        u = self.base.Tr * u            # rescale u
        u = u[self.base.crop_to_roi]    # Truncate u to ROI
        self.u_iter = self.base.Tr.flatten() * np.array(self.u_iter)                # rescale u_iter
        self.u_iter = self.u_iter[tuple((slice(None),)) + self.base.crop_to_roi]    # truncate u_iter to ROI

        # convert residual lists to arrays and reshape if needed
        self.subdomain_residuals = np.array(self.subdomain_residuals).T
        if self.subdomain_residuals.shape[0] < self.subdomain_residuals.shape[1]:
            self.subdomain_residuals = self.subdomain_residuals.T
        self.full_residuals = np.array(self.full_residuals)

        return u
