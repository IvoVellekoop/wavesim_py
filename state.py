import time
import numpy as np
from torch import tensor
from torch.linalg import norm
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from utilities import squeeze_


class State(object):
    def __init__(self, base: HelmholtzBase):
        self.base = base
        self.init_norm = None
        self.subdomain_residuals = defaultdict(list)  # Initialize empty dict of lists
        self.full_residuals = []
        self.u_iter = defaultdict(list)  # Initialize empty dict of lists
        self.iterations = self.base.max_iterations
        self.should_terminate = False
        self.start_time = time.time()
        self.sim_time = 0

    def next(self, t_dict, i):
        """ Log residuals and Check termination conditions to proceed to next iteration or not """
        subdomain_residuals = []
        for patch in self.base.domains_iterator:  # patch gives the 3-element position tuple of subdomain
            # compute and log residual for patch
            subdomain_residual = norm(t_dict[patch]).to(self.base.device) / self.init_norm
            self.subdomain_residuals[patch].append(subdomain_residual)
            subdomain_residuals.append(subdomain_residual)  # collect all subdomain residuals
        subdomain_residuals = tensor(subdomain_residuals, device=self.base.device)
        full_residual = norm(subdomain_residuals)
        self.full_residuals.append(full_residual)  # log residual for entire domain

        # Check termination conditions
        if (subdomain_residuals <= self.base.threshold_residual).all() \
                or (subdomain_residuals >= self.base.divergence_limit).all() \
                or full_residual <= self.base.threshold_residual \
                or full_residual >= self.base.divergence_limit \
                or i >= self.base.max_iterations - 1:
            print(f'Residual {full_residual:.2e}. Stopping at iteration {i + 1}')
            self.should_terminate = True
            self.iterations = i + 1
            self.sim_time = time.time() - self.start_time
            print('Simulation done (Time {} s)'.format(np.round(self.sim_time, 2)))

    def finalize(self, u):
        """ Rescale u and crop to ROI, and convert residual lists to arrays """
        for patch in self.base.domains_iterator:  # patch gives 3-element position tuple of subdomain (e.g., (0,0,0))
            current_patch = tuple([slice(patch[j] * self.base.domain_size[j],
                                         patch[j] * self.base.domain_size[j] + self.base.domain_size[j])
                                   for j in range(self.base.n_dims)])
            u[current_patch] = np.sqrt(self.base.scaling[patch]) * u[current_patch]  # rescale u
        u = u[self.base.crop2roi]  # Crop u to ROI

        # convert residuals to arrays and reshape if needed
        self.subdomain_residuals = tensor(list(map(list, self.subdomain_residuals.values())))
        if self.subdomain_residuals.shape[0] < self.subdomain_residuals.shape[1]:
            self.subdomain_residuals = self.subdomain_residuals.T
        self.full_residuals = tensor(self.full_residuals)
        return squeeze_(u)
