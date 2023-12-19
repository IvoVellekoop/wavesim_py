import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from state import State


class AnySim:
    def __init__(self, base: HelmholtzBase):
        self.base = base

    def iterate(self):
        """ AnySim update """

        u = np.zeros_like(self.base.s, dtype=np.complex64)  # field u, initialize with 0s
        restrict, extend = self.domain_decomp_operators()  # Construct restriction and extension operators
        state = State(self.base)
        state.init_norm = norm(np.sum(np.array([(self.map_domain(
            self.base.medium_operators[patch](self.base.propagator(
                self.map_domain(self.base.s, restrict, patch, self.base.n_dims),
                self.base.scaling[patch])), extend, patch, self.base.n_dims))
            for patch in self.base.domains_iterator]), axis=0))

        # Empty dicts of lists to store patch-wise source (s) and field (u)
        s_dict = defaultdict(list)
        u_dict = defaultdict(list)
        for patch in self.base.domains_iterator:
            # restrict full-domain source s to the patch subdomain, and apply scaling for that subdomain
            s_dict[patch] = 1j * np.sqrt(self.base.scaling[patch]) * self.map_domain(self.base.s, restrict,
                                                                                     patch, self.base.n_dims)
            # restrict full-domain field u to the patch subdomain
            u_dict[patch] = self.map_domain(u, restrict, patch, self.base.n_dims)

        for i in range(self.base.max_iterations):
            residual = 0
            for idx, patch in enumerate(self.base.domains_iterator):
                # idx gives the index in the list of domains_iterator
                # patch gives the 3-element position tuple of subdomain (e.g., (0,0,0))

                print(f'Iteration {i + 1}, sub-domain {patch}. ', end='\r')

                t1 = self.base.medium_operators[patch](u_dict[patch]) + s_dict[patch]  # B(u) + s
                # Communication between subdomains (Add the transfer_correction with previous and/or next subdomain)
                t1 = t1 - self.transfer_correction(u_dict, patch, idx)
                t1 = self.base.propagator(t1, self.base.scaling[patch])  # (L+1)^-1 t1
                t1 = self.base.medium_operators[patch](u_dict[patch] - t1)  # B(u - t1). subdomain residual

                state.log_subdomain_residual(norm(t1), patch)  # log residual for current subdomain

                u_dict[patch] = u_dict[patch] - (self.base.alpha * t1)  # update subdomain u

                # find the slice of full domain u and update
                patch_slice = self.base.patch_slice(patch)
                u[patch_slice] = u_dict[patch]

                state.log_u_iter(u, patch)  # collect u updates (store separately subdomain-wise)
                residual += self.map_domain(t1, extend, patch, self.base.n_dims)  # add subdomain residuals to get full
            state.log_full_residual(norm(residual))  # log residual for entire domain
            state.next(i)  # Check termination conditions
            if state.should_terminate:  # Proceed to next iteration or not
                break
        # return u and u_iter cropped to roi, residual arrays, and state object with information on run
        return state.finalize(u), state

    def domain_decomp_operators(self):
        """ Construct restriction and extension operators """
        restrict = [[] for _ in range(self.base.n_dims)]
        extend = [[] for _ in range(self.base.n_dims)]

        if self.base.total_domains == 1:
            [restrict[dim].append(1.) for dim in range(self.base.n_dims)]
            [extend[dim].append(1.) for dim in range(self.base.n_dims)]
        else:
            ones = np.eye(self.base.domain_size[0])
            restrict0_ = []
            [restrict0_.append(np.zeros((self.base.domain_size[dim], self.base.n_ext[dim]))) 
             for dim in range(self.base.n_dims)]
            for dim in range(self.base.n_dims):
                for patch in range(self.base.n_domains[dim]):
                    restrict_mid_ = restrict0_[dim].copy()
                    restrict_mid_[:, slice(patch * (self.base.domain_size[dim] - self.base.overlap[dim]),
                                           patch * (self.base.domain_size[dim] - self.base.overlap[dim])
                                           + self.base.domain_size[dim])] = ones
                    restrict[dim].append(restrict_mid_.T)
                    extend[dim].append(restrict_mid_)
        return restrict, extend

    @staticmethod
    def map_domain(x, mapping_operator, patch, n_dims):
        """ Map x to extended domain or restricted subdomain """
        # n_dims = np.squeeze(x).ndim
        for dim in range(n_dims):  # For applying in every dimension
            x = np.moveaxis(x, dim, -1)  # Transpose
            x = np.dot(x, mapping_operator[dim][patch[dim]])  # Apply (appropriate) mapping operator
            x = np.moveaxis(x, -1, dim)  # Transpose back
        return x.astype(np.complex64)

    def transfer_correction(self, x, current_patch, idx):
        """ Transfer correction from neighbouring subdomains to be added to t1 of current subdomain """
        x_transfer = np.zeros_like(x[current_patch], dtype=np.complex64)
        for idx_shift in [-1, +1]:  # Transfer wrt previous (-1) and next (+1) subdomain
            if 0 <= idx + idx_shift < len(self.base.domains_iterator):  # check if subdomain is on the edge
                neighbour_patch = self.base.domains_iterator[idx + idx_shift]  # get the neighbouring subdomain location
                x_neighbour = x[neighbour_patch].copy()  # get the neighbouring subdomain field
                # get the field(s) to transfer
                x_transfer += self.base.scaling[current_patch] * self.base.wrap_corr(x_neighbour, idx_shift)
                # x_transfer += self.transfer(x_neighbour, self.base.scaling[current_patch], idx_shift)
        return x_transfer
