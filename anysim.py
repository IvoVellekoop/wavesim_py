import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from state import State


def overlap_decay(x):
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


class AnySim:
    def __init__(self, base: HelmholtzBase):
        self.base = base
        self.u = np.zeros_like(self.base.s, dtype=np.csingle)  # field u, initialize with 0s
        self.u_dict = defaultdict(list)  # Empty dict of lists to store u's in each patch
        self.restriction = [[] for _ in range(self.base.n_dims)]
        self.extension = [[] for _ in range(self.base.n_dims)]
        self.domain_decomp_operators()  # Construct restriction and extension operators
        self.state = State(self.base)
        self.state.init_norm = norm(np.sum(np.array([(self.extend(self.base.medium_operators[patch]
                                                                  (self.base.propagator(self.restrict(self.base.s,
                                                                   patch), self.base.scaling[patch])), patch))
                                                     for patch in self.base.domains_iterator]), axis=0))

    def iterate(self):
        """ AnySim update """
        s_dict = defaultdict(list)  # Empty dict of lists to store source term s in each patch
        for patch in self.base.domains_iterator:
            # restrict full-domain source s to the patch subdomain, and apply scaling for that subdomain
            s_dict[patch] = self.base.Tl[patch] * self.restrict(self.base.s, patch)
            # restrict full-domain field u to the patch subdomain
            self.u_dict[patch] = self.restrict(self.u, patch)

        for i in range(self.base.max_iterations):
            residual = 0
            for idx, patch in enumerate(self.base.domains_iterator):
                # idx gives the index in the list of domains_iterator
                # patch gives the 3-element position tuple of subdomain (e.g., (0,0,0))
                print(f'Iteration {i + 1}, sub-domain {patch}. ', end='\r')
                t1 = self.base.medium_operators[patch](self.u_dict[patch]) + s_dict[patch]  # B(u) + s
                t1 = self.base.propagator(t1, self.base.scaling[patch])  # (L+1)^-1 t1
                t1 = self.base.medium_operators[patch](self.u_dict[patch] - t1)  # B(u - t1). subdomain residual
                self.state.log_subdomain_residual(norm(t1), patch)  # log residual for this subdomain

                # Communication between subdomains (Add the transfer_correction with previous and/or next subdomain)
                t1 = t1 + self.transfer_correction(patch, idx)
                self.u_dict[patch] = self.u_dict[patch] - (self.base.alpha * t1)  # update subdomain u

                # find the slice of full domain u to update
                patch_slice = tuple([slice(patch[j]*(self.base.domain_size[j]-self.base.overlap[j]),
                                    patch[j]*(self.base.domain_size[j]-self.base.overlap[j])+self.base.domain_size[j])
                                    for j in range(self.base.n_dims)])
                self.u[patch_slice] = self.u_dict[patch]

                self.state.log_u_iter(self.u, patch)  # collect u updates (store separately subdomain-wise)
                residual += self.extend(t1, patch)  # collect all subdomain residuals to update full residual
            self.state.log_full_residual(norm(residual))  # log residual for entire domain
            self.state.next(i)  # Check termination conditions
            if self.state.should_terminate:  # Proceed to next iteration or not
                break
        # return u and u_iter cropped to roi, residual arrays, and state object with information on run
        return self.state.finalize(self.u), self.state

    def domain_decomp_operators(self):
        """ Construct restriction, extension, and partition of unity operators """
        if self.base.total_domains == 1:
            [self.restriction[i].append(1.) for i in range(self.base.n_dims)]
            [self.extension[i].append(1.) for i in range(self.base.n_dims)]
        else:
            ones = np.eye(self.base.domain_size[0])
            restrict0_ = []
            [restrict0_.append(np.zeros((self.base.domain_size[i], self.base.n_ext[i])))
             for i in range(self.base.n_dims)]
            for i in range(self.base.n_dims):
                for patch in range(self.base.n_domains[i]):
                    restrict_mid_ = restrict0_[i].copy()
                    restrict_mid_[:, slice(patch * (self.base.domain_size[i] - self.base.overlap[i]),
                                           patch * (self.base.domain_size[i] - self.base.overlap[i])
                                           + self.base.domain_size[i])] = ones
                    self.restriction[i].append(restrict_mid_.T)
                    self.extension[i].append(restrict_mid_)

    def restrict(self, x, patch):
        """ Restrict full-domain 'x' to the patch subdomain """
        for i in range(self.base.n_dims):  # For applying in every dimension
            x = np.moveaxis(x, i, -1)  # Transpose
            x = np.dot(x, self.restriction[i][patch[i]])  # Apply (appropriate) restriction operator
            x = np.moveaxis(x, -1, i)  # Transpose back
        return x.astype(np.csingle)

    def extend(self, x, patch):
        """ Extend patch subdomain 'x' to full-domain """
        for i in range(self.base.n_dims):  # For applying in every dimension
            x = np.moveaxis(x, i, -1)  # Transpose
            x = np.dot(x, self.extension[i][patch[i]])  # Apply (appropriate) extension operator
            x = np.moveaxis(x, -1, i)  # Transpose back
        return x.astype(np.csingle)

    def transfer_correction(self, current_patch, idx):
        u_transfer = np.zeros_like(self.u_dict[current_patch], dtype=np.csingle)
        for idx_shift in [-1, +1]:  # Transfer wrt previous (-1) and next (+1) subdomain
            if 0 <= idx + idx_shift < len(self.base.domains_iterator):
                neighbour_patch = self.base.domains_iterator[idx + idx_shift]  # get the neighbouring subdomain location
                u_tr_temp = self.u_dict[neighbour_patch].copy()  # get the neighbouring subdomain field
                # get the field(s) to transfer
                u_transfer = u_transfer + self.base.transfer(u_tr_temp, self.base.scaling[current_patch], idx_shift)
        return u_transfer
