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
        self.restriction = [[] for _ in range(self.base.n_dims)]
        self.extension = [[] for _ in range(self.base.n_dims)]
        self.partition_of_unity = 1.
        self.domain_decomp_operators()  # Construct restriction, extension, and partition of unity operators
        self.state = State(self.base)
        self.state.init_norm = norm(np.sum(np.array([(self.extend(self.base.medium_operators[j]
                                                                  (self.base.propagator(self.restrict(self.base.s, j))),
                                                                  j))
                                                     for j in self.base.domains_iterator]), axis=0))

    def alt_schwarz_iterate(self):
        """ Alternating Schwarz. Trial implementation of direct differential version. 
            Gives the same results as the original iterate function implementation """
        # Initialize empty dict of lists
        u_temp = defaultdict(list)
        s_temp = defaultdict(list)
        for j in self.base.domains_iterator:
            s_temp[j] = self.restrict(self.base.s, j)  # restrict full-domain source s to the j-th subdomain

        pre_ = tuple([slice(None, 1) for _ in range(self.base.n_dims)])
        post_ = tuple([slice(-1, None) for _ in range(self.base.n_dims)])

        for i in range(self.base.max_iterations):
            residual = 0
            for idx, j in enumerate(self.base.domains_iterator):
                # j gives the 3-element position tuple of subdomain (e.g., (0,0,0))
                print(f'Iteration {i + 1}, sub-domain {j}. ', end='\r')
                u_temp[j] = self.restrict(self.u, j)  # restrict full-domain field u to the j-th subdomain
                tj = self.base.medium_operators[j](u_temp[j]) + s_temp[j]  # B(u) + s
                tj = self.base.propagator(tj)  # (L+1)^-1 (tj)
                tj = self.base.medium_operators[j](u_temp[j] - tj)  # B(u - tj). subdomain residual
                self.state.log_subdomain_residual(norm(np.dot(self.partition_of_unity, tj)), j)
                u_temp[j] = u_temp[j] - self.base.alpha * tj
                try:
                    u_temp[j][pre_] = u_temp[self.base.domains_iterator[idx - 1]][post_]
                    u_temp[j][post_] = u_temp[self.base.domains_iterator[idx + 1]][pre_]
                except ValueError:
                    pass

                tj = self.extend(tj, j)  # extend j-th subdomain tj to full-domain
                self.u = self.u - self.base.alpha * tj  # Update full-domain u. u = u - alpha*tj. Update overlaps only?
                self.state.log_u_iter(self.u, j)  # collect u subdomain updates
                residual += tj  # collect all subdomain residuals to update full residual
            self.state.log_full_residual(norm(residual))
            self.state.next(i)  # Check termination conditions
            if self.state.should_terminate:  # Proceed to next iteration or not
                break
        # return u and u_iter cropped to roi, residual arrays, and state object with information on run
        return self.state.finalize(self.u), self.state

    def iterate(self):
        """ AnySim update """
        # Initialize empty dict of lists
        s_temp = defaultdict(list)
        for j in self.base.domains_iterator:
            s_temp[j] = self.restrict(self.base.s, j)  # restrict full-domain source s to the j-th subdomain

        for i in range(self.base.max_iterations):
            residual = 0
            for j in self.base.domains_iterator:  # j gives the 3-element position tuple of subdomain (e.g., (0,0,0))
                print(f'Iteration {i + 1}, sub-domain {j}. ', end='\r')
                u_temp = self.restrict(self.u, j)  # restrict full-domain field u to the j-th subdomain
                tj = self.base.medium_operators[j](u_temp) + s_temp[j]  # B(u) + s
                tj = self.base.propagator(tj)  # (L+1)^-1 tj
                tj = self.base.medium_operators[j](u_temp - tj)  # B(u - tj). subdomain residual
                self.state.log_subdomain_residual(norm(np.dot(self.partition_of_unity, tj)), j)
                tj = self.extend(tj, j)  # extend j-th subdomain tj to full-domain
                self.u = self.u - self.base.alpha * tj  # Update full-domain u. u = u - alpha*tj. Update overlaps only?
                self.state.log_u_iter(self.u, j)  # collect u subdomain updates
                residual += tj  # collect all subdomain residuals to update full residual
            self.state.log_full_residual(norm(residual))
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
                for j in range(self.base.n_domains[i]):
                    restrict_mid_ = restrict0_[i].copy()
                    restrict_mid_[:, slice(j * (self.base.domain_size[i] - self.base.overlap[i]),
                                           j * (self.base.domain_size[i] - self.base.overlap[i])
                                           + self.base.domain_size[i])] = ones
                    self.restriction[i].append(restrict_mid_.T)
                    self.extension[i].append(restrict_mid_)

            decay = overlap_decay(self.base.overlap[0])
            self.partition_of_unity = np.diag(np.concatenate((decay, np.ones(self.base.domain_size[0] - 2 *
                                                                             self.base.overlap[0]), np.flip(decay)))).T

    def restrict(self, a, j):
        """ Restrict full-domain 'a' to the j-th subdomain """
        a_ = a.copy()
        for i in range(self.base.n_dims):  # For applying in every dimension
            a_ = np.moveaxis(a_, i, -1)  # Transpose
            a_ = np.dot(a_, self.restriction[i][j[i]])  # Apply (appropriate) restriction operator
            a_ = np.moveaxis(a_, -1, i)  # Transpose back
        return a_.astype(np.csingle)

    def extend(self, a, j):
        """ Extend j-th subdomain 'a' to full-domain """
        a_ = a.copy()
        for i in range(self.base.n_dims):  # For applying in every dimension
            a_ = np.moveaxis(a_, i, -1)  # Transpose
            a_ = np.dot(a_, self.partition_of_unity)  # Apply partition of unity to avoid redundancy
            a_ = np.dot(a_, self.extension[i][j[i]])  # Apply (appropriate) extension operator
            a_ = np.moveaxis(a_, -1, i)  # Transpose back
        return a_.astype(np.csingle)
