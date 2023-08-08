import time
import numpy as np
from anysim_base import AnySimBase
from numpy.linalg import norm

def overlap_decay(x):
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


class AnySim:
    def __init__(self, base: AnySimBase):
        self.base = base

        self.residual_i = [None] * len(self.base.range_total_domains)
        self.residual = [[]] * len(self.base.range_total_domains)
        self.full_residual = []
        self.u_iter = []
        self.u_list = []
        self.b_list = []
        self.restrict = []
        self.pou = []
        self.full_norm_gb = None
        self.restrict_n_partition()
        self.residual_initial()

    # AnySim update
    def iterate(self):
        s1 = time.time()


        tj = [None for _ in self.base.range_total_domains]

        for i in range(self.base.max_iterations):
            for j in self.base.range_total_domains:
                print('Iteration {}, sub-domain {}.'.format(i + 1, j + 1), end='\r')
                if self.base.total_domains == 1:
                    self.u_list[j] = self.base.u.copy()
                else:
                    self.u_list[j] = self.restrict[j] @ self.base.u
                tj[j] = self.base.medium_operators[j](self.u_list[j]) + self.b_list[j]
                tj[j] = self.base.propagator(tj[j])
                tj[j] = self.base.medium_operators[j](self.u_list[j] - tj[j])  # subdomain residual

                self.state.log_residual(j, self.pou[j] @ tj[j])
                self.residual_i[j] = norm(self.pou[j] @ tj[j]) / self.full_norm_gb
                self.residual[j].append(self.residual_i[j])

                self.u_list[j] = self.base.alpha * tj[j]
                # instead of this, simply update on overlapping regions?
                self.base.u = self.base.u - self.restrict[j].T @ self.pou[j] @ self.u_list[j]

            self.state.next(...)
            self.state.log_residual_full(... ) self.residual_full_domain(tj, i)
            self.u_iter.append(self.base.u)

            if self.state.should_terminate():
                break
        self.base.sim_time = time.time() - s1
        print('Simulation done (Time {} s)'.format(np.round(self.base.sim_time, 2)))
        return self.finalize()

    def restrict_n_partition(self):
        """Construct restriction operators (self.restrict) and partition of unity operators (self.pou)"""
        if self.base.total_domains == 1:
            self.u_list.append(self.base.u)
            self.b_list.append(self.base.b)
            self.pou = 1
        else:
            ones = np.eye(self.base.domain_size[0])
            restrict0 = np.zeros((self.base.domain_size[0], self.base.n_ext[0]))
            for i in self.base.range_total_domains:
                restrict_mid = restrict0.copy()
                restrict_mid[:, i * (self.base.domain_size[0] - self.base.overlap[0]): 
                             i * (self.base.domain_size[0] - self.base.overlap[0]) + self.base.domain_size[0]] = ones
                self.restrict.append(restrict_mid)

            decay = overlap_decay(self.base.overlap[0])
            pou1 = np.diag(np.concatenate((np.ones(self.base.domain_size[0] - self.base.overlap[0]), np.flip(decay))))
            self.pou.append(pou1)
            pou_mid = np.diag(
                np.concatenate((decay, np.ones(self.base.domain_size[0] - 2 * self.base.overlap[0]), np.flip(decay))))
            for _ in range(1, self.base.total_domains - 1):
                self.pou.append(pou_mid)
            pou_end = np.diag(np.concatenate((decay, np.ones(self.base.domain_size[0] - self.base.overlap[0]))))
            self.pou.append(pou_end)

            for j in self.base.range_total_domains:
                self.u_list.append(self.restrict[j] @ self.base.u)
                self.b_list.append(self.restrict[j] @ self.base.b)

    def residual_initial(self):
        # Normalize subdomain residual wrt preconditioned source
        self.full_norm_gb = norm(np.sum(np.array(
                [(self.restrict[j].T@self.pou[j]@self.base.medium_operators[j](self.base.propagator(self.b_list[j])))
                 for j in self.base.range_total_domains]), axis=0))

    # Residual collection and checking
    def residual_subdomain(self, tj, j):
        # To Normalize subdomain residual wrt preconditioned source

    def residual_full_domain(self, tj, i):
        if self.base.total_domains == 1:
            full_nr = np.linalg.norm(tj[0])
        else:
            full_nr = np.linalg.norm(
                np.sum(np.array([(self.restrict[j].T @ self.pou[j] @ tj[j])
                                 for j in self.base.range_total_domains]), axis=0))
        self.full_residual.append(full_nr / self.full_norm_gb)
        if self.full_residual[i] < self.base.threshold_residual:
            self.base.iterations = i
            print(f'Stopping. Iter {self.base.iterations + 1} '
                  f'residual {self.full_residual[i]:.2e}<={self.base.threshold_residual}')
            self.breaker = True
        self.base.residual_i = self.full_residual[i]

    def finalize(self):
        self.base.u = self.base.Tr * self.base.u

        # collect ...
        if self.base.n_dims > 1 and self.base.max_iterations > 500:
            self.u_iter = self.u_iter[::10]
        u_iter = self.base.Tr.flatten() * np.array(self.u_iter)
        u_iter = u_iter[tuple((slice(None),)) + self.base.crop_to_roi]

        # do something else
        self.base.residual = np.array(self.residual).T
        if self.base.residual.shape[0] < self.base.residual.shape[1]:
            self.base.residual = self.base.residual.T
        self.base.full_residual = np.array(self.full_residual)

        self.base.u = self.base.u[self.base.crop_to_roi]  # Truncate u to ROI
        return u_iter

