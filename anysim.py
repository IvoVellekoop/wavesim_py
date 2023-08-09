import numpy as np
from numpy.linalg import norm
from helmholtz_base import Helmholtz_Base
from state import State

def overlap_decay(x):
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


class AnySim:
    def __init__(self, base: Helmholtz_Base):
        self.base = base

        self.u = (np.zeros_like(self.base.b, dtype='complex_'))  # field u, initialize with 0

        self.u_list = []
        self.b_list = []
        self.restriction = []
        self.extension = []
        self.partition_of_unity = []
        self.restrict_n_partition()
        self.state = State(self.base)
        self.state.residual_initial(preconditioned_source = [(np.dot(self.extension[j], np.dot(self.partition_of_unity[j], self.base.medium_operators[j](self.base.propagator(self.b_list[j]))))) for j in range(self.base.total_domains)])

    def iterate(self):
        """ AnySim update """
        tj = [None] * self.base.total_domains
        for i in range(self.base.max_iterations):
            for j in range(self.base.total_domains):
                print('Iteration {}, sub-domain {}.'.format(i + 1, j + 1), end='\r')
                self.u_list[j] = np.dot(self.restriction[j], self.u)
                tj[j] = self.base.medium_operators[j](self.u_list[j]) + self.b_list[j]
                tj[j] = self.base.propagator(tj[j])
                tj[j] = self.base.medium_operators[j](self.u_list[j] - tj[j])  # subdomain residual

                self.state.log_subdomain_residual(j, residual_s = np.dot(self.partition_of_unity[j], tj[j]))

                self.u_list[j] = self.base.alpha * tj[j]
                # instead of this, simply update on overlapping regions?
                self.u = self.u - np.dot(self.extension[j], np.dot(self.partition_of_unity[j], self.u_list[j]))

            self.state.log_full_residual(residual_f = [( np.dot(self.extension[j], np.dot(self.partition_of_unity[j], tj[j])) ) for j in range(self.base.total_domains)])

            self.state.next(i, self.u)
            if self.state.should_terminate:
                break

        print('Simulation done (Time {} s)'.format(np.round(self.state.sim_time, 2)))
        return self.state.finalize(self.u), self.state

    def restrict_n_partition(self):
        """Construct restriction operators (self.restriction) and partition of unity operators (self.partition_of_unity)"""
        if self.base.total_domains == 1:
            self.u_list.append(self.u)
            self.b_list.append(self.base.b)
            self.restriction.append(1.)
            self.extension.append(1.)
            self.partition_of_unity.append(1.)
        else:
            ones = np.eye(self.base.domain_size[0])
            restrict0 = np.zeros((self.base.domain_size[0], self.base.n_ext[0]))
            for i in range(self.base.total_domains):
                restrict_mid = restrict0.copy()
                restrict_mid[:, i * (self.base.domain_size[0] - self.base.overlap[0]): 
                             i * (self.base.domain_size[0] - self.base.overlap[0]) + self.base.domain_size[0]] = ones
                self.restriction.append(restrict_mid)
                self.extension.append(restrict_mid.T)

            decay = overlap_decay(self.base.overlap[0])
            pou1 = np.diag(np.concatenate((np.ones(self.base.domain_size[0] - self.base.overlap[0]), np.flip(decay))))
            self.partition_of_unity.append(pou1)
            pou_mid = np.diag(
                np.concatenate((decay, np.ones(self.base.domain_size[0] - 2 * self.base.overlap[0]), np.flip(decay))))
            for _ in range(1, self.base.total_domains - 1):
                self.partition_of_unity.append(pou_mid)
            pou_end = np.diag(np.concatenate((decay, np.ones(self.base.domain_size[0] - self.base.overlap[0]))))
            self.partition_of_unity.append(pou_end)

            for j in range(self.base.total_domains):
                self.u_list.append(self.restriction[j] @ self.u)
                self.b_list.append(self.restriction[j] @ self.base.b)
