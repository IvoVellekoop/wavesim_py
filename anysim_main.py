import time
import numpy as np


def check_input_dims(a):
    for _ in range(3 - a.ndim):
        a = np.expand_dims(a, axis=-1)
    return a


def boundary_(x):
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def overlap_decay(x):
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


class AnySim:
    def __init__(self,
                 n=np.ones((1, 1, 1)),  # Refractive index distribution
                 wavelength=1.,  # Wavelength in um (micron)
                 ppw=4,  # points per wavelength
                 boundary_widths=(0, 0, 0),  # Width of absorbing boundaries
                 source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
                 n_domains=(1, 1, 1),  # Number of subdomains to decompose into, in each dimension
                 overlap=(0, 0, 0),  # Overlap between subdomains in each dimension
                 wrap_correction=None,  # Wrap-around correction. None, 'L_Omega' or 'L_corr'
                 cp=20,  # Corner points to include in case of 'L_corr' wrap-around correction
                 max_iterations=int(1.1e+3)):  # Maximum number iterations

        self.n = check_input_dims(n)
        self.n_dims = (np.squeeze(self.n)).ndim  # Number of dimensions in problem
        self.n_roi = np.array(self.n.shape)  # Num of points in ROI (Region of Interest)
        self.boundary_widths = self.check_input_len(boundary_widths, 0)
        self.bw_pre = np.floor(self.boundary_widths)
        self.bw_post = np.ceil(self.boundary_widths)
        self.wavelength = wavelength  # Wavelength in um (micron)
        self.ppw = ppw  # points per wavelength
        self.k0 = (1. * 2. * np.pi) / self.wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
        self.pixel_size = self.wavelength / self.ppw  # Grid pixel size in um (micron)
        self.n_ext = self.n_roi + self.bw_pre + self.bw_post
        self.b = check_input_dims(source)
        self.max_domain_size = 500
        if n_domains is None:
            self.n_domains = self.n_ext // self.max_domain_size  # n_ext/Max permissible size of sub-domain
        else:
            self.n_domains = self.check_input_len(n_domains,
                                                  1)  # Number of subdomains to decompose into in each dimension

        self.overlap = self.check_input_len(overlap, 0)  # Overlap between subdomains in each dimension

        if (self.n_domains == 1).all():  # If 1 domain, implies no domain decomposition
            self.domain_size = self.n_ext.copy()
        else:  # Else, domain decomposition
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains
            self.check_domain_size_max()
            self.check_domain_size_same()
            self.check_domain_size_int()

        self.bw_pre = self.bw_pre.astype(int)
        self.bw_post = self.bw_post.astype(int)
        self.n_ext = self.n_ext.astype(int)
        self.n_domains = self.n_domains
        self.domain_size[self.n_dims:] = 0
        self.domain_size = self.domain_size.astype(int)

        self.total_domains = np.prod(self.n_domains)
        self.range_total_domains = range(self.total_domains)

        self.medium_operators = []
        self.n_subdomain = None
        self.v0 = None
        self.v = None
        self.scaling = None
        self.Tl = None
        self.Tr = None
        self.n_fast_conv = None
        self.propagator = None
        self.u = None
        self.residual = None
        self.residual_i = None
        self.full_residual = None
        self.sim_time = None

        self.crop_to_roi = tuple([slice(self.bw_pre[i], -self.bw_post[i]) for i in range(self.n_dims)])

        self.wrap_correction = wrap_correction  # None, 'L_omega', OR 'L_corr'
        self.cp = cp  # number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

        self.max_iterations = max_iterations
        self.iterations = self.max_iterations - 1
        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]
        self.threshold_residual = 1.e-6
        self.iter_step = 1

    def setup_operators_n_initialize(self):  # function that calls all the other 'main' functions
        # Make operators: Medium b = 1 - v, and Propagator (L+1)^(-1)
        v_raw = self.k0 ** 2 * self.n ** 2
        v_raw = np.pad(v_raw, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])), mode='edge')

        if self.total_domains == 1:
            self.medium_operators.append(self.make_medium(v_raw, 'Both'))
        else:
            self.medium_operators.append(
                self.make_medium(v_raw[tuple([slice(0, self.domain_size[i]) for i in range(self.n_dims)])], 'left'))
            for d in range(1, self.total_domains - 1):
                self.medium_operators.append(self.make_medium(v_raw[tuple([slice(
                    d * (self.domain_size[i] - self.overlap[i]),
                    d * (self.domain_size[i] - self.overlap[i]) + self.domain_size[i]) for i in range(self.n_dims)])],
                                                              None))
            self.medium_operators.append(
                self.make_medium(v_raw[tuple([slice(-self.domain_size[i], None) for i in range(self.n_dims)])],
                                 'right'))
        self.make_propagator()

        # Scale the source term (and pad if boundaries)
        self.b = self.Tl * np.squeeze(
            np.pad(self.b, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])),
                   mode='constant'))  # source term y
        self.u = (np.zeros_like(self.b, dtype='complex_'))  # field u, initialize with 0

    def make_medium(self, v_raw, which_end=None):  # Medium B = 1 - V
        self.n_subdomain = v_raw.shape
        # give tiny non-zero minimum value to prevent division by zero in homogeneous media
        mu_min = (10.0 / (self.boundary_widths[:self.n_dims] * self.pixel_size)) if (
                self.boundary_widths != 0).any() else 0
        mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(self.n_subdomain[:self.n_dims]) * self.pixel_size)))
        v_min = np.imag((self.k0 + 1j * np.max(mu_min)) ** 2)
        v_max = 0.95
        self.v0 = (np.max(np.real(v_raw)) + np.min(np.real(v_raw))) / 2
        self.v0 = self.v0 + 1j * v_min
        self.v = -1j * (v_raw - self.v0)

        # if self.wrap_correction == 'L_corr':
        # 	p = 2*np.pi*np.fft.fftfreq(self.n_subdomain, self.pixel_size)
        # 	Lw_p = p**2
        # 	Lw = Finv @ np.diag(Lw_p.flatten()) @ F
        # 	L_corr = -np.real(Lw)
        # # Keep only upper and lower triangular corners of -Lw
        # 	L_corr[:-self.cp,:-self.cp] = 0; L_corr[self.cp:,self.cp:] = 0
        # 	self.v = self.v + 1j*L_corr

        self.scaling = v_max / np.max(np.abs(self.v))
        self.v = self.scaling * self.v
        self.Tr = np.sqrt(self.scaling)
        self.Tl = 1j * self.Tr

        b = 1 - self.v
        if which_end is None:
            b = np.squeeze(b)
        else:
            if which_end == 'Both':
                n_roi = self.n_roi
            elif which_end == 'left':
                n_roi = self.n_subdomain - self.bw_pre
            elif which_end == 'right':
                n_roi = self.n_subdomain - self.bw_post
            else:
                n_roi = self.n_subdomain - self.bw_pre - self.bw_post
            b = np.squeeze(self.pad_func(m=b, n_roi=n_roi, which_end=which_end).astype('complex_'))
        medium = lambda x: b * x
        return medium

    def make_propagator(self):  # (L+1)^(-1)
        if self.wrap_correction == 'L_omega':
            self.n_fast_conv = self.n_subdomain * 10
        else:
            self.n_fast_conv = self.n_subdomain

        l_p = (2 * np.pi * np.fft.fftfreq(self.n_fast_conv[0], self.pixel_size)) ** 2
        for d in range(1, self.n_dims):
            l_p = np.expand_dims(l_p, axis=-1) + np.expand_dims(
                (2 * np.pi * np.fft.fftfreq(self.n_fast_conv[d], self.pixel_size)) ** 2, axis=0)
        l_p = 1j * self.scaling * (l_p - self.v0)
        l_p_inv = np.squeeze(1 / (l_p + 1))
        if self.wrap_correction == 'L_omega':
            self.propagator = lambda x: (np.fft.ifftn(
                l_p_inv * np.fft.fftn(np.pad(x, (0, self.n_fast_conv - self.n_subdomain)))))[:self.n_subdomain]
        else:
            self.propagator = lambda x: (np.fft.ifftn(l_p_inv * np.fft.fftn(x)))

    # AnySim update
    def iterate(self):
        s1 = time.time()

        # Construct restriction operators (restrict) and partition of unity operators (pou)
        u_list = []
        b_list = []
        restrict = []
        pou = []
        if self.total_domains == 1:
            u_list.append(self.u)
            b_list.append(self.b)
            # To Normalize subdomain residual wrt preconditioned source
            full_norm_gb = np.linalg.norm(self.medium_operators[0](self.propagator(b_list[0])))
        else:
            ones = np.eye(self.domain_size[0])
            restrict0 = np.zeros((self.domain_size[0], self.n_ext[0]))
            for i in self.range_total_domains:
                restrict_mid = restrict0.copy()
                restrict_mid[:, i * (self.domain_size[0] - self.overlap[0]): 
                             i * (self.domain_size[0] - self.overlap[0]) + self.domain_size[0]] = ones
                restrict.append(restrict_mid)

            decay = overlap_decay(self.overlap[0])
            pou1 = np.diag(np.concatenate((np.ones(self.domain_size[0] - self.overlap[0]), np.flip(decay))))
            pou.append(pou1)
            pou_mid = np.diag(
                np.concatenate((decay, np.ones(self.domain_size[0] - 2 * self.overlap[0]), np.flip(decay))))
            for _ in range(1, self.total_domains - 1):
                pou.append(pou_mid)
            pou_end = np.diag(np.concatenate((decay, np.ones(self.domain_size[0] - self.overlap[0]))))
            pou.append(pou_end)

            for j in self.range_total_domains:
                u_list.append(restrict[j] @ self.u)
                b_list.append(restrict[j] @ self.b)

            # To Normalize subdomain residual wrt preconditioned source
            full_norm_gb = np.linalg.norm(np.sum(np.array(
                [(restrict[j].T @ pou[j] @ self.medium_operators[j](self.propagator(b_list[j]))) for j in
                 self.range_total_domains]), axis=0))

        tj = [None for _ in self.range_total_domains]
        residual_i = [None for _ in self.range_total_domains]
        residual = [[] for _ in self.range_total_domains]
        # u_iter = []
        breaker = False

        full_residual = []

        for i in range(self.max_iterations):
            for j in self.range_total_domains:
                print('Iteration {}, sub-domain {}.'.format(i + 1, j + 1), end='\r')
                # Main update START ---
                # if i % self.iter_step == 0:
                if self.total_domains == 1:
                    u_list[j] = self.u.copy()
                else:
                    u_list[j] = restrict[j] @ self.u
                tj[j] = self.medium_operators[j](u_list[j]) + b_list[j]
                tj[j] = self.propagator(tj[j])
                tj[j] = self.medium_operators[j](u_list[j] - tj[j])  # subdomain residual
                # --- continued below ---

                ''' Residual collection and checking '''
                # To Normalize subdomain residual wrt preconditioned source
                if self.total_domains == 1:
                    nr = np.linalg.norm(tj[j])
                else:
                    nr = np.linalg.norm(pou[j] @ tj[j])

                residual_i[j] = nr / full_norm_gb
                residual[j].append(residual_i[j])

                # --- continued below ---
                u_list[j] = self.alpha * tj[j]
                # if i % self.iter_step == 0:
                if self.total_domains == 1:
                    self.u = self.u - u_list[j]  # instead of this, simply update on overlapping regions?
                else:
                    self.u = self.u - restrict[j].T @ pou[j] @ u_list[j]  # instead, update on overlapping regions?
            # Main update END ---

            # Full Residual
            if self.total_domains == 1:
                full_nr = np.linalg.norm(tj[0])
            else:
                full_nr = np.linalg.norm(
                    np.sum(np.array([(restrict[j].T @ pou[j] @ tj[j]) for j in self.range_total_domains]), axis=0))

            full_residual.append(full_nr / full_norm_gb)
            if full_residual[i] < self.threshold_residual:
                self.iterations = i
                print(f'Stopping. Iter {self.iterations + 1} '
                      f'residual {full_residual[i]:.2e}<={self.threshold_residual}')
                breaker = True
                break
            self.residual_i = full_residual[i]

            if breaker:
                break
        # u_iter.append(self.u)
        self.u = self.Tr * self.u
        # self.u_iter = self.Tr.flatten() * np.array(u_iter)		## getting killed here

        # residual[1] = residual[1][::2]	# if update order is 0-1-2-1-0-... (i.e., 1 repeated twice in one iteration)
        self.residual = np.array(residual).T
        print(self.residual.shape)
        if self.residual.shape[0] < self.residual.shape[1]:
            self.residual = self.residual.T
        self.full_residual = np.array(full_residual)

        # Truncate u to ROI
        self.u = self.u[self.crop_to_roi]
        # self.u_iter = self.u_iter[tuple((slice(None),))+self.crop_to_roi]

        self.sim_time = time.time() - s1
        print('Simulation done (Time {} s)'.format(np.round(self.sim_time, 2)))

    def check_input_len(self, a, x):
        if isinstance(a, int) or isinstance(a, float):
            a = self.n_dims*tuple((a,)) + (3-self.n_dims) * (x,)
        elif len(a) == 1:
            a = self.n_dims*tuple(a) + (3-self.n_dims) * (x,)
        elif isinstance(a, list) or isinstance(a, tuple):
            a += (3 - len(a)) * (x,)
        if isinstance(a, np.ndarray):
            a = np.concatenate((a, np.zeros(3 - len(a))))
        return np.array(a).astype(int)

    # Ensure that domain size is the same across every dim
    def check_domain_size_same(self):
        while (self.domain_size[:self.n_dims] != np.max(self.domain_size[:self.n_dims])).any():
            self.bw_post[:self.n_dims] = self.bw_post[:self.n_dims] + self.n_domains[:self.n_dims] * (
                    np.max(self.domain_size[:self.n_dims]) - self.domain_size[:self.n_dims])
            self.n_ext = self.n_roi + self.bw_pre + self.bw_post
            self.domain_size[:self.n_dims] = (self.n_ext + (
                    (self.n_domains - 1) * self.overlap)) / self.n_domains

    # Ensure that domain size is less than 500 in every dim
    def check_domain_size_max(self):
        while (self.domain_size > self.max_domain_size).any():
            self.n_domains[np.where(self.domain_size > self.max_domain_size)] += 1
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    # Ensure that domain size is int
    def check_domain_size_int(self):
        while (self.domain_size % 1 != 0).any() or (self.bw_post % 1 != 0).any():
            self.bw_post += np.round(self.n_domains * (np.ceil(self.domain_size) - self.domain_size), 2)
            self.n_ext = self.n_roi + self.bw_pre + self.bw_post
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    # pad boundaries
    def pad_func(self, m, n_roi, which_end='Both'):
        full_filter = 1
        for i in range(self.n_dims):
            left_boundary = boundary_(np.floor(self.bw_pre[i]))
            right_boundary = np.flip(boundary_(np.ceil(self.bw_post[i])))
            if which_end == 'Both':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
            elif which_end == 'left':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i])))
            elif which_end == 'right':
                full_filter = np.concatenate((np.ones(n_roi[i]), right_boundary))

            m = np.moveaxis(m, i, -1) * full_filter
            m = np.moveaxis(m, -1, i)
        return m
