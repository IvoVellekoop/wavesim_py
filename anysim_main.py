import os
import time
import numpy as np
from datetime import date


def check_input_dims(a):
    for _ in range(3 - a.ndim):
        a = np.expand_dims(a, axis=-1)
    return a


def check_input_len(a, x):
    if isinstance(a, list) or isinstance(a, tuple):
        a += (3 - len(a)) * (x,)
    elif isinstance(a, int) or isinstance(a, float):
        a = tuple((a,)) + 2 * (x,)
    if isinstance(a, np.ndarray):
        a = np.concatenate((a, np.zeros(3 - len(a))))
    return np.array(a).astype(int)


def boundary_(x):
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def overlap_decay(x):
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


# Relative error
def relative_error(e, e_true):
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)


class AnySim:
    k0 = None
    n = None
    n_dims = None
    pixel_size = None
    n_roi = None
    n_ext = None

    b = None
    u = None

    boundary_widths = None
    bw_pre = None
    bw_post = None

    domain_size = None
    max_domain_size = None
    n_domains = None
    total_domains = None
    range_total_domains = None
    overlap = None

    wrap_correction = None

    max_iterations = None
    threshold_residual = None
    iter_step = None
    crop_to_roi = None

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

        AnySim.n = check_input_dims(n)
        AnySim.n_dims = (np.squeeze(AnySim.n)).ndim  # Number of dimensions in problem

        AnySim.n_roi = np.array(AnySim.n.shape)  # Num of points in ROI (Region of Interest)

        AnySim.boundary_widths = check_input_len(boundary_widths, 0)
        AnySim.bw_pre = np.floor(AnySim.boundary_widths)
        AnySim.bw_post = np.ceil(AnySim.boundary_widths)

        AnySim.wavelength = wavelength  # Wavelength in um (micron)
        AnySim.ppw = ppw  # points per wavelength
        AnySim.k0 = (1. * 2. * np.pi) / (
            AnySim.wavelength)  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
        AnySim.pixel_size = AnySim.wavelength / AnySim.ppw  # Grid pixel size in um (micron)

        AnySim.n_ext = AnySim.n_roi + AnySim.bw_pre + AnySim.bw_post
        AnySim.b = check_input_dims(source)

        AnySim.max_domain_size = 500
        if n_domains is None:
            AnySim.n_domains = AnySim.n_ext // AnySim.max_domain_size  # n_ext/Max permissible size of sub-domain
        else:
            AnySim.n_domains = check_input_len(n_domains, 1)  # Number of subdomains to decompose into in each dimension

        AnySim.overlap = check_input_len(overlap, 0)  # Overlap between subdomains in each dimension

        if (AnySim.n_domains == 1).all():  # If 1 domain, implies no domain decomposition
            AnySim.domain_size = AnySim.n_ext.copy()
        else:  # Else, domain decomposition
            AnySim.domain_size = (AnySim.n_ext + ((AnySim.n_domains - 1) * AnySim.overlap)) / AnySim.n_domains
            self.check_domain_size_max()
            self.check_domain_size_same()
            self.check_domain_size_int()

        AnySim.bw_pre = AnySim.bw_pre.astype(int)
        AnySim.bw_post = AnySim.bw_post.astype(int)
        AnySim.n_ext = AnySim.n_ext.astype(int)
        AnySim.n_domains = AnySim.n_domains
        AnySim.domain_size[AnySim.n_dims:] = 0
        AnySim.domain_size = AnySim.domain_size.astype(int)

        AnySim.total_domains = np.prod(AnySim.n_domains)
        AnySim.range_total_domains = range(AnySim.total_domains)
        self.operators = []
        self.Tl = None
        self.Tr = None
        self.v = None
        self.medium = None
        self.propagator = None
        self.n_fast_conv = None

        AnySim.crop_to_roi = tuple([slice(AnySim.bw_pre[i], -AnySim.bw_post[i]) for i in range(AnySim.n_dims)])

        AnySim.wrap_correction = wrap_correction  # None, 'L_omega', OR 'L_corr'
        AnySim.cp = cp  # number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

        AnySim.max_iterations = max_iterations
        AnySim.iterations = AnySim.max_iterations - 1

        ''' Create log folder / Check for existing log folder'''
        today = date.today()
        d1 = today.strftime("%Y%m%d")
        AnySim.log_dir = 'logs/Logs_' + d1 + '/'
        if not os.path.exists(AnySim.log_dir):
            os.makedirs(AnySim.log_dir)

        AnySim.run_id = d1
        AnySim.run_loc = AnySim.log_dir  # + AnySim.run_id
        if not os.path.exists(AnySim.run_loc):
            os.makedirs(AnySim.run_loc)

        AnySim.run_id += '_n_dims' + str(AnySim.n_dims) + '_abs' + str(AnySim.boundary_widths)
        if AnySim.wrap_correction:
            AnySim.run_id += '_' + AnySim.wrap_correction
        AnySim.run_id += '_n_domains' + str(AnySim.n_domains)

        AnySim.stats_file_name = AnySim.log_dir + d1 + '_stats.txt'

        self.print_details()  # print the simulation details

    @staticmethod
    def print_details():
        print(f'\n{AnySim.n_dims} dimensional problem')
        if AnySim.wrap_correction:
            print('Wrap correction: \t', AnySim.wrap_correction)
        print('Boundaries width: \t', AnySim.boundary_widths)
        if AnySim.total_domains > 1:
            print(
                f'Decomposing into {AnySim.n_domains} domains of size {AnySim.domain_size}, overlap {AnySim.overlap}')

    def setup_operators_n_init_variables(self):  # function that calls all the other 'main' functions
        # Make operators: Medium b = 1 - v, and Propagator (L+1)^(-1)
        v_raw = AnySim.k0 ** 2 * AnySim.n ** 2
        v_raw = np.pad(v_raw, (tuple([[AnySim.bw_pre[i], AnySim.bw_post[i]] for i in range(3)])), mode='edge')

        # v_raw_blocks = [[] for _ in range(AnySim.n_dims)]

        # for j in range(AnySim.n_domains[0]):
        # 	v_raw_blocks[0].append(v_raw[j*(AnySim.domain_size[0]-AnySim.overlap):j*(AnySim.domain_size[0]-AnySim.overlap)+AnySim.domain_size[0]])

        # for i in range(AnySim.n_dims-1):
        # 	print(f'{i} main dimension')
        # 	# temp_blocks = v_raw_blocks.copy()
        # 	# v_raw_blocks = [[] for _ in range(AnySim.n_dims)]
        # 	temp_blocks = v_raw_blocks.copy()
        # 	v_raw_blocks[i] = []
        # 	print(f'range 0 to {i+1}')
        # 	for j in range(AnySim.n_dims):
        # 		print('-'*50)
        # 		print(f'{j} internal dimension for reassigning')
        # 		print(f'size of temp_blocks[{i}] = {len(temp_blocks[i])}')
        # 		for k in range(AnySim.n_domains[i]):# range(len(temp_blocks[i])):
        # 			print(f'pick element {j} in k[{i}]')
        # 			print(f'assign to v_raw_blocks[{j}], elements {k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])} to {k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])+AnySim.domain_size[i+1]}')

        # 			assign_block = temp_blocks[i][j][..., k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1]):k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])+AnySim.domain_size[i+1] ]
        # 			v_raw_blocks[j].append(assign_block)

        # 			plt.imshow(np.abs(assign_block))
        # 			plt.colorbar()
        # 			plt.show()
        # 			plt.close()

        if AnySim.total_domains == 1:
            self.operators.append(self.make_operators(v_raw, 'Both'))
        else:
            self.operators.append(
                self.make_operators(v_raw[tuple([slice(0, AnySim.domain_size[i]) for i in range(AnySim.n_dims)])],
                                    'left'))
            for d in range(1, AnySim.total_domains - 1):
                # aa = v_raw[tuple([slice(d*(AnySim.domain_size[i]-AnySim.overlap[i]), d*(AnySim.domain_size[i]-AnySim.overlap[i])+AnySim.domain_size[i]) for i in range(AnySim.n_dims)])]
                # print(d, aa.shape)
                self.operators.append(self.make_operators(v_raw[tuple([slice(
                    d * (AnySim.domain_size[i] - AnySim.overlap[i]),
                    d * (AnySim.domain_size[i] - AnySim.overlap[i]) + AnySim.domain_size[i]) for i in
                    range(AnySim.n_dims)])], None))
            self.operators.append(
                self.make_operators(v_raw[tuple([slice(-AnySim.domain_size[i], None) for i in range(AnySim.n_dims)])],
                                    'right'))

        # Scale the source term (and pad if boundaries)
        AnySim.b = self.Tl * np.squeeze(
            np.pad(AnySim.b, (tuple([[AnySim.bw_pre[i], AnySim.bw_post[i]] for i in range(3)])),
                   mode='constant'))  # source term y
        AnySim.u = (np.zeros_like(AnySim.b, dtype='complex_'))  # field u, initialize with 0

    # Make the operators: Medium b = 1 - v and Propagator (L+1)^(-1)
    def make_operators(self, v_raw, which_end=None):
        n = v_raw.shape
        # give tiny non-zero minimum value to prevent division by zero in homogeneous media
        mu_min = (10.0 / (AnySim.boundary_widths[:AnySim.n_dims] * AnySim.pixel_size)) if (
                AnySim.boundary_widths != 0).any() else 0
        mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(n[:AnySim.n_dims]) * AnySim.pixel_size)))
        v_min = np.imag((AnySim.k0 + 1j * np.max(mu_min)) ** 2)
        v_max = 0.95
        v0 = (np.max(np.real(v_raw)) + np.min(np.real(v_raw))) / 2
        v0 = v0 + 1j * v_min
        self.v = -1j * (v_raw - v0)

        # if AnySim.wrap_correction == 'L_corr':
        # 	p = 2*np.pi*np.fft.fftfreq(n, AnySim.pixel_size)
        # 	Lw_p = p**2
        # 	Lw = Finv @ np.diag(Lw_p.flatten()) @ F
        # 	L_corr = -np.real(Lw)
        # # Keep only upper and lower triangular corners of -Lw
        # 	L_corr[:-AnySim.cp,:-AnySim.cp] = 0; L_corr[AnySim.cp:,AnySim.cp:] = 0
        # 	self.v = self.v + 1j*L_corr

        scaling = v_max / np.max(np.abs(self.v))
        self.v = scaling * self.v

        self.Tr = np.sqrt(scaling)
        self.Tl = 1j * self.Tr

        # b = 1 - v
        b = 1 - self.v
        if which_end is None:
            b = np.squeeze(b)
        else:
            if which_end == 'Both':
                n_roi = AnySim.n_roi
            elif which_end == 'left':
                n_roi = n - AnySim.bw_pre
            elif which_end == 'right':
                n_roi = n - AnySim.bw_post
            else:
                n_roi = n - AnySim.bw_pre - AnySim.bw_post
            b = np.squeeze(self.pad_func(m=b, n_roi=n_roi, which_end=which_end).astype('complex_'))
        self.medium = lambda x: b * x

        # Make Propagator (L+1)^(-1)
        if AnySim.wrap_correction == 'L_omega':
            self.n_fast_conv = n * 10
        else:
            self.n_fast_conv = n

        l_p = (2 * np.pi * np.fft.fftfreq(self.n_fast_conv[0], self.pixel_size)) ** 2
        for d in range(1, AnySim.n_dims):
            l_p = np.expand_dims(l_p, axis=-1) + np.expand_dims(
                (2 * np.pi * np.fft.fftfreq(self.n_fast_conv[d], self.pixel_size)) ** 2, axis=0)
        l_p = 1j * scaling * (l_p - v0)
        l_p_inv = np.squeeze(1 / (l_p + 1))
        if AnySim.wrap_correction == 'L_omega':
            self.propagator = lambda x: (np.fft.ifftn(l_p_inv * np.fft.fftn(np.pad(x, (0, self.n_fast_conv - n)))))[:n]
        else:
            self.propagator = lambda x: (np.fft.ifftn(l_p_inv * np.fft.fftn(x)))

        return [self.medium, self.propagator]

    # AnySim update
    def iterate(self):
        s1 = time.time()
        medium = 0
        propagator = 1

        alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]
        AnySim.threshold_residual = 1.e-6
        AnySim.iter_step = 1

        # Construct restriction operators (restrict) and partition of unity operators (pou)
        u = []
        b = []
        restrict = []
        pou = []
        if AnySim.total_domains == 1:
            u.append(AnySim.u)
            b.append(AnySim.b)
            # To Normalize subdomain residual wrt preconditioned source
            full_norm_gb = np.linalg.norm(self.operators[0][medium](self.operators[0][propagator](b[0])))
        else:
            ones = np.eye(AnySim.domain_size[0])
            restrict0 = np.zeros((AnySim.domain_size[0], AnySim.n_ext[0]))
            for i in AnySim.range_total_domains:
                restrict_mid = restrict0.copy()
                restrict_mid[:,
                             i * (AnySim.domain_size[0] - AnySim.overlap[0]):
                             i * (AnySim.domain_size[0] - AnySim.overlap[0]) + AnySim.domain_size[0]] = ones
                restrict.append(restrict_mid)

            decay = overlap_decay(AnySim.overlap[0])
            pou1 = np.diag(np.concatenate((np.ones(AnySim.domain_size[0] - AnySim.overlap[0]), np.flip(decay))))
            pou.append(pou1)
            pou_mid = np.diag(
                np.concatenate((decay, np.ones(AnySim.domain_size[0] - 2 * AnySim.overlap[0]), np.flip(decay))))
            for _ in range(1, AnySim.total_domains - 1):
                pou.append(pou_mid)
            pou_end = np.diag(np.concatenate((decay, np.ones(AnySim.domain_size[0] - AnySim.overlap[0]))))
            pou.append(pou_end)

            for j in AnySim.range_total_domains:
                u.append(restrict[j] @ AnySim.u)
                b.append(restrict[j] @ AnySim.b)

            # To Normalize subdomain residual wrt preconditioned source
            full_norm_gb = np.linalg.norm(np.sum(np.array(
                [(restrict[j].T @ pou[j] @ self.operators[j][medium](self.operators[j][propagator](b[j]))) for j in
                 AnySim.range_total_domains]), axis=0))

        tj = [None for _ in AnySim.range_total_domains]
        residual_i = [None for _ in AnySim.range_total_domains]
        residual = [[] for _ in AnySim.range_total_domains]
        # u_iter = []
        breaker = False

        full_residual = []

        for i in range(AnySim.max_iterations):
            for j in AnySim.range_total_domains:
                print('Iteration {}, sub-domain {}.'.format(i + 1, j + 1), end='\r')
                # Main update START ---
                # if i % AnySim.iter_step == 0:
                if AnySim.total_domains == 1:
                    u[j] = AnySim.u.copy()
                else:
                    u[j] = restrict[j] @ AnySim.u
                tj[j] = self.operators[j][medium](u[j]) + b[j]
                tj[j] = self.operators[j][propagator](tj[j])
                tj[j] = self.operators[j][medium](u[j] - tj[j])  # subdomain residual
                # --- continued below ---

                ''' Residual collection and checking '''
                # To Normalize subdomain residual wrt preconditioned source
                if AnySim.total_domains == 1:
                    nr = np.linalg.norm(tj[j])
                else:
                    nr = np.linalg.norm(pou[j] @ tj[j])

                residual_i[j] = nr / full_norm_gb
                residual[j].append(residual_i[j])

                # --- continued below ---
                u[j] = alpha * tj[j]
                # if i % AnySim.iter_step == 0:
                if AnySim.total_domains == 1:
                    AnySim.u = AnySim.u - u[j]  # instead of this, simply update on overlapping regions?
                else:
                    AnySim.u = AnySim.u - restrict[j].T @ pou[j] @ u[j]  # instead, update on overlapping regions?
            # Main update END ---

            # Full Residual
            if AnySim.total_domains == 1:
                full_nr = np.linalg.norm(tj[0])
            else:
                full_nr = np.linalg.norm(
                    np.sum(np.array([(restrict[j].T @ pou[j] @ tj[j]) for j in AnySim.range_total_domains]), axis=0))

            full_residual.append(full_nr / full_norm_gb)
            if full_residual[i] < AnySim.threshold_residual:
                AnySim.iterations = i
                print(f'Stopping. Iter {AnySim.iterations + 1} '
                      f'residual {full_residual[i]:.2e}<={AnySim.threshold_residual}')
                breaker = True
                break
            AnySim.residual_i = full_residual[i]

            if breaker:
                break
        # u_iter.append(AnySim.u)
        AnySim.u = self.Tr * AnySim.u
        # self.u_iter = self.Tr.flatten() * np.array(u_iter)		## getting killed here

        # residual[1] = residual[1][::2]	# if update order is 0-1-2-1-0-... (i.e., 1 repeated twice in one iteration)
        AnySim.residual = np.array(residual).T
        print(AnySim.residual.shape)
        if AnySim.residual.shape[0] < AnySim.residual.shape[1]:
            AnySim.residual = AnySim.residual.T
        AnySim.full_residual = np.array(full_residual)

        # Truncate u to ROI
        AnySim.u = AnySim.u[AnySim.crop_to_roi]
        # self.u_iter = self.u_iter[tuple((slice(None),))+AnySim.crop_to_roi]

        AnySim.sim_time = time.time() - s1
        print('Simulation done (Time {} s)'.format(np.round(AnySim.sim_time, 2)))

        return AnySim.u

    # Ensure that domain size is the same across every dim
    @staticmethod
    def check_domain_size_same():
        while (AnySim.domain_size[:AnySim.n_dims] != np.max(AnySim.domain_size[:AnySim.n_dims])).any():
            AnySim.bw_post[:AnySim.n_dims] = AnySim.bw_post[:AnySim.n_dims] + AnySim.n_domains[:AnySim.n_dims] * (
                    np.max(AnySim.domain_size[:AnySim.n_dims]) - AnySim.domain_size[:AnySim.n_dims])
            AnySim.n_ext = AnySim.n_roi + AnySim.bw_pre + AnySim.bw_post
            AnySim.domain_size[:AnySim.n_dims] = (AnySim.n_ext + (
                    (AnySim.n_domains - 1) * AnySim.overlap)) / AnySim.n_domains

    # Ensure that domain size is less than 500 in every dim
    @staticmethod
    def check_domain_size_max():
        while (AnySim.domain_size > AnySim.max_domain_size).any():
            AnySim.n_domains[np.where(AnySim.domain_size > AnySim.max_domain_size)] += 1
            AnySim.domain_size = (AnySim.n_ext + ((AnySim.n_domains - 1) * AnySim.overlap)) / AnySim.n_domains

    # Ensure that domain size is int
    @staticmethod
    def check_domain_size_int():
        while (AnySim.domain_size % 1 != 0).any() or (AnySim.bw_post % 1 != 0).any():
            AnySim.bw_post += np.round(AnySim.n_domains * (np.ceil(AnySim.domain_size) - AnySim.domain_size), 2)
            AnySim.n_ext = AnySim.n_roi + AnySim.bw_pre + AnySim.bw_post
            AnySim.domain_size = (AnySim.n_ext + ((AnySim.n_domains - 1) * AnySim.overlap)) / AnySim.n_domains

    # pad boundaries
    @staticmethod
    def pad_func(m, n_roi, which_end='Both'):
        full_filter = 1
        for i in range(AnySim.n_dims):
            left_boundary = boundary_(np.floor(AnySim.bw_pre[i]))
            right_boundary = np.flip(boundary_(np.ceil(AnySim.bw_post[i])))
            if which_end == 'Both':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
            elif which_end == 'left':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i])))
            elif which_end == 'right':
                full_filter = np.concatenate((np.ones(n_roi[i]), right_boundary))

            m = np.moveaxis(m, i, -1) * full_filter
            m = np.moveaxis(m, -1, i)
        return m

    # Compute and print relative error between u and some analytic/"ideal"/"expected" u_true
    @staticmethod
    def compare(u_true):
        # Print relative error between u and analytic solution (or Matlab result)
        AnySim.u_true = u_true

        if AnySim.u_true.shape[0] != AnySim.n_roi[0]:
            AnySim.u = AnySim.u[tuple([slice(0, AnySim.n_roi[i]) for i in range(AnySim.n_dims)])]
        # self.u_iter = self.u_iter[:, :AnySim.n_roi]

        AnySim.rel_err = relative_error(AnySim.u, AnySim.u_true)
        print('Relative error: {:.2e}'.format(AnySim.rel_err))
        return AnySim.rel_err
