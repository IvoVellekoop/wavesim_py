import numpy as np
from itertools import product
from scipy.linalg import dft


def check_input_dims(a):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - a.ndim):
        a = np.expand_dims(a, axis=-1)
    return a


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function """
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


class HelmholtzBase:
    def __init__(self,
                 n=np.ones((1, 1, 1)),  # Refractive index distribution
                 wavelength=1.,  # Wavelength in um (micron)
                 ppw=4,  # points per wavelength
                 boundary_widths=20,  # Width of absorbing boundaries
                 source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
                 n_domains=(1, 1, 1),  # Number of subdomains to decompose into, in each dimension
                 overlap=0,  # Overlap between subdomains in each dimension
                 wrap_correction=None,  # Wrap-around correction. None or 'wrap_corr' or 'L_omega'
                 cp=20,  # Corner points to include in case of 'wrap_corr' wrap-around correction
                 max_iterations=int(2.e+3),  # Maximum number iterations
                 setup_operators=True):  # Set up medium and propagator operators

        self.n = check_input_dims(n)
        self.n_dims = (np.squeeze(self.n)).ndim  # Number of dimensions in problem
        self.n_roi = np.array(self.n.shape, dtype=np.int16)  # Num of points in ROI (Region of Interest)
        self.boundary_widths = self.check_input_len(boundary_widths, 0)
        self.bw_pre = np.floor(self.boundary_widths).astype(np.int16)
        self.bw_post = np.ceil(self.boundary_widths).astype(np.float32)
        self.wavelength = wavelength  # Wavelength in um (micron)
        self.ppw = ppw  # points per wavelength
        self.k0 = (1. * 2. * np.pi) / self.wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
        self.pixel_size = self.wavelength / self.ppw  # Grid pixel size in um (micron)
        self.n_ext = self.n_roi + self.bw_pre + self.bw_post  # n_roi + boundaries on either side(s)
        self.s = check_input_dims(source).astype(np.float32)
        self.max_subdomain_size = 500  # max permissible size of one sub-domain
        if n_domains is None:
            self.n_domains = self.n_ext // self.max_subdomain_size
        else:
            self.n_domains = self.check_input_len(n_domains,
                                                  1)  # Number of subdomains to decompose into in each dimension

        self.overlap = self.check_input_len(overlap, 0).astype(np.int16)  # Overlap between subdomains in each dimension

        if (self.n_domains == 1).all():  # If 1 domain, implies no domain decomposition
            self.domain_size = self.n_ext.copy()
        else:  # Else, domain decomposition
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains
            self.check_domain_size_max()  # determines number of subdomains
            self.check_domain_size_same()  # all subdomains of same size
            self.check_domain_size_int()  # subdomain size is int

        self.bw_pre = self.bw_pre.astype(np.int16)
        self.bw_post = self.bw_post.astype(np.int16)
        self.n_ext = self.n_ext.astype(np.int16)
        self.n_domains = self.n_domains.astype(np.int16)
        self.domains_iterator = list(product(range(self.n_domains[0]), range(self.n_domains[1]),
                                             range(self.n_domains[2])))  # to iterate through subdomains in all dims
        self.domain_size[self.n_dims:] = 0
        self.domain_size = self.domain_size.astype(np.int16)

        self.total_domains = np.prod(self.n_domains).astype(np.int16)

        self.subdomain_Bs = []
        self.medium_operators = []
        self.v0 = None
        self.v = None
        self.fft_size = None
        self.scaling = None
        self.Tl = None
        self.Tr = None
        self.n_fft = None
        self.wrap_corr = None
        self.wrap_transfer = None
        self.full_b = None
        self.propagator = None
        self.transfer_info = None

        self.crop2roi = tuple([slice(self.bw_pre[i], -self.bw_post[i])
                               for i in range(self.n_dims)])  # crop array from n_ext to n_roi

        self.wrap_correction = wrap_correction  # None OR 'wrap_corr'
        self.cp = cp  # number of corner points (c.p.) in the upper and lower triangular corners of the wrap_corr matrix

        self.max_iterations = max_iterations
        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]
        self.threshold_residual = 1.e-6
        self.divergence_limit = 1.e+12

        self.print_details()
        if setup_operators:
            self.setup_operators_n_initialize()

    def print_details(self):
        """ Print main information about the problem """
        print(f'\n{self.n_dims} dimensional problem')
        if self.wrap_correction:
            print('Wrap correction: \t', self.wrap_correction)
        print('Boundaries widths (Pre): \t', self.bw_pre)
        print('\t\t (Post): \t', self.bw_post)
        if self.total_domains > 1:
            print(
                f'Decomposing into {self.n_domains} domains of size {self.domain_size}, overlap {self.overlap}')

    def setup_operators_n_initialize(self):
        """ Make (1) Medium b = 1 - v and (2) Propagator (L+1)^(-1) operators, and (3) pad and scale source """
        v_raw = self.k0 ** 2 * self.n ** 2
        v_raw = np.squeeze(np.pad(v_raw, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])), mode='edge'))
        self.full_b = self.make_b(v_raw)
        self.subdomain_Bs = {}
        self.medium_operators = {}
        for patch in self.domains_iterator:
            b_block = self.full_b[tuple([slice(patch[j] * (self.domain_size[j] - self.overlap[j]), 
                                  patch[j] * (self.domain_size[j] - self.overlap[j]) + self.domain_size[j])
                                  for j in range(self.n_dims)])]
            self.subdomain_Bs[patch] = b_block.copy()
            if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
                self.medium_operators[patch] = lambda x, b_ = b_block, subdomain_scaling = self.scaling[patch]: (
                        b_ * x - self.dot_ndim(x, subdomain_scaling * self.wrap_corr))
            else:
                self.medium_operators[patch] = lambda x, b_ = b_block: b_ * x

        self.s = np.squeeze(np.pad(self.s, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])), 
                                   mode='constant'))  # Pad the source term (scale later)

        self.propagator = self.make_propagator(self.domain_size[:self.n_dims])

    def make_b(self, v_raw):
        """ Make the medium matrix, B = 1 - V """
        # give tiny non-zero minimum value to prevent division by zero in homogeneous media
        mu_min = ((10.0 / (self.boundary_widths[:self.n_dims] * self.pixel_size)) if (
                self.boundary_widths != 0).any() else self.check_input_len(0, 0)).astype(np.float32)
        mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(v_raw.shape[:self.n_dims]) * self.pixel_size)))
        v_min = np.imag((self.k0 + 1j * np.max(mu_min)) ** 2)
        self.v0 = (np.max(np.real(v_raw)) + np.min(np.real(v_raw))) / 2
        self.v0 = self.v0 + 1j * v_min
        self.v = -1j * (v_raw - self.v0)

        if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
            self.fft_size = self.domain_size[0]
            lw = self.l_fft_operator(omega=1, truncate_to=1)

            # Option 1. EXACT. Using (Lo-lw) as the wrap-around correction
            lo = self.l_fft_operator(omega=20, truncate_to=1)
            self.wrap_corr = (1j * (lo-lw))
            # Option 2. APPROXIMATE. Replacing (Lo-lw) with -lw, or even -np.real(lw) (because real(lw)>>imag(lw))
            # self.wrap_corr = -1j * lw  # np.real(lw)

            # Truncate the wrap-around correction to square blocks of side cp in the upper and lower triangular corners
            # self.wrap_corr[:-self.cp, :-self.cp] = 0
            # self.wrap_corr[self.cp:, self.cp:] = 0

            # Transfer the one-sided wrap-around artefacts from one subdomain to another
            # apply subdomain_scaling => self.scaling[patch] later
            self.wrap_transfer = 1j * self.l_fft_operator(omega=20,
                                                          truncate_to=2)[:self.fft_size,
                                                                         self.fft_size:2*self.fft_size]
            # self.wrap_transfer = self.wrap_corr.copy()

            # change the scaling based on both v and wrap_corr
            wrap_corr_norm = np.linalg.norm(self.wrap_corr, 2)
        else:
            wrap_corr_norm = 0

        # # Option 1: Uniform scaling across the full domain
        scaling = 0.95 / max(np.max(np.abs(self.v)), wrap_corr_norm)
        # scaling = 0.027
        self.scaling = {}
        self.Tr = {}
        self.Tl = {}
        for patch in self.domains_iterator:
            current_patch = tuple([slice(patch[j] * (self.domain_size[j] - self.overlap[j]), 
                                  patch[j] * (self.domain_size[j] - self.overlap[j]) + self.domain_size[j])
                                  for j in range(self.n_dims)])
            v_temp = self.v[current_patch]
    
            # Option 1: Uniform scaling across the full domain
            self.scaling[patch] = scaling
            # # Option 2: Different scaling for different subdomains
            # self.scaling[patch] = (0.95 / max(np.max(np.abs(v_temp)), wrap_corr_norm))
    
            self.Tr[patch] = np.sqrt(self.scaling[patch])
            self.Tl[patch] = 1j * self.Tr[patch]
            self.v[current_patch] = self.scaling[patch] * v_temp

        b = 1 - self.v
        b = self.pad_func(m=b, n_roi=self.n_roi)
        return b

    def make_propagator(self, n):
        """ Make the propagator operator that does fast convolution with (l_p+1)^(-1) """
        if self.wrap_correction == 'L_omega':
            self.n_fft = n * 10
        else:
            self.n_fft = n

        l_p = self.coordinates_f()  # Fourier coordinates in n_dims

        # propagator: operator for fast convolution with (l_p+1)^-1

        # # Option 1: Uniform scaling across the full domain
        # l_p = 1j * self.scaling * (l_p - self.v0)  # Shift, scale, and multiply with 1j, l_p
        # l_inv = np.squeeze(1 / (l_p + 1))  # Invert (l_p + 1)

        # if self.wrap_correction == 'L_omega':
        #     propagator = lambda x: (np.fft.ifftn(l_inv * np.fft.fftn(np.pad(x, (0, self.n_fft[0] - n[0])))))[:n]
        # else:
        #     propagator = lambda x: np.fft.ifftn(l_inv * np.fft.fftn(x))

        # Option 2: Different scaling for different subdomains
        # Shift, and multiply with 1j, l_p (don't scale just yet. Scaling incorporated as an argument inside the
        # propagator operator definition)
        l_p = 1j * (l_p - self.v0)
        if self.wrap_correction == 'L_omega':
            propagator = lambda x, subdomain_scaling: (np.fft.ifftn(np.squeeze(1 / (subdomain_scaling * l_p + 1)) *
                                                       np.fft.fftn(np.pad(x, (0, self.n_fft[0] - n[0])))))[
                                                       tuple([slice(0, n[i]) for i in range(self.n_dims)])]
        else:
            propagator = lambda x, subdomain_scaling: np.fft.ifftn(np.squeeze(1 / (subdomain_scaling * l_p + 1)) *
                                                                   np.fft.fftn(x))
        return propagator

    def check_input_len(self, a, x):
        """ Convert 'a' to a 3-element numpy array, appropriately, i.e., either repeat, or add 0 or 1. """
        if isinstance(a, int) or isinstance(a, float):
            a = self.n_dims*tuple((a,)) + (3-self.n_dims) * (x,)
        elif len(a) == 1:
            a = self.n_dims*tuple(a) + (3-self.n_dims) * (x,)
        elif isinstance(a, list) or isinstance(a, tuple):
            a += (3 - len(a)) * (x,)
        if isinstance(a, np.ndarray):
            a = np.concatenate((a, np.zeros(3 - len(a))))
        return np.array(a, dtype=np.int16)

    def check_domain_size_same(self):
        """ Increase bw_post in dimension(s) until all subdomains are of the same size """
        while (self.domain_size[:self.n_dims] != np.max(self.domain_size[:self.n_dims])).any():
            self.bw_post[:self.n_dims] = self.bw_post[:self.n_dims] + self.n_domains[:self.n_dims] * (
                    np.max(self.domain_size[:self.n_dims]) - self.domain_size[:self.n_dims])
            self.n_ext = self.n_roi + self.bw_pre + self.bw_post
            self.domain_size[:self.n_dims] = (self.n_ext + (
                    (self.n_domains - 1) * self.overlap)) / self.n_domains

    def check_domain_size_max(self):
        """ Increase number of subdomains until subdomain size is less than max_subdomain_size """
        while (self.domain_size > self.max_subdomain_size).any():
            self.n_domains[np.where(self.domain_size > self.max_subdomain_size)] += 1
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    def check_domain_size_int(self):
        """ Increase bw_post in dimension(s) until the subdomain size is int """
        while (self.domain_size % 1 != 0).any() or (self.bw_post % 1 != 0).any():
            self.bw_post += np.round(self.n_domains * (np.ceil(self.domain_size) - self.domain_size), 2)
            self.n_ext = self.n_roi + self.bw_pre + self.bw_post
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    def l_fft_operator(self, omega=1, truncate_to=1):
        """ Make the operator that does fast convolution with L
        :param omega: Do the fast convolution over a domain size (omega) times (fft_size).
                      Default: omega=1, i.e., gives wrap-around artefacts.
        :param truncate_to: To truncate to (truncate_to) times (fft_size). Must be <= omega.
                  Default truncate_to=1, i.e., truncate to original domain size
        :return:
        """
        n = omega * self.fft_size
        f = np.array(dft(n))
        finv = np.conj(f).T/n
        l_p = ((2 * np.pi * np.fft.fftfreq(n, self.pixel_size)) ** 2)
        if omega > 1:
            m = truncate_to * self.fft_size
            trunc = np.zeros((m, n))
            trunc[:, :m] = np.eye(m)
            l_fft = trunc @ finv @ np.diag(l_p.ravel()) @ f @ trunc.T
        else:
            l_fft = finv @ np.diag(l_p.ravel()) @ f
        return l_fft.astype(np.complex64)

    def pad_func(self, m, n_roi):
        """ Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
        for i in range(self.n_dims):
            left_boundary = boundary_(self.bw_pre[i])
            right_boundary = np.flip(boundary_(self.bw_post[i]))
            full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
            m = np.moveaxis(m, i, -1) * full_filter
            m = np.moveaxis(m, -1, i)
        return m.astype(np.complex64)

    def coordinates_f(self):
        """ Fourier space coordinates for given size, spacing, and dimensions """
        l_p = ((2 * np.pi * np.fft.fftfreq(self.n_fft[0], self.pixel_size)) ** 2).astype(np.complex64)
        for d in range(1, self.n_dims):
            l_p = np.expand_dims(l_p, axis=-1) + np.expand_dims(
                (2 * np.pi * np.fft.fftfreq(self.n_fft[d], self.pixel_size)) ** 2, axis=0).astype(np.complex64)
        return l_p

    def dot_ndim(self, x, y):
        """ np.dot(x, y) over all axes of x.
        Here y is a 2-D array for fast-convolution or related operations that need to be applied to every axis of x """
        for i in range(self.n_dims):
            x = np.moveaxis(x, i, -1)  # Transpose
            x = np.dot(x, y)
            x = np.moveaxis(x, -1, i)  # Transpose back
        return x.astype(np.complex64)

    def transfer(self, x, subdomain_scaling, patch_shift):
        if patch_shift == -1:
            return self.dot_ndim(x, subdomain_scaling * self.wrap_transfer)
        elif patch_shift == +1:
            return self.dot_ndim(x, subdomain_scaling * np.flip(self.wrap_transfer))
