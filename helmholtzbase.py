import numpy as np
from itertools import product


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
                 overlap=20,  # Overlap between subdomains in each dimension
                 wrap_correction=None,  # Wrap-around correction. None, 'L_Omega' or 'L_corr'
                 cp=20,  # Corner points to include in case of 'L_corr' wrap-around correction
                 max_iterations=int(2.e+3),  # Maximum number iterations
                 setup_operators=True):  # Set up medium and propagator operators

        self.n = check_input_dims(n)
        self.n_dims = (np.squeeze(self.n)).ndim  # Number of dimensions in problem
        self.n_roi = np.array(self.n.shape, dtype=np.short)  # Num of points in ROI (Region of Interest)
        self.boundary_widths = self.check_input_len(boundary_widths, 0)
        self.bw_pre = np.floor(self.boundary_widths).astype(np.short)
        self.bw_post = np.ceil(self.boundary_widths).astype(np.short)
        self.wavelength = wavelength  # Wavelength in um (micron)
        self.ppw = ppw  # points per wavelength
        self.k0 = (1. * 2. * np.pi) / self.wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
        self.pixel_size = self.wavelength / self.ppw  # Grid pixel size in um (micron)
        self.n_ext = self.n_roi + self.bw_pre + self.bw_post  # n_roi + boundaries on either side(s)
        self.s = check_input_dims(source).astype(np.float16)
        self.max_subdomain_size = 500  # max permissible size of one sub-domain
        if n_domains is None:
            self.n_domains = self.n_ext // self.max_subdomain_size
        else:
            self.n_domains = self.check_input_len(n_domains,
                                                  1)  # Number of subdomains to decompose into in each dimension

        self.overlap = self.check_input_len(overlap, 0).astype(np.short)  # Overlap between subdomains in each dimension

        if (self.n_domains == 1).all():  # If 1 domain, implies no domain decomposition
            self.domain_size = self.n_ext.copy()
        else:  # Else, domain decomposition
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains
            self.check_domain_size_max()  # determines number of subdomains
            self.check_domain_size_same()  # all subdomains of same size
            self.check_domain_size_int()  # subdomain size is int

        self.bw_pre = self.bw_pre.astype(np.short)
        self.bw_post = self.bw_post.astype(np.short)
        self.n_ext = self.n_ext.astype(np.short)
        self.n_domains = self.n_domains.astype(np.short)
        self.domains_iterator = list(product(range(self.n_domains[0]), range(self.n_domains[1]),
                                             range(self.n_domains[2])))  # to iterate through subdomains in all dims
        self.domain_size[self.n_dims:] = 0
        self.domain_size = self.domain_size.astype(np.short)

        self.total_domains = np.prod(self.n_domains).astype(np.short)

        self.medium_operators = []
        self.v0 = None
        self.v = None
        self.scaling = None
        self.Tl = None
        self.Tr = None
        self.n_fast_conv = None
        self.propagator = None

        self.crop2roi = tuple([slice(self.bw_pre[i], -self.bw_post[i])
                               for i in range(self.n_dims)])  # crop array from n_ext to n_roi

        self.wrap_correction = wrap_correction  # None, 'L_omega', OR 'L_corr'
        self.cp = cp  # number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

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
        print('Boundaries width: \t', self.boundary_widths)
        if self.total_domains > 1:
            print(
                f'Decomposing into {self.n_domains} domains of size {self.domain_size}, overlap {self.overlap}')

    def setup_operators_n_initialize(self):
        """ Make Medium b = 1 - v and Propagator (L+1)^(-1) operators, and pad and scale source """
        v_raw = self.k0 ** 2 * self.n ** 2
        v_raw = np.pad(v_raw, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])), mode='edge')

        b = self.make_b(v_raw)

        self.medium_operators = {}
        for patch in self.domains_iterator:
            b_block = b[tuple([slice(patch[j] * (self.domain_size[j] - self.overlap[j]), 
                                     patch[j] * (self.domain_size[j] - self.overlap[j]) + self.domain_size[j])
                               for j in range(self.n_dims)])]
            self.medium_operators[patch] = lambda x, b_ = b_block: b_ * x
        self.make_propagator()
        self.s = self.Tl * np.squeeze(
            np.pad(self.s, (tuple([[self.bw_pre[i], self.bw_post[i]] for i in range(3)])),
                   mode='constant'))  # Scale the source term and pad

    def make_b(self, v_raw):
        """ Make the medium matrix, B = 1 - V """
        vraw_shape = v_raw.shape
        # give tiny non-zero minimum value to prevent division by zero in homogeneous media
        mu_min = ((10.0 / (self.boundary_widths[:self.n_dims] * self.pixel_size)) if (
                self.boundary_widths != 0).any() else 0).astype(np.float16)
        mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(vraw_shape[:self.n_dims]) * self.pixel_size)))
        v_min = np.imag((self.k0 + 1j * np.max(mu_min)) ** 2)
        self.v0 = (np.max(np.real(v_raw)) + np.min(np.real(v_raw))) / 2
        self.v0 = self.v0 + 1j * v_min
        self.v = -1j * (v_raw - self.v0)
        self.scaling = 0.95 / np.max(np.abs(self.v))
        self.v = self.scaling * self.v
        self.Tr = np.sqrt(self.scaling)
        self.Tl = 1j * self.Tr

        b = 1 - self.v
        b = np.squeeze(self.pad_func(m=b, n_roi=self.n_roi))  # apply ARL to b
        return b.astype(np.csingle)

    def make_propagator(self):
        """ Make the propagator operator that does fast convolution with (L+1)^(-1)"""
        n_subdomain = tuple(self.domain_size[:self.n_dims])
        if self.wrap_correction == 'L_omega':
            self.n_fast_conv = n_subdomain * 10
        else:
            self.n_fast_conv = n_subdomain

        # Fourier coordinates in n_dims
        l_p = ((2 * np.pi * np.fft.fftfreq(self.n_fast_conv[0], self.pixel_size)) ** 2).astype(np.csingle)
        for d in range(1, self.n_dims):
            l_p = np.expand_dims(l_p, axis=-1).astype(np.csingle) + np.expand_dims(
                (2 * np.pi * np.fft.fftfreq(self.n_fast_conv[d], self.pixel_size)) ** 2, axis=0).astype(np.csingle)
        l_p = 1j * self.scaling * (l_p - self.v0)
        l_p_inv = np.squeeze(1 / (l_p + 1))
        # propagator: operator for fast convolution with (L+1)^-1
        if self.wrap_correction == 'L_omega':
            self.propagator = lambda x: ((np.fft.ifftn(
                l_p_inv * np.fft.fftn(np.pad(x, (0, self.n_fast_conv - n_subdomain)))))[:n_subdomain]).astype(np.csingle)
        else:
            self.propagator = lambda x: (np.fft.ifftn(l_p_inv * np.fft.fftn(x))).astype(np.csingle)

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
        return np.array(a, dtype=np.short)

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

    def pad_func(self, m, n_roi, which_end='Both'):
        """ Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
        full_filter = 1
        for i in range(self.n_dims):
            left_boundary = boundary_(np.floor(self.bw_pre[i]))
            right_boundary = np.flip(boundary_(np.ceil(self.bw_post[i])))
            if which_end == 'Both':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
            elif which_end == 'pre':
                full_filter = np.concatenate((left_boundary, np.ones(n_roi[i])))
            elif which_end == 'post':
                full_filter = np.concatenate((np.ones(n_roi[i]), right_boundary))
            m = np.moveaxis(m, i, -1) * full_filter
            m = np.moveaxis(m, -1, i)
        return m
