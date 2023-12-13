from itertools import product
from scipy.linalg import dft
from scipy.fft import fftn, ifftn, fftfreq
from scipy.sparse import diags as spdiags
from scipy.sparse.linalg import norm as spnorm
from utilities import *


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
                 n_correction=8,  # number of points used in the wrapping correction
                 max_iterations=int(3.e+4),  # Maximum number iterations
                 setup_operators=True):  # Set up medium and propagator operators

        self.n = check_input_dims(n)
        self.n_dims = (np.squeeze(self.n)).ndim  # Number of dimensions in problem
        self.n_roi = np.array(self.n.shape)  # Num of points in ROI (Region of Interest)
        self.boundary_widths = self.check_input_len(boundary_widths, 0)
        self.boundary_pre = np.floor(self.boundary_widths).astype(int)
        self.boundary_post = np.ceil(self.boundary_widths).astype(np.float32)
        self.wavelength = wavelength  # Wavelength in um (micron)
        self.ppw = ppw  # points per wavelength
        self.k0 = (1. * 2. * np.pi) / self.wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
        self.pixel_size = self.wavelength / self.ppw  # Grid pixel size in um (micron)
        self.n_ext = self.n_roi + self.boundary_pre + self.boundary_post  # n_roi + boundaries on either side(s)
        self.s = check_input_dims(source).astype(np.float32)
        self.max_subdomain_size = 500  # max permissible size of one sub-domain
        if n_domains is None:
            self.n_domains = self.n_ext // self.max_subdomain_size
        else:
            self.n_domains = self.check_input_len(n_domains,
                                                  1)  # Number of subdomains to decompose into in each dimension

        self.overlap = self.check_input_len(overlap, 0).astype(int)  # Overlap between subdomains in each dimension

        if (self.n_domains == 1).all():  # If 1 domain, implies no domain decomposition
            self.domain_size = self.n_ext.copy()
        else:  # Else, domain decomposition
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains
            self.check_domain_size_max()  # determines number of subdomains
            self.check_domain_size_same()  # all subdomains of same size
            self.check_domain_size_int()  # subdomain size is int

        self.boundary_pre = self.boundary_pre.astype(int)
        self.boundary_post = self.boundary_post.astype(int)
        self.n_ext = self.n_ext.astype(int)
        self.n_domains = self.n_domains.astype(int)
        self.domains_iterator = list(product(range(self.n_domains[0]), range(self.n_domains[1]),
                                             range(self.n_domains[2])))  # to iterate through subdomains in all dims
        self.domain_size[self.n_dims:] = 0
        self.domain_size = self.domain_size.astype(int)

        self.total_domains = np.prod(self.n_domains).astype(int)

        self.v = None
        self.medium_operators = None
        self.propagator = None
        self.scaling = None
        self.wrap_corr = None
        self.wrap_transfer = None
        self.l_p = None

        self.crop2roi = tuple([slice(self.boundary_pre[i], -self.boundary_post[i])
                               for i in range(self.n_dims)])  # crop array from n_ext to n_roi

        self.wrap_correction = wrap_correction  # None OR 'wrap_corr'
        self.n_correction = n_correction  # number of corner points (c.p.) in the upper and lower triangular corners of the wrap_corr matrix

        self.max_iterations = max_iterations
        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]
        self.threshold_residual = 1.e-6
        self.divergence_limit = 1.e+12

        self.print_details()
        if setup_operators:
            self.initialize()

    def print_details(self):
        """ Print main information about the problem """
        print(f'\n{self.n_dims} dimensional problem')
        if self.wrap_correction:
            print('Wrap correction: \t', self.wrap_correction)
        print('Boundaries widths (Pre): \t', self.boundary_pre)
        print('\t\t (Post): \t', self.boundary_post)
        if self.total_domains > 1:
            print(
                f'Decomposing into {self.n_domains} domains of size {self.domain_size}, overlap {self.overlap}')

    def initialize(self):
        """ Get (1) Medium b = 1 - v - corrections and (2) Propagator (L+1)^(-1) operators, and (3) pad source """
        v_raw = self.k0 ** 2 * self.n ** 2
        # pad v_raw with boundaries using edge values
        v_raw = np.squeeze(np.pad(v_raw, (tuple([[self.boundary_pre[i], self.boundary_post[i]] for i in range(3)])),
                                  mode="edge"))
        # get the medium and propagator operators, and scaling
        self.medium_operators, self.propagator, self.scaling = self.make_operators(v_raw)
        # Pad the source term (scale later)
        self.s = np.squeeze(np.pad(self.s, (tuple([[self.boundary_pre[i], self.boundary_post[i]] for i in range(3)])), 
                                   mode="constant"))

    def make_operators(self, v_raw):
        """ Make the medium and propagator operators, and, if applicable,
        the wrapping correction and wrapping transfer operators. """
        # Make v
        v0 = (np.max(np.real(v_raw)) + np.min(np.real(v_raw))) / 2
        v0 = v0 + 1j * self.v_min(v_raw)
        self.v = -1j * (v_raw - v0)

        # Make the wrap_corr operator (If wrap_correction=True)
        d = self.domain_size[:self.n_dims]
        crop2domain = tuple([slice(0, d[i]) for i in range(self.n_dims)]) # crop from the padded domain to the original domain

        omega = 10  # compute the fft over omega times the domain size
        if self.wrap_correction == 'L_omega':
            n_fft = d * omega
        else:
            n_fft = d
        
        l_p = self.laplacian_sq_f(n_fft, self.n_dims, self.pixel_size)  # Fourier coordinates in n_dims
        if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
            # compute the 1-D convolution kernel (brute force) and take the wrapped part of it
            side = np.zeros(d, dtype=np.complex64)
            side[-1,...] = 1.0
            k_wrap = np.real(ifftn(l_p * fftn(side))[:, *(0,)*(self.n_dims - 1)]) # discard tiny imaginary part due to numerical errors

            # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
            wrap_matrix = np.zeros((self.n_correction, self.n_correction), dtype=np.complex64)
            for r in range(self.n_correction):
                size = r + 1
                wrap_matrix[r, :] = k_wrap[self.n_correction-size:2*self.n_correction-size]

            self.wrap_corr = lambda x: 1j * self.compute_wrap_corr(x, wrap_matrix)

        # Compute (and apply to v) the scaling. Scaling computation includes wrap_corr if wrap_correction=True
        scaling = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)
            v_norm = np.max(np.abs(self.v[patch_slice]))
            if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
                v_norm += self.n_dims * np.linalg.norm(wrap_matrix, 2)
            scaling[patch] = 0.95/v_norm
            self.v[patch_slice] = scaling[patch] * self.v[patch_slice]

        # Make b and apply ARL
        b = 1 - self.v
        b = pad_func(b, self.boundary_pre, self.boundary_post, self.n_roi, n_dims=self.n_dims)
        # Make the medium operator(s)
        medium_operators = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)
            b_block = b[patch_slice]
            if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
                medium_operators[patch] = lambda x, b_ = b_block: (b_ * x - scaling[patch] * self.wrap_corr(x))
            else:
                medium_operators[patch] = lambda x, b_ = b_block: b_ * x

        # Transfer the one-sided wrap-around artefacts from one subdomain to another [only works for 1D case]
        # apply subdomain_scaling i.e., scaling[patch] later
        if self.total_domains > 1:
            fft_size = self.domain_size[0]
            self.wrap_transfer = 1j * self.l_fft_matrix(self.domain_size, omega=20, truncate_to=2, 
                                                        pixel_size=self.pixel_size)[:fft_size, fft_size:2*fft_size]

        # Make the propagator operator that does fast convolution with (l_p+1)^(-1)
        self.l_p = 1j * (l_p - v0)  # Shift l_p and multiply with 1j (Scaling incorporated inside propagator operator)
        if self.wrap_correction == 'L_omega':
            propagator = lambda x, subdomain_scaling: (ifftn(np.squeeze(1 / (subdomain_scaling * self.l_p + 1)) *
                                                       fftn(x, n_fft)))[crop2domain]
        else:
            propagator = lambda x, subdomain_scaling: ifftn(np.squeeze(1 / (subdomain_scaling * self.l_p + 1)) *
                                                            fftn(x))

        return medium_operators, propagator, scaling

    def v_min(self, v_raw):
        """ give tiny non-zero minimum value to prevent division by zero in homogeneous media """
        mu_min = ((10.0 / (self.boundary_widths[:self.n_dims] * self.pixel_size)) if (
                self.boundary_widths != 0).any() else self.check_input_len(0, 0)).astype(np.float32)
        mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(v_raw.shape[:self.n_dims]) * self.pixel_size)))
        return np.imag((self.k0 + 1j * np.max(mu_min)) ** 2)
    
    def patch_slice(self, patch):
        """ Return the slice, i.e., indices for the current 'patch', i.e., the subdomain """
        return tuple([slice(patch[j] * (self.domain_size[j] - self.overlap[j]), 
                            patch[j] * (self.domain_size[j] - self.overlap[j]) + self.domain_size[j]) for
                      j in range(self.n_dims)])

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
        return np.array(a)

    def check_domain_size_same(self):
        """ Increase boundary_post in dimension(s) until all subdomains are of the same size """
        while (self.domain_size[:self.n_dims] != np.max(self.domain_size[:self.n_dims])).any():
            self.boundary_post[:self.n_dims] = self.boundary_post[:self.n_dims] + self.n_domains[:self.n_dims] * (
                    np.max(self.domain_size[:self.n_dims]) - self.domain_size[:self.n_dims])
            self.n_ext = self.n_roi + self.boundary_pre + self.boundary_post
            self.domain_size[:self.n_dims] = (self.n_ext + (
                    (self.n_domains - 1) * self.overlap)) / self.n_domains

    def check_domain_size_max(self):
        """ Increase number of subdomains until subdomain size is less than max_subdomain_size """
        while (self.domain_size > self.max_subdomain_size).any():
            self.n_domains[np.where(self.domain_size > self.max_subdomain_size)] += 1
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    def check_domain_size_int(self):
        """ Increase boundary_post in dimension(s) until the subdomain size is int """
        while (self.domain_size % 1 != 0).any() or (self.boundary_post % 1 != 0).any():
            self.boundary_post += np.round(self.n_domains * (np.ceil(self.domain_size) - self.domain_size), 2)
            self.n_ext = self.n_roi + self.boundary_pre + self.boundary_post
            self.domain_size = (self.n_ext + ((self.n_domains - 1) * self.overlap)) / self.n_domains

    @staticmethod
    def laplacian_sq_f(n_fft, n_dims, pixel_size=1.):
        """ Fourier space coordinates for given size, spacing, and dimensions """
        l_p = ((2 * np.pi * fftfreq(n_fft[0], pixel_size)) ** 2).astype(np.complex64)
        for d in range(1, n_dims):
            l_p = np.expand_dims(l_p, axis=-1) + np.expand_dims(
                (2 * np.pi * fftfreq(n_fft[d], pixel_size)) ** 2, axis=0).astype(np.complex64)
        return l_p

    @staticmethod
    def compute_wrap_corr(x, wrap_matrix):
        """ Function to compute the wrapping correction in 3 dimensions. 
        Possible efficiency improvement: Compute corrections as six separate arrays (for the edges only) 
        to save memory.
        :param x: Array to which wrapping correction is to be applied
        :param wrap_matrix: Non-cyclic convolution matrix with the wrapping artifacts
        :return: corr: Correction array corresponding to x
        """
        corr = np.zeros(x.shape, dtype='complex64')
        n_correction = wrap_matrix.shape[0]
        # construct slice to select the side pixels
        left = (slice(0, n_correction))
        right = (slice(-n_correction, None))
        for i in range(x.ndim):
            corr[left] += np.tensordot(wrap_matrix, x[right], ((0,), (0,)))
            corr[right] += np.tensordot(wrap_matrix, x[left], ((1,), (0,)))
            x = np.rollaxis(x, -1)
            corr = np.rollaxis(corr, -1)
        return corr

    @staticmethod
    def l_fft_matrix(domain_size, omega=1, truncate_to=1, pixel_size=1.):
        """ Make the operator that does fast convolution with L
        :param domain_size:
        :param omega: Do the fast convolution over a domain size (omega) times (domain_size).
                      Default: omega=1, i.e., gives wrap-around artefacts.
        :param truncate_to: To truncate to (truncate_to) times (domain_size). Must be <= omega.
                  Default truncate_to=1, i.e., truncate to original domain size
        :param pixel_size:
        :return:
        """
        n = omega * domain_size
        l_p = ((2 * np.pi * fftfreq(n[0], pixel_size)) ** 2).astype(np.complex64)
        f = dft_matrix(n[0])
        finv = np.conj(f).T/n[0]
        if omega > 1:
            m = truncate_to * domain_size[0]
            trunc = np.zeros((m, n[0]), dtype=np.complex64)
            trunc[:, :m] = np.eye(m)
            l_fft = trunc @ finv @ np.diag(l_p.ravel()) @ f @ trunc.T
        else:
            l_fft = finv @ np.diag(l_p.ravel()) @ f
        return l_fft.astype(np.complex64)

    def l_fft_operator(self, omega=1, truncate_to=1):
        """ Make the operator that does fast convolution with L
        :param self:
        :param omega: Do the fast convolution over a domain size (omega) times (domain_size).
                      Default: omega=1, i.e., gives wrap-around artefacts.
        :param truncate_to: To truncate to (truncate_to) times (domain_size). Must be <= omega.
                  Default truncate_to=1, i.e., truncate to original domain size
        :return:
        """
        d = self.n_dims
        n = omega * self.domain_size
        l_p = self.laplacian_sq_f(n, d, self.pixel_size)
        f = dft_matrix(n[0])
        finv = np.conj(f).T/n[0]
        if omega > 1:
            m = truncate_to * self.domain_size[0]
            trunc = np.zeros((m, n[0]))
            trunc[:, :m] = np.eye(m)
            l_fft = lambda x: self.dot_ndim(self.dot_ndim(l_p * self.dot_ndim(self.dot_ndim(x, trunc, d),
                                                                              f, d), finv, d), trunc.T, d)
        else:
            l_fft = lambda x: self.dot_ndim(l_p * self.dot_ndim(x, f, d), finv, d)
        return l_fft

    @staticmethod
    def dot_ndim(x, y, n_dims):  # np.einsum useful for this? https://stackoverflow.com/a/48290839/14384909
        """ np.dot(x, y) over all axes of x.
        y is a 2-D array for fast-convolution or related operations that need to be applied to every axis of x """
        for i in range(n_dims):
            x = np.moveaxis(x, i, -1)  # Transpose
            x = np.dot(x, y)
            x = np.moveaxis(x, -1, i)  # Transpose back
        return x.astype(np.complex64)

    def transfer(self, x, subdomain_scaling, patch_shift):
        if patch_shift == -1:
            return self.dot_ndim(x, subdomain_scaling * self.wrap_transfer, self.n_dims)
        elif patch_shift == +1:
            return self.dot_ndim(x, subdomain_scaling * np.flip(self.wrap_transfer), self.n_dims)


# n = np.ones((5, 6, 7), dtype=np.float32)
# source = np.zeros_like(n)
# source[0] = 1.
# base = HelmholtzBase(n=n, source=source, n_domains=1, boundary_widths=5, wrap_correction='wrap_corr')
# print('Done.')
