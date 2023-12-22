import numpy as np
from itertools import product
from scipy.fft import fftn, ifftn
from utilities import laplacian_sq_f, pad_func, preprocess


class HelmholtzBase:
    """" Class for generating medium (B) and propagator (L+1)^(-1) operators, scaling,
     and setting up wrapping and transfer corrections """
    def __init__(self, 
                 n=np.ones((1, 1, 1)),  # Refractive index distribution
                 source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
                 wavelength=1.,  # Wavelength in um (micron)
                 ppw=4,  # points per wavelength
                 boundary_widths=(20, 20, 20),  # Width of absorbing boundaries
                 n_domains=(1, 1, 1),  # Number of subdomains to decompose into, in each dimension
                 wrap_correction=None,  # Wrap-around correction. None or 'wrap_corr' or 'L_omega'
                 omega=10,  # compute the fft over omega times the domain size
                 n_correction=8,  # number of points used in the wrapping correction
                 max_iterations=int(1.e+4),  # Maximum number of iterations
                 setup_operators=True):  # Set up Medium (+corrections) and Propagator operators, and scaling

        self.pixel_size = wavelength / ppw  # Grid pixel size in um (micron)

        self.v = None
        self.l_p = None
        self.scaling = None

        self.wrap_correction = wrap_correction
        self.n_correction = n_correction
        self.wrap_corr = None
        self.wrap_transfer = None

        self.medium_operators = None
        self.propagator = None

        (self.n_roi, self.s, self.n_dims, self.boundary_widths, self.boundary_pre, self.boundary_post,
         self.n_domains, self.domain_size, self.omega, self.v_min, self.v_raw) = (
            preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega))

        # base = preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega)
        # for name, val in base.items():
        #     exec('self.'+name+' = val')

        # to iterate through subdomains in all dimensions
        self.domains_iterator = list(product(*(range(self.n_domains[i]) for i in range(3))))
        
        self.total_domains = np.prod(self.n_domains).astype(int)  # total number of domains across all dimensions
        
        self.crop2roi = tuple([slice(self.boundary_pre[d], -self.boundary_post[d])
                               for d in range(self.n_dims)])  # crop to n_roi, excluding boundaries

        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]

        # Stopping criteria
        self.max_iterations = int(max_iterations)
        self.threshold_residual = 1.e-6
        self.divergence_limit = 1.e+6

        self.print_details()
        if setup_operators:  # Get Medium (b = 1 - v + corrections) and Propagator (L+1)^(-1) operators, and scaling
            self.medium_operators, self.propagator, self.scaling = self.make_operators()

    def print_details(self):
        """ Print main information about the problem """
        print(f'\n{self.n_dims} dimensional problem')
        if self.wrap_correction:
            print('Wrap correction: \t', self.wrap_correction)
        print('Boundaries widths (Pre): \t', self.boundary_pre)
        print('\t\t (Post): \t', self.boundary_post)
        if self.total_domains > 1:
            print(f'Decomposing into {self.n_domains} domains of size {self.domain_size}')

    def make_operators(self):
        """ Make the medium and propagator operators, and, if applicable,
        the wrapping correction and wrapping transfer operators.
        :return: medium_operators, propagator, scaling"""
        # Make v
        v0 = (np.max(np.real(self.v_raw)) + np.min(np.real(self.v_raw))) / 2
        v0 = v0 + 1j * self.v_min
        self.v = -1j * (self.v_raw - v0)  # shift v_raw

        # Make the wrap_corr operator (If wrap_correction=True)
        if self.wrap_correction == 'L_omega':
            n_fft = self.domain_size * self.omega
        else:
            n_fft = self.domain_size.copy()
        
        l_p = laplacian_sq_f(self.n_dims, n_fft, self.pixel_size)  # Fourier coordinates in n_dims
        if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
            # compute the 1-D convolution kernel (brute force) and take the wrapped part of it
            side = np.zeros(self.domain_size, dtype=np.complex64)
            side[-1, ...] = 1.0
            k_wrap = np.real(ifftn(l_p * fftn(side))[:, 0, 0])  # discard tiny imaginary part due to numerical errors

            # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
            wrap_matrix = np.zeros((self.n_correction, self.n_correction), dtype=np.complex64)
            for r in range(self.n_correction):
                size = r + 1
                wrap_matrix[r, :] = k_wrap[self.n_correction-size:2*self.n_correction-size]

            self.wrap_corr = lambda x, idx_shift='all': 1j * self.compute_wrap_corr(x, wrap_matrix, idx_shift)

        # Compute the scaling, scale v. Scaling computation includes wrap_corr if it is used in wrapping &// transfer
        scaling = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)
            v_norm = np.max(np.abs(self.v[patch_slice]))
            if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
                v_norm += self.n_dims * np.linalg.norm(wrap_matrix, 2)
            scaling[patch] = 0.95/v_norm
            self.v[patch_slice] = scaling[patch] * self.v[patch_slice]  # Scale v patch/subdomain-wise

        # Make b and apply ARL
        b = 1 - self.v
        b = pad_func(b, self.boundary_pre, self.boundary_post, self.n_roi, self.n_dims)
        
        # Make the medium operator(s) patch/subdomain-wise
        medium_operators = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)  # get slice/indices for the subdomain patch
            b_patch = b[patch_slice]
            if self.wrap_correction == 'wrap_corr' or self.total_domains > 1:
                medium_operators[patch] = lambda x, b_ = b_patch: (b_ * x + scaling[patch] * self.wrap_corr(x))
            else:
                medium_operators[patch] = lambda x, b_ = b_patch: b_ * x

        # Make the propagator operator that does fast convolution with (l_p+1)^(-1)
        self.l_p = 1j * (l_p - v0)  # Shift l_p and multiply with 1j (Scaling incorporated inside propagator operator)
        if self.wrap_correction == 'L_omega':
            crop2domain = tuple([slice(0, self.domain_size[i]) for i in range(3)])  # map from padded to original domain
            propagator = lambda x, subdomain_scaling: (ifftn((1 / (subdomain_scaling * self.l_p + 1)) *
                                                       fftn(x, n_fft)))[crop2domain]
        else:
            propagator = lambda x, subdomain_scaling: ifftn((1 / (subdomain_scaling * self.l_p + 1)) * fftn(x))

        return medium_operators, propagator, scaling

    def patch_slice(self, patch):
        """ Return the slice, i.e., indices for the current 'patch', i.e., the subdomain """
        return tuple([slice(patch[d] * self.domain_size[d], 
                            patch[d] * self.domain_size[d] + self.domain_size[d]) for d in range(self.n_dims)])

    @staticmethod
    def compute_wrap_corr(x, wrap_matrix, idx_shift='all'):
        """ Function to compute the wrapping correction in 3 dimensions. 
        Possible efficiency improvement: Compute corrections as six separate arrays (for the edges only) to save memory
        :param x: Array to which wrapping correction is to be applied
        :param wrap_matrix: Non-cyclic convolution matrix with the wrapping artifacts
        :param idx_shift: Apply correction to left [-1], right [+1], or both ['all'] edges
        :return: corr: Correction array corresponding to x
        """
        corr = np.zeros(x.shape, dtype='complex64')
        n_correction = wrap_matrix.shape[0]
        # construct slice to select the side pixels
        left = (slice(0, n_correction))
        right = (slice(-n_correction, None))
        n_dims = np.squeeze(x).ndim
        for _ in range(n_dims):
            if idx_shift == -1 or idx_shift == 'all':
                corr[left] += np.tensordot(wrap_matrix, x[right], ((0,), (0,)))
            if idx_shift == +1 or idx_shift == 'all':
                corr[right] += np.tensordot(wrap_matrix, x[left], ((1,), (0,)))
            x = x.transpose((1, 2, 0))
            corr = corr.transpose((1, 2, 0))

        # Reshape corr back to original size
        for _ in range(3-n_dims):
            corr = corr.transpose((1, 2, 0))

        return corr

    def transfer_correction(self, x, current_patch, t1=0):
        """ Transfer correction from neighbouring subdomains to be added to t1 of current subdomain
        :param x: Dictionary with all patches/subdomains of x
        :param current_patch: Current subdomain, as a 3-element position tuple
        :return: x_transfer: Field(s) to transfer
        """
        x_transfer = np.zeros_like(x[current_patch], dtype=np.complex64)
        for d in range(self.n_dims):
            for idx_shift in [-1, +1]:  # Transfer wrt previous (-1) and next (+1) subdomain in axis (d)
                patch_shift = np.zeros_like(current_patch)
                patch_shift[d] = idx_shift
                neighbour_patch = tuple(np.array(current_patch) + patch_shift)  # get neighbouring subdomain location
                if neighbour_patch in self.domains_iterator:  # check if subdomain exists
                    x_transfer += self.scaling[current_patch] * self.wrap_corr(x[neighbour_patch], idx_shift)
        return x_transfer
