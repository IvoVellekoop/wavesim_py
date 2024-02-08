import numpy as np
from itertools import product
# from numpy.fft import fftn, ifftn
from collections import defaultdict
from utilities import laplacian_sq_f, pad_func, preprocess

import torch
from torch.fft import fftn, ifftn
from torchvision.transforms import Lambda
torch.set_default_dtype(torch.float32)


class HelmholtzBase:
    """" Class for generating medium (B) and propagator (L+1)^(-1) operators, scaling,
     and setting up wrapping and transfer corrections """
    def __init__(self, 
                 n=np.ones((1, 1, 1)),  # Refractive index distribution
                 source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
                 wavelength=1.,  # Wavelength in um (micron)
                 ppw=4,  # points per wavelength
                 boundary_widths=10,  # Width of absorbing boundaries
                 n_domains=1,  # Number of subdomains to decompose into, in each dimension
                 wrap_correction=None,  # Wrap-around correction. None or 'wrap_corr' or 'L_omega'
                 omega=10,  # compute the fft over omega times the domain size
                 n_correction=8,  # number of points used in the wrapping correction
                 max_iterations=int(1.e+4),  # Maximum number of iterations
                 setup_operators=True):  # Set up Medium (+corrections) and Propagator operators, and scaling

        (self.n_roi, self.s, self.n_dims, self.boundary_widths, self.boundary_pre, self.boundary_post,
         self.n_domains, self.domain_size, self.omega, self.v_min, self.v_raw, self.device) = (
            preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega))

        # base = preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega)
        # for name, val in base.items():
        #     exec('self.'+name+' = val')

        self.pixel_size = wavelength / ppw  # Grid pixel size in um (micron)

        self.v = None
        self.l_p = None
        self.scaling = None

        self.total_domains = np.prod(self.n_domains).astype(int)  # total number of domains across all dimensions
        if self.total_domains == 1:
            self.wrap_correction = wrap_correction
        else:
            self.wrap_correction = 'wrap_corr'

        smallest_domain_size = np.min(self.domain_size[:self.n_dims])
        if n_correction > smallest_domain_size/3:
            self.n_correction = int(smallest_domain_size/3)
        else:
            self.n_correction = n_correction

        self.wrap_matrix = None
        self.medium_operators = None
        self.propagator_operators = None
        self.l_plus1_operators = None

        # to iterate through subdomains in all dimensions
        self.domains_iterator = list(product(*(range(self.n_domains[i]) for i in range(3))))

        # crop to n_roi, excluding boundaries
        self.crop2roi = tuple([slice(self.boundary_pre[d], -self.boundary_post[d])
                               for d in range(self.n_dims)])

        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]

        # Stopping criteria
        self.max_iterations = int(max_iterations)
        self.threshold_residual = 1.e-6
        self.divergence_limit = 1.e+6

        self.print_details()
        if setup_operators:
            # Get Medium (b = 1 - v + corrections), Propagator (L+1)^(-1), and (L+1) operators, and Scaling
            (self.medium_operators, self.propagator_operators,
             self.l_plus1_operators, self.scaling) = self.make_operators()

    def print_details(self):
        """ Print main information about the problem """
        print(f'\n{self.n_dims} dimensional problem')
        if self.wrap_correction:
            print('Wrap correction: \t', self.wrap_correction)
            print('n_correction: \t', self.n_correction)
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
        if self.wrap_correction == 'wrap_corr':
            # compute the 1-D convolution kernel (brute force) and take the wrapped part of it
            side = torch.zeros(*self.domain_size, device=self.device)
            side[-1, ...] = 1.0
            k_wrap = np.real(ifftn(l_p * fftn(side))[:, 0, 0])  # discard tiny imaginary part due to numerical errors

            # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
            self.wrap_matrix = torch.zeros((self.n_correction, self.n_correction),
                                           dtype=torch.complex64, device=self.device)
            for r in range(self.n_correction):
                size = r + 1
                self.wrap_matrix[r, :] = k_wrap[self.n_correction-size:2*self.n_correction-size]

        # Compute the scaling, scale v. Scaling computation includes wrap_corr if it is used in wrapping &// transfer
        scaling = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)
            v_norm = np.max(np.abs(self.v[patch_slice]))
            if self.wrap_correction == 'wrap_corr':
                v_norm += self.n_dims * torch.linalg.norm(self.wrap_matrix, 2).cpu().numpy()
            scaling[patch] = 0.95/v_norm
            self.v[patch_slice] = scaling[patch] * self.v[patch_slice]  # Scale v patch/subdomain-wise

        # Make b and apply ARL
        b = 1 - self.v
        b = pad_func(b, self.boundary_pre, self.boundary_post, self.n_roi, self.n_dims)
        b = torch.tensor(b).to(self.device)

        # Make the medium operator(s) patch/subdomain-wise
        medium_operators = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)  # get slice/indices for the subdomain patch
            b_patch = b[patch_slice]
            medium_operators[patch] = Lambda(lambda x, b_=b_patch: b_ * x)

        # Make the propagator operator that does fast convolution with (l_p+1)^(-1)
        self.l_p = 1j * (l_p - v0)  # Shift l_p and multiply with 1j (Scaling incorporated inside propagator operator)
        propagator_operators = {}
        l_plus1_operators = {}
        if self.wrap_correction == 'L_omega':
            crop2domain = tuple([slice(0, self.domain_size[i]) for i in range(3)])  # map from padded to original domain
            for patch in self.domains_iterator:
                propagator_operators[patch] = Lambda(lambda x: (ifftn((1 / (scaling[patch] * self.l_p + 1)) *
                                                                      fftn(x, tuple(n_fft))))[crop2domain])
                l_plus1_operators[patch] = Lambda(lambda x: (ifftn((scaling[patch] * self.l_p + 1) *
                                                                   fftn(x, tuple(n_fft))))[crop2domain])
        else:
            for patch in self.domains_iterator:
                propagator_operators[patch] = Lambda(lambda x: ifftn((1 / (scaling[patch] * self.l_p + 1)) * fftn(x)))
                l_plus1_operators[patch] = Lambda(lambda x: ifftn((scaling[patch] * self.l_p + 1) * fftn(x)))

        return medium_operators, propagator_operators, l_plus1_operators, scaling

    def patch_slice(self, patch):
        """ Return the slice, i.e., indices for the current 'patch', i.e., the subdomain """
        return tuple([slice(patch[d] * self.domain_size[d], 
                            patch[d] * self.domain_size[d] + self.domain_size[d]) for d in range(self.n_dims)])

    def medium(self, x, y=None):
        """ Apply medium operators to subdomains/patches of x 
        :param x: Dict of List of arrays to which medium operators are to be applied
        :param y: Dict of List of arrays to add for first medium operator application 
                  in AnySim iteration, i.e., Bx + y
        :return: t: Dict of List of subdomain-wise Bx [+y]
        """
        t = defaultdict(list)
        for patch in self.domains_iterator:
            t[patch] = self.medium_operators[patch](x[patch])
            if y is not None:
                t[patch] += y[patch]
        t = self.apply_corrections(x, t, 'wrapping')
        t = self.apply_corrections(x, t, 'transfer')
        return t
    
    def l_plus1(self, x):
        """ Apply L+1 operators to subdomains/patches of x 
        :param x: Dict of List of arrays to which L+1 operators are to be applied
        :return: t: Dict of List of subdomain-wise (L+1)
        """
        t = defaultdict(list)
        for patch in self.domains_iterator:
            t[patch] = self.l_plus1_operators[patch](x[patch])
        return t

    def propagator(self, x):
        """ Apply propagator operators (L+1)^-1 to subdomains/patches of x 
        :param x: Dict of List of arrays to which propagator operators are to be applied
        :return: t: Dict of List of subdomain-wise (L+1)^-1
        """
        t = defaultdict(list)
        for patch in self.domains_iterator:
            t[patch] = self.propagator_operators[patch](x[patch])
        return t

    @staticmethod
    def compute_corrections(x, wrap_matrix):
        """ Function to compute the transfer corrections in 3 dimensions. 
            Corrections computed as six separate arrays (for the edges only) to save memory
        :param x: Array to which wrapping correction is to be applied
        :param wrap_matrix: Non-cyclic convolution matrix with the wrapping artifacts
        :return: corr: Dict of List of correction arrays corresponding to x
        """
        # construct slices to select the side pixels
        n_correction = wrap_matrix.shape[0]
        left = np.arange(0, n_correction)
        right = np.arange(-n_correction, 0)

        corr_dict = defaultdict(list)
        for dim in range(3):
            for idx_shift in [-1, +1]:  # Correction for left [-1] or right [+1] edge
                edge = (dim, idx_shift)
                if x.shape[0] == 1:
                    corr_dict[edge] = 0.0
                else:
                    if idx_shift == -1:
                        corr_dict[edge] = torch.tensordot(wrap_matrix, x[left], ([1,], [0,]))
                    if idx_shift == +1:
                        corr_dict[edge] = torch.tensordot(wrap_matrix, x[right], ([0,], [0,]))
                    # Reshape back to original size
                    for _ in range(edge[0]):
                        corr_dict[edge] = corr_dict[edge].permute(2, 0, 1)
            x = x.permute((1, 2, 0))

        return corr_dict

    def apply_corrections(self, f, t, corr_type):
        """ 
        :param f: Dict of List of arrays for which correction is computed
        :param t: Dict of List of arrays to which correction is to be applied
        :param corr_type: Type of correction: 'wrapping' or 'transfer'
        """
        if corr_type == 'transfer' and self.total_domains == 1:
            return t
        elif corr_type == 'wrapping' and self.wrap_correction != 'wrap_corr':
            return t
        elif corr_type != 'wrapping' and corr_type != 'transfer':
            raise TypeError("Specify corr_type = 'wrapping' or 'transfer'")
        else:
            # multiplier for adding (m = 1) or subtracting (m = -1) correction
            if corr_type == 'wrapping':
                m = 1  
            else:
                m = -1

            # For indexing edge patch of the domain of size n_correction (default=8) and applying correction to it
            left = [(slice(None),) * d + (slice(0, self.n_correction),) for d in range(3)]
            right = [(slice(None),) * d + (slice(-self.n_correction, None),) for d in range(3)]

            for from_patch in self.domains_iterator:
                # get Dict of List of (six) correction arrays corresponding to f's left and right edges in each axis
                f_corr = self.compute_corrections(f[from_patch], self.wrap_matrix)
                if corr_type == 'wrapping':
                    to_patch = from_patch
                for d in range(3):
                    for idx_shift in [-1, +1]:  # Transfer from left (-1)/right (+1) of "f" to right/left of "t"
                        if corr_type == 'transfer':
                            patch_shift = np.zeros_like(from_patch)
                            patch_shift[d] = idx_shift
                            to_patch = tuple(np.array(from_patch) + patch_shift)  # get neighbouring subdomain location

                        if to_patch in self.domains_iterator:  # check if subdomain exists
                            if idx_shift == -1:
                                t[to_patch][right[d]] += m * 1j * self.scaling[to_patch] * f_corr[d, idx_shift]
                            if idx_shift == +1:
                                t[to_patch][left[d]] += m * 1j * self.scaling[to_patch] * f_corr[d, idx_shift]
            return t
