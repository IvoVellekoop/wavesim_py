import numpy as np
from itertools import chain, product
from collections import defaultdict
from utilities import laplacian_sq_f, preprocess
import torch
from torch.fft import fftn, ifftn

torch.set_default_dtype(torch.float32)


class HelmholtzBase:
    """" Class for generating medium (B) and propagator (L+1)^(-1) operators, scaling,
     and setting up wrapping and transfer corrections """

    def __init__(self,
                 n=np.ones((1, 1, 1)),
                 source=np.zeros((1, 1, 1)),
                 wavelength=1.,
                 ppw=4,
                 boundary_widths=10,
                 n_domains=1,
                 wrap_correction=None,
                 omega=10,
                 n_correction=8,
                 scaling=None,
                 max_iterations=int(1.e+4),
                 setup_operators=True,
                 device=None):
        """ Takes input parameters for the HelmholtzBase class (and sets up the operators)
        :param n: Refractive index distribution. Default is np.ones((1, 1, 1)).
        :param source: Source term. Same size as n. Default is np.zeros((1, 1, 1)).
        :param wavelength: Wavelength in um (micron). Default is 1.
        :param ppw: Points per wavelength. Default is 4.
        :param boundary_widths: Width of absorbing boundaries. Default is 10.
        :param n_domains: Number of subdomains to decompose into, in each dimension. Default is 1.
        :param wrap_correction: Wrap-around correction. None or 'wrap_corr' or 'L_omega'. Default is None.
        :param omega: Compute the fft over omega times the domain size. Default is 10.
        :param n_correction: Number of points used in the wrapping correction. Default is 8.
        :param scaling: None or float for custom scaling. Default is None.
        :param max_iterations: Maximum number of iterations. Default is int(1.e+4).
        :param setup_operators: Bool, set up scaling, Medium (+corrections) and Propagator operators. Default is True.
        :param device: Device to use for computation. Default is None (all available devices are used).
        """

        # Takes the input parameters and returns these in the appropriate format, with more parameters for setting up
        # the Medium (+corrections) and Propagator operators, and scaling
        (self.n_roi, self.s, self.n_dims, self.boundary_widths, self.boundary_pre, self.boundary_post,
         self.n_domains, self.domain_size, self.omega, self.v_min, self.v_raw) = (
            preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega))

        # base = preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega)
        # for name, val in base.items():
        #     exec('self.'+name+' = val')

        self.pixel_size = wavelength / ppw  # Grid pixel size in um (micron)

        self.scaling = scaling

        self.total_domains = np.prod(self.n_domains).astype(int)  # total number of domains across all dimensions
        if self.total_domains == 1:
            self.wrap_correction = wrap_correction
        else:
            self.wrap_correction = 'wrap_corr'

        smallest_domain_size = np.min(self.domain_size[:self.n_dims])
        if n_correction > smallest_domain_size / 3:
            self.n_correction = int(smallest_domain_size / 3)
        else:
            self.n_correction = n_correction

        self.propagator_operators = None
        self.medium_operators = None
        self.wrap_matrices = None
        self.scaling = None
        self.l_plus1_operators = None

        # to iterate through subdomains in all dimensions
        self.domains_iterator = list(product(*(range(self.n_domains[i]) for i in range(3))))
        
        self.devices = {}  # to set a device for each subdomain
        self.device_list = []  # to make a list of all devices (to send wrapping matrix to)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            for idx, patch in enumerate(self.domains_iterator):
                device = f'cuda:{idx % device_count}' if device_count > 0 else 'cpu'
                self.devices[patch] = device
            self.device_list = [f'cuda:{i}' for i in range(device_count)] if device_count > 0 else ['cpu']
        else:
            self.device = device
            for patch in self.domains_iterator:
                self.devices[patch] = device
            self.device_list = ['cpu']

        # crop to n_roi, excluding boundaries
        self.crop2roi = tuple([slice(self.boundary_pre[d], -self.boundary_post[d])
                               for d in range(self.n_dims)])
        # map from padded to original domain
        self.crop2domain = None  # Only applicable when wrap_correction='L_omega'

        self.alpha = 0.75  # ~step size of the Richardson iteration \in (0,1]

        # Stopping criteria
        self.max_iterations = int(max_iterations)
        self.threshold_residual = 1.e-6
        self.divergence_limit = 1.e+6

        self.print_details()
        if setup_operators:
            # Get Propagator (L+1)^(-1) and Medium (b = 1 - v) operators, Wrapping Matrix (for wrapping and transfer
            # corrections), Scaling and (L+1) operators
            (self.propagator_operators, self.medium_operators, self.wrap_matrices, self.scaling,
             self.l_plus1_operators) = self.make_operators()

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
        v0 = 0.5 * (np.max(np.real(self.v_raw)) + np.min(np.real(self.v_raw)))
        v = -1j * (self.v_raw - v0)  # shift v_raw

        # Make the wrap_corr operator (If wrap_correction=True)
        if self.wrap_correction == 'L_omega':
            n_fft = self.domain_size * self.omega
        else:
            n_fft = self.domain_size.copy()

        wrap_matrices = {}
        if self.wrap_correction == 'wrap_corr':
            wrap_matrix = self.make_wrap_matrix(n_fft)
            for device in self.device_list:
                wrap_matrices[device] = wrap_matrix.to(device)

        # Compute the scaling, scale v. Scaling computation includes wrap_corr if it is used in wrapping &// transfer
        scaling = {}
        if self.scaling:
            for patch in self.domains_iterator:
                scaling[patch] = self.scaling
                patch_slice = self.patch_slice(patch)
                v[patch_slice] = scaling[patch] * v[patch_slice]  # Scale v patch/subdomain-wise
        else:
            # multiplier for v_norm computation. m=2 for wrapping + transfer correction when n_domain > 1
            m = 2 if self.total_domains > 1 else 1
            # Scaling option 1. Compute one scaling for entire domain
            v_norm = np.max(np.abs(v))
            if self.wrap_correction == 'wrap_corr':
                v_norm += m * self.n_dims * torch.linalg.norm(wrap_matrix, 2).cpu().numpy()
            v_norm = np.maximum(v_norm, self.v_min)  # minimum scaling if medium empty
            for patch in self.domains_iterator:
                patch_slice = self.patch_slice(patch)
                # # Scaling option 2. Compute scaling patch/subdomain-wise
                # v_norm = np.max(np.abs(v[patch_slice]))
                # if self.wrap_correction == 'wrap_corr':
                #     v_norm += m * self.n_dims * np.linalg.norm(wrap_matrix, 2)
                # v_norm = np.maximum(v_norm, self.v_min)  # minimum scaling if medium empty
                scaling[patch] = 0.95 / v_norm
                v[patch_slice] = scaling[patch] * v[patch_slice]  # Scale v patch/subdomain-wise

        # Make b
        b = torch.tensor(1 - v, dtype=torch.complex64, device=self.device)

        # Make the medium operator(s) patch/subdomain-wise
        medium_operators = {}
        for patch in self.domains_iterator:
            patch_slice = self.patch_slice(patch)  # get slice/indices for the subdomain patch
            b_p = b[patch_slice].to(self.devices[patch])
            medium_operators[patch] = lambda x, b_=b_p: b_ * x

        # # Make the propagator operator that does fast convolution with (l_p+1)^(-1)
        propagator_operators = {}
        l_plus1_operators = {}
        if self.wrap_correction == 'L_omega':
            # map from padded to original domain
            self.crop2domain = tuple([slice(0, self.domain_size[i]) for i in range(3)])
            for patch in self.domains_iterator:
                propagator_operators[patch] = lambda x, p_=patch: (
                    ifftn(1 / (1j * scaling[p_] *
                               (laplacian_sq_f(self.n_dims, n_fft, self.pixel_size, self.devices[p_]) - v0) + 1) *
                          fftn(x, tuple(n_fft))))[self.crop2domain]
                l_plus1_operators[patch] = lambda x, p_=patch: (
                    ifftn((1j * scaling[p_] *
                           (laplacian_sq_f(self.n_dims, n_fft, self.pixel_size, self.devices[p_]) - v0) + 1) *
                          fftn(x, tuple(n_fft))))
        else:
            for patch in self.domains_iterator:
                propagator_operators[patch] = lambda x, p_=patch: ifftn(
                    1 / (1j * scaling[p_] *
                         (laplacian_sq_f(self.n_dims, n_fft, self.pixel_size, self.devices[p_]) - v0) + 1) *
                    fftn(x))
                l_plus1_operators[patch] = lambda x, p_=patch: ifftn(
                    (1j * scaling[p_] *
                     (laplacian_sq_f(self.n_dims, n_fft, self.pixel_size, self.devices[p_]) - v0) + 1) *
                    fftn(x))

        return propagator_operators, medium_operators, wrap_matrices, scaling, l_plus1_operators

    def make_wrap_matrix(self, n_fft):
        """ Make the wrap matrix for the wrapping correction
        :param n_fft: Size of the padded domain for the fft 
        :return: wrap_matrix: A square matrix of side n_correction"""
        # compute the 1-D convolution kernel (brute force) and take the wrapped part of it
        side = torch.zeros(*self.domain_size, device=self.device)
        side[-1, ...] = 1.0
        k_wrap = np.real(ifftn(laplacian_sq_f(self.n_dims, n_fft, self.pixel_size, device=self.device) *
                               fftn(side))[:, 0, 0])  # discard tiny imaginary part due to numerical errors

        # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
        wrap_matrix = torch.zeros((self.n_correction, self.n_correction), dtype=torch.complex64, device=self.device)
        for r in range(self.n_correction):
            size = r + 1
            wrap_matrix[r, :] = k_wrap[self.n_correction - size:2 * self.n_correction - size]
        return wrap_matrix

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
        t = self.apply_corrections(x, t, 'wrapping')
        t = self.apply_corrections(x, t, 'transfer')
        if y:
            for patch in self.domains_iterator:
                t[patch] += y[patch]
        return t

    def l_plus1(self, x, crop=True):
        """ Apply L+1 operators to subdomains/patches of x 
        :param x: Dict of List of arrays to which L+1 operators are to be applied
        :param crop: Bool, whether to crop (L+1)x or not [only applicable when wrap_correction = 'L_omega']
        :return: t: Dict of List of subdomain-wise (L+1)
        """
        t = defaultdict(list)
        if crop and self.wrap_correction == 'L_omega':
            for patch in self.domains_iterator:
                t[patch] = self.l_plus1_operators[patch](x[patch])[self.crop2domain]
        else:
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
    def compute_corrections(x, wrap_matrix, device):
        """ Function to compute the wrapping/transfer corrections in 3 dimensions as six separate arrays
        for the edges of size n_correction (==wrap_matrix.shape[0 or 1])
        :param x: Array to which wrapping correction is to be applied
        :param wrap_matrix: Non-cyclic convolution matrix with the wrapping artifacts
        :param device: To ensure all corrections are on the same device
        :return: corr: Dict of List of correction arrays corresponding to x
        """

        # construct slices to select the side pixels
        n_correction = wrap_matrix.shape[0]
        left = [(slice(None),) * d + (slice(0, n_correction),) for d in range(3)]
        right = [(slice(None),) * d + (slice(-n_correction, None),) for d in range(3)]

        corr_dict = defaultdict(list)
        for dim in range(3):
            if x.shape[dim] != 1:
                # (dim, -1) indicates correction for left edge in dimension dim, (dim, +1) for the right edge

                # tensordot(wrap_matrix, x[slice]) gives the correction, but the uncontracted dimension of wrap matrix
                # (of size n_correction) is always at axis=0. It should be at axis=dim, and torch.moveaxis() does this
                corr_dict[(dim, -1)] = torch.moveaxis(torch.tensordot(wrap_matrix, x[left[dim]], ([1, ], [dim, ])),
                                                      0, dim)
                corr_dict[(dim, +1)] = torch.moveaxis(torch.tensordot(wrap_matrix, x[right[dim]], ([0, ], [dim, ])),
                                                      0, dim)
            else:  # no correction if dim doesn't exist
                zero_tensor = torch.zeros(1, device=device)
                corr_dict[(dim, -1)] = zero_tensor
                corr_dict[(dim, +1)] = zero_tensor
        return corr_dict

    def apply_corrections(self, f, t, corr_type, im=True):
        """ Function to apply the wrapping/transfer corrections computed from compute_corrections()
        :param f: Dict of List of arrays for which correction is computed
        :param t: Dict of List of arrays to which correction is to be applied
        :param corr_type: Type of correction: 'wrapping' or 'transfer'
        :param im: True/False for whether to multiply correction by 1j or not
        :return: t: With corrections (if any) added to original t
        """
        if corr_type == 'transfer' and self.total_domains == 1:
            return t
        elif corr_type == 'wrapping' and self.wrap_correction != 'wrap_corr':
            return t
        elif corr_type != 'wrapping' and corr_type != 'transfer':
            raise TypeError("Specify corr_type = 'wrapping' or 'transfer'")
        else:
            # multiplier for adding (m = 1) or subtracting (m = -1) correction
            m = 1
            if corr_type == 'transfer':
                m = -1

            # multiply correction by 1j (im=True) or not (im=False). For symmetry and accretivity check of corrections
            if im:
                m = m * 1j

            # For indexing edge patch of the domain of size n_correction (default=8) and applying correction to it
            # first n_correction points in each axis
            left = [(slice(None),) * d + (slice(0, self.n_correction),) for d in range(3)]
            # last n_correction points in each axis
            right = [(slice(None),) * d + (slice(-self.n_correction, None),) for d in range(3)]
            # rearrange slices in dictionary similar to compute_corrections output
            slices = list(chain.from_iterable(zip(right, left)))  # interleave right and left in an alternating way
            edges = list(product(*(range(3), [-1, 1])))  # [(dim, -1 or 1) for dim in range(3)]
            edge_dict = dict(zip(edges, slices))  # dictionary with keys similar to compute_corrections output

            for from_patch in self.domains_iterator:
                # get Dict of List of (six) correction arrays corresponding to f's left and right edges in each axis
                device = self.devices[from_patch]
                f_corr = self.compute_corrections(f[from_patch], self.wrap_matrices[device], device)
                if corr_type == 'wrapping':
                    to_patch = from_patch  # wrapping corrections added to same patch
                for d, i in edges:  # d: dim (0,1,2), i: correction for right (-1) or left (+1) edge
                    if corr_type == 'transfer':
                        # shift current patch by -1 or 1 in dim d to find neighbouring patch
                        # e.g. for d = 1, patch_shift = (0, i, 0) = (0, 1, 0) or (0, -1, 0)
                        patch_shift = (*(0,) * d, i, *(0,) * (3 - d - 1))
                        to_patch = tuple(np.add(from_patch, patch_shift))  # neighbouring patch
                    if to_patch in self.domains_iterator:  # check if patch/subdomain exists
                        t[to_patch][edge_dict[d, i]] += (m * self.scaling[to_patch] *
                                                         f_corr[d, i].to(self.devices[to_patch]))
            return t
