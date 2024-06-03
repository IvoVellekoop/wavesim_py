import torch
from utilities import is_zero
from .domain import Domain


class HelmholtzDomain(Domain):
    """Represents a single domain of the simulation.

    The `Domain` object encapsulates all data that is stored on a single computation node (e.g. a GPU or a node in a
    cluster), and provides methods to perform the basic operations that the Wavesim algorithm needs.

    Note:
        Domain currently works only for the Helmholtz equation and the PyTorch backend.
        If we want to have further functionality, this class should be refactored.
    """

    def __init__(self,
                 permittivity,
                 pixel_size: float,
                 periodic: tuple[bool, bool, bool],
                 n_boundary: int = 0,
                 n_slots=2,
                 stand_alone=True,
                 Vwrap=None,
                 ):
        """Construct a domain object with the given permittivity and allocate memory.

        Note: the permittivity array is stored in one of the temporary memory slots and will be overwritten during
              processing. This means that no copy is kept (to save memory), and the data should not be used after
              calling this function.
        Note: all operations performed on this domain will use the same pytorch device and data type as the
              permittivity array.

        Attributes:
            permittivity: permittivity map. Must be a 3-dimensional array of complex float32 or float64.
                Its shape (n_x, n_y, n_z) is used to determine the size of the domain, and the device and datatype are
                used for all operations.
            pixel_size: grid spacing (in wavelength units)
            periodic: tuple of three booleans indicating whether the domain is periodic in each dimension.
            n_boundary: Number of pixels used for the boundary correction.
            n_slots: number of arrays used for storing the field and temporary data.
            Vwrap: optional wrapping matrix, when omitted and not in stand-alone mode, the matrix will be computed.

            stand_alone: if True, the domain performs shifting and scaling of the scattering potential (based on the
                permittivity of this domain alone). In this stand-alone mode, no wrapping corrections are applied,
                 making it equivalent to the original Wavesim algorithm.
                 Set to False when part of a multi-domain, where the all subdomains need to be considered together to
                 compute the shift and scale factors.

         """
        permittivity = torch.tensor(permittivity)
        super().__init__(pixel_size, permittivity.shape, permittivity.device)

        # validate input arguments
        if n_slots < 2:
            raise ValueError("n_slots must be at least 2")
        if permittivity.ndim != 3 or not (
                permittivity.dtype == torch.complex64 or permittivity.dtype == torch.complex128):
            raise ValueError(
                f"Permittivity must be 3-dimensional and complex float32 or float64, not {permittivity.dtype}.")
        if any([n_boundary > 0.5 * self.shape[i] and not periodic[i] for i in range(3)]):
            raise ValueError(f"Domain boundary of {n_boundary} is too large for the given domain size {self.shape}")

        self._n_boundary = n_boundary
        self._Bscat = None
        self._periodic = periodic if n_boundary > 0 else \
            [True, True, True]  # allow manually disabling wrapping corrections by setting n_boundary=0
        self._source = None
        self._stand_alone = stand_alone

        # allocate memory for the side pixels
        # note: at the moment, compute_corrections does not support in-place operations,
        # so this pre-allocated memory is not used.

        # edge_slices is a list of 6 pairs of slices, each pair corresponding to one of the six faces of the domain.
        def compute_slice(dd):
            d = dd // 2
            if dd % 2 == 0:
                return (slice(None),) * d + (slice(0, n_boundary),)
            else:
                return (slice(None),) * d + (slice(-n_boundary, None),)

        self.edge_slices = [compute_slice(dd) for dd in range(6)]
        self.edges = [
            None if self._periodic[0] else torch.zeros_like(permittivity[self.edge_slices[0]]),
            None if self._periodic[0] else torch.zeros_like(permittivity[self.edge_slices[1]]),
            None if self._periodic[1] else torch.zeros_like(permittivity[self.edge_slices[2]]),
            None if self._periodic[1] else torch.zeros_like(permittivity[self.edge_slices[3]]),
            None if self._periodic[2] else torch.zeros_like(permittivity[self.edge_slices[4]]),
            None if self._periodic[2] else torch.zeros_like(permittivity[self.edge_slices[5]]),
        ]

        # compute the un-scaled laplacian kernel and the un-scaled wrapping correction matrices
        # This kernel is given by -(px² + py² + pz²), with p_ the Fourier space coordinate
        # We temporarily store the kernel in `propagator_kernel`.
        # The shift and scale functions convert it to 1 / (scale·(L+shift)+1)
        # todo: convert to on-the-fly computation as in MATLAB code so that we don't need to store the kernel
        self.inverse_propagator_kernel = 0.0j
        for dim in range(3):
            self.inverse_propagator_kernel = self.inverse_propagator_kernel + self._laplace_kernel(dim)
        self.propagator_kernel = None  # will be set in initialize_scale

        # allocate storage for temporary data, re-use the memory we got for the raw scattering potential
        # as one of the locations (which will be overwritten later)
        self._x = [permittivity] + [torch.zeros_like(permittivity) for _ in range(n_slots - 1)]

        # compute n²·k₀² (the raw scattering potential)
        # also compute the bounding box holding the values of the scattering potential in the complex plane.
        # note: wavelength [pixels] = 1/self.pixel_size, so k=n·2π·self.pixel_size
        permittivity.mul_(-(2.0 * torch.pi * self.pixel_size) ** 2)
        r_min, r_max = torch.aminmax(permittivity.real)
        i_min, i_max = torch.aminmax(permittivity.imag)
        self.V_bounds = torch.tensor((r_min, r_max, i_min, i_max))

        if stand_alone:
            # When in stand-alone mode, compute scaling factors now.
            self.Vwrap = [None, None, None]
            center = 0.5 * (r_min + r_max)  # + 0.5j * (i_min + i_max)
            V_norm = self.initialize_shift(center)
            self.initialize_scale(0.95j / V_norm)
        elif Vwrap is not None:
            # Use the provided wrapping matrices. This is used to ensure all subdomains use the same wrapping matrix
            self.Vwrap = [W.to(self.device) if W is not None else None for W in Vwrap]
        else:
            # Compute the wrapping correction matrices if none were provided
            # These matrices must be computed before initialize_scale, since they
            # affect the overall scaling.
            # place a point at -1,-1,-1 in slot 1 (which currently holds all zeros)
            # and then convolve the point with the inverse propagator kernel
            # we now have the wrap-around artefacts located at [:,-1,-1], [-1,:,-1] and [-1,-1,:]
            self._x[1][-1, -1, -1] = 1.0
            self.inverse_propagator(1, 1)
            self.Vwrap = [
                _make_wrap_matrix(self._x[1][:, -1, -1], n_boundary) if not self._periodic[0] else None,
                _make_wrap_matrix(self._x[1][-1, :, -1], n_boundary) if not self._periodic[1] else None,
                _make_wrap_matrix(self._x[1][-1, -1, :], n_boundary) if not self._periodic[2] else None,
            ]

        # compute the norm of Vwrap. Worst case: just add all norms
        # the factor 2 is because the same matrix is used twice (for domain transfer and wrapping correction)
        self.Vwrap_norm = 2.0 * sum([torch.linalg.norm(W, ord=2).item() for W in self.Vwrap if W is not None])

    ## Functions implementing the domain interface
    # add_source()
    # clear()
    # get()
    # inner_product()
    # medium()
    # mix()
    # propagator()
    # set_source()
    def add_source(self, slot: int, weight: float):
        if self._source is not None:
            torch.add(self._x[slot], self._source, out=self._x[slot], alpha=weight)

    def clear(self, slot: int):
        """Clears the data in the specified slot"""
        self._x[slot].zero_()

    def get(self, slot: int, copy=False):
        """Returns the data in the specified slot.

        param: slot: slot from which to return the data
        param: copy: if True, returns a copy of the data. Otherwise, may return the original data possible.
                     Note that this data may be overwritten by the next call to domain.
        """
        data = self._x[slot]
        return data.detach().clone() if copy else data

    def set(self, slot: int, data):
        """Copy the date into the specified slot"""
        self._x[slot].copy_(data)

    def inner_product(self, slot_a: int, slot_b: int):
        """Computes the inner product of two data vectors

        Note: the vectors may be represented as multidimensional arrays,
        but these arrays must be contiguous for this operation to work.
        Although it would be possible to use flatten(), this would create a
        copy when the array is not contiguous, causing a hidden performance hit.
        """
        retval = torch.vdot(self._x[slot_a].view(-1), self._x[slot_b].view(-1)).item()
        return retval if slot_a != slot_b else retval.real  # remove small imaginary part if present

    def medium(self, slot_in: int, slot_out: int):
        """Applies the operator 1-Vscat.

        Note: does not apply the wrapping correction. When part of a multi-domain,
        the wrapping correction is applied by the medium() function of the multi-domain object
        and this function should not be called directly.
        """
        torch.mul(self._Bscat, self._x[slot_in], out=self._x[slot_out])

    def mix(self, weight_a: float, slot_a: int, weight_b: float, slot_b: int, slot_out: int):
        """Mixes two data arrays and stores the result in the specified slot"""
        a = self._x[slot_a]
        b = self._x[slot_b]
        out = self._x[slot_out]
        if weight_a == 1.0:
            torch.add(a, b, alpha=weight_b, out=out)
        elif weight_a == 0.0:
            torch.mul(b, weight_b, out=out)
        elif weight_b == 1.0:
            torch.add(b, a, alpha=weight_a, out=out)
        elif weight_b == 0.0:
            torch.mul(a, weight_a, out=out)
        elif weight_a + weight_b == 1.0:
            torch.lerp(a, b, weight_b, out=out)
        elif slot_a == slot_out:
            a.mul_(weight_a)
            a.add_(b, alpha=weight_b)
        else:
            torch.mul(b, weight_b, out=out)
            out.add_(a, alpha=weight_a)

    def propagator(self, slot_in: int, slot_out: int):
        """Applies the operator (L+1)^-1 x.
        """
        # todo: convert to on-the-fly computation
        torch.fft.fftn(self._x[slot_in], out=self._x[slot_out])
        self._x[slot_out].mul_(self.propagator_kernel)
        torch.fft.ifftn(self._x[slot_out], out=self._x[slot_out])

    def inverse_propagator(self, slot_in: int, slot_out: int):
        """Applies the operator (L+1) x .

        This operation is not needed for the Wavesim algorithm, but is provided for testing purposes,
        and can be used to evaluate the residue of the solution.
        """
        # todo: convert to on-the-fly computation
        torch.fft.fftn(self._x[slot_in], out=self._x[slot_out])
        self._x[slot_out].mul_(self.inverse_propagator_kernel)
        torch.fft.ifftn(self._x[slot_out], out=self._x[slot_out])

    def set_source(self, source):
        """Sets the source term for this domain.
        """
        self._source = None
        if source is None or is_zero(source):
            return

        source = source.to(self.device)
        if source.is_sparse:
            source = source.coalesce()
            if len(source.indices()) == 0:
                return

        self._source = source.to(self.device, self._x[0].dtype)

    def initialize_shift(self, shift) -> float:
        """Shifts the scattering potential and propagator kernel, then returns the norm of the shifted operator."""
        self.inverse_propagator_kernel.add_(shift)
        self._x[0].add_(-shift)  # currently holds the scattering potential
        self.shift = shift
        return self._x[0].view(-1).abs().max().item()

    def initialize_scale(self, scale: complex):
        """Scales all operators.

        Computes Bscat (from the temporary storage 0), the propagator kernel (from the temporary value in
        propagator_kernel), and scales Vwrap.
        Attributes:
            scale: Scaling factor of the problem. Its magnitude is chosen such that the
                operator V = scale · (the scattering potential + the wrapping correction)
                 has norm < 1. The complex argument is chosen such that L+V is accretive.
        """

        # B = 1 - scale·(n² k₀² - shift). Scaling and shifting was already applied. 1-... not yet
        self.scale = scale
        self._Bscat = 1.0 - scale * self._x[0]

        # kernel = 1 / (scale·(L + shift) + 1). Shifting was already applied. scaling, +1 and reciprocal not yet
        self.inverse_propagator_kernel.multiply_(scale)
        self.inverse_propagator_kernel.add_(1.0)
        self.propagator_kernel = 1.0 / self.inverse_propagator_kernel
        if self.Vwrap is not None:
            self.Vwrap = [scale * W if W is not None else None for W in self.Vwrap]

    def compute_corrections(self, slot_in: int):
        """Computes the edge corrections by multiplying the first and last pixels of each line with the Vwrap matrix.

        The corrections are stored in self.edges.TODO: re-use this memory
        """
        """ Function to compute the wrapping/transfer corrections in 3 dimensions as six separate arrays
        for the edges of size n_correction (==wrap_matrix.shape[0 or 1])
        :param x: Array to which wrapping correction is to be applied
        :param wrap_matrix: Non-cyclic convolution matrix with the wrapping artifacts
        """
        for edge in range(6):
            axes = [1, ] if edge % 2 == 0 else [0, ]
            dim = edge // 2
            if self.Vwrap[dim] is None:
                continue

            # tensordot(wrap_matrix, x[slice]) gives the correction, but the uncontracted dimension of wrap matrix
            # (of size n_correction) is always at axis=0. It should be at axis=dim,
            # moveaxis moves the non-contracted dimension to the correct position
            # todo: convert to an in-place operation using the 'out' parameter
            self.edges[edge] = torch.moveaxis(
                torch.tensordot(a=self.Vwrap[dim], b=self._x[slot_in][self.edge_slices[edge]], dims=(axes, [dim, ]),
                                out=self.edges[edge]), 0,
                dim)

        return self.edges

    def apply_corrections(self, wrap_corrections, transfer_corrections, slot: int):
        """Apply  -1·wrapping/transfer corrections

        Transfer corrections correspond to a contribution from neighboring domains, and are added to the current domain.
        Wrap corrections correct for the periodicity of the fft. They are subtracted from the domain
        In this case, there is an additional factor of -1 because this function is called from `medium`, which applies
        1-V instead of V.
        Therefore, transfer corrections are now subtracted, and wrap corrections are added.

        :param slot: slot index for the data to which the corrections are applied. Operation is always in-place
        :param wrap_corrections: list of 6 corrections for wrap-around (may contain None for periodic boundary)
        :param transfer_corrections: list of 6 corrections coming from neighboring segments (may contain None for
                                     end of domain)
        """
        for edge in range(6):
            if wrap_corrections[edge] is not None and transfer_corrections[edge] is None:
                self._x[slot][self.edge_slices[edge]] += wrap_corrections[edge]
            elif transfer_corrections[edge] is not None and wrap_corrections[edge] is None:
                self._x[slot][self.edge_slices[edge]] -= transfer_corrections[edge]
            elif transfer_corrections[edge] is not None and wrap_corrections[edge] is not None:
                self._x[slot][self.edge_slices[edge]] += wrap_corrections[edge] - transfer_corrections[edge].to(
                    self.device)
            else:
                pass

    def _laplace_kernel(self, dim):
        """Compute the Fourier-domain kernel for the Laplace operator in the given dimension"""

        # original way (introduces wrapping artifacts in the kernel)
        # return -self.coordinates_f(dim) ** 2

        # new way: uses exact Laplace kernel in real space, and returns Fourier transform of that
        x = self.coordinates(dim, 'periodic')
        if x.numel() == 1:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        x = x * torch.pi / self.pixel_size
        c = torch.cos(x)
        s = torch.sin(x)
        x_kernel = 2.0 * c / x ** 2 - 2.0 * s / x ** 3 + s / x
        x_kernel[0, 0, 0] = 1.0 / 3.0  # remove singularity at x=0
        x_kernel *= -torch.pi ** 2 / self.pixel_size ** 2
        f_kernel = torch.fft.fftn(x_kernel)
        return -f_kernel.real


def _make_wrap_matrix(L_kernel, n_boundary):
    """ Make the matrices for the wrapping correction

    :param L_kernel: the kernel for the laplace operator
    :param n_boundary: the size of the correction matrix
    :param tmp: temporary storage, of the same size as the kernel. MUST be initialized to zero

    Note: the matrices need may not be identical for the different dimensions if the sizes are different
    Note: uses the temporary storage slot 1 for the intermediate results
    """

    # define a single point source at (0,0,0) and compute the (wrapped) convolution
    # with the forward kernel (L+1)
    kernel_section = L_kernel.real.ravel()

    # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
    wrap_matrix = torch.zeros((n_boundary, n_boundary), dtype=kernel_section.dtype, device=kernel_section.device)
    for r in range(n_boundary):
        size = r + 1
        wrap_matrix[r, :] = kernel_section[n_boundary - size:2 * n_boundary - size]
    return wrap_matrix
