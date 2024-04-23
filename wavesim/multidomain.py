import numpy as np
import torch
from torch import tensor
from utilities import partition, combine, list_to_array
from .domain import Domain
from .helmholtzdomain import HelmholtzDomain


class MultiDomain(Domain):
    """" Class for generating medium (B) and propagator (L+1)^(-1) operators, scaling,
     and setting up wrapping and transfer corrections """

    def __init__(self,
                 refractive_index,
                 pixel_size: float,
                 periodic: tuple[bool, bool, bool],
                 n_domains: tuple[int, int, int] = (1, 1, 1),
                 n_boundary: int = 8):
        """ Takes input parameters for the HelmholtzBase class (and sets up the operators)
        :param refractive_index: Refractive index distribution, must be 3-d.
        :param pixel_size: Grid spacing in wavelengths.
        :param periodic: Indicates for each dimension whether the simulation is periodic or not.
            For periodic dimensions, the field is wrapped around the domain.
        :param n_domains: number of domains to split the simulation into.
            If the domain size is not divisible by n_domains, the last domain will be slightly smaller than the other ones.
            In the future, the domain size may be adjusted to have an efficient fourier transform.
            Default is (1,1,1), no domain decomposition.
        :param n_boundary: Number of points used in the wrapping and domain transfer correction. Default is 8.
        """

        # Takes the input parameters and returns these in the appropriate format, with more parameters for setting up
        # the Medium (+corrections) and Propagator operators, and scaling
        # (self.n_roi, self.s, self.n_dims, self.boundary_widths, self.boundary_pre, self.boundary_post,
        # self.n_domains, self.domain_size, self.omega, self.v_min, self.v_raw) = (
        #   preprocess(n, pixel_size, n_domains))

        # validata input parameters
        if not refractive_index.ndim == 3:
            raise ValueError("The refractive index must be a 3D array")
        if not len(n_domains) == 3:
            raise ValueError("The number of domains must be a 3-tuple")

        # enumerate the cuda devices. We will assign the domains to the devices in a round-robin fashion.
        # we use the first GPU as primary device
        devices = [f'cuda:{device_id}' for device_id in
                   range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']
        super().__init__(pixel_size, refractive_index.shape, torch.device(devices[0]))
        self.periodic = np.array(periodic)
       
        # compute domain boundaries in each dimension
        if any([(n_boundary > self.shape[i] / n_domains[i] // 2) and not periodic[i] for i in range(3)]):
            raise ValueError(f"Domain boundary of {n_boundary} is too small for the given domain size")
        self.domains = np.empty(n_domains, dtype=HelmholtzDomain)
        self.n_domains = n_domains

        # distribute the refractive index map over the subdomains.
        ri_domains = partition(refractive_index, self.n_domains)
        for domain_index, ri_domain in enumerate(ri_domains.flat):
            ri_domain = torch.tensor(ri_domain, dtype=torch.complex64, device=devices[domain_index % len(devices)])
            self.domains.flat[domain_index] = HelmholtzDomain(refractive_index=ri_domain, pixel_size=pixel_size,
                                                              n_boundary=n_boundary, periodic=periodic,
                                                              stand_alone=False)

        # determine the optimal shift
        limits = np.array([domain.V_bounds for domain in self.domains.flat])
        r_min = np.min(limits[:, 0])
        r_max = np.max(limits[:, 1])
        i_min = np.min(limits[:, 2])
        i_max = np.max(limits[:, 3])
        center = 0.5 * (r_min + r_max) + 0.5j * (i_min + i_max)

        # shift L and V to minimize norm of V
        Vscat_norm = 0.0
        Vwrap_norm = 0.0
        for domain in self.domains.flat:
            Vscat_norm = np.maximum(Vscat_norm, domain.initialize_shift(center))
            Vwrap_norm = np.maximum(Vwrap_norm, domain.Vwrap_norm)

        # compute the scaling factor
        # apply the scaling to compute the final form of all operators in the iteration
        self.shift = center
        self.scale = 0.95j / (Vscat_norm + Vwrap_norm)
        for domain in self.domains.flat:
            domain.initialize_scale(self.scale)

    ## Functions implementing the domain interface
    # add_source()
    # clear()
    # get()
    # inner_product()
    # medium()
    # mix()
    # propagator()
    # set_source()
    def add_source(self, slot: int):
        """ Add the source to the field in slot """
        for domain in self.domains.flat:
            domain.add_source(slot)

    def clear(self, slot: int):
        """ Clear the field in the specified slot """
        for domain in self.domains.flat:
            domain.clear(slot)

    def get(self, slot: int, device=None):
        """ Get the field in the specified slot, this gathers the fields from all subdomains and puts them in one big array

         :param: device: device on which to store the data. Defaults to the primary device
        """
        domain_data = list_to_array([domain.get(slot) for domain in self.domains.flat], 1).reshape(self.domains.shape)
        return combine(domain_data, device)

    def set(self, slot: int, data):
        """Copy the date into the specified slot"""
        parts = partition(data, self.n_domains)
        for domain, part in zip(self.domains.flat, parts.flat):
            domain.set(slot, part)

    def inner_product(self, slot_a: int, slot_b: int):
        """ Compute the inner product of the fields in slots a and b

        Note: use sqrt(inner_product(slot_a, slot_a)) to compute the norm of the field in slot_a.
        There is a large but inconsistent difference in performance between
        vdot and linalg.norm. Execution time can vary a factor of 3 or more between the two, depending on the input size
        and whether the function is executed on the CPU or the GPU.
        """
        inner_product = 0.0
        for domain in self.domains.flat:
            inner_product += domain.inner_product(slot_a, slot_b)
        return inner_product

    def medium(self, slot_in: int, slot_out: int):
        """ Apply the medium operator B, including wrapping corrections
        """
        domain_edges = [domain.compute_corrections(slot_in) for domain in self.domains.flat]
        domain_edges = list_to_array(domain_edges, 2).reshape(*self.domains.shape, 6)

        for domain in self.domains.flat:
            domain.medium(slot_in, slot_out)

        # apply wrapping corrections. We subtract each correction from
        # the opposite side of the domain to compensate for the wrapping.
        # also, we add each correction to the opposite side of the neighbouring domain
        for idx, domain in enumerate(self.domains.flat):
            x0, x1, x2 = np.unravel_index(idx, self.domains.shape)
            # for the wrap corrections, take the corrections for this domain and swap them
            wrap_corrections = domain_edges[x0, x1, x2, (1, 0, 3, 2, 5, 4)]

            # for the transfer corrections, take the corrections for this domain and swap them
            transfer_corrections = [
                domain_edges[x0 - 1, x1, x2, 1] if x0 > 0 else None,
                domain_edges[x0 + 1, x1, x2, 0] if x0 < self.domains.shape[0] - 1 else None,
                domain_edges[x0, x1 - 1, x2, 3] if x1 > 0 else None,
                domain_edges[x0, x1 + 1, x2, 2] if x1 < self.domains.shape[1] - 1 else None,
                domain_edges[x0, x1, x2 - 1, 5] if x2 > 0 else None,
                domain_edges[x0, x1, x2 + 1, 4] if x2 < self.domains.shape[2] - 1 else None
            ]

            domain.apply_corrections(wrap_corrections, transfer_corrections, slot_out)

    def mix(self, weight_a: float, slot_a: int, weight_b: float, slot_b: int, slot_out: int):
        """ Mix the fields in slots a and b and store the result in slot_out """
        for domain in self.domains.flat:
            domain.mix(weight_a, slot_a, weight_b, slot_b, slot_out)

    def propagator(self, slot_in: int, slot_out: int):
        """ Apply propagator operators (L+1)^-1 to subdomains/patches of x
        """
        for domain in self.domains.flat:
            domain.propagator(slot_in, slot_out)

    def inverse_propagator(self, slot_in: int, slot_out: int):
        """ Apply inverse propagator operators L+1 to subdomains/patches of x
        """
        for domain in self.domains.flat:
            domain.inverse_propagator(slot_in, slot_out)

    def set_source(self, source):
        """ Split the source into subdomains and store in the subdomain states """
        for domain, source in zip(self.domains.flat, partition(source, self.n_domains).flat):
            domain.set_source(source)

    ## other functions (may become part of utilities?)
