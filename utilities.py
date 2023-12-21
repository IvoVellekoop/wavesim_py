import numpy as np
from scipy.sparse import dok_matrix


def preprocess(n=np.ones((1, 1, 1)),  # Refractive index distribution
               source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
               wavelength=1.,  # Wavelength in um (micron)
               ppw=4,  # points per wavelength
               boundary_widths=(20, 20, 20),  # Width of absorbing boundaries
               n_domains=(1, 1, 1),  # Number of subdomains to decompose into, in each dimension
               omega=10):  # compute the fft over omega times the domain size
    """ Set up parameters to pass to HelmholtzBase """

    n = check_input_dims(n)
    n_dims = (np.squeeze(n)).ndim  # Number of dimensions in problem
    n_roi = np.array(n.shape)  # Num of points in ROI (Region of Interest)

    boundary_widths = check_input_len(boundary_widths, 0, n_dims)
    boundary_pre = np.floor(boundary_widths)
    boundary_post = np.ceil(boundary_widths)

    n_ext = n_roi + boundary_pre + boundary_post  # n_roi + boundaries on either side(s)

    max_subdomain_size = 500  # max permissible size of one sub-domain
    # Number of subdomains to decompose into in each dimension
    if n_domains is None:
        n_domains = n_ext // max_subdomain_size
    else:
        n_domains = check_input_len(n_domains, 1, n_dims)

    # determines number of subdomains based on max size, ensures that all are of the same size (pads if necessary),
    # modifies boundary_post and n_ext, and casts parameters to int

    if (n_domains == 1).all():  # If 1 domain, implies no domain decomposition
        domain_size = n_ext.copy()
    else:  # Else, domain decomposition
        domain_size = n_ext/n_domains

        # Increase boundary_post in dimension(s) until all subdomains are of the same size
        while (domain_size[:n_dims] != np.max(domain_size[:n_dims])).any():
            boundary_post[:n_dims] += (n_domains[:n_dims] * (np.max(domain_size[:n_dims]) - domain_size[:n_dims]))
            n_ext = n_roi + boundary_pre + boundary_post
            domain_size[:n_dims] = n_ext/n_domains

        # Increase number of subdomains until subdomain size is less than max_subdomain_size
        while (domain_size > max_subdomain_size).any():
            n_domains[np.where(domain_size > max_subdomain_size)] += 1
            domain_size = n_ext/n_domains

        # Increase boundary_post in dimension(s) until the subdomain size is int
        while (domain_size % 1 != 0).any() or (boundary_post % 1 != 0).any():
            boundary_post += np.round(n_domains * (np.ceil(domain_size) - domain_size), 2)
            n_ext = n_roi + boundary_pre + boundary_post
            domain_size = n_ext/n_domains

    boundary_pre = boundary_pre.astype(int)
    boundary_post = boundary_post.astype(int)
    n_ext = n_ext.astype(int)
    n_domains = n_domains.astype(int)
    domain_size = domain_size.astype(int)

    k0 = (1. * 2. * np.pi) / wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
    v_raw = k0 ** 2 * n ** 2

    # pad v_raw with boundaries using edge values
    v_raw = pad_boundaries(v_raw, boundary_pre, boundary_post, mode="edge")

    """ give tiny non-zero minimum value to prevent division by zero in homogeneous media """
    pixel_size = wavelength / ppw  # Grid pixel size in um (micron)
    mu_min = ((10.0 / (boundary_widths[:n_dims] * pixel_size)) if (
            boundary_widths != 0).any() else check_input_len(0, 0, n_dims)).astype(np.float32)
    mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(v_raw.shape[:n_dims]) * pixel_size)))
    v_min = np.imag((k0 + 1j * np.max(mu_min)) ** 2)

    # Pad the source term (scale later)
    s = check_input_dims(source)
    s = pad_boundaries(s, boundary_pre, boundary_post, mode="constant")

    omega = check_input_len(omega, 1, n_dims)  # compute the fft over omega times the domain size
    
    return (n_roi, n_ext, s, n_dims, boundary_widths, boundary_pre, boundary_post,
            n_domains, domain_size, omega, v_min, v_raw)
    # return locals()


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function
    :param x: Size of the ARL
    :return boundary_: Boundary"""
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def check_input_dims(x):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - x.ndim):
        x = np.expand_dims(x, axis=-1)
    return x


def check_input_len(x, e, n_dims):
    """ Convert 'x' to a 3-element numpy array, appropriately, i.e., either repeat, or add 'e'. """
    if isinstance(x, int) or isinstance(x, float):
        x = n_dims*tuple((x,)) + (3-n_dims) * (e,)
    elif len(x) == 1:
        x = n_dims*tuple(x) + (3-n_dims) * (e,)
    elif isinstance(x, list) or isinstance(x, tuple):
        x += (3 - len(x)) * (e,)
    if isinstance(x, np.ndarray):
        x = np.concatenate((x, np.zeros(3 - len(x))))
    return np.array(x)


def dft_matrix(n):
    """ Create a discrete Fourier transform matrix of size n x n. Faster than scipy dft function """
    r = np.arange(n)
    omega = np.exp((-2 * np.pi * 1j) / n)  # remove the '-' for inverse fourier
    return np.vander(omega ** r, increasing=True).astype(np.complex64)  # faster than meshgrid


def full_matrix(operator, d):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) """
    nf = np.prod(d)
    m = dok_matrix((nf, nf), dtype=np.complex64)
    b = np.zeros(d, dtype=np.complex64)
    b.flat[0] = 1
    for i in range(nf):
        m[:, i] = operator(np.roll(b, i)).ravel()
    return m


# def full_matrix(operator, d):
#     """ Converts operator to an 2D square matrix of size d.
#     (operator should be a function taking a single column vector as input?) """
#     shape = list(d)
#     nf = np.prod(d)
#     m = dok_matrix((nf, nf), dtype=np.complex64)
#     # b = csr_matrix(([1], ([0],[0])), shape=(nf, 1), dtype=np.complex64)
#     b = np.zeros((nf, 1), dtype=np.complex64)
#     b[0] = 1
#     for i in range(nf):
#         m[:, i] = np.reshape(operator(np.reshape(b, shape)), (-1,))
#         # b.indices = (b.indices+1)%b.shape[0]
#         b = np.roll(b, (1, 0), axis=(0, 1))
#     return m


def pad_boundaries(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) """
    pad_width = tuple([[boundary_pre[i], boundary_post[i]] for i in range(3)])
    return np.pad(x, pad_width, mode)


def pad_func(m, boundary_pre, boundary_post, n_roi, n_dims):
    """ Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
    for i in range(n_dims):
        left_boundary = boundary_(boundary_pre[i])
        right_boundary = np.flip(boundary_(boundary_post[i]))
        full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
        m = np.moveaxis(m, i, -1) * full_filter
        m = np.moveaxis(m, -1, i)
    return m.astype(np.complex64)


def relative_error(e, e_true):
    """ Relative error ⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩ """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)
