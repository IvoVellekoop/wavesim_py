import numpy as np
from anysim_matlab import checkV, coordinates_f, pad_func

## AnySim (wavesim like, without smallest circle shifting for computing V0)
def AnySim_wvsim_like(boundaries_width, n, N_roi, pixel_size, k0, s, iters=int(1.e+4)):
    N_dim = n.ndim
    N = int(N_roi+2*boundaries_width)
    bw_l = int(np.floor(boundaries_width))
    bw_r = int(np.ceil(boundaries_width))

    ## make medium
    V0 = k0**2 * (np.max(np.real(n**2)) + np.min(np.real(n**2)))/2
    V = -1j * k0**2 * n**2 - V0
    ## uncommment if you want diag V in 1D
    # V = np.diag(V)
    
    Vmax = 0.95
    scaling = Vmax/checkV(V)
    V = scaling * V
    ## Check that ||V|| < 1 (0.95 here)
    vc = checkV(V)
    if vc < 1:
        pass
    else:
        return print('||V|| not < 1, but {}'.format(vc))

    ## B = 1 - V, and pad
    B = pad_func((1-V), boundaries_width, bw_l, bw_r, N_roi, N_dim, 0).astype('complex_')  # obj.medium(), i.e., medium (1-V)
    medium = lambda x: B * x
    ## uncommment if you want diag V in 1D
    # B = pad_func((1-np.diag(V)), boundaries_width, bw_l, bw_r, N_roi, N_dim, 0).astype('complex_')  # obj.medium(), i.e., medium (1-V)
    # B = np.diag(B)
    # medium = lambda x: B @ x

    ## make propagator
    Tr = np.sqrt(scaling)
    Tl = 1j * Tr.copy()
    L = (coordinates_f(1, N, pixel_size)**2).T
    for d in range(2,N_dim+1):
        L = L + coordinates_f(d, N, pixel_size)**2
    L = Tl * Tr * (L - 1j*V0)
    Lr = np.squeeze(1/(L+1))
    propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

    ## Check that A = L + V is accretive
    if N_dim == 1:
        L = np.diag(np.squeeze(L)[bw_l:-bw_r])
        V = np.diag(V)
    # else:
    #     L = L[bw_l:-bw_r, bw_l:-bw_r]
        A = L + V
        acc = np.min(np.linalg.eigvals(A + np.asarray(np.matrix(A).H)))
        if np.round(acc, 13) < 0:
            return print('A is not accretive. ', acc)

    ## pad the source term with zeros so that it is the same size as B.shape[0]
    b = Tl * np.pad(s, N_dim*((bw_l,bw_r),), mode='constant') # source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    u = Tr * u

    if N_dim == 1:
        u = u[bw_l:-bw_r]
    elif N_dim == 2:
        u = u[bw_l:-bw_r,bw_l:-bw_r]
    return u
