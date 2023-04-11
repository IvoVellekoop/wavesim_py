import numpy as np
from anysim_matlab import checkV, coordinates_f, DFT_matrix, relative_error

## AnySim with wrap-around correction (wavesim like, without smallest circle shifting for computing V0)
def AnySim_wrap(n, N, pixel_size, k0, b, iters=int(1.e+4), cp=5):
    N_dim = n.ndim

    ## Correction term
    F = DFT_matrix(N)
    Finv = np.asarray((np.matrix(F).H)/N)
    L = (coordinates_f(1, N, pixel_size)**2).T
    for d in range(2,N_dim+1):
        L = L + coordinates_f(d, N, pixel_size)**2
    Lw = Finv @ np.diag(L.flatten()) @ F

    # ## Option 1. Using (Lo-Lw) as the wrap-around correction
    Ones = np.eye(N)
    Omega = np.zeros((N, N*10))
    Omega[:,:N] = Ones
    p_O = 2*np.pi*np.fft.fftfreq(N*10, pixel_size)
    L_Omega = np.abs(p_O)**2
    F_Omega = DFT_matrix(N*10)
    Finv_Omega = np.asarray((np.matrix(F_Omega).H)/(N*10))
    Lo = Omega @ Finv_Omega @ np.diag(L_Omega.flatten()) @ F_Omega @ Omega.T
    L_corr1 = (Lo-Lw)
    L_corr = np.real(L_corr1) + 0j*np.imag(L_corr1)
    # ## Option 2. Replacing (Lo-Lw) with the upper and lower triangular corners of -Lw
    # L_corr = -Lw.copy()
    # L_corr[:-cp,:-cp] = 0
    # L_corr[cp:,cp:] = 0
    # print(relative_error(L_corr, L_corr1))

    ## make medium
    Vraw = -1j * ( np.diag(k0**2 * n**2) + L_corr )
    # points = np.concatenate((np.diag(Vraw), Vraw[:cp,-cp:].flatten(), Vraw[-cp:,:cp].flatten()))
    abs_im_Vraw = np.abs(np.imag(Vraw))
    V0 = ( np.max(abs_im_Vraw) + np.min(abs_im_Vraw) )/2
    V = Vraw - np.eye(N)*V0
    Vmax = 0.95
    # print(checkV(V), np.linalg.norm(V,2))
    scaling = Vmax/np.linalg.norm(V,2) #checkV(V) #
    V *= scaling
    ## Check that ||V|| < 1 (0.95 here)
    vc = np.linalg.norm(V,2) #checkV(V) #
    if vc < 1:
        pass
    else:
        return print('||V|| not < 1, but {}'.format(vc))

    ## B = 1 - V
    B = np.eye(N) - V
    medium = lambda x: B @ x

    ## make propagator
    Tr = np.sqrt(scaling)
    Tl = 1j * Tr.copy()
    L = Tl * Tr * (L - 1j*V0)
    Lr = np.squeeze(1/(L+1))
    propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

    ## Check that A = L + V is accretive
    L = np.diag(np.squeeze(L))
    A = L + V
    for _ in range(10): ## Repeat test for multiple random vectors
        z = np.random.rand(b.size,1) + 1j*np.random.randn(b.size,1)
        acc = np.real(np.matrix(z).H @ A @ z)
        if np.round(acc, 13) < 0:
            return print('A is not accretive. ', acc)

    ## pad the source term with zeros so that it is the same size as B.shape[0]
    b = Tl * b.copy() # source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    return Tr*u, L, V

## AnySim with L_Omega instead of wrapped around Laplacian (wavesim like, without smallest circle shifting for computing V0)
def AnySim_omega(n, N, pixel_size, k0, b, iters=int(1.e+4)):
    N_dim = n.ndim

    ## make medium
    Vraw = np.diag(-1j * k0**2 * n**2)
    abs_im_Vraw = np.abs(np.imag(Vraw))
    V0 = ( np.max(abs_im_Vraw) + np.min(abs_im_Vraw) )/2
    V = Vraw - np.eye(N)*V0
    Vmax = 0.95
    # print(checkV(V), np.linalg.norm(V,2))
    scaling = Vmax/np.linalg.norm(V,2) #checkV(V) #
    V *= scaling

    ## Check that ||V|| < 1 (0.95 here)
    vc = np.linalg.norm(V,2) #checkV(V) #
    if vc < 1:
        pass
    else:
        return print('||V|| not < 1, but {}'.format(vc))

    ## B = 1 - V
    B = np.eye(N) - V
    medium = lambda x: B @ x

    ## make propagator
    Tr = np.sqrt(scaling)
    Tl = 1j * Tr.copy()

    L_Omega = (coordinates_f(1, N*10, pixel_size)**2).T
    for d in range(2,N_dim+1):
        L_Omega = L_Omega + coordinates_f(d, N*10, pixel_size)**2
    L_Omega = Tl * Tr * (L_Omega - 1j*V0)
    Lr_Omega = np.squeeze(1/(L_Omega+1))
    Ones = np.eye(N)
    Omega = np.zeros((N, N*10))
    Omega[:,:N] = Ones
    F_Omega = DFT_matrix(N*10)
    Finv_Omega = np.asarray((np.matrix(F_Omega).H)/(N*10))
    prop_operator = Omega @ Finv_Omega @ np.diag(Lr_Omega.flatten()) @ F_Omega @ Omega.T
    propagator = lambda x: prop_operator @ x

    ## pad the source term with zeros so that it is the same size as B.shape[0]
    b = Tl * b.copy() # source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    return Tr*u
