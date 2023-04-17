import numpy as np
from anysim_matlab import checkV, DFT_matrix#, relative_error

## AnySim with wrap-around correction (wavesim like, without smallest circle shifting for computing V0)
def AnySim_wrap(n, N, pixel_size, k0, b, iters=int(1.e+4), cp=20):
    # ## Correction term
    F = DFT_matrix(N)
    Finv = np.asarray(np.matrix(F).H/N)
    p = 2*np.pi*np.fft.fftfreq(N, pixel_size)
    L = p**2
    Lw = Finv @ np.diag(L.flatten()) @ F

    ''' Option 1. Using (Lo-Lw) as the wrap-around correction '''
    # Ones = np.eye(N)
    # Omega = np.zeros((N, N*10))
    # Omega[:,:N] = Ones
    # p_O = 2*np.pi*np.fft.fftfreq(N*10, pixel_size)
    # L_Omega = p_O**2
    # F_Omega = DFT_matrix(N*10)
    # Finv_Omega = np.asarray(np.matrix(F_Omega).H/(N*10))
    # Lo = Omega @ Finv_Omega @ np.diag(L_Omega.flatten()) @ F_Omega @ Omega.T
    # L_corr = (Lo-Lw)
    # L_corr = np.real(L_corr1)
    ''' Option 2. 
        Replacing (Lo-Lw) with the upper and lower triangular corners of -np.real(Lw) 
        as the remaining portion and the imaginary part of (Lo-Lw) are of the order of 10^(-11)'''
    L_corr = -np.real(Lw).copy()
    L_corr[:-cp,:-cp] = 0; L_corr[cp:,cp:] = 0
    # print('relative_error(L_corr, L_corr1)', relative_error(L_corr, L_corr1))

    ## make medium
    V0 = k0**2 * (np.max(np.real(n**2)) + np.min(np.real(n**2)))/2
    V = np.diag(-1j * k0**2 * n**2 - V0) + 1j * L_corr
    Vmax = 0.95
    scaling = Vmax/np.linalg.norm(V,2) #checkV(V) #
    V = scaling * V

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
    Tl = 1j * Tr
    L = Tl * Tr * (L - 1j*V0)
    Lr = np.squeeze(1/(L+1))
    propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

    ## Check that A = L + V is accretive
    L = np.diag(np.squeeze(L))
    A = L + V
    acc = np.min(np.linalg.eigvals(A + np.asarray(np.matrix(A).H)))
    if np.round(acc, 13) < 0:
        return print('A is not accretive. ', acc)

    b = Tl * b # scale the source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 1.                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    return Tr*u

## AnySim with L_Omega instead of wrapped around Laplacian (wavesim like, without smallest circle shifting for computing V0)
def AnySim_omega(n, N, pixel_size, k0, b, iters=int(1.e+4)):
    ## make medium
    V0 = k0**2 * (np.max(np.real(n**2)) + np.min(np.real(n**2)))/2
    V = -1j * k0**2 * n**2 - V0
    Vmax = 0.95
    scaling = Vmax/checkV(V)    #checkV(V) gives the largest singular value of V
    V = scaling * V

    ## Check that ||V|| < 1 (0.95 here)
    vc = checkV(V)
    if vc < 1:
        pass
    else:
        return print('||V|| not < 1, but {}'.format(vc))

    B = 1 - V
    medium = lambda x: B * x

    ## make propagator
    Tr = np.sqrt(scaling)
    Tl = 1j * Tr
    p_O = 2*np.pi*np.fft.fftfreq(N*10, pixel_size)
    L_Omega = p_O**2
    L_Omega = Tl * Tr * (L_Omega - 1j*V0)
    Lr_Omega = np.squeeze(1/(L_Omega+1))
    propagator = lambda x: (np.fft.ifftn(Lr_Omega * np.fft.fftn( np.pad(x,(0,N*(10-1))) )))[:N]

    b = Tl * b # scale the source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    return Tr*u
