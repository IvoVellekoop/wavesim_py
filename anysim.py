import numpy as np
from anysim_small_circ_prob import checkV, coordinates_f, DFT_matrix, pad_func, relative_error

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
    # # Lw_inv = np.linalg.inv(Lw)
    # # L_corr = Lw_inv @ (Lo-Lw) @ Lw_inv
    # # L_corr1 = np.real(L_corr1)
    ''' Option 2. 
        Replacing (Lo-Lw) with the upper and lower triangular corners of -np.real(Lw) 
        as the remaining portion and the imaginary part of (Lo-Lw) are of the order of 10^(-11) '''
    L_corr = -np.real(Lw)                          # copy -Lw
    L_corr[:-cp,:-cp] = 0; L_corr[cp:,cp:] = 0  # Keep only upper and lower traingular corners of -Lw
    # # print('relative_error(L_corr1, L_corr)', relative_error(L_corr1, L_corr))

    ## make medium
    Vraw = k0**2 * n**2
    mu_min = 1.e+0/(N*pixel_size)   # give tiny non-zero minimum value to prevent division by zero in homogeneous media
    Vmin = np.imag( (k0 + 1j*np.max(mu_min))**2 )
    V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 + 1j*Vmin
    V = np.diag(-1j * (Vraw - V0)) + 1j * L_corr
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
    L = Tl * Tr * (L - V0)
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
    residual = []
    for i in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)       # residual
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            print('breaking @ iter {}, residual {:.2e}'.format(i+1, residual_i))
            break
        u = u - (alpha * t1)
    return Tr*u, np.array(residual)

## AnySim with L_Omega instead of wrapped around Laplacian (wavesim like, without smallest circle shifting for computing V0)
def AnySim_omega(n, N, pixel_size, k0, b, iters=int(1.e+4)):
    ## make medium
    Vraw = k0**2 * n**2
    mu_min = 1.e+0/(N*pixel_size)   # give tiny non-zero minimum value to prevent division by zero in homogeneous media
    Vmin = np.imag( (k0 + 1j*np.max(mu_min))**2 )
    V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 + 1j*Vmin
    V = -1j * (Vraw - V0)
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
    L_Omega = Tl * Tr * (L_Omega - V0)
    Lr_Omega = np.squeeze(1/(L_Omega+1))
    propagator = lambda x: (np.fft.ifftn(Lr_Omega * np.fft.fftn( np.pad(x,(0,N*(10-1))) )))[:N]

    b = Tl * b # scale the source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    residual = []
    for i in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            print('breaking @ iter {}, residual {:.2e}'.format(i+1, residual_i))
            break
        u = u - (alpha * t1)
    return Tr*u, np.array(residual)

## AnySim (wavesim like, without smallest circle shifting for computing V0)
def AnySim(boundaries_width, n, N_roi, pixel_size, k0, s, iters=int(1.e+4)):
    N_dim = n.ndim
    N = int(N_roi+2*boundaries_width)
    bw_l = int(np.floor(boundaries_width))
    bw_r = int(np.ceil(boundaries_width))

    ## make medium
    Vraw = k0**2 * n**2
    if boundaries_width == 0:
        mu_min = 0
    else:
        mu_min = 10/(boundaries_width*pixel_size)
    mu_min = np.maximum( mu_min, 1.e-3/(N*pixel_size) )
    Vmin = np.imag( (k0 + 1j*np.max(mu_min))**2 )
    V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 + 1j*Vmin
    V = -1j * (Vraw - V0)
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
    L = Tl * Tr * (L - V0)
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
    residual = []
    for i in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            print('breaking @ iter {}, residual {:.2e}'.format(i+1, residual_i))
            break
        u = u - (alpha * t1)
    u = Tr * u

    if N_dim == 1:
        u = u[bw_l:-bw_r]
    elif N_dim == 2:
        u = u[bw_l:-bw_r,bw_l:-bw_r]
    return u, np.array(residual)
