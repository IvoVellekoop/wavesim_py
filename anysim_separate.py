import os
import numpy as np
from scipy.linalg import eigvals
from datetime import date
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (8,8) #(14.32,8)
try:
	plt.rcParams['text.usetex'] = False
except:
	pass

## AnySim with wrap-around correction (wavesim like, without smallest circle shifting for computing V0)
def AnySim_wrap(test, small_circ_prob, wrap_around, n, N, pixel_size, k0, b, u_true, max_iters=int(6.e+5), cp=20):
    iters = max_iters - 1
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
    V = np.diag(-1j * (Vraw - V0)) #+ 1j * L_corr # sign of -1j * (Vraw-V0)??
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
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    residual = []
    u_iter = []
    for i in range(max_iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            iters = i
            print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(iters+1, residual_i))
            break
        u = u - (alpha * t1)
        u_iter.append(u)
    # Scale back
    u = Tr * u
    u_iter = Tr * np.array(u_iter)

    residual = np.array(residual)

    ## For plotting 
    post_process(test, small_circ_prob, wrap_around, N, pixel_size, u, iters, residual, u_iter, 0, u_true)

    return u, residual, u_iter

## AnySim with L_Omega instead of wrapped around Laplacian (wavesim like, without smallest circle shifting for computing V0)
def AnySim_omega(test, small_circ_prob, wrap_around, n, N, pixel_size, k0, b, u_true, max_iters=int(6.e+5)):
    iters = max_iters - 1
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
    u_iter = []
    for i in range(max_iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            iters = i
            print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(iters+1, residual_i))
            break
        u = u - (alpha * t1)
        u_iter.append(u)
    # Scale back
    u = Tr * u
    u_iter = Tr * np.array(u_iter)

    residual = np.array(residual)

    ## For plotting 
    post_process(test, small_circ_prob, wrap_around, N, pixel_size, u, iters, residual, u_iter, 0, u_true)

    return u, residual, u_iter

## AnySim (wavesim like, without smallest circle shifting for computing V0)
def AnySim(test, small_circ_prob, wrap_around, boundaries_width, n, N_roi, pixel_size, k0, s, u_true, max_iters=int(6.e+5)):
    N_dim = n.ndim
    N = int(N_roi+2*boundaries_width)
    bw_l = int(np.floor(boundaries_width))
    bw_r = int(np.ceil(boundaries_width))
    iters = max_iters - 1

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
    u_iter = []
    for i in range(max_iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        if i==0:
            normb = np.linalg.norm(t1)
        nr = np.linalg.norm(t1)
        residual_i = nr/normb
        residual.append(residual_i)
        if residual_i < 1.e-6:
            iters = i
            print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(iters+1, residual_i))
            break
        u = u - (alpha * t1)
        u_iter.append(u)

    # Scale back
    u = Tr * u
    u_iter = Tr * np.array(u_iter)

    # Truncate u to ROI
    if N_dim == 1:
        u = u[bw_l:-bw_r]
        u_iter = u_iter[:, bw_l:-bw_r]
    elif N_dim == 2:
        u = u[bw_l:-bw_r,bw_l:-bw_r]
        u_iter = u_iter[:, bw_l:-bw_r,bw_l:-bw_r]

    residual = np.array(residual)

    ## For plotting 
    post_process(test, small_circ_prob, wrap_around, N_roi, pixel_size, u, iters, residual, u_iter, boundaries_width, u_true)

    return u, residual, u_iter


### Other helper functions

## pad (1-V) to size N = N_roi + (2*boundaries_width)
def pad_func(M, boundaries_width, bw_l, bw_r, N_roi, N_dim, element_dimension=0):
    sz = M.shape[element_dimension+0:N_dim]
    if (boundaries_width != 0) & (sz == (1,)):
        M = np.tile(M, (int((np.ones(1) * (N_roi-1) + 1)[0]),))
    M = np.pad(M, ((bw_l,bw_r)), mode='edge')

    for d in range(N_dim):
        try:
            w = boundaries_width[d]
        except:
            w = boundaries_width

    if w>0:
        left_boundary = boundaries_window(np.floor(w))
        right_boundary = boundaries_window(np.ceil(w))
        full_filter = np.concatenate((left_boundary, np.ones((N_roi,1)), np.flip(right_boundary)))
        if N_dim == 1:
            M = M * np.squeeze(full_filter)
        else:
            M = full_filter.T * M * full_filter
    return M

def boundaries_window(L):
    x = np.expand_dims(np.arange(L)/(L-1), axis=1)
    a2 = np.expand_dims(np.array([-0.4891775, 0.1365995/2, -0.0106411/3]) / (0.3635819 * 2 * np.pi), axis=1)
    return np.sin(x * np.expand_dims(np.array([1, 2, 3]), axis=0) * 2 * np.pi) @ a2 + x

## DFT matrix
def DFT_matrix(N):
    l, m = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    return np.power( omega, l * m )

def coordinates_f(dimension, N, pixel_size):
    pixel_size_f = 2 * np.pi/(pixel_size*N)
    return np.expand_dims( fft_range(N) * pixel_size_f, axis=tuple(range(2-dimension)))

def fft_range(N):
    return np.fft.ifftshift(symrange(N))

def symrange(N):
    return range(-int(np.floor(N/2)),int(np.ceil(N/2)))

## Relative error
def relative_error(E_, E_true):
    return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )

## Check that V is a contraction
def checkV(A):
    return np.max(np.abs(A))

def post_process(test, small_circ_prob, wrap_around, N_roi, pixel_size, u, iters, residual, u_iter, boundaries_width, u_true):

    ''' Create log folder / Check for existing log folder'''
    today = date.today()
    d1 = today.strftime("%Y%m%d")
    log_dir = 'logs_separate/Logs_'+d1+'/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_id = d1 + '_' + test + '_scp' + str(int(small_circ_prob)) + '_wrap_' + wrap_around
    run_loc = log_dir + run_id
    if not os.path.exists(run_loc):
        os.makedirs(run_loc)
    
    plot_FieldNResidual(test, small_circ_prob, wrap_around, run_id, run_loc, N_roi, pixel_size, u, iters, residual, boundaries_width, u_true)
    plot_field_iters(test, small_circ_prob, wrap_around, run_id, run_loc, N_roi, pixel_size, u, iters, u_iter, boundaries_width, u_true)

# Plotting functions
def plot_FieldNResidual(test, small_circ_prob, wrap_around, run_id, run_loc, N_roi, pixel_size, u, iters, residual, boundaries_width=0, u_true=np.array([0])): # png
    plt.subplots(figsize=figsize, ncols=1, nrows=2)

    plt.subplot(2,1,1)
    x = np.arange(N_roi)*pixel_size
    if test == 'FreeSpace':
        tlabel = 'Analytic solution'
    elif test == '1D':
        u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])
        tlabel = 'Matlab solution'
    plt.plot(x, np.real(u_true), 'k', lw=2., label=tlabel)
    plt.plot(x, np.real(u), 'r', lw=1., label='RelErr = {:.2e}'.format(relative_error(u,u_true)))
    plt.title('Field')
    plt.ylabel('Amplitude')
    plt.xlabel('$x~[\lambda]$')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.loglog(np.arange(1,iters+2), residual, 'k', lw=1.5)
    plt.axhline(y=1.e-6, c='k', ls=':')
    plt.yticks([1.e+0, 1.e-2, 1.e-4, 1.e-6])
    plt.title('Residual. Iterations = {:.2e}'.format(iters+1))
    plt.ylabel('Residual')
    plt.xlabel('Iterations')
    plt.grid()

    if wrap_around == 'boundaries':
        plt.suptitle(f'Tackling wrap-around effects with: {wrap_around} (width = {int(boundaries_width)})')
    else:
        plt.suptitle(f'Tackling wrap-around effects with: {wrap_around}')
    plt.tight_layout()
    plt.savefig(f'{run_loc}/{run_id}_{iters+1}iters_FieldNResidual.png', bbox_inches='tight', pad_inches=0.03, dpi=100)
    # plt.draw()
    plt.close()

def plot_field_iters(test, small_circ_prob, wrap_around, run_id, run_loc, N_roi, pixel_size, u, iters, u_iter, boundaries_width=0, u_true=np.array([0])): # movie/animation/GIF
    u_iter = np.real(u_iter)
    x = np.arange(N_roi)*pixel_size

    fig = plt.figure(figsize=(14.32,8))
    if test == 'FreeSpace':
        tlabel = 'Analytic solution'
    elif test == '1D':
        u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])
        tlabel = 'Matlab solution'
    plt.plot(x, np.real(u_true), 'k:', lw=0.75, label=tlabel)
    plt.xlabel("$x$")
    plt.ylabel("Amplitude")
    plt.xlim([x[0]-x[1]*2,x[-1]+x[1]*2])
    plt.ylim([np.min(u_iter), np.max(u_iter)])
    plt.grid()
    line, = plt.plot([] , [], 'b', lw=1., animated=True)
    line.set_xdata(x)
    title = plt.title('')

    # Plot 100 or fewer frames. Takes much longer for any more frames.
    max_frames = 100
    if iters > max_frames:
        plot_iters = max_frames
        iters_trunc = np.linspace(0,iters-1,plot_iters).astype(int)
        u_iter_trunc = u_iter[iters_trunc]
    else:
        plot_iters = iters
        iters_trunc = np.arange(iters)
        u_iter_trunc = u_iter

    def animate(i):
        line.set_ydata(u_iter_trunc[i])		# update the data.
        if wrap_around == 'boundaries':
            title.set_text(f'Tackling wrap-around effects with: {wrap_around} (width = {int(boundaries_width)}). Iteration {iters_trunc[i]+1}')
        else:
            title.set_text(f'Tackling wrap-around effects with: {wrap_around}. Iteration {iters_trunc[i]+1}')
        return line, title,
    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=True, frames=plot_iters)
    writer = animation.FFMpegWriter(
        fps=10, metadata=dict(artist='Me'))
    #ani.save(f'{run_loc}/{run_id}_{iters+1}iters_Field.mp4', writer=writer)
    plt.close()

