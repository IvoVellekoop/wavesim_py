# Date: 2023-03-03 11:11
# Author: Swapnil
# Pytorch implementation of 20230301_wrap_solution.ipynb, i.e., 
# the analytical correction for wrap-around effects

#%% #@title import packages
import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np
from scipy.sparse import identity
import time

import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (14.32,8)
plt.rcParams['text.usetex'] = True

import plotly.graph_objects as go

#%% #@title simulation parameters and function definitions

lambd = 1.                  # wavelength in um (micron)
k0 = (1.*2.*np.pi)/(lambd)  # wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
epsilon = 1.e+0             # higher epsilon means more absorption and thus faster decay. Usually of the order of k0**2. Wavesim uses epsilon = 3 and anysim does not use it anymore; instead scaling factor c used

## Relative error
def relative_error(E, Ea):
    return np.mean( np.linalg.norm(E-Ea, ord=2) ) / np.mean( np.linalg.norm(Ea, ord=2) )

## DFT matrix
def DFT_matrix(N):
    l, m = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    F = np.power( omega, l * m )
    return np.matrix(F)

def DFT_matrix_torch(N):
    l, m = torch.meshgrid(torch.arange(N).to(device), torch.arange(N).to(device), indexing='ij')
    omega = torch.exp( torch.tensor(-2*torch.pi*1j/N) ).to(device)
    F = torch.pow(omega, l*m).to(device)
    return F

xh = 10.; xl = -xh
dx = 0.05#lambd/4#
x = np.arange(xl,xh+dx,dx)

# define a point source
s_loc = 0                      # source location (centre)
s_val = 1.

s1 = np.zeros_like(x, dtype='complex_')
if np.where(np.round(x,3)==s_loc)[0].size==1:
    s1[np.where(np.round(x,3)==s_loc)] = s_val
else:
    where_x_1 = np.where(np.round(x)==s_loc)
    select = int((where_x_1[0].size-1)/2)
    s1[where_x_1[0][select]] = s_val
    print(s1)
s1 = np.matrix(s1).T

## To compare numpy vs. pytorch (dense) vs. pytorch (sparse) implementations, uncomment the respective blocks below
#%% #@title Numpy. Main implementation block
print('-'*50)
print('Numpy')

start_np = time.time()
kn = 10
x_size = x.size
xW1 = kn*x_size
pW = 2*np.pi*np.fft.fftfreq(xW1, dx)
g_pW = 1/(np.abs(pW)**2 - k0**2 - 1.j*epsilon)
FG = np.diag(g_pW/dx)    #; print('FG', FG.shape, FG.device, FG.dtype)
# g_L_W = np.fft.ifftshift(np.fft.ifft(g_pW)/dx)
# FG = np.diag(np.fft.fft(g_L_W))#; print('FG', FG.shape)

Ones = np.eye(x_size, dtype='complex_')
O = np.zeros((x_size, xW1), dtype='complex_')#; print('O', O.shape)
O[:,:x_size] = Ones
F = DFT_matrix(xW1)#; print('F', F.shape)
Finv = (F.H)/xW1#; print('Finv', Finv.shape)

# noshift_FGFT = Finv @ FG @ F @ O.T
# FGFT = np.fft.ifftshift(noshift_FGFT, axes=0)#; print('FGFT', FGFT.shape)
FGFT = Finv @ FG @ F @ O.T
GO = O @ FGFT#; print('GO', GO.shape)

# ### EW over padded domain W
W = np.tile(Ones, kn)#; print('W', W.shape)
# GW = W @ FGFT#; print('GW', GW.shape)

### EW over simulation domain Omega. Not much of a difference (for 10 or 20 times the domain)
p_O = 2*np.pi*np.fft.fftfreq(x.size, dx)
g_p_O = 1/(np.abs(p_O)**2 - k0**2 - 1.j*epsilon)
FG_O = np.diag(g_p_O/dx)    #; print('FG_O', FG_O.shape)
F_O = DFT_matrix(x_size)    #; print('F_O.shape', F_O.shape)
Finv_O = (F_O.H)/x_size     #; print('Finv_O.shape', Finv_O.shape)
GW = Finv_O @ FG_O @ F_O    #; print('GW.shape', GW.shape)

EO1 = np.squeeze(np.asarray(GO.T @ s1))#; print('EO1', EO1.shape)
EW1 = np.squeeze(np.asarray(GW.T @ s1))#; print('EW1', EW1.shape)

GW_inv = np.linalg.inv(GW)
B = GW_inv @ (GW - GO) @ GW_inv
b = x_size
# b = 15#int(x_size/2+1)
# B[b:-b,:] = 0.; B[:,b:-b] = 0.
E_wrap = np.squeeze(np.asarray(GW @ B @ EW1))

## Plot Corrected field and E_L
E_corrected = EW1 - E_wrap
error1 = relative_error(E_corrected, EO1)
end_np = time.time() - start_np
print(b,'/',x_size, '(boundary points out of total points)')
print('Time,\t\t', np.round(end_np, 2))
print("Relative error,\t {:.2e}".format(error1))
print('-'*50)
# time.sleep(5.0)

#%% #@title 08-03-2023 New trials to bring down computations to Omega domain (simulation domain) instead of padded W domain
# E_wrap_new = np.squeeze(np.asarray((W-O) @ FGFT @ s1))
# W_rinv = (W/kn).T
W_rinv = (W.T@np.linalg.inv(W@W.T))
# E_wrap_new = np.squeeze(np.asarray((W-O) @ W_rinv @ W @ FGFT @ s1))
B_new = GW_inv @ (W-O) @ W_rinv
E_wrap_new = np.squeeze(np.asarray(GW @ B_new @ EW1))
print(relative_error(E_wrap_new, E_wrap))

plt.subplots(nrows=1, ncols=1, figsize=(figsize[0],figsize[1]))
# plt.subplot(1,2,1)
# plt.plot(np.real(EO1), 'r', label='E_L')
# plt.plot(np.real(EO_cpu), 'b:', label=r'$E_{\Omega}$')
# plt.title('Linear')
# plt.legend()

# plt.subplot(1,2,2)
plt.subplot(1,1,1)
plt.plot(np.real(E_wrap), 'r', label='E_wrap')
plt.plot(np.real(EW1), 'g--', label='EW1')
plt.plot(np.real(E_wrap_new), 'b:', label='E_wrap_new')
plt.title('Periodic')
plt.legend()
plt.show()

print('Done')

# #%% #@title Pytorch. All dense matrices.
# print('Pytorch (dense)')

# s_ = time.time()
# kn = 10
# x_size = x.size
# xW1 = kn*x_size
# pW = 2*torch.pi*torch.fft.fftfreq(xW1, dx)
# g_pW = 1/(torch.abs(pW)**2 - k0**2 - 1.j*epsilon)
# FG = torch.diag(g_pW/dx).to(device)                 #; print('FG', FG.shape, FG.device, FG.dtype)
# # g_L_W = torch.fft.ifftshift(torch.fft.ifft(g_pW)/dx)
# # FG = torch.diag(torch.fft.fft(g_L_W)).to(device)  #; print('FG', FG.shape, FG.device, FG.dtype)

# # F = torch.tensor(DFT_matrix(xW1)).to(device)#; print('F', F.shape, F.device, F.dtype)
# F = DFT_matrix_torch(xW1)                           #; print('F', F.shape, F.device, F.dtype)
# Finv = (F.H)/xW1                                    #; print('Finv', Finv.shape, Finv.device, Finv.dtype)

# Ones = torch.eye(x_size, device=device, dtype=torch.complex128)
# O = torch.zeros((x_size, xW1), device=device, dtype=torch.complex128)
# O[:,:x_size] = Ones                                 #; print('O', O.shape, O.device, O.dtype)
# W = torch.tile(Ones, (kn,))                         #; print('W', W.shape, W.device, W.dtype)

# # noshift_FGFT = torch.mm(Finv, torch.mm(FG, torch.sparse.mm(F, torch.transpose(O,0,1))))#; print('noshift_FGFT', noshift_FGFT.shape, noshift_FGFT.device, noshift_FGFT.dtype)
# # FGFT = torch.fft.ifftshift(noshift_FGFT, dim=0)   #; print('FGFT', FGFT.shape, FGFT.device, FGFT.dtype)
# FGFT = torch.mm(Finv, torch.mm(FG, torch.mm(F, torch.transpose(O,0,1))))#; print('FGFT', FGFT.shape, FGFT.device, FGFT.dtype)
# GO = torch.mm(O, FGFT)                              #; print('GO', GO.shape, GO.device, GO.dtype)
# GW = torch.mm(W, FGFT)                              #; print('GW', GW.shape, GW.device, GW.dtype)

# s = torch.tensor(s1).to(device)                     #; print('s', s.shape, s.device, s.dtype)

# EO = torch.mm(torch.transpose(GO,0,1), s)           #; print('EO', EO.shape, EO.device, EO.dtype)
# EW = torch.mm(torch.transpose(GW,0,1), s)           #; print('EW', EW.shape, EW.device, EW.dtype)

# EO_cpu = np.squeeze(np.asarray(EO.cpu().numpy()))
# EW_cpu = np.squeeze(np.asarray(EW.cpu().numpy()))

# GW_inv = torch.linalg.inv(GW)
# B = torch.mm(GW_inv, torch.mm((GW-GO), GW_inv))
# b = x_size
# # b = 15
# # B[b:-b,:] = 0.; B[:,b:-b] = 0.                      #; print('B', B.shape, B.device, B.dtype)
# E_wrap = np.squeeze(np.asarray((torch.mm(GW,torch.mm(B,EW))).cpu().numpy()))

# ## Plot Corrected field and E_L
# E_corrected = EW_cpu - E_wrap
# error = relative_error(E_corrected, EO_cpu)
# e_ = time.time() - s_
# print(b,'/',x_size, '(boundary points out of total points)')
# print('Time,\t\t', np.round(e_, 2))
# print("Relative error,\t {:.2e}".format(error))
# print('-'*50)
# # time.sleep(5.0)

# #%% #@title Pytorch. Sparse matrices wherever possible.
# print('Pytorch (sparse)')

# start = time.time()
# kn = 10
# x_size = x.size
# xW1 = kn*x_size
# pW = 2*torch.pi*torch.fft.fftfreq(xW1, dx)
# g_pW = 1/(torch.abs(pW)**2 - k0**2 - 1.j*epsilon)
# FG = torch.sparse.spdiags(g_pW/dx, torch.tensor([0]), (xW1, xW1)).to(device)#; print('FG', FG.shape, FG.device, FG.dtype)

# F = DFT_matrix_torch(xW1)                           #; print('F', F.shape, F.device, F.dtype)
# Finv = (F.H)/xW1                                    #; print('Finv', Finv.shape, Finv.device, Finv.dtype)

# Ones = torch.sparse.spdiags(torch.ones(x_size, dtype=torch.complex128), torch.tensor([0]), (x_size, x_size)).to(device)#; print('Ones', Ones.shape, Ones.device, Ones.dtype)
# O = torch.sparse_coo_tensor(torch.tensor((range(x_size),range(x_size))), torch.ones(x_size), [x_size,xW1], dtype=torch.complex128).to(device)#; print('O', O.shape, O.device, O.dtype)
# FGFT = torch.mm(Finv, torch.sparse.mm(FG, torch.sparse.mm(O, F).t()))#; print('FGFT', FGFT.shape, FGFT.device, FGFT.dtype)
# GO = torch.sparse.mm(O, FGFT)                       #; print('GO', GO.shape, GO.device, GO.dtype)

# ### EW over padded domain W
# W = torch.concat([Ones for _ in range(kn)], dim=1).to(device)#; print('W', W.shape, W.device, W.dtype)
# GW = torch.sparse.mm(W, FGFT)                       #; print('GW', GW.shape, GW.device, GW.dtype)

# # ### EW over simulation domain Omega. Not much of a difference (for 10 or 20 times the domain)
# # p_O = 2*torch.pi*torch.fft.fftfreq(x.size, dx)
# # g_p_O = 1/(torch.abs(p_O)**2 - k0**2 - 1.j*epsilon)
# # FG_O = torch.sparse.spdiags(g_p_O/dx, torch.tensor([0]), (x_size, x_size)).to(device)#; print('FG_O', FG_O.shape, FG_O.device, FG_O.dtype)
# # F_O = DFT_matrix_torch(x_size)                      #; print('F_O.shape', F_O.shape)
# # Finv_O = (F_O.H)/x_size                             #; print('Finv_O.shape', Finv_O.shape)
# # GW = torch.mm(Finv_O, torch.sparse.mm(FG_O, F_O))   #; print('GW.shape', GW.shape)

# i = torch.tensor([[np.where(np.round(x,3)==s_loc)[0][0]],[0]])
# v = torch.tensor([1.])
# s = torch.sparse_coo_tensor(i, v, [x_size,1], dtype=torch.complex128).to(device)#; print('s', s.shape, s.device, s.dtype)

# EO = torch.sparse.mm(s.t(), GO.t()).t()             #; print('EO', EO.shape, EO.device, EO.dtype)
# EW = torch.sparse.mm(s.t(), GW.t()).t()             #; print('EW', EW.shape, EW.device, EW.dtype)

# EO_cpu = np.squeeze(np.asarray(EO.cpu().numpy()))
# EW_cpu = np.squeeze(np.asarray(EW.cpu().numpy()))

# GW_inv = torch.linalg.inv(GW)
# B = torch.mm(GW_inv, torch.mm((GW-GO), GW_inv))
# b = x_size
# # b = 15
# # B[b:-b,:] = 0.; B[:,b:-b] = 0.                      #; print('B', B.shape, B.device, B.dtype)
# E_wrap = np.squeeze(np.asarray((torch.mm(GW,torch.mm(B,EW))).cpu().numpy()))

# ## Plot Corrected field and E_L
# E_corrected = EW_cpu - E_wrap
# error = relative_error(E_corrected, EO_cpu)
# end = time.time() - start
# print(b,'/',x_size, '(boundary points out of total points)')
# print('Time,\t\t', np.round(end, 2))
# print("Relative error,\t {:.2e}".format(error))
# print('-'*50)

# print('Done.')

# #%% #@title Compute and plot relative error vs. b
# # b_arr = np.arange(1, int(x_size/2+2))
# # error_list = []
# # for b in b_arr:
# #     M = torch.zeros_like(B)
# #     M[:b,:b] = 1.; M[-b:,-b:] = 1.; M[:b,-b:] = 1.; M[-b:,:b] = 1.  ### Corner-wise mask
# #     MB = torch.mul(M,B)
# #     # E_wrap_b = np.squeeze(np.asarray(GW @ MB @ EW))
# #     E_wrap_b = np.squeeze(np.asarray((torch.mm(GW,torch.mm(MB,EW))).cpu().numpy()))
# #     error_list.append(relative_error(EW_cpu - E_wrap_b, EO_cpu))
# #     print(b, end='\r')
# # errors = np.array(error_list)
# # # # print(np.round(errors, 2))
# # # np.set_printoptions(formatter={'float': lambda x: format(x, '.2E')})
# # # print(errors)

# # layout = go.Layout(title="Relative error vs. b (number of boundary gridpoints)",
# #                     xaxis=dict(title='b', range=[b_arr[0]-5, b_arr[-1]+5]), 
# #                     yaxis=dict(title='Relative error', type='log', tickformat='.0e'), 
# #                     height=400, width=800,
# #                     margin=dict(l=10, r=10, t=100, b=10),
# #                     font=dict(family="Times New Roman", size=18))
# # fig = go.Figure(layout=layout)
# # fig.add_trace(go.Scatter(x=b_arr, y=errors, mode='lines', name="Analytical", line=dict(color='red', width=1.5)))
# # fig.write_image('figures/1d_withM_vs_b_mat_pyt_sparse.pdf')
# # fig.write_html('figures/1d_withM_vs_b_mat_pyt_sparse.html')#, include_plotlyjs=False)
# # # fig.show()
# # print('Rel_err vs b plotted.')


#%% #@title Relative error and Plot to compare np. vs torch implementations
# print('Time np vs torch,\t\t', np.round(e1, 2), ',', np.round(e_, 2))
# print("Relative error np vs torch,\t {:.2e}, {:.2e}".format(error1, error))
# print("EO vs. EO1 {:.0e}".format(relative_error(EO_cpu, EO1)))
# print("EW vs. EW1 {:.0e}".format(relative_error(EW_cpu, EW1)))

# plt.subplots(nrows=1, ncols=2, figsize=(figsize[0],figsize[1]/2))
# plt.subplot(1,2,1)
# plt.plot(np.real(EO1), 'r', label='E_L')
# plt.plot(np.real(EO_cpu), 'b:', label=r'$E_{\Omega}$')
# plt.title('Linear')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(np.real(EW1), 'r', label='E')
# plt.plot(np.real(EW_cpu), 'b:', label=r'$E_W$')
# plt.title('Periodic')
# plt.legend()
# plt.show()
