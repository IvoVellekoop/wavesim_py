
import time
import numpy as np
from anysim_combined import AnySim

## Run tests
''' --- TESTS ---'''

## Custom user-defined run
N = 256         # medium size in pixels
n = np.ones(N)  # refractive index distribution
source_amplitude = 1.
source_location = 0
N_domains = [1,3]# np.arange(1,6)#      # Number of sub-domains for domain decomposition (=[1] for no domain decomposition)

s1 = time.time()

for idx,i in enumerate(N_domains):
    anysim = AnySim(N=N, n=n, source_amplitude=source_amplitude, source_location=source_location, N_domains=i)
    
    if idx==0:
        x = np.arange(0,anysim.N_roi*anysim.pixel_size,anysim.pixel_size)
        x = np.pad(x, (64,64), mode='constant')
        h = anysim.pixel_size
        k = anysim.k0
        phi = k * x

        E_theory = 1.0j*h/(2*k) * np.exp(1.0j*phi) - h/(4*np.pi*k) * (np.exp(1.0j * phi) * ( np.exp(1.0j * (k-np.pi/h) * x) - np.exp(1.0j * (k+np.pi/h) * x)) - np.exp(-1.0j * phi) * ( -np.exp(-1.0j * (k-np.pi/h) * x) + np.exp(-1.0j * (k+np.pi/h) * x)))
        # special case for values close to 0
        small = np.abs(k*x) < 1.e-10
        E_theory[small] = 1.0j * h/(2*k) * (1 + 2j * np.arctanh(h*k/np.pi)/np.pi); # exact value at 0.
        u_true = E_theory[64:-64]

    anysim.runit(u_true)

e1 = time.time() - s1
print('Total time (including plotting): ', np.round(e1,2))
print('-'*50)


'''
Default options for AnySim()
test =              'custom'
lambd =             1
ppw =               4
boundaries_width =  20
N =                 256
n =                 [0]
source_amplitude =  1.
source_location =   0
domain_decomp =     True
N_domains =         2
overlap =           20
wrap_correction =   'None'
cp =                20
'''


# s2 = time.time()
# anysim = AnySim(wrap_correction='L_omega')
# anysim.runit()
# e2 = time.time() - s2
# print('Total time (including plotting): ', np.round(e2,2))
# print('-'*50)

# s3 = time.time()
# anysim = AnySim(wrap_correction='L_corr')
# anysim.runit()
# e3 = time.time() - s3
# print('Total time (including plotting): ', np.round(e3,2))
# print('-'*50)

# print('Done')