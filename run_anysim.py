
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
N_domains = [1,3]# np.arange(1,6)# 

s1 = time.time()

for i in N_domains:
    # anysim = AnySim(N=N, n=n, source_amplitude=source_amplitude, source_location=source_location, N_domains=i)
    # anysim.runit()

    anysim = AnySim(test='FreeSpace', N_domains=i)
    anysim.runit()

    anysim = AnySim(test='1D', N_domains=i)
    anysim.runit()

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