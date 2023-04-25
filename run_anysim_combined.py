import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# topic = 'Helmholtz'
""" Helmholtz (the only one for now) or Maxwell """

# test = 'FreeSpace'
""" 
'FreeSpace'
    (Simulates free-space propagation and compares the result to the analytical solution), OR
'1D'
    (Simulates 1-D propagation of light through a slab of glass), OR
'2D'
    (Simulates propagation of light in a 2-D structure made of iron (uses the scalar wave equation)), OR 
'2D_low_contrast'
    (Simulates propagation of light in a 2-D structure made of fat and water (uses the scalar wave equation)) 
"""

# smallest_circle_problem = False
""" True (V0 as in AnySim) or False (V0 as in WaveSim) """

# wrap_around = 'boundaries'
""" 
'boundaries'
    (add absorbing boundaries), OR
'L_Omega'
    (Do the fast convolution over a much larger domain to eliminate wrap-around effects without absorbing boundaries), OR
'L_corr'
    (Add the wrap-aroound correction term to V to correct for the wrap-around effects without absorbing boundaries)
"""

import time

from anysim_combined import AnySim
if __name__ == '__main__':

    s1 = time.time()
    anysim = AnySim()
    u1 = anysim.runit()
    e1 = time.time() - s1
    print('Total time (including plotting): ', np.round(e1,2))
    print('-'*50)

    s2 = time.time()
    anysim = AnySim(wrap_around='L_Omega')
    u2 = anysim.runit()
    e2 = time.time() - s2
    print('Total time (including plotting): ', np.round(e2,2))
    print('-'*50)

    s3 = time.time()
    anysim = AnySim(wrap_around='L_corr')
    u3 = anysim.runit()
    e3 = time.time() - s3
    print('Total time (including plotting): ', np.round(e3,2))
    print('-'*50)

    print('Done')

    plt.show()
