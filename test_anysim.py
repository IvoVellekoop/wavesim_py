import numpy as np
from anysim_combined import AnySim
from scipy.io import loadmat
import unittest
# import pytest

class Tests_1D(unittest.TestCase):
    def test_1DFreeSpace(self):
        N = 256
        n = np.ones(N)
        N_domains = 1

        anysim = AnySim(test='FreeSpace', n=n, N=N, N_domains=N_domains)
        
        ## Compare with the analytic solution
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

        self.assertLessEqual( anysim.runit(u_true), 1.e-3)
    
    def test_1DGlassPlate(self):
        N = 256
        n = np.ones(N)
        n[99:130] = 1.5
        N_domains = 1

        u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])

        anysim = AnySim(test='1DGlassPlate', n=n, N=N, N_domains=N_domains)
        self.assertLessEqual( anysim.runit(u_true), 1.e-3)


if __name__ == '__main__':
    unittest.main()
    # unittest.main()
