import numpy as np
from anysim_combined import AnySim

def test_Contraction():
    N_roi = np.array([256])
    n = np.ones(tuple(N_roi))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim1D_FS = AnySim(test='Test_1DFreeSpace', N_roi=N_roi, n=n, source=source)
    anysim1D_FS.setup_operators_n_init_variables()
    # vc = np.max(np.abs(anysim1D_FS.V))
    vc = np.linalg.norm(np.diag(anysim1D_FS.V), 2)
    print(vc)
    assert vc < 1, f'||V|| not < 1, but {vc}'

def test_Accretive():
    N_roi = np.array([256])
    n = np.ones(tuple(N_roi))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim1D_FS = AnySim(test='Test_1DFreeSpace', N_roi=N_roi, n=n, source=source)
    anysim1D_FS.setup_operators_n_init_variables()

    L_plus_1_inv = anysim1D_FS.propagator(np.eye(anysim1D_FS.N_FastConv[0]))
    L_plus_1 = np.linalg.inv(L_plus_1_inv)
    B = anysim1D_FS.medium(np.eye(anysim1D_FS.N_FastConv[0]))
    A = L_plus_1 - B

    acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.conj(A).T))))
    
    assert np.round(acc, 7) >= 0, f'A is not accretive. {acc}'
