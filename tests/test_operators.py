import numpy as np
from anysim_main import AnySim

def test_Contraction():
    n = np.ones((256,1,1))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim1D_FS = AnySim(n=n, source=source)
    anysim1D_FS.setup_operators_n_initialize()
    # vc = np.max(np.abs(anysim1D_FS.v))
    vc = np.linalg.norm(np.diag(np.squeeze(anysim1D_FS.v)), 2)
    print(vc)
    assert vc < 1, f'||V|| not < 1, but {vc}'

def test_Accretive():
    n = np.ones((256,1,1))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim1D_FS = AnySim(n=n, source=source)
    anysim1D_FS.setup_operators_n_initialize()

    L_plus_1_inv = anysim1D_FS.propagator(np.eye(anysim1D_FS.n_fast_conv[0]))
    L_plus_1 = np.linalg.inv(L_plus_1_inv)
    B = anysim1D_FS.medium_operators[0](np.eye(anysim1D_FS.n_fast_conv[0]))
    A = L_plus_1 - B

    acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.conj(A).T))))
    
    assert np.round(acc, 7) >= 0, f'A is not accretive. {acc}'
