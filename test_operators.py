import numpy as np
from anysim_combined import AnySim

def test_Contraction():
    anysim = AnySim(test='Test_1DFreeSpace')
    # vc = np.max(np.abs(anysim.V))
    vc = np.linalg.norm(np.diag(anysim.V), 2)
    print(vc)
    assert vc < 1, f'||V|| not < 1, but {vc}'

def test_Accretive():
    anysim = AnySim(test='Test_1DFreeSpace')

    L_plus_1_inv = anysim.propagator(np.eye(anysim.N_FastConv[0]))
    L_plus_1 = np.linalg.inv(L_plus_1_inv)
    B = anysim.medium(np.eye(anysim.N_FastConv[0]))
    A = L_plus_1 - B

    acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.conj(A).T))))
    
    assert np.round(acc, 7) >= 0, f'A is not accretive. {acc}'
