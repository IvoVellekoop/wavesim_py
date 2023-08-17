import numpy as np
from helmholtzbase import HelmholtzBase


def test_accretive():
    """ Check that sum of propagator and medium operators (L + V) is accretive
        , i.e., has a non-negative real part"""
    n = np.ones((256, 1, 1)).astype(np.float32)
    source = np.zeros_like(n)
    source[0] = 1.
    anysim = HelmholtzBase(n=n, source=source)
    l_plus_1_inv = anysim.propagator(np.eye(anysim.n_fast_conv[0], dtype=np.single))
    l_plus_1 = np.linalg.inv(l_plus_1_inv)
    b = anysim.medium_operators[(0, 0, 0)](np.eye(anysim.n_fast_conv[0], dtype=np.single))
    a = l_plus_1 - b

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    assert np.round(acc, 5) >= 0, f'a is not accretive. {acc}'


def test_contraction():
    """ Test whether potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    n = np.ones((256, 1, 1)).astype(np.float32)
    source = np.zeros_like(n)
    source[0] = 1.
    anysim = HelmholtzBase(n=n, source=source)
    # vc = np.max(np.abs(anysim.v))
    vc = np.linalg.norm(np.diag(np.squeeze(anysim.v)), 2)
    print(vc)
    assert vc < 1, f'||V|| not < 1, but {vc}'
