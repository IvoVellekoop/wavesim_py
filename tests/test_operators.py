import numpy as np
from anysim_base import AnySimBase


def test_contraction():
    n = np.ones((256, 1, 1))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim_1d_fs = AnySimBase(n=n, source=source)
    anysim_1d_fs.setup_operators_n_initialize()
    # vc = np.max(np.abs(anysim_1d_fs.v))
    vc = np.linalg.norm(np.diag(np.squeeze(anysim_1d_fs.v)), 2)
    print(vc)
    assert vc < 1, f'||V|| not < 1, but {vc}'


def test_accretive():
    n = np.ones((256, 1, 1))
    source = np.zeros_like(n)
    source[0] = 1.
    anysim_1d_fs = AnySimBase(n=n, source=source)
    anysim_1d_fs.setup_operators_n_initialize()

    l_plus_1_inv = anysim_1d_fs.propagator(np.eye(anysim_1d_fs.n_fast_conv[0]))
    l_plus_1 = np.linalg.inv(l_plus_1_inv)
    b = anysim_1d_fs.medium_operators[0](np.eye(anysim_1d_fs.n_fast_conv[0]))
    a = l_plus_1 - b

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))

    assert np.round(acc, 7) >= 0, f'a is not accretive. {acc}'
