import numpy as np

from latent_capture.metrics import pass_movement, pca_2d, top_coordinate_share


def test_top_coordinate_share_finds_dominant():
    x = np.ones((10, 8), dtype=np.float32)
    x[:, 5] = 3.0
    idx, share = top_coordinate_share(x)
    assert idx == 5
    assert round(share, 4) == round(9 / 16, 4)


def test_pass_movement_zero_when_identical():
    p = np.tile(np.random.default_rng(0).normal(size=(4, 1, 6)), (1, 3, 1))
    assert np.allclose(pass_movement(p), [0.0, 0.0])


def test_pass_movement_orders_big_then_small():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 1.0]])
    m = pass_movement(a[None])
    assert m[0] > m[1] > 0


def test_pca_2d_shape_and_centering():
    y = pca_2d(np.random.default_rng(1).normal(size=(30, 12)))
    assert y.shape == (30, 2)
    assert np.allclose(y.mean(axis=0), 0, atol=1e-6)
