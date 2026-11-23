import numpy as np

from mapbuild.fit import fit_map, place


def blobs(seed=0):
    r = np.random.default_rng(seed)
    cents = r.normal(size=(4, 40)) * 4
    lab = np.repeat(np.arange(4), 60)
    x = cents[lab][:, None, :] + r.normal(size=(240, 3, 40)) * 0.3
    x += np.array([0.0, 5.0, 10.0])[None, :, None]  # a per-pass offset to be removed
    return x, lab


def test_fit_map_shapes_and_clipping():
    x, _ = blobs()
    m = fit_map(x, dims=64, cells=16, drop=7)
    assert m["components"].shape == (39, 39) and m["centres"].shape == (16, 39) and m["xy"].shape == (16, 2)


def test_place_separates_blobs_and_ignores_pass_offset():
    x, lab = blobs()
    m = fit_map(x, dims=8, cells=4, drop=7)
    c = place(m, x)
    assert c.shape == (240, 3)
    assert (c[:, 0] == c[:, 2]).mean() > 0.95  # same blob lands in the same cell on every pass
    assert all(len(set(c[lab == k, 0])) == 1 for k in range(4))


def test_place_uses_stored_means():
    x, _ = blobs()
    m = fit_map(x, dims=8, cells=4, drop=7)
    np.testing.assert_array_equal(place(m, x[:10]), place(m, x)[:10])
