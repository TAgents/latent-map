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


def test_dropped_coordinate_is_really_ignored():
    x, lab = blobs()
    m = fit_map(x, dims=8, cells=4, drop=7)
    loud = x.copy()
    loud[:, :, 7] = 1e6  # a huge value in the dropped coordinate must change nothing
    np.testing.assert_array_equal(place(m, loud), place(m, x))


def test_scale_passes_puts_passes_of_different_spread_in_the_same_cells():
    r = np.random.default_rng(1)
    cents = r.normal(size=(4, 30))
    lab = np.repeat(np.arange(4), 80)
    shared = r.normal(size=(320, 30)) * 3            # a common direction, so unit-normalising keeps the spread
    x = np.stack([cents[lab] * s + shared * 0 + r.normal(size=(320, 30)) * 0.02 * s for s in (1.0, 1.0, 1.0)], axis=1)
    x = x + 6.0                                      # shared offset dominates the norm
    x[:, 2] = (x[:, 2] - 6.0) * 4 + 6.0              # the last pass spreads four times wider
    plain = place(fit_map(x, dims=8, cells=4, drop=29), x)
    scaled_map = fit_map(x, dims=8, cells=4, drop=29, scale_passes=True)
    scaled = place(scaled_map, x)
    assert (scaled[:, 0] == scaled[:, 2]).mean() > 0.95
    assert (scaled[:, 0] == scaled[:, 2]).mean() >= (plain[:, 0] == plain[:, 2]).mean()
    assert scaled_map["pass_scales"].shape == (3,)


def test_fit_can_ignore_some_points_but_still_place_them():
    x, lab = blobs()
    m = fit_map(x[:200], dims=8, cells=4, drop=7)
    assert place(m, x).shape == (240, 3)
