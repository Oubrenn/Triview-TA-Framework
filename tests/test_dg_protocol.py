from src.datasets import ViewConfig, _sample_transform_params


def test_color_severity_id_matches_levels():
    cfg = ViewConfig(
        shift_bins=[0.0],
        scale_ratios=[1.0],
        color_bands=4,
        color_max_gain_db_levels=[1.0, 3.0, 6.0],
    )
    d0 = _sample_transform_params(cfg, num_bins=33, seed=123, domain_id=0)
    d1 = _sample_transform_params(cfg, num_bins=33, seed=123, domain_id=1)
    d2 = _sample_transform_params(cfg, num_bins=33, seed=123, domain_id=2)
    assert int(d0["color_severity_id"]) == 0
    assert int(d1["color_severity_id"]) == 1
    assert int(d2["color_severity_id"]) == 2
    assert float(d0["color_max_gain_db"]) == 1.0
    assert float(d1["color_max_gain_db"]) == 3.0
    assert float(d2["color_max_gain_db"]) == 6.0


def test_domain_id_roundtrip():
    cfg = ViewConfig(
        shift_bins=[-1.0, 1.0],
        scale_ratios=[0.9, 1.1],
        color_bands=4,
        color_max_gain_db_levels=[3.0, 6.0],
    )
    total = len(cfg.shift_bins) * len(cfg.scale_ratios) * len(cfg.color_max_gain_db_levels)
    for domain_id in range(total):
        draw = _sample_transform_params(cfg, num_bins=33, seed=77, domain_id=domain_id)
        assert int(draw["domain_id"]) == domain_id
