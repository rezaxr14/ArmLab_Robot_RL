import pytest

from armforge.config import RunConfig, apply_overrides, deep_merge, from_dict, load_config, save_config, to_dict


def test_defaults_roundtrip(tmp_path):
    cfg = RunConfig()
    save_config(cfg, tmp_path / "c.yaml")
    assert to_dict(load_config(tmp_path / "c.yaml")) == to_dict(cfg)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError, match="unknown"):
        from_dict(RunConfig, {"train": {"total_stepz": 1}})


def test_dotted_overrides_and_tuples(tmp_path):
    cfg = load_config(None, {"train.total_steps": 5, "env.physics.table_friction": [0.4, 0.01, 0.0002], "algo.hidden": [32, 32]})
    assert cfg.train.total_steps == 5
    assert cfg.env.physics.table_friction == (0.4, 0.01, 0.0002)
    assert cfg.algo.hidden == (32, 32)


def test_int_is_coerced_to_float():
    cfg = from_dict(RunConfig, {"algo": {"gamma": 1}})
    assert isinstance(cfg.algo.gamma, float)


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 9}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 9, "z": 2}} and a == {"x": {"y": 1, "z": 2}}
    assert apply_overrides(a, {"x.y": 3})["x"]["y"] == 3


def test_all_presets_load():
    from pathlib import Path

    for f in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        cfg = load_config(f)
        assert cfg.env.task.name in {"reach", "push", "pickplace"}, f
        assert cfg.train.total_steps > 0
