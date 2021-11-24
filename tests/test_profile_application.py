import shutil
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

import classy
from classy.scripts.cli.train import apply_profile_on_dir


@pytest.fixture
def hydra_test_env(tmpdir):

    tmpdir = Path(tmpdir.strpath)

    OmegaConf.save(
        OmegaConf.create(
            {
                "i": 0,
                "l": [0, 0],
                "d": {"a": "a", "b": "b", "c": "c"},
                "o": {"_target_": "my.custom.module", "p1": "${interpolation}"},
                "interpolation": "interpolation",
                "defaults": [{"x": "a"}, {"z": "a"}, "_self_"],
            }
        ),
        tmpdir / "root.yaml",
    )

    (tmpdir / "x").mkdir()
    OmegaConf.save(OmegaConf.create({"a": "0"}), tmpdir / "x" / "a.yaml")
    OmegaConf.save(OmegaConf.create({"b": "0"}), tmpdir / "x" / "b.yaml")
    OmegaConf.save(OmegaConf.create({"c": "0"}), tmpdir / "x" / "c.yaml")

    (tmpdir / "y").mkdir()
    OmegaConf.save(OmegaConf.create({"a": "0"}), tmpdir / "y" / "a.yaml")
    OmegaConf.save(OmegaConf.create({"b": "0"}), tmpdir / "y" / "b.yaml")

    (tmpdir / "z").mkdir()
    OmegaConf.save(OmegaConf.create([0, 1]), tmpdir / "z" / "a.yaml")

    return str(tmpdir), "root"


class TestProfileApplication:
    def _load_hydra_conf(self, hydra_config_dir, hydra_config_name, cli):
        with hydra.initialize_config_dir(config_dir=hydra_config_dir, job_name="test"):
            return hydra.compose(config_name=hydra_config_name, overrides=cli)

    def _run_skeleton(self, hydra_config_dir, hydra_config_name, profile, cli):

        before_cfg = self._load_hydra_conf(hydra_config_dir, hydra_config_name, cli)

        apply_profile_on_dir(profile, profile_name="", config_name=hydra_config_name, config_dir=hydra_config_dir)

        after_cfg = self._load_hydra_conf(hydra_config_dir, hydra_config_name, cli)

        return before_cfg, after_cfg

    def test_empty_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({}), cli=[])
        assert before_cfg == after_cfg

    def test_root_primitive_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"i": 1}), cli=[])

        assert before_cfg.i != after_cfg.i
        assert after_cfg.i == 1

        # check that the above change was the only one applied
        after_cfg.i = before_cfg.i
        assert before_cfg == after_cfg

    def test_none_on_root_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"o": None}), cli=[])

        assert before_cfg.o != after_cfg.o
        assert after_cfg.o is None

        # check that the above change was the only one applied
        after_cfg.o = before_cfg.o
        assert before_cfg == after_cfg

    def test_root_list_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"l": [1]}), cli=[])

        assert before_cfg.l != after_cfg.l
        assert after_cfg.l == [1]

        # check that the above change was the only one applied
        after_cfg.l = before_cfg.l
        assert before_cfg == after_cfg

    def test_root_dict_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(
            *hydra_test_env, profile=OmegaConf.create({"d": {"a": "_a", "d": "_d"}}), cli=[]
        )

        assert before_cfg.d != after_cfg

        # check replacement
        assert before_cfg.d.a != after_cfg.d.a
        assert after_cfg.d.a == "_a"

        # check addittivity (b has remained identical)
        assert before_cfg.d.b == after_cfg.d.b

        # check new insertion
        assert "d" not in before_cfg.d
        assert "d" in after_cfg.d

        # check that the above change was the only one applied
        after_cfg.d = before_cfg.d
        assert before_cfg == after_cfg

    def test_root_obj_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(
            *hydra_test_env, profile=OmegaConf.create({"o": {"_target_": "my.other.custom.module", "p2": "p2"}}), cli=[]
        )

        assert before_cfg.d != after_cfg

        # check _target has changed correctly
        assert before_cfg.o._target_ != after_cfg.o._target_
        assert after_cfg.o._target_ == "my.other.custom.module"

        # check p1 has been removed
        assert "p1" in before_cfg.o
        assert "p1" not in after_cfg.o

        # check p2 has been added
        assert "p2" not in before_cfg.o
        assert "p2" in after_cfg.o

    def test_root_additional_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"s": "s"}), cli=[])

        assert "s" not in before_cfg
        assert after_cfg.s == "s"

        # check that the above change was the only one applied
        del after_cfg.s
        assert before_cfg == after_cfg

    def test_root_interpolation_variable_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(
            *hydra_test_env, profile=OmegaConf.create({"interpolation": "another-interpolation"}), cli=[]
        )

        assert before_cfg.interpolation != after_cfg.interpolation
        assert after_cfg.interpolation == "another-interpolation"
        assert after_cfg.o.p1 == "another-interpolation"

        # check that the above change was the only one applied
        # this works also here, with an interpolation, as no materialization took place
        after_cfg.interpolation = before_cfg.interpolation
        assert before_cfg == after_cfg

    def test_dict_override_in_config_group_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"x": {"a": "1"}}), cli=[])

        assert before_cfg.x.a == "0"
        assert after_cfg.x.a == "1"

        # check that the above change was the only one applied
        # this works also here, with an interpolation, as no materialization took place
        after_cfg.x.a = before_cfg.x.a
        assert before_cfg == after_cfg

    def test_list_override_in_config_group_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"z": [1, 2]}), cli=[])

        assert before_cfg.z == [0, 1]
        assert after_cfg.z == [1, 2]

        # check that the above change was the only one applied
        # this works also here, with an interpolation, as no materialization took place
        after_cfg.z = before_cfg.z
        assert before_cfg == after_cfg

    def test_config_group_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"x": "b"}), cli=[])

        assert before_cfg.x != after_cfg.x

        # check a was deleted
        assert "a" in before_cfg.x
        assert "a" not in after_cfg.x

        # check b was introduced
        assert "b" not in before_cfg.x
        assert "b" in after_cfg.x

        # check that the above change was the only one applied
        after_cfg.x = before_cfg.x
        assert before_cfg == after_cfg

    def test_new_config_group_set_from_profile(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"y": "a"}), cli=[])

        assert before_cfg != after_cfg

        assert "y" not in before_cfg
        assert "y" in after_cfg
        assert after_cfg.y.a == "0"

        # check that the above change was the only one applied
        del after_cfg.y
        assert before_cfg == after_cfg

    def test_variable_intersection_between_profile_and_cli(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"i": "1"}), cli=["i=2"])

        assert after_cfg.i == 2

    def test_config_group_intersection_between_profile_and_cli(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({"x": "b"}), cli=["x=c"])

        assert after_cfg.x == OmegaConf.create({"c": "0"})

    def test_config_group_set_from_cli(self, hydra_test_env):
        before_cfg, after_cfg = self._run_skeleton(*hydra_test_env, profile=OmegaConf.create({}), cli=["+y=a"])

        assert after_cfg.y == OmegaConf.create({"a": "0"})
