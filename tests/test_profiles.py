from pathlib import Path

import hydra
import pytest

from classy.scripts.cli.train import apply_profiles_and_cli, ClassyBlame


def parse_test_spec(spec: str, hydra_test_env):
    path, content = None, []
    cli = []
    for line in spec.split("\n"):
        line = line.strip()
        if line == "":
            if path is not None:
                assert len(content) > 0
                if path == "cli":
                    for l in content:
                        assert not l.startswith(" ") or l.startswith("\t")
                        cli.append(l.replace(": ", "="))
                else:
                    output_path = f"{hydra_test_env}/{path}"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        f.write("\n".join(content) + "\n")
                path, content = None, []
        elif line.startswith("|"):
            content.append(line[2:])
        else:
            path, _content = line.split("|")
            path = path.rstrip()
            content.append(_content[1:])
    return cli


@pytest.fixture
def hydra_test_env(tmpdir):
    tmpdir = Path(tmpdir.strpath)
    (tmpdir / "profiles").mkdir()
    return str(tmpdir)


class TestProfiles:

    def _compute_final_config(self, spec, hydra_test_env):
        cli = parse_test_spec(spec, hydra_test_env=hydra_test_env)
        blames = apply_profiles_and_cli(
            config_name="root",
            config_dir=hydra_test_env,
            profile_path=f"{hydra_test_env}/profiles/profile.yaml",
            cli={ClassyBlame("cli"): cli},
            output_config_name="merged"
        )
        with hydra.initialize_config_dir(config_dir=hydra_test_env, job_name="test"):
            cfg = hydra.compose(config_name="merged")
        return cfg, blames

    def test_profile_overrides_value(self, hydra_test_env):
        """
        Check profile can override values
        """
        spec = """
        root.yaml               | a: 0

        profiles/profile.yaml   | a: 1
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == 1

    def test_profile_adds_value(self, hydra_test_env):
        """
        Check profile can add values
        """
        spec = """
        root.yaml               | a: 0

        profiles/profile.yaml   | b: 1
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg == {"a": 0, "b": 1}

    def test_cli_adds_value(self, hydra_test_env):
        """
        Check profile can add values
        """
        spec = """
        root.yaml               | a: 0

        profiles/profile.yaml   | b: 1

        cli                     | c: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg == {"a": 0, "b": 1, "c": 2}

    def test_profile_and_cli_override_value(self, hydra_test_env):
        """
        Check cli edits are prioritized over profile's
        """
        spec = """
        root.yaml               | a: 0

        profiles/profile.yaml   | a: 1

        cli                     | a: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == 2

    def test_profile_override_from_file_in_defaults(self, hydra_test_env):
        """
        Check profile overrides from file are correctly applied (file in root defaults)
        """
        spec = """
        root.yaml               | defaults:
                                |   - a: a1

        a/a1.yaml               | b: 0

        a/a2.yaml               | b: 1

        profiles/profile.yaml   | a: a2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == {"b": 1}

    def test_profile_override_from_file_not_in_defaults(self, hydra_test_env):
        """
        Check profile overrides from file are correctly applied (file not in root defaults)
        """
        spec = """
        root.yaml               | a: {}

        a/a2.yaml               | b: 1

        profiles/profile.yaml   | a: a2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == {"b": 1}

    def test_cli_override_on_profile_from_file(self, hydra_test_env):
        """
        Check cli edits are correctly applied over specs from file in profile
        """
        spec = """
        root.yaml               | a: {}

        a/a1.yaml               | b:
                                |   c: 0
                                |   d: 0

        profiles/profile.yaml   | a: a1

        cli                     | a.b.c: 1
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a.b == {"c": 1, "d": 0}

    def test_profile_override_target(self, hydra_test_env):
        """
        Check that, when profile changes _target_, profile's node entirely overwrites the original one
        (b is discarded and _target_ replaced)
        """
        spec = """
        root.yaml               | defaults:
                                | - a: a1

        a/a1.yaml               | _target_: A1
                                | b: 1

        profiles/profile.yaml   | a:
                                |   _target_: A2
                                |   c: 1
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a._target_ == "A2"
        assert "b" not in cfg.a
        assert cfg.a.c == 1

    def test_profile_override_not_on_target(self, hydra_test_env):
        """
        Check that, when profile does not change _target_, profile's node only updates the original one
        (_target_ is unchanged and b updated)
        """
        spec = """
        root.yaml               | defaults:
                                | - a: a1

        a/a1.yaml               | _target_: A1
                                | b: 1

        profiles/profile.yaml   | a:
                                |   b: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a._target_ == "A1"
        assert cfg.a.b == 2

    def test_profile_override_not_on_target_from_file(self, hydra_test_env):
        """
        Check that, when profile specifies a dict override from file (not on _target), profile's node only updates
        original one (_target_ is unchanged and b updated)
        """
        spec = """
        root.yaml               | defaults:
                                | - a: a1

        a/a1.yaml               | _target_: A1
                                | b: 1

        a/a2.yaml               | b: 2

        profiles/profile.yaml   | a: a2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a.b == 2

    def test_profile_override_list(self, hydra_test_env):
        """
        Check that, when profile specifies a list, it is appended to the corresponding original node
        """
        spec = """
        root.yaml               | defaults:
                                | - a: a1

        a/a1.yaml               | - a: 0
                                | - b: 0

        profiles/profile.yaml   | a:
                                |   - c: 0
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == [{"a": 0}, {"b": 0}, {"c": 0}]

    def test_profile_override_list_from_file(self, hydra_test_env):
        """
        Check that, when profile specifies a list override from file, it is appended to the corresponding original node
        """
        spec = """
        root.yaml               | defaults:
                                | - a: a1

        a/a1.yaml               | - a: 0
                                | - b: 0

        a/a2.yaml               | - c: 0

        profiles/profile.yaml   | a: a2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == [{"a": 0}, {"b": 0}, {"c": 0}]

    def test_interpolation_on_value(self, hydra_test_env):
        """
        Check interpolation is preserved after profile application
        """
        spec = """
        root.yaml               | a: 0
                                | b: ${a}

        profiles/profile.yaml   | a: 1

        cli                     | a: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.b == 2

    def test_interpolation_node_with_value_override(self, hydra_test_env):
        """
        Check that standard profile rules apply also on interpolated nodes (value override)
        """
        spec = """
        root.yaml               | a: 0
                                | b:
                                |   _target_: B1
                                |   c: 1
                                |   d: ${a}
                                | e: ${b}

        profiles/profile.yaml   | e:
                                |   c: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.b.c == 1
        assert cfg.e.c == 2

    def test_cli_override_on_profile_from_file_on_interpolation(self, hydra_test_env):
        """
        Check cli edits are correctly applied over specs from file in profile
        """
        spec = """
        root.yaml               | a: {}

        a/a1.yaml               | b:
                                |   c:
                                |     d: 0
                                |   e: ${a.b.c}

        profiles/profile.yaml   | a: a1

        cli                     | a.b.e.d: 1
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a.b.c == {"d": 0}
        assert cfg.a.b.e == {"d": 1}

    def test_interpolation_node_with_interpolation_preserved(self, hydra_test_env):
        """
        Check that standard profile rules apply also on interpolated nodes (interpolation preserved)
        """
        spec = """
        root.yaml               | a: 0
                                | b:
                                |   _target_: B1
                                |   c: 1
                                |   d: ${a}
                                | e: ${b}

        profiles/profile.yaml   | e:
                                |   c: 2

        cli                     | a: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.b.d == 2
        assert cfg.e.d == 2

    def test_interpolation_node_with_interpolation_discarded(self, hydra_test_env):
        """
        Check that standard profile rules apply also on interpolated nodes (interpolation discarded)
        """
        spec = """
        root.yaml               | a: 0
                                | b:
                                |   _target_: B1
                                |   c: 1
                                |   d: ${a}
                                | e: ${b}

        profiles/profile.yaml   | e:
                                |   c: 2
                                |   d: 4

        cli                     | a: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.b.d == 2
        assert cfg.e.d == 4

    def test_file_on_cli(self, hydra_test_env):
        """
        Check that, when cli overrides contain a file, it is handled as if it was written as part of profile
        """
        spec = """
        root.yaml               | v: 0
                                |
                                | defaults:
                                | - a: a1

        a/a1.yaml               | b: 1
                                | c: 1

        a/a2.yaml               | c: 3

        profiles/profile.yaml   | v: 1
                                | a:
                                |   b: 2

        cli                     | a: a2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.v == 1
        assert cfg.a == {"b": 1, "c": 3}

    def test_profile_override_to_null(self, hydra_test_env):
        """
        Check that, when cli overrides something to null, it is properly handled
        """
        spec = """
        root.yaml               | a:
                                |   b: 1
                                |   c: 2

        profiles/profile.yaml   | a: null
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a is None

    def test_profile_override_from_null(self, hydra_test_env):
        """
        Check that, when cli overrides null to something, it is properly handled
        """
        spec = """
        root.yaml               | a: null

        profiles/profile.yaml   | a:
                                |   b: 1
                                |   c: 2
        """
        cfg, blames = self._compute_final_config(spec, hydra_test_env)
        assert cfg.a == {"b": 1, "c": 2}
