# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.1] - Unreleased

### Added

- Python 3.14 support (`requires-python = ">=3.11,<3.15"`). On Python 3.14
  this needs `pyiron_workflow_vasp>=0.2.1` (the first vasp release that allows
  3.14); resolvers select it automatically, so the declared floor stays at
  0.2.0, which remains the true API floor.

## [0.2.0] - Unreleased

### Changed

- Ported to the `pyiron_workflow` 0.19/0.20 line (`pyiron_workflow>=0.19,<0.21`,
  `flowrep>=0.6,<0.7`) and to `pyiron_workflow_vasp>=0.2.0,<0.3`.
- Python 3.11-3.13 supported.

### Fixed

- `collect_structures` and `concat_and_save` accept the dict shape produced by
  the default parser.
- Packaging: the version is single-sourced from `pyiron_workflow_assyst/__init__.py`,
  dependency bounds and project metadata corrected, a `LICENSE` file is shipped,
  and the VASP example is no longer packaged. A plain `pip install` now works
  in a fresh environment with build isolation.
- Tests run without a real POTCAR library or `.pyiron_vasp_config`.

### Added

- CI: `clean-install` workflow (isolated sdist/wheel build, install into an
  empty venv, `pip check`, test suite) and a `release` workflow that publishes
  to PyPI via trusted publishing.
