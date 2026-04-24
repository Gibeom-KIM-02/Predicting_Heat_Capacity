from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError('YAML config must load into a dictionary.')

    cfg['config_path'] = config_path
    cfg['project_root'] = _resolve_project_root(cfg, config_path)
    cfg['input_root'] = _resolve_input_root(cfg)
    cfg['output_root'] = _resolve_output_root(cfg)
    cfg['resolved_thermoml_json_roots'] = _resolve_thermoml_json_roots(cfg)
    return cfg


def _resolve_project_root(cfg: dict[str, Any], config_path: Path) -> Path:
    raw = cfg.get('project_root', '.')
    return _resolve_path(raw, config_path.parent)


def _resolve_input_root(cfg: dict[str, Any]) -> Path:
    project_root = cfg['project_root']
    input_cfg = cfg.get('input', {})
    raw = input_cfg.get('root', 'input')
    return _resolve_path(raw, project_root)


def _resolve_output_root(cfg: dict[str, Any]) -> Path:
    project_root = cfg['project_root']

    # Backward compatibility: old key output_root still works.
    if 'output_root' in cfg and not isinstance(cfg['output_root'], Path):
        return _resolve_path(cfg['output_root'], project_root)

    output_cfg = cfg.get('output', {})
    raw = output_cfg.get('root', 'output/CP_PIPELINE_OUTPUT')
    return _resolve_path(raw, project_root)


def _resolve_thermoml_json_roots(cfg: dict[str, Any]) -> list[Path]:
    project_root = cfg['project_root']
    input_root = cfg['input_root']
    input_cfg = cfg.get('input', {})

    subdirs = input_cfg.get('thermoml_subdirs')
    explicit_roots = cfg.get('thermoml_json_roots')

    if subdirs is not None:
        if not isinstance(subdirs, list):
            raise TypeError("input.thermoml_subdirs must be a list.")

        bad = [x for x in subdirs if not isinstance(x, (str, Path))]
        if bad:
            raise TypeError(
                "All entries in input.thermoml_subdirs must be strings. "
                f"Bad entries: {bad!r}. "
                "If you intended folder names like 10.1007, wrap them in quotes in YAML."
            )

        return [_resolve_path(subdir, input_root) for subdir in subdirs]

    if explicit_roots is not None:
        if not isinstance(explicit_roots, list):
            raise TypeError("thermoml_json_roots must be a list.")

        bad = [x for x in explicit_roots if not isinstance(x, (str, Path))]
        if bad:
            raise TypeError(
                "All entries in thermoml_json_roots must be strings or Paths. "
                f"Bad entries: {bad!r}"
            )

        return [_resolve_path(root, project_root) for root in explicit_roots]

    raise KeyError(
        'Provide either input.thermoml_subdirs or thermoml_json_roots in the YAML config.'
    )


def prepare_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    output_root = Path(cfg['output_root']).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_dir = output_root / 'raw_master'
    clean_dir = output_root / 'clean_master'
    window_dir = output_root / 'derived_datasets'

    for d in [raw_dir, clean_dir, window_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        'project_root': Path(cfg['project_root']).resolve(),
        'config_path': Path(cfg['config_path']).resolve(),
        'input_root': Path(cfg['input_root']).resolve(),
        'output_root': output_root,
        'raw_dir': raw_dir,
        'clean_dir': clean_dir,
        'window_dir': window_dir,
    }

def _resolve_path(raw_path: Any, base_dir: Path) -> Path:
    if raw_path is None:
        raise ValueError("Path value is None.")

    if isinstance(raw_path, Path):
        path = raw_path
    elif isinstance(raw_path, (str, bytes)):
        path = Path(raw_path)
    elif isinstance(raw_path, (int, float)):
        raise TypeError(
            f"Path-like config value must be a string, not {type(raw_path).__name__}: {raw_path!r}. "
            "If this is a directory name like 10.1007, quote it in YAML."
        )
    else:
        raise TypeError(
            f"Unsupported path value type: {type(raw_path).__name__} ({raw_path!r})"
        )

    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()

