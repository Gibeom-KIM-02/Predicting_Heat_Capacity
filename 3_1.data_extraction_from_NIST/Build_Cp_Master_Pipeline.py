#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.cleaning import build_clean_master
from scripts.config_utils import load_config, prepare_output_dirs
from scripts.dataset_builder import build_representative_dataset, build_temperature_window_dataset
from scripts.structure_features import configure_rdkit_logging
from scripts.thermoml_parser import build_raw_master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build raw/clean/windowed Cp master datasets from ThermoML.'
    )
    parser.add_argument('--config', required=True, help='YAML config path')
    return parser.parse_args()


def run_pipeline(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    paths = prepare_output_dirs(cfg)

    configure_rdkit_logging(str(cfg.get('rdkit_log_level', 'warning')))

    print('=' * 80)
    print('Cp master pipeline')
    print('=' * 80)
    print(f"Config       : {paths['config_path']}")
    print(f"Project root : {paths['project_root']}")
    print(f"Input root   : {paths['input_root']}")
    print(f"Output root  : {paths['output_root']}")
    print('ThermoML roots:')
    for root in cfg['resolved_thermoml_json_roots']:
        print(f'  - {root}')

    raw_df = build_raw_master(cfg, paths)
    clean_df = build_clean_master(raw_df, cfg, paths)
    build_temperature_window_dataset(clean_df, cfg, paths)
    build_representative_dataset(clean_df, cfg, paths)

    print('\n' + '=' * 80)
    print('Pipeline finished')
    print('=' * 80)
    print(f"RAW    : {paths['raw_dir']}")
    print(f"CLEAN  : {paths['clean_dir']}")
    print(f"WINDOW : {paths['window_dir']}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == '__main__':
    main()