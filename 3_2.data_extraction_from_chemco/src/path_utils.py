from pathlib import Path


def resolve_project_paths(cfg: dict) -> dict:
    paths_cfg = cfg["paths"]
    files_cfg = cfg["files"]
    logging_cfg = cfg["logging"]
    saving_cfg = cfg["saving"]

    project_root = Path(".").resolve()

    input_dir = project_root / paths_cfg["input_dir"]
    output_dir = project_root / paths_cfg["output_dir"]
    log_dir = project_root / paths_cfg["log_dir"]

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    input_files = [input_dir / name for name in files_cfg["input_files"]]

    output_file = output_dir / files_cfg["output_file"]
    summary_file = output_dir / files_cfg["summary_file"]

    cp_only_file = output_dir / files_cfg["cp_only_file"]
    cp_liquid_file = output_dir / files_cfg["cp_liquid_file"]
    cp_solid_file = output_dir / files_cfg["cp_solid_file"]
    cp_gas_file = output_dir / files_cfg["cp_gas_file"]

    log_file = log_dir / logging_cfg["log_file"]

    partial_suffix = saving_cfg.get("partial_suffix", "_partial")
    partial_output_file = output_file.with_name(
        f"{output_file.stem}{partial_suffix}{output_file.suffix}"
    )

    return {
        "project_root": project_root,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "log_dir": log_dir,
        "input_files": input_files,
        "output_file": output_file,
        "partial_output_file": partial_output_file,
        "summary_file": summary_file,
        "cp_only_file": cp_only_file,
        "cp_liquid_file": cp_liquid_file,
        "cp_solid_file": cp_solid_file,
        "cp_gas_file": cp_gas_file,
        "log_file": log_file,
    }


def build_partial_output_path(output_file: Path, suffix: str = "_partial") -> Path:
    return output_file.with_name(f"{output_file.stem}{suffix}{output_file.suffix}")
