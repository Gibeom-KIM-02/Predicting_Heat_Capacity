import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


OUTPUT_COLUMNS = [
    "Cp_Value_Raw",
    "Cp_Unit_Raw",
    "Cp_Value_J_per_mol_K",
    "Cp_Phase",
    "Cp_Temperature_K",
    "Tboil_Value_Raw",
    "Tboil_Unit_Raw",
    "Tboil_K",
    "Tfus_Value_Raw",
    "Tfus_Unit_Raw",
    "Tfus_K",
    "Chemeo_Source_URL",
    "Chemeo_Status",
    "Chemeo_Selected_Name",
    "Chemeo_Selected_SMILES",
    "Chemeo_Selected_Formula",
    "Chemeo_Candidate_Count",
]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_file: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("chemeo_scraper")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def read_input_csvs(input_dir: str, input_files: list[str], logger) -> Optional[pd.DataFrame]:
    """
    Read multiple CSV files and concatenate them into one DataFrame.
    All files are expected to share the same column schema.
    """
    encodings_to_try = ["utf-8", "cp949", "latin1"]
    frames = []

    for file_name in input_files:
        file_path = Path(input_dir) / file_name
        if not file_path.exists():
            logger.error(f"Input file not found: {file_path}")
            return None

        loaded = False
        for enc in encodings_to_try:
            try:
                logger.info(f"Trying to read: {file_path} | encoding={enc}")
                df_part = pd.read_csv(file_path, low_memory=False, encoding=enc)
                df_part["__source_file__"] = file_name
                frames.append(df_part)
                logger.info(f"Loaded: {file_path} | rows={len(df_part)}")
                loaded = True
                break
            except UnicodeDecodeError:
                logger.warning(f"UnicodeDecodeError for {file_path} with encoding={enc}")
            except Exception as e:
                logger.error(f"Failed reading {file_path} with encoding={enc}: {e}")
                return None

        if not loaded:
            logger.error(f"Could not read input file with any encoding: {file_path}")
            return None

    if not frames:
        logger.error("No input files were loaded.")
        return None

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Concatenated input rows: {len(df)} from {len(frames)} file(s)")
    return df


def detect_smiles_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    for col in ["smiles", "SMILES", "isosmiles"]:
        if col in df.columns:
            logger.info(f"Detected SMILES column: {col}")
            return col
    logger.error(f"No SMILES-related column found. Available columns: {df.columns.tolist()}")
    return None


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_partial_output(df: pd.DataFrame, output_path: Path, logger: logging.Logger) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Partial save completed: {output_path}")


def load_resume_partial(partial_output_file: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not partial_output_file.exists():
        return None
    try:
        df_partial = pd.read_csv(partial_output_file, low_memory=False)
        logger.info(f"Loaded partial resume file: {partial_output_file}")
        return df_partial
    except Exception as e:
        logger.warning(f"Failed to load partial resume file: {e}")
        return None
