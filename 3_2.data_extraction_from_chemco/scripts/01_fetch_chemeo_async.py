from pathlib import Path
import sys
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_config, setup_logger
from src.path_utils import resolve_project_paths
from src.chemeo_scraper import run_scraper


def main():
    cfg = load_config("config.yaml")
    paths = resolve_project_paths(cfg)

    logger = setup_logger(
        log_file=str(paths["log_file"]),
        log_level=cfg["logging"].get("log_level", "INFO")
    )

    logger.info("Starting Chemeo scraper...")
    logger.info("Directory convention: input/ output/ logs/ scripts/ src/")
    logger.info("Input files:")
    for p in paths["input_files"]:
        logger.info(f"  - {p}")
    logger.info(f"Output file : {paths['output_file']}")
    logger.info(f"Summary file: {paths['summary_file']}")
    logger.info(f"Log file    : {paths['log_file']}")

    asyncio.run(run_scraper(cfg, logger))


if __name__ == "__main__":
    main()