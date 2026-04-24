import asyncio
import logging
import random
import urllib.parse
from typing import Optional
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from src.chemeo_parser import (
    choose_best_cp_row,
    choose_first_exact_title_row,
    collect_search_candidates,
    convert_cp_to_j_per_mol_k,
    convert_temperature_to_k,
    extract_dl_mapping,
    extract_page_name,
    parse_numeric_value,
    parse_property_rows,
    score_candidate,
)
from src.io_utils import (
    detect_smiles_column,
    ensure_output_columns,
    load_resume_partial,
    read_input_csvs,
    save_partial_output,
)
from src.path_utils import build_partial_output_path, resolve_project_paths


async def request_with_retry(client: httpx.AsyncClient, url: str, cfg: dict, logger: logging.Logger, context: str = ""):
    retry_cfg = cfg["retry"]
    website_cfg = cfg["website"]
    max_retries = retry_cfg["max_retries"]
    base_delay = retry_cfg["retry_base_delay_sec"]
    retry_status_codes = set(retry_cfg["retry_status_codes"])

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.get(url, follow_redirects=True, timeout=website_cfg["request_timeout"])
            if response.status_code in retry_status_codes:
                logger.warning(f"{context} HTTP {response.status_code} on attempt {attempt}/{max_retries}: {url}")
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * attempt + random.uniform(0, 0.5))
                    continue
            response.raise_for_status()
            return response
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            logger.warning(f"{context} Transient network error on attempt {attempt}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(base_delay * attempt + random.uniform(0, 0.5))
                continue
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"{context} Non-success HTTP response after retries: {e.response.status_code} | {url}")
            return None
        except Exception as e:
            logger.error(f"{context} Unexpected request error: {e} | {url}")
            return None
    return None


async def fetch_chemeo_data(client: httpx.AsyncClient, smiles, index: int, total: int, cfg: dict, logger: logging.Logger, input_formula: Optional[str] = None):
    parsing_cfg = cfg["parsing"]
    website_cfg = cfg["website"]

    if pd.isna(smiles) or str(smiles).strip().lower() == "nan":
        logger.warning(f"[{index+1}/{total}] Empty or NaN SMILES encountered.")
        return (index, None, None, None, None, None, None, None, None, None, None, None, None, "EMPTY_SMILES", None, None, None, 0)

    smiles_str = str(smiles).strip()
    search_url = f"{website_cfg['base_url']}/search?q={urllib.parse.quote(smiles_str)}"
    target_temperature = float(parsing_cfg["target_temperature"])
    temperature_tolerance = float(parsing_cfg["temperature_tolerance"])
    context = f"[{index+1}/{total}] {smiles_str}"

    try:
        search_resp = await request_with_retry(client, search_url, cfg, logger, context=context + " | search")
        if search_resp is None:
            return (index, None, None, None, None, None, None, None, None, None, None, None, None, "SEARCH_REQUEST_FAILED", None, None, None, 0)

        search_soup = BeautifulSoup(search_resp.text, "html.parser")
        candidates = collect_search_candidates(search_soup, website_cfg["base_url"])
        if not candidates:
            return (index, None, None, None, None, None, None, None, None, None, None, None, search_url, "SEARCH_NO_CANDIDATE", None, None, None, 0)

        best_candidate = None
        best_candidate_soup = None
        best_score = -10**9
        max_candidates_to_check = int(parsing_cfg.get("max_candidates_to_check", 5))

        for cand in candidates[:max_candidates_to_check]:
            detail_resp = await request_with_retry(client, cand["url"], cfg, logger, context=context + f" | candidate {cand['url']}")
            if detail_resp is None:
                continue
            cand_soup = BeautifulSoup(detail_resp.text, "html.parser")
            dl_map = extract_dl_mapping(cand_soup)
            cand_smiles = dl_map.get("SMILES")
            cand_formula = dl_map.get("Formula")
            cand_name_page = extract_page_name(cand_soup) or cand.get("name")
            score = score_candidate(smiles_str, input_formula, cand_smiles, cand_formula, cand_name_page)
            logger.info(f"{context} Candidate score={score} | name={cand_name_page} | smiles={cand_smiles} | formula={cand_formula} | url={cand['url']}")
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_candidate_soup = cand_soup

        if best_candidate is None or best_candidate_soup is None:
            return (index, None, None, None, None, None, None, None, None, None, None, None, search_url, "NO_VALID_CANDIDATE_PAGE", None, None, None, len(candidates))

        source_url = best_candidate["url"]
        rows_data = parse_property_rows(best_candidate_soup)
        if not rows_data:
            return (index, None, None, None, None, None, None, None, None, None, None, None, source_url, "DETAIL_PAGE_NO_PROPERTY_ROWS", None, None, None, len(candidates))

        cp_value_raw = cp_unit_raw = cp_value_j_per_mol_k = cp_phase = cp_temperature_k = None
        tboil_value_raw = tboil_unit_raw = tboil_k = None
        tfus_value_raw = tfus_unit_raw = tfus_k = None

        best_cp = choose_best_cp_row(rows_data, target_temperature, temperature_tolerance)
        if best_cp is not None:
            cp_value_raw = best_cp["value_raw"]
            cp_unit_raw = best_cp["unit_raw"]
            cp_phase = best_cp["phase"]
            cp_temperature_k = best_cp["temp_k"]
            cp_std, cp_std_unit = convert_cp_to_j_per_mol_k(best_cp["value_numeric"], cp_unit_raw)
            cp_value_j_per_mol_k = cp_std if cp_std_unit == "J/mol·K" else None

        row_tboil = choose_first_exact_title_row(rows_data, "Normal Boiling Point Temperature")
        if row_tboil is not None:
            tboil_value_raw = row_tboil["value_text"]
            tboil_unit_raw = row_tboil["unit_text"]
            tboil_numeric = parse_numeric_value(row_tboil["value_text"])
            if tboil_numeric is not None:
                tboil_std, tboil_std_unit = convert_temperature_to_k(tboil_numeric, tboil_unit_raw)
                tboil_k = tboil_std if tboil_std_unit == "K" else None

        row_tfus = choose_first_exact_title_row(rows_data, "Normal melting (fusion) point")
        if row_tfus is not None:
            tfus_value_raw = row_tfus["value_text"]
            tfus_unit_raw = row_tfus["unit_text"]
            tfus_numeric = parse_numeric_value(row_tfus["value_text"])
            if tfus_numeric is not None:
                tfus_std, tfus_std_unit = convert_temperature_to_k(tfus_numeric, tfus_unit_raw)
                tfus_k = tfus_std if tfus_std_unit == "K" else None

        selected_dl = extract_dl_mapping(best_candidate_soup)
        selected_name = extract_page_name(best_candidate_soup) or best_candidate.get("name")
        selected_smiles = selected_dl.get("SMILES")
        selected_formula = selected_dl.get("Formula")
        candidate_count = len(candidates)
        status = "OK"
        if cp_value_j_per_mol_k is None and tboil_k is None and tfus_k is None:
            status = "DETAIL_FOUND_BUT_NO_TARGET_PROPERTY"

        return (
            index,
            cp_value_raw, cp_unit_raw, cp_value_j_per_mol_k, cp_phase, cp_temperature_k,
            tboil_value_raw, tboil_unit_raw, tboil_k,
            tfus_value_raw, tfus_unit_raw, tfus_k,
            source_url, status, selected_name, selected_smiles, selected_formula, candidate_count,
        )
    except Exception as e:
        logger.error(f"{context} Unexpected parsing failure: {e}")
        return (index, None, None, None, None, None, None, None, None, None, None, None, None, "EXCEPTION", None, None, None, 0)


async def run_scraper(cfg: dict, logger: logging.Logger) -> None:
    paths = resolve_project_paths(cfg)

    input_dir = paths["input_dir"]
    input_files = cfg["files"]["input_files"]
    output_file = paths["output_file"]
    partial_output_file = paths["partial_output_file"]
    summary_file = paths["summary_file"]

    logger.info("Loading input CSV...")
    df = read_input_csvs(
        input_dir=input_dir,
        input_files=input_files,
        logger=logger
    )

    if df is None:
        logger.error("Stopping because input CSV could not be loaded.")
        return

    smiles_col = detect_smiles_column(df, logger)
    if smiles_col is None:
        logger.error("Stopping because no SMILES column was found.")
        return

    df = ensure_output_columns(df)
    df_partial = load_resume_partial(partial_output_file, logger)
    if df_partial is not None and len(df_partial) == len(df):
        for col in df_partial.columns:
            df[col] = df_partial[col]

    total = len(df)
    semaphore = asyncio.Semaphore(cfg["concurrency"]["concurrency_limit"])
    limits = httpx.Limits(
        max_keepalive_connections=cfg["concurrency"]["max_keepalive_connections"],
        max_connections=cfg["concurrency"]["max_connections"],
    )

    async with httpx.AsyncClient(
        limits=limits,
        verify=cfg["website"]["verify_ssl"],
        headers={"User-Agent": cfg["website"]["user_agent"]},
    ) as client:
        completed_count = 0

        async def sem_task(i: int, smiles_value):
            nonlocal completed_count
            async with semaphore:
                await asyncio.sleep(
                    random.uniform(
                        cfg["saving"]["sleep_min_sec"],
                        cfg["saving"]["sleep_max_sec"]
                    )
                )
                input_formula = df.at[i, "Molecular_Formula"] if "Molecular_Formula" in df.columns else None
                result = await fetch_chemeo_data(client, smiles_value, i, total, cfg, logger, input_formula=input_formula)
                columns = [
                    "Cp_Value_Raw", "Cp_Unit_Raw", "Cp_Value_J_per_mol_K", "Cp_Phase", "Cp_Temperature_K",
                    "Tboil_Value_Raw", "Tboil_Unit_Raw", "Tboil_K",
                    "Tfus_Value_Raw", "Tfus_Unit_Raw", "Tfus_K",
                    "Chemeo_Source_URL", "Chemeo_Status", "Chemeo_Selected_Name",
                    "Chemeo_Selected_SMILES", "Chemeo_Selected_Formula", "Chemeo_Candidate_Count",
                ]
                for offset, col in enumerate(columns, start=1):
                    df.at[i, col] = result[offset]
                completed_count += 1
                if cfg["saving"]["partial_save_every"] > 0 and completed_count % cfg["saving"]["partial_save_every"] == 0:
                    save_partial_output(df, partial_output_file, logger)

        tasks = []
        skipped_count = 0
        for i in range(total):
            already_done = pd.notna(df.at[i, "Chemeo_Status"]) and str(df.at[i, "Chemeo_Status"]).strip() != ""
            if already_done:
                skipped_count += 1
                continue
            tasks.append(sem_task(i, df.at[i, smiles_col]))

        logger.info(f"Resume mode: skipped {skipped_count} completed row(s), running {len(tasks)} row(s).")
        if tasks:
            await asyncio.gather(*tasks)
        else:
            logger.info("No remaining rows to process.")

    df.to_csv(output_file, index=False, encoding=cfg["saving"]["output_encoding"])
    save_partial_output(df, partial_output_file, logger)
    logger.info(f"Final output saved to: {output_file}")

    matched_cp = df["Cp_Value_J_per_mol_K"].notna().sum()
    matched_tb = df["Tboil_K"].notna().sum()
    matched_tf = df["Tfus_K"].notna().sum()
    matched_url = df["Chemeo_Source_URL"].notna().sum()
    status_counts = df["Chemeo_Status"].value_counts(dropna=False).to_dict()

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Chemeo scraping summary\n")
        f.write("======================\n")
        f.write(f"Input rows                : {len(df)}\n")
        f.write(f"Cp standardized rows      : {matched_cp}\n")
        f.write(f"Tboil standardized rows   : {matched_tb}\n")
        f.write(f"Tfus standardized rows    : {matched_tf}\n")
        f.write(f"URL captured rows         : {matched_url}\n")
        f.write(f"Target temperature        : {cfg['parsing']['target_temperature']}\n")
        f.write(f"Tolerance (K)             : {cfg['parsing']['temperature_tolerance']}\n")
        f.write(f"Final output file         : {output_file}\n")
        f.write(f"Partial output file       : {partial_output_file}\n")
        f.write(f"Log file                  : {paths['log_file']}\n")
        f.write("Status counts:\n")
        for k, v in status_counts.items():
            f.write(f"  {k}: {v}\n")

    logger.info(f"Summary saved to: {summary_file}")
