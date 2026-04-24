import re
from typing import Optional

from bs4 import BeautifulSoup


def normalize_unit_text(text: str) -> str:
    if text is None:
        return ""
    t = text.strip().lower()
    t = t.replace("×", "*").replace("·", "*").replace("−", "-").replace(" ", "")
    return t


def normalize_smiles_text(smiles: Optional[str]) -> Optional[str]:
    if smiles is None:
        return None
    return smiles.strip().replace(" ", "")


def normalized_formula(formula: Optional[str]) -> Optional[str]:
    if formula is None:
        return None
    return formula.strip().replace(" ", "")


def has_isotopic_formula(formula: Optional[str]) -> bool:
    if not formula:
        return False
    return ("D" in formula.strip()) or ("T" in formula.strip())


def is_probably_isotopic_name(name: Optional[str]) -> bool:
    if not name:
        return False
    lower = name.lower()
    isotopic_markers = [
        "[2h", "[3h", "[13c", "[14c", "[15n", "[18o", "[17o", "[34s",
        "deuter", "tritium", "triti", "d-", "d ", "d6", "d5", "d4", "d3", "d2",
        "t2", "toluene-d", "ethanol-d", "benzene-d", "water-t",
    ]
    return any(marker in lower for marker in isotopic_markers)


def score_candidate(input_smiles, input_formula, candidate_page_smiles, candidate_page_formula, candidate_name) -> int:
    score = 0
    in_smi = normalize_smiles_text(input_smiles)
    cand_smi = normalize_smiles_text(candidate_page_smiles)
    in_formula = normalized_formula(input_formula)
    cand_formula = normalized_formula(candidate_page_formula)

    if cand_smi == in_smi:
        score += 100
    if in_formula and cand_formula:
        if in_formula == cand_formula:
            score += 35
        else:
            score -= 35
    if candidate_name and is_probably_isotopic_name(candidate_name):
        score -= 40
    else:
        score += 5
    if has_isotopic_formula(cand_formula):
        score -= 60
    return score


def parse_numeric_value(text: str) -> Optional[float]:
    if not text:
        return None
    interval = re.findall(r"[-+]?\d*\.?\d+", text)
    if not interval:
        return None
    try:
        nums = [float(x) for x in interval]
    except ValueError:
        return None
    if text.strip().startswith("[") and len(nums) >= 2:
        return sum(nums[:2]) / 2.0
    return nums[0]


def convert_cp_to_j_per_mol_k(value: float, unit_text: Optional[str]):
    u = normalize_unit_text(unit_text)
    mapping = {
        "j/mol*k": value,
        "j/mol·k": value,
        "j/molk": value,
        "j*mol^-1*k^-1": value,
        "jmol^-1k^-1": value,
        "kj/mol*k": value * 1000.0,
        "kj/mol·k": value * 1000.0,
        "kj/molk": value * 1000.0,
        "kj*mol^-1*k^-1": value * 1000.0,
        "kjmol^-1k^-1": value * 1000.0,
        "cal/mol*k": value * 4.184,
        "cal/mol·k": value * 4.184,
        "cal/molk": value * 4.184,
        "cal*mol^-1*k^-1": value * 4.184,
        "calmol^-1k^-1": value * 4.184,
        "kcal/mol*k": value * 4184.0,
        "kcal/mol·k": value * 4184.0,
        "kcal/molk": value * 4184.0,
        "kcal*mol^-1*k^-1": value * 4184.0,
        "kcalmol^-1k^-1": value * 4184.0,
        "j/kmol*k": value / 1000.0,
        "j/kmol·k": value / 1000.0,
        "jkmol^-1k^-1": value / 1000.0,
        "kj/kmol*k": value,
        "kj/kmol·k": value,
        "kjkmol^-1k^-1": value,
        "cal/kmol*k": value * 4.184 / 1000.0,
        "cal/kmol·k": value * 4.184 / 1000.0,
        "calkmol^-1k^-1": value * 4.184 / 1000.0,
        "kcal/kmol*k": value * 4184.0 / 1000.0,
        "kcal/kmol·k": value * 4184.0 / 1000.0,
        "kcalkmol^-1k^-1": value * 4184.0 / 1000.0,
    }
    if u in mapping:
        return mapping[u], "J/mol·K"
    return None, "UNKNOWN"


def convert_temperature_to_k(value: float, unit_text: Optional[str]):
    u = normalize_unit_text(unit_text)
    if u in ["k", "kelvin"]:
        return value, "K"
    if u in ["c", "°c", "degc", "celsius"]:
        return value + 273.15, "K"
    if u in ["f", "°f", "degf", "fahrenheit"]:
        return (value - 32.0) * 5.0 / 9.0 + 273.15, "K"
    return None, "UNKNOWN"


def extract_dl_mapping(soup: BeautifulSoup) -> dict:
    data = {}
    root = soup.find(id="details-sidebar")
    dl = root.find("dl") if root else soup.find("dl")
    if not dl:
        return data
    current_key = None
    for child in dl.children:
        if getattr(child, "name", None) == "dt":
            current_key = child.get_text(" ", strip=True)
        elif getattr(child, "name", None) == "dd" and current_key is not None:
            data[current_key] = child.get_text(" ", strip=True)
            current_key = None
    return data


def extract_page_name(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(" ", strip=True)
        m = re.match(r"Chemical Properties of (.+?) \(CAS ", text)
        if m:
            return m.group(1).strip()
        return text.strip()
    if soup.title:
        text = soup.title.get_text(" ", strip=True)
        m = re.match(r"(.+?) \(CAS ", text)
        if m:
            return m.group(1).strip()
    return None


def collect_search_candidates(search_soup: BeautifulSoup, base_url: str) -> list[dict]:
    candidates = []
    seen = set()
    table = search_soup.find("table")
    if table:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            link = tr.find("a", href=re.compile(r"^/cid/"))
            if not link:
                continue
            href = link.get("href", "").strip()
            if not href or href in seen:
                continue
            if href.endswith((".pdf", ".xls", ".2dmol")):
                continue
            seen.add(href)
            name = tds[1].get_text(" ", strip=True) if len(tds) >= 2 else None
            cas = tds[2].get_text(" ", strip=True) if len(tds) >= 3 else None
            candidates.append({"href": href, "url": base_url + href, "name": name, "cas": cas})
    if not candidates:
        for a in search_soup.find_all("a", href=re.compile(r"^/cid/")):
            href = a.get("href", "").strip()
            if not href or href in seen:
                continue
            if href.endswith((".pdf", ".xls", ".2dmol")):
                continue
            seen.add(href)
            candidates.append({"href": href, "url": base_url + href, "name": a.get_text(" ", strip=True) or None, "cas": None})
    return candidates


def parse_property_rows(soup: BeautifulSoup) -> list[dict]:
    rows_data = []
    for tr in soup.find_all("tr"):
        cols = tr.find_all("td")
        if len(cols) < 4:
            continue
        first_span = cols[0].find("span")
        property_title = first_span.get("title", "").strip() if first_span else ""
        property_text = cols[0].get_text(" ", strip=True)
        value_text = cols[1].get_text(" ", strip=True)
        unit_text = cols[2].get_text(" ", strip=True)
        extra_text = None
        source_text = None
        if len(cols) == 4:
            source_text = cols[3].get_text(" ", strip=True)
        elif len(cols) >= 5:
            extra_text = cols[3].get_text(" ", strip=True)
            source_text = cols[4].get_text(" ", strip=True)
        rows_data.append({
            "property_title": property_title,
            "property_text": property_text,
            "value_text": value_text,
            "unit_text": unit_text,
            "extra_text": extra_text,
            "source_text": source_text,
            "ncols": len(cols),
        })
    return rows_data


def choose_best_cp_row(rows_data: list[dict], target_temperature: float, tolerance: float) -> Optional[dict]:
    cp_titles = {
        "Ideal gas heat capacity": "Gas",
        "Liquid phase heat capacity": "Liquid",
        "Solid phase heat capacity": "Solid",
    }
    phase_priority = {"Liquid": 0, "Gas": 1, "Solid": 2, "Unknown": 3}
    candidates = []
    for row in rows_data:
        title = row["property_title"]
        if title not in cp_titles:
            continue
        temp_text = row["extra_text"]
        if temp_text is None:
            continue
        temp_value = parse_numeric_value(temp_text)
        if temp_value is None or abs(temp_value - target_temperature) > tolerance:
            continue
        cp_value = parse_numeric_value(row["value_text"])
        if cp_value is None:
            continue
        phase = cp_titles[title]
        candidates.append({
            "phase": phase,
            "value_raw": row["value_text"],
            "unit_raw": row["unit_text"],
            "value_numeric": cp_value,
            "temp_k": temp_value,
            "row": row,
        })
    if not candidates:
        return None
    return min(candidates, key=lambda x: (abs(x["temp_k"] - target_temperature), phase_priority.get(x["phase"], 9)))


def choose_first_exact_title_row(rows_data: list[dict], target_title: str) -> Optional[dict]:
    for row in rows_data:
        if row["property_title"] == target_title:
            return row
    return None
