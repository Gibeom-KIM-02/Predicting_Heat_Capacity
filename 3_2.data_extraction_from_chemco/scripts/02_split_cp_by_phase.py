from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config("config.yaml")
    output_dir = Path(cfg["paths"]["output_dir"])
    main_csv = output_dir / cfg["files"]["output_file"]

    cp_only_file = output_dir / "PubChem_Cp_only.csv"
    cp_liquid_file = output_dir / "PubChem_Cp_liquid.csv"
    cp_solid_file = output_dir / "PubChem_Cp_solid.csv"
    cp_gas_file = output_dir / "PubChem_Cp_gas.csv"

    df = pd.read_csv(main_csv, low_memory=False)
    df_cp = df[df["Cp_Value_J_per_mol_K"].notna()].copy()
    df_cp.to_csv(cp_only_file, index=False, encoding="utf-8-sig")

    df_liquid = df_cp[df_cp["Cp_Phase"].astype(str).str.lower() == "liquid"].copy()
    df_solid = df_cp[df_cp["Cp_Phase"].astype(str).str.lower() == "solid"].copy()
    df_gas = df_cp[df_cp["Cp_Phase"].astype(str).str.lower() == "gas"].copy()

    df_liquid.to_csv(cp_liquid_file, index=False, encoding="utf-8-sig")
    df_solid.to_csv(cp_solid_file, index=False, encoding="utf-8-sig")
    df_gas.to_csv(cp_gas_file, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(f"  {cp_only_file} rows = {len(df_cp)}")
    print(f"  {cp_liquid_file} rows = {len(df_liquid)}")
    print(f"  {cp_solid_file} rows = {len(df_solid)}")
    print(f"  {cp_gas_file} rows = {len(df_gas)}")


if __name__ == "__main__":
    main()
