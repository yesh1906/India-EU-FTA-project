from pathlib import Path
import pandas as pd

# 1. PROJECT PATHS
# ----------------------------
# This file lives in: source/01_comtrade_data.py
# So we go one level up to reach the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# 2. RAW FILE MAP
# ----------------------------
# Adjust these names only if upon installation the file names differ.
FILES = {
    "total_ind_china": RAW_DIR / "total_ind_chn.csv",
    "total_ind_usa": RAW_DIR / "total_ind_usa.csv",
    "total_eu_india": RAW_DIR / "total_eu_ind.csv",
    "hs2_ind_china": RAW_DIR / "h2_ind_chn.csv",
    "hs2_ind_usa": RAW_DIR / "h2_ind_usa.csv",
    "hs2_eu_india": RAW_DIR / "h2_eu_ind.csv",
}


# 3. HELPER FUNCTIONS
# ----------------------------
def check_files_exist(file_map: dict[str, Path]) -> None:
    """
    Check that every expected raw file exists before we try to load it.
    This prevents confusing pandas later.
    """
    print("\nChecking raw files...\n")
    missing = []

    for label, path in file_map.items():
        if path.exists():
            print(f"[OK] {label}: {path.name}")
        else:
            print(f"[MISSING] {label}: {path}")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "\nSome raw files are missing. Fix the filenames or move the files into data/raw/ first."
        )


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load one CSV file into a DataFrame.
    Comtrade exports appear to have a trailing empty field at the end of each row,
    so we explicitly ignore anything beyond the first 47 columns.
    """
    df = pd.read_csv(
        path,
        low_memory=False,
        index_col=False,
        usecols=range(47)
    )
    return df

def inspect_raw_layout(path: Path) -> None:
    """
    Print the first few raw lines of the file so we can see
    how the CSV is actually structured on disk. Later, during cleaning,
    """
    print("\n" + "=" * 80)
    print(f"RAW FILE INSPECTION: {path.name}")
    print("=" * 80)

    with open(path, "r", encoding="utf-8-sig") as f:
        for i in range(5):
            line = f.readline().rstrip("\n")
            print(f"Line {i+1}: {line[:500]}")

def basic_summary(name: str, df: pd.DataFrame) -> None:
    """
    Print a basic summary for one dataset:
    - shape
    - columns
    - key unique values if they exist
    """
    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("\nData types (first 10 columns):")
    print(df.dtypes.head(10))

    print("\nPreview of selected columns:")
    preview_cols = ["refYear", "reporterDesc", "partnerDesc", "flowDesc", "cmdCode", "cmdDesc", "primaryValue"]
    existing_preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[existing_preview_cols].head())

    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    # Key fields we expect in Comtrade-style extracts
    key_fields = [
        "refYear",
        "reporterDesc",
        "partnerDesc",
        "flowDesc",
        "cmdCode",
        "cmdDesc",
        "primaryValue",
    ]

    print("\nKey field checks:")
    for field in key_fields:
        print(f"  - {field}: {'FOUND' if field in df.columns else 'MISSING'}")

    # Print unique values for important fields if they exist
    print("\nSample unique values:")
    for field in ["refYear", "reporterDesc", "partnerDesc", "flowDesc", "cmdCode", "cmdDesc"]:
        if field in df.columns:
            values = df[field].dropna().unique().tolist()
            preview = values[:10]
            print(f"  - {field}: {preview}")

    # Missing-value snapshot for key fields
    print("\nMissing values in key fields:")
    for field in key_fields:
        if field in df.columns:
            missing_count = df[field].isna().sum()
            print(f"  - {field}: {missing_count}")

    print("\nFirst 5 rows:")
    print(df.head())


def compare_structures(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Compare the column structure across all datasets.
    This tells us whether files are aligned or if some need special handling.
    """
    print("\n" + "=" * 80)
    print("COLUMN STRUCTURE COMPARISON")
    print("=" * 80)

    reference_name = list(datasets.keys())[0]
    reference_cols = set(datasets[reference_name].columns)

    print(f"Reference dataset: {reference_name}")

    for name, df in datasets.items():
        cols = set(df.columns)
        missing_vs_ref = reference_cols - cols
        extra_vs_ref = cols - reference_cols

        print(f"\n{name}:")
        print(f"  Total columns: {len(cols)}")
        print(f"  Missing vs reference: {sorted(missing_vs_ref) if missing_vs_ref else 'None'}")
        print(f"  Extra vs reference: {sorted(extra_vs_ref) if extra_vs_ref else 'None'}")


def india_perspective_notes(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Print a reminder about which files are already from India's perspective
    and which one(s) will need to be flipped later.
    """
    print("\n" + "=" * 80)
    print("INDIA-PERSPECTIVE CHECK")
    print("=" * 80)

    for name, df in datasets.items():
        reporter = df["reporterDesc"].dropna().unique().tolist() if "reporterDesc" in df.columns else []
        partner = df["partnerDesc"].dropna().unique().tolist() if "partnerDesc" in df.columns else []

        print(f"\n{name}:")
        print(f"  Reporter(s): {reporter}")
        print(f"  Partner(s): {partner}")

        if "eu" in name.lower():
            print("  NOTE: This dataset is likely from the EU reporter side and will need flow reversal later.")
        else:
            print("  NOTE: This dataset is likely already from India's reporter side.")


# 4. MAIN SCRIPT
# ----------------------------
def main() -> None:
    print("\nStarting raw comtrade data inspection...")

    # Step 1: confirm files exist
    check_files_exist(FILES)

    # Step 1b: inspect one raw file directly
    inspect_raw_layout(FILES["total_ind_china"])

    # Step 2: load all files
    datasets = {}
    for name, path in FILES.items():
        datasets[name] = load_csv(path)

    # Step 3: print a basic summary for each
    for name, df in datasets.items():
        basic_summary(name, df)

    # Step 4: compare column structures across datasets
    compare_structures(datasets)

    # Step 5: print India-perspective notes
    india_perspective_notes(datasets)

    print("\nDone. Next step: write the cleaning/harmonisation script.")


if __name__ == "__main__":
    main()