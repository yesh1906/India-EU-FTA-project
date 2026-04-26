from pathlib import Path
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 40)

# 1. PROJECT PATHS
# ----------------------------
# This Python file lives inside the "source" folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Path where the cleaned datasets will be saved
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# Create the clean folder automatically if it does not exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# 2. RAW FILE MAP
# ----------------------------
# These are the six raw files downloaded earlier.
# If you ever rename the raw files, update the names here.
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
def load_csv(path: Path) -> pd.DataFrame:
    #Load one raw CSV file into a pandas DataFrame.
    return pd.read_csv(
        path,
        low_memory=False,
        index_col=False,
        usecols=range(47)  # We use usecols=range(47) as comtrade export has an extra trailing empty field at the end of each row.
                           # Restricting to the first 47 columns avoids misalignment.
    )


def keep_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the variables that matter for the analysis.
    Not all 47 raw columns are needed.
    For now, only these 7 columns are relevant:
    """
    cols_to_keep = [
        "refYear",
        "reporterDesc",
        "partnerDesc",
        "flowDesc",
        "cmdCode",
        "cmdDesc",
        "primaryValue",
    ]
    return df[cols_to_keep].copy()


def standardize_partner_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cleaner and more consistent reporter/partner labels.
    This helps later when we group or plot the data.
    """
    df = df.copy()

    partner_map = {
        "USA": "United States",
        "China": "China",
        "India": "India",
        "European Union": "European Union",
    }

    reporter_map = {
        "India": "India",
        "European Union": "European Union",
    }

    df["partner"] = df["partnerDesc"].replace(partner_map)
    df["reporter"] = df["reporterDesc"].replace(reporter_map)
    return df


def standardize_reporter_partner_india_perspective(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    For India-reporter files:
        reporter stays India
        partner stays China / United States
    For EU-reporter files:
        reporter becomes India, partner becomes European Union
    This gives us one uniform bilateral structure from India's perspective across all cleaned datasets.
    """
    df = df.copy()

    if "eu" in dataset_name.lower():
        df["reporter"] = "India"
        df["partner"] = "European Union"
    return df

def standardize_flow_india_perspective(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Standardize all trade flows so they are written from India's perspective.
    For India-reporter files:
        Import = India's import
        Export = India's export
    For EU-reporter files:
        EU Export to India = India Import from EU
        EU Import from India = India Export to EU
    """
    df = df.copy()
    if "eu" in dataset_name.lower():
        flow_map = {
            "Export": "Import",
            "Import": "Export",
        }
        df["flow_india"] = df["flowDesc"].replace(flow_map)
        df["source_side"] = "EU_reporter_reversed"
    else:
        df["flow_india"] = df["flowDesc"]
        df["source_side"] = "India_reporter"
    return df


def add_partner_group(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Add one clean partner-group label for analysis.
    This is useful because later we want one simple column saying:
    - China
    - United States
    - European Union
    """
    df = df.copy()

    if "china" in dataset_name.lower():
        df["partner_group"] = "China"
    elif "usa" in dataset_name.lower():
        df["partner_group"] = "United States"
    elif "eu" in dataset_name.lower():
        df["partner_group"] = "European Union"
    else:
        df["partner_group"] = "Unknown"
    return df


def filter_common_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 2020-2024 for now.
    As:
    - China and US files stop at 2024
    - EU files include 2025
    To keep all three partners comparable, we use the shared common years first.
    """
    return df[df["refYear"].between(2020, 2024)].copy()


def clean_one_dataset(path: Path, dataset_name: str, dataset_type: str) -> pd.DataFrame:
#Full cleaning pipeline for one dataset.

    df = load_csv(path)
    df = keep_useful_columns(df)
    df = standardize_partner_names(df)
    df = standardize_flow_india_perspective(df, dataset_name)
    df = standardize_reporter_partner_india_perspective(df, dataset_name)
    df = add_partner_group(df, dataset_name)
    df = filter_common_years(df)

    # Add a label showing whether this is totals data or hs2 data
    df["dataset_type"] = dataset_type

    # Convert commodity code to string for consistency
    df["cmdCode"] = df["cmdCode"].astype(str)

    # Rename primaryValue into a cleaner and more readable name
    df = df.rename(columns={"primaryValue": "trade_value_usd"})

    # Reorder columns into a cleaner structure
    ordered_cols = [
        "refYear",
        "reporter",
        "partner",
        "partner_group",
        "flow_india",
        "cmdCode",
        "cmdDesc",
        "trade_value_usd",
        "dataset_type",
        "source_side",
    ]
    return df[ordered_cols].copy()

def print_clean_summary(name: str, df: pd.DataFrame) -> None:
    """
    Print a short summary of each cleaned dataset
    so it can be verified that the cleaning worked.
    """
    print("\n" + "=" * 80)
    print(f"CLEANED DATASET: {name}")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Years:", sorted(df["refYear"].unique().tolist()))
    print("Partner groups:", df["partner_group"].unique().tolist())
    print("India-perspective flows:", df["flow_india"].unique().tolist())

    preview_cols = [
        "refYear",
        "partner_group",
        "flow_india",
        "cmdCode",
        "cmdDesc",
        "trade_value_usd",
    ]
    print(df[preview_cols].head())


# 4. MAIN CLEANING WORKFLOW
# ----------------------------
def main() -> None:
    """
    Main workflow:
    - clean totals files
    - combine totals
    - clean hs2 files
    - combine hs2
    - save final outputs
    """
    print("\nStarting cleaning and harmonisation...\n")
    # --------------------------------------------------
    # A. CLEAN TOTALS FILES
    # --------------------------------------------------
    totals_frames = []
    totals_map = {
        "total_ind_china": FILES["total_ind_china"],
        "total_ind_usa": FILES["total_ind_usa"],
        "total_eu_india": FILES["total_eu_india"],
    }

    for name, path in totals_map.items():
        cleaned = clean_one_dataset(path, name, dataset_type="totals")
        print_clean_summary(name, cleaned)
        totals_frames.append(cleaned)

    # Combine all cleaned totals into one DataFrame
    clean_totals = pd.concat(totals_frames, ignore_index=True)
    
    # --------------------------------------------------------
    # B. CLEAN HS2 FILES
    # --------------------------------------------------------
    hs2_frames = []
    hs2_map = {
        "hs2_ind_china": FILES["hs2_ind_china"],
        "hs2_ind_usa": FILES["hs2_ind_usa"],
        "hs2_eu_india": FILES["hs2_eu_india"],
    }

    for name, path in hs2_map.items():
        cleaned = clean_one_dataset(path, name, dataset_type="hs2")
        print_clean_summary(name, cleaned)
        hs2_frames.append(cleaned)

    # Combine all cleaned hs2 datasets into one DataFrame
    clean_hs2 = pd.concat(hs2_frames, ignore_index=True)

    # --------------------------------------------------------
    # C. SAVE OUTPUT FILES
    # --------------------------------------------------------
    totals_path = CLEAN_DIR / "totals_clean.csv"
    hs2_path = CLEAN_DIR / "hs2_clean.csv"

    clean_totals.to_csv(totals_path, index=False)
    clean_hs2.to_csv(hs2_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL CLEAN DATASETS SAVED")
    print("=" * 80)
    print(f"totals_clean.csv -> {totals_path}")
    print(f"hs2_clean.csv    -> {hs2_path}")

    print("\nCombined totals shape:", clean_totals.shape)
    print("Combined HS2 shape:", clean_hs2.shape)

    print("\nTotals preview:")
    print(clean_totals.head())

    print("\nHS2 preview:")
    print(clean_hs2.head())


if __name__ == "__main__":
    main()