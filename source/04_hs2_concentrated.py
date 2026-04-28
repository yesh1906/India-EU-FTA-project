from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 50)

# ------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

HS2_PATH = CLEAN_DIR / "hs2_clean.csv"
# ------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------
def shorten_label(text: str, max_len: int = 55) -> str:
    """
    Shorten very long HS2 labels so horizontal bar charts remain readable.
    """
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
def pct_label(x: float) -> str:
    #Format a share value as a percentage string.
    return f"{x:.2%}"

def load_hs2(path: Path) -> pd.DataFrame:
    """
    Load the cleaned HS2 dataset.
    This stage is descriptive rather than regression-based.
    We are analysing HS2 sector structure, concentration, and change over time.
    No causal or econometric model is estimated here yet.
    """
    return pd.read_csv(path)
# ------------------------------------------------------
# 3. CORE SUMMARY TABLES
# ------------------------------------------------------
def build_sector_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the core HS2 summary:
    year x partner x flow x sector.
    This is the main long-format dataset used for sector-level analysis.
    """
    out = (
        df.groupby(
            ["refYear", "partner_group", "flow_india", "cmdCode", "cmdDesc"],
            as_index=False
        )["trade_value_usd"]
        .sum()
        .sort_values(
            ["partner_group", "flow_india", "refYear", "trade_value_usd"],
            ascending=[True, True, True, False]
        )
    )
    return out

def add_sector_shares(sector_year_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Within each year-partner-flow bucket, compute the sector's share
    of total trade value.
    """
    df = sector_year_summary.copy()

    totals = (
        df.groupby(["refYear", "partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .rename(columns={"trade_value_usd": "partner_flow_total_usd"})
    )

    df = df.merge(
        totals,
        on=["refYear", "partner_group", "flow_india"],
        how="left"
    )

    df["sector_share"] = df["trade_value_usd"] / df["partner_flow_total_usd"]
    return df

def build_cumulative_sector_summary(df_with_shares: pd.DataFrame) -> pd.DataFrame:
    #Build a cumulative 2020–2024 sector summary for each partner and flow.
    out = (
        df_with_shares.groupby(
            ["partner_group", "flow_india", "cmdCode", "cmdDesc"],
            as_index=False
        )["trade_value_usd"]
        .sum()
    )

    totals = (
        out.groupby(["partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .rename(columns={"trade_value_usd": "partner_flow_total_usd"})
    )

    out = out.merge(totals, on=["partner_group", "flow_india"], how="left")
    out["sample_share"] = out["trade_value_usd"] / out["partner_flow_total_usd"]

    out = out.sort_values(
        ["partner_group", "flow_india", "trade_value_usd"],
        ascending=[True, True, False]
    )

    return out

def build_hhi_yearly(df_with_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly HHI (Herfindahl-Hirschman Index) for partner-flow concentration.
    HHI = sum of squared sector shares
    Interpretation:
    - higher HHI = more concentrated trade structure
    - lower HHI = more diversified trade structure
    """
    out = (
        df_with_shares.groupby(["refYear", "partner_group", "flow_india"], as_index=False)
        .agg(hhi=("sector_share", lambda s: (s ** 2).sum()))
        .sort_values(["flow_india", "partner_group", "refYear"])
    )
    return out

def build_share_change_2020_2024(df_with_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Compare sector shares in 2020 versus 2024.
    This helps identify rising and declining sectors.
    """
    df = df_with_shares[df_with_shares["refYear"].isin([2020, 2024])].copy()

    pivot = (
        df.pivot_table(
            index=["partner_group", "flow_india", "cmdCode", "cmdDesc"],
            columns="refYear",
            values="sector_share",
            aggfunc="sum"
        )
        .reset_index()
    )

    if 2020 not in pivot.columns:
        pivot[2020] = 0
    if 2024 not in pivot.columns:
        pivot[2024] = 0

    pivot = pivot.rename(columns={2020: "share_2020", 2024: "share_2024"})
    pivot["share_change"] = pivot["share_2024"] - pivot["share_2020"]

    return pivot.sort_values(
        ["partner_group", "flow_india", "share_change"],
        ascending=[True, True, False]
    )

def build_import_exposure_candidates(share_change: pd.DataFrame) -> pd.DataFrame:
    """
    Identify candidate import-exposure sectors.
    Simple descriptive logic:
    - focus only on imports, since comparing from 03 scripts suggests export concentration isn't a major concern
    - sectors with relatively high 2024 share
    - sectors whose share increased over time
    """
    df = share_change.copy()
    df = df[df["flow_india"] == "Import"].copy()

    df["exposure_score"] = df["share_2024"] + df["share_change"]
    df = df[df["share_2024"] > 0].copy()

    return df.sort_values(["partner_group", "exposure_score"], ascending=[True, False])

# ------------------------------------------------------
# 4. SAVE TABLES
# ------------------------------------------------------
def save_tables(
    sector_year_summary: pd.DataFrame,
    cumulative_sector_summary: pd.DataFrame,
    hhi_yearly: pd.DataFrame,
    share_change: pd.DataFrame,
    import_exposure_candidates: pd.DataFrame,
) -> None:
#Save the main HS2 analytical tables.
    sector_year_summary.to_csv(TABLES_DIR / "hs2_sector_year_summary.csv", index=False)
    cumulative_sector_summary.to_csv(TABLES_DIR / "hs2_cumulative_sector_summary.csv", index=False)
    hhi_yearly.to_csv(TABLES_DIR / "hs2_hhi_yearly.csv", index=False)
    share_change.to_csv(TABLES_DIR / "hs2_share_change_2020_2024.csv", index=False)
    import_exposure_candidates.to_csv(TABLES_DIR / "hs2_import_exposure_candidates.csv", index=False)

# ------------------------------------------------------
# 5. FIGURES
# ------------------------------------------------------
def fig_import_concentration_hhi(hhi_yearly: pd.DataFrame) -> None:
    """
    Figure 1:
    Import concentration by partner over time using HHI.
    """
    plot_df = hhi_yearly[hhi_yearly["flow_india"] == "Import"].copy()

    plt.figure(figsize=(10, 6))

    for partner, group in plot_df.groupby("partner_group"):
        group = group.sort_values("refYear")
        plt.plot(group["refYear"], group["hhi"], marker="o", label=partner)

    plt.title("HS2 Import Concentration by Partner (HHI), 2020–2024")
    plt.xlabel("Year")
    plt.ylabel("HHI")
    plt.xticks(sorted(plot_df["refYear"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_hs2_import_concentration_hhi.png", dpi=300)
    plt.close()

def fig_export_concentration_hhi(hhi_yearly: pd.DataFrame) -> None:
    """
    Figure 2:
    Export concentration by partner over time using HHI.
    """
    plot_df = hhi_yearly[hhi_yearly["flow_india"] == "Export"].copy()

    plt.figure(figsize=(10, 6))

    for partner, group in plot_df.groupby("partner_group"):
        group = group.sort_values("refYear")
        plt.plot(group["refYear"], group["hhi"], marker="o", label=partner)

    plt.title("HS2 Export Concentration by Partner (HHI), 2020–2024")
    plt.xlabel("Year")
    plt.ylabel("HHI")
    plt.xticks(sorted(plot_df["refYear"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_hs2_export_concentration_hhi.png", dpi=300)
    plt.close()

def plot_top_sectors(
    df_with_shares: pd.DataFrame,
    partner: str,
    flow: str,
    year: int,
    title: str,
    xlabel: str,
    output_name: str,
    top_n: int = 10,
) -> None:
    """
    Generic horizontal bar chart for top sector shares.
    Used for import/export sector profiles by partner.
    """
    plot_df = df_with_shares[
        (df_with_shares["partner_group"] == partner) &
        (df_with_shares["flow_india"] == flow) &
        (df_with_shares["refYear"] == year)
    ].copy()

    plot_df = plot_df.sort_values("sector_share", ascending=False).head(top_n).copy()
    plot_df["cmdDesc_short"] = plot_df["cmdDesc"].apply(shorten_label)
    plot_df = plot_df.sort_values("sector_share", ascending=True)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(plot_df["cmdDesc_short"], plot_df["sector_share"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("HS2 sector")

    for bar, value in zip(bars, plot_df["sector_share"]):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            pct_label(value),
            va="center"
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_top_share_changes(
    share_change: pd.DataFrame,
    partner: str,
    flow: str,
    title: str,
    xlabel: str,
    output_name: str,
    top_n: int = 10,
    direction: str = "increase",
) -> None:
    """
    Plot the sectors with the biggest increase or decrease in share
    between 2020 and 2024.
    direction:
        'increase' -> largest positive share changes
        'decrease' -> largest negative share changes
    """
    plot_df = share_change[
        (share_change["partner_group"] == partner) &
        (share_change["flow_india"] == flow)
    ].copy()

    if direction == "increase":
        plot_df = plot_df.sort_values("share_change", ascending=False).head(top_n).copy()
    else:
        plot_df = plot_df.sort_values("share_change", ascending=True).head(top_n).copy()

    plot_df["cmdDesc_short"] = plot_df["cmdDesc"].apply(shorten_label)
    plot_df = plot_df.sort_values("share_change", ascending=True)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(plot_df["cmdDesc_short"], plot_df["share_change"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("HS2 sector")

    for bar, value in zip(bars, plot_df["share_change"]):
        xpos = bar.get_width()
        plt.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            pct_label(value),
            va="center"
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------
# 6. ANALYSIS NOTES
# ------------------------------------------------------
def print_analysis_notes(
    hhi_yearly: pd.DataFrame,
    import_exposure_candidates: pd.DataFrame,
    df_with_shares: pd.DataFrame,
    share_change: pd.DataFrame,
) -> None:
#Print analytical takeaways
    print("\n" + "=" * 80)
    print("HS2 ANALYSIS NOTES")
    print("=" * 80)

    latest_year = hhi_yearly["refYear"].max()

    latest_import_hhi = hhi_yearly[
        (hhi_yearly["refYear"] == latest_year) &
        (hhi_yearly["flow_india"] == "Import")
    ].sort_values("hhi", ascending=False)

    if not latest_import_hhi.empty:
        row = latest_import_hhi.iloc[0]
        print(
            f"Most concentrated import relationship in {latest_year}: "
            f"{row['partner_group']} (HHI = {row['hhi']:.4f})"
        )

    latest_export_hhi = hhi_yearly[
        (hhi_yearly["refYear"] == latest_year) &
        (hhi_yearly["flow_india"] == "Export")
    ].sort_values("hhi", ascending=False)

    if not latest_export_hhi.empty:
        row = latest_export_hhi.iloc[0]
        print(
            f"Most concentrated export relationship in {latest_year}: "
            f"{row['partner_group']} (HHI = {row['hhi']:.4f})"
        )

    print(f"\nTop import sectors in {latest_year}:")
    for partner in ["China", "European Union", "United States"]:
        subset = df_with_shares[
            (df_with_shares["partner_group"] == partner) &
            (df_with_shares["flow_india"] == "Import") &
            (df_with_shares["refYear"] == latest_year)
        ].sort_values("sector_share", ascending=False)

        if not subset.empty:
            row = subset.iloc[0]
            print(f"- {partner}: {row['cmdDesc']} ({row['sector_share']:.2%})")

    print(f"\nTop export sectors in {latest_year}:")
    for partner in ["European Union", "United States"]:
        subset = df_with_shares[
            (df_with_shares["partner_group"] == partner) &
            (df_with_shares["flow_india"] == "Export") &
            (df_with_shares["refYear"] == latest_year)
        ].sort_values("sector_share", ascending=False)

        if not subset.empty:
            row = subset.iloc[0]
            print(f"- {partner}: {row['cmdDesc']} ({row['sector_share']:.2%})")

    print("\nLargest sector-share increases (2020 to 2024):")
    for partner, flow in [("China", "Import"), ("European Union", "Export"), ("United States", "Export")]:
        subset = share_change[
            (share_change["partner_group"] == partner) &
            (share_change["flow_india"] == flow)
        ].sort_values("share_change", ascending=False)

        if not subset.empty:
            row = subset.iloc[0]
            print(
                f"- {partner} {flow}: {row['cmdDesc']} "
                f"(change = {row['share_change']:.2%}, 2024 share = {row['share_2024']:.2%})"
            )

    print("\nTop import-exposure candidates by partner:")
    for partner, group in import_exposure_candidates.groupby("partner_group"):
        top_group = group.head(3)
        print(f"\n{partner}:")
        print(top_group[["cmdCode", "cmdDesc", "share_2024", "share_change", "exposure_score"]])

# ------------------------------------------------------
# 7. MAIN WORKFLOW
# ------------------------------------------------------
def main() -> None:
    """
    Methodology for this stage:
    - use the cleaned HS2 dataset
    - build sector-level summaries by year, partner, and flow
    - compute sector shares within each bilateral relationship
    - measure concentration using HHI
    - compare 2020 and 2024 sector shares
    - identify candidate exposure sectors on the import side
    - compare import baskets across China, EU, and US
    - compare export baskets to EU and US
    - produce sector figures for later interpretation in the report
This stage is descriptive rather than regression-based.
No causal or econometric model is estimated here yet.
    """
    print("\nStarting HS2 concentration and exposure analysis...\n")
    hs2 = load_hs2(HS2_PATH)

    sector_year_summary = build_sector_year_summary(hs2)
    df_with_shares = add_sector_shares(sector_year_summary)
    cumulative_sector_summary = build_cumulative_sector_summary(df_with_shares)
    hhi_yearly = build_hhi_yearly(df_with_shares)
    share_change = build_share_change_2020_2024(df_with_shares)
    import_exposure_candidates = build_import_exposure_candidates(share_change)

    save_tables(
        sector_year_summary,
        cumulative_sector_summary,
        hhi_yearly,
        share_change,
        import_exposure_candidates,
    )
    # Concentration figures
    fig_import_concentration_hhi(hhi_yearly)
    fig_export_concentration_hhi(hhi_yearly)

    # 2024 import structure by partner
    plot_top_sectors(
        df_with_shares=df_with_shares,
        partner="China",
        flow="Import",
        year=2024,
        title="Top 10 Indian Import Sectors from China, 2024",
        xlabel="Share of India imports from China",
        output_name="fig_hs2_top_import_sectors_china_2024.png",
    )

    plot_top_sectors(
        df_with_shares=df_with_shares,
        partner="European Union",
        flow="Import",
        year=2024,
        title="Top 10 Indian Import Sectors from the European Union, 2024",
        xlabel="Share of India imports from EU",
        output_name="fig_hs2_top_import_sectors_eu_2024.png",
    )

    plot_top_sectors(
        df_with_shares=df_with_shares,
        partner="United States",
        flow="Import",
        year=2024,
        title="Top 10 Indian Import Sectors from the United States, 2024",
        xlabel="Share of India imports from US",
        output_name="fig_hs2_top_import_sectors_us_2024.png",
    )

    # 2024 export structure by partner
    plot_top_sectors(
        df_with_shares=df_with_shares,
        partner="European Union",
        flow="Export",
        year=2024,
        title="Top 10 Indian Export Sectors to the European Union, 2024",
        xlabel="Share of India exports to EU",
        output_name="fig_hs2_top_export_sectors_eu_2024.png",
    )

    plot_top_sectors(
        df_with_shares=df_with_shares,
        partner="United States",
        flow="Export",
        year=2024,
        title="Top 10 Indian Export Sectors to the United States, 2024",
        xlabel="Share of India exports to US",
        output_name="fig_hs2_top_export_sectors_us_2024.png",
    )

    # Share-change figures
    plot_top_share_changes(
        share_change=share_change,
        partner="China",
        flow="Import",
        title="Top Increases in India's Import Share from China, 2020 to 2024",
        xlabel="Change in sector share",
        output_name="fig_hs2_share_change_import_china_2020_2024.png",
        direction="increase",
    )

    plot_top_share_changes(
        share_change=share_change,
        partner="European Union",
        flow="Export",
        title="Top Increases in India's Export Share to the European Union, 2020 to 2024",
        xlabel="Change in sector share",
        output_name="fig_hs2_share_change_export_eu_2020_2024.png",
        direction="increase",
    )

    plot_top_share_changes(
        share_change=share_change,
        partner="United States",
        flow="Export",
        title="Top Increases in India's Export Share to the United States, 2020 to 2024",
        xlabel="Change in sector share",
        output_name="fig_hs2_share_change_export_us_2020_2024.png",
        direction="increase",
    )

    print("\n" + "=" * 80)
    print("TABLE PREVIEWS")
    print("=" * 80)

    print("\nSector-year summary:")
    print(sector_year_summary.head())

    print("\nCumulative sector summary:")
    print(cumulative_sector_summary.head())

    print("\nHHI yearly:")
    print(hhi_yearly.head())

    print("\nShare change 2020 to 2024:")
    print(share_change.head())

    print_analysis_notes(hhi_yearly, import_exposure_candidates, df_with_shares, share_change)

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Tables -> {TABLES_DIR}")
    print(f"Figures -> {FIGURES_DIR}")

if __name__ == "__main__":
    main()