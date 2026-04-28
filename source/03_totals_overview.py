from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 40)

# ------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# 2. LOAD CLEAN TOTALS DATA
# ------------------------------------------------------
TOTALS_PATH = CLEAN_DIR / "totals_clean.csv"

def load_totals(path: Path) -> pd.DataFrame:
    """
    Load the cleaned totals dataset.
    At this stage we are working only with the totals file because
    the goal is to establish the broad bilateral trade structure
    before focuing on to the HS2 sector.
    """
    df = pd.read_csv(path)
    return df

# ------------------------------------------------------
# 3. BUILD SUMMARY TABLES
# ------------------------------------------------------
def build_partner_flow_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the core yearly bilateral table:
    year x partner x India-perspective flow.
    """
    summary = (
        df.groupby(["refYear", "partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .sort_values(["partner_group", "flow_india", "refYear"])
    )
    return summary

def build_partner_total_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total imports, exports, and combined trade by partner
    across the full 2020–2024 sample window.
    """
    pivot = (
        df.groupby(["partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .pivot(index="partner_group", columns="flow_india", values="trade_value_usd")
        .fillna(0)
        .reset_index()
    )

    # Ensure both columns exist
    if "Import" not in pivot.columns:
        pivot["Import"] = 0
    if "Export" not in pivot.columns:
        pivot["Export"] = 0

    pivot["Total_Trade"] = pivot["Import"] + pivot["Export"]
    pivot["Trade_Balance"] = pivot["Export"] - pivot["Import"]

    pivot = pivot.rename_axis(None, axis=1)
    pivot = pivot.sort_values("Total_Trade", ascending=False)

    return pivot


def build_trade_balance_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly trade balance by partner from India's perspective.
    Trade balance = exports - imports
    Positive = surplus
    Negative = deficit
    """
    yearly = (
        df.groupby(["refYear", "partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .pivot(index=["refYear", "partner_group"], columns="flow_india", values="trade_value_usd")
        .fillna(0)
        .reset_index()
    )
    if "Import" not in yearly.columns:
        yearly["Import"] = 0
    if "Export" not in yearly.columns:
        yearly["Export"] = 0

    yearly["trade_balance_usd"] = yearly["Export"] - yearly["Import"]
    yearly = yearly.rename_axis(None, axis=1)

    return yearly

# ------------------------------------------------------
# 4. SAVE TABLES
# ------------------------------------------------------
def save_tables(
    partner_flow_year_summary: pd.DataFrame,
    partner_total_summary: pd.DataFrame,
    trade_balance_yearly: pd.DataFrame,
) -> None:
    """
    Save the main overview tables for later use in Quarto.
    """
    partner_flow_year_summary.to_csv(TABLES_DIR / "partner_flow_year_summary.csv", index=False)
    partner_total_summary.to_csv(TABLES_DIR / "partner_total_summary.csv", index=False)
    trade_balance_yearly.to_csv(TABLES_DIR / "partner_trade_balance.csv", index=False)

# ------------------------------------------------------
# 5. FIGURES
# ------------------------------------------------------
def to_billions(series: pd.Series) -> pd.Series:
    """
    Convert USD values into USD billions for cleaner plotting.
    """
    return series / 1_000_000_000

def fig_trade_by_partner_flow(partner_flow_year_summary: pd.DataFrame) -> None:
    """
    Figure 1:
    Line chart of bilateral trade by partner and flow over time.
    Values are shown in USD billions for readability.
    """
    plt.figure(figsize=(11, 6))

    for (partner, flow), group in partner_flow_year_summary.groupby(["partner_group", "flow_india"]):
        group = group.sort_values("refYear")
        plt.plot(
            group["refYear"],
            to_billions(group["trade_value_usd"]),
            marker="o",
            label=f"{partner} - {flow}"
        )

    plt.title("India's Bilateral Trade by Partner and Flow, 2020–2024")
    plt.xlabel("Year")
    plt.ylabel("Trade value (USD billions)")
    plt.xticks(sorted(partner_flow_year_summary["refYear"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_trade_by_partner_flow.png", dpi=300)
    plt.close()

def fig_total_trade_by_partner(partner_total_summary: pd.DataFrame) -> None:
    """
    Figure 2:
    Total trade volume by partner across the full sample period.
    """
    plot_df = partner_total_summary.sort_values("Total_Trade", ascending=False).copy()
    plot_df["Total_Trade_Bn"] = to_billions(plot_df["Total_Trade"])

    plt.figure(figsize=(9, 6))
    bars = plt.bar(plot_df["partner_group"], plot_df["Total_Trade_Bn"])

    plt.title("Total India Trade Volume by Partner, 2020–2024")
    plt.xlabel("Partner")
    plt.ylabel("Total trade value (USD billions)")

    # Optional value labels
    for bar, value in zip(bars, plot_df["Total_Trade_Bn"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_total_trade_by_partner.png", dpi=300)
    plt.close()

def fig_trade_balance_by_partner(trade_balance_yearly: pd.DataFrame) -> None:
    """
    Figure 3:
    Line chart of India's bilateral trade balance by partner over time.
    """
    plt.figure(figsize=(11, 6))

    for partner, group in trade_balance_yearly.groupby("partner_group"):
        group = group.sort_values("refYear")
        plt.plot(
            group["refYear"],
            to_billions(group["trade_balance_usd"]),
            marker="o",
            label=partner
        )

    plt.axhline(0, linewidth=1)
    plt.title("India's Bilateral Trade Balance by Partner, 2020–2024")
    plt.xlabel("Year")
    plt.ylabel("Trade balance (USD billions)")
    plt.xticks(sorted(trade_balance_yearly["refYear"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_trade_balance_by_partner.png", dpi=300)
    plt.close()


def fig_partner_trade_share(partner_total_summary: pd.DataFrame) -> None:
    """
    Figure 4:
    Partner share of cumulative trade volume across the sample period.
    This is a sample-period aggregate share, not a year-by-year share.
    """
    plot_df = partner_total_summary.copy()
    grand_total = plot_df["Total_Trade"].sum()
    plot_df["Trade_Share"] = plot_df["Total_Trade"] / grand_total

    plt.figure(figsize=(9, 6))
    bars = plt.bar(plot_df["partner_group"], plot_df["Trade_Share"])

    plt.title("Partner Share of Cumulative India Trade, 2020–2024")
    plt.xlabel("Partner")
    plt.ylabel("Share of sample-period trade")

    # Optional value labels
    for bar, value in zip(bars, plot_df["Trade_Share"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_partner_trade_share.png", dpi=300)
    plt.close()

# ------------------------------------------------------
# 6. ANALYTICAL TAKEAWAYS
# ------------------------------------------------------
def print_analysis_takeaways(
    partner_total_summary: pd.DataFrame,
    trade_balance_yearly: pd.DataFrame,
) -> None:
    """
    Print compact interpretations.
    """
    print("\n" + "=" * 80)
    print("OVERVIEW")
    print("=" * 80)

    # Largest partner by total trade
    largest_partner = partner_total_summary.iloc[0]["partner_group"]
    largest_trade = partner_total_summary.iloc[0]["Total_Trade"]
    print(f"Largest partner by total trade volume (2020–2024): {largest_partner} ({largest_trade:,.0f} USD)")

    # Biggest surplus and deficit overall
    balance_ranked = partner_total_summary.sort_values("Trade_Balance")
    biggest_deficit_partner = balance_ranked.iloc[0]["partner_group"]
    biggest_deficit_value = balance_ranked.iloc[0]["Trade_Balance"]

    biggest_surplus_partner = balance_ranked.iloc[-1]["partner_group"]
    biggest_surplus_value = balance_ranked.iloc[-1]["Trade_Balance"]

    print(f"Largest overall trade deficit: {biggest_deficit_partner} ({biggest_deficit_value:,.0f} USD)")
    print(f"Largest overall trade surplus: {biggest_surplus_partner} ({biggest_surplus_value:,.0f} USD)")

    # 2024 snapshot/overview
    latest_year = trade_balance_yearly["refYear"].max()
    latest = trade_balance_yearly[trade_balance_yearly["refYear"] == latest_year].copy()

    print(f"\nTrade balance snapshot for {latest_year}:")
    print(latest[["partner_group", "trade_balance_usd"]].sort_values("trade_balance_usd"))


# ------------------------------------------------------
# 7. MAIN WORKFLOW
# ------------------------------------------------------
def main() -> None:
    """
    Methodology for this stage:
    - Use the cleaned totals dataset only
    - Construct yearly bilateral summaries
    - Compare imports, exports, and total trade by partner
    - Measure trade balance from India's perspective
    - Save tables and overview figures
    """
    print("\nStarting trade overview analysis...\n")

    totals = load_totals(TOTALS_PATH)

    partner_flow_year_summary = build_partner_flow_year_summary(totals)
    partner_total_summary = build_partner_total_summary(totals)
    trade_balance_yearly = build_trade_balance_yearly(totals)

    save_tables(partner_flow_year_summary, partner_total_summary, trade_balance_yearly)

    fig_trade_by_partner_flow(partner_flow_year_summary)
    fig_total_trade_by_partner(partner_total_summary)
    fig_trade_balance_by_partner(trade_balance_yearly)
    fig_partner_trade_share(partner_total_summary)

    print("\n" + "=" * 80)
    print("TABLE PREVIEWS")
    print("=" * 80)
    print("\nPartner-flow-year summary:")
    print(partner_flow_year_summary.head())

    print("\nPartner total summary:")
    print(partner_total_summary)

    print("\nTrade balance yearly:")
    print(trade_balance_yearly.head())

    print_analysis_takeaways(partner_total_summary, trade_balance_yearly)

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Tables -> {TABLES_DIR}")
    print(f"Figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()