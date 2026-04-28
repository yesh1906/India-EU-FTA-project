from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

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
# 2. LOAD AND PREPARE DATA
# ------------------------------------------------------
def load_hs2(path: Path) -> pd.DataFrame:
    """
    Load the cleaned HS2 dataset.
    This stage adds a small Unit 5-style modelling component:
    trend-based forecasting of key sector shares.
    It is a forward-looking projection based on recent trends/data.
    """
    return pd.read_csv(path)

def build_sector_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to year x partner x flow x sector level.
    """
    out = (
        df.groupby(
            ["refYear", "partner_group", "flow_india", "cmdCode", "cmdDesc"],
            as_index=False
        )["trade_value_usd"]
        .sum()
    )
    return out

def add_sector_shares(sector_year_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sector's share within each year-partner-flow basket.
    """
    df = sector_year_summary.copy()

    totals = (
        df.groupby(["refYear", "partner_group", "flow_india"], as_index=False)["trade_value_usd"]
        .sum()
        .rename(columns={"trade_value_usd": "partner_flow_total_usd"})
    )

    df = df.merge(totals, on=["refYear", "partner_group", "flow_india"], how="left")
    df["sector_share"] = df["trade_value_usd"] / df["partner_flow_total_usd"]

    return df

# ------------------------------------------------------
# 3. SERIES SELECTION
# ------------------------------------------------------
def get_target_series(df_with_shares: pd.DataFrame) -> pd.DataFrame:
    """
    Select the strategically relevant series that we want to forecast.
    Chosen series:
    - China Import, HS2 85: electrical machinery
    - European Union Export, HS2 85: electrical machinery
    - United States Export, HS2 85: electrical machinery
    - European Union Import, HS2 84: machinery/mechanical appliances --> 
      this is could help us predict the outlook for trade between India and EU, when the FTA commences.
    """
    targets = [
        {"partner_group": "China", "flow_india": "Import", "cmdCode": "85", "series_name": "China Import - HS2 85"},
        {"partner_group": "European Union", "flow_india": "Export", "cmdCode": "85", "series_name": "EU Export - HS2 85"},
        {"partner_group": "United States", "flow_india": "Export", "cmdCode": "85", "series_name": "US Export - HS2 85"},
        {"partner_group": "European Union", "flow_india": "Import", "cmdCode": "84", "series_name": "EU Import - HS2 84"},
    ]

    frames = []

    df_with_shares = df_with_shares.copy()
    df_with_shares["cmdCode"] = df_with_shares["cmdCode"].astype(str)

    for target in targets:
        subset = df_with_shares[
            (df_with_shares["partner_group"] == target["partner_group"]) &
            (df_with_shares["flow_india"] == target["flow_india"]) &
            (df_with_shares["cmdCode"] == target["cmdCode"])
        ].copy()

        if not subset.empty:
            subset["series_name"] = target["series_name"]
            frames.append(subset)

    if not frames:
        raise ValueError("No target series were found in the HS2 dataset.")

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["series_name", "refYear"])

# ------------------------------------------------------
# 4. TREND MODEL AND FORECAST
# ------------------------------------------------------
def fit_linear_trend_and_forecast(series_df: pd.DataFrame, forecast_years: list[int]) -> pd.DataFrame:
    """
    Fit a simple linear trend:

        sector_share_t = alpha + beta * year_t + error_t

    Then forecast sector shares for future years.
    This is an illustrative trend projection, not a structural forecast.
    """
    df = series_df.sort_values("refYear").copy()

    x = df["refYear"].to_numpy(dtype=float)
    y = df["sector_share"].to_numpy(dtype=float)

    # Fit linear trend using numpy polyfit
    beta, alpha = np.polyfit(x, y, 1)

    # In-sample fitted values
    df["fitted_share"] = alpha + beta * df["refYear"]

    # Forecast future values
    forecast_df = pd.DataFrame({"refYear": forecast_years})
    forecast_df["sector_share"] = np.nan
    forecast_df["fitted_share"] = alpha + beta * forecast_df["refYear"]

    # Attach identifying info
    for col in ["series_name", "partner_group", "flow_india", "cmdCode", "cmdDesc"]:
        forecast_df[col] = df.iloc[0][col]

    df["is_forecast"] = False
    forecast_df["is_forecast"] = True

    out = pd.concat([df, forecast_df], ignore_index=True)

    # Store regression coefficients for reference
    out["alpha"] = alpha
    out["beta"] = beta

    return out

def build_all_forecasts(target_series: pd.DataFrame, forecast_years: list[int]) -> pd.DataFrame:
    """
    Build forecasts for all selected series.
    """
    frames = []

    for series_name, group in target_series.groupby("series_name"):
        forecasted = fit_linear_trend_and_forecast(group, forecast_years)
        frames.append(forecasted)

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------
# 5. SAVE TABLES
# ------------------------------------------------------
def save_forecast_tables(all_forecasts: pd.DataFrame) -> None:
    """
    Save the full forecast table and a compact future-only table.
    """
    all_forecasts.to_csv(TABLES_DIR / "forecast_sector_share_trends.csv", index=False)

    future_only = all_forecasts[all_forecasts["is_forecast"]].copy()
    future_only.to_csv(TABLES_DIR / "forecast_sector_share_future_only.csv", index=False)


# ------------------------------------------------------
# 6. PLOTTING
# ------------------------------------------------------
def pct_label(x: float) -> str:
    return f"{x:.2%}"

def plot_forecast_series(series_df: pd.DataFrame, output_name: str) -> None:
    """
    Plot actual sector shares plus fitted trend and forecasts.
    """
    df = series_df.sort_values("refYear").copy()

    actual = df[df["is_forecast"] == False].copy()
    forecast = df[df["is_forecast"] == True].copy()

    plt.figure(figsize=(10, 6))

    # Actual observed values
    plt.plot(
        actual["refYear"],
        actual["sector_share"],
        marker="o",
        label="Actual share"
    )

    # Fitted/forecast line
    plt.plot(
        df["refYear"],
        df["fitted_share"],
        marker="o",
        linestyle="--",
        label="Trend / forecast"
    )

    # Add a vertical divider between actual and forecast
    last_actual_year = actual["refYear"].max()
    plt.axvline(last_actual_year, linewidth=1)

    title = f"{df.iloc[0]['series_name']} Sector Share Trend and Forecast"
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Sector share")

    # Improve y-axis labels by showing percentage scale
    y_ticks = plt.yticks()[0]
    plt.yticks(y_ticks, [pct_label(y) for y in y_ticks])

    plt.xticks(df["refYear"].astype(int).tolist())
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

def save_all_forecast_figures(all_forecasts: pd.DataFrame) -> None:
    """
    Save one figure per forecast series.
    """
    for series_name, group in all_forecasts.groupby("series_name"):
        safe_name = (
            series_name.lower()
            .replace(" ", "_")
            .replace("-", "")
        )
        output_name = f"fig_forecast_{safe_name}.png"
        plot_forecast_series(group, output_name)

# ------------------------------------------------------
# 7. PRINT ANALYSIS NOTES
# ------------------------------------------------------
def print_forecast_notes(all_forecasts: pd.DataFrame) -> None:
    """
    Print compact notes for the forecast section.
    """
    print("\n" + "=" * 80)
    print("FORECAST ANALYSIS NOTES")
    print("=" * 80)

    future_only = all_forecasts[all_forecasts["is_forecast"]].copy()

    for series_name, group in future_only.groupby("series_name"):
        group = group.sort_values("refYear")
        beta = group["beta"].iloc[0]
        latest_forecast = group.iloc[-1]

        direction = "increasing" if beta > 0 else "decreasing"

        print(f"\n{series_name}")
        print(f"Trend slope (beta): {beta:.6f}")
        print(f"Interpretation: sector share is {direction} over time under the simple trend model.")
        print(
            f"Forecast {int(latest_forecast['refYear'])}: "
            f"{latest_forecast['fitted_share']:.2%}"
        )

# ------------------------------------------------------
# 8. MAIN WORKFLOW
# ------------------------------------------------------
def main() -> None:
    """
    Methodology for this stage:
    - use the cleaned HS2 dataset
    - construct yearly sector-share series
    - select a small number of strategically relevant series
    - fit simple linear trend models
    - forecast 2025 and 2026 sector shares
    - save forecast tables and figures
    This is a forward-looking but modest modelling extension.
    """
    print("\nStarting trend-based sector-share forecasting...\n")

    hs2 = load_hs2(HS2_PATH)
    sector_year_summary = build_sector_year_summary(hs2)
    df_with_shares = add_sector_shares(sector_year_summary)

    target_series = get_target_series(df_with_shares)

    print("\nSelected target series:")
    print(target_series[["refYear", "series_name", "partner_group", "flow_india", "cmdCode", "cmdDesc", "sector_share"]])

    forecast_years = [2025, 2026]
    all_forecasts = build_all_forecasts(target_series, forecast_years)

    save_forecast_tables(all_forecasts)
    save_all_forecast_figures(all_forecasts)

    print("\nForecast table preview:")
    print(all_forecasts.head(12))

    print_forecast_notes(all_forecasts)

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Forecast tables -> {TABLES_DIR}")
    print(f"Forecast figures -> {FIGURES_DIR}")

if __name__ == "__main__":
    main()