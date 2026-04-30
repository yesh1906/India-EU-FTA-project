# India–EU FTA Trade Diversification Project

## Project overview
This project analyses whether the India–EU Free Trade Agreement can be understood as a diversification strategy that reduces India’s long-run dependence on concentrated Chinese imports, particularly in electrical machinery.

## Research question
Can the India–EU FTA help India reduce strategic import dependence on China by strengthening export diversification and sectoral trade resilience?

## Data
- Source: UN Comtrade
- Period: 2020–2024
- Levels used:
  - total bilateral goods trade
  - HS2 sector-level goods trade

## Repository structure
- `source/` Python scripts for cleaning, analysis, and forecasting
- `data/raw/` raw downloaded trade files
- `data/clean/` cleaned datasets
- `output/figures/` generated charts
- `output/tables/` generated tables
- `blog.qmd` Quarto blog file

## Reproducibility
Run scripts in order:
1. `01_comtrade_data.py`
2. `02_cleaning_data.py`
3. `03_totals_overview.py`
4. `04_hs2_concentrated.py`
5. `05_forecast_trends.py`

Then render the Quarto file.

## Outputs
- GitHub repository: [paste repo link]
- Project endpoint: [paste rendered blog link]