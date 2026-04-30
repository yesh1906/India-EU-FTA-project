# India–EU FTA Trade Diversification Project

## Project overview
This project analyses whether the India–EU Free Trade Agreement can be understood as a diversification strategy that reduces India’s long-run dependence on concentrated Chinese imports, particularly in electrical machinery.

## Research question
Can the India–EU FTA help India reduce strategic import dependence on China by strengthening export diversification and sectoral trade resilience?

## Data
- **Source:** UN Comtrade
- **Period:** 2020–2024
- **Levels used:**
  - total bilateral goods trade
  - HS2 sector-level goods trade

## Repository structure
- `source/` Python scripts for cleaning, analysis, and forecasting
- `data/raw/` raw downloaded trade files
- `data/clean/` cleaned datasets
- `output/figures/` generated charts
- `output/tables/` generated tables
- `blog.qmd` Quarto blog file
- `blog.html` rendered blog output
- `requirements.txt` required Python libraries

## Reproducibility

### Option 1: Render from saved outputs
The Quarto file can be rendered directly from the saved outputs without rerunning all Python scripts, as the generated figures and tables are already stored in the repository.

### Option 2: To Reproduce the full analysis from scratch
First, install the required Python libraries:

```bash
pip install -r requirements.txt

python source/01_comtrade_data.py
python source/02_cleaning_data.py
python source/03_totals_overview.py
python source/04_hs2_concentrated.py
python source/05_forecast_trends.py

quarto render blog.qm
```
## Outputs
- GitHub repository: https://github.com/yesh1906/India-EU-FTA-project 
- Project endpoint: 