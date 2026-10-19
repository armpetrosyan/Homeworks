# Homework 1 — Bass Diffusion Model

## Predicting the Diffusion of Samsung Micro RGB TVs Using Historical OLED TV Data

### Project Overview

This project forecasts the worldwide adoption of **Samsung Micro RGB** TVs, one of TIME's Best Inventions of 2025 ([link](https://time.com/collections/best-inventions-2025/7318277/samsung-micro-rgb/)), using the **Bass Diffusion Model**.

OLED TVs are used as the look-alike innovation. Worldwide OLED TV shipments from 2015 to 2025 are used to estimate p, q and M, and these are then used to forecast adoption of Micro RGB / RGB LED TVs from 2025 to 2040.

The analysis was done in Python and Jupyter Notebook.

### Project Structure

```text
HW1/
├── bass_model.ipynb
├── README.md
├── data/
│   ├── oled_tv_shipments.csv
│   ├── analyst_forecasts_rgb_tv.csv
│   ├── oled_bass_fit_results.csv
│   └── micro_rgb_forecast.csv
├── img/
│   ├── oled_annual_shipments.png
│   ├── oled_bass_model_comparison.png
│   ├── micro_rgb_annual_adopters.png
│   └── micro_rgb_cumulative_adopters.png
└── report/
    ├── report.pdf
    ├── report_source.md
    └── references.md
```

**Files**

- `bass_model.ipynb`: main notebook with all the code (Bass model functions, fitting, forecast and graphs).
- `data/oled_tv_shipments.csv`: worldwide OLED TV shipments 2015–2025 (IHS Markit / Omdia).
- `data/analyst_forecasts_rgb_tv.csv`: Omdia and TrendForce forecasts for RGB LED TVs, used only to check the model.
- `data/oled_bass_fit_results.csv`: actual vs fitted OLED shipments.
- `data/micro_rgb_forecast.csv`: forecast of new and cumulative adopters per year (low, base, high).
- `img/`: graphs created by the notebook.
- `report/`: final report (PDF), its markdown source and the reference list.

### Methodology

1. Collect worldwide OLED TV shipment data (2015–2025).
2. Plot the historical shipments.
3. Estimate p, q and M of the discrete-time Bass model with nonlinear least squares.
4. Check the fit with RMSE and R².
5. Estimate the market potential for Micro RGB / RGB LED TVs with a Fermi estimate.
6. Forecast annual and cumulative adopters for 2025–2040.
7. Compare the forecast with Omdia and TrendForce forecasts.

### Key Results

**OLED TV Bass Model Parameters**

| Parameter | Value |
|---|---:|
| Innovation coefficient (p) | 0.0151 |
| Imitation coefficient (q) | 0.440 |
| Market potential (M) | 56.7 million |

**Model Performance**

| Metric | Value |
|---|---:|
| RMSE | 0.64 million |
| R² | 0.92 |

**Micro RGB / RGB LED TV Forecast (base case)**

| Metric | Estimated Value |
|---|---:|
| Market potential | 70.1 million (range 38.9–116.8) |
| Peak adoption year | 2033 |
| Peak annual adopters | 8.17 million |
| Cumulative adopters by 2030 | 16.6 million |
| Cumulative adopters by 2035 | 53.9 million |

These numbers are a forecast based on the model assumptions, not actual sales figures.

### Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- SciPy

### How to Run the Project

1. Download or clone the repository.
2. Open a terminal in the `HW1` folder.
3. Install the packages if needed:

```bash
python3 -m pip install notebook pandas numpy matplotlib scipy
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `bass_model.ipynb`.
6. Select **Kernel → Restart Kernel and Run All Cells**.

The notebook reads the data, fits the model, creates the forecast and saves the tables in `data/` and the graphs in `img/`.

### Final Report

The full report is in `report/report.pdf`.
