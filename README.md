#  Data Cleaning and Analysis Tool

## Overview
This project provides a comprehensive data cleaning and analysis tool using Python. It includes functionality for cleaning data, handling missing values, detecting and removing outliers, fetching stock data, and generating interactive visualizations and reports.

## Features
- **Data Cleaning:**
  - Standardizes column names
  - Removes duplicate rows
  - Handles missing values using different strategies (interpolation, forward fill, backward fill, dropping)
  - Detects and removes outliers using IQR or Z-score methods

- **Data Analysis:**
  - Fetches stock market data using `yfinance`
  - Reads CSV files for data analysis
  - Generates summary statistics
  - Logs all data cleaning and transformation steps

- **Visualization:**
  - Bar plots, heatmaps, and scatter plots using `matplotlib` and `seaborn`
  - Interactive visualizations using `plotly`

- **Reporting:**
  - Generates a PDF report summarizing data insights

## Installation
To use this project, install the required dependencies:
```sh
pip install pandas numpy matplotlib seaborn plotly yfinance fpdf
```

## Usage
### Data Cleaning
```python
from data_cleaning import EnhancedDataCleaner

# Load dataset
df = pd.read_csv('data.csv')

# Clean data
cleaner = EnhancedDataCleaner(df)
cleaned_df = (cleaner
               .clean_column_names()
               .remove_duplicates()
               .handle_missing_values(strategy='interpolate')
               .remove_outliers(method='iqr')
               .get_cleaned_data())
```

### Data Analysis
```python
from data_analysis import AdvancedDataAnalyzer

analyzer = AdvancedDataAnalyzer()
analyzer.fetch_stock_data(['AAPL', 'MSFT'], '2023-01-01', '2024-01-01')
analyzer.clean_data()
analyzer.create_bar_plot(x_col='date', y_col='close', title='Stock Prices')
analyzer.generate_pdf_report('report.pdf')
```

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- yfinance
- fpdf

## ScreenShots

<img width="350" alt="1" src="https://github.com/user-attachments/assets/de967a7a-e653-46d0-91ac-78089e5f0b81" />
<img width="350" alt="2" src="https://github.com/user-attachments/assets/22ee0bd3-3626-405f-a1a0-e27ebf921d5c" />
<img width="350" alt="3" src="https://github.com/user-attachments/assets/de5ff5ec-05e8-4b53-aac4-e2ce5d73c966" />


