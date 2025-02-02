# Data Analysis and Cleaning Tool

## Overview

This project provides data analysis and cleaning tool that enables users to fetch, clean, analyze, and visualize data efficiently. The tool integrates multiple Python libraries, including Pandas, NumPy, Seaborn, Plotly, and Matplotlib, to offer comprehensive data processing capabilities.

## Features

- **Data Cleaning:**
  - Standardizes column names
  - Removes duplicate records
  - Handles missing values using interpolation, forward fill, or backward fill
  - Removes outliers using IQR or Z-score methods
- **Data Analysis & Visualization:**
  - Generates bar plots, scatter plots, and heatmaps
  - Supports interactive data visualizations using Plotly
- **Data Sources:**
  - Fetches stock market data using Yahoo Finance (`yfinance`)
  - Reads data from CSV files
- **Automated Reporting:**
  - Generates a PDF report containing summary statistics and cleaning logs

## Installation

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn plotly yfinance fpdf
```

## Usage

### 1. Fetch Data

You can fetch data either from a CSV file or from Yahoo Finance.

#### Fetch CSV Data:

```python
analyzer = AdvancedDataAnalyzer()
analyzer.fetch_csv_data("path/to/your/file.csv")
```

#### Fetch Stock Market Data:

```python
analyzer.fetch_stock_data(["AAPL", "GOOGL"], "2023-01-01", "2023-12-31")
```

### 2. Clean Data

```python
analyzer.clean_data()
```

### 3. Generate Visualizations

#### Create Bar Plot:

```python
analyzer.create_bar_plot('country', 'estimate', 'Country vs Estimate')
```

#### Create Heatmap:

```python
analyzer.create_heatmap(['Estimate', 'NextYear'], 'Correlation Heatmap')
```

#### Create Scatter Plot:

```python
analyzer.create_scatter_plot('year', 'estimate', 'Year vs Estimate')
```

### 4. Generate Report

```python
analyzer.generate_pdf_report("analysis_report.pdf")
```
