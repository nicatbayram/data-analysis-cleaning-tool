import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from io import StringIO
import yfinance as yf
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataCleaner:
    """Enhanced version of DataCleaner with additional features."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []
    
    def log_operation(self, operation):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cleaning_log.append(f"{timestamp}: {operation}")
    
    def clean_column_names(self):
        """Standardize column names."""
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        self.log_operation("Standardized column names")
        return self
    
    def remove_duplicates(self, subset=None):
        """Remove duplicate rows."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed_rows = initial_rows - len(self.df)
        self.log_operation(f"Removed {removed_rows} duplicate rows")
        return self
    
    def handle_missing_values(self, strategy='interpolate'):
        """Handle missing values with multiple strategies."""
        initial_nulls = self.df.isnull().sum().sum()
        
        if strategy == 'interpolate':
            self.df = self.df.interpolate(method='linear')
        elif strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'ffill':
            self.df = self.df.fillna(method='ffill')
        elif strategy == 'bfill':
            self.df = self.df.fillna(method='bfill')
        
        final_nulls = self.df.isnull().sum().sum()
        self.log_operation(f"Handled {initial_nulls - final_nulls} missing values using {strategy}")
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """Remove outliers using IQR or Z-score method."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(self.df)
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[~((self.df[col] < (Q1 - threshold * IQR)) | 
                                  (self.df[col] > (Q3 + threshold * IQR)))]

            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        
        removed_rows = initial_rows - len(self.df)
        self.log_operation(f"Removed {removed_rows} outliers using {method} method")
        return self
    
    def get_cleaned_data(self):
        """Return the cleaned DataFrame."""
        return self.df.copy()

class AdvancedDataAnalyzer:
    """
    Enhanced data analysis class with interactive visualizations,
    multiple data sources, and automated reporting.
    """
    
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
        self.cleaning_log = []
    
    def fetch_stock_data(self, symbols, start_date, end_date):
        """Fetch stock market data using yfinance."""
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data[symbol] = ticker.history(start=start_date, end=end_date)
            
            self.raw_data = pd.concat(data, axis=1)
            self.log_operation("Stock data fetched using yfinance")
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
    
    def fetch_csv_data(self, file_path):
        """Fetch data from a CSV file."""
        try:
            self.raw_data = pd.read_csv(file_path, on_bad_lines='skip')
            self.log_operation("Data fetched from CSV file")
        except Exception as e:
            print(f"Error fetching CSV data: {str(e)}")
    
    def log_operation(self, operation):
        """Log operations with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cleaning_log.append(f"{timestamp}: {operation}")
    
    def clean_data(self):
        """Clean the raw data using enhanced cleaning methods."""
        if self.raw_data is None:
            print("No data to clean. Please fetch data first.")
            return
        
        cleaner = EnhancedDataCleaner(self.raw_data)
        self.cleaned_data = (cleaner
                           .clean_column_names()
                           .remove_duplicates()
                           .handle_missing_values(strategy='interpolate')
                           .remove_outliers(method='iqr')
                           .get_cleaned_data())
        
        self.cleaning_log.extend(cleaner.cleaning_log)
    
    def create_bar_plot(self, x_col, y_col, title):
        """Create a bar plot using Matplotlib."""
        x_col = x_col.lower()
        y_col = y_col.lower()
        
        if x_col not in self.cleaned_data.columns or y_col not in self.cleaned_data.columns:
            print(f"Available columns: {list(self.cleaned_data.columns)}")
            raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in data")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.cleaned_data, x=x_col, y=y_col, palette='Blues_d')
        plt.title(title)
        plt.xlabel(x_col.title())
        plt.ylabel(y_col.title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        self.log_operation(f"Created bar plot: {title}")
    
    def create_heatmap(self, columns, title):
        """Create a heatmap using Seaborn."""
        correlation = self.cleaned_data[columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        self.log_operation(f"Created heatmap: {title}")
    
    def create_scatter_plot(self, x_col, y_col, title):
        """Create a scatter plot using Seaborn."""
        x_col = x_col.lower()
        y_col = y_col.lower()
        
        if x_col not in self.cleaned_data.columns or y_col not in self.cleaned_data.columns:
            print(f"Available columns: {list(self.cleaned_data.columns)}")
            raise ValueError(f"Columns '{x_col}' or '{y_col}' not found in data")
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.cleaned_data, x=x_col, y=y_col, color='blue')
        plt.title(title)
        plt.xlabel(x_col.title())
        plt.ylabel(y_col.title())
        plt.tight_layout()
        plt.show()
        self.log_operation(f"Created scatter plot: {title}")
    
    def generate_pdf_report(self, filename="analysis_report.pdf"):
        """Generate a PDF report with analysis results."""
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
        pdf.ln(10)
        
        # Add summary statistics
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Summary Statistics", ln=True)
        pdf.set_font("Arial", "", 10)
        
        summary = self.cleaned_data.describe().round(2)
        for col in summary.columns:
            pdf.cell(0, 10, f"\n{col}:", ln=True)
            for stat, value in summary[col].items():
                pdf.cell(0, 10, f"{stat}: {value}", ln=True)
            pdf.ln(5)
        
        # Add cleaning log
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Data Cleaning Log", ln=True)
        pdf.set_font("Arial", "", 10)
        for log in self.cleaning_log:
            pdf.cell(0, 10, log, ln=True)
        
        # Save the report
        pdf.output(filename)
        print(f"Report generated: {filename}")

def main():
    """Example usage of the data analysis tools."""
    # Create analyzer instance
    analyzer = AdvancedDataAnalyzer()
    
    # Fetch data from the specified CSV file
    print("Fetching data from CSV file...")
    csv_file_path = "C:/Users/nicat/Desktop/Data/unsd-citypopulation-year-fm.csv.crdownload"
    analyzer.fetch_csv_data(csv_file_path)
    
    if analyzer.raw_data is not None:
        # Update column names
        analyzer.raw_data.columns = ["Country", "Year", "Type", "Gender", "Location", "AreaType", "EstimateType", "FigureType", "NextYear", "Estimate", "Additional"]
        
        # Clean data
        analyzer.clean_data()
        
        # Check available columns
        print("Available columns:", list(analyzer.cleaned_data.columns))
        
        # Create a bar plot
        analyzer.create_bar_plot('country', 'estimate', 'Country vs Estimate')
        
        # Create a heatmap
        analyzer.create_heatmap(['Estimate', 'NextYear'], 'Correlation Heatmap')
        
        # Create a scatter plot
        analyzer.create_scatter_plot('year', 'estimate', 'Year vs Estimate')
        
        # Generate report
        analyzer.generate_pdf_report()
    else:
        print("Failed to fetch data from CSV file.")

if __name__ == "__main__":
    main()