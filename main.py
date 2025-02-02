import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from io import StringIO
import yfinance as yf
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')
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
        
   
        
    def filter_top_n_countries(self, column='estimate', n=10, aggregation='mean'):
        """Filter top N countries."""
        if self.cleaned_data is None:
            print("Clean the data first.")
            return
            
        grouped_data = (self.cleaned_data.groupby('country')[column]
                       .agg(aggregation)
                       .sort_values(ascending=False)
                       .head(n))
        return grouped_data.index.tolist()
    
    def create_enhanced_bar_plot(self, x_col, y_col, title, top_n=10):
        """Improved bar chart rendering."""
        x_col = x_col.lower()
        y_col = y_col.lower()
        
        # Get top N countries
        top_countries = self.filter_top_n_countries(y_col, top_n)
        filtered_data = self.cleaned_data[self.cleaned_data['country'].isin(top_countries)]
        
        # Calculate average values ​​by country
        plot_data = (filtered_data.groupby('country')[y_col]
                    .mean()
                    .sort_values(ascending=True))
        
        # Visualization
        plt.figure(figsize=(12, 8))
        bars = plt.barh(plot_data.index, plot_data.values)
        
        # Visual improvements
        plt.title(title, pad=20, fontsize=14)
        plt.xlabel(y_col.title(), fontsize=12)
        plt.ylabel(x_col.title(), fontsize=12)
        
        # Add values ​​on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}',
                    ha='left', va='center', fontsize=10)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        self.log_operation(f"Created enhanced bar plot: {title} (Top {top_n} countries)")
    
    def create_interactive_scatter_plot(self, x_col, y_col, title, top_n=10):
        """Create an interactive scatter plot using Plotly."""
        x_col = x_col.lower()
        y_col = y_col.lower()
        
        # Get top N countries
        top_countries = self.filter_top_n_countries(y_col, top_n)
        filtered_data = self.cleaned_data[self.cleaned_data['country'].isin(top_countries)]
        
        fig = px.scatter(filtered_data, x=x_col, y=y_col, color='country',
                        title=title, hover_data=['country', 'year'])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend_title_text='Ülkeler',
            height=600
        )
        
        fig.show()
        self.log_operation(f"Created interactive scatter plot: {title} (Top {top_n} countries)")
    
    def create_time_series_plot(self, y_col, title, top_n=5):
        """Create a time series chart."""
        y_col = y_col.lower()
        
        # Get top N countries
        top_countries = self.filter_top_n_countries(y_col, top_n)
        filtered_data = self.cleaned_data[self.cleaned_data['country'].isin(top_countries)]
        
        plt.figure(figsize=(12, 6))
        
        for country in top_countries:
            country_data = filtered_data[filtered_data['country'] == country]
            plt.plot(country_data['year'], country_data[y_col], 
                    marker='o', label=country)
        
        plt.title(title, pad=20, fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(y_col.title(), fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        self.log_operation(f"Created time series plot: {title} (Top {top_n} countries)")

def main():
    """Example usage."""
    analyzer = AdvancedDataAnalyzer()
    
    print("Retrieving data from CSV file...")
    csv_file_path = "C:/Users/nicat/Desktop/Data/unsd-citypopulation-year-fm.csv.crdownload"
    analyzer.fetch_csv_data(csv_file_path)
    
    if analyzer.raw_data is not None:
        # Update column names
        analyzer.raw_data.columns = ["Country", "Year", "Type", "Gender", "Location", 
                                   "AreaType", "EstimateType", "FigureType", "NextYear", 
                                   "Estimate", "Additional"]
        
        # Clenaing data
        analyzer.clean_data()
        
        # İmproved visualizations
        analyzer.create_enhanced_bar_plot('country', 'estimate', 
                                        'Top 10 Countries with the Highest Population', top_n=10)
        
        analyzer.create_interactive_scatter_plot('year', 'estimate', 
                                               'Population Distribution by Years', top_n=10)
        
        analyzer.create_time_series_plot('estimate', 
                                       'Time Series of the 5 Countries with the Highest Population', top_n=5)
        
      # Generate report
        analyzer.generate_pdf_report()
    else:
        print("Unable to retrieve data from CSV file.")

if __name__ == "__main__":
    main()