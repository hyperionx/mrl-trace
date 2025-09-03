import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AnalysisTools:
    def __init__(self):
        """Initialize analysis tools with plotting configuration"""
        # Set style for matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistical summary"""
        stats_dict = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            stats_dict['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].value_counts().head(5).to_dict(),
                'missing_count': df[col].isnull().sum()
            }
        
        return stats_dict
    
    def plot_distribution(self, df: pd.DataFrame, column: str, plot_type: str = 'histogram') -> go.Figure:
        """Plot distribution of a column"""
        fig = go.Figure()
        
        if df[column].dtype in ['object', 'category']:
            # Categorical data
            value_counts = df[column].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f'Distribution of {column}',
                        labels={'x': column, 'y': 'Count'})
        else:
            # Numerical data
            if plot_type == 'histogram':
                fig = px.histogram(df, x=column, title=f'Distribution of {column}')
            elif plot_type == 'box':
                fig = px.box(df, y=column, title=f'Box Plot of {column}')
            elif plot_type == 'violin':
                fig = px.violin(df, y=column, title=f'Violin Plot of {column}')
        
        fig.update_layout(height=400)
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Plot correlation matrix for numerical columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numerical columns for correlation analysis")
            return None
        
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, 
                       title='Correlation Matrix',
                       color_continuous_scale='RdBu',
                       aspect='auto')
        fig.update_layout(height=500)
        return fig
    
    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        """Create scatter plot"""
        if color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}')
        else:
            fig = px.scatter(df, x=x_col, y=y_col,
                           title=f'{y_col} vs {x_col}')
        fig.update_layout(height=400)
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Plot time series data"""
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        fig = px.line(df_copy, x=date_col, y=value_col,
                     title=f'Time Series: {value_col} over time')
        fig.update_layout(height=400)
        return fig
    
    def statistical_tests(self, df: pd.DataFrame, col1: str, col2: str = None, test_type: str = 'normality') -> Dict[str, Any]:
        """Perform statistical tests"""
        results = {}
        
        if test_type == 'normality':
            # Shapiro-Wilk test for normality
            if df[col1].dtype in ['float64', 'int64']:
                statistic, p_value = stats.shapiro(df[col1].dropna())
                results = {
                    'test': 'Shapiro-Wilk Normality Test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
        
        elif test_type == 'correlation':
            # Pearson correlation test
            if col2 and df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
                correlation, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                results = {
                    'test': 'Pearson Correlation Test',
                    'correlation': correlation,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
        
        elif test_type == 'ttest':
            # Independent t-test
            if col2 and df[col1].dtype in ['float64', 'int64'] and df[col2].dtype in ['float64', 'int64']:
                statistic, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
                results = {
                    'test': 'Independent T-Test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
        
        return results
    
    def create_summary_dashboard(self, df: pd.DataFrame) -> List[go.Figure]:
        """Create a comprehensive dashboard with multiple plots"""
        figures = []
        
        # Basic info
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Missing values plot
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig_missing = px.bar(x=missing_data.index, y=missing_data.values,
                               title='Missing Values by Column',
                               labels={'x': 'Columns', 'y': 'Missing Count'})
            figures.append(fig_missing)
        
        # Distribution plots for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Limit to first 10 columns to avoid layout issues
            cols_to_plot = numeric_cols[:10]
            n_cols = len(cols_to_plot)
            
            # Calculate appropriate vertical spacing
            vertical_spacing = max(0.02, 1.0 / (n_cols + 1))
            
            # Create subplot for distributions
            fig_dist = make_subplots(rows=n_cols, cols=1,
                                   subplot_titles=cols_to_plot,
                                   vertical_spacing=vertical_spacing)
            
            for i, col in enumerate(cols_to_plot, 1):
                fig_dist.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col),
                    row=i, col=1
                )
            
            fig_dist.update_layout(height=200 * n_cols, 
                                 title_text="Distribution of Numerical Variables")
            figures.append(fig_dist)
            
            if len(numeric_cols) > 10:
                st.info(f"Showing distributions for first 10 numerical columns. Total numerical columns: {len(numeric_cols)}")
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            corr_fig = self.plot_correlation_matrix(df)
            if corr_fig:
                figures.append(corr_fig)
        
        return figures
    
    def generate_insights(self, df: pd.DataFrame) -> str:
        """Generate AI insights about the data"""
        insights = []
        
        # Basic insights
        insights.append(f"Dataset contains {df.shape[0]} observations and {df.shape[1]} features.")
        
        # Missing data insights
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 10]
        if len(high_missing) > 0:
            insights.append(f"High missing data (>10%) in: {', '.join(high_missing.index)}")
        
        # Numerical insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Numerical variables: {', '.join(numeric_cols)}")
            
            # Check for outliers using IQR
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    insights.append(f"Potential outliers detected in {col}: {len(outliers)} points")
        
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Categorical variables: {', '.join(categorical_cols)}")
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count < 10:
                    insights.append(f"{col} has {unique_count} unique values")
                else:
                    insights.append(f"{col} has many unique values ({unique_count})")
        
        return "\n".join(insights)
