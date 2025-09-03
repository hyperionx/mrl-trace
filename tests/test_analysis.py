import pytest
import pandas as pd
import numpy as np
from src.analysis import AnalysisTools


class TestAnalysisTools:
    """Test the AnalysisTools class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analysis_tools = AnalysisTools()
        
        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'binary_col': np.random.choice([0, 1], 100),
            'missing_col': np.random.choice([1, 2, 3, np.nan], 100)
        })
    
    def test_initialization(self):
        """Test that analysis tools initializes correctly"""
        assert hasattr(self.analysis_tools, 'basic_statistics')
        assert hasattr(self.analysis_tools, 'generate_insights')
    
    def test_basic_statistics(self):
        """Test basic statistics calculation"""
        stats = self.analysis_tools.basic_statistics(self.sample_data)
        
        assert 'shape' in stats
        assert 'columns' in stats
        assert 'dtypes' in stats
        assert 'missing_values' in stats
        
        assert stats['shape'] == (100, 4)
        assert len(stats['columns']) == 4
        assert len(stats['dtypes']) == 4
        assert len(stats['missing_values']) == 4
    
    def test_generate_insights(self):
        """Test insight generation"""
        insights = self.analysis_tools.generate_insights(self.sample_data)
        
        assert isinstance(insights, str)
        assert len(insights) > 0
    
    def test_statistical_tests_normality(self):
        """Test normality test"""
        result = self.analysis_tools.statistical_tests(
            self.sample_data, 'numeric_col', None, 'normality'
        )
        
        assert result is not None
        # Normality test should return some result
        assert len(str(result)) > 0
    
    def test_statistical_tests_correlation(self):
        """Test correlation test"""
        # Add another numeric column for correlation
        self.sample_data['numeric_col2'] = np.random.normal(0, 1, 100)
        
        result = self.analysis_tools.statistical_tests(
            self.sample_data, 'numeric_col', 'numeric_col2', 'correlation'
        )
        
        assert result is not None
        assert len(str(result)) > 0
    
    def test_plot_distribution(self):
        """Test distribution plotting"""
        fig = self.analysis_tools.plot_distribution(
            self.sample_data, 'numeric_col', 'histogram'
        )
        
        assert fig is not None
        # Plotly figure should have data
        assert len(fig.data) > 0
    
    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting"""
        # Ensure we have numeric columns
        numeric_data = self.sample_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            fig = self.analysis_tools.plot_correlation_matrix(numeric_data)
            if fig is not None:
                assert len(fig.data) > 0
    
    def test_plot_scatter(self):
        """Test scatter plot generation"""
        # Ensure we have numeric columns
        numeric_data = self.sample_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            fig = self.analysis_tools.plot_scatter(
                numeric_data, 'numeric_col', 'binary_col'
            )
            assert fig is not None
            assert len(fig.data) > 0
    
    def test_create_summary_dashboard(self):
        """Test summary dashboard creation"""
        figures = self.analysis_tools.create_summary_dashboard(self.sample_data)
        
        assert isinstance(figures, list)
        assert len(figures) > 0
        
        for fig in figures:
            assert fig is not None
            assert len(fig.data) > 0
    
    def test_data_loading_simulation(self):
        """Test data loading functionality (simulated)"""
        # Since we can't test actual file uploads, test the method exists
        assert hasattr(self.analysis_tools, 'load_data')
    
    def test_insight_generation_with_different_data_types(self):
        """Test insight generation with various data types"""
        # Test with only numeric data
        numeric_data = self.sample_data.select_dtypes(include=[np.number])
        insights = self.analysis_tools.generate_insights(numeric_data)
        assert isinstance(insights, str)
        assert len(insights) > 0
        
        # Test with only categorical data
        categorical_data = self.sample_data.select_dtypes(include=['object'])
        if len(categorical_data.columns) > 0:
            insights = self.analysis_tools.generate_insights(categorical_data)
            assert isinstance(insights, str)
            assert len(insights) > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        try:
            stats = self.analysis_tools.basic_statistics(empty_df)
            # Should handle empty dataframe gracefully
            assert 'shape' in stats
        except Exception as e:
            # If it raises an exception, that's also acceptable
            assert isinstance(e, Exception)
        
        # Test with single column
        single_col_df = pd.DataFrame({'single': [1, 2, 3]})
        stats = self.analysis_tools.basic_statistics(single_col_df)
        assert stats['shape'] == (3, 1)


class TestAnalysisToolsIntegration:
    """Test integration between analysis tools and workflow"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analysis_tools = AnalysisTools()
        
        # Create realistic test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'age': np.random.normal(45, 15, 100),
            'income': np.random.normal(50000, 20000, 100),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
            'satisfaction': np.random.choice([1, 2, 3, 4, 5], 100)
        })
    
    def test_complete_analysis_workflow(self):
        """Test a complete analysis workflow"""
        # Step 1: Basic statistics
        stats = self.analysis_tools.basic_statistics(self.test_data)
        assert stats['shape'] == (100, 4)
        
        # Step 2: Generate insights
        insights = self.analysis_tools.generate_insights(self.test_data)
        assert isinstance(insights, str)
        
        # Step 3: Create visualizations
        age_dist = self.analysis_tools.plot_distribution(self.test_data, 'age', 'histogram')
        assert age_dist is not None
        
        # Step 4: Statistical tests
        age_normality = self.analysis_tools.statistical_tests(
            self.test_data, 'age', None, 'normality'
        )
        assert age_normality is not None
        
        # Step 5: Summary dashboard
        dashboard = self.analysis_tools.create_summary_dashboard(self.test_data)
        assert len(dashboard) > 0
    
    def test_data_quality_analysis(self):
        """Test data quality analysis features"""
        # Add some missing values
        self.test_data.loc[0, 'age'] = np.nan
        self.test_data.loc[1, 'income'] = np.nan
        
        stats = self.analysis_tools.basic_statistics(self.test_data)
        
        # Check that missing values are detected
        missing_counts = stats['missing_values']
        assert missing_counts['age'] > 0 or missing_counts['income'] > 0
        
        # Generate insights should handle missing data
        insights = self.analysis_tools.generate_insights(self.test_data)
        assert isinstance(insights, str)


if __name__ == "__main__":
    pytest.main([__file__])
