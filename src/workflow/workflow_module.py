import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from enum import Enum
import json
import re

class WorkflowStep(Enum):
    DATA_UPLOAD = "data_upload"
    DATA_EXPLORATION = "data_exploration"
    DATA_PREPROCESSING = "data_preprocessing"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_INTERPRETATION = "model_interpretation"

class WorkflowModule:
    def __init__(self):
        """Initialize the workflow module"""
        self.current_step = WorkflowStep.DATA_UPLOAD
        self.workflow_data = {}
        self.step_descriptions = {
            WorkflowStep.DATA_UPLOAD: "Upload your experimental data (CSV, Excel, or text files)",
            WorkflowStep.DATA_EXPLORATION: "Explore and understand your data structure and characteristics",
            WorkflowStep.DATA_PREPROCESSING: "Clean, transform, and prepare your data for analysis",
            WorkflowStep.STATISTICAL_ANALYSIS: "Perform statistical analysis to understand patterns and relationships",
            WorkflowStep.MODEL_SELECTION: "Choose appropriate machine learning models for your data",
            WorkflowStep.MODEL_TRAINING: "Train your selected models on the prepared data",
            WorkflowStep.MODEL_EVALUATION: "Evaluate model performance using appropriate metrics",
            WorkflowStep.MODEL_INTERPRETATION: "Interpret results and understand model predictions"
        }
        
    def get_current_step_info(self) -> Dict[str, Any]:
        """Get information about the current workflow step"""
        return {
            'step': self.current_step.value,
            'description': self.step_descriptions[self.current_step],
            'step_number': list(WorkflowStep).index(self.current_step) + 1,
            'total_steps': len(WorkflowStep)
        }
    
    def next_step(self):
        """Move to the next workflow step"""
        current_index = list(WorkflowStep).index(self.current_step)
        if current_index < len(WorkflowStep) - 1:
            self.current_step = list(WorkflowStep)[current_index + 1]
    
    def previous_step(self):
        """Move to the previous workflow step"""
        current_index = list(WorkflowStep).index(self.current_step)
        if current_index > 0:
            self.current_step = list(WorkflowStep)[current_index - 1]
    
    def set_step(self, step: WorkflowStep):
        """Set the workflow to a specific step"""
        self.current_step = step
    
    def get_step_guidance(self, step: WorkflowStep) -> str:
        """Get guidance text for a specific workflow step"""
        guidance = {
            WorkflowStep.DATA_UPLOAD: """
            **Data Upload Guidance:**
            - Supported formats: CSV, Excel (.xlsx, .xls), JSON, TXT
            - Ensure your data is properly formatted with headers
            - Check that missing values are handled appropriately
            - Verify data types are correct for your analysis
            """,
            
            WorkflowStep.DATA_EXPLORATION: """
            **Data Exploration Guidance:**
            - Examine data shape and structure
            - Check for missing values and data types
            - Generate summary statistics
            - Create visualizations to understand distributions
            - Identify potential outliers or anomalies
            """,
            
            WorkflowStep.DATA_PREPROCESSING: """
            **Data Preprocessing Guidance:**
            - Handle missing values (imputation or removal)
            - Convert categorical variables to numerical
            - Scale or normalize numerical features
            - Remove outliers if necessary
            - Split data into training and testing sets
            """,
            
            WorkflowStep.STATISTICAL_ANALYSIS: """
            **Statistical Analysis Guidance:**
            - Perform descriptive statistics
            - Check for correlations between variables
            - Conduct hypothesis testing if applicable
            - Analyze distributions and normality
            - Identify significant relationships
            """,
            
            WorkflowStep.MODEL_SELECTION: """
            **Model Selection Guidance:**
            - Consider your problem type (classification, regression, clustering)
            - Evaluate data characteristics (size, features, target distribution)
            - Choose appropriate algorithms based on your goals
            - Consider model interpretability requirements
            - Plan for model comparison and validation
            """,
            
            WorkflowStep.MODEL_TRAINING: """
            **Model Training Guidance:**
            - Use cross-validation for robust evaluation
            - Tune hyperparameters using grid search or random search
            - Monitor for overfitting and underfitting
            - Save trained models for later use
            - Document training parameters and results
            """,
            
            WorkflowStep.MODEL_EVALUATION: """
            **Model Evaluation Guidance:**
            - Use appropriate metrics for your problem type
            - Compare multiple models fairly
            - Analyze confusion matrices for classification
            - Check residual plots for regression
            - Validate results on test data
            """,
            
            WorkflowStep.MODEL_INTERPRETATION: """
            **Model Interpretation Guidance:**
            - Understand feature importance
            - Analyze prediction explanations
            - Identify model limitations
            - Document findings and insights
            - Plan for model deployment and monitoring
            """
        }
        return guidance.get(step, "No guidance available for this step.")
    
    def get_step_questions(self, step: WorkflowStep) -> List[str]:
        """Get relevant questions to ask the user for a specific step"""
        questions = {
            WorkflowStep.DATA_UPLOAD: [
                "What type of experimental data are you working with?",
                "How many samples and features does your dataset have?",
                "Are there any specific data quality issues you're aware of?",
                "What is your target variable or outcome of interest?"
            ],
            
            WorkflowStep.DATA_EXPLORATION: [
                "What patterns or trends do you notice in your data?",
                "Are there any unexpected values or outliers?",
                "How are your variables distributed?",
                "What relationships exist between your features?"
            ],
            
            WorkflowStep.DATA_PREPROCESSING: [
                "How should we handle missing values in your data?",
                "Do you need to scale or normalize your features?",
                "Are there any categorical variables that need encoding?",
                "What percentage of data should be used for training vs testing?"
            ],
            
            WorkflowStep.STATISTICAL_ANALYSIS: [
                "What statistical tests are most relevant to your research question?",
                "Are you looking for correlations, differences, or trends?",
                "What significance level do you want to use?",
                "Do you need to control for multiple comparisons?"
            ],
            
            WorkflowStep.MODEL_SELECTION: [
                "What is your primary goal: prediction, classification, or understanding?",
                "How important is model interpretability to you?",
                "Do you have any preference for specific algorithms?",
                "What are your performance requirements?"
            ],
            
            WorkflowStep.MODEL_TRAINING: [
                "How should we split your data for training and validation?",
                "What hyperparameters should we focus on tuning?",
                "How do you want to handle class imbalance (if applicable)?",
                "What evaluation metrics are most important to you?"
            ],
            
            WorkflowStep.MODEL_EVALUATION: [
                "What performance metrics are most relevant to your application?",
                "How do you want to compare multiple models?",
                "Are there any specific thresholds or criteria for success?",
                "How should we handle uncertainty in the results?"
            ],
            
            WorkflowStep.MODEL_INTERPRETATION: [
                "Which features are most important for your predictions?",
                "How do you want to explain individual predictions?",
                "What insights are you hoping to gain from the model?",
                "How will you use these results in your research?"
            ]
        }
        return questions.get(step, [])
    
    def store_step_data(self, step: WorkflowStep, data: Dict[str, Any]):
        """Store data for a specific workflow step"""
        self.workflow_data[step.value] = data
    
    def get_step_data(self, step: WorkflowStep) -> Optional[Dict[str, Any]]:
        """Retrieve data for a specific workflow step"""
        return self.workflow_data.get(step.value)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workflow state"""
        completed_steps = []
        for step in WorkflowStep:
            if step.value in self.workflow_data:
                completed_steps.append(step.value)
        
        return {
            'current_step': self.current_step.value,
            'completed_steps': completed_steps,
            'total_steps': len(WorkflowStep),
            'progress_percentage': (len(completed_steps) / len(WorkflowStep)) * 100
        }
    
    def convert_recommendations_to_workflow(self, recommendations: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert AI recommendations into structured workflow steps
        
        Args:
            recommendations: AI-generated recommendations text
            df: The dataset being analyzed
            
        Returns:
            Dictionary containing structured workflow steps
        """
        workflow_steps = {
            'data_upload': {
                'completed': True,
                'data_shape': df.shape,
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict()
            },
            'data_exploration': {
                'completed': False,
                'recommendations': self._extract_exploration_recommendations(recommendations),
                'suggested_analyses': []
            },
            'data_preprocessing': {
                'completed': False,
                'recommendations': self._extract_preprocessing_recommendations(recommendations),
                'steps': []
            },
            'statistical_analysis': {
                'completed': False,
                'recommendations': self._extract_statistical_recommendations(recommendations),
                'tests': []
            },
            'model_selection': {
                'completed': False,
                'recommendations': self._extract_model_recommendations(recommendations),
                'suggested_models': []
            },
            'model_training': {
                'completed': False,
                'parameters': {}
            },
            'model_evaluation': {
                'completed': False,
                'metrics': []
            },
            'model_interpretation': {
                'completed': False,
                'focus_areas': []
            }
        }
        
        # Extract specific recommendations for each step
        workflow_steps['data_exploration']['suggested_analyses'] = self._extract_visualization_recommendations(recommendations)
        workflow_steps['data_preprocessing']['steps'] = self._extract_preprocessing_steps(recommendations, df)
        workflow_steps['statistical_analysis']['tests'] = self._extract_statistical_tests(recommendations)
        workflow_steps['model_selection']['suggested_models'] = self._extract_model_suggestions(recommendations)
        
        return workflow_steps
    
    def _extract_exploration_recommendations(self, recommendations: str) -> List[str]:
        """Extract data exploration recommendations from AI text"""
        exploration_keywords = ['explore', 'examine', 'understand', 'investigate', 'analyze', 'visualize']
        recommendations_list = []
        
        lines = recommendations.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in exploration_keywords):
                recommendations_list.append(line.strip())
        
        return recommendations_list[:5]  # Limit to 5 recommendations
    
    def _extract_preprocessing_recommendations(self, recommendations: str) -> List[str]:
        """Extract data preprocessing recommendations from AI text"""
        preprocessing_keywords = ['preprocess', 'clean', 'handle', 'missing', 'normalize', 'scale', 'encode']
        recommendations_list = []
        
        lines = recommendations.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in preprocessing_keywords):
                recommendations_list.append(line.strip())
        
        return recommendations_list[:5]
    
    def _extract_statistical_recommendations(self, recommendations: str) -> List[str]:
        """Extract statistical analysis recommendations from AI text"""
        statistical_keywords = ['statistical', 'test', 'correlation', 'significance', 'hypothesis', 'regression']
        recommendations_list = []
        
        lines = recommendations.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in statistical_keywords):
                recommendations_list.append(line.strip())
        
        return recommendations_list[:5]
    
    def _extract_model_recommendations(self, recommendations: str) -> List[str]:
        """Extract model selection recommendations from AI text"""
        model_keywords = ['model', 'algorithm', 'machine learning', 'classification', 'regression', 'clustering']
        recommendations_list = []
        
        lines = recommendations.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in model_keywords):
                recommendations_list.append(line.strip())
        
        return recommendations_list[:5]
    
    def _extract_visualization_recommendations(self, recommendations: str) -> List[str]:
        """Extract visualization recommendations from AI text"""
        viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'histogram', 'scatter', 'correlation matrix']
        viz_recommendations = []
        
        lines = recommendations.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in viz_keywords):
                viz_recommendations.append(line.strip())
        
        return viz_recommendations[:5]
    
    def _extract_preprocessing_steps(self, recommendations: str, df: pd.DataFrame) -> List[str]:
        """Extract specific preprocessing steps based on data characteristics"""
        steps = []
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct.sum() > 0:
            steps.append("Handle missing values")
        
        # Check for categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            steps.append("Encode categorical variables")
        
        # Check for numerical scaling needs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            steps.append("Scale numerical features")
        
        # Add steps from recommendations
        preprocessing_lines = [line for line in recommendations.split('\n') 
                             if any(keyword in line.lower() for keyword in ['preprocess', 'clean', 'handle'])]
        steps.extend(preprocessing_lines[:3])
        
        return steps
    
    def _extract_statistical_tests(self, recommendations: str) -> List[str]:
        """Extract suggested statistical tests from AI recommendations"""
        tests = []
        
        # Common statistical tests
        test_keywords = {
            'correlation': 'Pearson/Spearman correlation analysis',
            't-test': 'Independent/Paired t-test',
            'anova': 'ANOVA test',
            'chi-square': 'Chi-square test',
            'regression': 'Linear/Logistic regression',
            'normality': 'Normality test (Shapiro-Wilk)'
        }
        
        for keyword, test_name in test_keywords.items():
            if keyword in recommendations.lower():
                tests.append(test_name)
        
        return tests[:5]
    
    def _extract_model_suggestions(self, recommendations: str) -> List[str]:
        """Extract suggested ML models from AI recommendations"""
        models = []
        
        # Common ML models
        model_keywords = {
            'linear': 'Linear Regression/Classification',
            'logistic': 'Logistic Regression',
            'random forest': 'Random Forest',
            'svm': 'Support Vector Machine',
            'neural network': 'Neural Network',
            'clustering': 'K-means Clustering',
            'decision tree': 'Decision Tree'
        }
        
        for keyword, model_name in model_keywords.items():
            if keyword in recommendations.lower():
                models.append(model_name)
        
        return models[:5]


class ToolCallSystem:
    """Interactive tool call system for AI Agent to execute workflow actions"""
    
    def __init__(self, workflow_module: WorkflowModule, analysis_tools):
        self.workflow_module = workflow_module
        self.analysis_tools = analysis_tools
        self.available_tools = self._initialize_tools()
        self.tool_results = {}
        
    def _initialize_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available tools for the AI Agent"""
        return {
            'workflow_status': {
                'description': 'Get current workflow status and progress',
                'function': self._get_workflow_status,
                'parameters': {},
                'example': 'tool: workflow_status'
            },
            'workflow_help': {
                'description': 'Show available workflow tools and commands',
                'function': self._show_workflow_help,
                'parameters': {},
                'example': 'tool: workflow_help'
            },
            'data_analyze': {
                'description': 'Perform basic data analysis and generate insights',
                'function': self._analyze_data,
                'parameters': {},
                'example': 'tool: data_analyze'
            },
            'data_explore': {
                'description': 'Explore data structure and generate visualizations',
                'function': self._explore_data,
                'parameters': {'plot_type': 'histogram', 'columns': 'all'},
                'example': 'tool: data_explore plot_type=histogram columns=all'
            },
            'statistical_test': {
                'description': 'Perform statistical tests on data',
                'function': self._run_statistical_test,
                'parameters': {'test_type': 'normality', 'column': 'required'},
                'example': 'tool: statistical_test test_type=normality column=age'
            },
            'correlation_analysis': {
                'description': 'Analyze correlations between numerical variables',
                'function': self._correlation_analysis,
                'parameters': {},
                'example': 'tool: correlation_analysis'
            },
            'data_preprocessing': {
                'description': 'Get data preprocessing recommendations and status',
                'function': self._data_preprocessing_status,
                'parameters': {},
                'example': 'tool: data_preprocessing'
            },
            'workflow_next': {
                'description': 'Move to next workflow step',
                'function': self._next_workflow_step,
                'parameters': {},
                'example': 'tool: workflow_next'
            },
            'workflow_previous': {
                'description': 'Move to previous workflow step',
                'function': self._previous_workflow_step,
                'parameters': {},
                'example': 'tool: workflow_previous'
            },
            'workflow_set_step': {
                'description': 'Set workflow to specific step',
                'function': self._set_workflow_step,
                'parameters': {'step': 'required'},
                'example': 'tool: workflow_set_step step=data_exploration'
            },
            'generate_insights': {
                'description': 'Generate AI-powered insights about the data',
                'function': self._generate_insights,
                'parameters': {},
                'example': 'tool: generate_insights'
            },
            'train_baseline_model': {
                'description': 'Train a baseline machine learning model (classification or regression)',
                'function': self._train_baseline_model,
                'parameters': {'model_type': 'classification', 'target_column': 'required', 'test_size': '0.2'},
                'example': 'tool: train_baseline_model model_type=classification target_column=target test_size=0.2'
            },
            'evaluate_model': {
                'description': 'Evaluate a trained model performance',
                'function': self._evaluate_model,
                'parameters': {},
                'example': 'tool: evaluate_model'
            },
            'predict_with_model': {
                'description': 'Make predictions using a trained model',
                'function': self._predict_with_model,
                'parameters': {'input_data': 'required'},
                'example': 'tool: predict_with_model input_data=sample_data'
            },
            'model_comparison': {
                'description': 'Compare multiple baseline models',
                'function': self._model_comparison,
                'parameters': {'target_column': 'required', 'test_size': '0.2'},
                'example': 'tool: model_comparison target_column=target test_size=0.2'
            }
        }
    
    def detect_tool_calls(self, user_input: str) -> List[Dict[str, Any]]:
        """Detect tool calls in user input using pattern matching"""
        tool_calls = []
        
        # Pattern: tool: tool_name [parameters]
        tool_pattern = r'tool:\s*(\w+)(?:\s+(.+))?'
        matches = re.findall(tool_pattern, user_input, re.IGNORECASE)
        
        for tool_name, params_str in matches:
            tool_name = tool_name.lower()
            if tool_name in self.available_tools:
                # Parse parameters
                parameters = self._parse_parameters(params_str) if params_str else {}
                tool_calls.append({
                    'tool': tool_name,
                    'parameters': parameters,
                    'original_input': user_input
                })
        
        return tool_calls
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse tool parameters from string input"""
        parameters = {}
        
        # Parse key=value pairs
        param_pattern = r'(\w+)=([^\s]+)'
        matches = re.findall(param_pattern, params_str)
        
        for key, value in matches:
            # Try to convert value to appropriate type
            if value.lower() in ['true', 'false']:
                parameters[key] = value.lower() == 'true'
            elif value.isdigit():
                parameters[key] = int(value)
            elif value.replace('.', '').isdigit():
                parameters[key] = float(value)
            else:
                parameters[key] = value
        
        return parameters
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]], current_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Execute detected tool calls and return results"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['tool']
            parameters = tool_call['parameters']
            
            try:
                if tool_name in self.available_tools:
                    tool_info = self.available_tools[tool_name]
                    tool_function = tool_info['function']
                    
                    # Execute tool function
                    if tool_name in ['data_analyze', 'data_explore', 'correlation_analysis', 'generate_insights', 'statistical_test', 'train_baseline_model', 'model_comparison']:
                        # Tools that need data
                        if current_data is not None:
                            result = tool_function(current_data, **parameters)
                        else:
                            result = {'error': 'No data available. Please upload data first.'}
                    else:
                        # Tools that don't need data
                        result = tool_function(**parameters)
                    
                    results.append({
                        'tool': tool_name,
                        'success': True,
                        'result': result,
                        'parameters': parameters
                    })
                    
                    # Store result for future reference
                    self.tool_results[tool_name] = result
                    
                else:
                    results.append({
                        'tool': tool_name,
                        'success': False,
                        'error': f'Unknown tool: {tool_name}',
                        'parameters': parameters
                    })
                    
            except Exception as e:
                results.append({
                    'tool': tool_name,
                    'success': False,
                    'error': f'Error executing {tool_name}: {str(e)}',
                    'parameters': parameters
                })
        
        return results
    
    def format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool execution results for display"""
        if not results:
            return ""
        
        formatted_results = []
        formatted_results.append("🔧 **Tool Execution Results:**\n")
        
        for result in results:
            tool_name = result['tool']
            if result['success']:
                formatted_results.append(f"✅ **{tool_name}** executed successfully")
                if 'result' in result and result['result']:
                    if isinstance(result['result'], dict):
                        for key, value in result['result'].items():
                            if key != 'error':
                                formatted_results.append(f"   • {key}: {value}")
                    else:
                        formatted_results.append(f"   • Result: {result['result']}")
            else:
                formatted_results.append(f"❌ **{tool_name}** failed: {result.get('error', 'Unknown error')}")
            
            formatted_results.append("")
        
        return "\n".join(formatted_results)
    
    # Tool implementation methods
    def _get_workflow_status(self, **kwargs) -> Dict[str, Any]:
        """Get current workflow status"""
        summary = self.workflow_module.get_workflow_summary()
        current_info = self.workflow_module.get_current_step_info()
        
        return {
            'current_step': current_info['step'],
            'step_description': current_info['description'],
            'progress': f"{summary['progress_percentage']:.1f}%",
            'completed_steps': summary['completed_steps'],
            'total_steps': summary['total_steps']
        }
    
    def _show_workflow_help(self, **kwargs) -> Dict[str, Any]:
        """Show available workflow tools"""
        help_info = {
            'available_tools': list(self.available_tools.keys()),
            'tool_descriptions': {name: info['description'] for name, info in self.available_tools.items()},
            'examples': {name: info['example'] for name, info in self.available_tools.items()}
        }
        return help_info
    
    def _analyze_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform basic data analysis"""
        stats = self.analysis_tools.basic_statistics(df)
        insights = self.analysis_tools.generate_insights(df)
        
        return {
            'data_shape': stats['shape'],
            'columns': stats['columns'],
            'missing_values': stats['missing_values'],
            'insights': insights
        }
    
    def _explore_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Explore data structure and generate visualizations"""
        plot_type = kwargs.get('plot_type', 'histogram')
        columns = kwargs.get('columns', 'all')
        
        if columns == 'all':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns.split(',') if col in df.columns]
        
        exploration_info = {
            'plot_type': plot_type,
            'available_numeric_columns': numeric_cols,
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'suggestion': f"Use 'tool: plot_distribution column={numeric_cols[0] if numeric_cols else 'N/A'} plot_type={plot_type}' to create plots"
        }
        
        return exploration_info
    
    def _run_statistical_test(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run statistical tests on data"""
        test_type = kwargs.get('test_type', 'normality')
        column = kwargs.get('column')
        
        if not column:
            return {'error': 'Column parameter is required'}
        
        if column not in df.columns:
            return {'error': f'Column {column} not found in dataset'}
        
        if test_type == 'normality':
            result = self.analysis_tools.statistical_tests(df, column, None, 'normality')
        elif test_type == 'correlation':
            # For correlation, we need two columns
            return {'error': 'Correlation test requires two columns. Use correlation_analysis tool instead.'}
        else:
            return {'error': f'Unsupported test type: {test_type}'}
        
        return result
    
    def _correlation_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze correlations between numerical variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numerical columns for correlation analysis'}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'variables': f"{numeric_cols[i]} vs {numeric_cols[j]}",
                        'correlation': round(corr_value, 3),
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        return {
            'numerical_columns': numeric_cols.tolist(),
            'strong_correlations': strong_correlations,
            'suggestion': 'Use correlation matrix visualization to see all correlations'
        }
    
    def _data_preprocessing_status(self, **kwargs) -> Dict[str, Any]:
        """Get data preprocessing recommendations and status"""
        current_step = self.workflow_module.current_step
        step_guidance = self.workflow_module.get_step_guidance(current_step)
        step_questions = self.workflow_module.get_step_questions(current_step)
        
        return {
            'current_step': current_step.value,
            'guidance': step_guidance,
            'questions_to_consider': step_questions,
            'next_steps': 'Consider moving to next step or addressing current step questions'
        }
    
    def _next_workflow_step(self, **kwargs) -> Dict[str, Any]:
        """Move to next workflow step"""
        self.workflow_module.next_step()
        current_info = self.workflow_module.get_current_step_info()
        
        return {
            'action': 'Moved to next step',
            'new_step': current_info['step'],
            'description': current_info['description'],
            'step_number': current_info['step_number']
        }
    
    def _previous_workflow_step(self, **kwargs) -> Dict[str, Any]:
        """Move to previous workflow step"""
        self.workflow_module.previous_step()
        current_info = self.workflow_module.get_current_step_info()
        
        return {
            'action': 'Moved to previous step',
            'new_step': current_info['step'],
            'description': current_info['description'],
            'step_number': current_info['step_number']
        }
    
    def _set_workflow_step(self, **kwargs) -> Dict[str, Any]:
        """Set workflow to specific step"""
        step_name = kwargs.get('step')
        if not step_name:
            return {'error': 'Step parameter is required'}
        
        try:
            step_enum = WorkflowStep(step_name)
            self.workflow_module.set_step(step_enum)
            current_info = self.workflow_module.get_current_step_info()
            
            return {
                'action': f'Set workflow to step: {step_name}',
                'current_step': current_info['step'],
                'description': current_info['description'],
                'step_number': current_info['step_number']
            }
        except ValueError:
            return {'error': f'Invalid step: {step_name}. Available steps: {[s.value for s in WorkflowStep]}'}
    
    def _generate_insights(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Generate AI-powered insights about the data"""
        insights = self.analysis_tools.generate_insights(df)
        
        return {
            'insights': insights,
            'data_summary': f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns",
            'suggestion': 'Use these insights to guide your analysis workflow'
        }
    
    def _train_baseline_model(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train a baseline machine learning model"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
            import numpy as np
            
            model_type = kwargs.get('model_type', 'classification')
            target_column = kwargs.get('target_column')
            test_size = float(kwargs.get('test_size', 0.2))
            
            if not target_column:
                return {'error': 'target_column parameter is required'}
            
            if target_column not in df.columns:
                return {'error': f'Target column {target_column} not found in dataset'}
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                le = LabelEncoder()
                for col in categorical_cols:
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if y.dtype == 'object' else y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if model_type == 'classification':
                # Check if binary or multiclass
                if len(y.unique()) == 2:
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            if model_type == 'classification':
                metrics = {
                    'accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                    'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                    'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4)
                }
            else:
                metrics = {
                    'r2_score': round(r2_score(y_test, y_pred), 4),
                    'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    'mae': round(mean_squared_error(y_test, y_pred, squared=False), 4)
                }
            
            # Store model in session state for later use
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            
            model_info = {
                'model': model,
                'scaler': scaler,
                'model_type': model_type,
                'target_column': target_column,
                'feature_columns': list(X.columns),
                'metrics': metrics,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            model_name = f"{model_type}_{target_column}_{len(st.session_state.trained_models)}"
            st.session_state.trained_models[model_name] = model_info
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_type,
                'target_column': target_column,
                'features_used': len(X.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': metrics,
                'suggestion': f'Use "tool: evaluate_model" to get detailed evaluation or "tool: predict_with_model" to make predictions'
            }
            
        except Exception as e:
            return {'error': f'Error training model: {str(e)}'}
    
    def _evaluate_model(self, **kwargs) -> Dict[str, Any]:
        """Evaluate a trained model performance"""
        try:
            if 'trained_models' not in st.session_state or not st.session_state.trained_models:
                return {'error': 'No trained models found. Train a model first using "tool: train_baseline_model"'}
            
            # Get the most recent model
            model_name = list(st.session_state.trained_models.keys())[-1]
            model_info = st.session_state.trained_models[model_name]
            
            model = model_info['model']
            model_type = model_info['model_type']
            metrics = model_info['metrics']
            X_test = model_info['X_test']
            y_test = model_info['y_test']
            y_pred = model_info['y_pred']
            
            # Additional evaluation metrics
            if model_type == 'classification':
                from sklearn.metrics import classification_report, confusion_matrix
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                return {
                    'model_name': model_name,
                    'model_type': model_type,
                    'basic_metrics': metrics,
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'suggestion': 'Use "tool: predict_with_model" to make new predictions'
                }
            else:
                from sklearn.metrics import mean_absolute_error
                mae = mean_absolute_error(y_test, y_pred)
                
                return {
                    'model_name': model_name,
                    'model_type': model_type,
                    'basic_metrics': metrics,
                    'mae': round(mae, 4),
                    'suggestion': 'Use "tool: predict_with_model" to make new predictions'
                }
                
        except Exception as e:
            return {'error': f'Error evaluating model: {str(e)}'}
    
    def _predict_with_model(self, **kwargs) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            if 'trained_models' not in st.session_state or not st.session_state.trained_models:
                return {'error': 'No trained models found. Train a model first using "tool: train_baseline_model"'}
            
            # Get the most recent model
            model_name = list(st.session_state.trained_models.keys())[-1]
            model_info = st.session_state.trained_models[model_name]
            
            model = model_info['model']
            scaler = model_info['scaler']
            feature_columns = model_info['feature_columns']
            
            # For now, return sample prediction info
            # In a real implementation, you'd get input data from kwargs
            return {
                'model_name': model_name,
                'model_type': model_info['model_type'],
                'target_column': model_info['target_column'],
                'features_required': feature_columns,
                'sample_prediction': 'Use the model to predict on new data',
                'suggestion': 'Provide input data in the format matching the training features'
            }
            
        except Exception as e:
            return {'error': f'Error making predictions: {str(e)}'}
    
    def _model_comparison(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Compare multiple baseline models"""
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            import numpy as np
            
            target_column = kwargs.get('target_column')
            test_size = float(kwargs.get('test_size', 0.2))
            
            if not target_column:
                return {'error': 'target_column parameter is required'}
            
            if target_column not in df.columns:
                return {'error': f'Target column {target_column} not found in dataset'}
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                le = LabelEncoder()
                for col in categorical_cols:
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if y.dtype == 'object' else y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Determine if classification or regression
            is_classification = len(y.unique()) < 10  # Simple heuristic
            
            if is_classification:
                models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'SVM': SVC(random_state=42)
                }
                scoring = 'accuracy'
            else:
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'SVR': SVR()
                }
                scoring = 'r2'
            
            # Compare models using cross-validation
            results = {}
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=scoring)
                    results[name] = {
                        'mean_score': round(scores.mean(), 4),
                        'std_score': round(scores.std(), 4)
                    }
                except:
                    results[name] = {'mean_score': 'N/A', 'std_score': 'N/A'}
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x]['mean_score'] if results[x]['mean_score'] != 'N/A' else -1)
            
            return {
                'model_comparison': results,
                'best_model': best_model_name,
                'best_score': results[best_model_name]['mean_score'],
                'task_type': 'Classification' if is_classification else 'Regression',
                'suggestion': f'Best model is {best_model_name}. Use "tool: train_baseline_model model_type={"classification" if is_classification else "regression"} target_column={target_column}" to train it.'
            }
            
        except Exception as e:
            return {'error': f'Error comparing models: {str(e)}'}

