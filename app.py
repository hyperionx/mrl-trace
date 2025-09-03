import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os
from xai_sdk import Client
from xai_sdk.chat import user, system
from src.workflow import WorkflowModule, WorkflowStep, ToolCallSystem
from src.analysis import AnalysisTools
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import subprocess
import psutil
import json
import sys
import logging
from datetime import datetime
import random

# Import utility classes and functions
from src.utils import (
    ChatProcessor, BackendManager, SessionManager, ChatInterface, TestingTools,
    setup_logging, validate_api_key, is_api_key_ready, DEFAULT_PROMPT_TEMPLATES,
    SYNTHETIC_QUERIES, SYNTHETIC_RESPONSES
)

# Import async backend components
try:
    from src.backend.async_client import AsyncBackendClient, RequestPriority
    ASYNC_BACKEND_AVAILABLE = True
except ImportError:
    ASYNC_BACKEND_AVAILABLE = False
    st.warning("Async backend not available. Install requirements-async.txt for full functionality.")

# Global utility instances
backend_manager = BackendManager()
session_manager = SessionManager()
chat_interface = None
testing_tools = TestingTools()

# Initialize logger
logger = setup_logging()

def log_user_interaction(query: str, response: str, template_used: str, response_time: float = None):
    """Log user interactions for analysis and validation"""
    try:
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "template_used": template_used,
            "response_time": response_time,
            "session_id": session_manager.user_session_id if session_manager.user_session_id else "none"
        }
        
        logger.info(f"User Interaction: {json.dumps(interaction_data, indent=2)}")
        
        # Also log to a structured JSON file for analysis
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        interactions_file = f"{log_dir}/interactions.json"
        
        # Load existing interactions or create new file
        if os.path.exists(interactions_file):
            with open(interactions_file, 'r') as f:
                try:
                    interactions = json.load(f)
                except json.JSONDecodeError:
                    interactions = []
        else:
            interactions = []
        
        # Add new interaction
        interactions.append(interaction_data)
        
        # Save updated interactions
        with open(interactions_file, 'w') as f:
            json.dump(interactions, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

def submit_async_request(request_type: str, **kwargs):
    """Submit a request to the async backend"""
    if not session_manager.async_backend_enabled or not session_manager.async_client or not session_manager.user_session_id:
        return None
    
    try:
        async def submit_request():
            async with session_manager.async_client:
                if request_type == 'chat':
                    return await session_manager.async_client.submit_chat(
                        session_manager.user_session_id, 
                        kwargs.get('user_id', 'default_user'),
                        kwargs.get('message', ''),
                        kwargs.get('priority', RequestPriority.NORMAL),
                        kwargs.get('timeout', 30.0)
                    )
                elif request_type == 'tool':
                    return await session_manager.async_client.execute_tool(
                        session_manager.user_session_id,
                        kwargs.get('user_id', 'default_user'),
                        kwargs.get('tool_name', ''),
                        kwargs.get('parameters', {}),
                        kwargs.get('priority', RequestPriority.NORMAL),
                        kwargs.get('timeout', 60.0)
                    )
                elif request_type == 'analysis':
                    return await session_manager.async_client.submit_analysis(
                        session_manager.user_session_id,
                        kwargs.get('user_id', 'default_user'),
                        kwargs.get('analysis_type', ''),
                        kwargs.get('parameters', {}),
                        kwargs.get('priority', RequestPriority.NORMAL),
                        kwargs.get('timeout', 120.0)
                    )
                else:
                    return None
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, submit_request())
            return future.result(timeout=30)
            
    except Exception as e:
        st.error(f"❌ Async request failed: {e}")
        return None

def get_request_status(request_id: str):
    """Get the status of an async request"""
    if not session_manager.async_backend_enabled or not session_manager.async_client:
        return None
    
    try:
        async def check_status():
            async with session_manager.async_client:
                return await session_manager.async_client.get_request_status(request_id)
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, check_status())
            return future.result(timeout=10)
            
    except Exception as e:
        st.error(f"❌ Status check failed: {e}")
        return None

def await_sync_chat(grok_client, enhanced_prompt: str, data_context: str, prompt_template: str = "general_assistant") -> str:
    """Execute synchronous chat with Grok client using tuned prompts"""
    try:
        # Use Grok to generate response
        chat = grok_client.chat.create(model="grok-4")
        
        # Safely get chat history and prompt templates
        chat_history = st.session_state.get("chat_history", [])
        prompt_templates = st.session_state.get('prompt_templates', DEFAULT_PROMPT_TEMPLATES)
        current_data = st.session_state.get('current_data', None)
        
        # Add previous context from chat history (limit to last 10 messages to avoid token limits)
        recent_messages = chat_history[-10:-1] if len(chat_history) > 1 else []
        for msg in recent_messages:
            if msg["role"] == "user":
                chat.append(user(msg["content"]))
            else:
                chat.append(system(msg["content"]))
        
        # Get the selected prompt template
        template = prompt_templates.get(
            prompt_template, 
            DEFAULT_PROMPT_TEMPLATES["general_assistant"]
        )
        
        # Build enhanced system prompt with template and data context
        # Note: Don't add conflicting instructions that override the template
        system_prompt = f"{template['system_prompt']} Data Context: {data_context if current_data is not None else 'No data loaded'}."
        
        chat.append(system(system_prompt))
        chat.append(user(enhanced_prompt))
        
        # Generate response with error handling
        response = chat.sample()
        if response and hasattr(response, 'content'):
            return response.content
        else:
            return "I apologize, but I couldn't generate a response. Please try again."
        
    except Exception as e:
        st.error(f"❌ Chat failed: {e}")
        return f"I apologize, but I encountered an error: {str(e)}. Please check your API key and try again."

def extract_tool_calls_from_response(response_text: str) -> List[str]:
    """Extract tool calls from AI response text"""
    tool_calls = []
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('`tool:') and line.endswith('`'):
            # Extract the tool call from backticks
            tool_call = line[1:-1]  # Remove backticks
            tool_calls.append(tool_call)
        elif line.startswith('- `tool:') and line.endswith('`'):
            # Extract the tool call from markdown list format
            tool_call = line[3:-1]  # Remove "- `" and "`"
            tool_calls.append(tool_call)
    
    return tool_calls

# Tool call buttons are no longer needed as tools are executed automatically
# when AI actions are accepted

def _render_tool_output(tool_name: str, result: Any):
    """Render actual tool functionality based on tool type"""
    # Handle the case where result is a list of execution results
    if isinstance(result, list) and len(result) > 0:
        # Take the first result if it's a list
        result = result[0]
    
    # Check if the result has the expected structure
    if isinstance(result, dict):
        # Check if success is at the top level
        if 'success' in result and result['success']:
            # Extract the actual tool result from the execution results
            tool_result = result.get('result', {})
            
            if tool_name == 'data_explore':
                _render_data_explore_output(tool_result)
            elif tool_name == 'data_analyze':
                _render_data_analyze_output(tool_result)
            elif tool_name == 'statistical_test':
                _render_statistical_test_output(tool_result)
            elif tool_name == 'correlation_analysis':
                _render_correlation_analysis_output(tool_result)
            elif tool_name == 'generate_insights':
                _render_insights_output(tool_result)
            else:
                # Fallback to generic display
                st.success("✅ Tool executed successfully!")
                st.json(tool_result)
        else:
            # Handle error cases
            if isinstance(result, str):
                st.error(result)
            else:
                st.error("Tool execution failed")
                st.json(result)
    else:
        # Handle other result types
        if isinstance(result, str):
            st.info(result)
        else:
            st.json(result)

def _render_data_explore_output(tool_result: dict):
    """Render data exploration tool output with actual plots"""
    import uuid
    
    # Generate a unique identifier for this function call
    unique_id = str(uuid.uuid4())[:8]
    
    st.success("✅ Data exploration completed!")
    
    # Display available columns info
    if 'available_numeric_columns' in tool_result:
        st.write("**📊 Available Numeric Columns:**")
        if tool_result['available_numeric_columns']:
            st.write(tool_result['available_numeric_columns'])
        else:
            st.info("No numeric columns available for plotting")
    
    if 'categorical_columns' in tool_result:
        st.write("**🏷️ Categorical Columns:**")
        st.write(tool_result['categorical_columns'])
    
    # Generate and display actual plots if we have data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        # Create sample plots for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**📈 Sample Visualizations:**")
            
            # Create a histogram for the first numeric column
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Histogram:**")
                try:
                    import plotly.express as px
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True, key=f"histogram_{numeric_cols[0]}_{unique_id}")
                except Exception as e:
                    st.error(f"Could not create histogram: {e}")
            
            with col2:
                st.write("**Box Plot:**")
                try:
                    import plotly.express as px
                    fig = px.box(df, y=numeric_cols[0], title=f"Box Plot of {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True, key=f"boxplot_{numeric_cols[0]}_{unique_id}")
                except Exception as e:
                    st.error(f"Could not create box plot: {e}")
            
            # Create correlation heatmap if we have multiple numeric columns
            if len(numeric_cols) > 1:
                st.write("**🔥 Correlation Heatmap:**")
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    corr_matrix = df[numeric_cols].corr()
                    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns))
                    fig.update_layout(title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True, key=f"correlation_heatmap_{unique_id}")
                except Exception as e:
                    st.error(f"Could not create correlation heatmap: {e}")
        
        # Handle categorical columns if no numeric columns available
        elif 'categorical_columns' in tool_result and tool_result['categorical_columns']:
            st.write("**📊 Categorical Data Analysis:**")
            
            # Create bar chart for categorical columns
            categorical_cols = tool_result['categorical_columns']
            if categorical_cols:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Bar Chart:**")
                    try:
                        import plotly.express as px
                        # Count occurrences of each category
                        category_counts = df[categorical_cols[0]].value_counts()
                        fig = px.bar(x=category_counts.index, y=category_counts.values, 
                                   title=f"Distribution of {categorical_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{categorical_cols[0]}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create bar chart: {e}")
                
                with col2:
                    st.write("**Pie Chart:**")
                    try:
                        import plotly.express as px
                        # Count occurrences of each category
                        category_counts = df[categorical_cols[0]].value_counts()
                        fig = px.pie(values=category_counts.values, names=category_counts.index, 
                                   title=f"Distribution of {categorical_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{categorical_cols[0]}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create pie chart: {e}")
        
        # Show data preview
        st.write("**📋 Data Preview:**")
        st.dataframe(df.head(10))
        
        # Show basic statistics
        st.write("**📊 Basic Statistics:**")
        st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"Data types: {dict(df.dtypes)}")
        
        # Show missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**❓ Missing Values:**")
            st.write(missing_data[missing_data > 0])
        else:
            st.success("✅ No missing values found!")

def _render_data_analyze_output(tool_result: dict):
    """Render data analysis tool output with enhanced visualizations and insights"""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    st.success("✅ Data analysis completed!")
    
    # Display basic analysis results
    if 'data_shape' in tool_result:
        st.write(f"**📏 Dataset Shape:** {tool_result['data_shape']}")
    
    if 'columns' in tool_result:
        st.write("**📋 Columns:**")
        st.write(tool_result['columns'])
    
    if 'missing_values' in tool_result:
        st.write("**❓ Missing Values:**")
        st.write(tool_result['missing_values'])
    
    if 'insights' in tool_result:
        st.write("**💡 AI-Generated Insights:**")
        st.write(tool_result['insights'])
    
    # Create enhanced visualizations if we have data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        st.write("**📊 Enhanced Data Analysis Visualizations:**")
        
        # Get column information
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Show comprehensive data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Rows", f"{df.shape[0]:,}")
        
        with col2:
            st.metric("📋 Columns", f"{df.shape[1]:,}")
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("❓ Missing", f"{missing_pct:.1f}%")
        
        with col4:
            st.metric("🔢 Numeric", f"{len(numeric_cols)}")
        
        # Show data quality metrics
        if len(df.columns) > 0:
            st.write("**🔍 Data Quality Metrics:**")
            
            # Calculate data quality metrics
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completeness = ((total_cells - missing_cells) / total_cells) * 100
                st.metric("📊 Completeness", f"{completeness:.1f}%")
            
            with col2:
                uniqueness = ((df.shape[0] - duplicate_rows) / df.shape[0]) * 100
                st.metric("🆔 Uniqueness", f"{uniqueness:.1f}%")
            
            with col3:
                st.metric("📈 Data Types", f"{len(df.dtypes.unique())}")
        
        # Show column type distribution
        if len(df.columns) > 0:
            st.write("**🔍 Column Type Analysis:**")
            
            dtype_counts = df.dtypes.value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    import plotly.express as px
                    fig = px.pie(
                        values=dtype_counts.values,
                        names=dtype_counts.index.astype(str),
                        title="Column Types Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"analyze_dtype_pie_{unique_id}")
                except Exception as e:
                    st.error(f"Could not create column type pie chart: {e}")
            
            with col2:
                st.write("**Column Type Details:**")
                for dtype, count in dtype_counts.items():
                    st.write(f"• **{dtype}:** {count} column(s)")
        
        # Show missing values analysis
        if df.isnull().sum().sum() > 0:
            st.write("**❓ Missing Values Analysis:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Missing values by column
                    missing_by_col = df.isnull().sum()
                    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
                    
                    if len(missing_by_col) > 0:
                        fig = px.bar(
                            x=missing_by_col.index,
                            y=missing_by_col.values,
                            title="Missing Values by Column"
                        )
                        fig.update_layout(
                            xaxis_title="Columns",
                            yaxis_title="Missing Count"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"analyze_missing_bar_{unique_id}")
                
                except Exception as e:
                    st.error(f"Could not create missing values bar chart: {e}")
            
            with col2:
                try:
                    # Missing values heatmap
                    missing_data = df.isnull().astype(int)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=missing_data.values,
                        x=missing_data.columns,
                        y=list(range(len(missing_data))),  # Convert range to list
                        colorscale='Reds',
                        showscale=True
                    ))
                    
                    fig.update_layout(
                        title="Missing Values Pattern",
                        xaxis_title="Columns",
                        yaxis_title="Row Index",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"analyze_missing_heatmap_{unique_id}")
                
                except Exception as e:
                    st.error(f"Could not create missing values heatmap: {e}")
        
        # Show numeric data analysis
        if len(numeric_cols) > 0:
            st.write("**🔢 Numeric Data Analysis:**")
            
            # Show descriptive statistics
            st.write("**📊 Descriptive Statistics:**")
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats, use_container_width=True)
            
            # Show distribution plots for first few numeric columns
            st.write("**📈 Distribution Analysis:**")
            
            # Limit to first 3 numeric columns to avoid too many plots
            display_cols = numeric_cols[:3]
            
            for col in display_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        import plotly.express as px
                        fig = px.histogram(
                            df, x=col,
                            title=f"Distribution of {col}",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"analyze_hist_{col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create histogram for {col}: {e}")
                
                with col2:
                    try:
                        import plotly.express as px
                        fig = px.box(
                            df, y=col,
                            title=f"Box Plot of {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"analyze_box_{col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create box plot for {col}: {e}")
        
        # Show categorical data analysis
        if len(categorical_cols) > 0:
            st.write("**🏷️ Categorical Data Analysis:**")
            
            # Limit to first 2 categorical columns
            display_cat_cols = categorical_cols[:2]
            
            for col in display_cat_cols:
                try:
                    # Count values and show top categories
                    value_counts = df[col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart of top categories
                        top_categories = value_counts.head(15)
                        fig = px.bar(
                            x=top_categories.index,
                            y=top_categories.values,
                            title=f"Category Distribution in {col}"
                        )
                        fig.update_layout(
                            xaxis_title="Categories",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"analyze_bar_{col}_{unique_id}")
                    
                    with col2:
                        # Show category statistics
                        st.write(f"**{col} Analysis:**")
                        st.write(f"• **Unique values:** {len(value_counts)}")
                        st.write(f"• **Most common:** {value_counts.index[0]} ({value_counts.iloc[0]} times)")
                        st.write(f"• **Least common:** {value_counts.index[-1]} ({value_counts.iloc[-1]} times)")
                        
                        # Show top 10 categories
                        st.write("**Top 10 Categories:**")
                        for i, (category, count) in enumerate(value_counts.head(10).items()):
                            st.write(f"{i+1}. **{category}:** {count}")
                
                except Exception as e:
                    st.error(f"Could not analyze categorical column {col}: {e}")
    
    # Display all other results
    st.write("**📋 Complete Analysis Results:**")
    for key, value in tool_result.items():
        if key not in ['data_shape', 'columns', 'missing_values', 'insights']:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def _render_statistical_test_output(tool_result: dict):
    """Render statistical test tool output with actual statistical analysis"""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    # Check if there was an error in the tool execution
    if 'error' in tool_result:
        st.error(f"❌ **Statistical Test Failed:** {tool_result['error']}")
        
        # Provide helpful guidance based on the error
        if 'Column parameter is required' in tool_result['error']:
            st.info("""
            **🔧 How to fix this error:**
            
            The `statistical_test` tool requires a `column` parameter to specify which column to test.
            
            **Correct usage examples:**
            - `tool: statistical_test test_type=normality column=age`
            - `tool: statistical_test test_type=normality column=perimeter_mean`
            - `tool: statistical_test test_type=normality column=diagnosis`
            
            **Available columns in your dataset:**""")
            
            # Show available columns if we have data
            if 'current_data' in st.session_state and st.session_state.current_data is not None:
                df = st.session_state.current_data
                st.write("**📋 Available Columns:**")
                
                # Show numeric columns (recommended for statistical tests)
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.write("**🔢 Numeric Columns (Recommended for tests):**")
                    for col in numeric_cols:
                        st.write(f"• `{col}`")
                
                # Show categorical columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    st.write("**🏷️ Categorical Columns:**")
                    for col in categorical_cols:
                        st.write(f"• `{col}`")
                
                st.info("""
                **💡 Tip:** For normality tests, use numeric columns. For categorical data, consider using other analysis tools.
                """)
        
        elif 'Column' in tool_result['error'] and 'not found' in tool_result['error']:
            st.info("""
            **🔧 How to fix this error:**
            
            The specified column was not found in your dataset. Please check the column name spelling and case sensitivity.
            """)
        
        elif 'Unsupported test type' in tool_result['error']:
            st.info("""
            **🔧 How to fix this error:**
            
            The specified test type is not supported. Currently supported test types:
            - `normality` - Tests if data follows normal distribution (Shapiro-Wilk test)
            
            **Correct usage:**
            `tool: statistical_test test_type=normality column=your_column_name`
            """)
        
        # Show the complete error for debugging
        st.write("**📋 Complete Error Details:**")
        st.json(tool_result)
        return
    
    # If no error, proceed with normal rendering
    st.success("✅ Statistical test completed!")
    
    # Display test results in a structured way
    if 'test_type' in tool_result:
        st.write(f"**📊 Test Type:** {tool_result['test_type']}")
    
    if 'test_statistic' in tool_result:
        st.write(f"**🔢 Test Statistic:** {tool_result['test_statistic']:.4f}")
    
    if 'p_value' in tool_result:
        # Color code p-value based on significance
        p_value = tool_result['p_value']
        if p_value < 0.001:
            st.write(f"**🎯 P-Value:** {p_value:.6f} (*** Highly Significant)")
        elif p_value < 0.01:
            st.write(f"**🎯 P-Value:** {p_value:.6f} (** Very Significant)")
        elif p_value < 0.05:
            st.write(f"**🎯 P-Value:** {p_value:.6f} (* Significant)")
        else:
            st.write(f"**🎯 P-Value:** {p_value:.6f} (Not Significant)")
    
    if 'degrees_of_freedom' in tool_result:
        st.write(f"**📐 Degrees of Freedom:** {tool_result['degrees_of_freedom']}")
    
    if 'sample_size' in tool_result:
        st.write(f"**👥 Sample Size:** {tool_result['sample_size']}")
    
    if 'effect_size' in tool_result:
        effect_size = tool_result['effect_size']
        st.write(f"**📏 Effect Size:** {effect_size:.4f}")
        
        # Interpret effect size
        if abs(effect_size) < 0.1:
            st.info("💡 Effect Size: Negligible")
        elif abs(effect_size) < 0.3:
            st.info("💡 Effect Size: Small")
        elif abs(effect_size) < 0.5:
            st.info("💡 Effect Size: Medium")
        else:
            st.info("💡 Effect Size: Large")
    
    # Create visualizations if we have data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        # Show data distribution for the tested variable
        if 'tested_column' in tool_result and tool_result['tested_column'] in df.columns:
            tested_col = tool_result['tested_column']
            
            if df[tested_col].dtype in ['int64', 'float64']:
                st.write("**📈 Data Distribution Visualization:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        import plotly.express as px
                        fig = px.histogram(df, x=tested_col, title=f"Distribution of {tested_col}")
                        st.plotly_chart(fig, use_container_width=True, key=f"stat_test_hist_{tested_col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create histogram: {e}")
                
                with col2:
                    try:
                        import plotly.express as px
                        fig = px.box(df, y=tested_col, title=f"Box Plot of {tested_col}")
                        st.plotly_chart(fig, use_container_width=True, key=f"stat_test_box_{tested_col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create box plot: {e}")
        
        # Show Q-Q plot for normality test if applicable
        if 'test_type' in tool_result and 'normality' in tool_result['test_type'].lower():
            st.write("**📊 Normality Check (Q-Q Plot):**")
            try:
                import plotly.graph_objects as go
                import numpy as np
                from scipy import stats
                
                # Get numeric data
                numeric_data = df.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 0:
                    col = numeric_data.columns[0]
                    data = df[col].dropna()
                    
                    # Create Q-Q plot
                    theoretical_quantiles = stats.probplot(data, dist="norm")[0]
                    sample_quantiles = stats.probplot(data, dist="norm")[1]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Data Points'
                    ))
                    
                    # Add reference line
                    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Normal',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"Q-Q Plot for {col}",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"qq_plot_{col}_{unique_id}")
            except Exception as e:
                st.error(f"Could not create Q-Q plot: {e}")
    
    # Show interpretation
    st.write("**💡 Statistical Interpretation:**")
    if 'p_value' in tool_result and 'test_statistic' in tool_result:
        p_value = tool_result['p_value']
        if p_value < 0.05:
            st.success("✅ **Result:** The test result is statistically significant (p < 0.05)")
            st.info("This suggests that the observed effect is unlikely to have occurred by chance alone.")
        else:
            st.warning("⚠️ **Result:** The test result is not statistically significant (p ≥ 0.05)")
            st.info("This suggests that the observed effect could have occurred by chance.")
    
    # Display all other results
    st.write("**📋 Complete Test Results:**")
    for key, value in tool_result.items():
        if key not in ['test_type', 'test_statistic', 'p_value', 'degrees_of_freedom', 'sample_size', 'effect_size', 'tested_column']:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def _render_correlation_analysis_output(tool_result: dict):
    """Render correlation analysis tool output with actual correlation visualizations"""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    st.success("✅ Correlation analysis completed!")
    
    if 'numerical_columns' in tool_result:
        st.write("**📊 Analyzed Columns:**")
        st.write(tool_result['numerical_columns'])
    
    if 'strong_correlations' in tool_result:
        st.write("**🔗 Strong Correlations:**")
        for corr in tool_result['strong_correlations']:
            st.write(f"• {corr['variables']}: {corr['correlation']} ({corr['strength']})")
    
    # Create correlation heatmap if we have data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        # Get numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            st.write("**🔥 Correlation Heatmap:**")
            try:
                import plotly.graph_objects as go
                import numpy as np
                
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 3),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title="Correlation Matrix Heatmap",
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    width=600,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"corr_heatmap_{unique_id}")
                
                # Add correlation interpretation
                st.write("**💡 Correlation Interpretation:**")
                st.info("""
                - **Red**: Strong positive correlation (close to +1)
                - **Blue**: Strong negative correlation (close to -1)
                - **White**: No correlation (close to 0)
                - **Darker colors**: Stronger correlations
                """)
                
            except Exception as e:
                st.error(f"Could not create correlation heatmap: {e}")
        
        # Show scatter plots for strong correlations
        if 'strong_correlations' in tool_result and len(tool_result['strong_correlations']) > 0:
            st.write("**📈 Strong Correlation Scatter Plots:**")
            
            # Limit to first 4 strong correlations to avoid too many plots
            strong_corrs = tool_result['strong_correlations'][:4]
            
            for i, corr in enumerate(strong_corrs):
                try:
                    # Parse variables (assuming format like "var1 vs var2" or "var1, var2")
                    if ' vs ' in corr['variables']:
                        var1, var2 = corr['variables'].split(' vs ')
                    elif ', ' in corr['variables']:
                        var1, var2 = corr['variables'].split(', ')
                    else:
                        # Try to extract from the string
                        vars_list = [col for col in numeric_cols if col in corr['variables']]
                        if len(vars_list) >= 2:
                            var1, var2 = vars_list[0], vars_list[1]
                        else:
                            continue
                    
                    var1 = var1.strip()
                    var2 = var2.strip()
                    
                    if var1 in df.columns and var2 in df.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            import plotly.express as px
                            fig = px.scatter(
                                df, x=var1, y=var2,
                                title=f"{var1} vs {var2}",
                                trendline="ols"
                            )
                            fig.update_layout(
                                xaxis_title=var1,
                                yaxis_title=var2
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"scatter_{var1}_{var2}_{unique_id}")
                        
                        with col2:
                            # Show correlation details
                            st.write(f"**Correlation Details:**")
                            st.write(f"**Variables:** {var1} ↔ {var2}")
                            st.write(f"**Correlation:** {corr['correlation']:.4f}")
                            st.write(f"**Strength:** {corr['strength']}")
                            
                            # Interpret correlation strength
                            corr_val = abs(corr['correlation'])
                            if corr_val >= 0.8:
                                st.success("**Very Strong** correlation")
                            elif corr_val >= 0.6:
                                st.info("**Strong** correlation")
                            elif corr_val >= 0.4:
                                st.warning("**Moderate** correlation")
                            else:
                                st.info("**Weak** correlation")
                
                except Exception as e:
                    st.error(f"Could not create scatter plot for correlation: {e}")
    
    # Show correlation summary
    if 'strong_correlations' in tool_result:
        st.write("**📊 Correlation Summary:**")
        
        # Count correlations by strength
        strength_counts = {}
        for corr in tool_result['strong_correlations']:
            strength = corr['strength']
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        for strength, count in strength_counts.items():
            st.write(f"• **{strength}:** {count} correlation(s)")
    
    # Display all other results
    st.write("**📋 Complete Analysis Results:**")
    for key, value in tool_result.items():
        if key not in ['numerical_columns', 'strong_correlations']:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def _render_insights_output(tool_result: dict):
    """Render insights tool output with enhanced data visualization and analysis"""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    
    st.success("✅ Insights generated!")
    
    # Display AI-generated insights
    if 'insights' in tool_result:
        st.write("**💡 AI-Generated Insights:**")
        
        # Handle different insight formats
        if isinstance(tool_result['insights'], list):
            for i, insight in enumerate(tool_result['insights']):
                st.write(f"**{i+1}.** {insight}")
        elif isinstance(tool_result['insights'], str):
            # Split by newlines or periods for better formatting
            insights = tool_result['insights'].replace('\n', ' ').split('. ')
            for i, insight in enumerate(insights):
                if insight.strip():
                    st.write(f"**{i+1}.** {insight.strip()}")
        else:
            st.write(tool_result['insights'])
    
    # Display data summary
    if 'data_summary' in tool_result:
        st.write("**📊 Data Summary:**")
        st.write(tool_result['data_summary'])
    
    # Create enhanced visualizations if we have data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        st.write("**📈 Enhanced Data Visualizations:**")
        
        # Get column information
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Show data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📏 Total Rows", f"{df.shape[0]:,}")
        
        with col2:
            st.metric("📋 Total Columns", f"{df.shape[1]:,}")
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("❓ Missing Data", f"{missing_pct:.1f}%")
        
        # Show data types distribution
        if len(df.columns) > 0:
            st.write("**🔍 Data Types Distribution:**")
            
            dtype_counts = df.dtypes.value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    import plotly.express as px
                    fig = px.pie(
                        values=dtype_counts.values,
                        names=dtype_counts.index.astype(str),
                        title="Data Types Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"dtype_pie_{unique_id}")
                except Exception as e:
                    st.error(f"Could not create data type pie chart: {e}")
            
            with col2:
                st.write("**Data Type Details:**")
                for dtype, count in dtype_counts.items():
                    st.write(f"• **{dtype}:** {count} column(s)")
        
        # Show missing values heatmap
        if df.isnull().sum().sum() > 0:
            st.write("**❓ Missing Values Pattern:**")
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create missing values heatmap
                missing_data = df.isnull().astype(int)
                
                fig = go.Figure(data=go.Heatmap(
                    z=missing_data.values,
                    x=missing_data.columns,
                    y=list(range(len(missing_data))),  # Convert range to list
                    colorscale='Reds',
                    showscale=True
                ))
                
                fig.update_layout(
                    title="Missing Values Pattern (Red = Missing)",
                    xaxis_title="Columns",
                    yaxis_title="Row Index",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"missing_heatmap_{unique_id}")
                
                # Show missing values summary
                missing_summary = df.isnull().sum()
                missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
                
                if len(missing_summary) > 0:
                    st.write("**Missing Values by Column:**")
                    for col, missing_count in missing_summary.items():
                        missing_pct = (missing_count / len(df)) * 100
                        st.write(f"• **{col}:** {missing_count} ({missing_pct:.1f}%)")
                
            except Exception as e:
                st.error(f"Could not create missing values heatmap: {e}")
        
        # Show distribution of first few numeric columns
        if len(numeric_cols) > 0:
            st.write("**📊 Numeric Data Distributions:**")
            
            # Limit to first 4 numeric columns to avoid too many plots
            display_cols = numeric_cols[:4]
            
            for i, col in enumerate(display_cols):
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        import plotly.express as px
                        fig = px.histogram(
                            df, x=col,
                            title=f"Distribution of {col}",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"insight_hist_{col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create histogram for {col}: {e}")
                
                with col2:
                    try:
                        import plotly.express as px
                        fig = px.box(
                            df, y=col,
                            title=f"Box Plot of {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"insight_box_{col}_{unique_id}")
                    except Exception as e:
                        st.error(f"Could not create box plot for {col}: {e}")
        
        # Show categorical data insights
        if len(categorical_cols) > 0:
            st.write("**🏷️ Categorical Data Insights:**")
            
            # Limit to first 3 categorical columns
            display_cat_cols = categorical_cols[:3]
            
            for col in display_cat_cols:
                try:
                    # Count values and show top categories
                    value_counts = df[col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart of top categories
                        top_categories = value_counts.head(10)
                        fig = px.bar(
                            x=top_categories.index,
                            y=top_categories.values,
                            title=f"Top Categories in {col}"
                        )
                        fig.update_layout(
                            xaxis_title="Categories",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"insight_bar_{col}_{unique_id}")
                    
                    with col2:
                        # Show category statistics
                        st.write(f"**{col} Statistics:**")
                        st.write(f"• **Unique values:** {len(value_counts)}")
                        st.write(f"• **Most common:** {value_counts.index[0]} ({value_counts.iloc[0]} times)")
                        st.write(f"• **Least common:** {value_counts.index[-1]} ({value_counts.iloc[-1]} times)")
                        
                        # Show top 5 categories
                        st.write("**Top 5 Categories:**")
                        for i, (category, count) in enumerate(value_counts.head().items()):
                            st.write(f"{i+1}. **{category}:** {count}")
                
                except Exception as e:
                    st.error(f"Could not analyze categorical column {col}: {e}")
    
    # Display all other results
    st.write("**📋 Complete Insights Results:**")
    for key, value in tool_result.items():
        if key not in ['insights', 'data_summary']:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def create_user_session(user_id: str, api_key: str):
    """Create a user session with the async backend"""
    return session_manager.create_user_session(user_id, api_key)

def main():
    # Set page configuration first (must be first Streamlit command)
    st.set_page_config(page_title="AI Agent for Experimental Scientists", layout="wide")
    
    # Load environment variables from .env file
    load_dotenv()

    # Get Grok API key from environment as fallback
    GROK_API_KEY_ENV = os.getenv("API_KEY")

    # Initialize session state for API key
    if 'api_key' not in st.session_state:
        st.session_state.api_key = GROK_API_KEY_ENV or ""

    # Configure Grok client only if API key is available and valid
    grok_client = None
    model = None
    
    if st.session_state.api_key:
        # Simple validation check
        is_valid, error_message = validate_api_key(st.session_state.api_key)
        
        if is_valid:
            # API key is valid, create the client
            grok_client = Client(
                api_key=st.session_state.api_key,
                timeout=3600,  # Longer timeout for actual operations
            )
            model = grok_client.chat.create(model="grok-4")

    # Sidebar for API key configuration
    with st.sidebar:
        st.title("🔧 Configuration")
        
        # Initialize modules first (needed for chat interface)
        if 'workflow_module' not in st.session_state:
            st.session_state.workflow_module = WorkflowModule()

        if 'analysis_tools' not in st.session_state:
            st.session_state.analysis_tools = AnalysisTools()

        if 'tool_call_system' not in st.session_state:
            st.session_state.tool_call_system = ToolCallSystem(
                st.session_state.workflow_module, 
                st.session_state.analysis_tools
            )

        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        

        
        # Data Upload Section
        with st.expander("📁 Upload Your Data", expanded=False):
            uploaded_file = st.file_uploader("Upload your dataset for AI analysis", type=['csv', 'xlsx', 'xls', 'json'])
            
            if uploaded_file is not None:
                # Load data
                df = st.session_state.analysis_tools.load_data(uploaded_file)
                if df is not None:
                    st.session_state.current_data = df
                    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    st.info(f"📊 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
                    
                    # Show basic data info
                    st.write("**Columns:**")
                    st.write(", ".join(df.columns.tolist()[:10]) + ("..." if len(df.columns) > 10 else ""))
                    
                    # Show data types
                    st.write("**Data Types:**")
                    dtypes_list = list(df.dtypes.items())
                    for col, dtype in dtypes_list[:5]:
                        st.write(f"- {col}: {dtype}")
                    if len(dtypes_list) > 5:
                        st.write(f"... and {len(dtypes_list) - 5} more columns")
                else:
                    st.error("❌ Failed to load data. Please check your file format.")
            else:
                st.info("💡 Upload a dataset to get started with AI analysis")
        
        # ⚡ Async Backend Status (at the top of sidebar)
        with st.expander("⚡ Async Backend Status", expanded=False):
            if ASYNC_BACKEND_AVAILABLE:
                # Initialize async backend state
                if 'async_backend_initialized' not in st.session_state:
                    st.session_state.async_backend_initialized = False
                
                # Check if backend is running
                backend_running = backend_manager.check_backend_running()
                
                # Determine the current status and show only one message
                if backend_running and st.session_state.async_backend_initialized:
                    status_message = "✅ Async backend is running and connected!"
                    status_type = "success"
                elif backend_running and not st.session_state.async_backend_initialized:
                    status_message = "⚠️ Backend running but async client not connected. Using synchronous mode."
                    status_type = "warning"
                else:
                    status_message = "❌ Async backend is not running"
                    status_type = "error"
                
                # Display the single status message
                if status_type == "success":
                    st.success(status_message)
                elif status_type == "warning":
                    st.warning(status_message)
                else:
                    st.error(status_message)
                
                # Action buttons based on status
                if backend_running and st.session_state.async_backend_initialized:
                    # Backend is connected - show stats and debug options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Backend Stats", key="backend_stats_btn"):
                            debug_info = backend_manager.debug_backend_status()
                            st.json(debug_info)
                    with col2:
                        if st.button("🔍 Debug Backend", key="debug_backend_btn"):
                            st.info("Backend is healthy and responding to requests.")
                
                elif backend_running and not st.session_state.async_backend_initialized:
                    # Backend running but not connected - try to connect automatically first
                    if 'auto_connect_attempted' not in st.session_state:
                        st.session_state.auto_connect_attempted = False
                    
                    if not st.session_state.auto_connect_attempted:
                        # Try automatic connection first
                        st.info("🚀 Backend detected, attempting automatic connection...")
                        with st.spinner("Connecting to async backend..."):
                            success = session_manager.initialize_async_backend()
                            if success:
                                st.session_state.async_backend_initialized = True
                                st.success("✅ Async backend connected automatically!")
                                st.rerun()
                            else:
                                st.session_state.auto_connect_attempted = True
                                st.warning("⚠️ Automatic connection failed, manual connection required")
                    
                    # Show manual connection options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Connect Client", key="connect_client_btn"):
                            with st.spinner("Connecting to async backend..."):
                                success = session_manager.initialize_async_backend()
                                if success:
                                    st.session_state.async_backend_initialized = True
                                    st.success("✅ Async client connected successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to connect async client")
                                    st.info("💡 Try the 'Test Connection' button to verify connectivity")
                    with col2:
                        if st.button("🧪 Test Connection", key="test_connection_btn"):
                            with st.expander("🔍 Connection Test Results", expanded=True):
                                st.write("**Testing async client connection...**")
                                try:
                                    async def test_connection():
                                        try:
                                            async with AsyncBackendClient() as client:
                                                # Test root endpoint
                                                async with client.session.get(f"{client.base_url}/") as response:
                                                    root_status = response.status
                                                    root_text = await response.text()
                                                
                                                # Test health endpoint
                                                health_response = await client.health_check()
                                                health_success = health_response.success
                                                health_data = health_response.data
                                                
                                                return {
                                                    "root_endpoint": {"status": root_status, "response": root_text[:100]},
                                                    "health_endpoint": {"success": health_success, "data": health_data},
                                                    "connection_success": True
                                                }
                                        except Exception as e:
                                            return {"connection_success": False, "error": str(e)}
                                    
                                    # Run the test
                                    with ThreadPoolExecutor() as executor:
                                        future = executor.submit(asyncio.run, test_connection())
                                        test_result = future.result(timeout=20)
                                    
                                    st.json(test_result)
                                    
                                    if test_result.get("connection_success"):
                                        st.success("✅ Async client connection test successful!")
                                        # Automatically update the state since the test succeeded
                                        st.session_state.async_backend_initialized = True
                                        st.success("🔄 **State Updated**: Async client is now connected!")
                                        st.info("💡 The app will refresh automatically to show the connected status.")
                                        # Force a rerun to update the UI
                                        time.sleep(1)  # Brief delay to show the success message
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Async client connection test failed: {test_result.get('error')}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Test failed: {e}")
                
                else:
                    # Backend not running - show start options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🚀 Start Backend", key="start_backend_btn"):
                            with st.spinner("Starting async backend..."):
                                success = backend_manager.start_backend_process()
                                if success:
                                    st.success("✅ Backend started successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to start backend")
                    with col2:
                        if st.button("📊 Check Status", key="check_status_btn"):
                            st.info("Backend is not running. Use 'Start Backend' to begin.")
            
            else:
                st.warning("⚠️ **Async Backend**: Not available")
                st.info("💡 Install async dependencies with 'uv sync --extra async' for enhanced performance")

        # Testing and Analytics Section 
        with st.expander("🧪 Testing & Analytics", expanded=False):
            # Synthetic data generation
            st.write("**📊 Generate Test Data**")
            st.write("Generate synthetic user interactions for testing system robustness")
            
            col1, col2 = st.columns(2)
            with col1:
                num_samples = st.number_input("Number of samples", min_value=1, max_value=20, value=5, key="synthetic_samples")
            
            with col2:
                if st.button("🎲 Generate Data", key="generate_synthetic_btn"):
                    with st.spinner("Generating synthetic data..."):
                        synthetic_data = testing_tools.generate_synthetic_data(num_samples)
                        st.success(f"✅ Generated {len(synthetic_data)} synthetic interactions")
                        
                        # Display sample data
                        st.write("**Sample Generated Data:**")
                        for i, data in enumerate(synthetic_data[:3], 1):
                            st.write(f"{i}. **Query:** {data['query'][:50]}...")
                            st.write(f"   **Template:** {data['template_used']}")
                            st.write(f"   **Response Time:** {data['response_time']:.2f}s")
                            st.write("---")
            
            st.markdown("---")
            
            # System simulation
            st.write("**🔄 System Simulation**")
            st.write("Simulate multiple user interactions to test system robustness")
            
            col1, col2 = st.columns(2)
            with col1:
                num_interactions = st.number_input("Interactions to simulate", min_value=5, max_value=50, value=10, key="simulation_interactions")
            
            with col2:
                if st.button("🚀 Run Simulation", key="run_simulation_btn"):
                    with st.spinner("Running system simulation..."):
                        simulated_data = testing_tools.simulate_user_interactions(num_interactions)
                        st.success(f"✅ Simulation completed with {len(simulated_data)} interactions")
                        st.info("💡 Check the logs folder for detailed interaction data")
            
            st.markdown("---")
            
            # Analytics dashboard
            st.write("**📈 Analytics Dashboard**")
            st.write("View system performance and user interaction analytics")
            
            # Check if interactions file exists
            interactions_file = "logs/interactions.json"
            if os.path.exists(interactions_file):
                try:
                    with open(interactions_file, 'r') as f:
                        interactions = json.load(f)
                    
                    if interactions:
                        st.write(f"**Total Interactions:** {len(interactions)}")
                        
                        # Calculate statistics
                        response_times = [i.get('response_time', 0) for i in interactions if i.get('response_time')]
                        templates_used = [i.get('template_used', 'unknown') for i in interactions]
                        
                        if response_times:
                            avg_response_time = sum(response_times) / len(response_times)
                            st.write(f"**Average Response Time:** {avg_response_time:.2f}s")
                        
                        # Template usage analysis
                        template_counts = {}
                        for template in templates_used:
                            template_counts[template] = template_counts.get(template, 0) + 1
                        
                        st.write("**Template Usage:**")
                        for template, count in template_counts.items():
                            st.write(f"- {template}: {count} times")
                        
                        # Recent interactions
                        st.write("**Recent Interactions:**")
                        recent = interactions[-5:] if len(interactions) > 5 else interactions
                        for i, interaction in enumerate(recent, 1):
                            st.write(f"{i}. {interaction['query'][:40]}... → {interaction['template_used']}")
                    
                except Exception as e:
                    st.error(f"❌ Failed to load analytics: {e}")
            else:
                st.info("📊 No interaction data available yet. Start chatting to generate analytics!")
                
        # API Key Management
        with st.expander("🔑 API Key Configuration", expanded=False):
            api_key_input = st.text_input(
                "Grok API Key",
                value=st.session_state.api_key,
                type="password",
                help="Enter your Grok API key. You can get one from https://console.x.ai/",
                key="api_key_input"
            )
            
            # Update session state when API key changes
            if api_key_input != st.session_state.api_key:
                st.session_state.api_key = api_key_input
                st.rerun()
            
            # Show API key status
            if st.session_state.api_key:
                # Simple validation check
                is_valid, error_message = validate_api_key(st.session_state.api_key)
                
                if is_valid:
                    st.success("✅ API key valid and ready")
                    st.info("🚀 AI features are now enabled!")
                else:
                    st.error(f"🔑 **API Key Issue**: {error_message}")
                    
                    # Show specific help based on error type
                    if "Invalid API key" in error_message:
                        st.info("💡 **Help**: Double-check your API key from https://console.x.ai/")
                    elif "no credits" in error_message:
                        st.info("💡 **Help**: Visit https://console.x.ai/ to add credits to your account")
                    elif "Permission denied" in error_message:
                        st.info("💡 **Help**: Check your API key permissions in the x.ai console")
                    else:
                        st.info("💡 **Help**: Please check your API key and try again")
            else:
                st.warning("⚠️ No API key provided")
                st.info("💡 AI features will be disabled without a valid API key")
            
            # Show management buttons
            if st.session_state.api_key:
                if st.button("🗑️ Clear Key", key="clear_key_btn"):
                    st.session_state.api_key = ""
                    st.rerun()
        
        # Prompt Tuning Configuration
        st.markdown("---")
        st.header("🎯 Prompt Tuning")
        
        # Initialize prompt templates in session state
        if 'prompt_templates' not in st.session_state:
            st.session_state.prompt_templates = DEFAULT_PROMPT_TEMPLATES.copy()
        
        if 'selected_prompt_template' not in st.session_state:
            st.session_state.selected_prompt_template = "general_assistant"
        
        # Prompt template selector
        template_options = {k: v['name'] for k, v in st.session_state.prompt_templates.items()}
        selected_template = st.selectbox(
            "Select AI Personality:",
            options=list(template_options.keys()),
            format_func=lambda x: template_options[x],
            key="prompt_template_selector"
        )
        
        if selected_template != st.session_state.selected_prompt_template:
            st.session_state.selected_prompt_template = selected_template
            st.rerun()
        
        # Show current template description
        current_template = st.session_state.prompt_templates[selected_template]
        st.info(f"**{current_template['name']}**: {current_template['description']}")
        
        # Custom prompt editor - only show when Custom is selected
        if selected_template == "custom":
            with st.expander("✏️ Customize Prompt Template", expanded=False):
                st.write("Modify the system prompt for the selected template:")
                
                custom_prompt = st.text_area(
                    "System Prompt:",
                    value=current_template['system_prompt'],
                    height=200,
                    key=f"custom_prompt_{selected_template}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Changes", key=f"save_prompt_{selected_template}"):
                        st.session_state.prompt_templates[selected_template]['system_prompt'] = custom_prompt
                        st.success("✅ Prompt template updated!")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Reset to Default", key=f"reset_prompt_{selected_template}"):
                        st.session_state.prompt_templates[selected_template] = DEFAULT_PROMPT_TEMPLATES[selected_template].copy()
                        st.success("✅ Prompt template reset to default!")
                        st.rerun()
            
                # Export/Import functionality
                st.markdown("---")
                st.write("**Export/Import Templates:**")
                
                # Export row
                if st.button("📤 Export Templates", key="export_templates", use_container_width=True):
                    templates_json = json.dumps(st.session_state.prompt_templates, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=templates_json,
                        file_name="prompt_templates.json",
                        mime="application/json",
                        use_container_width=True
                        )
                
                # Import row
                uploaded_templates = st.file_uploader(
                    "Import Templates",
                    type=['json'],
                    key="import_templates"
                )
                
                if uploaded_templates is not None:
                    try:
                        imported_templates = json.load(uploaded_templates)
                        if st.button("📥 Import Templates", key="import_templates_btn", use_container_width=True):
                            st.session_state.prompt_templates.update(imported_templates)
                            st.success("✅ Templates imported successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to parse templates: {e}")
        
        # 💬 AI Chat Interface - At the bottom of sidebar
        # Create chat interface in sidebar
        global chat_interface
        if chat_interface is None:
            chat_interface = ChatInterface(grok_client, st.session_state.tool_call_system, st.session_state.get('current_data'))
        
        # Always show chat interface at the bottom
        chat_interface.create_sidebar_chat(
            use_async_mode=st.session_state.get('use_async_mode', False),
            async_backend_enabled=session_manager.async_backend_enabled,
            user_session_id=session_manager.user_session_id
        )

    st.title("AI Agent for Experimental Scientists")
    st.write("This assistant helps you analyze data and create custom analysis tools through AI chat interactions.")
    
    # Show API key status in main area
    if not st.session_state.api_key:
        st.warning("🔑 **API Key Required**: Please configure your Grok API key in the sidebar to enable AI features.")
    elif grok_client is not None:
        st.success("✅ **AI Features Enabled**: Your API key is valid and ready to use.")
    else:
        st.error("❌ **API Key Issue**: Please check your API key in the sidebar.")
    
    # Main Dashboard
    st.markdown("---")
    st.header("🎯 AI-Powered Analysis Dashboard")
    st.write("Simple, fast, and effective data analysis tools.")
    
    # Check if data is available
    if st.session_state.current_data is None:
        st.warning("⚠️ No data uploaded. Please upload data above to get started.")
        st.info("💡 **Getting Started:**\n"
                "1. Upload your dataset using the sidebar\n"
                "2. Chat with the AI agent for analysis recommendations\n"
                "3. Use the simple tools for quick analysis")
    else:
        df = st.session_state.current_data
        analysis_tools = st.session_state.analysis_tools
        
        # Tabbed Dashboard Interface
        st.subheader("📊 Dashboard Tabs")
        
        # Initialize dynamic tabs in session state
        if 'ai_generated_tabs' not in st.session_state:
            st.session_state.ai_generated_tabs = []
        
        # Create base tabs
        base_tabs = ["📊 Data Overview"]
        
        # Add AI-generated tabs
        if st.session_state.ai_generated_tabs:
            base_tabs.extend([tab['name'] for tab in st.session_state.ai_generated_tabs])
        
        # Create tabs
        tabs = st.tabs(base_tabs)
        
        # Data Overview Tab
        with tabs[0]:
            st.subheader("📊 Data Overview")
            
            # Data metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            # Quick Actions
            st.subheader("🚀 Quick Actions")
            
            # Data Exploration
            if st.button("📊 Explore Data", key="explore_data"):
                st.subheader("📊 Data Exploration Results")
                
                # Basic info
                st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                
                # Data types
                st.write("**Data Types:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
                
                # Missing values
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    st.write("**Missing Values:**")
                    for col, missing_count in missing_data.items():
                        if missing_count > 0:
                            st.write(f"- {col}: {missing_count} missing values")
                else:
                    st.success("✅ No missing values found!")
                
                # Sample data
                st.write("**Sample Data (first 5 rows):**")
                st.dataframe(df.head())
            
            # Statistical Summary
            if st.button("📈 Statistical Summary", key="stat_summary"):
                st.subheader("📈 Statistical Summary")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.write("**Numerical Columns Summary:**")
                    st.dataframe(df[numeric_cols].describe())
                    
                    # Correlation matrix for numerical columns
                    if len(numeric_cols) >= 2:
                        st.write("**Correlation Matrix:**")
                        corr_matrix = df[numeric_cols].corr()
                        st.dataframe(corr_matrix)
                else:
                    st.info("No numerical columns found for statistical analysis.")
            
            # Data Cleaning
            if st.button("🧹 Data Cleaning", key="data_cleaning"):
                st.subheader("🧹 Data Cleaning Tools")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Remove Missing Values", key="remove_missing"):
                        df_clean = df.dropna()
                        st.session_state.current_data = df_clean
                        st.success(f"✅ Removed {len(df) - len(df_clean)} rows with missing values")
                        st.rerun()
                
                with col2:
                    if st.button("Fill Missing with Mean", key="fill_mean"):
                        df_filled = df.copy()
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        for col in numeric_cols:
                            if df_filled[col].isnull().sum() > 0:
                                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
                        st.session_state.current_data = df_filled
                        st.success("✅ Filled missing values with column means")
                        st.rerun()
            
            # Simple Visualization
            st.subheader("📊 Quick Visualizations")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    plot_col = st.selectbox("Select column for plot:", numeric_cols, key="plot_col")
                    plot_type = st.selectbox("Plot type:", ['histogram', 'box'], key="plot_type")
                    
                    if st.button("Generate Plot", key="gen_plot"):
                        try:
                            fig = analysis_tools.plot_distribution(df, plot_col, plot_type)
                            st.plotly_chart(fig, use_container_width=True, key=f"distribution_plot_{plot_col}_{plot_type}_{i}_{j}")
                        except Exception as e:
                            st.error(f"Plot generation failed: {e}")
                
                with col2:
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
                        y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
                        
                        if st.button("Generate Scatter Plot", key="gen_scatter"):
                            try:
                                scatter_fig = analysis_tools.plot_scatter(df, x_col, y_col, None)
                                st.plotly_chart(scatter_fig, use_container_width=True, key=f"scatter_plot_{x_col}_{y_col}_{i}_{j}")
                            except Exception as e:
                                st.error(f"Scatter plot generation failed: {e}")
                    else:
                        st.info("Need at least 2 numerical columns for scatter plots.")
            else:
                st.info("No numerical columns found for visualization.")
        
        # AI-Generated Tabs
        for i, tab_content in enumerate(st.session_state.ai_generated_tabs):
            with tabs[i + 1]:  # +1 because first tab is Data Overview
                st.subheader(f"🤖 {tab_content['name']}")
                st.write(tab_content.get('description', ''))
                
                # Display content based on tab type
                if tab_content.get('content_type') == 'data_analysis':
                    st.write("**🔍 Data Analysis Tools:**")
                    
                    # Show AI recommendation context
                    if tab_content.get('ai_source'):
                        with st.expander("📋 AI Recommendation Context", expanded=False):
                            st.write(tab_content.get('ai_source', ''))
                    
                    # Check if tools were already executed
                    if tab_content.get('tools_executed') and tab_content.get('executed_results'):
                        for j, result in enumerate(tab_content['executed_results']):
                            # Get tool name for the expander title
                            tool_name = tab_content.get('tool_names', [])[j] if tab_content.get('tool_names') and j < len(tab_content['tool_names']) else f"Tool {j+1}"
                            with st.expander(f"🔧 {tool_name} Results", expanded=False):
                                if isinstance(result, str):
                                    st.info(result)
                                else:
                                    # Render actual tool functionality based on tool type
                                    _render_tool_output(tool_name, result)
                    else:
                        # Show extracted tools as functional buttons (fallback)
                        extracted_tools = tab_content.get('extracted_tools', [])
                        if extracted_tools:
                            st.write("**🔧 AI-Recommended Tools to Execute:**")
                            for j, tool_call in enumerate(extracted_tools):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.code(tool_call, language="bash")
                                with col2:
                                    if st.button(f"Execute", key=f"execute_tool_{i}_{j}"):
                                        # Execute the tool call using the tool call system
                                        detected_tools = st.session_state.tool_call_system.detect_tool_calls(tool_call)
                                        if detected_tools:
                                            with st.spinner("🔧 Executing tool..."):
                                                results = st.session_state.tool_call_system.execute_tool_calls(
                                                    detected_tools, 
                                                    st.session_state.current_data
                                                )
                                                
                                                # Format and display tool results
                                                tool_results = st.session_state.tool_call_system.format_tool_results(results)
                                                
                                                # Add tool results to chat history
                                                st.session_state["chat_history"].append({
                                                    "role": "assistant", 
                                                    "content": f"Tool Execution Results:\n\n{tool_results}"
                                                })
                                                
                                                st.success("Tool executed successfully!")
                                                st.rerun()
                                        else:
                                            st.error("Invalid tool call format")
                        else:
                            st.info("No specific tools extracted from AI recommendation.")
                    
                    # Show the custom content if available
                    if tab_content.get('custom_content'):
                        st.markdown("---")
                        st.markdown(tab_content.get('custom_content', ''))
                
                elif tab_content.get('content_type') == 'statistical_testing':
                    st.write("**📊 Statistical Testing Tools:**")
                    
                    # Show AI recommendation context
                    if tab_content.get('ai_source'):
                        with st.expander("📋 AI Recommendation Context", expanded=False):
                            st.write(tab_content.get('ai_source', ''))
                    
                    # Check if tools were already executed
                    if tab_content.get('tools_executed') and tab_content.get('executed_results'):
                        for j, result in enumerate(tab_content['executed_results']):
                            # Get tool name for the expander title
                            tool_name = tab_content.get('tool_names', [])[j] if tab_content.get('tool_names') and j < len(tab_content['tool_names']) else f"Tool {j+1}"
                            with st.expander(f"🔧 {tool_name} Results", expanded=False):
                                if isinstance(result, str):
                                    st.info(result)
                                else:
                                    # Render actual tool functionality based on tool type
                                    _render_tool_output(tool_name, result)
                    else:
                        # Show extracted tools as functional buttons (fallback)
                        extracted_tools = tab_content.get('extracted_tools', [])
                        if extracted_tools:
                            st.write("**🔧 AI-Recommended Tools to Execute:**")
                            for j, tool_call in enumerate(extracted_tools):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.code(tool_call, language="bash")
                                with col2:
                                    if st.button(f"Execute", key=f"execute_tool_{i}_{j}"):
                                        # Execute the tool call using the tool call system
                                        detected_tools = st.session_state.tool_call_system.detect_tool_calls(tool_call)
                                        if detected_tools:
                                            with st.spinner("🔧 Executing tool..."):
                                                results = st.session_state.tool_call_system.execute_tool_calls(
                                                    detected_tools, 
                                                    st.session_state.current_data
                                                )
                                                
                                                # Format and display tool results
                                                tool_results = st.session_state.tool_call_system.format_tool_results(results)
                                            
                                                # Add tool results to chat history
                                                st.session_state["chat_history"].append({
                                                    "role": "assistant", 
                                                    "content": f"Tool Execution Results:\n\n{tool_results}"
                                                })
                                                
                                                st.success("Tool executed successfully!")
                                                st.rerun()
                                        else:
                                            st.error("Invalid tool call format")
                        else:
                            st.info("No specific tools extracted from AI recommendation.")
                    
                    # Show the custom content if available
                    if tab_content.get('custom_content'):
                        st.markdown("---")
                        st.markdown(tab_content.get('custom_content', ''))
                
                elif tab_content.get('content_type') == 'machine_learning':
                    st.write("**🤖 Machine Learning Tools:**")
                    
                    # Show AI recommendation context
                    if tab_content.get('ai_source'):
                        with st.expander("📋 AI Recommendation Context", expanded=False):
                            st.write(tab_content.get('ai_source', ''))
                    
                    # Check if tools were already executed
                    if tab_content.get('tools_executed') and tab_content.get('executed_results'):
                        for j, result in enumerate(tab_content['executed_results']):
                            # Get tool name for the expander title
                            tool_name = tab_content.get('tool_names', [])[j] if tab_content.get('tool_names') and j < len(tab_content['tool_names']) else f"Tool {j+1}"
                            with st.expander(f"🔧 {tool_name} Results", expanded=False):
                                if isinstance(result, str):
                                    st.info(result)
                                else:
                                    # Render actual tool functionality based on tool type
                                    _render_tool_output(tool_name, result)
                    else:
                        # Show extracted tools as functional buttons (fallback)
                        extracted_tools = tab_content.get('extracted_tools', [])
                        if extracted_tools:
                            st.write("**🔧 AI-Recommended Tools to Execute:**")
                            for j, tool_call in enumerate(extracted_tools):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.code(tool_call, language="bash")
                                with col2:
                                    if st.button(f"Execute", key=f"execute_tool_{i}_{j}"):
                                        # Execute the tool call using the tool call system
                                        detected_tools = st.session_state.tool_call_system.detect_tool_calls(tool_call)
                                        if detected_tools:
                                            with st.spinner("🔧 Executing tool..."):
                                                results = st.session_state.tool_call_system.execute_tool_calls(
                                                    detected_tools, 
                                                    st.session_state.current_data
                                                )
                                                
                                                # Format and display tool results
                                                tool_results = st.session_state.tool_call_system.format_tool_results(results)
                                                
                                                # Add tool results to chat history
                                                st.session_state["chat_history"].append({
                                                    "role": "assistant", 
                                                    "content": f"Tool Execution Results:\n\n{tool_results}"
                                                })
                                                
                                                st.success("Tool executed successfully!")
                                                st.rerun()
                                        else:
                                            st.error("Invalid tool call format")
                        else:
                            st.info("No specific tools extracted from AI recommendation.")
                    
                    # Show the custom content if available
                    if tab_content.get('custom_content'):
                        st.markdown("---")
                        st.markdown(tab_content.get('custom_content', ''))
                
                elif tab_content.get('content_type') == 'custom':
                    # Check if tools were already executed
                    if tab_content.get('tools_executed') and tab_content.get('executed_results'):
                        for j, result in enumerate(tab_content['executed_results']):
                            # Get tool name for the expander title
                            tool_name = tab_content.get('tool_names', [])[j] if tab_content.get('tool_names') and j < len(tab_content['tool_names']) else f"Tool {j+1}"
                            with st.expander(f"🔧 {tool_name} Results", expanded=False):
                                if isinstance(result, str):
                                    st.info(result)
                                else:
                                    # Render actual tool functionality based on tool type
                                    _render_tool_output(tool_name, result)
                    else:
                        # Show extracted tools as functional buttons (fallback)
                        extracted_tools = tab_content.get('extracted_tools', [])
                        if extracted_tools:
                            st.write("**🔧 AI-Recommended Tools to Execute:**")
                            for j, tool_call in enumerate(extracted_tools):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.code(tool_call, language="bash")
                                with col2:
                                    if st.button(f"Execute", key=f"execute_tool_{i}_{j}"):
                                        # Execute the tool call using the tool call system
                                        detected_tools = st.session_state.tool_call_system.detect_tool_calls(tool_call)
                                        if detected_tools:
                                            with st.spinner("🔧 Executing tool..."):
                                                results = st.session_state.tool_call_system.execute_tool_calls(
                                                    detected_tools, 
                                                    st.session_state.current_data
                                                )
                                                
                                                # Format and display tool results
                                                tool_results = st.session_state.tool_call_system.format_tool_results(results)
                                                
                                                # Add tool results to chat history
                                                st.session_state["chat_history"].append({
                                                    "role": "assistant", 
                                                    "content": f"Tool Execution Results:\n\n{tool_results}"
                                                })
                                                
                                                st.success("Tool executed successfully!")
                                                st.rerun()
                                        else:
                                            st.error("Invalid tool call format")
                    
                    # Show the custom content prominently
                    st.markdown(tab_content.get('custom_content', 'Custom content from AI agent.'))
                    
                    # Show AI source if available
                    if tab_content.get('ai_source'):
                        with st.expander("📋 AI Recommendation Source", expanded=False):
                            st.write(tab_content.get('ai_source', ''))
                
                # Tab management
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Remove Tab", key=f"remove_tab_{i}"):
                        st.session_state.ai_generated_tabs.pop(i)
                        st.success("Tab removed!")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Refresh Tab", key=f"refresh_tab_{i}"):
                        st.info("Tab refreshed!")
                        st.rerun()
        

        st.markdown("---")
        # Tab Management Section
        with st.expander("🎛️ Tab Management", expanded=False):
            st.subheader("📝 Create New Tab")
            
            # Add new tab manually (for testing/demonstration)
            new_tab_name = st.text_input("New Tab Name:", key="new_tab_name", placeholder="e.g., Advanced Analysis")
            new_tab_type = st.selectbox("Tab Type:", [
                'data_analysis', 'statistical_testing', 'machine_learning', 'custom'
            ], key="new_tab_type")
            
            if st.button("➕ Add Tab", key="add_tab"):
                if new_tab_name:
                    new_tab = {
                        'name': new_tab_name,
                        'content_type': new_tab_type,
                        'description': f'AI-generated {new_tab_type.replace("_", " ").title()} tab',
                        'custom_content': f'This tab was created for {new_tab_type.replace("_", " ")} purposes.'
                    }
                    st.session_state.ai_generated_tabs.append(new_tab)
                    st.success(f"Added new tab: {new_tab_name}")
                    st.rerun()
                else:
                    st.warning("Please enter a tab name")
            
            st.markdown("---")
            st.subheader("🗑️ Tab Operations")
            
            # Clear all AI-generated tabs
            if st.button("🗑️ Clear All AI Tabs", key="clear_ai_tabs"):
                st.session_state.ai_generated_tabs = []
                st.success("All AI-generated tabs cleared!")
                st.rerun()
            
            st.markdown("---")
            st.subheader("📊 Tab Status")
            
            # Show tab status
            if st.session_state.ai_generated_tabs:
                st.info(f"📊 **Current Tabs:** {len(st.session_state.ai_generated_tabs)} AI-generated tabs available")
                for i, tab in enumerate(st.session_state.ai_generated_tabs):
                    st.write(f"- {tab['name']} ({tab['content_type'].replace('_', ' ').title()})")
            else:
                st.info("💡 **No AI-generated tabs yet.** Chat with the AI agent to create custom analysis tabs!")
            
            st.markdown("---")
            st.subheader("💡 Instructions")
            st.info("💡 **How to use this tabbed dashboard:**\n"
                    "1. **Data Overview tab** contains all basic analysis tools\n"
                    "2. **Chat with the AI agent** to generate custom analysis tabs\n"
                    "3. **AI-generated tabs** will appear automatically based on your needs\n"
                    "4. **Manage tabs** using the controls above")


def cleanup_on_exit():
    """Cleanup function to stop backend processes"""
    backend_manager.stop_backend_process()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_on_exit()
