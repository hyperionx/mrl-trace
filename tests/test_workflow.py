import pytest
import pandas as pd
import numpy as np
from src.workflow import WorkflowModule, WorkflowStep, ToolCallSystem
from src.analysis import AnalysisTools


class TestWorkflowStep:
    """Test the WorkflowStep enum"""
    
    def test_workflow_steps_exist(self):
        """Test that all expected workflow steps exist"""
        expected_steps = [
            'data_upload', 'data_exploration', 'data_preprocessing',
            'statistical_analysis', 'model_selection', 'model_training',
            'model_evaluation', 'model_interpretation'
        ]
        
        for step_name in expected_steps:
            assert hasattr(WorkflowStep, step_name.upper().replace(' ', '_'))


class TestWorkflowModule:
    """Test the WorkflowModule class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.workflow = WorkflowModule()
    
    def test_initialization(self):
        """Test that workflow initializes correctly"""
        assert self.workflow.current_step == WorkflowStep.DATA_UPLOAD
        assert len(self.workflow.step_descriptions) == 8
    
    def test_next_step(self):
        """Test moving to next step"""
        initial_step = self.workflow.current_step
        self.workflow.next_step()
        assert self.workflow.current_step != initial_step
        assert self.workflow.current_step == WorkflowStep.DATA_EXPLORATION
    
    def test_previous_step(self):
        """Test moving to previous step"""
        # Move to second step first
        self.workflow.next_step()
        second_step = self.workflow.current_step
        
        # Then move back
        self.workflow.previous_step()
        assert self.workflow.current_step == WorkflowStep.DATA_UPLOAD
    
    def test_set_step(self):
        """Test setting workflow to specific step"""
        self.workflow.set_step(WorkflowStep.MODEL_SELECTION)
        assert self.workflow.current_step == WorkflowStep.MODEL_SELECTION
    
    def test_get_current_step_info(self):
        """Test getting current step information"""
        info = self.workflow.get_current_step_info()
        assert 'step' in info
        assert 'description' in info
        assert 'step_number' in info
        assert 'total_steps' in info
        assert info['step_number'] == 1
        assert info['total_steps'] == 8
    
    def test_get_step_guidance(self):
        """Test getting step guidance"""
        guidance = self.workflow.get_step_guidance(WorkflowStep.DATA_UPLOAD)
        assert isinstance(guidance, str)
        assert len(guidance) > 0
        assert "Data Upload Guidance" in guidance
    
    def test_get_step_questions(self):
        """Test getting step questions"""
        questions = self.workflow.get_step_questions(WorkflowStep.DATA_UPLOAD)
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
    
    def test_store_and_get_step_data(self):
        """Test storing and retrieving step data"""
        test_data = {'key': 'value', 'number': 42}
        self.workflow.store_step_data(WorkflowStep.DATA_UPLOAD, test_data)
        
        retrieved_data = self.workflow.get_step_data(WorkflowStep.DATA_UPLOAD)
        assert retrieved_data == test_data
    
    def test_get_workflow_summary(self):
        """Test getting workflow summary"""
        summary = self.workflow.get_workflow_summary()
        assert 'current_step' in summary
        assert 'completed_steps' in summary
        assert 'total_steps' in summary
        assert 'progress_percentage' in summary
        assert summary['total_steps'] == 8
        assert summary['progress_percentage'] == 0.0


class TestToolCallSystem:
    """Test the ToolCallSystem class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.workflow = WorkflowModule()
        self.analysis_tools = AnalysisTools()
        self.tool_system = ToolCallSystem(self.workflow, self.analysis_tools)
    
    def test_initialization(self):
        """Test that tool call system initializes correctly"""
        assert hasattr(self.tool_system, 'available_tools')
        assert hasattr(self.tool_system, 'tool_results')
        assert len(self.tool_system.available_tools) > 0
    
    def test_available_tools_structure(self):
        """Test that available tools have correct structure"""
        for tool_name, tool_info in self.tool_system.available_tools.items():
            assert 'description' in tool_info
            assert 'function' in tool_info
            assert 'parameters' in tool_info
            assert 'example' in tool_info
    
    def test_detect_tool_calls(self):
        """Test tool call detection"""
        # Test with valid tool call
        user_input = "I want to check my workflow status tool: workflow_status"
        tool_calls = self.tool_system.detect_tool_calls(user_input)
        assert len(tool_calls) == 1
        assert tool_calls[0]['tool'] == 'workflow_status'
        
        # Test with no tool calls
        user_input = "Just a regular question about data"
        tool_calls = self.tool_system.detect_tool_calls(user_input)
        assert len(tool_calls) == 0
    
    def test_parse_parameters(self):
        """Test parameter parsing"""
        params_str = "test_type=normality column=age test_size=0.2"
        params = self.tool_system._parse_parameters(params_str)
        
        assert params['test_type'] == 'normality'
        assert params['column'] == 'age'
        assert params['test_size'] == 0.2
    
    def test_workflow_status_tool(self):
        """Test workflow status tool"""
        result = self.tool_system._get_workflow_status()
        assert 'current_step' in result
        assert 'step_description' in result
        assert 'progress' in result
    
    def test_workflow_help_tool(self):
        """Test workflow help tool"""
        result = self.tool_system._show_workflow_help()
        assert 'available_tools' in result
        assert 'tool_descriptions' in result
        assert 'examples' in result
    
    def test_statistical_test_tool_execution(self):
        """Test that statistical_test tool executes with data parameter"""
        # Create sample data
        sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'height': [170, 175, 180, 165, 185],
            'weight': [70, 75, 80, 65, 85]
        })
        
        # Test tool call detection
        tool_calls = self.tool_system.detect_tool_calls("tool: statistical_test test_type=normality column=age")
        assert len(tool_calls) == 1
        assert tool_calls[0]['tool'] == 'statistical_test'
        
        # Test tool execution with data
        results = self.tool_system.execute_tool_calls(tool_calls, sample_data)
        assert len(results) == 1
        assert results[0]['success'] == True
        assert results[0]['tool'] == 'statistical_test'


class TestWorkflowIntegration:
    """Test integration between workflow components"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.workflow = WorkflowModule()
        self.analysis_tools = AnalysisTools()
        self.tool_system = ToolCallSystem(self.workflow, self.analysis_tools)
    
    def test_workflow_progression_with_tools(self):
        """Test that tools can progress workflow"""
        # Start at first step
        assert self.workflow.current_step == WorkflowStep.DATA_UPLOAD
        
        # Use tool to move to next step
        result = self.tool_system._next_workflow_step()
        assert result['action'] == 'Moved to next step'
        assert self.workflow.current_step == WorkflowStep.DATA_EXPLORATION
        
        # Check that workflow status reflects change
        status = self.tool_system._get_workflow_status()
        assert status['current_step'] == 'data_exploration'
    
    def test_workflow_set_step_with_tools(self):
        """Test setting workflow step with tools"""
        # Set to a specific step
        result = self.tool_system._set_workflow_step(step='model_selection')
        assert result['action'] == 'Set workflow to step: model_selection'
        assert self.workflow.current_step == WorkflowStep.MODEL_SELECTION
    
    def test_invalid_step_handling(self):
        """Test handling of invalid step names"""
        result = self.tool_system._set_workflow_step(step='invalid_step')
        assert 'error' in result
        assert 'Invalid step' in result['error']


if __name__ == "__main__":
    pytest.main([__file__])
