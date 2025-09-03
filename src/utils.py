import streamlit as st
import asyncio
import time
import logging
import json
import os
import random
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Import async backend components
try:
    from src.backend.async_client import AsyncBackendClient, RequestPriority
    ASYNC_BACKEND_AVAILABLE = True
except ImportError:
    ASYNC_BACKEND_AVAILABLE = False

# Synthetic data for testing
SYNTHETIC_QUERIES = [
    "How do I analyze the correlation between variables in my dataset?",
    "What statistical test should I use for comparing two groups?",
    "Can you help me design an experiment with proper controls?",
    "How do I preprocess my data before machine learning?",
    "What visualization would best show the distribution of my data?",
    "How do I handle missing values in my dataset?",
    "What's the best way to validate my machine learning model?",
    "How do I interpret p-values in my statistical analysis?",
    "Can you suggest tools for data exploration?",
    "What's the difference between parametric and non-parametric tests?",
    "How do I calculate effect size for my study?",
    "What sample size do I need for my experiment?",
    "How do I check for normality in my data?",
    "What's the best way to handle outliers?",
    "How do I perform a power analysis?"
]

SYNTHETIC_RESPONSES = [
    "For correlation analysis, I recommend using Pearson's correlation coefficient for normally distributed data or Spearman's rank correlation for non-parametric data. You can use the `correlation_analysis` tool to explore relationships between variables.",
    "For comparing two groups, use a t-test for normally distributed data or Mann-Whitney U test for non-parametric data. The `statistical_test` tool can help you perform these analyses.",
    "For experimental design, ensure you have proper randomization, control groups, and blinding. Consider using the `workflow_help` tool to plan your experimental workflow.",
    "Data preprocessing should include handling missing values, scaling features, and encoding categorical variables. Use the `data_preprocessing` tool to automate these steps.",
    "For data distribution visualization, histograms and box plots work well. The `data_explore` tool can generate appropriate visualizations for your data type."
]

# Prompt tuning configuration
DEFAULT_PROMPT_TEMPLATES = {
    "data_analysis": {
        "name": "Data Analysis Expert",
        "description": "Specialized in statistical analysis and data interpretation",
        "system_prompt": "You are a data analysis expert. CRITICAL: You MUST ONLY suggest tools that are available in this system. NEVER suggest external tools like Jupyter Notebook, Python scripts, or other software. Available tools: data_analyze, data_explore, statistical_test, correlation_analysis, data_preprocessing, generate_insights. When helping with data analysis: 1) Ask what the user wants to know about their data, 2) Tell the user EXACTLY what you will do on the main screen (be specific about actions, not vague), 3) ALWAYS include the exact tool commands in your response using the format 'tool: tool_name [parameters]' so they can be captured and executed, 4) ALWAYS suggest which tab to create on the main screen for the next analysis step with natural language explanation. Example: 'To understand your data distribution, I will: 1) Create a Data Exploration tab and run tool: data_explore plot_type=histogram to show distribution plots, 2) Execute tool: data_analyze to generate summary statistics in a new tab, 3) Run tool: correlation_analysis to create a correlation matrix visualization. Next step: Create a 'Data Exploration' tab on the main screen where I will automatically generate these visualizations and statistics for you. This tab will contain all the tools needed to explore your data distributions, correlations, and generate insights.'"
    },
    "experimental_design": {
        "name": "Experimental Design Specialist",
        "description": "Expert in research methodology and experimental planning",
        "system_prompt": "You are an experimental design expert. CRITICAL: You MUST ONLY suggest tools that are available in this system. NEVER suggest external tools like Jupyter Notebook, Python scripts, or other software. Available tools: data_analyze, data_explore, statistical_test, correlation_analysis, generate_insights. When helping with experiments: 1) Ask about the research goal, 2) Tell the user EXACTLY what you will do on the main screen (be specific about actions, not vague), 3) ALWAYS include the exact tool commands in your response using the format 'tool: tool_name [parameters]' so they can be captured and executed, 4) ALWAYS suggest which tab to create on the main screen for the next analysis step with natural language explanation. Example: 'What is your main research question? For robust results, I will: 1) Create a Data Analysis tab and run tool: data_analyze to explore your experimental data, 2) Execute tool: statistical_test to validate your experimental design assumptions, 3) Generate insights about your experimental setup. Next step: Create a 'Statistical Analysis' tab on the main screen where I will automatically run these tests and show you the results. This tab will provide comprehensive statistical validation tools for your experimental design, including hypothesis testing and correlation analysis.'"
    },
    "machine_learning": {
        "name": "Machine Learning Engineer",
        "description": "Specialized in ML model development and evaluation",
        "system_prompt": "You are a machine learning expert. CRITICAL: You MUST ONLY suggest tools that are available in this system. NEVER suggest external tools like Jupyter Notebook, Python scripts, or other software. Available tools: data_preprocessing, train_baseline_model, evaluate_model, predict_with_model, model_comparison. When helping with ML: 1) Ask about the prediction goal, 2) Tell the user EXACTLY what you will do on the main screen (be specific about actions, not vague), 3) ALWAYS include the exact tool commands in your response using the format 'tool: tool_name [parameters]' so they can be captured and executed, 4) ALWAYS suggest which tab to create on the main screen for the next analysis step with natural language explanation. Example: 'What are you trying to predict? For ML success, I will: 1) Create a Data Preprocessing tab and run tool: data_preprocessing to clean your data, 2) Execute tool: train_baseline_model to build your prediction model, 3) Run tool: evaluate_model to show you the model performance metrics. Next step: Create a 'Machine Learning Model' tab on the main screen where I will automatically train and evaluate your model, showing you the results. This tab will contain all the ML tools needed for data preprocessing, model training, evaluation, and prediction.'"
    },
    "general_assistant": {
        "name": "General AI Assistant",
        "description": "Versatile AI assistant for scientific research",
        "system_prompt": "You are a helpful AI assistant for scientists. CRITICAL: You MUST ONLY suggest tools that are available in this system. NEVER suggest external tools like Jupyter Notebook, Python scripts, or other software. Available tools: workflow_status, workflow_help, data_analyze, data_explore, statistical_test, correlation_analysis, data_preprocessing, generate_insights, train_baseline_model, evaluate_model, predict_with_model, model_comparison. Your approach: 1) Ask clarifying questions, 2) Tell the user EXACTLY what you will do on the main screen (be specific about actions, not vague), 3) ALWAYS include the exact tool commands in your response using the format 'tool: tool_name [parameters]' so they can be captured and executed, 4) ALWAYS suggest which tab to create on the main screen for the next analysis step with natural language explanation. Example: 'What would you like to analyze? I will: 1) Create a Data Analysis tab and run tool: data_analyze to start your analysis, 2) Execute tool: workflow_status to show your current progress, 3) Generate insights based on your data. Next step: Create a 'Data Analysis' tab on the main screen where I will automatically perform these analyses and display the results for you. This tab will provide a comprehensive analysis workspace with tools for data exploration, statistical testing, and insight generation.'"
    },
    "custom": {
        "name": "Custom AI Personality",
        "description": "Fully customizable AI personality - edit the system prompt below",
        "system_prompt": "You are a custom AI assistant for experimental scientists. CRITICAL: You MUST ONLY suggest tools that are available in this system. NEVER suggest external tools like Jupyter Notebook, Python scripts, or other software. ALWAYS tell the user EXACTLY what you will do on the main screen (be specific about actions, not vague), ALWAYS include the exact tool commands in your response using the format 'tool: tool_name [parameters]' so they can be captured and executed, and ALWAYS suggest which tab to create on the main screen for the next analysis step with natural language explanation. This system prompt has been customized by the user to match their specific needs and preferences. Please adapt your responses according to the custom instructions provided in this prompt."
    }
}

class ChatProcessor:
    """Centralized chat processing to eliminate duplicate code"""
    
    def __init__(self, grok_client, tool_call_system, current_data):
        self.grok_client = grok_client
        self.tool_call_system = tool_call_system
        self.current_data = current_data
    
    def process_chat_input(self, user_input: str, use_async_mode: bool = False, 
                          async_backend_enabled: bool = False, user_session_id: str = None):
        """Process chat input and handle AI responses - centralized logic"""
        try:
            # Note: User message is already added to chat history by ChatInterface
            # Check for tool calls first
            tool_calls = self.tool_call_system.detect_tool_calls(user_input)
            
            if tool_calls:
                return self._execute_tools(tool_calls, use_async_mode, async_backend_enabled, user_session_id)
            else:
                return self._process_ai_chat(user_input, use_async_mode, async_backend_enabled, user_session_id)
                
        except Exception as e:
            st.error(f"❌ Error processing chat input: {e}")
            return None
    
    def _add_user_message(self, user_input: str):
        """Add user message to chat history"""
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state["chat_history"].append({
            "role": "user", 
            "content": f"[{timestamp}] {user_input}"
        })
    
    def _execute_tools(self, tool_calls: List[str], use_async_mode: bool, 
                       async_backend_enabled: bool, user_session_id: str):
        """Execute tool calls - centralized logic"""
        with st.spinner("🔧 Executing tools..."):
            if use_async_mode and async_backend_enabled and user_session_id:
                return self._execute_tools_async(tool_calls, user_session_id)
            else:
                return self._execute_tools_sync(tool_calls)
    
    def _execute_tools_async(self, tool_calls: List[str], user_session_id: str):
        """Execute tools using async backend"""
        try:
            from src.backend.async_client import AsyncBackendClient, RequestPriority
            
            tool_response = self._submit_async_request(
                'tool',
                tool_name=tool_calls[0],
                parameters={'tool_calls': tool_calls},
                user_id=f"user_{hash(time.time())}",
                priority=RequestPriority.HIGH,
                user_session_id=user_session_id
            )
            
            if tool_response and tool_response.success:
                st.success(f"⚡ Tool execution submitted to async backend (ID: {tool_response.request_id[:8]}...)")
                
                with st.spinner("⏳ Waiting for async completion..."):
                    final_status = self._get_request_status(tool_response.request_id, user_session_id)
                    if final_status and final_status.success:
                        tool_results = final_status.data.get('results', 'Tool executed successfully')
                    else:
                        tool_results = 'Tool execution completed via async backend'
            else:
                st.warning("⚠️ Async tool execution failed, falling back to sync mode")
                tool_results = self._execute_tools_sync(tool_calls)
                
            return tool_results
            
        except Exception as e:
            st.error(f"Async tool execution failed: {e}")
            return self._execute_tools_sync(tool_calls)
    
    def _execute_tools_sync(self, tool_calls: List[str]):
        """Execute tools synchronously"""
        results = self.tool_call_system.execute_tool_calls(tool_calls, self.current_data)
        return self.tool_call_system.format_tool_results(results)
    
    def _process_ai_chat(self, user_input: str, use_async_mode: bool, 
                         async_backend_enabled: bool, user_session_id: str):
        """Process AI chat - centralized logic"""
        # Enhance user input with data context if available
        enhanced_prompt, data_context = self._enhance_prompt_with_context(user_input)
        
        if self.grok_client is not None:
            selected_template = st.session_state.get('selected_prompt_template', 'general_assistant')
            start_time = time.time()
            
            if use_async_mode and async_backend_enabled and user_session_id:
                assistant_message = self._process_chat_async(enhanced_prompt, user_session_id)
            else:
                assistant_message = self._process_chat_sync(enhanced_prompt, data_context, selected_template)
            
            # Calculate response time and log interaction
            response_time = time.time() - start_time
            self._log_user_interaction(enhanced_prompt, assistant_message, selected_template, response_time)
            
            # Add assistant message to chat history
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state["chat_history"].append({
                "role": "assistant", 
                "content": f"[{timestamp}] {assistant_message}"
            })
            
            return assistant_message
        else:
            st.error("❌ Grok client not available. Please check your API key.")
            return None
    
    def _enhance_prompt_with_context(self, user_input: str):
        """Enhance user input with data context"""
        if self.current_data is not None:
            df = self.current_data
            data_context = f"""
            Context: You are helping an experimental scientist with a dataset containing {df.shape[0]} rows and {df.shape[1]} columns.
            Columns: {list(df.columns)}
            Data types: {df.dtypes.to_dict()}
            
            User question: {user_input}
            
            Please provide helpful, specific advice for their experimental analysis.
            """
            return data_context, data_context
        return user_input, user_input
    
    def _process_chat_async(self, enhanced_prompt: str, user_session_id: str):
        """Process chat using async backend"""
        with st.spinner("⚡ AI is thinking (async mode)..."):
            chat_response = self._submit_async_request(
                'chat',
                message=enhanced_prompt,
                user_id=f"user_{hash(time.time())}",
                priority=RequestPriority.NORMAL,
                user_session_id=user_session_id
            )
            
            if chat_response and chat_response.success:
                st.success(f"⚡ Chat request submitted to async backend (ID: {chat_response.request_id[:8]}...)")
                
                with st.spinner("⏳ Waiting for async completion..."):
                    final_status = self._get_request_status(chat_response.request_id, user_session_id)
                    if final_status and final_status.success:
                        return final_status.data.get('response', 'AI response received via async backend')
                    else:
                        return 'AI response completed via async backend'
            else:
                st.warning("⚠️ Async chat failed, falling back to sync mode")
                return self._process_chat_sync(enhanced_prompt, "", "general_assistant")
    
    def _process_chat_sync(self, enhanced_prompt: str, data_context: str, selected_template: str):
        """Process chat synchronously"""
        # st.info("🔄 Using synchronous chat mode...")
        return self._await_sync_chat(enhanced_prompt, data_context, selected_template)
    
    def _await_sync_chat(self, enhanced_prompt: str, data_context: str, prompt_template: str):
        """Execute synchronous chat with Grok client using tuned prompts"""
        try:
            chat = self.grok_client.chat.create(model="grok-4")
            
            # Safely get chat history and prompt templates
            chat_history = st.session_state.get("chat_history", [])
            prompt_templates = st.session_state.get('prompt_templates', {})
            current_data = self.current_data
            
            # Add previous context from chat history (limit to last 10 messages to avoid token limits)
            recent_messages = chat_history[-10:-1] if len(chat_history) > 1 else []
            for msg in recent_messages:
                if msg["role"] == "user":
                    from xai_sdk.chat import user
                    chat.append(user(msg["content"]))
                else:
                    from xai_sdk.chat import system
                    chat.append(system(msg["content"]))
            
            # Get the selected prompt template
            template = prompt_templates.get(
                prompt_template, 
                DEFAULT_PROMPT_TEMPLATES["general_assistant"]
            )
            
            # Build enhanced system prompt with template and data context
            # Note: Don't add conflicting instructions that override the template
            system_prompt = f"{template['system_prompt']} Data Context: {data_context if current_data is not None else 'No data loaded'}."
            
            from xai_sdk.chat import system, user
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
    
    def _submit_async_request(self, request_type: str, **kwargs):
        """Submit a request to the async backend"""
        try:
            async def submit_request():
                async with AsyncBackendClient() as client:
                    if request_type == 'chat':
                        return await client.submit_chat(
                            kwargs.get('user_session_id'), 
                            kwargs.get('user_id', 'default_user'),
                            kwargs.get('message', ''),
                            kwargs.get('priority', RequestPriority.NORMAL),
                            kwargs.get('timeout', 30.0)
                        )
                    elif request_type == 'tool':
                        return await client.execute_tool(
                            kwargs.get('user_session_id'),
                            kwargs.get('user_id', 'default_user'),
                            kwargs.get('tool_name', ''),
                            kwargs.get('parameters', {}),
                            kwargs.get('priority', RequestPriority.NORMAL),
                            kwargs.get('timeout', 60.0)
                        )
                    else:
                        return None
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, submit_request())
                return future.result(timeout=30)
                
        except Exception as e:
            st.error(f"❌ Async request failed: {e}")
            return None
    
    def _get_request_status(self, request_id: str, user_session_id: str):
        """Get the status of an async request"""
        try:
            async def check_status():
                async with AsyncBackendClient() as client:
                    return await client.get_request_status(request_id)
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, check_status())
                return future.result(timeout=10)
                
        except Exception as e:
            st.error(f"❌ Status check failed: {e}")
            return None
    
    def _log_user_interaction(self, query: str, response: str, template_used: str, response_time: float):
        """Log user interactions for analysis and validation"""
        try:
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "template_used": template_used,
                "response_time": response_time,
                "session_id": st.session_state.get('user_session_id', 'none')
            }
            
            # Log to file
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
            logging.error(f"Failed to log interaction: {e}")


class BackendManager:
    """Centralized backend management to eliminate duplicate code"""
    
    def __init__(self):
        self.backend_process = None
    
    def check_backend_running(self):
        """Check if the FastAPI backend is already running"""
        try:
            import requests
            endpoints = [
                "http://localhost:8000/health",
                "http://localhost:8000/",
                "http://localhost:8000/docs"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=2)
                    if response.status_code == 200:
                        return True
                except Exception:
                    continue
            
            return False
        except Exception:
            return False
    
    def debug_backend_status(self):
        """Debug function to check backend status"""
        try:
            import requests
            endpoints = [
                "http://localhost:8000/health",
                "http://localhost:8000/",
                "http://localhost:8000/docs"
            ]
            
            debug_info = {}
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=2)
                    debug_info[endpoint] = {
                        "status_code": response.status_code,
                        "response": response.text[:100] if response.text else "No content"
                    }
                except Exception as e:
                    debug_info[endpoint] = {"error": str(e)}
            
            return debug_info
        except Exception as e:
            return {"error": f"Debug failed: {e}"}
    
    def start_backend_process(self):
        """Start the FastAPI backend as a subprocess"""
        if self.backend_process and self.backend_process.poll() is None:
            return True  # Already running
        
        try:
            import subprocess
            import sys
            import os
            
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Start the async server
            self.backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "src.backend.async_server:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=script_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a bit for the server to start
            time.sleep(3)
            
            # Check if it's running
            return self.check_backend_running()
                
        except Exception as e:
            st.error(f"Failed to start backend: {e}")
            return False
    
    def stop_backend_process(self):
        """Stop the FastAPI backend process"""
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                self.backend_process = None
                return True
            except:
                try:
                    self.backend_process.kill()
                    self.backend_process = None
                    return True
                except:
                    return False
        return True
    
    def auto_start_backend(self):
        """Automatically start the FastAPI backend if not running"""
        if not self.check_backend_running():
            st.info("🚀 Auto-starting FastAPI backend...")
            with st.spinner("Starting backend server..."):
                if self.start_backend_process():
                    st.success("✅ Backend started successfully!")
                    return True
                else:
                    st.error("❌ Failed to start backend automatically")
                    return False
        return True


class SessionManager:
    """Centralized session management to eliminate duplicate code"""
    
    def __init__(self):
        self.user_session_id = None
        self.async_backend_enabled = False
        self.async_client = None
    
    def initialize_async_backend(self):
        """Initialize the async backend client"""
        if not ASYNC_BACKEND_AVAILABLE:
            return False
        
        try:
            async def check_backend():
                try:
                    async with AsyncBackendClient() as client:
                        try:
                            async with client.session.get(f"{client.base_url}/") as response:
                                if response.status == 200:
                                    try:
                                        health_response = await client.health_check()
                                        print(f"Health check result: {health_response.success}")
                                    except Exception:
                                        pass
                                    return True
                                else:
                                    return False
                        except Exception:
                            return False
                except Exception as e:
                    print(f"Async client initialization failed: {e}")
                    return False
            
            # Run async check in thread
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, check_backend())
                backend_available = future.result(timeout=15)
            
            if backend_available:
                self.async_client = AsyncBackendClient()
                self.async_backend_enabled = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"initialize_async_backend failed: {e}")
            return False
    
    def create_user_session(self, user_id: str, api_key: str):
        """Create a user session with the async backend"""
        if not self.async_backend_enabled or not self.async_client:
            return None
        
        try:
            async def create_session():
                async with self.async_client:
                    response = await self.async_client.create_session(user_id, api_key)
                    if response.success:
                        return response.data.get('session_id')
                    return None
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, create_session())
                session_id = future.result(timeout=10)
            
            if session_id:
                self.user_session_id = session_id
                st.success(f"✅ Session created: {session_id[:8]}...")
                return session_id
            else:
                st.error("❌ Failed to create session")
                return None
                
        except Exception as e:
            st.error(f"❌ Session creation failed: {e}")
            return None


class ChatInterface:
    """Centralized chat interface management"""
    
    def __init__(self, grok_client, tool_call_system, current_data):
        self.grok_client = grok_client
        self.tool_call_system = tool_call_system
        self.current_data = current_data
        self.chat_processor = ChatProcessor(grok_client, tool_call_system, current_data)
    
    def create_sidebar_chat(self, use_async_mode: bool = False, async_backend_enabled: bool = False, user_session_id: str = None):
        """Create the chat interface in the sidebar with auto-scrolling to latest messages"""
        st.divider()
        st.header("💬 AI Chat")
        st.write("Ask questions about your data and analysis.")
        
        # Initialize chat history if not exists
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Check for pending AI responses and process them
        if st.session_state.get('pending_ai_response', False):
            self._process_pending_ai_response()

        # Show info about Accept/Reject functionality
        with st.expander("ℹ️ How to use Accept/Reject buttons", expanded=False):
            st.info("""
            **Accept/Reject AI Actions:**
            
            💡 **Accept Button**: Creates a new tab on the main dashboard based on AI recommendations
            ❌ **Reject Button**: Dismisses the AI recommendation and continues the conversation
            
            **How it works:**
            1. Chat with the AI about your data analysis needs
            2. When the AI gives recommendations, use the Accept button to create a new analysis tab
            3. The new tab will contain tools and content based on the AI's suggestions
            4. Use the Reject button if you want different recommendations
            """)
        
        # Create custom CSS for better chat display
        self._create_chat_css()
        
        # Display chat history
        self._display_chat_history()
        
        # Add chat control buttons
        self._add_chat_controls()
        
        # Chat interface at the very bottom of sidebar
        self._create_chat_input()
    
    def _create_chat_css(self):
        """Create custom CSS for chat interface"""
        st.markdown("""
        <style>
        .chat-message {
            padding: 10px;
            margin: 6px 0;
            border-radius: 10px;
            border-left: 3px solid;
            font-size: 0.9em;
            line-height: 1.3;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left-color: #2196f3;
            color: #1565c0;
            margin-left: 15px;
        }
        .assistant-message {
            background-color: #f3e5f5;
            border-left-color: #9c27b0;
            color: #6a1b9a;
            margin-right: 15px;
        }
        .chat-container {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: #fafafa;
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .chat-messages-wrapper {
            flex: 1;
            overflow-y: auto;
        }
        .empty-chat-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _display_chat_history(self):
        """Display chat history container"""
        st.subheader("Chat History")
        
        if st.session_state["chat_history"]:
            # Display existing messages (newest at top)
            chat_html = '<div class="chat-container" id="chatContainer">'
            chat_html += '<div class="chat-messages-wrapper">'
            # Reverse the order to show newest messages at the top
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "user":
                    chat_html += f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>'
                else:
                    # Add indicator for AI messages that can be accepted
                    ai_indicator = '<span style="color: #9c27b0; font-size: 0.8em;">💡</span>'
                    chat_html += f'<div class="chat-message assistant-message"><strong>AI:</strong> {msg["content"]} {ai_indicator}</div>'
            chat_html += '</div></div>'
            
            # Add tool call buttons for the last AI response below the chat container
            if st.session_state["chat_history"]:
                last_ai_msg = None
                for msg in reversed(st.session_state["chat_history"]):
                    if msg["role"] == "assistant":
                        last_ai_msg = msg
                        break
                
                if last_ai_msg:
                    # Tool calls are now handled automatically when AI action is accepted
                    # No need to show them in sidebar
                    pass
        else:
            # Display empty chat container with placeholder
            chat_html = '''
            <div class="chat-container" id="chatContainer">
                <div class="chat-messages-wrapper">
                    <div class="empty-chat-placeholder">
                        💡 Start a conversation by typing a message below!<br>
                    </div>
                </div>
            </div>
            '''
        
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Simple JavaScript for basic scrolling
        st.markdown(
            """
            <script>
            // Simple function to scroll to top (newest messages)
            function scrollToTop() {
                const chatContainer = document.getElementById('chatContainer');
                if (chatContainer) {
                    const messagesWrapper = chatContainer.querySelector('.chat-messages-wrapper');
                    if (messagesWrapper) {
                        messagesWrapper.scrollTop = 0;
                    }
                }
            }
            
            // Scroll to top when page loads to show newest messages
            setTimeout(scrollToTop, 100);
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def _add_chat_controls(self):
        """Add chat control buttons"""
        # Check if there are tool calls or next step recommendations in the last AI message
        has_ai_action = False
        if st.session_state["chat_history"]:
            last_ai_msg = None
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "assistant":
                    last_ai_msg = msg
                    break
            
            if last_ai_msg:
                # Check for tool calls
                tool_calls = self._extract_tool_calls_from_response(last_ai_msg["content"])
                has_tool_calls = len(tool_calls) > 0
                
                # Check for next step recommendations
                has_next_step = "Next step:" in last_ai_msg["content"] or "next step:" in last_ai_msg["content"]
                
                has_ai_action = has_tool_calls or has_next_step
        
        # Always show Clear Chat button
        if has_ai_action:
            # Show indicator that AI action is available
            st.info("💡 **AI Action Available**: The AI has provided actionable recommendations. You can accept or reject them below.")
            
            # Show all three buttons when AI action is detected
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state["chat_history"] = []
                    st.rerun()
            
            with col2:
                if st.button("✅ Accept AI Action", key="accept_ai_action"):
                    self._handle_accept_ai_action()
            
            with col3:
                if st.button("❌ Reject AI Action", key="reject_ai_action"):
                    st.info("❌ AI action rejected. You can continue chatting or ask for different recommendations.")
        else:
            # Only show Clear Chat button when no AI action
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()
    
    def _create_chat_input(self):
        """Create chat input interface"""
        # Use a form for better input handling and immediate updates
        with st.form(key="chat_input_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask questions about your data, analysis, or experimental design:", 
                key="chat_input", 
                placeholder="Type your question here..."
            )
            
            # Handle form submission
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.form_submit_button("Send", key="chat_send_form"):
                    if user_input and user_input.strip():
                        self._process_chat_input(user_input.strip(), use_async_mode=False, async_backend_enabled=False, user_session_id=None)
            
            with col2:
                if st.form_submit_button("Clear", key="chat_clear_form"):
                    # Don't modify chat_input directly, let the form handle clearing
                    pass
    
    def _process_chat_input(self, user_input: str, use_async_mode: bool = False, async_backend_enabled: bool = False, user_session_id: str = None):
        """Process chat input using the chat processor"""
        # Add user message to chat history immediately for instant display
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state["chat_history"].append({
            "role": "user", 
            "content": f"[{timestamp}] {user_input}"
        })
        
        # Store the input for processing in the next cycle
        st.session_state['pending_user_input'] = user_input
        st.session_state['pending_ai_response'] = True
        
        # Force UI refresh to show the user message immediately
        st.rerun()
    
    def _process_pending_ai_response(self):
        """Process pending AI response in the background"""
        try:
            # Get the pending user input
            user_input = st.session_state.get('pending_user_input', '')
            if not user_input:
                st.session_state.pending_ai_response = False
                return
            
            # Process the AI response using the chat processor
            result = self.chat_processor.process_chat_input(
                user_input, 
                use_async_mode=False, 
                async_backend_enabled=False, 
                user_session_id=None
            )
            
            # Clear the pending state
            st.session_state.pending_ai_response = False
            st.session_state.pending_user_input = ''
            
            # If we got a result, refresh the UI
            if result is not None:
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing AI response: {e}")
            st.session_state.pending_ai_response = False
            st.session_state.pending_user_input = ''
    
    def _extract_tool_calls_from_response(self, response_text: str) -> List[str]:
        """Extract tool calls from AI response text"""
        tool_calls = []
        
        # Method 1: Look for tool calls wrapped in backticks
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
        
        # Method 2: Look for tool calls in the text without backticks
        if not tool_calls:
            # Use regex to find tool: commands in the text
            import re
            tool_pattern = r'tool:\s*(\w+(?:\s+[^,\n]+)?)'
            matches = re.findall(tool_pattern, response_text, re.IGNORECASE)
            for match in matches:
                tool_call = f"tool: {match.strip()}"
                if tool_call not in tool_calls:
                    tool_calls.append(tool_call)
        
        # Method 3: Look for specific tool mentions in the text
        if not tool_calls:
            # Check for common tool patterns in the text
            common_tools = [
                'data_analyze', 'data_explore', 'statistical_test', 'correlation_analysis',
                'data_preprocessing', 'generate_insights', 'train_baseline_model',
                'evaluate_model', 'predict_with_model', 'model_comparison'
            ]
            
            for tool in common_tools:
                if tool in response_text.lower():
                    tool_call = f"tool: {tool}"
                    if tool_call not in tool_calls:
                        tool_calls.append(tool_call)
        
        # Debug: Log what was found
        if tool_calls:
            st.info(f"🔍 Extracted {len(tool_calls)} tool calls: {tool_calls}")
        else:
            st.info(f"🔍 No tool calls found in response. Response preview: {response_text[:200]}...")
        
        return tool_calls
    
    # Tool call buttons are no longer needed as tools are executed automatically
    # when AI actions are accepted
    
    def _handle_accept_ai_action(self):
        """Handle accepting AI action to create new tab and execute tools"""
        # Get the last AI message to extract recommendations
        if st.session_state["chat_history"]:
            last_ai_msg = None
            for msg in reversed(st.session_state["chat_history"]):
                if msg["role"] == "assistant":
                    last_ai_msg = msg
                    break
            
            if last_ai_msg:
                # Create a new tab based on AI recommendations
                ai_content = last_ai_msg["content"]
                new_tab = self._create_tab_from_ai_recommendation(ai_content)
                if new_tab:
                    if 'ai_generated_tabs' not in st.session_state:
                        st.session_state.ai_generated_tabs = []
                    
                    # Execute tool calls automatically if available
                    extracted_tools = new_tab.get('extracted_tools', [])
                    if extracted_tools:
                        # Execute all available tools
                        all_results = []
                        tool_names = []
                        for tool_call in extracted_tools:
                            try:
                                # Extract tool name from tool call (skip "tool:" prefix)
                                if tool_call and tool_call.startswith("tool:"):
                                    tool_name = tool_call.split()[1] if len(tool_call.split()) > 1 else "Unknown Tool"
                                else:
                                    tool_name = tool_call.split()[0] if tool_call else "Unknown Tool"
                                tool_names.append(tool_name)
                                
                                # Execute the tool call using the tool call system
                                detected_tools = st.session_state.tool_call_system.detect_tool_calls(tool_call)
                                if detected_tools:
                                    results = st.session_state.tool_call_system.execute_tool_calls(
                                        detected_tools, 
                                        st.session_state.current_data
                                    )
                                    all_results.append(results)
                                else:
                                    all_results.append(f"Invalid tool call format: {tool_call}")
                            except Exception as e:
                                all_results.append(f"Error executing {tool_call}: {str(e)}")
                        
                        # Store the execution results and tool names in the tab
                        new_tab['executed_results'] = all_results
                        new_tab['tool_names'] = tool_names
                        new_tab['tools_executed'] = True
                    
                    st.session_state.ai_generated_tabs.append(new_tab)
                    st.success(f"✅ Created new tab: {new_tab['name']}")
                    # Note: st.rerun() removed to prevent UI conflicts
                else:
                    st.warning("⚠️ Could not extract actionable recommendation from AI response")
            else:
                st.warning("⚠️ No AI message found to accept")
    


    def _create_tab_from_ai_recommendation(self, ai_content: str) -> dict:
        """Create a new tab based on AI recommendations in the content"""
        try:
            # Extract key information from AI response
            content_lower = ai_content.lower()
            
            # Try to extract tab name from "Next step:" recommendation
            tab_name = None
            tab_type = 'custom'
            description = "AI-generated analysis recommendation"
            
            # Look for "Next step:" pattern to extract specific tab name
            if "Next step:" in ai_content or "next step:" in ai_content:
                lines = ai_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("Next step:") or line.startswith("next step:"):
                        # Extract the tab name from quotes or after "Create a"
                        if "Create a '" in line and "' tab" in line:
                            # Extract text between single quotes
                            start_idx = line.find("'") + 1
                            end_idx = line.find("'", start_idx)
                            if start_idx > 0 and end_idx > start_idx:
                                tab_name = line[start_idx:end_idx]
                        elif "Create a " in line and " tab" in line:
                            # Extract text after "Create a " and before " tab"
                            start_idx = line.find("Create a ") + 8
                            end_idx = line.find(" tab", start_idx)
                            if start_idx > 8 and end_idx > start_idx:
                                tab_name = line[start_idx:end_idx]
                        break
            
            # If no specific tab name found, determine based on content analysis
            if not tab_name:
                if any(word in content_lower for word in ['data analysis', 'explore', 'investigate', 'examine']):
                    tab_type = 'data_analysis'
                    tab_name = "🔍 Data Analysis"
                    description = "AI-recommended data analysis tools and insights"
                elif any(word in content_lower for word in ['statistical', 'statistics', 'test', 'correlation', 'regression']):
                    tab_type = 'statistical_testing'
                    tab_name = "📊 Statistical Testing"
                    description = "AI-recommended statistical analysis and testing tools"
                elif any(word in content_lower for word in ['machine learning', 'ml', 'model', 'predict', 'classification']):
                    tab_type = 'machine_learning'
                    tab_name = "🤖 Machine Learning"
                    description = "AI-recommended machine learning tools and models"
                elif any(word in content_lower for word in ['visualization', 'plot', 'chart', 'graph']):
                    tab_type = 'data_analysis'
                    tab_name = "📈 Advanced Visualizations"
                    description = "AI-recommended visualization and plotting tools"
                elif any(word in content_lower for word in ['clean', 'preprocess', 'transform', 'feature']):
                    tab_type = 'data_analysis'
                    tab_name = "🧹 Data Preprocessing"
                    description = "AI-recommended data cleaning and preprocessing tools"
                else:
                    tab_type = 'custom'
                    tab_name = "💡 AI Recommendation"
                    description = "AI-generated analysis recommendation"
            
            # Add emoji prefix if not already present
            if not any(emoji in tab_name for emoji in ['🔍', '📊', '🤖', '📈', '🧹', '💡']):
                if 'data' in tab_name.lower() or 'analysis' in tab_name.lower():
                    tab_name = f"🔍 {tab_name}"
                elif 'statistical' in tab_name.lower() or 'test' in tab_name.lower():
                    tab_name = f"📊 {tab_name}"
                elif 'machine' in tab_name.lower() or 'learning' in tab_name.lower():
                    tab_name = f"🤖 {tab_name}"
                elif 'visualization' in tab_name.lower() or 'plot' in tab_name.lower():
                    tab_name = f"📈 {tab_name}"
                else:
                    tab_name = f"💡 {tab_name}"
            
            # Extract tool calls from the AI response
            tool_calls = self._extract_tool_calls_from_response(ai_content)
            
            # Create custom content with actual tool calls
            if tool_calls:
                custom_content = f"""
                **📋 Results Available:**
                Tools have been automatically executed when you accepted this AI action. 
                View the results in the tab content above.
                
                **💡 Next Steps:**
                - Review the analysis results above
                - Ask follow-up questions about the results
                - Request additional analysis as needed
                """
            else:
                # If no tool calls found, create content based on the AI recommendation
                custom_content = f"""
                **💡 Analysis Plan:**
                The AI has provided recommendations for your data analysis. While specific tool commands weren't detected, you can:
                
                1. **Ask follow-up questions** in the chat to get specific tool recommendations
                2. **Request specific analysis** like "Can you show me correlation analysis tools?"
                3. **Ask for data exploration** tools to start your analysis
                
                **🔧 Available Tools in This System:**
                - `tool: data_analyze` - Basic data analysis and insights
                - `tool: data_explore` - Data exploration with visualizations
                - `tool: statistical_test` - Statistical testing
                - `tool: correlation_analysis` - Correlation analysis
                - `tool: data_preprocessing` - Data preprocessing
                - `tool: generate_insights` - AI-powered insights
                
                **💬 Next Steps:**
                - Ask the AI for specific analysis recommendations
                - Request help with particular data questions
                - The AI will automatically execute tools when you accept recommendations
                """
            
            return {
                'name': tab_name,
                'content_type': tab_type,
                'description': description,
                'custom_content': custom_content,
                'ai_source': ai_content[:200] + '...' if len(ai_content) > 200 else ai_content,
                'extracted_tools': tool_calls,  # Store the actual extracted tools
                'full_ai_content': ai_content  # Store full content for reference
            }
            
        except Exception as e:
            st.error(f"Error creating tab from AI recommendation: {e}")
            return None


class TestingTools:
    """Centralized testing and analytics tools"""
    
    @staticmethod
    def generate_synthetic_data(num_samples: int = 5) -> List[Dict[str, str]]:
        """Generate synthetic user interactions for testing"""
        synthetic_data = []
        
        for i in range(min(num_samples, len(SYNTHETIC_QUERIES))):
            query = SYNTHETIC_QUERIES[i]
            response = SYNTHETIC_RESPONSES[i % len(SYNTHETIC_RESPONSES)]
            
            synthetic_data.append({
                "query": query,
                "response": response,
                "template_used": random.choice(["data_analysis", "experimental_design", "machine_learning", "general_assistant"]),
                "response_time": random.uniform(1.0, 5.0)
            })
        
        return synthetic_data
    
    @staticmethod
    def simulate_user_interactions(num_interactions: int = 10):
        """Simulate multiple user interactions for testing system robustness"""
        logger = setup_logging()
        logger.info(f"Starting simulation of {num_interactions} user interactions")
        
        synthetic_data = TestingTools.generate_synthetic_data(num_interactions)
        
        for i, interaction in enumerate(synthetic_data, 1):
            logger.info(f"Simulating interaction {i}/{num_interactions}")
            
            # Log the simulated interaction
            TestingTools._log_user_interaction(
                query=interaction["query"],
                response=interaction["response"],
                template_used=interaction["template_used"],
                response_time=interaction["response_time"]
            )
            
            # Simulate processing time
            time.sleep(0.1)
        
        logger.info(f"Simulation completed. Generated {len(synthetic_data)} interactions")
        return synthetic_data
    
    @staticmethod
    def _log_user_interaction(query: str, response: str, template_used: str, response_time: float = None):
        """Log user interactions for analysis and validation"""
        try:
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "template_used": template_used,
                "response_time": response_time,
                "session_id": "simulation"
            }
            
            # Log to file
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
            logging.error(f"Failed to log interaction: {e}")


def setup_logging():
    """Setup logging configuration for user interactions"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique log file for each session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/user_interactions_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_api_key(api_key):
    """Validate the API key by testing a real API connection"""
    if not api_key:
        return False, "No API key provided"
    
    # Basic format validation
    if len(api_key) < 10:
        return False, "❌ API key too short. Please check your API key format."
    
    # Check for common placeholder patterns
    if api_key.lower() in ['abc', 'test', 'demo', 'placeholder', 'your_api_key_here', '***']:
        return False, "❌ Please enter a real API key, not a placeholder."
    
    # Check if it looks like a real API key (should contain alphanumeric characters)
    if not any(c.isalnum() for c in api_key):
        return False, "❌ API key format appears invalid. Please check your API key."
    
    try:
        from xai_sdk import Client
        
        # Test the API key with a real API call
        test_client = Client(
            api_key=api_key,
            timeout=15,  # Short timeout for validation
        )
        
        # Create a chat session and make a simple test call
        test_chat = test_client.chat.create(model="grok-4")
        
        # Actually test the API key by making a minimal request
        from xai_sdk.chat import user
        test_chat.append(user("Hello"))
        test_response = test_chat.sample()
        
        # If we get here, the API key is valid
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific API key errors
        if "Incorrect API key provided" in error_msg:
            return False, "❌ Invalid API key. Please check your API key and try again."
        elif "doesn't have any credits yet" in error_msg:
            return False, "❌ API key has no credits. Please add credits to your x.ai account."
        elif "PERMISSION_DENIED" in error_msg:
            return False, "❌ Permission denied. Please check your API key permissions."
        elif "INVALID_ARGUMENT" in error_msg:
            return False, "❌ Invalid API key format or argument."
        elif "timeout" in error_msg.lower():
            return False, "❌ API validation timed out. Please check your connection and try again."
        elif "authentication" in error_msg.lower():
            return False, "❌ Authentication failed. Please check your API key."
        elif "unauthorized" in error_msg.lower():
            return False, "❌ Unauthorized. Please check your API key permissions."
        else:
            return False, f"❌ API connection error: {error_msg}"


def is_api_key_ready():
    """Check if the API key is ready for use"""
    if not st.session_state.get('api_key', ''):
        return False
    
    # Simple validation check
    is_valid, _ = validate_api_key(st.session_state.api_key)
    return is_valid
