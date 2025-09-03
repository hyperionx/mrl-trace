"""
FastAPI-based async server for handling multiple user requests
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .async_manager import AsyncRequestManager
from .request_handler import RequestType, RequestPriority

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="Chat message content")
    priority: str = Field("normal", description="Request priority (low, normal, high, urgent)")
    timeout: float = Field(30.0, description="Request timeout in seconds")


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    priority: str = Field("normal", description="Request priority")
    timeout: float = Field(60.0, description="Request timeout in seconds")


class DataAnalysisRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    priority: str = Field("normal", description="Request priority")
    timeout: float = Field(120.0, description="Request timeout in seconds")


class WorkflowUpdateRequest(BaseModel):
    action: str = Field(..., description="Workflow action to perform")
    data: Dict[str, Any] = Field(default_factory=dict, description="Workflow data")
    priority: str = Field("high", description="Request priority")
    timeout: float = Field(30.0, description="Request timeout in seconds")


class FileUploadRequest(BaseModel):
    filename: str = Field(..., description="Name of the file")
    file_data: str = Field(..., description="Base64 encoded file data")
    priority: str = Field("normal", description="Request priority")
    timeout: float = Field(60.0, description="Request timeout in seconds")


class ModelTrainingRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to train")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    priority: str = Field("low", description="Request priority")
    timeout: float = Field(300.0, description="Request timeout in seconds")


class ResponseModel(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    request_id: str = Field(..., description="Unique request ID")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")


class StatusResponse(BaseModel):
    status: str = Field(..., description="System status")
    components: Dict[str, Any] = Field(..., description="Component statuses")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    health: str = Field(..., description="Overall system health")


# Global async manager instance
async_manager: Optional[AsyncRequestManager] = None


def get_async_manager() -> AsyncRequestManager:
    """Dependency to get the async manager instance"""
    if async_manager is None:
        raise HTTPException(status_code=503, detail="Async manager not initialized")
    return async_manager


def get_priority_enum(priority_str: str) -> RequestPriority:
    """Convert priority string to enum"""
    priority_map = {
        'low': RequestPriority.LOW,
        'normal': RequestPriority.NORMAL,
        'high': RequestPriority.HIGH,
        'urgent': RequestPriority.URGENT
    }
    
    if priority_str.lower() not in priority_map:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {priority_str}")
    
    return priority_map[priority_str.lower()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global async_manager
    
    # Startup
    logger.info("Starting async backend server...")
    
    try:
        # Initialize async manager
        async_manager = AsyncRequestManager(
            max_connections=50,
            max_sessions=200,
            max_concurrent_requests=100,
            max_queue_size=500
        )
        
        # Start the manager
        await async_manager.start()
        logger.info("Async backend server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start async backend server: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down async backend server...")
        
        if async_manager:
            try:
                await async_manager.graceful_shutdown(timeout=30.0)
                logger.info("Async backend server shut down successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="AI Agent Async Backend",
    description="Asynchronous backend for handling multiple user requests efficiently",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "AI Agent Async Backend is running"}


@app.get("/health", response_model=StatusResponse)
async def health_check(manager: AsyncRequestManager = Depends(get_async_manager)):
    """Health check endpoint"""
    try:
        status_info = await manager.get_system_status()
        return StatusResponse(
            status=status_info['status'],
            components={
                'connection_pool': status_info['connection_pool'],
                'session_manager': status_info['session_manager'],
                'request_handler': status_info['request_handler']
            },
            performance=status_info['performance_metrics'],
            health=status_info['system_health']
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/session/create", response_model=ResponseModel)
async def create_session(
    user_id: str,
    api_key: str,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Create a new user session"""
    try:
        async with manager.get_session_context(user_id, api_key) as session_id:
            return ResponseModel(
                success=True,
                request_id=session_id,
                message="Session created successfully",
                data={"session_id": session_id, "user_id": user_id}
            )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/chat", response_model=ResponseModel)
async def submit_chat(
    session_id: str,
    user_id: str,
    request: ChatRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Submit a chat request"""
    try:
        priority = get_priority_enum(request.priority)
        
        request_id = await manager.submit_chat_request(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            priority=priority,
            timeout=request.timeout
        )
        
        # Add message to session history
        await manager.add_chat_message(session_id, "user", request.message)
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="Chat request submitted successfully",
            data={"request_id": request_id, "session_id": session_id}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit chat request: {str(e)}")


@app.post("/tool/execute", response_model=ResponseModel)
async def execute_tool(
    session_id: str,
    user_id: str,
    request: ToolExecutionRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Execute a tool"""
    try:
        priority = get_priority_enum(request.priority)
        
        request_id = await manager.submit_tool_execution_request(
            session_id=session_id,
            user_id=user_id,
            tool_name=request.tool_name,
            tool_params=request.parameters,
            priority=priority,
            timeout=request.timeout
        )
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="Tool execution request submitted successfully",
            data={"request_id": request_id, "tool": request.tool_name}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit tool execution request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit tool execution request: {str(e)}")


@app.post("/analysis", response_model=ResponseModel)
async def submit_analysis(
    session_id: str,
    user_id: str,
    request: DataAnalysisRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Submit a data analysis request"""
    try:
        priority = get_priority_enum(request.priority)
        
        request_id = await manager.submit_data_analysis_request(
            session_id=session_id,
            user_id=user_id,
            analysis_type=request.analysis_type,
            data_params=request.parameters,
            priority=priority,
            timeout=request.timeout
        )
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="Data analysis request submitted successfully",
            data={"request_id": request_id, "analysis_type": request.analysis_type}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit data analysis request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit data analysis request: {str(e)}")


@app.post("/workflow/update", response_model=ResponseModel)
async def update_workflow(
    session_id: str,
    user_id: str,
    request: WorkflowUpdateRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Update workflow"""
    try:
        priority = get_priority_enum(request.priority)
        
        request_id = await manager.submit_workflow_update_request(
            session_id=session_id,
            user_id=user_id,
            workflow_action=request.action,
            workflow_data=request.data,
            priority=priority,
            timeout=request.timeout
        )
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="Workflow update request submitted successfully",
            data={"request_id": request_id, "action": request.action}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit workflow update request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit workflow update request: {str(e)}")


@app.post("/file/upload", response_model=ResponseModel)
async def upload_file(
    session_id: str,
    user_id: str,
    request: FileUploadRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Upload a file"""
    try:
        priority = get_priority_enum(request.priority)
        
        # Decode base64 file data
        import base64
        file_data = base64.b64decode(request.file_data)
        
        request_id = await manager.submit_file_upload_request(
            session_id=session_id,
            user_id=user_id,
            filename=request.filename,
            file_data=file_data,
            priority=priority,
            timeout=request.timeout
        )
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="File upload request submitted successfully",
            data={"request_id": request_id, "filename": request.filename}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit file upload request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit file upload request: {str(e)}")


@app.post("/model/train", response_model=ResponseModel)
async def train_model(
    session_id: str,
    user_id: str,
    request: ModelTrainingRequest,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Train a machine learning model"""
    try:
        priority = get_priority_enum(request.priority)
        
        request_id = await manager.submit_model_training_request(
            session_id=session_id,
            user_id=user_id,
            model_type=request.model_type,
            training_params=request.parameters,
            priority=priority,
            timeout=request.timeout
        )
        
        return ResponseModel(
            success=True,
            request_id=request_id,
            message="Model training request submitted successfully",
            data={"request_id": request_id, "model_type": request.model_type}
        )
        
    except Exception as e:
        logger.error(f"Failed to submit model training request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit model training request: {str(e)}")


@app.get("/request/{request_id}/status")
async def get_request_status(
    request_id: str,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Get the status of a request"""
    try:
        status_info = await manager.get_request_status(request_id)
        if status_info is None:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get request status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


@app.delete("/request/{request_id}")
async def cancel_request(
    request_id: str,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Cancel a request"""
    try:
        success = await manager.cancel_request(request_id)
        if not success:
            raise HTTPException(status_code=404, detail="Request not found or cannot be cancelled")
        
        return {"message": "Request cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel request: {str(e)}")


@app.get("/session/{session_id}")
async def get_session_info(
    session_id: str,
    manager: AsyncRequestManager = Depends(get_async_manager)
):
    """Get session information"""
    try:
        session_info = await manager.get_session_info(session_id)
        if session_info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_info.session_id,
            "user_id": session_info.user_id,
            "created_at": session_info.created_at.isoformat(),
            "last_activity": session_info.last_activity.isoformat(),
            "is_active": session_info.is_active,
            "request_count": session_info.request_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@app.get("/stats")
async def get_system_stats(manager: AsyncRequestManager = Depends(get_async_manager)):
    """Get system statistics"""
    try:
        return await manager.get_system_status()
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(
        "src.backend.async_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
