"""
Async Request Manager - Main coordinator for the async backend
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .connection_pool import AsyncConnectionPool
from .session_manager import SessionManager
from .request_handler import RequestHandler, RequestType, RequestPriority

logger = logging.getLogger(__name__)


class AsyncRequestManager:
    """
    Main coordinator for the async backend system
    """
    
    def __init__(
        self,
        max_connections: int = 20,
        max_sessions: int = 100,
        max_concurrent_requests: int = 50,
        max_queue_size: int = 200
    ):
        # Initialize components
        self.connection_pool = AsyncConnectionPool(max_connections=max_connections)
        self.session_manager = SessionManager(max_sessions=max_sessions)
        self.request_handler = RequestHandler(
            max_concurrent_requests=max_concurrent_requests,
            max_queue_size=max_queue_size
        )
        
        # System state
        self._running = False
        self._startup_lock = asyncio.Lock()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'peak_concurrent_users': 0,
            'system_uptime': 0.0
        }
        
        logger.info("AsyncRequestManager initialized")
    
    async def start(self):
        """Start all backend components"""
        if self._running:
            logger.warning("AsyncRequestManager is already running")
            return
        
        async with self._startup_lock:
            if self._running:
                return
            
            try:
                # Start all components
                await self.connection_pool.start()
                await self.session_manager.start()
                await self.request_handler.start()
                
                self._running = True
                logger.info("AsyncRequestManager started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start AsyncRequestManager: {e}")
                await self.stop()
                raise
    
    async def stop(self):
        """Stop all backend components"""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # Stop all components
            await asyncio.gather(
                self.connection_pool.stop(),
                self.session_manager.stop(),
                self.request_handler.stop(),
                return_exceptions=True
            )
            
            logger.info("AsyncRequestManager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping AsyncRequestManager: {e}")
    
    @asynccontextmanager
    async def get_session_context(self, user_id: str, api_key: str):
        """
        Context manager for session management
        
        Args:
            user_id: User identifier
            api_key: User's API key
            
        Yields:
            Session ID for the user
        """
        session_id = await self.session_manager.create_session(user_id, api_key)
        
        try:
            yield session_id
        finally:
            # Session cleanup can be handled by the session manager's timeout mechanism
            pass
    
    async def submit_chat_request(
        self,
        session_id: str,
        user_id: str,
        message: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0
    ) -> str:
        """
        Submit a chat request
        
        Args:
            session_id: Session ID
            user_id: User ID
            message: Chat message
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'message': message,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.CHAT,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted chat request {request_id} for user {user_id}")
        return request_id
    
    async def submit_tool_execution_request(
        self,
        session_id: str,
        user_id: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 60.0
    ) -> str:
        """
        Submit a tool execution request
        
        Args:
            session_id: Session ID
            user_id: User ID
            tool_name: Name of the tool to execute
            tool_params: Tool parameters
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'tool': tool_name,
            'parameters': tool_params,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.TOOL_EXECUTION,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted tool execution request {request_id} for user {user_id}")
        return request_id
    
    async def submit_data_analysis_request(
        self,
        session_id: str,
        user_id: str,
        analysis_type: str,
        data_params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 120.0
    ) -> str:
        """
        Submit a data analysis request
        
        Args:
            session_id: Session ID
            user_id: User ID
            analysis_type: Type of analysis
            data_params: Analysis parameters
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'analysis_type': analysis_type,
            'parameters': data_params,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.DATA_ANALYSIS,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted data analysis request {request_id} for user {user_id}")
        return request_id
    
    async def submit_workflow_update_request(
        self,
        session_id: str,
        user_id: str,
        workflow_action: str,
        workflow_data: Dict[str, Any],
        priority: RequestPriority = RequestPriority.HIGH,
        timeout: float = 30.0
    ) -> str:
        """
        Submit a workflow update request
        
        Args:
            session_id: Session ID
            user_id: User ID
            workflow_action: Workflow action to perform
            workflow_data: Workflow data
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'action': workflow_action,
            'data': workflow_data,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.WORKFLOW_UPDATE,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted workflow update request {request_id} for user {user_id}")
        return request_id
    
    async def submit_file_upload_request(
        self,
        session_id: str,
        user_id: str,
        filename: str,
        file_data: bytes,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 60.0
    ) -> str:
        """
        Submit a file upload request
        
        Args:
            session_id: Session ID
            user_id: User ID
            filename: Name of the file
            file_data: File data
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'filename': filename,
            'file_data': file_data,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.FILE_UPLOAD,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted file upload request {request_id} for user {user_id}")
        return request_id
    
    async def submit_model_training_request(
        self,
        session_id: str,
        user_id: str,
        model_type: str,
        training_params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.LOW,
        timeout: float = 300.0
    ) -> str:
        """
        Submit a model training request
        
        Args:
            session_id: Session ID
            user_id: User ID
            model_type: Type of model to train
            training_params: Training parameters
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Request ID
        """
        data = {
            'model_type': model_type,
            'parameters': training_params,
            'session_id': session_id
        }
        
        request_id = await self.request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.MODEL_TRAINING,
            data=data,
            priority=priority,
            timeout=timeout
        )
        
        # Update performance metrics
        self.performance_metrics['total_requests'] += 1
        
        logger.debug(f"Submitted model training request {request_id} for user {user_id}")
        return request_id
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a request
        
        Args:
            request_id: Request ID to check
            
        Returns:
            Request status or None if not found
        """
        return await self.request_handler.get_request_status(request_id)
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request
        
        Args:
            request_id: Request ID to cancel
            
        Returns:
            True if cancelled, False otherwise
        """
        return await self.request_handler.cancel_request(request_id)
    
    async def get_session_info(self, session_id: str):
        """
        Get session information
        
        Args:
            session_id: Session ID to get info for
            
        Returns:
            Session information
        """
        return await self.session_manager.get_session(session_id)
    
    async def update_session_data(self, session_id: str, **kwargs) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session ID to update
            **kwargs: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        return await self.session_manager.update_session_data(session_id, **kwargs)
    
    async def add_chat_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a chat message to session history
        
        Args:
            session_id: Session ID to add message to
            role: Message role
            content: Message content
            
        Returns:
            True if successful, False otherwise
        """
        return await self.session_manager.add_chat_message(session_id, role, content)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        
        Returns:
            System status information
        """
        if not self._running:
            return {'status': 'stopped'}
        
        # Get status from all components
        connection_status = self.connection_pool.get_pool_status()
        session_status = self.session_manager.get_session_stats()
        handler_status = self.request_handler.get_handler_stats()
        
        # Calculate current concurrent users
        current_concurrent = session_status['active_sessions']
        if current_concurrent > self.performance_metrics['peak_concurrent_users']:
            self.performance_metrics['peak_concurrent_users'] = current_concurrent
        
        return {
            'status': 'running',
            'connection_pool': connection_status,
            'session_manager': session_status,
            'request_handler': handler_status,
            'performance_metrics': self.performance_metrics,
            'system_health': self._calculate_system_health(
                connection_status, session_status, handler_status
            )
        }
    
    def _calculate_system_health(
        self,
        connection_status: Dict[str, Any],
        session_status: Dict[str, Any],
        handler_status: Dict[str, Any]
    ) -> str:
        """Calculate overall system health"""
        # Check connection pool health
        if connection_status['active_connections'] >= connection_status['max_connections'] * 0.9:
            return 'warning'
        
        # Check session manager health
        if session_status['total_sessions'] >= session_status['max_sessions'] * 0.9:
            return 'warning'
        
        # Check request handler health
        if handler_status['active_requests'] >= handler_status['max_concurrent_requests'] * 0.9:
            return 'warning'
        
        # Check for high error rates
        total_requests = handler_status['total_requests_processed'] + handler_status['total_requests_failed']
        if total_requests > 0:
            error_rate = handler_status['total_requests_failed'] / total_requests
            if error_rate > 0.1:  # More than 10% error rate
                return 'critical'
            elif error_rate > 0.05:  # More than 5% error rate
                return 'warning'
        
        return 'healthy'
    
    async def reset_system(self):
        """Reset all system components (useful for testing)"""
        if self._running:
            await self.stop()
        
        # Reset all components
        await self.connection_pool.reset_pool()
        await self.session_manager.reset_stats()
        await self.request_handler.reset_stats()
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'peak_concurrent_users': 0,
            'system_uptime': 0.0
        }
        
        logger.info("System reset completed")
    
    async def graceful_shutdown(self, timeout: float = 30.0):
        """
        Perform graceful shutdown
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Starting graceful shutdown...")
        
        try:
            # Stop accepting new requests
            self._running = False
            
            # Wait for existing requests to complete
            await asyncio.wait_for(self.stop(), timeout=timeout)
            
            logger.info("Graceful shutdown completed")
            
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timed out, forcing stop")
            await self.stop()
