"""
Request Handler for processing user requests asynchronously
"""
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import traceback

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class RequestType(Enum):
    """Types of requests"""
    CHAT = "chat"
    TOOL_EXECUTION = "tool_execution"
    DATA_ANALYSIS = "data_analysis"
    WORKFLOW_UPDATE = "workflow_update"
    FILE_UPLOAD = "file_upload"
    MODEL_TRAINING = "model_training"


@dataclass
class Request:
    """Request information"""
    request_id: str
    session_id: str
    user_id: str
    request_type: RequestType
    priority: RequestPriority
    data: Dict[str, Any]
    created_at: float
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    completed: bool = False


class RequestHandler:
    """
    Handles user requests asynchronously with priority queuing
    """
    
    def __init__(self, max_concurrent_requests: int = 20, max_queue_size: int = 100):
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        
        # Priority queues for different request types
        self.request_queues: Dict[RequestType, asyncio.PriorityQueue] = {
            request_type: asyncio.PriorityQueue(maxsize=max_queue_size)
            for request_type in RequestType
        }
        
        # Active requests
        self.active_requests: Dict[str, Request] = {}
        self.request_workers: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'total_requests_processed': 0,
            'total_requests_failed': 0,
            'total_requests_timeout': 0,
            'average_processing_time': 0.0,
            'queue_lengths': {rt.value: 0 for rt in RequestType}
        }
        
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the request handler"""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks for each request type
        for request_type in RequestType:
            worker = asyncio.create_task(self._request_worker(request_type))
            self.request_workers.append(worker)
        
        # Start statistics update task
        stats_task = asyncio.create_task(self._update_stats_loop())
        self.request_workers.append(stats_task)
        
        logger.info(f"Request handler started with {len(RequestType)} workers")
    
    async def stop(self):
        """Stop the request handler"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all worker tasks
        for worker in self.request_workers:
            worker.cancel()
        
        # Wait for all workers to finish
        if self.request_workers:
            await asyncio.gather(*self.request_workers, return_exceptions=True)
        
        self.request_workers.clear()
        logger.info("Request handler stopped")
    
    async def submit_request(
        self,
        session_id: str,
        user_id: str,
        request_type: RequestType,
        data: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit a new request for processing
        
        Args:
            session_id: Session ID
            user_id: User ID
            request_type: Type of request
            data: Request data
            priority: Request priority
            timeout: Request timeout in seconds
            callback: Optional callback function
            
        Returns:
            Request ID
        """
        import uuid
        
        request_id = str(uuid.uuid4())
        request = Request(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            request_type=request_type,
            priority=priority,
            data=data,
            created_at=time.time(),
            timeout=timeout,
            callback=callback
        )
        
        # Calculate priority score (higher priority = lower score for queue ordering)
        priority_score = (RequestPriority.URGENT.value - priority.value, request.created_at)
        
        try:
            # Add to appropriate queue
            await self.request_queues[request_type].put((priority_score, request))
            
            # Update queue length statistics
            self.stats['queue_lengths'][request_type.value] = self.request_queues[request_type].qsize()
            
            logger.debug(f"Submitted request {request_id} of type {request_type.value} with priority {priority.value}")
            return request_id
            
        except asyncio.QueueFull:
            logger.warning(f"Request queue full for type {request_type.value}")
            raise RuntimeError(f"Request queue full for type {request_type.value}")
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a request
        
        Args:
            request_id: Request ID to check
            
        Returns:
            Request status dictionary or None if not found
        """
        async with self._lock:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return {
                    'request_id': request.request_id,
                    'status': 'processing' if not request.completed else 'completed',
                    'created_at': request.created_at,
                    'completed': request.completed,
                    'result': request.result,
                    'error': request.error,
                    'processing_time': time.time() - request.created_at if request.completed else None
                }
            return None
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request
        
        Args:
            request_id: Request ID to cancel
            
        Returns:
            True if cancelled, False if not found or already processing
        """
        # Note: This is a simplified implementation
        # In a real system, you'd need more sophisticated cancellation logic
        logger.info(f"Request cancellation requested for {request_id}")
        return True
    
    async def _request_worker(self, request_type: RequestType):
        """Worker task for processing requests of a specific type"""
        queue = self.request_queues[request_type]
        
        while self._running:
            try:
                # Get next request from queue
                priority_score, request = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Process the request
                await self._process_request(request)
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in request worker for {request_type.value}: {e}")
                continue
    
    async def _process_request(self, request: Request):
        """Process a single request"""
        start_time = time.time()
        
        try:
            # Add to active requests
            async with self._lock:
                self.active_requests[request.request_id] = request
            
            # Process based on request type
            if request.request_type == RequestType.CHAT:
                result = await self._handle_chat_request(request)
            elif request.request_type == RequestType.TOOL_EXECUTION:
                result = await self._handle_tool_execution(request)
            elif request.type == RequestType.DATA_ANALYSIS:
                result = await self._handle_data_analysis(request)
            elif request.request_type == RequestType.WORKFLOW_UPDATE:
                result = await self._handle_workflow_update(request)
            elif request.request_type == RequestType.FILE_UPLOAD:
                result = await self._handle_file_upload(request)
            elif request.request_type == RequestType.MODEL_TRAINING:
                result = await self._handle_model_training(request)
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
            
            # Set result and mark as completed
            request.result = result
            request.completed = True
            
            # Call callback if provided
            if request.callback:
                try:
                    await request.callback(result)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_requests_processed'] += 1
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_requests_processed'] - 1) + processing_time) /
                self.stats['total_requests_processed']
            )
            
            logger.info(f"Request {request.request_id} completed in {processing_time:.2f}s")
            
        except asyncio.TimeoutError:
            request.error = "Request timeout"
            request.completed = True
            self.stats['total_requests_timeout'] += 1
            logger.warning(f"Request {request.request_id} timed out")
            
        except Exception as e:
            request.error = str(e)
            request.completed = True
            self.stats['total_requests_failed'] += 1
            logger.error(f"Error processing request {request.request_id}: {e}")
            logger.debug(f"Request traceback: {traceback.format_exc()}")
            
        finally:
            # Remove from active requests
            async with self._lock:
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
    
    async def _handle_chat_request(self, request: Request) -> Dict[str, Any]:
        """Handle chat request"""
        # This would integrate with your AI chat system
        # For now, return a placeholder response
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            'type': 'chat_response',
            'content': f"Processed chat request for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _handle_tool_execution(self, request: Request) -> Dict[str, Any]:
        """Handle tool execution request"""
        # This would integrate with your tool call system
        await asyncio.sleep(0.2)  # Simulate processing time
        return {
            'type': 'tool_execution',
            'tool': request.data.get('tool'),
            'result': f"Executed tool for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _handle_data_analysis(self, request: Request) -> Dict[str, Any]:
        """Handle data analysis request"""
        # This would integrate with your analysis tools
        await asyncio.sleep(0.5)  # Simulate processing time
        return {
            'type': 'data_analysis',
            'analysis_type': request.data.get('analysis_type'),
            'result': f"Completed analysis for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _handle_workflow_update(self, request: Request) -> Dict[str, Any]:
        """Handle workflow update request"""
        # This would integrate with your workflow system
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            'type': 'workflow_update',
            'step': request.data.get('step'),
            'result': f"Updated workflow for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _handle_file_upload(self, request: Request) -> Dict[str, Any]:
        """Handle file upload request"""
        # This would handle file processing
        await asyncio.sleep(0.3)  # Simulate processing time
        return {
            'type': 'file_upload',
            'filename': request.data.get('filename'),
            'result': f"Processed file for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _handle_model_training(self, request: Request) -> Dict[str, Any]:
        """Handle model training request"""
        # This would handle ML model training
        await asyncio.sleep(1.0)  # Simulate longer processing time
        return {
            'type': 'model_training',
            'model_type': request.data.get('model_type'),
            'result': f"Trained model for user {request.user_id}",
            'timestamp': time.time()
        }
    
    async def _update_stats_loop(self):
        """Background task to update statistics"""
        while self._running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                # Update queue lengths
                for request_type in RequestType:
                    self.stats['queue_lengths'][request_type.value] = self.request_queues[request_type].qsize()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics for monitoring"""
        return {
            'running': self._running,
            'active_requests': len(self.active_requests),
            'max_concurrent_requests': self.max_concurrent_requests,
            'max_queue_size': self.max_queue_size,
            **self.stats
        }
    
    async def reset_stats(self):
        """Reset statistics (useful for testing)"""
        async with self._lock:
            self.stats = {
                'total_requests_processed': 0,
                'total_requests_failed': 0,
                'total_requests_timeout': 0,
                'average_processing_time': 0.0,
                'queue_lengths': {rt.value: 0 for rt in RequestType}
            }
