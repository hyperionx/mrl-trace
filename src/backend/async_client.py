"""
Async Client Library for easy integration with the async backend
"""
import asyncio
import aiohttp
import json
import base64
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AsyncResponse:
    """Response from async backend"""
    success: bool
    request_id: str
    message: str
    data: Optional[Dict[str, Any]] = None
    status_code: int = 200


class AsyncBackendClient:
    """
    Client for interacting with the async backend
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Start the client session"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._headers
            )
            logger.info("Async backend client started")
    
    async def stop(self):
        """Stop the client session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Async backend client stopped")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> AsyncResponse:
        """Make a request to the backend"""
        if self.session is None:
            raise RuntimeError("Client not started. Call start() first.")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params
            ) as response:
                response_data = await response.json()
                
                return AsyncResponse(
                    success=response_data.get('success', False),
                    request_id=response_data.get('request_id', ''),
                    message=response_data.get('message', ''),
                    data=response_data.get('data'),
                    status_code=response.status
                )
                
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            return AsyncResponse(
                success=False,
                request_id='',
                message=f"Request failed: {str(e)}",
                status_code=500
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return AsyncResponse(
                success=False,
                request_id='',
                message=f"Unexpected error: {str(e)}",
                status_code=500
            )
    
    async def health_check(self) -> AsyncResponse:
        """Check backend health"""
        return await self._make_request('GET', '/health')
    
    async def create_session(self, user_id: str, api_key: str) -> AsyncResponse:
        """Create a new user session"""
        params = {'user_id': user_id, 'api_key': api_key}
        return await self._make_request('POST', '/session/create', params=params)
    
    async def submit_chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
        priority: Union[RequestPriority, str] = RequestPriority.NORMAL,
        timeout: float = 30.0
    ) -> AsyncResponse:
        """Submit a chat request"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        data = {
            'message': message,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/chat', data=data, params=params)
    
    async def execute_tool(
        self,
        session_id: str,
        user_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        priority: Union[RequestPriority, str] = RequestPriority.NORMAL,
        timeout: float = 60.0
    ) -> AsyncResponse:
        """Execute a tool"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        data = {
            'tool_name': tool_name,
            'parameters': parameters,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/tool/execute', data=data, params=params)
    
    async def submit_analysis(
        self,
        session_id: str,
        user_id: str,
        analysis_type: str,
        parameters: Dict[str, Any],
        priority: Union[RequestPriority, str] = RequestPriority.NORMAL,
        timeout: float = 120.0
    ) -> AsyncResponse:
        """Submit a data analysis request"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        data = {
            'analysis_type': analysis_type,
            'parameters': parameters,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/analysis', data=data, params=params)
    
    async def update_workflow(
        self,
        session_id: str,
        user_id: str,
        action: str,
        data: Dict[str, Any],
        priority: Union[RequestPriority, str] = RequestPriority.HIGH,
        timeout: float = 30.0
    ) -> AsyncResponse:
        """Update workflow"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        request_data = {
            'action': action,
            'data': data,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/workflow/update', data=request_data, params=params)
    
    async def upload_file(
        self,
        session_id: str,
        user_id: str,
        filename: str,
        file_data: Union[bytes, str],
        priority: Union[RequestPriority, str] = RequestPriority.NORMAL,
        timeout: float = 60.0
    ) -> AsyncResponse:
        """Upload a file"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        # Convert file data to base64 if it's bytes
        if isinstance(file_data, bytes):
            file_data = base64.b64encode(file_data).decode('utf-8')
        
        data = {
            'filename': filename,
            'file_data': file_data,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/file/upload', data=data, params=params)
    
    async def train_model(
        self,
        session_id: str,
        user_id: str,
        model_type: str,
        parameters: Dict[str, Any],
        priority: Union[RequestPriority, str] = RequestPriority.LOW,
        timeout: float = 300.0
    ) -> AsyncResponse:
        """Train a machine learning model"""
        if isinstance(priority, RequestPriority):
            priority = priority.value
        
        data = {
            'model_type': model_type,
            'parameters': parameters,
            'priority': priority,
            'timeout': timeout
        }
        
        params = {'session_id': session_id, 'user_id': user_id}
        return await self._make_request('POST', '/model/train', data=data, params=params)
    
    async def get_request_status(self, request_id: str) -> AsyncResponse:
        """Get the status of a request"""
        return await self._make_request('GET', f'/request/{request_id}/status')
    
    async def cancel_request(self, request_id: str) -> AsyncResponse:
        """Cancel a request"""
        return await self._make_request('DELETE', f'/request/{request_id}')
    
    async def get_session_info(self, session_id: str) -> AsyncResponse:
        """Get session information"""
        return await self._make_request('GET', f'/session/{session_id}')
    
    async def get_system_stats(self) -> AsyncResponse:
        """Get system statistics"""
        return await self._make_request('GET', '/stats')
    
    async def wait_for_request_completion(
        self,
        request_id: str,
        check_interval: float = 1.0,
        max_wait_time: float = 300.0
    ) -> AsyncResponse:
        """
        Wait for a request to complete
        
        Args:
            request_id: Request ID to wait for
            check_interval: How often to check status (seconds)
            max_wait_time: Maximum time to wait (seconds)
            
        Returns:
            Final request status
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if we've exceeded max wait time
            if asyncio.get_event_loop().time() - start_time > max_wait_time:
                return AsyncResponse(
                    success=False,
                    request_id=request_id,
                    message="Request timed out waiting for completion",
                    status_code=408
                )
            
            # Get request status
            status_response = await self.get_request_status(request_id)
            
            if not status_response.success:
                return status_response
            
            # Check if request is completed
            if status_response.data and status_response.data.get('completed', False):
                return status_response
            
            # Wait before checking again
            await asyncio.sleep(check_interval)
    
    async def batch_submit_requests(
        self,
        session_id: str,
        user_id: str,
        requests: List[Dict[str, Any]]
    ) -> List[AsyncResponse]:
        """
        Submit multiple requests in batch
        
        Args:
            session_id: Session ID
            user_id: User ID
            requests: List of request dictionaries
            
        Returns:
            List of responses
        """
        tasks = []
        
        for request_data in requests:
            request_type = request_data.get('type')
            if request_type == 'chat':
                task = self.submit_chat(
                    session_id, user_id,
                    request_data['message'],
                    request_data.get('priority', RequestPriority.NORMAL),
                    request_data.get('timeout', 30.0)
                )
            elif request_type == 'tool':
                task = self.execute_tool(
                    session_id, user_id,
                    request_data['tool_name'],
                    request_data.get('parameters', {}),
                    request_data.get('priority', RequestPriority.NORMAL),
                    request_data.get('timeout', 60.0)
                )
            elif request_type == 'analysis':
                task = self.submit_analysis(
                    session_id, user_id,
                    request_data['analysis_type'],
                    request_data.get('parameters', {}),
                    request_data.get('priority', RequestPriority.NORMAL),
                    request_data.get('timeout', 120.0)
                )
            else:
                logger.warning(f"Unknown request type: {request_type}")
                continue
            
            tasks.append(task)
        
        # Execute all requests concurrently
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error responses
            processed_responses = []
            for response in responses:
                if isinstance(response, Exception):
                    processed_responses.append(AsyncResponse(
                        success=False,
                        request_id='',
                        message=f"Request failed: {str(response)}",
                        status_code=500
                    ))
                else:
                    processed_responses.append(response)
            
            return processed_responses
        
        return []


# Convenience functions for quick usage
async def quick_chat(
    message: str,
    user_id: str = "default_user",
    api_key: str = "default_key",
    base_url: str = "http://localhost:8000"
) -> AsyncResponse:
    """Quick chat function for simple usage"""
    async with AsyncBackendClient(base_url) as client:
        # Create session
        session_response = await client.create_session(user_id, api_key)
        if not session_response.success:
            return session_response
        
        session_id = session_response.data['session_id']
        
        # Submit chat
        return await client.submit_chat(session_id, user_id, message)


async def quick_tool_execution(
    tool_name: str,
    parameters: Dict[str, Any],
    user_id: str = "default_user",
    api_key: str = "default_key",
    base_url: str = "http://localhost:8000"
) -> AsyncResponse:
    """Quick tool execution function for simple usage"""
    async with AsyncBackendClient(base_url) as client:
        # Create session
        session_response = await client.create_session(user_id, api_key)
        if not session_response.success:
            return session_response
        
        session_id = session_response.data['session_id']
        
        # Execute tool
        return await client.execute_tool(session_id, user_id, tool_name, parameters)


# Example usage
async def example_usage():
    """Example of how to use the async client"""
    async with AsyncBackendClient() as client:
        # Check health
        health = await client.health_check()
        print(f"Backend health: {health.message}")
        
        # Create session
        session_response = await client.create_session("user123", "api_key_123")
        if session_response.success:
            session_id = session_response.data['session_id']
            print(f"Session created: {session_id}")
            
            # Submit chat request
            chat_response = await client.submit_chat(
                session_id, "user123", "Hello, how are you?"
            )
            print(f"Chat submitted: {chat_response.message}")
            
            # Wait for completion
            final_status = await client.wait_for_request_completion(chat_response.request_id)
            print(f"Final status: {final_status.message}")
            
            # Execute tool
            tool_response = await client.execute_tool(
                session_id, "user123", "data_analyze", {}
            )
            print(f"Tool executed: {tool_response.message}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
