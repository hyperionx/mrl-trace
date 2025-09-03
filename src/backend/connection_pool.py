"""
Connection Pool for managing API connections efficiently
"""
import asyncio
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a connection"""
    client_id: str
    created_at: float
    last_used: float
    is_active: bool
    error_count: int = 0


class ConnectionPool:
    """
    Manages a pool of API connections for efficient request handling
    """
    
    def __init__(self, max_connections: int = 10, max_idle_time: int = 300):
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time  # 5 minutes default
        self.connections: Dict[str, ConnectionInfo] = {}
        self.active_connections = 0
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the connection pool and cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Connection pool started")
    
    async def stop(self):
        """Stop the connection pool and cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Connection pool stopped")
    
    async def get_connection(self, client_id: str) -> Optional[str]:
        """
        Get an available connection for a client
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Connection ID if available, None otherwise
        """
        async with self._lock:
            # Check if client already has an active connection
            if client_id in self.connections:
                conn_info = self.connections[client_id]
                if conn_info.is_active:
                    conn_info.last_used = time.time()
                    return client_id
            
            # Check if we can create a new connection
            if self.active_connections < self.max_connections:
                conn_info = ConnectionInfo(
                    client_id=client_id,
                    created_at=time.time(),
                    last_used=time.time(),
                    is_active=True
                )
                self.connections[client_id] = conn_info
                self.active_connections += 1
                logger.debug(f"Created new connection for client {client_id}")
                return client_id
            
            # No available connections
            logger.warning(f"No available connections for client {client_id}")
            return None
    
    async def release_connection(self, client_id: str):
        """
        Release a connection back to the pool
        
        Args:
            client_id: Client ID to release connection for
        """
        async with self._lock:
            if client_id in self.connections:
                conn_info = self.connections[client_id]
                if conn_info.is_active:
                    conn_info.is_active = False
                    self.active_connections -= 1
                    logger.debug(f"Released connection for client {client_id}")
    
    async def mark_connection_error(self, client_id: str):
        """
        Mark a connection as having an error
        
        Args:
            client_id: Client ID with connection error
        """
        async with self._lock:
            if client_id in self.connections:
                conn_info = self.connections[client_id]
                conn_info.error_count += 1
                
                # If too many errors, deactivate the connection
                if conn_info.error_count >= 3:
                    if conn_info.is_active:
                        self.active_connections -= 1
                    conn_info.is_active = False
                    logger.warning(f"Deactivated connection for client {client_id} due to errors")
    
    async def _cleanup_loop(self):
        """Background task to clean up idle connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_connections(self):
        """Remove idle connections that exceed max_idle_time"""
        current_time = time.time()
        to_remove = []
        
        async with self._lock:
            for client_id, conn_info in self.connections.items():
                if (current_time - conn_info.last_used) > self.max_idle_time:
                    to_remove.append(client_id)
            
            for client_id in to_remove:
                conn_info = self.connections[client_id]
                if conn_info.is_active:
                    self.active_connections -= 1
                del self.connections[client_id]
                logger.debug(f"Removed idle connection for client {client_id}")
    
    def get_pool_status(self) -> Dict:
        """Get current pool status for monitoring"""
        return {
            'total_connections': len(self.connections),
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'available_connections': self.max_connections - self.active_connections
        }
    
    async def reset_pool(self):
        """Reset the connection pool (useful for testing)"""
        async with self._lock:
            self.connections.clear()
            self.active_connections = 0
            logger.info("Connection pool reset")


class AsyncConnectionPool(ConnectionPool):
    """
    Async-compatible connection pool with additional async features
    """
    
    async def acquire_connection(self, client_id: str, timeout: float = 5.0) -> Optional[str]:
        """
        Acquire a connection with timeout
        
        Args:
            client_id: Client ID to acquire connection for
            timeout: Timeout in seconds
            
        Returns:
            Connection ID if acquired, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            connection = await self.get_connection(client_id)
            if connection:
                return connection
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout acquiring connection for client {client_id}")
        return None
    
    @asynccontextmanager
    async def connection_context(self, client_id: str, timeout: float = 5.0):
        """
        Context manager for automatic connection management
        
        Args:
            client_id: Client ID to manage connection for
            timeout: Timeout for acquiring connection
        """
        connection = await self.acquire_connection(client_id, timeout)
        if not connection:
            raise TimeoutError(f"Could not acquire connection for client {client_id}")
        
        try:
            yield connection
        finally:
            await self.release_connection(client_id)
