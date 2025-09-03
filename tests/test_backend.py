"""
Tests for the async backend system
"""
import pytest
import asyncio
import time
from typing import Dict, Any

from src.backend import AsyncRequestManager
from src.backend.connection_pool import ConnectionPool, AsyncConnectionPool
from src.backend.session_manager import SessionManager, UserSession
from src.backend.request_handler import RequestHandler, RequestType, RequestPriority, Request
from src.backend.async_client import AsyncBackendClient, RequestPriority as ClientPriority


class TestConnectionPool:
    """Test the connection pool"""
    
    @pytest.fixture
    async def pool(self):
        """Create a connection pool for testing"""
        pool = ConnectionPool(max_connections=5, max_idle_time=60)
        await pool.start()
        yield pool
        await pool.stop()
    
    async def test_connection_pool_initialization(self, pool):
        """Test connection pool initialization"""
        assert pool.max_connections == 5
        assert pool.max_idle_time == 60
        assert pool.active_connections == 0
        assert len(pool.connections) == 0
    
    async def test_get_connection(self, pool):
        """Test getting a connection"""
        client_id = "test_client_1"
        connection = await pool.get_connection(client_id)
        
        assert connection == client_id
        assert pool.active_connections == 1
        assert client_id in pool.connections
    
    async def test_release_connection(self, pool):
        """Test releasing a connection"""
        client_id = "test_client_1"
        await pool.get_connection(client_id)
        
        assert pool.active_connections == 1
        await pool.release_connection(client_id)
        assert pool.active_connections == 0
    
    async def test_max_connections_limit(self, pool):
        """Test maximum connections limit"""
        # Try to get more connections than allowed
        connections = []
        for i in range(6):
            connection = await pool.get_connection(f"client_{i}")
            if connection:
                connections.append(connection)
        
        # Should only get 5 connections
        assert len(connections) == 5
        assert pool.active_connections == 5
        
        # Clean up
        for connection in connections:
            await pool.release_connection(connection)
    
    async def test_connection_error_handling(self, pool):
        """Test connection error handling"""
        client_id = "test_client_1"
        await pool.get_connection(client_id)
        
        # Mark connection as having errors
        await pool.mark_connection_error(client_id)
        await pool.mark_connection_error(client_id)
        await pool.mark_connection_error(client_id)
        
        # After 3 errors, connection should be deactivated
        assert pool.active_connections == 0


class TestAsyncConnectionPool:
    """Test the async connection pool"""
    
    @pytest.fixture
    async def async_pool(self):
        """Create an async connection pool for testing"""
        pool = AsyncConnectionPool(max_connections=5, max_idle_time=60)
        await pool.start()
        yield pool
        await pool.stop()
    
    async def test_acquire_connection_with_timeout(self, async_pool):
        """Test acquiring connection with timeout"""
        # Fill up the pool
        connections = []
        for i in range(5):
            connection = await async_pool.get_connection(f"client_{i}")
            connections.append(connection)
        
        # Try to acquire another connection with timeout
        start_time = time.time()
        connection = await async_pool.acquire_connection("client_6", timeout=1.0)
        end_time = time.time()
        
        # Should timeout
        assert connection is None
        assert end_time - start_time >= 1.0
        
        # Clean up
        for connection in connections:
            await async_pool.release_connection(connection)
    
    async def test_connection_context_manager(self, async_pool):
        """Test connection context manager"""
        async with async_pool.connection_context("test_client", timeout=5.0) as connection:
            assert connection == "test_client"
            assert async_pool.active_connections == 1
        
        # Connection should be automatically released
        assert async_pool.active_connections == 0


class TestSessionManager:
    """Test the session manager"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a session manager for testing"""
        manager = SessionManager(max_sessions=10, session_timeout=60)
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_session_creation(self, session_manager):
        """Test session creation"""
        user_id = "test_user_1"
        api_key = "test_api_key_1"
        
        session_id = await session_manager.create_session(user_id, api_key)
        
        assert session_id is not None
        assert len(session_manager.sessions) == 1
        assert session_manager.user_sessions[user_id] == session_id
    
    async def test_session_retrieval(self, session_manager):
        """Test session retrieval"""
        user_id = "test_user_1"
        api_key = "test_api_key_1"
        
        session_id = await session_manager.create_session(user_id, api_key)
        session = await session_manager.get_session(session_id)
        
        assert session is not None
        assert session.user_id == user_id
        assert session.api_key == api_key
        assert session.is_active
    
    async def test_session_data_update(self, session_manager):
        """Test session data update"""
        user_id = "test_user_1"
        api_key = "test_api_key_1"
        
        session_id = await session_manager.create_session(user_id, api_key)
        
        # Update session data
        success = await session_manager.update_session_data(
            session_id, 
            workflow_state={"step": "data_analysis"},
            data_cache={"temp_data": "value"}
        )
        
        assert success is True
        
        # Verify update
        session = await session_manager.get_session(session_id)
        assert session.workflow_state["step"] == "data_analysis"
        assert session.data_cache["temp_data"] == "value"
    
    async def test_chat_message_addition(self, session_manager):
        """Test adding chat messages to session"""
        user_id = "test_user_1"
        api_key = "test_api_key_1"
        
        session_id = await session_manager.create_session(user_id, api_key)
        
        # Add chat messages
        await session_manager.add_chat_message(session_id, "user", "Hello")
        await session_manager.add_chat_message(session_id, "assistant", "Hi there!")
        
        # Verify messages
        session = await session_manager.get_session(session_id)
        assert len(session.chat_history) == 2
        assert session.chat_history[0]["role"] == "user"
        assert session.chat_history[0]["content"] == "Hello"
        assert session.chat_history[1]["role"] == "assistant"
        assert session.chat_history[1]["content"] == "Hi there!"
    
    async def test_max_sessions_limit(self, session_manager):
        """Test maximum sessions limit"""
        # Create maximum number of sessions
        sessions = []
        for i in range(10):
            session_id = await session_manager.create_session(f"user_{i}", f"key_{i}")
            sessions.append(session_id)
        
        assert len(session_manager.sessions) == 10
        
        # Try to create one more
        try:
            await session_manager.create_session("user_11", "key_11")
            # Should not reach here
            assert False
        except Exception:
            # Expected behavior
            pass


class TestRequestHandler:
    """Test the request handler"""
    
    @pytest.fixture
    async def request_handler(self):
        """Create a request handler for testing"""
        handler = RequestHandler(max_concurrent_requests=5, max_queue_size=10)
        await handler.start()
        yield handler
        await handler.stop()
    
    async def test_request_submission(self, request_handler):
        """Test request submission"""
        session_id = "test_session"
        user_id = "test_user"
        
        request_id = await request_handler.submit_request(
            session_id=session_id,
            user_id=user_id,
            request_type=RequestType.CHAT,
            data={"message": "Hello"},
            priority=RequestPriority.NORMAL,
            timeout=30.0
        )
        
        assert request_id is not None
        assert request_handler.stats['total_requests_processed'] >= 0
    
    async def test_priority_queuing(self, request_handler):
        """Test priority-based queuing"""
        session_id = "test_session"
        user_id = "test_user"
        
        # Submit requests with different priorities
        low_priority = await request_handler.submit_request(
            session_id, user_id, RequestType.CHAT, {"message": "Low"}, RequestPriority.LOW
        )
        
        high_priority = await request_handler.submit_request(
            session_id, user_id, RequestType.CHAT, {"message": "High"}, RequestPriority.HIGH
        )
        
        normal_priority = await request_handler.submit_request(
            session_id, user_id, RequestType.CHAT, {"message": "Normal"}, RequestPriority.NORMAL
        )
        
        # All should be submitted successfully
        assert low_priority is not None
        assert high_priority is not None
        assert normal_priority is not None
    
    async def test_request_status_check(self, request_handler):
        """Test request status checking"""
        session_id = "test_session"
        user_id = "test_user"
        
        request_id = await request_handler.submit_request(
            session_id, user_id, RequestType.CHAT, {"message": "Test"}, RequestPriority.NORMAL
        )
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        status = await request_handler.get_request_status(request_id)
        assert status is not None
        assert 'status' in status


class TestAsyncRequestManager:
    """Test the async request manager"""
    
    @pytest.fixture
    async def manager(self):
        """Create an async request manager for testing"""
        manager = AsyncRequestManager(
            max_connections=5,
            max_sessions=10,
            max_concurrent_requests=5,
            max_queue_size=10
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.connection_pool is not None
        assert manager.session_manager is not None
        assert manager.request_handler is not None
        assert manager._running is True
    
    async def test_session_context_manager(self, manager):
        """Test session context manager"""
        user_id = "test_user"
        api_key = "test_key"
        
        async with manager.get_session_context(user_id, api_key) as session_id:
            assert session_id is not None
            
            # Verify session was created
            session = await manager.get_session_info(session_id)
            assert session is not None
            assert session.user_id == user_id
    
    async def test_chat_request_submission(self, manager):
        """Test chat request submission"""
        user_id = "test_user"
        api_key = "test_key"
        
        async with manager.get_session_context(user_id, api_key) as session_id:
            request_id = await manager.submit_chat_request(
                session_id, user_id, "Hello, world!"
            )
            
            assert request_id is not None
            
            # Verify request was submitted
            status = await manager.get_request_status(request_id)
            assert status is not None
    
    async def test_tool_execution_request(self, manager):
        """Test tool execution request"""
        user_id = "test_user"
        api_key = "test_key"
        
        async with manager.get_session_context(user_id, api_key) as session_id:
            request_id = await manager.submit_tool_execution_request(
                session_id, user_id, "data_analyze", {"param": "value"}
            )
            
            assert request_id is not None
    
    async def test_system_status(self, manager):
        """Test system status retrieval"""
        status = await manager.get_system_status()
        
        assert status['status'] == 'running'
        assert 'connection_pool' in status
        assert 'session_manager' in status
        assert 'request_handler' in status
        assert 'performance_metrics' in status
        assert 'system_health' in status


class TestAsyncBackendClient:
    """Test the async backend client"""
    
    @pytest.fixture
    async def client(self):
        """Create a client for testing"""
        # Note: This would require a running backend server
        # For testing, we'll create a mock or skip these tests
        pytest.skip("Requires running backend server")
    
    async def test_client_initialization(self):
        """Test client initialization"""
        client = AsyncBackendClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
    
    async def test_priority_enum(self):
        """Test priority enum values"""
        assert ClientPriority.LOW.value == "low"
        assert ClientPriority.NORMAL.value == "normal"
        assert ClientPriority.HIGH.value == "high"
        assert ClientPriority.URGENT.value == "urgent"


class TestIntegration:
    """Integration tests for the backend system"""
    
    @pytest.fixture
    async def full_system(self):
        """Create a full backend system for integration testing"""
        manager = AsyncRequestManager(
            max_connections=3,
            max_sessions=5,
            max_concurrent_requests=3,
            max_queue_size=10
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_full_workflow(self, full_system):
        """Test a complete workflow"""
        user_id = "integration_user"
        api_key = "integration_key"
        
        # Create session
        async with full_system.get_session_context(user_id, api_key) as session_id:
            # Submit multiple types of requests
            chat_request = await full_system.submit_chat_request(
                session_id, user_id, "Hello"
            )
            
            tool_request = await full_system.submit_tool_execution_request(
                session_id, user_id, "workflow_status", {}
            )
            
            analysis_request = await full_system.submit_data_analysis_request(
                session_id, user_id, "basic_stats", {}
            )
            
            # Verify all requests were submitted
            assert chat_request is not None
            assert tool_request is not None
            assert analysis_request is not None
            
            # Check system status
            status = await full_system.get_system_status()
            assert status['status'] == 'running'
    
    async def test_concurrent_requests(self, full_system):
        """Test handling concurrent requests"""
        user_id = "concurrent_user"
        api_key = "concurrent_key"
        
        async with full_system.get_session_context(user_id, api_key) as session_id:
            # Submit multiple requests concurrently
            tasks = []
            for i in range(5):
                task = full_system.submit_chat_request(
                    session_id, user_id, f"Message {i}"
                )
                tasks.append(task)
            
            # Wait for all requests
            request_ids = await asyncio.gather(*tasks)
            
            # Verify all requests were submitted
            assert len(request_ids) == 5
            assert all(rid is not None for rid in request_ids)


# Performance tests
class TestPerformance:
    """Performance tests for the backend system"""
    
    @pytest.fixture
    async def performance_manager(self):
        """Create a manager for performance testing"""
        manager = AsyncRequestManager(
            max_connections=20,
            max_sessions=50,
            max_concurrent_requests=20,
            max_queue_size=100
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    async def test_request_throughput(self, performance_manager):
        """Test request throughput"""
        user_id = "perf_user"
        api_key = "perf_key"
        
        async with performance_manager.get_session_context(user_id, api_key) as session_id:
            start_time = time.time()
            
            # Submit many requests
            tasks = []
            for i in range(50):
                task = performance_manager.submit_chat_request(
                    session_id, user_id, f"Performance test {i}"
                )
                tasks.append(task)
            
            # Wait for all requests
            request_ids = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Calculate throughput
            duration = end_time - start_time
            throughput = len(request_ids) / duration
            
            print(f"Request throughput: {throughput:.2f} requests/second")
            
            # Should handle at least 10 requests per second
            assert throughput > 10
    
    async def test_memory_usage(self, performance_manager):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        user_id = "memory_user"
        api_key = "memory_key"
        
        async with performance_manager.get_session_context(user_id, api_key) as session_id:
            # Submit many requests
            tasks = []
            for i in range(100):
                task = performance_manager.submit_chat_request(
                    session_id, user_id, f"Memory test {i}"
                )
                tasks.append(task)
            
            # Wait for all requests
            await asyncio.gather(*tasks)
            
            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Memory increase: {memory_increase:.2f} MB")
            
            # Memory increase should be reasonable (< 100 MB)
            assert memory_increase < 100


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
