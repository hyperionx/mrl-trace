"""
Session Manager for handling user sessions efficiently
"""
import asyncio
import time
import uuid
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    api_key: str
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    data_cache: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    request_count: int = 0
    error_count: int = 0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.request_count += 1
    
    def is_expired(self, max_idle_time: int = 3600) -> bool:
        """Check if session is expired"""
        return (datetime.now() - self.last_activity).total_seconds() > max_idle_time


class SessionManager:
    """
    Manages user sessions for efficient request handling
    """
    
    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # 1 hour default
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id mapping
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats = {
            'total_sessions_created': 0,
            'total_sessions_expired': 0,
            'total_requests_processed': 0,
            'total_errors': 0
        }
    
    async def start(self):
        """Start the session manager and cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session manager started")
    
    async def stop(self):
        """Stop the session manager and cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Session manager stopped")
    
    async def create_session(self, user_id: str, api_key: str) -> str:
        """
        Create a new session for a user
        
        Args:
            user_id: Unique user identifier
            api_key: User's API key
            
        Returns:
            Session ID
        """
        async with self._lock:
            # Check if user already has an active session
            if user_id in self.user_sessions:
                existing_session_id = self.user_sessions[user_id]
                if existing_session_id in self.sessions:
                    existing_session = self.sessions[existing_session_id]
                    if existing_session.is_active and not existing_session.is_expired(self.session_timeout):
                        # Reactivate existing session
                        existing_session.is_active = True
                        existing_session.update_activity()
                        existing_session.api_key = api_key
                        logger.debug(f"Reactivated existing session {existing_session_id} for user {user_id}")
                        return existing_session_id
            
            # Check if we can create a new session
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest expired session
                await self._remove_oldest_expired_session()
            
            # Create new session
            session_id = str(uuid.uuid4())
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                api_key=api_key
            )
            
            self.sessions[session_id] = session
            self.user_sessions[user_id] = session_id
            self._stats['total_sessions_created'] += 1
            
            logger.info(f"Created new session {session_id} for user {user_id}")
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session by ID
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            UserSession if found and active, None otherwise
        """
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_active and not session.is_expired(self.session_timeout):
                    session.update_activity()
                    return session
                else:
                    # Session expired or inactive, remove it
                    await self._remove_session(session_id)
            return None
    
    async def get_session_by_user(self, user_id: str) -> Optional[UserSession]:
        """
        Get a session by user ID
        
        Args:
            user_id: User ID to get session for
            
        Returns:
            UserSession if found and active, None otherwise
        """
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            return await self.get_session(session_id)
        return None
    
    async def update_session_data(self, session_id: str, **kwargs) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session ID to update
            **kwargs: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_active:
                    for key, value in kwargs.items():
                        if hasattr(session, key):
                            setattr(session, key, value)
                        else:
                            # Store in data_cache for custom data
                            session.data_cache[key] = value
                    
                    session.update_activity()
                    return True
            return False
    
    async def add_chat_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a chat message to session history
        
        Args:
            session_id: Session ID to add message to
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_active:
                    session.chat_history.append({
                        'role': role,
                        'content': content,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Keep only last 100 messages to prevent memory issues
                    if len(session.chat_history) > 100:
                        session.chat_history = session.chat_history[-100:]
                    
                    session.update_activity()
                    return True
            return False
    
    async def deactivate_session(self, session_id: str):
        """
        Deactivate a session
        
        Args:
            session_id: Session ID to deactivate
        """
        async with self._lock:
            await self._remove_session(session_id)
    
    async def _remove_session(self, session_id: str):
        """Remove a session completely"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Remove from user mapping
            if session.user_id in self.user_sessions:
                del self.user_sessions[session.user_id]
            
            # Remove session
            del self.sessions[session_id]
            
            logger.debug(f"Removed session {session_id}")
    
    async def _remove_oldest_expired_session(self):
        """Remove the oldest expired session"""
        oldest_session = None
        oldest_time = None
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                if oldest_time is None or session.last_activity < oldest_time:
                    oldest_session = session_id
                    oldest_time = session.last_activity
        
        if oldest_session:
            await self._remove_session(oldest_session)
            self._stats['total_sessions_expired'] += 1
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        expired_sessions = []
        
        async with self._lock:
            for session_id, session in self.sessions.items():
                if session.is_expired(self.session_timeout):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._remove_session(session_id)
                self._stats['total_sessions_expired'] += 1
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics for monitoring"""
        active_sessions = sum(1 for s in self.sessions.values() if s.is_active and not s.is_expired(self.session_timeout))
        
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': active_sessions,
            'max_sessions': self.max_sessions,
            'session_timeout': self.session_timeout,
            **self._stats
        }
    
    async def reset_stats(self):
        """Reset statistics (useful for testing)"""
        async with self._lock:
            self._stats = {
                'total_sessions_created': 0,
                'total_sessions_expired': 0,
                'total_requests_processed': 0,
                'total_errors': 0
            }
