"""
Secure Authentication & Session Management
Handles secure credential injection, session persistence, and CAPTCHA queue management.
"""

import os
import json
import asyncio
import logging
import hashlib
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import aiofiles
import pickle

logger = logging.getLogger(__name__)

@dataclass
class CaptchaChallenge:
    """Represents a CAPTCHA challenge in the queue."""
    id: str
    platform: str
    task_id: str
    captcha_type: str
    image_data: Optional[bytes]
    challenge_data: Dict[str, Any]
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    cooldown_until: Optional[datetime] = None
    
    def is_ready_for_retry(self) -> bool:
        """Check if challenge is ready for retry."""
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False
        return self.retry_count < self.max_retries

@dataclass
class SessionData:
    """Represents persistent session data."""
    platform: str
    account_email: str
    cookies: Dict[str, Any]
    local_storage: Dict[str, Any]
    session_storage: Dict[str, Any]
    user_agent: str
    fingerprint: Dict[str, Any]
    created_at: datetime
    last_used: datetime
    expires_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        # Session is valid if used within last 24 hours
        return (datetime.now() - self.last_used).total_seconds() < 86400

class SecureAuthManager:
    """Manages secure authentication, session persistence, and CAPTCHA queue."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Session storage
        self.sessions_file = config.get('sessions_file', 'encrypted_sessions.dat')
        self.sessions: Dict[str, SessionData] = {}
        
        # CAPTCHA queue
        self.captcha_queue: List[CaptchaChallenge] = []
        self.captcha_handlers = {}
        
        # Load existing sessions
        asyncio.create_task(self._load_sessions())
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from environment or generate new one."""
        # Try to get key from environment
        env_key = os.getenv('SURVEY_ENCRYPTION_KEY')
        if env_key:
            return base64.urlsafe_b64decode(env_key.encode())
        
        # Try to load from file
        key_file = self.config.get('encryption_key_file', '.encryption_key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        
        # Generate new key
        password = os.getenv('SURVEY_MASTER_PASSWORD', 'default_password_change_me').encode()
        salt = os.getenv('SURVEY_SALT', 'default_salt_change_me').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        # Save key to file
        with open(key_file, 'wb') as f:
            f.write(key)
        
        logger.info("Generated new encryption key")
        return key
    
    def get_credentials_from_env(self, platform: str) -> Optional[Dict[str, str]]:
        """Get credentials from environment variables."""
        platform_upper = platform.upper()
        
        # Try platform-specific environment variables
        email = os.getenv(f'{platform_upper}_EMAIL')
        password = os.getenv(f'{platform_upper}_PASSWORD')
        
        if email and password:
            return {'email': email, 'password': password}
        
        # Try generic environment variables
        email = os.getenv('SURVEY_EMAIL')
        password = os.getenv('SURVEY_PASSWORD')
        
        if email and password:
            return {'email': email, 'password': password}
        
        return None
    
    def get_credentials_from_secrets_manager(self, platform: str) -> Optional[Dict[str, str]]:
        """Get credentials from external secrets manager (AWS Secrets Manager, etc.)."""
        # This would integrate with AWS Secrets Manager, Azure Key Vault, etc.
        # For now, return None - can be extended based on deployment environment
        secrets_manager_url = os.getenv('SECRETS_MANAGER_URL')
        if not secrets_manager_url:
            return None
        
        # TODO: Implement actual secrets manager integration
        logger.info(f"Secrets manager integration not implemented for {platform}")
        return None
    
    def encrypt_credentials(self, credentials: Dict[str, str]) -> str:
        """Encrypt credentials for secure storage."""
        credentials_json = json.dumps(credentials)
        encrypted_data = self.cipher_suite.encrypt(credentials_json.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, str]:
        """Decrypt stored credentials."""
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_credentials.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return {}
    
    async def _load_sessions(self):
        """Load encrypted sessions from disk."""
        if not os.path.exists(self.sessions_file):
            return
        
        try:
            async with aiofiles.open(self.sessions_file, 'rb') as f:
                encrypted_data = await f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            sessions_data = pickle.loads(decrypted_data)
            
            # Convert to SessionData objects and filter valid sessions
            for session_key, session_dict in sessions_data.items():
                session = SessionData(**session_dict)
                if session.is_valid():
                    self.sessions[session_key] = session
                else:
                    logger.info(f"Removing expired session for {session.account_email}")
            
            logger.info(f"Loaded {len(self.sessions)} valid sessions")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self.sessions = {}
    
    async def _save_sessions(self):
        """Save encrypted sessions to disk."""
        try:
            # Convert SessionData objects to dictionaries
            sessions_data = {k: asdict(v) for k, v in self.sessions.items()}
            
            # Serialize and encrypt
            serialized_data = pickle.dumps(sessions_data)
            encrypted_data = self.cipher_suite.encrypt(serialized_data)
            
            async with aiofiles.open(self.sessions_file, 'wb') as f:
                await f.write(encrypted_data)
            
            logger.debug(f"Saved {len(self.sessions)} sessions")
            
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def get_session_key(self, platform: str, account_email: str) -> str:
        """Generate session key for platform and account."""
        return hashlib.sha256(f"{platform}:{account_email}".encode()).hexdigest()
    
    async def get_session(self, platform: str, account_email: str) -> Optional[SessionData]:
        """Get existing session if valid."""
        session_key = self.get_session_key(platform, account_email)
        session = self.sessions.get(session_key)
        
        if session and session.is_valid():
            session.last_used = datetime.now()
            await self._save_sessions()
            return session
        
        # Remove invalid session
        if session:
            del self.sessions[session_key]
            await self._save_sessions()
        
        return None
    
    async def save_session(self, platform: str, account_email: str, cookies: Dict[str, Any], 
                          local_storage: Dict[str, Any], session_storage: Dict[str, Any],
                          user_agent: str, fingerprint: Dict[str, Any]):
        """Save session data for reuse."""
        session_key = self.get_session_key(platform, account_email)
        
        session = SessionData(
            platform=platform,
            account_email=account_email,
            cookies=cookies,
            local_storage=local_storage,
            session_storage=session_storage,
            user_agent=user_agent,
            fingerprint=fingerprint,
            created_at=datetime.now(),
            last_used=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)  # Sessions expire after 7 days
        )
        
        self.sessions[session_key] = session
        await self._save_sessions()
        
        logger.info(f"Saved session for {account_email} on {platform}")
    
    async def restore_session(self, browser_manager, platform: str, account_email: str) -> bool:
        """Restore browser session from saved data."""
        session = await self.get_session(platform, account_email)
        if not session:
            return False
        
        try:
            # Restore cookies
            for cookie in session.cookies.get('cookies', []):
                await browser_manager.page.context.add_cookies([cookie])
            
            # Restore local storage
            for key, value in session.local_storage.items():
                await browser_manager.page.evaluate(f"localStorage.setItem('{key}', '{value}')")
            
            # Restore session storage
            for key, value in session.session_storage.items():
                await browser_manager.page.evaluate(f"sessionStorage.setItem('{key}', '{value}')")
            
            logger.info(f"Restored session for {account_email} on {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session: {e}")
            return False
    
    async def add_captcha_challenge(self, platform: str, task_id: str, captcha_type: str,
                                  image_data: Optional[bytes] = None, 
                                  challenge_data: Dict[str, Any] = None) -> str:
        """Add CAPTCHA challenge to queue."""
        challenge_id = hashlib.sha256(f"{platform}:{task_id}:{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        challenge = CaptchaChallenge(
            id=challenge_id,
            platform=platform,
            task_id=task_id,
            captcha_type=captcha_type,
            image_data=image_data,
            challenge_data=challenge_data or {},
            created_at=datetime.now()
        )
        
        self.captcha_queue.append(challenge)
        logger.info(f"Added CAPTCHA challenge {challenge_id} to queue")
        
        return challenge_id
    
    def get_captcha_queue(self) -> List[Dict[str, Any]]:
        """Get current CAPTCHA queue status."""
        return [
            {
                'id': challenge.id,
                'platform': challenge.platform,
                'task_id': challenge.task_id,
                'captcha_type': challenge.captcha_type,
                'created_at': challenge.created_at.isoformat(),
                'retry_count': challenge.retry_count,
                'ready_for_retry': challenge.is_ready_for_retry(),
                'cooldown_until': challenge.cooldown_until.isoformat() if challenge.cooldown_until else None
            }
            for challenge in self.captcha_queue
        ]
    
    async def retry_captcha_challenge(self, challenge_id: str) -> Optional[CaptchaChallenge]:
        """Get CAPTCHA challenge for retry."""
        for challenge in self.captcha_queue:
            if challenge.id == challenge_id and challenge.is_ready_for_retry():
                challenge.retry_count += 1
                challenge.cooldown_until = datetime.now() + timedelta(minutes=5 * challenge.retry_count)
                return challenge
        
        return None
    
    async def resolve_captcha_challenge(self, challenge_id: str, success: bool = True):
        """Mark CAPTCHA challenge as resolved."""
        self.captcha_queue = [c for c in self.captcha_queue if c.id != challenge_id]
        
        if success:
            logger.info(f"CAPTCHA challenge {challenge_id} resolved successfully")
        else:
            logger.warning(f"CAPTCHA challenge {challenge_id} failed to resolve")
    
    async def cleanup_expired_challenges(self):
        """Remove expired CAPTCHA challenges."""
        cutoff_time = datetime.now() - timedelta(hours=1)  # Remove challenges older than 1 hour
        
        initial_count = len(self.captcha_queue)
        self.captcha_queue = [
            c for c in self.captcha_queue 
            if c.created_at > cutoff_time and c.retry_count < c.max_retries
        ]
        
        removed_count = initial_count - len(self.captcha_queue)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired CAPTCHA challenges")
    
    async def get_secure_credentials(self, platform: str) -> Optional[Dict[str, str]]:
        """Get credentials from the most secure available source."""
        # Try secrets manager first
        credentials = self.get_credentials_from_secrets_manager(platform)
        if credentials:
            logger.info(f"Retrieved credentials for {platform} from secrets manager")
            return credentials
        
        # Try environment variables
        credentials = self.get_credentials_from_env(platform)
        if credentials:
            logger.info(f"Retrieved credentials for {platform} from environment variables")
            return credentials
        
        logger.warning(f"No secure credentials found for {platform}")
        return None
    
    async def start_periodic_cleanup(self):
        """Start periodic cleanup of expired sessions and challenges."""
        while True:
            try:
                await self.cleanup_expired_challenges()
                
                # Clean up expired sessions
                initial_session_count = len(self.sessions)
                self.sessions = {k: v for k, v in self.sessions.items() if v.is_valid()}
                
                removed_sessions = initial_session_count - len(self.sessions)
                if removed_sessions > 0:
                    await self._save_sessions()
                    logger.info(f"Cleaned up {removed_sessions} expired sessions")
                
                # Wait 10 minutes before next cleanup
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error