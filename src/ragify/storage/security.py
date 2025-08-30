"""
Security Manager for comprehensive data protection and access control.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import structlog

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import bcrypt

from ..models import PrivacyLevel
from ..exceptions import SecurityViolationError, AccessDeniedError


class SecurityManager:
    """
    Comprehensive security manager for data protection and access control.
    
    Implements:
    - Symmetric and asymmetric encryption
    - Access control and role-based permissions
    - Audit logging and compliance
    - Key management and rotation
    - Security monitoring and alerts
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        security_level: str = "standard"
    ):
        """
        Initialize the security manager.
        
        Args:
            encryption_key: Symmetric encryption key
            private_key_path: Path to RSA private key
            audit_log_path: Path to audit log file
            security_level: Security level (basic, standard, high, enterprise)
        """
        self.logger = structlog.get_logger(__name__)
        self.security_level = security_level
        
        # Initialize encryption
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
        
        # Initialize asymmetric encryption
        self.private_key = None
        self.public_key = None
        if private_key_path:
            self._load_rsa_keys(private_key_path)
        else:
            self._generate_rsa_keys()
        
        # Initialize audit logging
        self.audit_log_path = audit_log_path or "audit.log"
        self.audit_log = []
        self._load_audit_log()
        
        # Security policies
        self.security_policies = self._get_security_policies()
        
        # Access control
        self.user_roles = {}
        self.role_permissions = self._get_role_permissions()
        self.access_tokens = {}
        
        # Security monitoring
        self.security_events = []
        self.failed_attempts = {}
        self.lockout_threshold = 5
        self.lockout_duration = 300  # 5 minutes
        
        # Key rotation
        self.key_rotation_interval = 30 * 24 * 3600  # 30 days
        self.last_key_rotation = time.time()
        
        self.logger.info(f"Security manager initialized with level: {security_level}")
    
    def _generate_encryption_key(self) -> str:
        """Generate a new symmetric encryption key."""
        return Fernet.generate_key().decode()
    
    def _generate_rsa_keys(self) -> None:
        """Generate RSA key pair for asymmetric encryption."""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Store keys
            self.private_key = private_key
            self.public_key = public_key
            
            self.logger.info("RSA key pair generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate RSA keys: {e}")
            raise SecurityViolationError("key_generation", "RSA", str(e))
    
    def _load_rsa_keys(self, key_path: str) -> None:
        """Load RSA keys from file."""
        try:
            key_file = Path(key_path)
            if not key_file.exists():
                self.logger.warning(f"RSA key file not found: {key_path}")
                self._generate_rsa_keys()
                return
            
            with open(key_file, 'rb') as f:
                private_key_data = f.read()
                self.private_key = serialization.load_pem_private_key(
                    private_key_data,
                    password=None
                )
                self.public_key = self.private_key.public_key()
            
            self.logger.info(f"RSA keys loaded from: {key_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load RSA keys: {e}")
            self._generate_rsa_keys()
    
    def _get_security_policies(self) -> Dict[str, Any]:
        """Get security policies based on security level."""
        policies = {
            "basic": {
                "encryption_required": False,
                "access_logging": True,
                "audit_logging": False,
                "key_rotation": False,
                "session_timeout": 3600,  # 1 hour
                "max_failed_attempts": 3,
                "lockout_duration": 300,  # 5 minutes
            },
            "standard": {
                "encryption_required": True,
                "access_logging": True,
                "audit_logging": True,
                "key_rotation": True,
                "session_timeout": 1800,  # 30 minutes
                "max_failed_attempts": 5,
                "lockout_duration": 600,  # 10 minutes
            },
            "high": {
                "encryption_required": True,
                "access_logging": True,
                "audit_logging": True,
                "key_rotation": True,
                "session_timeout": 900,  # 15 minutes
                "max_failed_attempts": 3,
                "lockout_duration": 1800,  # 30 minutes
                "mfa_required": True,
            },
            "enterprise": {
                "encryption_required": True,
                "access_logging": True,
                "audit_logging": True,
                "key_rotation": True,
                "session_timeout": 600,  # 10 minutes
                "max_failed_attempts": 2,
                "lockout_duration": 3600,  # 1 hour
                "mfa_required": True,
                "compliance_reporting": True,
            }
        }
        
        return policies.get(self.security_level, policies["standard"])
    
    def _get_role_permissions(self) -> Dict[str, List[str]]:
        """Get role-based permissions."""
        return {
            "admin": [
                "read", "write", "delete", "admin", "audit", "security"
            ],
            "manager": [
                "read", "write", "audit"
            ],
            "analyst": [
                "read", "write"
            ],
            "viewer": [
                "read"
            ],
            "guest": [
                "read_public"
            ]
        }
    
    def _load_audit_log(self) -> None:
        """Load audit log from file."""
        try:
            if Path(self.audit_log_path).exists():
                with open(self.audit_log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                event = json.loads(line)
                                self.audit_log.append(event)
                            except json.JSONDecodeError:
                                continue
                
                self.logger.info(f"Loaded {len(self.audit_log)} audit log entries")
        except Exception as e:
            self.logger.warning(f"Failed to load audit log: {e}")
    
    async def encrypt_data(self, data: str, encryption_type: str = "symmetric") -> str:
        """
        Encrypt data using specified encryption method.
        
        Args:
            data: Data to encrypt
            encryption_type: Type of encryption (symmetric, asymmetric, hybrid)
            
        Returns:
            Encrypted data
        """
        try:
            if encryption_type == "symmetric":
                return await self._encrypt_symmetric(data)
            elif encryption_type == "asymmetric":
                return await self._encrypt_asymmetric(data)
            elif encryption_type == "hybrid":
                return await self._encrypt_hybrid(data)
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            raise SecurityViolationError("encryption", encryption_type, str(e))
    
    async def decrypt_data(self, encrypted_data: str, encryption_type: str = "symmetric") -> str:
        """
        Decrypt data using specified encryption method.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            encryption_type: Type of encryption used
            
        Returns:
            Decrypted data
        """
        try:
            if encryption_type == "symmetric":
                return await self._decrypt_symmetric(encrypted_data)
            elif encryption_type == "asymmetric":
                return await self._decrypt_asymmetric(encrypted_data)
            elif encryption_type == "hybrid":
                return await self._decrypt_hybrid(encrypted_data)
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise SecurityViolationError("decryption", encryption_type, str(e))
    
    async def _encrypt_symmetric(self, data: str) -> str:
        """Encrypt data using symmetric encryption (AES)."""
        try:
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.encryption_key.encode()[:32]),
                modes.CBC(iv)
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = data.encode()
            block_size = 16
            padding_length = block_size - (len(padded_data) % block_size)
            padded_data += bytes([padding_length] * padding_length)
            
            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            result = base64.b64encode(iv + encrypted_data).decode()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt symmetrically: {e}")
            raise
    
    async def _decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption (AES)."""
        try:
            # Decode base64
            data = base64.b64decode(encrypted_data.encode())
            
            # Extract IV and encrypted data
            iv = data[:16]
            encrypted = data[16:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.encryption_key.encode()[:32]),
                modes.CBC(iv)
            )
            decryptor = cipher.decryptor()
            
            # Decrypt
            decrypted_data = decryptor.update(encrypted) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted_data[-1]
            decrypted_data = decrypted_data[:-padding_length]
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt symmetrically: {e}")
            raise
    
    async def _encrypt_asymmetric(self, data: str) -> str:
        """Encrypt data using asymmetric encryption (RSA)."""
        try:
            if not self.public_key:
                raise SecurityViolationError("encryption", "RSA", "Public key not available")
            
            # Encrypt data with public key
            encrypted_data = self.public_key.encrypt(
                data.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt asymmetrically: {e}")
            raise
    
    async def _decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption (RSA)."""
        try:
            if not self.private_key:
                raise SecurityViolationError("decryption", "RSA", "Private key not available")
            
            # Decode base64
            data = base64.b64decode(encrypted_data.encode())
            
            # Decrypt data with private key
            decrypted_data = self.private_key.decrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt asymmetrically: {e}")
            raise
    
    async def _encrypt_hybrid(self, data: str) -> str:
        """Encrypt data using hybrid encryption (AES + RSA)."""
        try:
            # Generate random AES key
            aes_key = secrets.token_bytes(32)
            
            # Encrypt data with AES
            aes_encrypted = await self._encrypt_with_key(data, aes_key)
            
            # Encrypt AES key with RSA
            rsa_encrypted_key = await self._encrypt_asymmetric(aes_key.hex())
            
            # Combine encrypted key and data
            result = {
                'encrypted_key': rsa_encrypted_key,
                'encrypted_data': aes_encrypted
            }
            
            return base64.b64encode(json.dumps(result).encode()).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt with hybrid method: {e}")
            raise
    
    async def _decrypt_hybrid(self, encrypted_data: str) -> str:
        """Decrypt data using hybrid encryption."""
        try:
            # Decode base64 and parse JSON
            data = json.loads(base64.b64decode(encrypted_data.encode()).decode())
            
            # Decrypt AES key with RSA
            aes_key = bytes.fromhex(await self._decrypt_asymmetric(data['encrypted_key']))
            
            # Decrypt data with AES key
            decrypted_data = await self._decrypt_with_key(data['encrypted_data'], aes_key)
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt with hybrid method: {e}")
            raise
    
    async def _encrypt_with_key(self, data: str, key: bytes) -> str:
        """Encrypt data with a specific AES key."""
        try:
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Pad data
            padded_data = data.encode()
            block_size = 16
            padding_length = block_size - (len(padded_data) % block_size)
            padded_data += bytes([padding_length] * padding_length)
            
            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            result = base64.b64encode(iv + encrypted_data).decode()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt with key: {e}")
            raise
    
    async def _decrypt_with_key(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt data with a specific AES key."""
        try:
            # Decode base64
            data = base64.b64decode(encrypted_data.encode())
            
            # Extract IV and encrypted data
            iv = data[:16]
            encrypted = data[16:]
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            # Decrypt
            decrypted_data = decryptor.update(encrypted) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted_data[-1]
            decrypted_data = decrypted_data[:-padding_length]
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt with key: {e}")
            raise
    
    async def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to hash password: {e}")
            raise SecurityViolationError("password_hashing", "bcrypt", str(e))
    
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to verify password: {e}")
            return False
    
    async def generate_access_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """Generate a secure access token."""
        try:
            # Create token payload
            payload = {
                'user_id': user_id,
                'role': role,
                'permissions': permissions,
                'issued_at': time.time(),
                'expires_at': time.time() + self.security_policies['session_timeout']
            }
            
            # Sign token with HMAC
            token_data = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                self.encryption_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Combine payload and signature
            token = base64.b64encode(f"{token_data}.{signature}".encode()).decode()
            
            # Store token
            self.access_tokens[token] = payload
            
            # Log token generation
            await self.log_security_event(
                user_id=user_id,
                event_type="token_generated",
                details=f"Role: {role}, Permissions: {permissions}"
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to generate access token: {e}")
            raise SecurityViolationError("token_generation", "HMAC", str(e))
    
    async def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an access token."""
        try:
            if token not in self.access_tokens:
                return None
            
            payload = self.access_tokens[token]
            
            # Check expiration
            if time.time() > payload['expires_at']:
                del self.access_tokens[token]
                return None
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Failed to validate access token: {e}")
            return None
    
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission to perform action on resource."""
        try:
            # Get user role
            user_role = self.user_roles.get(user_id, "guest")
            
            # Get role permissions
            permissions = self.role_permissions.get(user_role, [])
            
            # Check if action is allowed
            if action in permissions:
                # Log access
                await self.log_access_event(
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    allowed=True
                )
                return True
            else:
                # Log denied access
                await self.log_access_event(
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    allowed=False
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to check permission: {e}")
            return False
    
    async def log_access_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an access event."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'allowed': allowed,
                'ip_address': details.get('ip_address', 'unknown') if details else 'unknown',
                'user_agent': details.get('user_agent', 'unknown') if details else 'unknown',
                'session_id': details.get('session_id', 'unknown') if details else 'unknown'
            }
            
            # Add to audit log
            self.audit_log.append(event)
            
            # Save to file
            await self._save_audit_log()
            
            # Check for security violations
            if not allowed:
                await self._handle_security_violation(user_id, resource, action)
                
        except Exception as e:
            self.logger.error(f"Failed to log access event: {e}")
    
    async def log_security_event(
        self,
        user_id: str,
        event_type: str,
        details: str,
        severity: str = "info"
    ) -> None:
        """Log a security event."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'event_type': event_type,
                'details': details,
                'severity': severity
            }
            
            # Add to security events
            self.security_events.append(event)
            
            # Log to audit log if enabled
            if self.security_policies['audit_logging']:
                self.audit_log.append(event)
                await self._save_audit_log()
            
            # Handle high severity events
            if severity in ["high", "critical"]:
                await self._handle_security_alert(event)
                
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    async def _handle_security_violation(self, user_id: str, resource: str, action: str) -> None:
        """Handle security violations."""
        try:
            # Increment failed attempts
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = 0
            
            # Check if user is already locked out
            if isinstance(self.failed_attempts[user_id], dict):
                # User is already locked out, don't increment
                return
            
            self.failed_attempts[user_id] += 1
            
            # Check if user should be locked out
            if self.failed_attempts[user_id] >= self.security_policies['max_failed_attempts']:
                await self._lockout_user(user_id)
                
            # Log security event
            await self.log_security_event(
                user_id=user_id,
                event_type="access_denied",
                details=f"Resource: {resource}, Action: {action}",
                severity="warning"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle security violation: {e}")
    
    async def _lockout_user(self, user_id: str) -> None:
        """Lock out a user due to too many failed attempts."""
        try:
            # Set lockout time
            lockout_until = time.time() + self.security_policies['lockout_duration']
            
            # Store lockout information
            self.failed_attempts[user_id] = {
                'locked_until': lockout_until,
                'reason': 'too_many_failed_attempts'
            }
            
            # Log security event
            await self.log_security_event(
                user_id=user_id,
                event_type="user_locked_out",
                details=f"Locked until {datetime.fromtimestamp(lockout_until)}",
                severity="high"
            )
            
            self.logger.warning(f"User {user_id} locked out due to security violations")
            
        except Exception as e:
            self.logger.error(f"Failed to lockout user: {e}")
    
    async def _handle_security_alert(self, event: Dict[str, Any]) -> None:
        """Handle high severity security alerts."""
        try:
            # Log critical security event
            self.logger.critical(
                f"SECURITY ALERT: {event['event_type']} by user {event['user_id']}",
                event_details=event
            )
            
            # In production, this would trigger:
            # - Email alerts
            # - SMS notifications
            # - Security team paging
            # - Incident response procedures
            
        except Exception as e:
            self.logger.error(f"Failed to handle security alert: {e}")
    
    async def _save_audit_log(self) -> None:
        """Save audit log to file."""
        try:
            with open(self.audit_log_path, 'w') as f:
                for event in self.audit_log:
                    f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save audit log: {e}")
    
    async def rotate_encryption_keys(self) -> None:
        """Rotate encryption keys."""
        try:
            # Check if rotation is needed
            if not self.security_policies['key_rotation']:
                return
            
            current_time = time.time()
            if current_time - self.last_key_rotation < self.key_rotation_interval:
                return
            
            # Generate new keys
            new_encryption_key = self._generate_encryption_key()
            self._generate_rsa_keys()
            
            # Update keys
            old_encryption_key = self.encryption_key
            self.encryption_key = new_encryption_key
            self.fernet = Fernet(self.encryption_key.encode())
            
            # Log key rotation
            await self.log_security_event(
                user_id="system",
                event_type="key_rotation",
                details="Encryption keys rotated successfully",
                severity="info"
            )
            
            self.last_key_rotation = current_time
            
            self.logger.info("Encryption keys rotated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to rotate encryption keys: {e}")
    
    async def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        try:
            # Calculate security metrics
            total_events = len(self.security_events)
            high_severity_events = len([e for e in self.security_events if e['severity'] in ['high', 'critical']])
            locked_out_users = len([u for u in self.failed_attempts.values() if isinstance(u, dict)])
            
            # Calculate security score
            security_score = max(0, 100 - (high_severity_events * 10) - (locked_out_users * 5))
            
            return {
                'security_level': self.security_level,
                'total_events': total_events,
                'high_severity_events': high_severity_events,
                'locked_out_users': locked_out_users,
                'security_score': security_score,
                'last_key_rotation': datetime.fromtimestamp(self.last_key_rotation).isoformat(),
                'policies': self.security_policies
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate security report: {e}")
            return {}
    
    async def cleanup_expired_tokens(self) -> None:
        """Clean up expired access tokens."""
        try:
            current_time = time.time()
            expired_tokens = []
            
            for token, payload in self.access_tokens.items():
                if current_time > payload['expires_at']:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.access_tokens[token]
            
            if expired_tokens:
                self.logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired tokens: {e}")
    
    async def close(self) -> None:
        """Clean up security manager resources."""
        try:
            # Save audit log
            await self._save_audit_log()
            
            # Clean up expired tokens
            await self.cleanup_expired_tokens()
            
            # Rotate keys if needed
            await self.rotate_encryption_keys()
            
            self.logger.info("Security manager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing security manager: {e}")
