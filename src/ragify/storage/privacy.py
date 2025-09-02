"""
Privacy Manager for context data protection and privacy controls.
"""

import asyncio
import re
import base64
import json
import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
import structlog

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import bcrypt

# Import security manager
from .security import SecurityManager

from ..models import Context, PrivacyLevel
from ..exceptions import PrivacyViolationError, EncryptionError, SecurityViolationError


class PrivacyManager:
    """
    Privacy manager for context data protection and privacy controls.
    
    Handles data anonymization, encryption, access controls, and
    compliance with privacy regulations.
    """
    
    def __init__(self, default_privacy_level: Optional[PrivacyLevel] = None, encryption_key: Optional[str] = None, security_level: str = "standard"):
        """
        Initialize the privacy manager.
        
        Args:
            default_privacy_level: Default privacy level for operations (defaults to PRIVATE)
            encryption_key: Optional encryption key (will generate if not provided)
            security_level: Security level for the security manager
        """
        self.default_privacy_level = default_privacy_level or PrivacyLevel.PRIVATE
        self.logger = structlog.get_logger(__name__)
        
        # Initialize security manager
        self.security_manager = SecurityManager(
            encryption_key=encryption_key,
            security_level=security_level
        )
        
        # Initialize encryption (legacy support)
        self.encryption_key = encryption_key or self._generate_encryption_key()
        if isinstance(self.encryption_key, str):
            # If it's a string, try to decode it as base64
            try:
                key_bytes = base64.urlsafe_b64decode(self.encryption_key + '==')
                self.fernet = Fernet(key_bytes)
            except Exception:
                # If that fails, generate a new key
                self.encryption_key = self._generate_encryption_key()
                self.fernet = Fernet(self.encryption_key)
        else:
            self.fernet = Fernet(self.encryption_key)
        
        # Key Management Infrastructure
        self._initialize_key_management()
        
        # Privacy policies
        self.privacy_policies = {
            PrivacyLevel.PUBLIC: {
                'anonymization': False,
                'encryption': False,
                'retention_days': 7,
                'access_logging': False,
            },
            PrivacyLevel.PRIVATE: {
                'anonymization': True,
                'encryption': False,
                'retention_days': 30,
                'access_logging': True,
            },
            PrivacyLevel.ENTERPRISE: {
                'anonymization': True,
                'encryption': True,
                'retention_days': 90,
                'access_logging': True,
            },
            PrivacyLevel.RESTRICTED: {
                'anonymization': True,
                'encryption': True,
                'retention_days': 365,
                'access_logging': True,
                'audit_trail': True,
            },
        }
        
        # Sensitive data patterns
        self.sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
    
    def _generate_encryption_key(self) -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _initialize_key_management(self):
        """Initialize key management infrastructure."""
        # Key storage directory
        self.keys_dir = Path("keys")
        self.keys_dir.mkdir(exist_ok=True)
        
        # Key metadata file
        self.keys_metadata_file = self.keys_dir / "keys_metadata.json"
        self.keys_backup_file = self.keys_dir / "keys_backup.json"
        
        # Key rotation settings
        self.key_rotation_days = 90  # Rotate keys every 90 days
        self.max_key_age_days = 365  # Maximum key age before forced rotation
        self.backup_retention_days = 730  # Keep backups for 2 years
        
        # Key derivation settings
        self.key_derivation_settings = {
            'pbkdf2': {
                'iterations': 100000,
                'hash_algorithm': hashes.SHA256(),
                'key_length': 32
            },
            'scrypt': {
                'n': 16384,  # CPU cost
                'r': 8,      # Memory cost
                'p': 1,      # Parallelization
                'key_length': 32
            },
            'argon2': {
                'iterations': 3,      # Number of iterations
                'memory_cost': 65536, # Memory usage in KiB
                'lanes': 4,          # Number of parallel lanes
                'key_length': 32
            }
        }
        
        # Load existing key metadata
        self.keys_metadata = self._load_keys_metadata()
        
        # Initialize master key for key encryption
        self._initialize_master_key()
    
    def _initialize_master_key(self):
        """Initialize or load master key for encrypting other keys."""
        master_key_file = self.keys_dir / "master.key"
        
        if master_key_file.exists():
            try:
                with open(master_key_file, 'rb') as f:
                    self.master_key = f.read()
                self.logger.info("Master key loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load master key: {e}")
                self._generate_new_master_key()
        else:
            self._generate_new_master_key()
    
    def _generate_new_master_key(self):
        """Generate a new master key."""
        try:
            self.master_key = Fernet.generate_key()
            master_key_file = self.keys_dir / "master.key"
            
            with open(master_key_file, 'wb') as f:
                f.write(self.master_key)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(master_key_file, 0o600)
            
            self.logger.info("New master key generated successfully")
        except Exception as e:
            self.logger.error(f"Failed to generate master key: {e}")
            raise EncryptionError("generate", "master_key", str(e))
    
    def _load_keys_metadata(self) -> Dict[str, Any]:
        """Load keys metadata from file."""
        try:
            if self.keys_metadata_file.exists():
                with open(self.keys_metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load keys metadata: {e}")
            return {}
    
    def _save_keys_metadata(self):
        """Save keys metadata to file."""
        try:
            with open(self.keys_metadata_file, 'w') as f:
                json.dump(self.keys_metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save keys metadata: {e}")
    
    def _encrypt_key_with_master(self, key_data: bytes) -> bytes:
        """Encrypt a key using the master key."""
        try:
            fernet = Fernet(self.master_key)
            return fernet.encrypt(key_data)
        except Exception as e:
            self.logger.error(f"Failed to encrypt key with master: {e}")
            raise EncryptionError("encrypt", "master_key", str(e))
    
    def _decrypt_key_with_master(self, encrypted_key: bytes) -> bytes:
        """Decrypt a key using the master key."""
        try:
            fernet = Fernet(self.master_key)
            return fernet.decrypt(encrypted_key)
        except Exception as e:
            self.logger.error(f"Failed to decrypt key with master: {e}")
            raise EncryptionError("decrypt", "master_key", str(e))
    
    async def derive_key_from_password_enhanced(
        self, 
        password: str, 
        salt: Optional[bytes] = None,
        algorithm: str = "pbkdf2",
        store_key: bool = True,
        key_id: Optional[str] = None
    ) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Enhanced key derivation with enterprise-grade features.
        
        Args:
            password: Password to derive key from
            salt: Optional salt (will generate if not provided)
            algorithm: Key derivation algorithm (pbkdf2, scrypt, argon2)
            store_key: Whether to store the derived key securely
            key_id: Optional key identifier for storage
            
        Returns:
            Tuple of (derived_key, key_id, metadata)
        """
        try:
            # Input validation
            if not password or not isinstance(password, str):
                raise SecurityViolationError("key_derivation", "password_validation", "Invalid password provided")
            
            if len(password) < 8:
                raise SecurityViolationError("key_derivation", "password_validation", "Password must be at least 8 characters")
            
            # Generate salt if not provided
            if salt is None:
                salt = secrets.token_bytes(32)
            
            # Validate algorithm
            if algorithm not in self.key_derivation_settings:
                raise SecurityViolationError("key_derivation", "algorithm_validation", f"Unsupported algorithm: {algorithm}")
            
            # Derive key using selected algorithm
            derived_key = await self._derive_key_with_algorithm(password, salt, algorithm)
            
            # Generate key ID if not provided
            if key_id is None:
                key_id = self._generate_key_id(password, salt)
            
            # Create key metadata
            metadata = {
                'algorithm': algorithm,
                'salt': base64.b64encode(salt).decode('utf-8'),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'last_used': datetime.now(timezone.utc).isoformat(),
                'usage_count': 0,
                'key_hash': self._hash_key_for_validation(derived_key),
                'strength_score': self._calculate_key_strength(password, algorithm),
                'derivation_settings': self.key_derivation_settings[algorithm].copy()
            }
            
            # Store key securely if requested
            if store_key:
                await self._store_key_securely(key_id, derived_key, metadata)
            
            self.logger.info(f"Key derived successfully using {algorithm} algorithm")
            return derived_key, key_id, metadata
            
        except SecurityViolationError:
            # Re-raise SecurityViolationError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to derive key: {e}")
            raise EncryptionError("key_derivation", "enhanced", str(e))
    
    async def _derive_key_with_algorithm(self, password: str, salt: bytes, algorithm: str) -> bytes:
        """Derive key using the specified algorithm."""
        try:
            if algorithm == "pbkdf2":
                settings = self.key_derivation_settings['pbkdf2']
                kdf = PBKDF2HMAC(
                    algorithm=settings['hash_algorithm'],
                    length=settings['key_length'],
                    salt=salt,
                    iterations=settings['iterations'],
                )
                return kdf.derive(password.encode())
            
            elif algorithm == "scrypt":
                settings = self.key_derivation_settings['scrypt']
                kdf = Scrypt(
                    salt=salt,
                    length=settings['key_length'],
                    n=settings['n'],
                    r=settings['r'],
                    p=settings['p'],
                )
                return kdf.derive(password.encode())
            
            elif algorithm == "argon2":
                settings = self.key_derivation_settings['argon2']
                kdf = Argon2id(
                    salt=salt,
                    length=settings['key_length'],
                    iterations=settings['iterations'],
                    lanes=settings['lanes'],
                    memory_cost=settings['memory_cost'],
                )
                return kdf.derive(password.encode())
            
            else:
                raise SecurityViolationError("key_derivation", "algorithm_validation", f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            self.logger.error(f"Failed to derive key with {algorithm}: {e}")
            raise EncryptionError("key_derivation", algorithm, str(e))
    
    def _generate_key_id(self, password: str, salt: bytes) -> str:
        """Generate a unique key identifier."""
        # Create a hash of password + salt + timestamp for uniqueness
        unique_data = f"{password}{base64.b64encode(salt).decode('utf-8')}{datetime.now(timezone.utc).isoformat()}"
        key_hash = hashlib.sha256(unique_data.encode()).hexdigest()
        return f"key_{key_hash[:16]}"
    
    def _hash_key_for_validation(self, key: bytes) -> str:
        """Create a hash of the key for validation purposes."""
        return hashlib.sha256(key).hexdigest()
    
    def _calculate_key_strength(self, password: str, algorithm: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0
        
        # Length bonus
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
        
        # Complexity bonus
        if re.search(r'[A-Z]', password):
            score += 15
        if re.search(r'[a-z]', password):
            score += 15
        if re.search(r'\d', password):
            score += 15
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        
        # Algorithm bonus
        if algorithm == "argon2":
            score += 10
        elif algorithm == "scrypt":
            score += 8
        elif algorithm == "pbkdf2":
            score += 5
        
        return min(100, score)
    
    async def _store_key_securely(self, key_id: str, key_data: bytes, metadata: Dict[str, Any]):
        """Store a key securely with encryption and metadata."""
        try:
            # Encrypt the key with master key
            encrypted_key = self._encrypt_key_with_master(key_data)
            
            # Store encrypted key
            key_file = self.keys_dir / f"{key_id}.key"
            with open(key_file, 'wb') as f:
                f.write(encrypted_key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            # Update metadata
            self.keys_metadata[key_id] = metadata
            self._save_keys_metadata()
            
            # Create backup
            await self._backup_key(key_id, encrypted_key, metadata)
            
            self.logger.info(f"Key {key_id} stored securely")
            
        except Exception as e:
            self.logger.error(f"Failed to store key {key_id}: {e}")
            raise EncryptionError("store", "key", str(e))
    
    async def _backup_key(self, key_id: str, encrypted_key: bytes, metadata: Dict[str, Any]):
        """Create a backup of the key."""
        try:
            backup_data = {
                'key_id': key_id,
                'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
                'metadata': metadata,
                'backup_created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Load existing backups
            backups = []
            if self.keys_backup_file.exists():
                with open(self.keys_backup_file, 'r') as f:
                    backups = json.load(f)
            
            # Add new backup
            backups.append(backup_data)
            
            # Keep only recent backups within retention period
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.backup_retention_days)
            backups = [
                b for b in backups 
                if datetime.fromisoformat(b['backup_created_at']) > cutoff_date
            ]
            
            # Save backups
            with open(self.keys_backup_file, 'w') as f:
                json.dump(backups, f, indent=2, default=str)
            
            self.logger.info(f"Key {key_id} backed up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to backup key {key_id}: {e}")
    
    async def retrieve_key(self, key_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Retrieve a stored key."""
        try:
            if key_id not in self.keys_metadata:
                raise SecurityViolationError("key_retrieval", "key_not_found", f"Key {key_id} not found")
            
            # Load encrypted key
            key_file = self.keys_dir / f"{key_id}.key"
            if not key_file.exists():
                raise SecurityViolationError("key_retrieval", "key_file_not_found", f"Key file for {key_id} not found")
            
            with open(key_file, 'rb') as f:
                encrypted_key = f.read()
            
            # Decrypt key
            key_data = self._decrypt_key_with_master(encrypted_key)
            
            # Validate key integrity
            if not self._validate_key_integrity(key_id, key_data):
                raise SecurityViolationError("key_retrieval", "integrity_check_failed", f"Key {key_id} integrity check failed")
            
            # Update usage metadata
            self.keys_metadata[key_id]['last_used'] = datetime.now(timezone.utc).isoformat()
            self.keys_metadata[key_id]['usage_count'] += 1
            self._save_keys_metadata()
            
            self.logger.info(f"Key {key_id} retrieved successfully")
            # Add key_id to metadata for return
            metadata_with_id = self.keys_metadata[key_id].copy()
            metadata_with_id['key_id'] = key_id
            return key_data, metadata_with_id
            
        except SecurityViolationError:
            # Re-raise SecurityViolationError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key_id}: {e}")
            raise EncryptionError("retrieve", "key", str(e))
    
    def _validate_key_integrity(self, key_id: str, key_data: bytes) -> bool:
        """Validate key integrity using stored hash."""
        try:
            if key_id not in self.keys_metadata:
                return False
            
            stored_hash = self.keys_metadata[key_id]['key_hash']
            current_hash = self._hash_key_for_validation(key_data)
            
            return hmac.compare_digest(stored_hash, current_hash)
            
        except Exception as e:
            self.logger.error(f"Key integrity validation failed: {e}")
            return False
    
    async def rotate_key(self, key_id: str, new_password: str, algorithm: str = "pbkdf2") -> str:
        """Rotate an existing key with a new password."""
        try:
            # Retrieve existing key metadata
            if key_id not in self.keys_metadata:
                raise SecurityViolationError("key_rotation", "key_not_found", f"Key {key_id} not found")
            
            old_metadata = self.keys_metadata[key_id]
            
            # Derive new key
            new_key, new_key_id, new_metadata = await self.derive_key_from_password_enhanced(
                new_password, 
                algorithm=algorithm,
                store_key=True
            )
            
            # Mark old key for deletion
            old_metadata['deprecated_at'] = datetime.now(timezone.utc).isoformat()
            old_metadata['replaced_by'] = new_key_id
            self._save_keys_metadata()
            
            # Schedule old key deletion (in production, this would be a background task)
            await self._schedule_key_deletion(key_id, delay_days=7)
            
            self.logger.info(f"Key {key_id} rotated to {new_key_id}")
            return new_key_id
            
        except Exception as e:
            self.logger.error(f"Failed to rotate key {key_id}: {e}")
            raise EncryptionError("rotate", "key", str(e))
    
    async def _schedule_key_deletion(self, key_id: str, delay_days: int):
        """Schedule a key for deletion after a delay."""
        try:
            # In a production environment, this would use a task queue
            # For now, we'll just mark it for deletion
            if key_id in self.keys_metadata:
                self.keys_metadata[key_id]['scheduled_for_deletion'] = (
                    datetime.now(timezone.utc) + timedelta(days=delay_days)
                ).isoformat()
                self._save_keys_metadata()
            
            self.logger.info(f"Key {key_id} scheduled for deletion in {delay_days} days")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule key deletion: {e}")
    
    async def delete_key(self, key_id: str, force: bool = False) -> bool:
        """Delete a key and its metadata."""
        try:
            if key_id not in self.keys_metadata:
                return False
            
            metadata = self.keys_metadata[key_id]
            
            # Check if key is scheduled for deletion or force delete
            if not force and 'scheduled_for_deletion' in metadata:
                scheduled_date = datetime.fromisoformat(metadata['scheduled_for_deletion'])
                if datetime.now(timezone.utc) < scheduled_date:
                    raise SecurityViolationError("key_deletion", "premature_deletion", f"Key {key_id} is not yet scheduled for deletion")
            
            # Remove key file
            key_file = self.keys_dir / f"{key_id}.key"
            if key_file.exists():
                os.remove(key_file)
            
            # Remove metadata
            del self.keys_metadata[key_id]
            self._save_keys_metadata()
            
            self.logger.info(f"Key {key_id} deleted successfully")
            return True
            
        except SecurityViolationError:
            # Re-raise SecurityViolationError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Failed to delete key {key_id}: {e}")
            return False
    
    async def get_key_metadata(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific key."""
        metadata = self.keys_metadata.get(key_id)
        if metadata:
            metadata_with_id = metadata.copy()
            metadata_with_id['key_id'] = key_id
            return metadata_with_id
        return None
    
    async def list_keys(self) -> List[Dict[str, Any]]:
        """List all stored keys with metadata."""
        return [
            {'key_id': key_id, **metadata}
            for key_id, metadata in self.keys_metadata.items()
        ]
    
    async def check_key_health(self) -> Dict[str, Any]:
        """Check the health of all stored keys."""
        try:
            total_keys = len(self.keys_metadata)
            healthy_keys = 0
            expired_keys = 0
            weak_keys = 0
            
            current_time = datetime.now(timezone.utc)
            
            for key_id, metadata in self.keys_metadata.items():
                # Check if key file exists
                key_file = self.keys_dir / f"{key_id}.key"
                if not key_file.exists():
                    continue
                
                # Check key age
                created_at = datetime.fromisoformat(metadata['created_at'])
                key_age_days = (current_time - created_at).days
                
                if key_age_days > self.max_key_age_days:
                    expired_keys += 1
                elif key_age_days > self.key_rotation_days:
                    # Mark for rotation
                    if 'needs_rotation' not in metadata:
                        metadata['needs_rotation'] = True
                        metadata['rotation_recommended_at'] = current_time.isoformat()
                else:
                    healthy_keys += 1
                
                # Check key strength
                if metadata.get('strength_score', 0) < 50:
                    weak_keys += 1
            
            # Save updated metadata
            self._save_keys_metadata()
            
            return {
                'total_keys': total_keys,
                'healthy_keys': healthy_keys,
                'expired_keys': expired_keys,
                'weak_keys': weak_keys,
                'keys_needing_rotation': expired_keys + sum(
                    1 for m in self.keys_metadata.values() 
                    if m.get('needs_rotation', False)
                ),
                'health_score': max(0, 100 - (expired_keys * 20) - (weak_keys * 10))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check key health: {e}")
            return {}
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired keys."""
        try:
            cleaned_count = 0
            current_time = datetime.now(timezone.utc)
            
            for key_id, metadata in list(self.keys_metadata.items()):
                created_at = datetime.fromisoformat(metadata['created_at'])
                key_age_days = (current_time - created_at).days
                
                if key_age_days > self.max_key_age_days:
                    if await self.delete_key(key_id, force=True):
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} expired keys")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
    
    async def apply_privacy_controls(
        self,
        context: Context,
        privacy_level: PrivacyLevel,
    ) -> Context:
        """
        Apply privacy controls to a context.
        
        Args:
            context: Context to apply privacy controls to
            privacy_level: Privacy level to apply
            
        Returns:
            Context with privacy controls applied
        """
        try:
            self.logger.info(f"Applying privacy controls for level: {privacy_level}")
            
            # Get privacy policy
            policy = self.privacy_policies.get(privacy_level, self.privacy_policies[PrivacyLevel.PRIVATE])
            
            # Apply controls
            protected_context = context.model_copy()
            
            # Anonymize if required
            if policy['anonymization']:
                protected_context = await self._anonymize_context(protected_context)
            
            # Encrypt if required
            if policy['encryption']:
                protected_context = await self._encrypt_context(protected_context)
            
            # Update metadata
            protected_context.metadata.update({
                'privacy_level': privacy_level.value,
                'privacy_applied_at': asyncio.get_event_loop().time(),
                'anonymized': policy['anonymization'],
                'encrypted': policy['encryption'],
            })
            
            self.logger.info(f"Successfully applied privacy controls for level: {privacy_level}")
            return protected_context
            
        except Exception as e:
            self.logger.error(f"Failed to apply privacy controls: {e}")
            raise PrivacyViolationError("privacy_controls", privacy_level.value, str(e))
    
    async def check_privacy_compliance(
        self,
        context: Context,
        required_level: PrivacyLevel,
    ) -> bool:
        """
        Check if context complies with privacy requirements.
        
        Args:
            context: Context to check
            required_level: Required privacy level
            
        Returns:
            True if compliant, False otherwise
        """
        try:
            current_level = PrivacyLevel(context.metadata.get('privacy_level', self.default_privacy_level.value))
            
            # Check if current level meets or exceeds required level
            compliance = self._privacy_level_compliance(current_level, required_level)
            
            if not compliance:
                self.logger.warning(
                    f"Privacy compliance check failed: current={current_level}, required={required_level}"
                )
            
            return compliance
            
        except Exception as e:
            self.logger.error(f"Failed to check privacy compliance: {e}")
            return False
    
    async def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect sensitive data in content.
        
        Args:
            content: Content to analyze
            
        Returns:
            List of detected sensitive data
        """
        try:
            import re
            
            detected_data = []
            
            for data_type, pattern in self.sensitive_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    detected_data.append({
                        'type': data_type,
                        'value': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                    })
            
            return detected_data
            
        except Exception as e:
            self.logger.error(f"Failed to detect sensitive data: {e}")
            return []
    
    async def anonymize_content(self, content: str) -> str:
        """
        Anonymize sensitive data in content using secure hashing.
        
        Args:
            content: Content to anonymize
            
        Returns:
            Anonymized content
        """
        try:
            anonymized_content = content
            
            # Replace sensitive data patterns with secure hashes
            for data_type, pattern in self.sensitive_patterns.items():
                matches = re.finditer(pattern, anonymized_content)
                
                # Replace matches in reverse order to maintain positions
                for match in reversed(list(matches)):
                    sensitive_data = match.group(0)
                    
                    # Create a secure hash of the sensitive data
                    salt = bcrypt.gensalt()
                    hashed = bcrypt.hashpw(sensitive_data.encode('utf-8'), salt)
                    hash_short = base64.urlsafe_b64encode(hashed[:8]).decode('utf-8')
                    
                    # Replace with anonymized version
                    replacement = f"[{data_type.upper()}_HASH_{hash_short}]"
                    anonymized_content = (
                        anonymized_content[:match.start()] + 
                        replacement + 
                        anonymized_content[match.end():]
                    )
            
            # Additional anonymization for names (basic pattern)
            name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            name_matches = re.finditer(name_pattern, anonymized_content)
            
            for match in reversed(list(name_matches)):
                name = match.group(0)
                # Create hash for name
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(name.encode('utf-8'), salt)
                hash_short = base64.urlsafe_b64encode(hashed[:8]).decode('utf-8')
                
                replacement = f"[NAME_HASH_{hash_short}]"
                anonymized_content = (
                    anonymized_content[:match.start()] + 
                    replacement + 
                    anonymized_content[match.end():]
                )
            
            return anonymized_content
            
        except Exception as e:
            self.logger.error(f"Failed to anonymize content: {e}")
            return content
    
    async def encrypt_content(self, content: str) -> str:
        """
        Encrypt content using Fernet symmetric encryption.
        
        Args:
            content: Content to encrypt
            
        Returns:
            Encrypted content (base64 encoded)
        """
        try:
            # Encrypt the content
            encrypted_data = self.fernet.encrypt(content.encode('utf-8'))
            
            # Return base64 encoded encrypted data
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt content: {e}")
            return content
    
    async def decrypt_content(self, encrypted_content: str) -> str:
        """
        Decrypt content using Fernet symmetric encryption.
        
        Args:
            encrypted_content: Encrypted content (base64 encoded)
            
        Returns:
            Decrypted content
        """
        try:
            # Check if content is actually encrypted
            if not encrypted_content or len(encrypted_content) < 100:
                # Likely not encrypted, return as-is
                return encrypted_content
            
            # Decode base64 and decrypt
            encrypted_data = base64.urlsafe_b64decode(encrypted_content.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt content: {e}")
            return encrypted_content
    
    async def _anonymize_context(self, context: Context) -> Context:
        """Anonymize sensitive data in context."""
        try:
            # Anonymize query
            anonymized_query = await self.anonymize_content(context.query)
            
            # Anonymize chunks
            anonymized_chunks = []
            for chunk in context.chunks:
                anonymized_content = await self.anonymize_content(chunk.content)
                anonymized_chunk = chunk.model_copy()
                anonymized_chunk.content = anonymized_content
                anonymized_chunks.append(anonymized_chunk)
            
            # Create anonymized context
            anonymized_context = context.model_copy()
            anonymized_context.query = anonymized_query
            anonymized_context.chunks = anonymized_chunks
            
            return anonymized_context
            
        except Exception as e:
            self.logger.error(f"Failed to anonymize context: {e}")
            return context
    
    async def _encrypt_context(self, context: Context) -> Context:
        """Encrypt context content."""
        try:
            # Encrypt query
            encrypted_query = await self.encrypt_content(context.query)
            
            # Encrypt chunks
            encrypted_chunks = []
            for chunk in context.chunks:
                encrypted_content = await self.encrypt_content(chunk.content)
                encrypted_chunk = chunk.model_copy()
                encrypted_chunk.content = encrypted_content
                encrypted_chunks.append(encrypted_chunk)
            
            # Create encrypted context
            encrypted_context = context.model_copy()
            encrypted_context.query = encrypted_query
            encrypted_context.chunks = encrypted_chunks
            
            return encrypted_context
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt context: {e}")
            return context
    
    def _privacy_level_compliance(
        self,
        current_level: PrivacyLevel,
        required_level: PrivacyLevel,
    ) -> bool:
        """Check if current privacy level meets required level."""
        # Define privacy level hierarchy (higher = more restrictive)
        level_hierarchy = {
            PrivacyLevel.PUBLIC: 1,
            PrivacyLevel.PRIVATE: 2,
            PrivacyLevel.ENTERPRISE: 3,
            PrivacyLevel.RESTRICTED: 4,
        }
        
        current_value = level_hierarchy.get(current_level, 0)
        required_value = level_hierarchy.get(required_level, 0)
        
        return current_value >= required_value
    
    async def get_privacy_report(self, context: Context) -> Dict[str, Any]:
        """
        Generate privacy report for a context.
        
        Args:
            context: Context to analyze
            
        Returns:
            Privacy report
        """
        try:
            # Detect sensitive data
            sensitive_data = await self.detect_sensitive_data(context.query)
            for chunk in context.chunks:
                chunk_sensitive = await self.detect_sensitive_data(chunk.content)
                sensitive_data.extend(chunk_sensitive)
            
            # Get privacy level
            privacy_level = PrivacyLevel(context.metadata.get('privacy_level', self.default_privacy_level.value))
            policy = self.privacy_policies.get(privacy_level, {})
            
            return {
                'privacy_level': privacy_level.value,
                'sensitive_data_count': len(sensitive_data),
                'sensitive_data_types': list(set(item['type'] for item in sensitive_data)),
                'anonymized': context.metadata.get('anonymized', False),
                'encrypted': context.metadata.get('encrypted', False),
                'retention_days': policy.get('retention_days', 30),
                'access_logging': policy.get('access_logging', False),
                'audit_trail': policy.get('audit_trail', False),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate privacy report: {e}")
            return {}
    
    async def encrypt_with_security_manager(self, data: str, encryption_type: str = "symmetric") -> str:
        """
        Encrypt data using the security manager.
        
        Args:
            data: Data to encrypt
            encryption_type: Type of encryption (symmetric, asymmetric, hybrid)
            
        Returns:
            Encrypted data
        """
        try:
            return await self.security_manager.encrypt_data(data, encryption_type)
        except Exception as e:
            self.logger.error(f"Failed to encrypt with security manager: {e}")
            # Fallback to legacy encryption
            return await self.encrypt_content(data)
    
    async def decrypt_with_security_manager(self, encrypted_data: str, encryption_type: str = "symmetric") -> str:
        """
        Decrypt data using the security manager.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            encryption_type: Type of encryption used
            
        Returns:
            Decrypted data
        """
        try:
            return await self.security_manager.decrypt_data(encrypted_data, encryption_type)
        except Exception as e:
            self.logger.error(f"Failed to decrypt with security manager: {e}")
            # Fallback to legacy decryption
            return await self.decrypt_content(encrypted_data)
    
    async def check_user_permission(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission to perform action on resource.
        
        Args:
            user_id: User ID
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if permission granted, False otherwise
        """
        try:
            return await self.security_manager.check_permission(user_id, resource, action)
        except Exception as e:
            self.logger.error(f"Failed to check user permission: {e}")
            return False
    
    async def generate_user_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """
        Generate access token for user.
        
        Args:
            user_id: User ID
            role: User role
            permissions: List of permissions
            
        Returns:
            Access token
        """
        try:
            return await self.security_manager.generate_access_token(user_id, role, permissions)
        except Exception as e:
            self.logger.error(f"Failed to generate user token: {e}")
            raise
    
    async def get_security_report(self) -> Dict[str, Any]:
        """
        Get security report from security manager.
        
        Returns:
            Security report
        """
        try:
            return await self.security_manager.get_security_report()
        except Exception as e:
            self.logger.error(f"Failed to get security report: {e}")
            return {}
    
    async def close(self) -> None:
        """Close privacy manager and security manager."""
        try:
            await self.security_manager.close()
            self.logger.info("Privacy manager closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing privacy manager: {e}")

    async def filter_by_privacy(
        self,
        chunks: List[Any],
        user_clearance: PrivacyLevel,
        user_id: str
    ) -> List[Any]:
        """
        Filter chunks based on user's privacy clearance level.
        
        Args:
            chunks: List of context chunks to filter
            user_clearance: User's privacy clearance level
            user_id: User ID for logging
            
        Returns:
            Filtered list of chunks
        """
        try:
            accessible_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'source') and hasattr(chunk.source, 'privacy_level'):
                    chunk_privacy = chunk.source.privacy_level
                    if self._privacy_level_compliance(chunk_privacy, user_clearance):
                        accessible_chunks.append(chunk)
                else:
                    # If no privacy level specified, assume public
                    accessible_chunks.append(chunk)
            
            # Log access
            await self.log_access_event(
                user_id=user_id,
                resource="chunk_filtering",
                action="read",
                timestamp=datetime.utcnow(),
                privacy_level=user_clearance
            )
            
            return accessible_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to filter chunks by privacy: {e}")
            return chunks

    async def anonymize_chunks(
        self,
        chunks: List[Any],
        anonymization_level: str = "medium"
    ) -> List[Any]:
        """
        Anonymize a list of chunks.
        
        Args:
            chunks: List of chunks to anonymize
            anonymization_level: Level of anonymization (low, medium, high)
            
        Returns:
            List of anonymized chunks
        """
        try:
            anonymized_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'content'):
                    anonymized_content = await self.anonymize_content(chunk.content)
                    anonymized_chunk = chunk.model_copy()
                    anonymized_chunk.content = anonymized_content
                    anonymized_chunks.append(anonymized_chunk)
                else:
                    anonymized_chunks.append(chunk)
            
            return anonymized_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to anonymize chunks: {e}")
            return chunks

    async def check_access(
        self,
        resource_path: str,
        user_role: str,
        user_clearance: PrivacyLevel
    ) -> bool:
        """
        Check if user has access to a resource.
        
        Args:
            resource_path: Path to the resource
            user_role: User's role
            user_clearance: User's clearance level
            
        Returns:
            True if access is granted, False otherwise
        """
        try:
            # Simple access control logic - can be enhanced
            if user_clearance == PrivacyLevel.RESTRICTED:
                return True  # Highest clearance has access to everything
            
            if "public" in resource_path.lower():
                return True
            
            if "financial" in resource_path.lower() and user_clearance == PrivacyLevel.ENTERPRISE:
                return True
            
            if "customer" in resource_path.lower() and user_clearance in [PrivacyLevel.ENTERPRISE, PrivacyLevel.CONFIDENTIAL]:
                return True
            
            return user_clearance == PrivacyLevel.PUBLIC and "public" in resource_path.lower()
            
        except Exception as e:
            self.logger.error(f"Failed to check access: {e}")
            return False

    async def log_access_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        timestamp: datetime,
        privacy_level: PrivacyLevel
    ) -> None:
        """
        Log an access event for compliance.
        
        Args:
            user_id: ID of the user
            resource: Resource being accessed
            action: Action performed
            timestamp: When the action occurred
            privacy_level: Privacy level of the resource
        """
        try:
            # In a real implementation, this would write to a database or log file
            self.logger.info(
                "Access event logged",
                user_id=user_id,
                resource=resource,
                action=action,
                timestamp=timestamp.isoformat(),
                privacy_level=privacy_level.value
            )
        except Exception as e:
            self.logger.error(f"Failed to log access event: {e}")

    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for the specified time period.
        
        Args:
            start_date: Start of reporting period
            end_date: End of reporting period
            
        Returns:
            Compliance report
        """
        try:
            # Generate real compliance report from actual data
            total_events = 0
            violations = 0
            privacy_level_counts = {
                'public': 0,
                'private': 0,
                'enterprise': 0,
                'restricted': 0
            }
            
            # Count events by privacy level (this would query actual logs in production)
            # For now, we'll use the access log if available
            if hasattr(self, 'access_log') and self.access_log:
                for event in self.access_log:
                    event_date = event.get('timestamp')
                    if start_date <= event_date <= end_date:
                        total_events += 1
                        privacy_level = event.get('privacy_level', 'private')
                        if privacy_level in privacy_level_counts:
                            privacy_level_counts[privacy_level] += 1
                        
                        # Check for violations (access denied events)
                        if event.get('action') == 'access_denied':
                            violations += 1
            
            # Calculate compliance score
            compliance_score = 100.0
            if total_events > 0:
                compliance_score = max(0.0, 100.0 - (violations / total_events) * 100.0)
            
            return {
                'total_events': total_events,
                'violations': violations,
                'compliance_score': round(compliance_score, 2),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'privacy_levels': privacy_level_counts
            }
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {}
