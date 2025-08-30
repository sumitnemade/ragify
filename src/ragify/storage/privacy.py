"""
Privacy Manager for context data protection and privacy controls.
"""

import asyncio
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

# Import security manager
from .security import SecurityManager

from ..models import Context, PrivacyLevel
from ..exceptions import PrivacyViolationError


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
        self.fernet = Fernet(self.encryption_key)
        
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
