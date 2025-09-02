"""
Test suite for Privacy Key Derivation and Management.

Tests enterprise-grade key management features including:
- Secure key derivation with multiple algorithms
- Key storage and encryption
- Key rotation and lifecycle management
- Key validation and integrity checks
- Backup and recovery systems
- Key health monitoring
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import base64
import hashlib

from ragify.storage.privacy import PrivacyManager
from ragify.models import PrivacyLevel
from ragify.exceptions import SecurityViolationError, EncryptionError


class TestPrivacyKeyDerivation:
    """Test basic key derivation functionality."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        # Use a temporary directory for keys
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        # Override keys directory to use temp directory
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        # Cleanup
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_basic_key_derivation(self, privacy_manager):
        """Test basic key derivation with PBKDF2."""
        password = "SecurePassword123!"
        salt = b"test_salt_32_bytes_long_string"
        
        key = privacy_manager._derive_key_from_password(password, salt)
        
        assert key is not None
        assert len(key) > 0
        assert isinstance(key, bytes)
    
    @pytest.mark.asyncio
    async def test_enhanced_key_derivation_pbkdf2(self, privacy_manager):
        """Test enhanced key derivation with PBKDF2 algorithm."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, algorithm="pbkdf2"
        )
        
        assert derived_key is not None
        assert len(derived_key) > 0
        assert key_id.startswith("key_")
        assert metadata['algorithm'] == "pbkdf2"
        assert metadata['strength_score'] > 0
        assert 'salt' in metadata
        assert 'created_at' in metadata
    
    @pytest.mark.asyncio
    async def test_enhanced_key_derivation_scrypt(self, privacy_manager):
        """Test enhanced key derivation with Scrypt algorithm."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, algorithm="scrypt"
        )
        
        assert derived_key is not None
        assert len(derived_key) > 0
        assert key_id.startswith("key_")
        assert metadata['algorithm'] == "scrypt"
        assert metadata['strength_score'] > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_key_derivation_argon2(self, privacy_manager):
        """Test enhanced key derivation with Argon2 algorithm."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, algorithm="argon2"
        )
        
        assert derived_key is not None
        assert len(derived_key) > 0
        assert key_id.startswith("key_")
        assert metadata['algorithm'] == "argon2"
        assert metadata['strength_score'] > 0
    
    @pytest.mark.asyncio
    async def test_key_derivation_with_custom_salt(self, privacy_manager):
        """Test key derivation with a custom salt."""
        password = "SecurePassword123!"
        custom_salt = b"custom_salt_32_bytes_long_string"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, salt=custom_salt, algorithm="pbkdf2"
        )
        
        assert derived_key is not None
        assert metadata['salt'] == base64.b64encode(custom_salt).decode('utf-8')
    
    @pytest.mark.asyncio
    async def test_key_derivation_without_storage(self, privacy_manager):
        """Test key derivation without storing the key."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, store_key=False
        )
        
        assert derived_key is not None
        assert key_id.startswith("key_")
        # Key should not be stored
        assert len(privacy_manager.keys_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_key_derivation_with_custom_id(self, privacy_manager):
        """Test key derivation with a custom key ID."""
        password = "SecurePassword123!"
        custom_id = "custom_key_123"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password, key_id=custom_id
        )
        
        assert key_id == custom_id
        assert custom_id in privacy_manager.keys_metadata


class TestKeyValidationAndSecurity:
    """Test key validation and security features."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_password_validation_short_password(self, privacy_manager):
        """Test that short passwords are rejected."""
        password = "short"
        
        with pytest.raises(SecurityViolationError, match="Password must be at least 8 characters"):
            await privacy_manager.derive_key_from_password_enhanced(password)
    
    @pytest.mark.asyncio
    async def test_password_validation_none_password(self, privacy_manager):
        """Test that None passwords are rejected."""
        with pytest.raises(SecurityViolationError, match="Invalid password provided"):
            await privacy_manager.derive_key_from_password_enhanced(None)
    
    @pytest.mark.asyncio
    async def test_password_validation_empty_password(self, privacy_manager):
        """Test that empty passwords are rejected."""
        with pytest.raises(SecurityViolationError, match="Invalid password provided"):
            await privacy_manager.derive_key_from_password_enhanced("")
    
    @pytest.mark.asyncio
    async def test_unsupported_algorithm(self, privacy_manager):
        """Test that unsupported algorithms are rejected."""
        password = "SecurePassword123!"
        
        with pytest.raises(SecurityViolationError, match="Unsupported algorithm"):
            await privacy_manager.derive_key_from_password_enhanced(password, algorithm="unsupported")
    
    @pytest.mark.asyncio
    async def test_key_strength_calculation(self, privacy_manager):
        """Test password strength calculation."""
        # Weak password
        weak_password = "password"
        _, _, metadata_weak = await privacy_manager.derive_key_from_password_enhanced(
            weak_password, store_key=False
        )
        
        # Strong password
        strong_password = "SecurePassword123!@#"
        _, _, metadata_strong = await privacy_manager.derive_key_from_password_enhanced(
            strong_password, store_key=False
        )
        
        assert metadata_weak['strength_score'] < metadata_strong['strength_score']
        assert metadata_strong['strength_score'] > 70  # Should be high for strong password
    
    @pytest.mark.asyncio
    async def test_key_integrity_validation(self, privacy_manager):
        """Test key integrity validation."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        # Test integrity validation
        is_valid = privacy_manager._validate_key_integrity(key_id, derived_key)
        assert is_valid is True
        
        # Test with corrupted key
        corrupted_key = derived_key + b"corruption"
        is_valid = privacy_manager._validate_key_integrity(key_id, corrupted_key)
        assert is_valid is False


class TestKeyStorageAndRetrieval:
    """Test key storage and retrieval functionality."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_key_storage_and_retrieval(self, privacy_manager):
        """Test storing and retrieving a key."""
        password = "SecurePassword123!"
        
        # Store key
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        # Verify key is stored
        assert key_id in privacy_manager.keys_metadata
        assert privacy_manager.keys_metadata_file.exists()
        
        # Retrieve key
        retrieved_key, retrieved_metadata = await privacy_manager.retrieve_key(key_id)
        
        assert retrieved_key == derived_key
        assert retrieved_metadata['key_id'] == key_id
        assert retrieved_metadata['usage_count'] == 1  # Should increment usage count
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_key(self, privacy_manager):
        """Test retrieving a key that doesn't exist."""
        with pytest.raises(SecurityViolationError, match="Key nonexistent_key not found"):
            await privacy_manager.retrieve_key("nonexistent_key")
    
    @pytest.mark.asyncio
    async def test_key_file_permissions(self, privacy_manager):
        """Test that key files have restrictive permissions."""
        password = "SecurePassword123!"
        
        await privacy_manager.derive_key_from_password_enhanced(password)
        
        # Check that key files exist and have restrictive permissions
        for key_id in privacy_manager.keys_metadata:
            key_file = privacy_manager.keys_dir / f"{key_id}.key"
            assert key_file.exists()
            
            # Check permissions (should be 600 - owner read/write only)
            stat = key_file.stat()
            assert oct(stat.st_mode)[-3:] == "600"
    
    @pytest.mark.asyncio
    async def test_key_backup_creation(self, privacy_manager):
        """Test that key backups are created."""
        password = "SecurePassword123!"
        
        await privacy_manager.derive_key_from_password_enhanced(password)
        
        # Check that backup file exists
        assert privacy_manager.keys_backup_file.exists()
        
        # Load backup data
        with open(privacy_manager.keys_backup_file, 'r') as f:
            import json
            backups = json.load(f)
        
        assert len(backups) > 0
        assert 'key_id' in backups[0]
        assert 'encrypted_key' in backups[0]
        assert 'metadata' in backups[0]


class TestKeyRotationAndLifecycle:
    """Test key rotation and lifecycle management."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_key_rotation(self, privacy_manager):
        """Test key rotation functionality."""
        # Create initial key
        password1 = "SecurePassword123!"
        derived_key1, key_id1, metadata1 = await privacy_manager.derive_key_from_password_enhanced(
            password1
        )
        
        # Rotate key with new password
        password2 = "NewSecurePassword456!"
        new_key_id = await privacy_manager.rotate_key(key_id1, password2)
        
        # Verify old key is marked for deletion
        old_metadata = privacy_manager.keys_metadata[key_id1]
        assert 'deprecated_at' in old_metadata
        assert 'replaced_by' in old_metadata
        assert old_metadata['replaced_by'] == new_key_id
        
        # Verify new key exists
        assert new_key_id in privacy_manager.keys_metadata
        new_metadata = privacy_manager.keys_metadata[new_key_id]
        assert new_metadata['algorithm'] == metadata1['algorithm']
    
    @pytest.mark.asyncio
    async def test_key_deletion_scheduling(self, privacy_manager):
        """Test key deletion scheduling."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        # Schedule deletion
        await privacy_manager._schedule_key_deletion(key_id, delay_days=1)
        
        # Verify scheduling
        updated_metadata = privacy_manager.keys_metadata[key_id]
        assert 'scheduled_for_deletion' in updated_metadata
        
        # Try to delete before scheduled time (should fail)
        with pytest.raises(SecurityViolationError):
            await privacy_manager.delete_key(key_id, force=False)
        
        # Force delete should work
        result = await privacy_manager.delete_key(key_id, force=True)
        assert result is True
        assert key_id not in privacy_manager.keys_metadata
    
    @pytest.mark.asyncio
    async def test_key_health_checking(self, privacy_manager):
        """Test key health monitoring."""
        # Create a key
        password = "SecurePassword123!"
        await privacy_manager.derive_key_from_password_enhanced(password)
        
        # Check health
        health = await privacy_manager.check_key_health()
        
        assert 'total_keys' in health
        assert 'healthy_keys' in health
        assert 'expired_keys' in health
        assert 'weak_keys' in health
        assert 'health_score' in health
        assert health['total_keys'] > 0
        assert health['health_score'] > 0
    
    @pytest.mark.asyncio
    async def test_expired_key_cleanup(self, privacy_manager):
        """Test cleanup of expired keys."""
        # Create a key
        password = "SecurePassword123!"
        await privacy_manager.derive_key_from_password_enhanced(password)
        
        # Manually set key as expired by modifying metadata
        for key_id in privacy_manager.keys_metadata:
            privacy_manager.keys_metadata[key_id]['created_at'] = (
                datetime.now(timezone.utc) - timedelta(days=400)
            ).isoformat()
        
        # Cleanup expired keys
        cleaned_count = await privacy_manager.cleanup_expired_keys()
        
        assert cleaned_count > 0
        assert len(privacy_manager.keys_metadata) == 0


class TestKeyManagementOperations:
    """Test key management operations."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_list_keys(self, privacy_manager):
        """Test listing all keys."""
        # Create multiple keys
        passwords = ["Password1!", "Password2!", "Password3!"]
        for password in passwords:
            await privacy_manager.derive_key_from_password_enhanced(password)
        
        # List keys
        keys_list = await privacy_manager.list_keys()
        
        assert len(keys_list) == len(passwords)
        for key_info in keys_list:
            assert 'key_id' in key_info
            assert 'algorithm' in key_info
            assert 'created_at' in key_info
            assert 'strength_score' in key_info
    
    @pytest.mark.asyncio
    async def test_get_key_metadata(self, privacy_manager):
        """Test getting metadata for a specific key."""
        password = "SecurePassword123!"
        
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        # Get metadata
        retrieved_metadata = await privacy_manager.get_key_metadata(key_id)
        
        assert retrieved_metadata is not None
        assert retrieved_metadata['key_id'] == key_id
        assert retrieved_metadata['algorithm'] == metadata['algorithm']
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key_metadata(self, privacy_manager):
        """Test getting metadata for a nonexistent key."""
        metadata = await privacy_manager.get_key_metadata("nonexistent_key")
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_master_key_management(self, privacy_manager):
        """Test master key generation and management."""
        # Verify master key exists
        assert hasattr(privacy_manager, 'master_key')
        assert privacy_manager.master_key is not None
        
        # Verify master key file exists
        master_key_file = privacy_manager.keys_dir / "master.key"
        assert master_key_file.exists()
        
        # Verify master key file permissions
        stat = master_key_file.stat()
        assert oct(stat.st_mode)[-3:] == "600"


class TestPrivacyKeyDerivationIntegration:
    """Integration tests for privacy key derivation."""
    
    @pytest.fixture
    async def privacy_manager(self):
        """Create a privacy manager instance for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        manager.keys_dir = Path(temp_dir)
        manager.keys_metadata_file = Path(temp_dir) / "keys_metadata.json"
        manager.keys_backup_file = Path(temp_dir) / "keys_backup.json"
        
        # Re-initialize master key in the new directory
        manager._initialize_master_key()
        
        yield manager
        
        await manager.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_key_lifecycle(self, privacy_manager):
        """Test complete key lifecycle from creation to deletion."""
        # 1. Create key
        password = "SecurePassword123!"
        derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        # 2. Verify key is stored and accessible
        assert key_id in privacy_manager.keys_metadata
        retrieved_key, retrieved_metadata = await privacy_manager.retrieve_key(key_id)
        assert retrieved_key == derived_key
        
        # 3. Check key health
        health = await privacy_manager.check_key_health()
        assert health['total_keys'] == 1
        assert health['healthy_keys'] == 1
        
        # 4. Rotate key
        new_password = "NewSecurePassword456!"
        new_key_id = await privacy_manager.rotate_key(key_id, new_password)
        
        # 5. Verify rotation
        assert new_key_id in privacy_manager.keys_metadata
        assert privacy_manager.keys_metadata[key_id]['deprecated_at'] is not None
        
        # 6. Clean up
        await privacy_manager.delete_key(key_id, force=True)
        await privacy_manager.delete_key(new_key_id, force=True)
        
        # 7. Verify cleanup
        assert len(privacy_manager.keys_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_algorithm_support(self, privacy_manager):
        """Test support for multiple key derivation algorithms."""
        algorithms = ["pbkdf2", "scrypt", "argon2"]
        passwords = ["Password1!", "Password2!", "Password3!"]
        
        for algorithm, password in zip(algorithms, passwords):
            derived_key, key_id, metadata = await privacy_manager.derive_key_from_password_enhanced(
                password, algorithm=algorithm
            )
            
            assert metadata['algorithm'] == algorithm
            assert derived_key is not None
        
        # Verify all keys are stored
        assert len(privacy_manager.keys_metadata) == len(algorithms)
        
        # List all keys
        keys_list = await privacy_manager.list_keys()
        assert len(keys_list) == len(algorithms)
        
        # Check that all algorithms are represented
        stored_algorithms = [key['algorithm'] for key in keys_list]
        for algorithm in algorithms:
            assert algorithm in stored_algorithms
    
    @pytest.mark.asyncio
    async def test_key_strength_analysis(self, privacy_manager):
        """Test password strength analysis across different algorithms."""
        # Test different password strengths
        weak_password = "password"
        medium_password = "Password123"
        strong_password = "SecurePassword123!@#"
        
        # Derive keys with different algorithms
        algorithms = ["pbkdf2", "scrypt", "argon2"]
        
        for algorithm in algorithms:
            # Weak password
            _, _, weak_metadata = await privacy_manager.derive_key_from_password_enhanced(
                weak_password, algorithm=algorithm, store_key=False
            )
            
            # Medium password
            _, _, medium_metadata = await privacy_manager.derive_key_from_password_enhanced(
                medium_password, algorithm=algorithm, store_key=False
            )
            
            # Strong password
            _, _, strong_metadata = await privacy_manager.derive_key_from_password_enhanced(
                strong_password, algorithm=algorithm, store_key=False
            )
            
            # Verify strength scores are in expected order
            assert weak_metadata['strength_score'] < medium_metadata['strength_score']
            assert medium_metadata['strength_score'] < strong_metadata['strength_score']
            
            # Verify algorithm bonus is applied
            if algorithm == "argon2":
                assert strong_metadata['strength_score'] > 80
            elif algorithm == "scrypt":
                assert strong_metadata['strength_score'] > 70
            elif algorithm == "pbkdf2":
                assert strong_metadata['strength_score'] > 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
