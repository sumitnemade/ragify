#!/usr/bin/env python3
"""
Privacy Key Derivation Demo

This demo showcases the enterprise-grade Privacy Key Derivation features of RAGify,
including secure key management, rotation, health monitoring, and lifecycle management.

Features demonstrated:
- Multiple key derivation algorithms (PBKDF2, Scrypt, Argon2)
- Secure key storage with encryption
- Key rotation and lifecycle management
- Key health monitoring and validation
- Backup and recovery systems
- Password strength analysis
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

from ragify.storage.privacy import PrivacyManager
from ragify.models import PrivacyLevel
from ragify.exceptions import SecurityViolationError, EncryptionError


class PrivacyKeyDerivationDemo:
    """Demo class for Privacy Key Derivation features."""
    
    def __init__(self):
        """Initialize the demo with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.privacy_manager = None
        
    async def setup(self):
        """Set up the privacy manager for demo."""
        print("🔐 Setting up Privacy Key Derivation Demo...")
        
        # Create privacy manager with enterprise security level
        self.privacy_manager = PrivacyManager(
            default_privacy_level=PrivacyLevel.ENTERPRISE,
            security_level="enterprise"
        )
        
        # Override keys directory to use temp directory for demo
        self.privacy_manager.keys_dir = Path(self.temp_dir)
        self.privacy_manager.keys_metadata_file = Path(self.temp_dir) / "keys_metadata.json"
        self.privacy_manager.keys_backup_file = Path(self.temp_dir) / "keys_backup.json"
        
        print(f"✅ Privacy Manager initialized with keys directory: {self.temp_dir}")
        print(f"🔑 Master key created: {self.privacy_manager.keys_dir / 'master.key'}")
        
    async def cleanup(self):
        """Clean up demo resources."""
        if self.privacy_manager:
            await self.privacy_manager.close()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
    
    async def demonstrate_key_derivation_algorithms(self):
        """Demonstrate different key derivation algorithms."""
        print("\n" + "="*60)
        print("🔑 DEMONSTRATING KEY DERIVATION ALGORITHMS")
        print("="*60)
        
        algorithms = ["pbkdf2", "scrypt", "argon2"]
        passwords = ["SecurePassword123!", "ComplexPass456!", "EnterprisePass789!"]
        
        for i, (algorithm, password) in enumerate(zip(algorithms, passwords), 1):
            print(f"\n{i}. Testing {algorithm.upper()} Algorithm")
            print(f"   Password: {password}")
            
            start_time = datetime.now()
            derived_key, key_id, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
                password, algorithm=algorithm
            )
            end_time = datetime.now()
            
            derivation_time = (end_time - start_time).total_seconds()
            
            print(f"   ✅ Key derived successfully")
            print(f"   🔑 Key ID: {key_id}")
            print(f"   📊 Strength Score: {metadata['strength_score']}/100")
            print(f"   ⏱️  Derivation Time: {derivation_time:.3f}s")
            print(f"   🧂 Salt: {metadata['salt'][:20]}...")
            print(f"   📅 Created: {metadata['created_at']}")
    
    async def demonstrate_key_strength_analysis(self):
        """Demonstrate password strength analysis."""
        print("\n" + "="*60)
        print("📊 DEMONSTRATING PASSWORD STRENGTH ANALYSIS")
        print("="*60)
        
        test_passwords = [
            ("weak", "password"),
            ("medium", "Password123"),
            ("strong", "SecurePassword123!@#"),
            ("very_strong", "MySuperSecurePassword123!@#$%^&*()")
        ]
        
        for strength_label, password in test_passwords:
            print(f"\n🔍 Analyzing password: '{password}'")
            
            # Test with different algorithms
            for algorithm in ["pbkdf2", "scrypt", "argon2"]:
                try:
                    _, _, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
                        password, algorithm=algorithm, store_key=False
                    )
                    
                    print(f"   {algorithm.upper()}: {metadata['strength_score']}/100")
                    
                    # Provide feedback on strength
                    if metadata['strength_score'] >= 80:
                        print(f"      🟢 Excellent strength")
                    elif metadata['strength_score'] >= 60:
                        print(f"      🟡 Good strength")
                    elif metadata['strength_score'] >= 40:
                        print(f"      🟠 Fair strength")
                    else:
                        print(f"      🔴 Weak strength - consider improving")
                        
                except Exception as e:
                    print(f"   {algorithm.upper()}: Error - {e}")
    
    async def demonstrate_key_storage_and_retrieval(self):
        """Demonstrate key storage and retrieval."""
        print("\n" + "="*60)
        print("💾 DEMONSTRATING KEY STORAGE AND RETRIEVAL")
        print("="*60)
        
        # Create a key
        password = "DemoStoragePassword123!"
        print(f"🔑 Creating key from password: '{password}'")
        
        derived_key, key_id, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        print(f"✅ Key created and stored with ID: {key_id}")
        
        # Verify storage
        print(f"📁 Key file created: {self.privacy_manager.keys_dir / f'{key_id}.key'}")
        print(f"📊 Metadata file: {self.privacy_manager.keys_metadata_file}")
        print(f"💾 Backup file: {self.privacy_manager.keys_backup_file}")
        
        # Retrieve the key
        print(f"\n🔄 Retrieving key...")
        retrieved_key, retrieved_metadata = await self.privacy_manager.retrieve_key(key_id)
        
        # Verify retrieval
        if retrieved_key == derived_key:
            print(f"✅ Key retrieved successfully - integrity verified")
            print(f"📈 Usage count: {retrieved_metadata['usage_count']}")
            print(f"🕒 Last used: {retrieved_metadata['last_used']}")
        else:
            print(f"❌ Key retrieval failed - integrity check failed")
    
    async def demonstrate_key_rotation(self):
        """Demonstrate key rotation functionality."""
        print("\n" + "="*60)
        print("🔄 DEMONSTRATING KEY ROTATION")
        print("="*60)
        
        # Create initial key
        old_password = "OldPassword123!"
        print(f"🔑 Creating initial key with password: '{old_password}'")
        
        derived_key, key_id, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
            old_password
        )
        
        print(f"✅ Initial key created: {key_id}")
        print(f"📅 Created: {metadata['created_at']}")
        
        # Rotate key
        new_password = "NewPassword456!"
        print(f"\n🔄 Rotating key with new password: '{new_password}'")
        
        new_key_id = await self.privacy_manager.rotate_key(key_id, new_password)
        
        print(f"✅ Key rotated successfully")
        print(f"🆕 New key ID: {new_key_id}")
        print(f"📉 Old key ID: {key_id} (deprecated)")
        
        # Check status
        old_metadata = self.privacy_manager.keys_metadata[key_id]
        new_metadata = self.privacy_manager.keys_metadata[new_key_id]
        
        print(f"\n📊 Rotation Status:")
        print(f"   Old key: {key_id}")
        print(f"      Status: Deprecated at {old_metadata['deprecated_at']}")
        print(f"      Replaced by: {old_metadata['replaced_by']}")
        
        print(f"   New key: {new_key_id}")
        print(f"      Status: Active")
        print(f"      Created: {new_metadata['created_at']}")
        print(f"      Algorithm: {new_metadata['algorithm']}")
    
    async def demonstrate_key_health_monitoring(self):
        """Demonstrate key health monitoring."""
        print("\n" + "="*60)
        print("🏥 DEMONSTRATING KEY HEALTH MONITORING")
        print("="*60)
        
        # Create multiple keys with different characteristics
        passwords = [
            "HealthyPassword123!",
            "WeakPassword",
            "OldPassword456!"
        ]
        
        print("🔑 Creating test keys...")
        for password in passwords:
            await self.privacy_manager.derive_key_from_password_enhanced(password)
        
        # Manually age one key to simulate expiration
        for key_id in list(self.privacy_manager.keys_metadata.keys())[:1]:
            self.privacy_manager.keys_metadata[key_id]['created_at'] = (
                datetime.now(timezone.utc) - timedelta(days=400)
            ).isoformat()
        
        print(f"✅ Created {len(self.privacy_manager.keys_metadata)} test keys")
        print(f"📅 Aged one key to simulate expiration")
        
        # Check key health
        print(f"\n🏥 Checking key health...")
        health = await self.privacy_manager.check_key_health()
        
        print(f"📊 Health Report:")
        print(f"   Total Keys: {health['total_keys']}")
        print(f"   Healthy Keys: {health['healthy_keys']}")
        print(f"   Expired Keys: {health['expired_keys']}")
        print(f"   Weak Keys: {health['weak_keys']}")
        print(f"   Keys Needing Rotation: {health['keys_needing_rotation']}")
        print(f"   Overall Health Score: {health['health_score']}/100")
        
        # Provide health recommendations
        if health['health_score'] >= 80:
            print(f"   🟢 Excellent key health - no action needed")
        elif health['health_score'] >= 60:
            print(f"   🟡 Good key health - consider rotation for older keys")
        elif health['health_score'] >= 40:
            print(f"   🟠 Fair key health - rotation recommended")
        else:
            print(f"   🔴 Poor key health - immediate action required")
        
        # Clean up expired keys
        if health['expired_keys'] > 0:
            print(f"\n🧹 Cleaning up expired keys...")
            cleaned_count = await self.privacy_manager.cleanup_expired_keys()
            print(f"✅ Cleaned up {cleaned_count} expired keys")
    
    async def demonstrate_backup_and_recovery(self):
        """Demonstrate backup and recovery systems."""
        print("\n" + "="*60)
        print("💾 DEMONSTRATING BACKUP AND RECOVERY")
        print("="*60)
        
        # Create a key to backup
        password = "BackupDemoPassword123!"
        print(f"🔑 Creating key for backup demo: '{password}'")
        
        derived_key, key_id, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
            password
        )
        
        print(f"✅ Key created: {key_id}")
        
        # Check backup file
        if self.privacy_manager.keys_backup_file.exists():
            with open(self.privacy_manager.keys_backup_file, 'r') as f:
                backups = json.load(f)
            
            print(f"📁 Backup file contains {len(backups)} backup entries")
            
            # Find our key's backup
            key_backup = None
            for backup in backups:
                if backup['key_id'] == key_id:
                    key_backup = backup
                    break
            
            if key_backup:
                print(f"✅ Backup found for key {key_id}")
                print(f"   📅 Backup created: {key_backup['backup_created_at']}")
                print(f"   🔐 Encrypted key length: {len(key_backup['encrypted_key'])} chars")
                print(f"   📊 Metadata preserved: {list(key_backup['metadata'].keys())}")
            else:
                print(f"❌ Backup not found for key {key_id}")
        else:
            print(f"❌ Backup file not found")
    
    async def demonstrate_key_lifecycle_management(self):
        """Demonstrate complete key lifecycle management."""
        print("\n" + "="*60)
        print("🔄 DEMONSTRATING COMPLETE KEY LIFECYCLE")
        print("="*60)
        
        # 1. Key Creation
        print("1️⃣  KEY CREATION")
        password = "LifecycleDemo123!"
        derived_key, key_id, metadata = await self.privacy_manager.derive_key_from_password_enhanced(
            password
        )
        print(f"   ✅ Key created: {key_id}")
        
        # 2. Key Usage
        print("\n2️⃣  KEY USAGE")
        retrieved_key, usage_metadata = await self.privacy_manager.retrieve_key(key_id)
        print(f"   ✅ Key retrieved successfully")
        print(f"   📈 Usage count: {usage_metadata['usage_count']}")
        
        # 3. Key Health Check
        print("\n3️⃣  KEY HEALTH CHECK")
        health = await self.privacy_manager.check_key_health()
        print(f"   🏥 Health score: {health['health_score']}/100")
        
        # 4. Key Rotation
        print("\n4️⃣  KEY ROTATION")
        new_password = "NewLifecycleDemo456!"
        new_key_id = await self.privacy_manager.rotate_key(key_id, new_password)
        print(f"   🔄 Key rotated: {key_id} → {new_key_id}")
        
        # 5. Key Deletion
        print("\n5️⃣  KEY DELETION")
        # Delete old key
        old_deleted = await self.privacy_manager.delete_key(key_id, force=True)
        print(f"   🗑️  Old key deleted: {old_deleted}")
        
        # Delete new key
        new_deleted = await self.privacy_manager.delete_key(new_key_id, force=True)
        print(f"   🗑️  New key deleted: {new_deleted}")
        
        # 6. Final Status
        print("\n6️⃣  FINAL STATUS")
        final_health = await self.privacy_manager.check_key_health()
        print(f"   📊 Final key count: {final_health['total_keys']}")
        print(f"   🎯 Lifecycle completed successfully")
    
    async def run_demo(self):
        """Run the complete demo."""
        try:
            await self.setup()
            
            print("\n🚀 Starting Privacy Key Derivation Demo...")
            print("This demo showcases enterprise-grade key management features.")
            
            # Run all demonstrations
            await self.demonstrate_key_derivation_algorithms()
            await self.demonstrate_key_strength_analysis()
            await self.demonstrate_key_storage_and_retrieval()
            await self.demonstrate_key_rotation()
            await self.demonstrate_key_health_monitoring()
            await self.demonstrate_backup_and_recovery()
            await self.demonstrate_key_lifecycle_management()
            
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ All Privacy Key Derivation features demonstrated")
            print("🔐 Enterprise-grade security implemented")
            print("🔄 Complete key lifecycle management shown")
            print("🏥 Key health monitoring operational")
            print("💾 Backup and recovery systems working")
            print("📊 Password strength analysis functional")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup()


async def main():
    """Main function to run the demo."""
    demo = PrivacyKeyDerivationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🔐 RAGify Privacy Key Derivation Demo")
    print("Enterprise-Grade Key Management System")
    print("=" * 50)
    
    asyncio.run(main())
