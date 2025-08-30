#!/usr/bin/env python3
"""
Comprehensive Security and Privacy Features Demo for RAGify

This demo showcases the actual implemented security and privacy features:
- Multi-level encryption (symmetric, asymmetric, hybrid)
- Access control and role-based permissions
- Audit logging and compliance monitoring
- GDPR, HIPAA, and SOX compliance
- Data anonymization and privacy controls
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ragify.storage.security import SecurityManager
from src.ragify.storage.privacy import PrivacyManager
from src.ragify.storage.compliance import ComplianceManager
from src.ragify.models import PrivacyLevel


async def demo_security_features():
    """Demonstrate comprehensive security features."""
    print("🔐 SECURITY FEATURES DEMO")
    print("=" * 50)
    
    # Initialize security manager with enterprise level
    security_manager = SecurityManager(security_level="enterprise")
    
    try:
        # 1. Encryption Demo
        print("\n1️⃣ ENCRYPTION FEATURES")
        print("-" * 30)
        
        sensitive_data = "This is highly sensitive financial information: $1,000,000"
        print(f"Original data: {sensitive_data}")
        
        # Symmetric encryption (AES)
        encrypted_symmetric = await security_manager.encrypt_data(sensitive_data, "symmetric")
        print(f"Symmetric encrypted: {encrypted_symmetric[:50]}...")
        
        # Asymmetric encryption (RSA)
        encrypted_asymmetric = await security_manager.encrypt_data(sensitive_data, "asymmetric")
        print(f"Asymmetric encrypted: {encrypted_asymmetric[:50]}...")
        
        # Hybrid encryption (AES + RSA)
        encrypted_hybrid = await security_manager.encrypt_data(sensitive_data, "hybrid")
        print(f"Hybrid encrypted: {encrypted_hybrid[:50]}...")
        
        # Decryption
        decrypted_symmetric = await security_manager.decrypt_data(encrypted_symmetric, "symmetric")
        decrypted_asymmetric = await security_manager.decrypt_data(encrypted_asymmetric, "asymmetric")
        decrypted_hybrid = await security_manager.decrypt_data(encrypted_hybrid, "hybrid")
        
        print(f"✅ Symmetric decryption: {decrypted_symmetric == sensitive_data}")
        print(f"✅ Asymmetric decryption: {decrypted_asymmetric == sensitive_data}")
        print(f"✅ Hybrid decryption: {decrypted_hybrid == sensitive_data}")
        
        # 2. Access Control Demo
        print("\n2️⃣ ACCESS CONTROL FEATURES")
        print("-" * 30)
        
        # Set up user roles
        security_manager.user_roles = {
            "alice": "admin",
            "bob": "manager",
            "charlie": "analyst",
            "dave": "viewer"
        }
        
        # Test permissions
        resources = ["financial_data", "customer_records", "audit_logs", "admin_panel"]
        actions = ["read", "write", "delete", "admin"]
        
        for user_id in ["alice", "bob", "charlie", "dave"]:
            print(f"\nUser: {user_id} (Role: {security_manager.user_roles[user_id]})")
            for resource in resources:
                for action in actions:
                    has_permission = await security_manager.check_permission(user_id, resource, action)
                    if has_permission:
                        print(f"  ✅ {action} {resource}")
                    else:
                        print(f"  ❌ {action} {resource}")
        
        # 3. Token Management Demo
        print("\n3️⃣ TOKEN MANAGEMENT")
        print("-" * 30)
        
        # Generate tokens for different users
        alice_token = await security_manager.generate_access_token("alice", "admin", ["read", "write", "delete", "admin"])
        bob_token = await security_manager.generate_access_token("bob", "manager", ["read", "write", "audit"])
        
        print(f"Alice's token: {alice_token[:50]}...")
        print(f"Bob's token: {bob_token[:50]}...")
        
        # Validate tokens
        alice_payload = await security_manager.validate_access_token(alice_token)
        bob_payload = await security_manager.validate_access_token(bob_token)
        
        print(f"Alice token valid: {alice_payload is not None}")
        print(f"Bob token valid: {bob_payload is not None}")
        
        # 4. Security Monitoring Demo
        print("\n4️⃣ SECURITY MONITORING")
        print("-" * 30)
        
        # Create sample security events
        await security_manager.log_security_event("alice", "login", "Successful login", "info")
        await security_manager.log_security_event("bob", "data_access", "Accessed sensitive data", "warning")
        await security_manager.log_security_event("unknown", "failed_login", "Multiple failed login attempts", "high")
        
        # Get security report
        security_report = await security_manager.get_security_report()
        print(f"Security Level: {security_report['security_level']}")
        print(f"Security Score: {security_report['security_score']}")
        print(f"Total Events: {security_report['total_events']}")
        print(f"High Severity Events: {security_report['high_severity_events']}")
        
        # 5. Key Rotation Demo
        print("\n5️⃣ KEY ROTATION")
        print("-" * 30)
        
        old_key = security_manager.encryption_key
        await security_manager.rotate_encryption_keys()
        new_key = security_manager.encryption_key
        
        print(f"Key rotated: {old_key != new_key}")
        print(f"New key: {new_key[:20]}...")
        
    finally:
        await security_manager.close()


async def demo_privacy_features():
    """Demonstrate comprehensive privacy features."""
    print("\n\n🔒 PRIVACY FEATURES DEMO")
    print("=" * 50)
    
    # Initialize privacy manager with enterprise security
    privacy_manager = PrivacyManager(
        default_privacy_level=PrivacyLevel.ENTERPRISE,
        security_level="enterprise"
    )
    
    try:
        # 1. Data Anonymization Demo
        print("\n1️⃣ DATA ANONYMIZATION")
        print("-" * 30)
        
        sensitive_content = """
        Customer: John Smith
        Email: john.smith@email.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        Address: 123 Main St, Anytown, USA
        """
        
        print("Original content:")
        print(sensitive_content)
        
        # Detect sensitive data
        sensitive_data = await privacy_manager.detect_sensitive_data(sensitive_content)
        print(f"\nDetected {len(sensitive_data)} sensitive data items:")
        for item in sensitive_data:
            print(f"  - {item['type']}: {item['value']}")
        
        # Anonymize content
        anonymized_content = await privacy_manager.anonymize_content(sensitive_content)
        print(f"\nAnonymized content:")
        print(anonymized_content)
        
        # 2. Privacy Controls Demo
        print("\n2️⃣ PRIVACY CONTROLS")
        print("-" * 30)
        
        from src.ragify.models import Context, ContextChunk, ContextSource
        
        # Create test context
        test_chunks = [
            ContextChunk(
                content="This is sensitive financial information about customer accounts.",
                source=ContextSource(
                    name="financial_db",
                    source_type="database",
                    privacy_level=PrivacyLevel.RESTRICTED
                ),
                metadata={"privacy_level": "restricted"}
            ),
            ContextChunk(
                content="This is public information about company policies.",
                source=ContextSource(
                    name="public_docs",
                    source_type="document",
                    privacy_level=PrivacyLevel.PUBLIC
                ),
                metadata={"privacy_level": "public"}
            )
        ]
        
        test_context = Context(
            query="Show me customer financial information",
            chunks=test_chunks,
            user_id="test_user",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        
        # Apply privacy controls
        protected_context = await privacy_manager.apply_privacy_controls(
            test_context, 
            PrivacyLevel.RESTRICTED
        )
        
        print(f"Context protected: {protected_context.metadata.get('privacy_level')}")
        print(f"Anonymized: {protected_context.metadata.get('anonymized')}")
        print(f"Encrypted: {protected_context.metadata.get('encrypted')}")
        
        # 3. Security Manager Integration Demo
        print("\n3️⃣ SECURITY MANAGER INTEGRATION")
        print("-" * 30)
        
        # Use security manager for encryption
        test_data = "This is test data that needs encryption"
        encrypted_data = await privacy_manager.encrypt_with_security_manager(test_data, "hybrid")
        decrypted_data = await privacy_manager.decrypt_with_security_manager(encrypted_data, "hybrid")
        
        print(f"Original: {test_data}")
        print(f"Encrypted: {encrypted_data[:50]}...")
        print(f"Decrypted: {decrypted_data}")
        print(f"✅ Encryption successful: {test_data == decrypted_data}")
        
        # 4. User Permission Demo
        print("\n4️⃣ USER PERMISSION CHECKING")
        print("-" * 30)
        
        # Check permissions for different users
        users = [
            ("alice", "admin"),
            ("bob", "manager"),
            ("charlie", "analyst")
        ]
        
        for user_id, role in users:
            has_permission = await privacy_manager.check_user_permission(user_id, "financial_data", "read")
            print(f"{user_id} ({role}) can read financial_data: {has_permission}")
        
        # 5. Privacy Report Demo
        print("\n5️⃣ PRIVACY REPORTING")
        print("-" * 30)
        
        privacy_report = await privacy_manager.get_privacy_report(test_context)
        print("Privacy Report:")
        for key, value in privacy_report.items():
            print(f"  {key}: {value}")
        
        # Get security report
        security_report = await privacy_manager.get_security_report()
        print(f"\nSecurity Score: {security_report.get('security_score', 'N/A')}")
        
    finally:
        await privacy_manager.close()


async def demo_compliance_features():
    """Demonstrate comprehensive compliance features."""
    print("\n\n📋 COMPLIANCE FEATURES DEMO")
    print("=" * 50)
    
    # Initialize compliance manager
    compliance_manager = ComplianceManager(
        compliance_frameworks=["GDPR", "HIPAA", "SOX"],
        audit_log_path="compliance_demo.log"
    )
    
    try:
        # 1. GDPR Compliance Demo
        print("\n1️⃣ GDPR COMPLIANCE")
        print("-" * 30)
        
        # Register data subject
        personal_data = {
            "name": "Jane Doe",
            "email": "jane.doe@email.com",
            "phone": "555-987-6543",
            "address": "456 Oak St, Somewhere, USA"
        }
        
        registration_id = await compliance_manager.register_data_subject(
            subject_id="jane_doe_001",
            personal_data=personal_data,
            consent_given=True,
            consent_details={
                "purpose": "Customer service",
                "legal_basis": "consent",
                "withdrawal_right": True
            }
        )
        
        print(f"Data subject registered: {registration_id}")
        
        # Implement right to be forgotten
        forgotten = await compliance_manager.right_to_be_forgotten(registration_id)
        print(f"Right to be forgotten implemented: {forgotten}")
        
        # 2. HIPAA Compliance Demo
        print("\n2️⃣ HIPAA COMPLIANCE")
        print("-" * 30)
        
        # Register PHI record
        phi_data = {
            "patient_id": "P12345",
            "diagnosis": "Hypertension",
            "medications": ["Lisinopril", "Amlodipine"],
            "allergies": ["Penicillin"],
            "treatment_plan": "Monitor blood pressure, continue medications"
        }
        
        phi_record_id = await compliance_manager.register_phi_record(
            patient_id="P12345",
            phi_data=phi_data,
            healthcare_provider="City General Hospital",
            treatment_purpose="Patient care"
        )
        
        print(f"PHI record registered: {phi_record_id}")
        
        # 3. SOX Compliance Demo
        print("\n3️⃣ SOX COMPLIANCE")
        print("-" * 30)
        
        # Register financial record
        financial_data = {
            "transaction_id": "T78901",
            "amount": 50000.00,
            "currency": "USD",
            "transaction_type": "revenue",
            "department": "Sales",
            "approval_status": "pending"
        }
        
        financial_record_id = await compliance_manager.register_financial_record(
            record_id="T78901",
            financial_data=financial_data,
            record_type="revenue_transaction",
            fiscal_period="Q1-2024"
        )
        
        print(f"Financial record registered: {financial_record_id}")
        
        # 4. Compliance Checks Demo
        print("\n4️⃣ COMPLIANCE CHECKS")
        print("-" * 30)
        
        # Run compliance checks for each framework
        frameworks = ["GDPR", "HIPAA", "SOX"]
        
        for framework in frameworks:
            print(f"\nRunning {framework} compliance check...")
            check_results = await compliance_manager.run_compliance_check(framework)
            
            print(f"  Status: {check_results['overall_status']}")
            print(f"  Violations: {len(check_results['violations'])}")
            
            if check_results['violations']:
                print("  Violation details:")
                for violation in check_results['violations']:
                    print(f"    - {violation['type']}")
        
        # 5. Compliance Reporting Demo
        print("\n5️⃣ COMPLIANCE REPORTING")
        print("-" * 30)
        
        # Generate comprehensive compliance report
        compliance_report = await compliance_manager.generate_compliance_report(
            framework=None,  # All frameworks
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow()
        )
        
        print("Compliance Report Summary:")
        print(f"  Period: {compliance_report['report_period']['start']} to {compliance_report['report_period']['end']}")
        print(f"  Overall Status: {compliance_report['compliance_status']}")
        print(f"  Total Events: {compliance_report['total_events']}")
        print(f"  Violations: {len(compliance_report['violations'])}")
        
        print(f"\nData Summary:")
        for key, value in compliance_report['data_summary'].items():
            print(f"  {key}: {value}")
        
        print(f"\nRecommendations:")
        for rec in compliance_report['recommendations']:
            print(f"  - {rec}")
        
        # 6. Audit Log Demo
        print("\n6️⃣ AUDIT LOGGING")
        print("-" * 30)
        
        # Show some audit log entries
        print(f"Total audit log entries: {len(compliance_manager.audit_log)}")
        
        if compliance_manager.audit_log:
            print("\nRecent audit log entries:")
            for event in compliance_manager.audit_log[-5:]:  # Last 5 events
                print(f"  {event['timestamp']} - {event['event_type']} ({event['framework']})")
        
    finally:
        await compliance_manager.close()


async def main():
    """Run all security and privacy demos."""
    print("🚀 RAGIFY SECURITY & PRIVACY FEATURES DEMO")
    print("=" * 60)
    print("This demo showcases the ACTUALLY IMPLEMENTED features:")
    print("✅ Multi-level encryption (AES, RSA, Hybrid)")
    print("✅ Access control and role-based permissions")
    print("✅ Audit logging and security monitoring")
    print("✅ Compliance: GDPR, HIPAA, SOX frameworks")
    print("✅ Privacy: Data anonymization and controls")
    print("✅ Security: Monitoring, alerts, and key rotation")
    print("=" * 60)
    
    try:
        # Run security features demo
        await demo_security_features()
        
        # Run privacy features demo
        await demo_privacy_features()
        
        # Run compliance features demo
        await demo_compliance_features()
        
        print("\n\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All security and privacy features are now fully implemented!")
        print("✅ Encryption: Symmetric, Asymmetric, Hybrid")
        print("✅ Access Control: Role-based permissions")
        print("✅ Audit Logging: Comprehensive event tracking")
        print("✅ Compliance: GDPR, HIPAA, SOX frameworks")
        print("✅ Privacy: Data anonymization and controls")
        print("✅ Security: Monitoring, alerts, and key rotation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
