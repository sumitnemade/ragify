#!/usr/bin/env python3
"""
Privacy Management Demo for Ragify Framework

This demo showcases the comprehensive privacy management capabilities
including data anonymization, access control, and compliance features.
"""

import asyncio
import json
from datetime import datetime
from uuid import uuid4
from src.ragify.storage.privacy import PrivacyManager
from src.ragify.models import (
    Context, ContextChunk, ContextSource, SourceType, 
    PrivacyLevel, PrivacyRule, AccessControl
)

async def demo_privacy_levels():
    """Demonstrate different privacy levels and their effects."""
    
    print("🔒 Privacy Levels Demo")
    print("=" * 50)
    
    # Initialize privacy manager
    privacy_manager = PrivacyManager()
    
    # Create test data with different privacy levels
    test_chunks = [
        ContextChunk(
            content="This is public information about machine learning basics",
            source=ContextSource(
                id=str(uuid4()),
                name="Public ML Guide",
                source_type=SourceType.DOCUMENT,
                privacy_level=PrivacyLevel.PUBLIC
            ),
            metadata={"category": "education", "tags": ["ml", "ai"]}
        ),
        ContextChunk(
            content="Internal company policy on data handling procedures",
            source=ContextSource(
                id=str(uuid4()),
                name="Company Policy",
                source_type=SourceType.DOCUMENT,
                privacy_level=PrivacyLevel.PRIVATE
            ),
            metadata={"category": "policy", "department": "hr"}
        ),
        ContextChunk(
            content="Confidential customer data including personal information",
            source=ContextSource(
                id=str(uuid4()),
                name="Customer Database",
                source_type=SourceType.DATABASE,
                privacy_level=PrivacyLevel.ENTERPRISE
            ),
            metadata={"category": "customer", "sensitivity": "high"}
        ),
        ContextChunk(
            content="Highly sensitive financial information and trade secrets",
            source=ContextSource(
                id=str(uuid4()),
                name="Financial Records",
                source_type=SourceType.DATABASE,
                privacy_level=PrivacyLevel.RESTRICTED
            ),
            metadata={"category": "financial", "classification": "secret"}
        )
    ]
    
    print(f"📄 Created {len(test_chunks)} test chunks with different privacy levels")
    
    # Test privacy filtering for different user roles
    user_roles = [
        {"id": "public_user", "role": "public", "clearance": PrivacyLevel.PUBLIC},
        {"id": "employee", "role": "employee", "clearance": PrivacyLevel.PRIVATE},
        {"id": "manager", "role": "manager", "clearance": PrivacyLevel.ENTERPRISE},
        {"id": "executive", "role": "executive", "clearance": PrivacyLevel.RESTRICTED}
    ]
    
    for user in user_roles:
        print(f"\n👤 Testing access for {user['role']} (clearance: {user['clearance']})")
        print("-" * 50)
        
        # Filter chunks based on user's clearance level
        accessible_chunks = await privacy_manager.filter_by_privacy(
            chunks=test_chunks,
            user_clearance=user['clearance'],
            user_id=user['id']
        )
        
        print(f"✅ Accessible chunks: {len(accessible_chunks)}/{len(test_chunks)}")
        
        for i, chunk in enumerate(accessible_chunks, 1):
            privacy_level = chunk.source.privacy_level
            print(f"  {i}. [{privacy_level}] {chunk.content[:60]}...")
            print(f"     Source: {chunk.source.name}")

async def demo_data_anonymization():
    """Demonstrate data anonymization capabilities."""
    
    print(f"\n🕵️  Data Anonymization Demo")
    print("=" * 50)
    
    privacy_manager = PrivacyManager()
    
    # Create sensitive data
    sensitive_chunks = [
        ContextChunk(
            content="Customer John Doe (ID: 12345) purchased $500 worth of products on 2024-01-15",
            source=ContextSource(
                id=str(uuid4()),
                name="Sales Database",
                source_type=SourceType.DATABASE,
                privacy_level=PrivacyLevel.ENTERPRISE
            ),
            metadata={"customer_id": "12345", "amount": 500, "date": "2024-01-15"}
        ),
        ContextChunk(
            content="Employee Jane Smith (SSN: 123-45-6789) works in the Engineering department",
            source=ContextSource(
                id=str(uuid4()),
                name="HR Database",
                source_type=SourceType.DATABASE,
                privacy_level=PrivacyLevel.RESTRICTED
            ),
            metadata={"employee_id": "EMP001", "department": "Engineering"}
        )
    ]
    
    print("📝 Original sensitive data:")
    for i, chunk in enumerate(sensitive_chunks, 1):
        print(f"  {i}. {chunk.content}")
    
    # Anonymize data
    print(f"\n🔒 Anonymizing data...")
    anonymized_chunks = await privacy_manager.anonymize_chunks(
        chunks=sensitive_chunks,
        anonymization_level="high"
    )
    
    print("✅ Anonymized data:")
    for i, chunk in enumerate(anonymized_chunks, 1):
        print(f"  {i}. {chunk.content}")
        print(f"     Metadata: {chunk.metadata}")

async def demo_access_control():
    """Demonstrate access control and permission management."""
    
    print(f"\n🚪 Access Control Demo")
    print("=" * 50)
    
    privacy_manager = PrivacyManager()
    
    # Create access control rules
    access_rules = [
        PrivacyRule(
            resource_pattern="customer/*",
            allowed_roles=["manager", "executive"],
            required_clearance=PrivacyLevel.ENTERPRISE,
            time_restrictions={"start_hour": 9, "end_hour": 17}
        ),
        PrivacyRule(
            resource_pattern="financial/*",
            allowed_roles=["executive"],
            required_clearance=PrivacyLevel.RESTRICTED,
            location_restrictions=["office", "vpn"]
        ),
        PrivacyRule(
            resource_pattern="public/*",
            allowed_roles=["*"],
            required_clearance=PrivacyLevel.PUBLIC
        )
    ]
    
    print("📋 Access Control Rules:")
    for i, rule in enumerate(access_rules, 1):
        print(f"  {i}. {rule.resource_pattern}")
        print(f"     Roles: {rule.allowed_roles}")
        print(f"     Clearance: {rule.required_clearance}")
    
    # Test access control
    test_resources = [
        {"path": "customer/12345", "user_role": "employee", "clearance": PrivacyLevel.PRIVATE},
        {"path": "customer/12345", "user_role": "manager", "clearance": PrivacyLevel.ENTERPRISE},
        {"path": "financial/reports", "user_role": "manager", "clearance": PrivacyLevel.ENTERPRISE},
        {"path": "financial/reports", "user_role": "executive", "clearance": PrivacyLevel.RESTRICTED},
        {"path": "public/docs", "user_role": "public", "clearance": PrivacyLevel.PUBLIC}
    ]
    
    print(f"\n🔍 Testing access control:")
    print("-" * 40)
    
    for resource in test_resources:
        access_granted = await privacy_manager.check_access(
            resource_path=resource["path"],
            user_role=resource["user_role"],
            user_clearance=resource["clearance"]
        )
        
        status = "✅ GRANTED" if access_granted else "❌ DENIED"
        print(f"  {resource['path']} - {resource['user_role']} ({resource['clearance']}): {status}")

async def demo_compliance_features():
    """Demonstrate compliance and audit features."""
    
    print(f"\n📊 Compliance and Audit Demo")
    print("=" * 50)
    
    privacy_manager = PrivacyManager()
    
    # Create sample data access events
    access_events = [
        {
            "user_id": "user123",
            "resource": "customer/12345",
            "action": "read",
            "timestamp": datetime.utcnow(),
            "privacy_level": PrivacyLevel.ENTERPRISE
        },
        {
            "user_id": "user456",
            "resource": "financial/reports",
            "action": "write",
            "timestamp": datetime.utcnow(),
            "privacy_level": PrivacyLevel.RESTRICTED
        }
    ]
    
    print("📝 Logging access events...")
    for event in access_events:
        await privacy_manager.log_access_event(
            user_id=event["user_id"],
            resource=event["resource"],
            action=event["action"],
            timestamp=event["timestamp"],
            privacy_level=event["privacy_level"]
        )
        print(f"  ✅ Logged: {event['user_id']} {event['action']} {event['resource']}")
    
    # Generate compliance report
    print(f"\n📊 Generating compliance report...")
    compliance_report = await privacy_manager.generate_compliance_report(
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow()
    )
    
    print("✅ Compliance Report Generated:")
    print(f"  - Total access events: {compliance_report.get('total_events', 0)}")
    print(f"  - Privacy violations: {compliance_report.get('violations', 0)}")
    print(f"  - Compliance score: {compliance_report.get('compliance_score', 0):.1f}%")

async def main():
    """Run all privacy management demos."""
    
    print("🔒 Ragify Privacy Management Demo")
    print("=" * 60)
    print("This demo showcases privacy controls, data anonymization,")
    print("access control, and compliance features.")
    print()
    
    try:
        # Run privacy level demo
        await demo_privacy_levels()
        
        # Run data anonymization demo
        await demo_data_anonymization()
        
        # Run access control demo
        await demo_access_control()
        
        # Run compliance features demo
        await demo_compliance_features()
        
        print(f"\n🎉 Privacy Management Demo completed successfully!")
        print("Key features demonstrated:")
        print("  ✅ Privacy level filtering")
        print("  ✅ Data anonymization")
        print("  ✅ Access control rules")
        print("  ✅ Compliance and audit logging")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
