#!/usr/bin/env python3
"""
Updates Engine Demo for Ragify Framework

This demo showcases the updates engine capabilities including
incremental updates, change detection, and synchronization.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
from ragify.engines.updates import UpdatesEngine
from ragify.models import (
    Context, ContextChunk, ContextSource, SourceType, 
    PrivacyLevel, RelevanceScore, UpdateType
)

async def demo_incremental_updates():
    """Demonstrate incremental update capabilities."""
    
    print("🔄 Incremental Updates Demo")
    print("=" * 50)
    
    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        
        # Initialize updates engine
        updates_engine = UpdatesEngine(
            storage_path=storage_path,
            change_detection=True,
            incremental_processing=True
        )
        
        try:
            # Connect to updates engine
            await updates_engine.connect()
            print(f"✅ Connected to updates engine at: {storage_path}")
            
            # Create initial context
            initial_context = Context(
                query="machine learning overview",
                chunks=[
                    ContextChunk(
                        content="Machine learning is a subset of artificial intelligence",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="ML Guide v1",
                            source_type=SourceType.DOCUMENT,
                            version="1.0"
                        ),
                        relevance_score=RelevanceScore(score=0.9),
                        created_at=datetime.utcnow() - timedelta(days=7)
                    )
                ],
                user_id="demo_user"
            )
            
            # Store initial context
            context_id = await updates_engine.store_context(initial_context)
            print(f"✅ Stored initial context: {context_id}")
            
            # Create updated context with new information
            updated_context = Context(
                query="machine learning overview",
                chunks=[
                    ContextChunk(
                        content="Machine learning is a subset of artificial intelligence",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="ML Guide v2",
                            source_type=SourceType.DOCUMENT,
                            version="2.0"
                        ),
                        relevance_score=RelevanceScore(score=0.95),
                        created_at=datetime.utcnow()
                    ),
                    ContextChunk(
                        content="Deep learning has revolutionized the field in recent years",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="ML Guide v2",
                            source_type=SourceType.DOCUMENT,
                            version="2.0"
                        ),
                        relevance_score=RelevanceScore(score=0.9),
                        created_at=datetime.utcnow()
                    )
                ],
                user_id="demo_user"
            )
            
            # Process incremental update
            print(f"\n🔄 Processing incremental update...")
            update_result = await updates_engine.process_update(
                context_id=context_id,
                updated_context=updated_context,
                update_type=UpdateType.INCREMENTAL
            )
            
            print(f"✅ Update processed:")
            print(f"  - New chunks added: {update_result.get('chunks_added', 0)}")
            print(f"  - Chunks modified: {update_result.get('chunks_modified', 0)}")
            print(f"  - Chunks removed: {update_result.get('chunks_removed', 0)}")
            print(f"  - Update timestamp: {update_result.get('update_timestamp')}")
            
            # Retrieve updated context
            final_context = await updates_engine.get_context(context_id)
            if final_context:
                print(f"✅ Final context has {len(final_context.chunks)} chunks")
                for i, chunk in enumerate(final_context.chunks, 1):
                    print(f"  {i}. {chunk.content[:60]}...")
            
        except Exception as e:
            print(f"❌ Incremental updates failed: {e}")
        finally:
            await updates_engine.close()

async def demo_change_detection():
    """Demonstrate change detection capabilities."""
    
    print(f"\n🔍 Change Detection Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        
        updates_engine = UpdatesEngine(
            storage_path=storage_path,
            change_detection=True,
            change_threshold=0.1
        )
        
        try:
            await updates_engine.connect()
            print(f"✅ Connected to updates engine at: {storage_path}")
            
            # Create multiple versions of content
            versions = [
                {
                    "version": "1.0",
                    "content": "Machine learning basics for beginners",
                    "timestamp": datetime.utcnow() - timedelta(days=30)
                },
                {
                    "version": "1.1",
                    "content": "Machine learning basics for beginners with examples",
                    "timestamp": datetime.utcnow() - timedelta(days=15)
                },
                {
                    "version": "2.0",
                    "content": "Machine learning fundamentals and practical applications",
                    "timestamp": datetime.utcnow()
                }
            ]
            
            # Store versions and detect changes
            context_id = None
            for i, version in enumerate(versions):
                context = Context(
                    query="machine learning basics",
                    chunks=[
                        ContextChunk(
                            content=version["content"],
                            source=ContextSource(
                                id=str(uuid4()),
                                name=f"ML Guide {version['version']}",
                                source_type=SourceType.DOCUMENT,
                                version=version["version"]
                            ),
                            created_at=version["timestamp"]
                        )
                    ],
                    user_id="demo_user"
                )
                
                if i == 0:
                    # Store first version
                    context_id = await updates_engine.store_context(context)
                    print(f"✅ Stored version {version['version']}: {context_id}")
                else:
                    # Process update and detect changes
                    print(f"\n🔄 Processing version {version['version']}...")
                    change_analysis = await updates_engine.analyze_changes(
                        context_id=context_id,
                        new_content=version["content"]
                    )
                    
                    print(f"✅ Change analysis for {version['version']}:")
                    print(f"  - Content similarity: {change_analysis.get('similarity', 0):.2f}")
                    print(f"  - Change magnitude: {change_analysis.get('change_magnitude', 0):.2f}")
                    print(f"  - Significant changes: {change_analysis.get('significant_changes', False)}")
                    
                    if change_analysis.get('significant_changes', False):
                        print(f"  - Change type: {change_analysis.get('change_type', 'unknown')}")
                        print(f"  - Change description: {change_analysis.get('change_description', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Change detection failed: {e}")
        finally:
            await updates_engine.close()

async def demo_synchronization():
    """Demonstrate synchronization capabilities."""
    
    print(f"\n🔄 Synchronization Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source_storage")
        target_path = os.path.join(temp_dir, "target_storage")
        
        # Initialize source and target engines
        source_engine = UpdatesEngine(
            storage_path=source_path,
            change_detection=True
        )
        
        target_engine = UpdatesEngine(
            storage_path=target_path,
            change_detection=True
        )
        
        try:
            await source_engine.connect()
            await target_engine.connect()
            print(f"✅ Connected to source: {source_path}")
            print(f"✅ Connected to target: {target_path}")
            
            # Create source data
            source_context = Context(
                query="synchronization test",
                chunks=[
                    ContextChunk(
                        content="This content will be synchronized",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="Sync Source",
                            source_type=SourceType.DOCUMENT
                        )
                    )
                ],
                user_id="demo_user"
            )
            
            source_id = await source_engine.store_context(source_context)
            print(f"✅ Created source context: {source_id}")
            
            # Perform initial synchronization
            print(f"\n🔄 Performing initial synchronization...")
            sync_result = await source_engine.synchronize_with(
                target_engine=target_engine,
                sync_mode="full",
                include_metadata=True
            )
            
            print(f"✅ Initial sync completed:")
            print(f"  - Contexts synced: {sync_result.get('contexts_synced', 0)}")
            print(f"  - Chunks synced: {sync_result.get('chunks_synced', 0)}")
            print(f"  - Sync time: {sync_result.get('sync_time', 0):.2f}s")
            
            # Update source data
            updated_source_context = Context(
                query="synchronization test updated",
                chunks=[
                    ContextChunk(
                        content="This content has been updated and needs resync",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="Sync Source Updated",
                            source_type=SourceType.DOCUMENT
                        )
                    )
                ],
                user_id="demo_user"
            )
            
            await source_engine.process_update(
                context_id=source_id,
                updated_context=updated_source_context,
                update_type=UpdateType.FULL
            )
            print(f"✅ Updated source context")
            
            # Perform incremental synchronization
            print(f"\n🔄 Performing incremental synchronization...")
            incremental_sync = await source_engine.synchronize_with(
                target_engine=target_engine,
                sync_mode="incremental",
                since_timestamp=datetime.utcnow() - timedelta(minutes=5)
            )
            
            print(f"✅ Incremental sync completed:")
            print(f"  - Changes synced: {incremental_sync.get('changes_synced', 0)}")
            print(f"  - Sync time: {incremental_sync.get('sync_time', 0):.2f}s")
            
            # Verify synchronization
            target_contexts = await target_engine.list_contexts()
            print(f"✅ Target now contains: {len(target_contexts)} contexts")
            
        except Exception as e:
            print(f"❌ Synchronization failed: {e}")
        finally:
            await source_engine.close()
            await target_engine.close()

async def demo_update_scheduling():
    """Demonstrate update scheduling capabilities."""
    
    print(f"\n⏰ Update Scheduling Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        
        updates_engine = UpdatesEngine(
            storage_path=storage_path,
            change_detection=True,
            scheduling_enabled=True
        )
        
        try:
            await updates_engine.connect()
            print(f"✅ Connected to updates engine at: {storage_path}")
            
            # Schedule periodic updates
            print(f"📅 Scheduling periodic updates...")
            
            # Schedule daily update
            daily_schedule = await updates_engine.schedule_update(
                schedule_type="daily",
                time="09:00",
                timezone="UTC",
                update_function="refresh_daily_data"
            )
            print(f"✅ Daily update scheduled: {daily_schedule}")
            
            # Schedule weekly update
            weekly_schedule = await updates_engine.schedule_update(
                schedule_type="weekly",
                day="monday",
                time="06:00",
                timezone="UTC",
                update_function="refresh_weekly_data"
            )
            print(f"✅ Weekly update scheduled: {weekly_schedule}")
            
            # Schedule custom interval update
            custom_schedule = await updates_engine.schedule_update(
                schedule_type="interval",
                interval_minutes=30,
                update_function="refresh_realtime_data"
            )
            print(f"✅ Custom interval update scheduled: {custom_schedule}")
            
            # List scheduled updates
            scheduled_updates = await updates_engine.list_scheduled_updates()
            print(f"\n📋 Scheduled Updates:")
            for update in scheduled_updates:
                print(f"  - {update.get('schedule_type')}: {update.get('update_function')}")
                print(f"    Next run: {update.get('next_run')}")
                print(f"    Status: {update.get('status')}")
            
            # Test immediate update execution
            print(f"\n🚀 Testing immediate update execution...")
            immediate_result = await updates_engine.execute_scheduled_update(
                schedule_id=daily_schedule
            )
            
            print(f"✅ Immediate update executed:")
            print(f"  - Success: {immediate_result.get('success', False)}")
            print(f"  - Execution time: {immediate_result.get('execution_time', 0):.2f}s")
            
            # Cancel a scheduled update
            print(f"\n❌ Canceling custom interval update...")
            await updates_engine.cancel_scheduled_update(custom_schedule)
            print(f"✅ Custom interval update canceled")
            
        except Exception as e:
            print(f"❌ Update scheduling failed: {e}")
        finally:
            await updates_engine.close()

async def main():
    """Run all updates engine demos."""
    
    print("🔄 Ragify Updates Engine Demo")
    print("=" * 60)
    print("This demo showcases incremental updates, change detection,")
    print("synchronization, and update scheduling capabilities.")
    print()
    
    try:
        # Run incremental updates demo
        await demo_incremental_updates()
        
        # Run change detection demo
        await demo_change_detection()
        
        # Run synchronization demo
        await demo_synchronization()
        
        # Run update scheduling demo
        await demo_update_scheduling()
        
        print(f"\n🎉 Updates Engine Demo completed successfully!")
        print("Key features demonstrated:")
        print("  ✅ Incremental updates")
        print("  ✅ Change detection")
        print("  ✅ Data synchronization")
        print("  ✅ Update scheduling")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
