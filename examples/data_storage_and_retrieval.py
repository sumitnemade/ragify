#!/usr/bin/env python3
"""
Storage Engine Demo for Ragify Framework

This demo showcases the storage engine capabilities including
data persistence, backup/restore, and storage optimization.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from ragify.engines.storage import StorageEngine
from ragify.models import (
    Context, ContextChunk, ContextSource, SourceType, 
    PrivacyLevel, RelevanceScore
)

async def demo_storage_operations():
    """Demonstrate basic storage operations."""
    
    print("💾 Storage Operations Demo")
    print("=" * 50)
    
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        
        # Initialize storage engine
        storage_engine = StorageEngine(
            storage_path=storage_path,
            storage_type="file",
            compression=True,
            encryption=False
        )
        
        try:
            # Connect to storage
            await storage_engine.connect()
            print(f"✅ Connected to storage at: {storage_path}")
            
            # Create test context
            test_context = Context(
                query="machine learning basics",
                chunks=[
                    ContextChunk(
                        content="Machine learning is a subset of artificial intelligence",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="ML Guide",
                            source_type=SourceType.DOCUMENT,
                            privacy_level=PrivacyLevel.PUBLIC
                        ),
                        relevance_score=RelevanceScore(score=0.9),
                        created_at=datetime.utcnow()
                    ),
                    ContextChunk(
                        content="Deep learning uses neural networks with multiple layers",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="DL Book",
                            source_type=SourceType.DOCUMENT,
                            privacy_level=PrivacyLevel.PUBLIC
                        ),
                        relevance_score=RelevanceScore(score=0.8),
                        created_at=datetime.utcnow()
                    )
                ],
                user_id="demo_user"
            )
            
            # Store context
            context_id = await storage_engine.store_context(test_context)
            print(f"✅ Stored context with ID: {context_id}")
            
            # Retrieve context
            retrieved_context = await storage_engine.get_context(context_id)
            if retrieved_context:
                print(f"✅ Retrieved context: {retrieved_context.query}")
                print(f"   Chunks: {len(retrieved_context.chunks)}")
            else:
                print("❌ Failed to retrieve context")
            
            # List stored contexts
            context_ids = await storage_engine.list_contexts()
            print(f"✅ Total contexts in storage: {len(context_ids)}")
            
            # Delete context
            await storage_engine.delete_context(context_id)
            print(f"✅ Deleted context: {context_id}")
            
            # Verify deletion
            context_ids_after = await storage_engine.list_contexts()
            print(f"✅ Contexts after deletion: {len(context_ids_after)}")
            
        except Exception as e:
            print(f"❌ Storage operations failed: {e}")
        finally:
            await storage_engine.close()

async def demo_backup_restore():
    """Demonstrate backup and restore functionality."""
    
    print(f"\n🔄 Backup and Restore Demo")
    print("=" * 50)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        backup_path = os.path.join(temp_dir, "backup")
        
        # Initialize storage engine
        storage_engine = StorageEngine(
            storage_path=storage_path,
            storage_type="file",
            compression=True
        )
        
        try:
            await storage_engine.connect()
            print(f"✅ Connected to storage at: {storage_path}")
            
            # Create multiple test contexts
            contexts = []
            for i in range(3):
                context = Context(
                    query=f"test query {i}",
                    chunks=[
                        ContextChunk(
                            content=f"Test content {i}",
                            source=ContextSource(
                                id=str(uuid4()),
                                name=f"Test Source {i}",
                                source_type=SourceType.DOCUMENT
                            )
                        )
                    ],
                    user_id="demo_user"
                )
                context_id = await storage_engine.store_context(context)
                contexts.append(context_id)
                print(f"✅ Stored context {i+1}: {context_id}")
            
            # Create backup
            print(f"\n💾 Creating backup...")
            backup_file = await storage_engine.create_backup(
                backup_path=backup_path,
                backup_name="demo_backup"
            )
            print(f"✅ Backup created: {backup_file}")
            
            # Verify backup exists
            backup_files = await storage_engine.list_backups(backup_path)
            print(f"✅ Available backups: {len(backup_files)}")
            
            # Clear storage
            for context_id in contexts:
                await storage_engine.delete_context(context_id)
            print(f"✅ Cleared storage")
            
            # Restore from backup
            print(f"\n🔄 Restoring from backup...")
            restored_contexts = await storage_engine.restore_backup(
                backup_file=backup_file,
                restore_path=storage_path
            )
            print(f"✅ Restored {len(restored_contexts)} contexts")
            
            # Verify restoration
            context_ids_after = await storage_engine.list_contexts()
            print(f"✅ Contexts after restore: {len(context_ids_after)}")
            
        except Exception as e:
            print(f"❌ Backup/restore failed: {e}")
        finally:
            await storage_engine.close()

async def demo_storage_optimization():
    """Demonstrate storage optimization features."""
    
    print(f"\n⚡ Storage Optimization Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "ragify_storage")
        
        # Initialize storage engine with optimization
        storage_engine = StorageEngine(
            storage_path=storage_path,
            storage_type="file",
            compression=True,
            deduplication=True,
            indexing=True
        )
        
        try:
            await storage_engine.connect()
            print(f"✅ Connected to optimized storage at: {storage_path}")
            
            # Create duplicate content for deduplication test
            duplicate_chunks = [
                ContextChunk(
                    content="This is duplicate content that should be deduplicated",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Source 1",
                        source_type=SourceType.DOCUMENT
                    )
                ),
                ContextChunk(
                    content="This is duplicate content that should be deduplicated",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Source 2",
                        source_type=SourceType.DOCUMENT
                    )
                )
            ]
            
            # Store contexts with duplicate content
            for i in range(2):
                context = Context(
                    query=f"duplicate test {i}",
                    chunks=duplicate_chunks,
                    user_id="demo_user"
                )
                await storage_engine.store_context(context)
                print(f"✅ Stored context {i+1} with duplicate content")
            
            # Run optimization
            print(f"\n🔧 Running storage optimization...")
            optimization_stats = await storage_engine.optimize_storage()
            
            print("✅ Optimization completed:")
            print(f"  - Space saved: {optimization_stats.get('space_saved', 0)} bytes")
            print(f"  - Duplicates removed: {optimization_stats.get('duplicates_removed', 0)}")
            print(f"  - Indexes created: {optimization_stats.get('indexes_created', 0)}")
            
            # Get storage statistics
            storage_stats = await storage_engine.get_storage_stats()
            print(f"\n📊 Storage Statistics:")
            print(f"  - Total size: {storage_stats.get('total_size', 0)} bytes")
            print(f"  - Context count: {storage_stats.get('context_count', 0)}")
            print(f"  - Chunk count: {storage_stats.get('chunk_count', 0)}")
            print(f"  - Compression ratio: {storage_stats.get('compression_ratio', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Storage optimization failed: {e}")
        finally:
            await storage_engine.close()

async def demo_storage_migration():
    """Demonstrate storage migration between different backends."""
    
    print(f"\n🚚 Storage Migration Demo")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source_storage")
        target_path = os.path.join(temp_dir, "target_storage")
        
        # Initialize source storage
        source_storage = StorageEngine(
            storage_path=source_path,
            storage_type="file"
        )
        
        # Initialize target storage
        target_storage = StorageEngine(
            storage_path=target_path,
            storage_type="file",
            compression=True
        )
        
        try:
            await source_storage.connect()
            await target_storage.connect()
            print(f"✅ Connected to source: {source_path}")
            print(f"✅ Connected to target: {target_path}")
            
            # Create test data in source
            test_context = Context(
                query="migration test",
                chunks=[
                    ContextChunk(
                        content="This content will be migrated",
                        source=ContextSource(
                            id=str(uuid4()),
                            name="Migration Source",
                            source_type=SourceType.DOCUMENT
                        )
                    )
                ],
                user_id="demo_user"
            )
            
            source_id = await source_storage.store_context(test_context)
            print(f"✅ Created context in source: {source_id}")
            
            # Migrate data
            print(f"\n🚚 Migrating data...")
            migration_result = await source_storage.migrate_to(
                target_storage=target_storage,
                include_metadata=True,
                verify_integrity=True
            )
            
            print(f"✅ Migration completed:")
            print(f"  - Contexts migrated: {migration_result.get('contexts_migrated', 0)}")
            print(f"  - Chunks migrated: {migration_result.get('chunks_migrated', 0)}")
            print(f"  - Migration time: {migration_result.get('migration_time', 0):.2f}s")
            
            # Verify migration
            target_contexts = await target_storage.list_contexts()
            print(f"✅ Target storage now contains: {len(target_contexts)} contexts")
            
        except Exception as e:
            print(f"❌ Storage migration failed: {e}")
        finally:
            await source_storage.close()
            await target_storage.close()

async def main():
    """Run all storage engine demos."""
    
    print("💾 Ragify Storage Engine Demo")
    print("=" * 60)
    print("This demo showcases storage operations, backup/restore,")
    print("optimization, and migration capabilities.")
    print()
    
    try:
        # Run storage operations demo
        await demo_storage_operations()
        
        # Run backup/restore demo
        await demo_backup_restore()
        
        # Run storage optimization demo
        await demo_storage_optimization()
        
        # Run storage migration demo
        await demo_storage_migration()
        
        print(f"\n🎉 Storage Engine Demo completed successfully!")
        print("Key features demonstrated:")
        print("  ✅ Basic storage operations")
        print("  ✅ Backup and restore")
        print("  ✅ Storage optimization")
        print("  ✅ Data migration")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
