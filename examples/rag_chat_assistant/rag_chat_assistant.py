"""
RAG-based Chat Assistant using RAGify Plugin

A Streamlit-based chatbot that leverages RAGify for intelligent context retrieval
and OpenAI for response generation. Users can upload PDF documents and ask
questions that get answered using retrieved context from the documents.
"""

import os
import sys
import asyncio
import tempfile
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import shutil

# Add the src directory to Python path to import RAGify
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import RAGify components
from ragify import ContextOrchestrator
from ragify.models import PrivacyLevel, SourceType
from ragify.sources.document import DocumentSource
from ragify.storage.vector_db import VectorDatabase

# Import OpenAI
import openai
from openai import OpenAI

# Import environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGChatAssistant:
    """
    RAG-based Chat Assistant using RAGify plugin.
    
    This class handles:
    - PDF document processing and embedding
    - Vector database management
    - Context retrieval using RAGify
    - OpenAI response generation
    """
    
    def __init__(self):
        """Initialize the RAG Chat Assistant."""
        self.orchestrator = None
        self.vector_db = None
        self.document_source = None
        self.is_initialized = False
        self.conversation_history = []
        self.uploaded_files_dir = Path("uploaded_documents")
        
        # Create uploaded documents directory
        self.uploaded_files_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.max_chunks = int(os.getenv("MAX_CHUNKS", 10))
        self.min_relevance = float(os.getenv("MIN_RELEVANCE", 0.5))
        self.max_tokens = int(os.getenv("MAX_TOKENS", 4000))
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAGify components."""
        try:
            # Initialize vector database
            vector_db_url = os.getenv("VECTOR_DB_URL", "memory://")
            self.vector_db = VectorDatabase(vector_db_url)
            
            # Initialize context orchestrator
            self.orchestrator = ContextOrchestrator(
                vector_db_url=vector_db_url,
                cache_url=os.getenv("CACHE_URL", "memory://"),
                privacy_level=PrivacyLevel.PRIVATE
            )
            
            self.is_initialized = True
            st.success("✅ RAGify components initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAGify components: {e}")
            self.is_initialized = False
    
    async def process_pdf_document(self, pdf_file) -> bool:
        """
        Process uploaded PDF document using RAGify.
        
        Args:
            pdf_file: Uploaded PDF file from Streamlit
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                st.error("RAGify components not initialized")
                return False
            
            # Save uploaded file to persistent location
            filename = pdf_file.name
            file_path = self.uploaded_files_dir / filename
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            try:
                # Create document source with persistent file path
                self.document_source = DocumentSource(
                    name="uploaded_pdf",
                    source_type=SourceType.DOCUMENT,
                    url=str(file_path),  # Use persistent path
                    chunk_size=1000,
                    overlap=200
                )
                
                # Add source to orchestrator
                self.orchestrator.add_source(self.document_source)
                
                # Test document processing
                test_chunks = await self.document_source.get_chunks(
                    query="test",
                    max_chunks=1
                )
                
                if test_chunks:
                    st.success(f"✅ PDF processed successfully! Extracted {len(test_chunks)} chunks")
                    
                    # Store document info
                    st.session_state['document_processed'] = True
                    st.session_state['document_name'] = filename
                    st.session_state['chunk_count'] = len(test_chunks)
                    st.session_state['document_path'] = str(file_path)
                    
                    return True
                else:
                    st.warning("⚠️ PDF processed but no chunks extracted")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
                # Clean up file if processing failed
                if file_path.exists():
                    file_path.unlink()
                return False
                    
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            return False
    
    async def retrieve_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant context using RAGify.
        
        Args:
            query: User query
            
        Returns:
            Dict containing context information
        """
        try:
            if not self.is_initialized or not self.document_source:
                return {
                    'success': False,
                    'error': 'Document not processed or RAGify not initialized'
                }
            
            # Get context using RAGify
            context_response = await self.orchestrator.get_context(
                query=query,
                max_chunks=self.max_chunks,
                min_relevance=self.min_relevance,
                max_tokens=self.max_tokens
            )
            
            context = context_response.context
            
            # Extract relevant information
            context_info = {
                'success': True,
                'chunks': [],
                'total_chunks': len(context.chunks),
                'total_tokens': context.total_tokens,
                'processing_time': context_response.processing_time,
                'cache_hit': context_response.cache_hit
            }
            
            # Process chunks
            for chunk in context.chunks:
                chunk_info = {
                    'content': chunk.content,
                    'source': chunk.source.name,
                    'relevance_score': chunk.relevance_score.score if chunk.relevance_score else 0.0,
                    'token_count': chunk.token_count or 0
                }
                context_info['chunks'].append(chunk_info)
            
            return context_info
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_response(self, query: str, context_info: Dict[str, Any]) -> str:
        """
        Generate response using OpenAI with retrieved context.
        
        Args:
            query: User query
            context_info: Retrieved context information
            
        Returns:
            Generated response string
        """
        try:
            if not context_info.get('success'):
                return f"Sorry, I couldn't retrieve relevant context. Error: {context_info.get('error', 'Unknown error')}"
            
            # Prepare context for LLM
            context_text = "\n\n".join([
                f"Context {i+1} (Relevance: {chunk['relevance_score']:.2f}):\n{chunk['content']}"
                for i, chunk in enumerate(context_info['chunks'])
            ])
            
            # Create prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Always base your answers on the given context. If the context doesn't contain enough information 
            to answer the question, say so. Be accurate and helpful."""
            
            user_prompt = f"""Context Information:
{context_text}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, please say so."""
            
            # Generate response
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        
        Args:
            query: User query
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve context
            st.info("🔍 Retrieving relevant context...")
            context_info = await self.retrieve_context(query)
            
            if not context_info.get('success'):
                return {
                    'success': False,
                    'response': context_info.get('error', 'Unknown error'),
                    'context_info': None,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Generate response
            st.info("🤖 Generating AI response...")
            response = await self.generate_response(query, context_info)
            
            # Step 3: Prepare result
            result = {
                'success': True,
                'response': response,
                'context_info': context_info,
                'processing_time': time.time() - start_time
            }
            
            # Add to conversation history
            self.conversation_history.append({
                'query': query,
                'response': response,
                'context_info': context_info,
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'response': f"Error processing query: {str(e)}",
                'context_info': None,
                'processing_time': time.time() - start_time
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        st.session_state['conversation_history'] = []
    
    def cleanup_files(self):
        """Clean up uploaded files."""
        try:
            if self.uploaded_files_dir.exists():
                shutil.rmtree(self.uploaded_files_dir)
                self.uploaded_files_dir.mkdir(exist_ok=True)
        except Exception as e:
            st.warning(f"Could not clean up files: {e}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG-based Chat Assistant")
    st.markdown("Powered by **RAGify** - Intelligent Context Orchestration")
    
    # Initialize session state
    if 'chat_assistant' not in st.session_state:
        st.session_state['chat_assistant'] = RAGChatAssistant()
    
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    if 'document_processed' not in st.session_state:
        st.session_state['document_processed'] = False
    
    chat_assistant = st.session_state['chat_assistant']
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Process document asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        chat_assistant.process_pdf_document(uploaded_file)
                    )
                    loop.close()
                    
                    if success:
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process document")
        
        # Document info
        if st.session_state.get('document_processed'):
            st.success("✅ Document Ready")
            st.info(f"📄 {st.session_state.get('document_name', 'Unknown')}")
            st.info(f"🔢 {st.session_state.get('chunk_count', 0)} chunks extracted")
        
        # Configuration
        st.header("⚙️ Configuration")
        max_chunks = st.slider("Max Chunks", 1, 20, chat_assistant.max_chunks)
        min_relevance = st.slider("Min Relevance", 0.0, 1.0, chat_assistant.min_relevance, 0.1)
        
        # Update configuration
        chat_assistant.max_chunks = max_chunks
        chat_assistant.min_relevance = min_relevance
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            chat_assistant.clear_conversation()
            st.rerun()
        
        # Clean up files
        if st.button("🧹 Clean Up Files"):
            chat_assistant.cleanup_files()
            st.success("Files cleaned up!")
        
        # Status info
        st.header("📊 Status")
        st.success("✅ RAGify Connected")
        st.info("Simple and clean implementation")
    
    # Main content
    if not st.session_state.get('document_processed'):
        st.info("👆 Please upload a PDF document in the sidebar to get started!")
        
        # Show example
        with st.expander("📖 How it works"):
            st.markdown("""
            This RAG-based Chat Assistant uses the **RAGify** plugin to:
            
            1. **📚 Process PDF Documents**: Extract and chunk text content
            2. **🔍 Intelligent Retrieval**: Use vector embeddings to find relevant context
            3. **🤖 AI Generation**: Generate accurate responses using OpenAI
            4. **📊 Simple & Clean**: No external dependencies, just core functionality
            
            **Key Features:**
            - Multi-source context fusion
            - Intelligent conflict resolution
            - Real-time context retrieval
            - Privacy-controlled data handling
            """)
        
        return
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    for i, conv in enumerate(chat_assistant.get_conversation_history()):
        with st.chat_message("user"):
            st.write(conv['query'])
        
        with st.chat_message("assistant"):
            st.write(conv['response'])
            
            # Show context info in expandable section
            if conv.get('context_info') and conv['context_info'].get('success'):
                with st.expander(f"🔍 Context Used ({len(conv['context_info']['chunks'])} chunks)"):
                    for j, chunk in enumerate(conv['context_info']['chunks']):
                        st.markdown(f"**Chunk {j+1}** (Relevance: {chunk['relevance_score']:.2f})")
                        st.text(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process query asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(chat_assistant.process_query(prompt))
                loop.close()
                
                if result['success']:
                    st.write(result['response'])
                    
                    # Show processing info
                    st.info(f"⏱️ Processing time: {result['processing_time']:.2f}s")
                    
                    # Show context info
                    if result['context_info']:
                        with st.expander(f"🔍 Context Used ({result['context_info']['total_chunks']} chunks)"):
                            for i, chunk in enumerate(result['context_info']['chunks']):
                                st.markdown(f"**Chunk {i+1}** (Relevance: {chunk['relevance_score']:.2f})")
                                st.text(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                                st.divider()
                else:
                    st.error(result['response'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Powered by <strong>RAGify</strong> - Intelligent Context Orchestration Framework<br/>
        Built with Streamlit and OpenAI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
