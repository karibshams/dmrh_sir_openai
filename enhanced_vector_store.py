"""
Enhanced FAISS vector store creation with improved chunking for tables and structured data.
Preserves tables, maintains semantic coherence, and adds metadata.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import re

load_dotenv()

class EnhancedVectorStoreCreator:
    def __init__(self, data_dir="data", output_path="vectorstore1/db_faiss"):
        self.data_dir = data_dir
        self.output_path = output_path
    
    def preprocess_documents(self, documents):
        """
        Preprocess documents to preserve tables and structured data.
        Add metadata to track document source and content type.
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            # Add source metadata
            if not doc.metadata:
                doc.metadata = {}
            
            # Identify if content contains tables
            if self._is_table_content(doc.page_content):
                doc.metadata['content_type'] = 'table'
                doc.metadata['preserve'] = True
            else:
                doc.metadata['content_type'] = 'narrative'
            
            doc.metadata['doc_id'] = i
            processed_docs.append(doc)
        
        return processed_docs
    
    def _is_table_content(self, text):
        """Detect if text contains table-like structure"""
        table_indicators = [
            '|', '─', '═', '┌', '┐', '└', '┘',
            'Credits', 'Prerequisite', 'Course', 'Fee',
            'Department', 'Semester', 'Requirements'
        ]
        return any(indicator in text for indicator in table_indicators)
    
    def create_vector_store(self):
        """
        Create enhanced FAISS vector store with optimized chunking.
        """
        try:
            print("\n" + "=" * 70)
            print("🔨 Building Enhanced Vector Store from PDFs")
            print("=" * 70 + "\n")
            
            # Load PDFs
            print("📚 Loading PDF documents...")
            documents = []
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
            
            if not pdf_files:
                print(f"❌ No PDF files found in {self.data_dir}")
                return False
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.data_dir, pdf_file)
                print(f"  - Loading: {pdf_file}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Add source filename to metadata
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file
                
                documents.extend(docs)
            
            print(f"✓ Loaded {len(documents)} pages\n")
            
            # Preprocess documents
            print("🔍 Preprocessing documents...")
            documents = self.preprocess_documents(documents)
            print("✓ Documents preprocessed with metadata\n")
            
            # Smart chunking strategy
            print("✂️  Splitting documents with smart chunking...")
            
            # Separate table and narrative content
            table_docs = [d for d in documents if d.metadata.get('content_type') == 'table']
            narrative_docs = [d for d in documents if d.metadata.get('content_type') != 'table']
            
            chunks = []
            
            # For tables: larger chunks to preserve structure
            if table_docs:
                print("  - Processing tables (preserving structure)...")
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=300,
                    separators=[
                        "\n\n",
                        "\n",
                        " ",
                        ""
                    ]
                )
                table_chunks = table_splitter.split_documents(table_docs)
                chunks.extend(table_chunks)
                print(f"    ✓ Created {len(table_chunks)} table chunks")
            
            # For narrative: balanced chunks for context
            if narrative_docs:
                print("  - Processing narrative content...")
                narrative_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=250,
                    separators=[
                        "\n\n",
                        "\nFaculty",
                        "\nCourses",
                        "\nDepartment",
                        "\nProgram",
                        "\nRequirement",
                        "\n",
                        " ",
                        ""
                    ]
                )
                narrative_chunks = narrative_splitter.split_documents(narrative_docs)
                chunks.extend(narrative_chunks)
                print(f"    ✓ Created {len(narrative_chunks)} narrative chunks")
            
            print(f"✓ Total chunks created: {len(chunks)}\n")
            
            # Create embeddings
            print("🤖 Creating embeddings...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Embeddings model loaded\n")
            
            # Build FAISS vector store
            print("📦 Building FAISS vector store...")
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model
            )
            print("✓ Vector store created\n")
            
            # Save vector store
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            print(f"💾 Saving vector store to {self.output_path}...")
            vector_store.save_local(self.output_path)
            print("✓ Vector store saved\n")
            
            print("=" * 70)
            print("✅ Enhanced vector store creation completed!")
            print("=" * 70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    creator = EnhancedVectorStoreCreator()
    success = creator.create_vector_store()
    if not success:
        exit(1)