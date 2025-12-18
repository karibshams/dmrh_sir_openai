"""
FIXED enhanced_vector_store.py
- Smaller chunks (400-600 tokens)
- Page-level metadata
- Section tagging (Curriculum, Fees, Faculty, etc.)
- Bulletin year tracking
- Better table preservation
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
    def __init__(self, data_dir="data", output_path="vectorstore2/db_faiss"):
        self.data_dir = data_dir
        self.output_path = output_path
    
    def detect_section(self, text):
        """Detect document section type"""
        text_lower = text.lower()
        
        section_keywords = {
            'curriculum': ['curriculum', 'course', 'program structure', 'credits required'],
            'fees': ['fee', 'tuition', 'cost', 'payment', 'charges', 'bdt'],
            'faculty': ['faculty', 'professor', 'lecturer', 'instructor', 'department', 'dr.', 'dr '],
            'admission': ['admission', 'requirement', 'apply', 'prerequisite', 'gpa'],
            'graduation': ['graduation', 'degree', 'cgpa', 'requirement', 'complete'],
            'course_list': ['cse', 'eng', 'bba', 'course code', 'credits'],
        }
        
        for section, keywords in section_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return section
        
        return 'general'
    
    def extract_bulletin_year(self, text, filename):
        """Extract bulletin year from text or filename"""
        # Try to find year in text
        year_match = re.search(r'20\d{2}', text)
        if year_match:
            return year_match.group(0)
        
        # Try to find year in filename
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return year_match.group(0)
        
        return "2024"  # Default
    
    def preprocess_documents(self, documents):
        """
        Add comprehensive metadata to documents
        """
        processed_docs = []
        
        for doc_idx, doc in enumerate(documents):
            if not doc.metadata:
                doc.metadata = {}
            
            # Add source file
            source_file = doc.metadata.get('source', 'unknown.pdf')
            doc.metadata['source_file'] = source_file
            
            # Add page number
            page_num = doc.metadata.get('page', doc_idx)
            doc.metadata['page'] = page_num
            
            # Detect section
            section = self.detect_section(doc.page_content)
            doc.metadata['section'] = section
            
            # Extract bulletin year
            bulletin_year = self.extract_bulletin_year(doc.page_content, source_file)
            doc.metadata['bulletin_year'] = bulletin_year
            
            # Detect if content is table
            is_table = self._is_table_content(doc.page_content)
            doc.metadata['content_type'] = 'table' if is_table else 'narrative'
            doc.metadata['is_table'] = is_table
            
            # Add document ID
            doc.metadata['doc_id'] = f"{source_file}_{page_num}"
            
            processed_docs.append(doc)
        
        return processed_docs
    
    def _is_table_content(self, text):
        """Detect if text contains table-like structure"""
        table_indicators = [
            '|', '─', '═', '┌', '┐', '└', '┘',
            'Course Code', 'Credits', 'Prerequisite',
            'Department', 'Semester', 'Title',
            'Total', 'Requirements', 'Program'
        ]
        return sum(indicator in text for indicator in table_indicators) >= 2
    
    def create_vector_store(self):
        """
        Create enhanced FAISS vector store with proper chunking
        """
        try:
            print("\n" + "=" * 70)
            print("🔨 Building FIXED Enhanced Vector Store")
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
                    doc.metadata['source'] = pdf_file
                
                documents.extend(docs)
            
            print(f"✓ Loaded {len(documents)} pages\n")
            
            # Preprocess with metadata
            print("🔍 Preprocessing documents with metadata...")
            documents = self.preprocess_documents(documents)
            print("✓ Metadata added (page, section, year, content_type)\n")
            
            # Smart chunking strategy - FIXED with smaller chunks
            print("✂️  Splitting documents with improved chunking...")
            
            table_docs = [d for d in documents if d.metadata.get('is_table')]
            narrative_docs = [d for d in documents if not d.metadata.get('is_table')]
            
            chunks = []
            
            # For tables: preserve structure with smaller chunks
            if table_docs:
                print("  - Processing tables (preserving structure)...")
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,      # FIXED: Reduced from 2000
                    chunk_overlap=150,   # FIXED: Reduced from 300
                    separators=[
                        "\n\n",
                        "\n",
                        " ",
                        ""
                    ]
                )
                table_chunks = table_splitter.split_documents(table_docs)
                chunks.extend(table_chunks)
                print(f"    ✓ Created {len(table_chunks)} table chunks (600 tokens max)")
            
            # For narrative: balanced chunks
            if narrative_docs:
                print("  - Processing narrative content...")
                narrative_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,      # FIXED: Reduced from 1200
                    chunk_overlap=100,   # FIXED: Reduced from 250
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
                print(f"    ✓ Created {len(narrative_chunks)} narrative chunks (500 tokens max)")
            
            print(f"✓ Total chunks created: {len(chunks)}\n")
            
            # Display chunk stats
            print("📊 Chunk Metadata Statistics:")
            sections = {}
            for chunk in chunks:
                section = chunk.metadata.get('section', 'unknown')
                sections[section] = sections.get(section, 0) + 1
            
            for section, count in sorted(sections.items()):
                print(f"    - {section}: {count} chunks")
            print()
            
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
            print("✅ FIXED vector store creation completed!")
            print("=" * 70)
            print("\n✨ Improvements made:")
            print("  ✓ Chunk size reduced (500-600 tokens)")
            print("  ✓ Page numbers tracked")
            print("  ✓ Section types tagged")
            print("  ✓ Bulletin year recorded")
            print("  ✓ Better table preservation")
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