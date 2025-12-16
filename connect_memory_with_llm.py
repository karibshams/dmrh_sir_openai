"""
Script to create and save the FAISS vector store from PDF documents.
Optimized chunking for 524-page EWU bulletin.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(data_dir="data", output_path="vectorstore/db_faiss"):
    """
    Create FAISS vector store with optimal chunking for large academic documents.
    """
    try:
        print("\n" + "=" * 60)
        print("🔨 Building Vector Store from PDFs")
        print("=" * 60 + "\n")
        
        print("📚 Loading PDF documents...")
        documents = []
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {data_dir}")
            return False
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_dir, pdf_file)
            print(f"  - Loading: {pdf_file}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        
        print(f"✓ Loaded {len(documents)} pages\n")
        
        # CRITICAL FIX: Larger chunks for academic structure
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # INCREASED from 400
            chunk_overlap=200,  # INCREASED from 150
            separators=[
                "\n\n",
                "\nFaculty",
                "\nCourses",
                "\nDepartment",
                "\n",
                " ",
                ""
            ]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks\n")
        
        print("🤖 Creating embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ Embeddings ready\n")
        
        print("📦 Building FAISS vector store...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        print("✓ Vector store created\n")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"💾 Saving vector store to {output_path}...")
        vector_store.save_local(output_path)
        print("✓ Vector store saved\n")
        
        print("=" * 60)
        print("✅ Vector store creation completed!")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_vector_store()
    if not success:
        exit(1)