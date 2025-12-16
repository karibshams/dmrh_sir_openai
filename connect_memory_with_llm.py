"""
Script to create and save the FAISS vector store from PDF documents.
Run this before starting the Streamlit app.
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
    Create and save FAISS vector store from PDF files.
    
    Args:
        data_dir: Directory containing PDF files
        output_path: Path where the vector store will be saved
    """
    try:
        print("\n" + "=" * 60)
        print("🔨 Building Vector Store from PDFs")
        print("=" * 60 + "\n")
        
        # Load PDF documents
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
        
        # Split documents into smaller chunks for better retrieval
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks\n")
        
        # Create embeddings
        print("🤖 Creating embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ Embeddings ready\n")
        
        # Create FAISS vector store
        print("📦 Building FAISS vector store...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        print("✓ Vector store created\n")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the vector store
        print(f"💾 Saving vector store to {output_path}...")
        vector_store.save_local(output_path)
        print("✓ Vector store saved\n")
        
        print("=" * 60)
        print("✅ Vector store creation completed successfully!")
        print("=" * 60 + "\n")
        print("You can now run: streamlit run ewu_academic_ui.py\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_vector_store()
    if not success:
        exit(1)