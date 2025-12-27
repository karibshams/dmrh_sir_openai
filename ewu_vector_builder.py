from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os, re
from dotenv import load_dotenv

load_dotenv()

class EWUVectorBuilder:
    def __init__(self, data_dir="data", output_path="ewu_db/faiss"):
        self.data_dir = data_dir
        self.output_path = output_path
    
    def detect_type(self, text):
        """Minimal content type detection"""
        text_lower = text.lower()
        
        if any(k in text_lower for k in ['professor', 'lecturer', 'designation', 'department']):
            return 'faculty'
        if any(k in text_lower for k in ['|', '─', '═', 'code', 'credits', 'prerequisite']):
            return 'table'
        if any(k in text_lower for k in ['subject to', 'except', 'must maintain', 'only if']):
            return 'rules'
        if any(k in text_lower for k in ['peo', 'po', 'co', 'mapping']):
            return 'matrix'
        
        return 'narrative'
    
    def extract_year(self, text, filename):
        """Extract academic year"""
        year = re.search(r'20\d{2}', text)
        if not year:
            year = re.search(r'20\d{2}', filename)
        return year.group(0) if year else "2025"
    
    def mark_content(self, text):
        """Mark table, rules, faculty data"""
        if '|' in text or any(x in text for x in ['─', '═', '┌']):
            text = f"[TABLE]\n{text}\n[/TABLE]"
        
        if any(x in text.lower() for x in ['subject to', 'except', 'must maintain']):
            text = re.sub(r'(subject to|except|must maintain|only if)([^.]*\.)', 
                         r'[RULE: \1\2]', text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess(self, documents):
        """Add metadata and mark content"""
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            
            source = doc.metadata.get('source', 'unknown.pdf')
            doc.metadata['source_file'] = source
            doc.metadata['page'] = doc.metadata.get('page', 0)
            doc.metadata['content_type'] = self.detect_type(doc.page_content)
            doc.metadata['year'] = self.extract_year(doc.page_content, source)
            
            # Mark important content
            doc.page_content = self.mark_content(doc.page_content)
        
        return documents
    
    def create(self):
        print("\n" + "="*60)
        print("Building EWU Vector Store (LEAN VERSION)")
        print("="*60 + "\n")
        
        # Load PDFs
        documents = []
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDFs in {self.data_dir}")
            return False
        
        for pdf_file in pdf_files:
            print(f"📄 Loading: {pdf_file}")
            loader = PyPDFLoader(os.path.join(self.data_dir, pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = pdf_file
            documents.extend(docs)
        
        print(f"✅ Loaded {len(documents)} pages\n")
        
        # Preprocess
        documents = self.preprocess(documents)
        
        # Smart chunking based on type
        chunks = []
        
        faculty_docs = [d for d in documents if d.metadata['content_type'] == 'faculty']
        table_docs = [d for d in documents if d.metadata['content_type'] == 'table']
        rule_docs = [d for d in documents if d.metadata['content_type'] == 'rules']
        other_docs = [d for d in documents if d.metadata['content_type'] in ['narrative', 'matrix']]
        
        # Faculty: keep intact for accuracy
        if faculty_docs:
            print(f"📋 Faculty rosters: {len(faculty_docs)} docs")
            faculty_split = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=300, separators=["\n\n", "\n", " "]
            )
            chunks.extend(faculty_split.split_documents(faculty_docs))
        
        # Tables: preserve structure
        if table_docs:
            print(f"📊 Tables: {len(table_docs)} docs")
            table_split = RecursiveCharacterTextSplitter(
                chunk_size=2500, chunk_overlap=400, separators=["[TABLE]", "\n\n", "\n"]
            )
            chunks.extend(table_split.split_documents(table_docs))
        
        # Rules: keep conditions together
        if rule_docs:
            print(f"⚖️  Rules: {len(rule_docs)} docs")
            rule_split = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=300, separators=["[RULE:", "\n\n", "\n"]
            )
            chunks.extend(rule_split.split_documents(rule_docs))
        
        # Others
        if other_docs:
            print(f"📝 Other: {len(other_docs)} docs")
            other_split = RecursiveCharacterTextSplitter(
                chunk_size=1200, chunk_overlap=250, separators=["\n\n", "\n"]
            )
            chunks.extend(other_split.split_documents(other_docs))
        
        print(f"✅ Total chunks: {len(chunks)}\n")
        
        # Embed
        print("🔧 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("🗄️  Building FAISS...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        vector_store.save_local(self.output_path)
        
        print(f"✅ Saved to {self.output_path}\n")
        print("="*60)
        print("Vector store ready!")
        print("="*60 + "\n")
        
        return True

if __name__ == "__main__":
    builder = EWUVectorBuilder()
    if not builder.create():
        exit(1)