from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import re

load_dotenv()

class EnhancedVectorStoreCreator:
    def __init__(self, data_dir="data", output_path="final_vectorstore/db_faiss"):
        self.data_dir = data_dir
        self.output_path = output_path
    
    def detect_section(self, text):
        text_lower = text.lower()
        
        section_keywords = {
            'curriculum': ['curriculum', 'course', 'program structure', 'credits required', 'credit distribution', 'major requirements'],
            'fees': ['fee', 'tuition', 'cost', 'payment', 'charges', 'bdt', 'scholarship', 'financial'],
            'faculty': ['faculty', 'professor', 'lecturer', 'instructor', 'department', 'dr.', 'dr ', 'chairperson'],
            'admission': ['admission', 'requirement', 'apply', 'prerequisite', 'gpa', 'eligibility', 'applicant'],
            'graduation': ['graduation', 'degree', 'cgpa', 'requirement', 'complete', 'convocation'],
            'course_list': ['cse', 'eng', 'bba', 'course code', 'credits', 'credit hours'],
            'policy': ['policy', 'regulation', 'rule', 'guideline', 'procedure', 'protocol'],
            'mission': ['mission', 'vision', 'objective', 'goal', 'philosophy'],
            'mapping': ['peo', 'po', 'co', 'mapping', 'outcome', 'attainment'],
            'metadata': ['at a glance', 'contact', 'address', 'phone', 'email', 'website'],
            'grading': ['grade', 'grading system', 'gpa', 'cgpa', 'marks', 'evaluation'],
            'flowchart': ['flowchart', 'semester wise', 'year wise', 'sequence'],
        }
        
        for section, keywords in section_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return section
        
        return 'general'
    
    def extract_bulletin_year(self, text, filename):
        year_match = re.search(r'20\d{2}', text)
        if year_match:
            return year_match.group(0)
        
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return year_match.group(0)
        
        return "2024"
    
    def detect_content_type(self, text):
        table_indicators = ['|', '─', '═', '┌', '┐', '└', '┘', 'Course Code', 'Credits', 'Prerequisite', 'Total']
        matrix_indicators = ['peo', 'po', 'co', 'mapping', '✓', '×']
        list_indicators = text.count('\n•') + text.count('\n-') + text.count('\n*')
        numeric_indicators = len(re.findall(r'\d+\.\d+|\d+%|cgpa|gpa', text.lower()))
        
        table_score = sum(indicator in text for indicator in table_indicators)
        matrix_score = sum(indicator in text.lower() for indicator in matrix_indicators)
        
        if matrix_score >= 3:
            return 'matrix'
        elif table_score >= 3:
            return 'table'
        elif list_indicators >= 3:
            return 'list'
        elif numeric_indicators >= 5:
            return 'numeric'
        elif 'prerequisite:' in text.lower() or 'co-requisite:' in text.lower():
            return 'cross_reference'
        else:
            return 'narrative'
    
    def preprocess_documents(self, documents):
        processed_docs = []
        
        for doc_idx, doc in enumerate(documents):
            if not doc.metadata:
                doc.metadata = {}
            
            source_file = doc.metadata.get('source', 'unknown.pdf')
            doc.metadata['source_file'] = source_file
            
            page_num = doc.metadata.get('page', doc_idx)
            doc.metadata['page'] = page_num
            
            section = self.detect_section(doc.page_content)
            doc.metadata['section'] = section
            
            bulletin_year = self.extract_bulletin_year(doc.page_content, source_file)
            doc.metadata['bulletin_year'] = bulletin_year
            
            content_type = self.detect_content_type(doc.page_content)
            doc.metadata['content_type'] = content_type
            
            doc.metadata['doc_id'] = f"{source_file}_{page_num}"
            
            processed_docs.append(doc)
        
        return processed_docs
    
    def create_vector_store(self):
        try:
            print("\n" + "=" * 70)
            print("Building Enhanced Vector Store for EWU")
            print("=" * 70 + "\n")
            
            print("Loading PDF documents...")
            documents = []
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
            
            if not pdf_files:
                print(f"No PDF files found in {self.data_dir}")
                return False
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.data_dir, pdf_file)
                print(f"  - Loading: {pdf_file}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata['source'] = pdf_file
                
                documents.extend(docs)
            
            print(f"Loaded {len(documents)} pages\n")
            
            print("Preprocessing documents with metadata...")
            documents = self.preprocess_documents(documents)
            print("Metadata added\n")
            
            print("Splitting documents with content-aware chunking...")
            
            narrative_docs = [d for d in documents if d.metadata.get('content_type') == 'narrative']
            table_docs = [d for d in documents if d.metadata.get('content_type') == 'table']
            matrix_docs = [d for d in documents if d.metadata.get('content_type') == 'matrix']
            list_docs = [d for d in documents if d.metadata.get('content_type') == 'list']
            numeric_docs = [d for d in documents if d.metadata.get('content_type') == 'numeric']
            cross_ref_docs = [d for d in documents if d.metadata.get('content_type') == 'cross_reference']
            
            chunks = []
            
            if narrative_docs:
                print(f"  - Processing narrative content ({len(narrative_docs)} docs)...")
                narrative_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                    separators=["\n\n", "\nFaculty", "\nDepartment", "\nProgram", "\n", " ", ""]
                )
                narrative_chunks = narrative_splitter.split_documents(narrative_docs)
                chunks.extend(narrative_chunks)
                print(f"    Created {len(narrative_chunks)} narrative chunks")
            
            if table_docs:
                print(f"  - Processing tables ({len(table_docs)} docs)...")
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                table_chunks = table_splitter.split_documents(table_docs)
                chunks.extend(table_chunks)
                print(f"    Created {len(table_chunks)} table chunks")
            
            if matrix_docs:
                print(f"  - Processing matrices ({len(matrix_docs)} docs)...")
                matrix_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", ""]
                )
                matrix_chunks = matrix_splitter.split_documents(matrix_docs)
                chunks.extend(matrix_chunks)
                print(f"    Created {len(matrix_chunks)} matrix chunks")
            
            if list_docs:
                print(f"  - Processing lists ({len(list_docs)} docs)...")
                list_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", " ", ""]
                )
                list_chunks = list_splitter.split_documents(list_docs)
                chunks.extend(list_chunks)
                print(f"    Created {len(list_chunks)} list chunks")
            
            if numeric_docs:
                print(f"  - Processing numeric data ({len(numeric_docs)} docs)...")
                numeric_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", " ", ""]
                )
                numeric_chunks = numeric_splitter.split_documents(numeric_docs)
                chunks.extend(numeric_chunks)
                print(f"    Created {len(numeric_chunks)} numeric chunks")
            
            if cross_ref_docs:
                print(f"  - Processing cross-references ({len(cross_ref_docs)} docs)...")
                cross_ref_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=150,
                    separators=["\n\n", "\nPrerequisite", "\nCo-requisite", "\n", " ", ""]
                )
                cross_ref_chunks = cross_ref_splitter.split_documents(cross_ref_docs)
                chunks.extend(cross_ref_chunks)
                print(f"    Created {len(cross_ref_chunks)} cross-reference chunks")
            
            print(f"\nTotal chunks created: {len(chunks)}\n")
            
            print("Chunk Statistics:")
            sections = {}
            for chunk in chunks:
                section = chunk.metadata.get('section', 'unknown')
                sections[section] = sections.get(section, 0) + 1
            
            for section, count in sorted(sections.items()):
                print(f"    - {section}: {count} chunks")
            print()
            
            print("Creating embeddings...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("Embeddings model loaded\n")
            
            print("Building FAISS vector store...")
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model
            )
            print("Vector store created\n")
            
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            print(f"Saving vector store to {self.output_path}...")
            vector_store.save_local(self.output_path)
            print("Vector store saved\n")
            
            print("=" * 70)
            print("Vector store creation completed successfully")
            print("=" * 70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    creator = EnhancedVectorStoreCreator()
    success = creator.create_vector_store()
    if not success:
        exit(1)