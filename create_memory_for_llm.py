from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
    
    def load(self):
        try:
            print("🤖 Loading embeddings...")
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
            print("✓ Embeddings loaded")
            return self.embedding_model
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class VectorStoreManager:
    def __init__(self, db_path="vectorstore/db_faiss", embedding_model=None):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.db = None
    
    def load(self):
        try:
            print("📂 Loading vector database...")
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Vector store not found at {self.db_path}")
            self.db = FAISS.load_local(
                self.db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print("✓ Database loaded")
            return self.db
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class LLMManager:
    def __init__(self, model="gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.llm = None
    
    def load(self):
        try:
            print("🚀 Loading OpenAI LLM...")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.3,
                max_tokens=1024
            )
            print(f"✓ LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        self.system_prompt = """You are an AI-powered Academic Assistant for East West University (EWU).

INSTRUCTIONS:
1. Answer using the provided EWU document context.
2. If context contains relevant information, use it and refine for clarity.
3. If context is empty/irrelevant:
   - Say clearly: "This information is not available in the retrieved EWU documents."
   - Do NOT assume or use general knowledge for EWU-specific data (courses, fees, faculty, policies).
4. For general academic questions, you may use general knowledge.
5. Prioritize EWU document information always.
6. Be concise and student-friendly."""
        
        self.custom_prompt_template = """{system_prompt}

EWU Document Context:
{{context}}

Question:
{{question}}

Answer based on the context. If context is insufficient, state that clearly."""
    
    def get_template(self):
        return PromptTemplate(
            template=self.custom_prompt_template.format(system_prompt=self.system_prompt),
            input_variables=["context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="vectorstore/db_faiss", model="gpt-4o-mini", debug=False):
        self.db_path = db_path
        self.model = model
        self.debug = debug
        self.embedding_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager(db_path=db_path)
        self.llm_manager = LLMManager(model=model)
        self.prompt_manager = PromptManager()
        self.qa_chain = None
        self.retriever = None
    
    def initialize(self):
        try:
            print("\n" + "=" * 60)
            print("🔧 Initializing Assistant")
            print("=" * 60 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\n✅ Ready!")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise
    
    def create_qa_chain(self):
        try:
            print("⛓️  Creating QA chain...")
            # CRITICAL FIX: Increased k from 5 to 15 for large documents
            retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 15}
            )
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm_manager.llm
                | StrOutputParser()
            )
            self.retriever = retriever
            print("✓ QA chain created (k=15 for better recall)")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def query(self, question, normalize=True):
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # FIX 5: Question normalization for better retrieval
            if normalize:
                question_normalized = f"East West University EWU {question}"
            else:
                question_normalized = question
            
            # DEBUG: Print retrieved chunks
            if self.debug:
                print("\n🔍 Retrieved Chunks:")
                docs = self.retriever.invoke(question_normalized)
                for i, doc in enumerate(docs[:5], 1):
                    print(f"\n[{i}] (relevance context)")
                    print(doc.page_content[:300] + "...\n")
            
            answer = self.qa_chain.invoke(question_normalized)
            return {'answer': answer}
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

if __name__ == "__main__":
    # Initialize with debug=True to see retrieved chunks
    assistant = AcademicAssistant(debug=True)
    assistant.initialize()
    print("Type 'quit' to exit\n")
    while True:
        try:
            question = input("❓ Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if not question:
                continue
            result = assistant.query(question)
            print(f"\n📝 Answer:\n{result['answer']}\n")
        except KeyboardInterrupt:
            print("\n👋 Ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")