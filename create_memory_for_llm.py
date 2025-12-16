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

Your primary responsibility is to answer questions using the provided academic context from official EWU documents.

Rules:
- Always prioritize information from the provided context.
- Answers must be short, clear, and directly related to the question.
- Do NOT provide unnecessary explanations or long answers.
- If the question is general (e.g., greetings), respond briefly and politely.
- If the answer is not found in the context, you may use general academic knowledge ONLY if it does not involve EWU-specific rules, credits, fees, policies, or regulations.
- Never invent or assume EWU policies, courses, credits, or academic rules.
- If EWU-specific information is not available, clearly state that it is not found in the official documents.
- Maintain an academic, student-friendly, and professional tone."""
        
        self.custom_prompt_template = """{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer concisely and accurately. Prefer EWU document information.Limit the answer to what is necessary to address the question."""
    
    def get_template(self):
        return PromptTemplate(
            template=self.custom_prompt_template.format(system_prompt=self.system_prompt),
            input_variables=["context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="vectorstore/db_faiss", model="gpt-4o-mini"):
        self.db_path = db_path
        self.model = model
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
            # Create a simple retrieval chain using Runnable
            retriever = self.vector_manager.db.as_retriever(search_kwargs={'k': 3})
            prompt = self.prompt_manager.get_template()
            
            # Build the chain: retriever -> prompt -> llm
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm_manager.llm
                | StrOutputParser()
            )
            self.retriever = retriever  # Store retriever for getting source docs
            print("✓ QA chain created")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def query(self, question):
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            # Get the answer from the chain
            answer = self.qa_chain.invoke(question)
            # Get source documents from the retriever
            sources = self.retriever.invoke(question)
            return {
                'answer': answer,
                'sources': sources
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def format_output(self, result):
        output = "\n" + "=" * 60
        output += "\n📝 ANSWER:\n"
        output += result['answer']
        output += "\n\n" + "-" * 60
        output += "\n📄 SOURCES:\n"
        for i, doc in enumerate(result['sources'], 1):
            output += f"\n{i}. {doc.metadata.get('source', 'Unknown')}\n"
            output += f"   {doc.page_content[:150]}...\n"
        output += "=" * 60 + "\n"
        return output
    
    def interactive_session(self):
        self.initialize()
        print("Type 'quit' to exit\n")
        while True:
            try:
                question = input("❓ Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                if not question:
                    continue
                result = self.query(question)
                print(self.format_output(result))
            except KeyboardInterrupt:
                print("\n👋 Ended!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    assistant = AcademicAssistant()
    assistant.interactive_session()