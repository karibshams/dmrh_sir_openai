from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
from dotenv import load_dotenv
import re
from datetime import datetime

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
1. For GENERAL questions (definitions, concepts, explanations): Answer directly and simply without requiring documents.
2. For EWU-SPECIFIC questions (courses, faculty, fees, policies): Prioritize document context.
3. Use chat history to understand context and provide follow-up answers.
4. If EWU-specific info not in documents, say: "This is not available in EWU documents."
5. Be concise, helpful, and student-friendly.
6. Remember previous answers in this conversation."""
        
        self.custom_prompt_template = """System: {system_prompt}

Chat History:
{{chat_history}}

EWU Document Context:
{{context}}

Student Question:
{{question}}

Answer based on context and chat history. For general questions, answer directly."""
    
    def get_template(self):
        template_str = self.custom_prompt_template.format(system_prompt=self.system_prompt)
        return PromptTemplate(
            template=template_str,
            input_variables=["chat_history", "context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="vectorstore/db_faiss", model="gpt-4o-mini"):
        self.db_path = db_path
        self.model = model
        self.embedding_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager(db_path=db_path)
        self.llm_manager = LLMManager(model=model)
        self.prompt_manager = PromptManager()
        
        # FIX 1: Simple chat history list
        self.chat_history = []
        
        self.qa_chain = None
        self.retriever = None
    
    def initialize(self):
        try:
            print("\n" + "=" * 60)
            print("🔧 Initializing Assistant with Chat Memory")
            print("=" * 60 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\n✅ Ready with conversation memory!")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise
    
    def create_qa_chain(self):
        try:
            print("⛓️  Creating QA chain with memory...")
            
            # FIX 3: Increase k to 20 for better recall
            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 20}
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            def get_chat_history(_):
                """FIX 1: Get chat history from list"""
                history_text = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}"
                    for msg in self.chat_history[-6:]
                ])
                return history_text if history_text else "No previous messages"
            
            # FIX 1: Build chain with memory
            self.qa_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "chat_history": RunnableLambda(get_chat_history)
                }
                | prompt
                | self.llm_manager.llm
                | StrOutputParser()
            )
            print("✓ QA chain created (k=20, with memory)")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def query(self, question):
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # FIX 4: Normalize query (fix spelling, expand)
            normalized_question = question.lower()
            replacements = {
                "facullty": "faculty",
                "coursse": "course",
                "cse": "computer science engineering",
                "eee": "electrical electronics engineering",
                "bba": "business administration",
            }
            for old, new in replacements.items():
                normalized_question = normalized_question.replace(old, new)
            normalized_question = f"East West University {normalized_question}"
            
            # Get answer from chain
            answer = self.qa_chain.invoke(normalized_question)
            
            # FIX 1: Save to chat history
            self.chat_history.append({"role": "student", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {'answer': answer}
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def clear_memory(self):
        """Clear conversation history"""
        self.chat_history = []
        print("✓ Chat history cleared")

if __name__ == "__main__":
    assistant = AcademicAssistant()
    assistant.initialize()
    print("Type 'quit' to exit | Type 'clear' to reset memory\n")
    while True:
        try:
            question = input("❓ Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if question.lower() == 'clear':
                assistant.clear_memory()
                continue
            if not question:
                continue
            result = assistant.query(question)
            print(f"\n📝 Answer:\n{result['answer']}\n")
        except KeyboardInterrupt:
            print("\n👋 Ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")