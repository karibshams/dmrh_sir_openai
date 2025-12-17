from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
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
    def __init__(self, db_path="vectorstore1/db_faiss", embedding_model=None):
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
                temperature=0.2,
                max_tokens=1500
            )
            print(f"✓ LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        self.system_prompt = """You are an AI-powered Academic Assistant for East West University (EWU).

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

1. PRIMARY PRIORITY - EWU DOCUMENTS FIRST:
   - Always search EWU documents first for answers
   - If information is found in documents, use ONLY that information
   - Mark answers with [FROM EWU DOCUMENTS]
   
2. SECONDARY - GENERAL KNOWLEDGE (Only if not in EWU docs):
   - If EWU document context is empty or doesn't contain answer, then use general knowledge
   - Mark these answers with [FROM GENERAL KNOWLEDGE]
   - Be honest about the source

3. ANSWER STRUCTURE:
   - For EWU-specific questions: Always cite from documents
   - Show exact values (fees, credits, prerequisites) from documents ONLY
   - Never fabricate or estimate EWU-specific data
   - If unsure, say "This specific information is not available in EWU documents"

4. CONFIDENCE SCORING:
   - [HIGH CONFIDENCE] - Found directly in documents
   - [MEDIUM CONFIDENCE] - Inferred from documents
   - [LOW CONFIDENCE] - Using general knowledge only

5. QUESTION CLASSIFICATION:
   - GENERAL (definitions, concepts): Answer with general knowledge + mark as such
   - EWU-SPECIFIC (courses, fees, policies, faculty): MUST come from documents
   - HYBRID (uses both): Clearly separate which part is from where

6. ERROR HANDLING:
   - Never make up fees, credits, or department info
   - Never hallucinate faculty names or positions
   - If information missing: Say "This is not available in current EWU documents"

7. CONTEXT USAGE:
   - Use chat history to understand context
   - Provide follow-up answers based on conversation flow
   - Remember previous answers to maintain consistency

Be concise, helpful, student-friendly, and ALWAYS honest about information sources."""
        
        self.custom_prompt_template = """System: {system_prompt}

Chat History:
{{chat_history}}

EWU Document Context (if available):
{{context}}

Student Question:
{{question}}

Response Guidelines:
- If context is provided and answers question: Use ONLY context, mark as [FROM EWU DOCUMENTS]
- If context is empty or irrelevant: Use general knowledge, mark as [FROM GENERAL KNOWLEDGE]
- Show confidence level: [HIGH/MEDIUM/LOW CONFIDENCE]
- Never mix sources without clarity
- For EWU-specific data: Be exact or say "not available in documents"

Answer:"""
    
    def get_template(self):
        template_str = self.custom_prompt_template.format(system_prompt=self.system_prompt)
        return PromptTemplate(
            template=template_str,
            input_variables=["chat_history", "context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="vectorstore1/db_faiss", model="gpt-4o-mini"):
        self.db_path = db_path
        self.model = model
        self.embedding_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager(db_path=db_path)
        self.llm_manager = LLMManager(model=model)
        self.prompt_manager = PromptManager()
        
        self.chat_history = []
        self.qa_chain = None
        self.retriever = None
    
    def initialize(self):
        try:
            print("\n" + "=" * 70)
            print("🔧 Initializing Enhanced Assistant with Hybrid Knowledge")
            print("=" * 70 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\n✅ Ready with hybrid knowledge system!")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise
    
    def normalize_query(self, question):
        """
        Normalize and expand query for better retrieval.
        Fixes typos and adds context.
        """
        normalized = question.lower().strip()
        
        # Fix common typos
        replacements = {
            "facullty": "faculty",
            "coursse": "course",
            "cse": "computer science engineering",
            "eee": "electrical electronics engineering",
            "bba": "business administration",
            "feee": "fee",
            "creditt": "credit",
            "perquisite": "prerequisite",
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Add EWU context if not present
        if "east west" not in normalized and "ewu" not in normalized:
            normalized = f"East West University {normalized}"
        
        return normalized
    
    def create_qa_chain(self):
        try:
            print("⛓️  Creating QA chain with enhanced retrieval...")
            
            # Increased k for better recall of relevant documents
            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 25}
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                """Format retrieved documents with metadata"""
                formatted = []
                for doc in docs:
                    source = doc.metadata.get('source_file', 'Unknown')
                    content_type = doc.metadata.get('content_type', 'text')
                    formatted.append(
                        f"[{content_type.upper()} - {source}]\n{doc.page_content}"
                    )
                
                return "\n\n---\n\n".join(formatted) if formatted else "[NO DOCUMENTS FOUND]"
            
            def get_chat_history(_):
                """Get recent chat history for context"""
                history_text = "\n".join([
                    f"{msg['role'].title()}: {msg['content'][:200]}..."
                    if len(msg['content']) > 200
                    else f"{msg['role'].title()}: {msg['content']}"
                    for msg in self.chat_history[-8:]
                ])
                return history_text if history_text else "[NO PREVIOUS MESSAGES]"
            
            # Build enhanced chain
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
            
            print("✓ QA chain created (k=25, enhanced retrieval, hybrid knowledge)")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def query(self, question):
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # Normalize query
            normalized_question = self.normalize_query(question)
            
            # Get answer from chain
            answer = self.qa_chain.invoke(normalized_question)
            
            # Save to chat history
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