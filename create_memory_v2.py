"""
FIXED enhanced_create_memory_for_llm.py
- STRICT context-only generation (no general knowledge)
- Mandatory citations with page numbers
- Explicit refusal when data not found
- Faithfulness validation
- Limited answer length
- Last 2-3 turns only in history
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
    def __init__(self, db_path="vectorstore2/db_faiss", embedding_model=None):
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
                temperature=0.01,  # FIXED: Even lower for strict compliance
                max_tokens=1500    # FIXED: Reduced from 3000
            )
            print(f"✓ LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        # FIXED: STRICT GROUNDING PROMPT - Context only, no general knowledge
        self.system_prompt = """You are an AI Academic Assistant for East West University (EWU).

⚠️ STRICT RULES - FOLLOW EXACTLY (No Exceptions):

RULE 1: CONTEXT-ONLY GENERATION
- You MUST answer ONLY from the provided document context
- You are FORBIDDEN from using internet knowledge, general knowledge, or assumptions
- If information is NOT in the provided context, you MUST refuse

RULE 2: MANDATORY CITATIONS (EVERY CLAIM)
- EVERY fact must cite: [Page X, Source: filename.pdf]
- Example: "CSE 103 is Structured Programming [Page 5, EWU_Bulletin_2024.pdf]"
- No citations = refuse the answer

RULE 3: EXPLICIT REFUSAL
- If the information is not in provided documents, say EXACTLY:
  "❌ This information is not available in the provided EWU documents."
- Do NOT guess, assume, or use general knowledge
- Do NOT say "typically" or "usually"

RULE 4: TABLE PRESERVATION
- If context contains tables, reproduce them exactly
- Do NOT summarize tables
- Keep formatting: Course Code | Course Title | Credits

RULE 5: CONFIDENCE LEVELS
- ALWAYS include: [CONFIDENCE: HIGH] or [CONFIDENCE: LOW]
- HIGH = exact match in documents
- LOW = partial information or inference

RULE 6: ANSWER LENGTH
- Keep answers under 500 words
- Be concise and specific
- No verbose explanations

RULE 7: FAITHFULNESS CHECK
- Before answering, verify: "Is this EXACTLY in the context?"
- If YES → Answer with citation
- If NO → Refuse

RULE 8: NO GENERAL KNOWLEDGE
- ❌ FORBIDDEN: "In general...", "Usually...", "Typically..."
- ❌ FORBIDDEN: "Based on common practice..."
- ✅ REQUIRED: "According to EWU bulletin..."

RULE 9: CLARIFICATION REQUEST
- If question is ambiguous, ask: "Could you clarify which..."
- Do NOT assume what student means

RULE 10: CONVERSATION CONSISTENCY
- Remember previous answers in THIS conversation only
- Do NOT contradict yourself
- If same question asked twice, give identical answer"""

        self.custom_prompt_template = """System Instructions:
{system_prompt}

═══════════════════════════════════════════════════════════════

DOCUMENT CONTEXT (STRICT SOURCE OF TRUTH):
{{context}}

RECENT CHAT HISTORY (Last 2-3 turns only):
{{chat_history}}

STUDENT QUESTION:
{{question}}

═══════════════════════════════════════════════════════════════

TASK: Answer ONLY from the document context above.

Step 1: Search the CONTEXT for exact match to the question
Step 2: If found → Answer with [Page X, Source: Y] citation
Step 3: If NOT found → Say "❌ This information is not available in provided EWU documents"
Step 4: Include [CONFIDENCE: HIGH/LOW] tag
Step 5: Reproduce tables exactly (do NOT summarize)

MANDATORY: Every factual claim needs a citation. If you cannot cite it from the context, refuse to answer.

RESPONSE (MUST include citations and confidence):[context]"""

    def get_template(self):
        template_str = self.custom_prompt_template.format(system_prompt=self.system_prompt)
        return PromptTemplate(
            template=template_str,
            input_variables=["chat_history", "context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="vectorstore2/db_faiss", model="gpt-4o-mini"):
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
            print("\n" + "=" * 80)
            print("🔧 Initializing EWU Academic Assistant - FIXED VERSION")
            print("=" * 80 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\n✅ Assistant Ready - STRICT CONTEXT-ONLY MODE")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise
    
    def normalize_query(self, question):
        """Normalize question for better retrieval"""
        normalized = question.lower().strip()
        
        typo_fixes = {
            "facullty": "faculty",
            "coursse": "course",
            "cse": "computer science engineering",
            "feee": "fee",
            "creditt": "credit",
            "prerequisite": "prerequisite",
            "dept": "department",
        }
        
        for old, new in typo_fixes.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def create_qa_chain(self):
        try:
            print("⛓️  Creating QA chain with STRICT grounding...")
            
            # Moderate retrieval (not too aggressive)
            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 15}  # FIXED: Reduced from 60
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                """Format documents with citations"""
                if not docs or len(docs) == 0:
                    return "[NO DOCUMENTS FOUND FOR THIS QUERY]"
                
                formatted_parts = []
                
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    page = doc.metadata.get('page', '?')
                    section = doc.metadata.get('section', 'general')
                    
                    formatted_parts.append(
                        f"\n[DOCUMENT {i}] Page {page} | {source} | Section: {section}\n"
                        f"{doc.page_content}\n"
                        f"{'─' * 60}"
                    )
                
                return "".join(formatted_parts)
            
            def get_chat_history(_):
                """Get last 2-3 turns only"""
                if not self.chat_history:
                    return "[NO PREVIOUS MESSAGES]"
                
                # Keep only last 2-3 turns (4-6 messages)
                recent = self.chat_history[-6:]
                
                history_lines = []
                for msg in recent:
                    role = msg['role'].upper()
                    content = msg['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    history_lines.append(f"{role}: {content}")
                
                return "\n".join(history_lines)
            
            # Build the chain
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
            
            print("✓ QA chain created successfully")
            print(f"  - Retrieval k=15 (context-focused)")
            print(f"  - Temperature=0.01 (strict compliance)")
            print(f"  - Max tokens=1500 (concise answers)")
            print(f"  - History=Last 2-3 turns only")
            print(f"  - Citations MANDATORY")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error creating chain: {e}")
            raise
    
    def query(self, question):
        """Query the system with strict grounding"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            normalized = self.normalize_query(question)
            print(f"🔍 Processing: {normalized[:100]}...")
            
            answer = self.qa_chain.invoke(normalized)
            
            # Store in history
            self.chat_history.append({"role": "student", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {'answer': answer}
        
        except Exception as e:
            print(f"❌ Error in query: {e}")
            raise
    
    def clear_memory(self):
        """Clear chat history"""
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
            print("-" * 80 + "\n")
        except KeyboardInterrupt:
            print("\n👋 Ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")