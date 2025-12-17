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
                temperature=0.05,  # VERY LOW for strict instruction following
                max_tokens=3000
            )
            print(f"✓ LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        # CRITICAL SYSTEM PROMPT - MUST BE FOLLOWED EXACTLY
        self.system_prompt = """You are an AI Academic Assistant for East West University (EWU).

ABSOLUTE RULES - FOLLOW 100%:

RULE 1: DOCUMENT CONTEXT PRIORITY
- First, ALWAYS check if the provided document context answers the question
- If information is in the provided context, USE IT and cite: [FROM EWU DOCUMENTS]
- ONLY use general knowledge if document context is EMPTY or doesn't contain the answer

RULE 2: MANDATORY SOURCE TAGGING (MUST BE IN EVERY RESPONSE)
Format MUST be exactly:
[SOURCE: EWU_DOCUMENTS] [CONFIDENCE: HIGH]
OR
[SOURCE: GENERAL_KNOWLEDGE] [CONFIDENCE: MEDIUM]
OR
[SOURCE: GENERAL_KNOWLEDGE] [CONFIDENCE: LOW]

RULE 3: WHEN TO USE WHICH SOURCE
- Document context has the answer → [SOURCE: EWU_DOCUMENTS] [CONFIDENCE: HIGH]
- Document context has partial info → [SOURCE: EWU_DOCUMENTS] [CONFIDENCE: MEDIUM]
- Document context is empty/irrelevant → [SOURCE: GENERAL_KNOWLEDGE] [CONFIDENCE: MEDIUM/LOW]

RULE 4: FACULTY SEARCH (CRITICAL)
- Faculty lists are in document context
- When asked about ANY faculty member, check the faculty list in context
- If name appears in faculty list → [SOURCE: EWU_DOCUMENTS] + Include their position
- NEVER say "not available" if faculty list is in the context provided

RULE 5: SPECIFIC INFORMATION (EXACT VALUES)
- Fees, credits, course codes → MUST come from documents
- Faculty names and positions → MUST come from documents  
- Course details → MUST come from documents
- NEVER estimate or guess these values
- If not in documents, say: "This specific information is not available in provided EWU documents"

RULE 6: GENERAL KNOWLEDGE (ALLOWED ONLY WHEN DOCUMENTS DON'T HAVE IT)
- Definitions and concepts → Can use general knowledge
- Historical context → Can use general knowledge
- If using general knowledge, ALWAYS mark it and explain why documents didn't have it

RULE 7: ANSWER STRUCTURE
[SOURCE TAG] [CONFIDENCE TAG]

[Your answer with proper citations]

RULE 8: NO CONTRADICTIONS
- If you already told user something from documents, keep it consistent
- Use chat history to maintain consistency
- Don't change answers based on different phrasings of same question"""

        self.custom_prompt_template = """System Instructions:
{system_prompt}

═══════════════════════════════════════════════════════════════

DOCUMENT CONTEXT PROVIDED (Search this FIRST):
{{context}}

CHAT HISTORY (for consistency):
{{chat_history}}

STUDENT QUESTION:
{{question}}

═══════════════════════════════════════════════════════════════

TASK: Answer the question following the ABSOLUTE RULES above.

Step 1: Check if DOCUMENT CONTEXT contains information about the question
Step 2: If YES → Use it with [SOURCE: EWU_DOCUMENTS] tag
Step 3: If NO → Use general knowledge with [SOURCE: GENERAL_KNOWLEDGE] tag
Step 4: Include proper [CONFIDENCE: ...] tag
Step 5: Answer with complete information

RESPONSE (MUST include source and confidence tags):"""

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
        
        self.chat_history = []
        self.qa_chain = None
        self.retriever = None
    
    def initialize(self):
        try:
            print("\n" + "=" * 80)
            print("🔧 Initializing EWU Academic Assistant - DOCUMENT FIRST APPROACH")
            print("=" * 80 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\n✅ Assistant Ready - Using Documents First + General Knowledge Hybrid")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise
    
    def normalize_query(self, question):
        """
        Normalize question for better retrieval.
        Expand search terms without changing meaning.
        """
        normalized = question.lower().strip()
        
        # Typo fixes
        typo_fixes = {
            "facullty": "faculty",
            "coursse": "course",
            "cse": "computer science engineering",
            "eee": "electrical electronics engineering",
            "bba": "business administration",
            "feee": "fee",
            "creditt": "credit",
            "prerequisite": "prerequisite",
            "proffesor": "professor",
            "dept": "department",
        }
        
        for old, new in typo_fixes.items():
            normalized = normalized.replace(old, new)
        
        # Add context for better retrieval
        if any(word in normalized for word in ["who", "faculty", "professor", "lecturer", "instructor"]):
            if "faculty" not in normalized:
                normalized += " faculty members"
            if "department" not in normalized:
                normalized += " department"
        
        # Add university context
        if "east west" not in normalized and "ewu" not in normalized:
            normalized = f"east west university {normalized}"
        
        return normalized
    
    def create_qa_chain(self):
        try:
            print("⛓️  Creating QA chain with aggressive retrieval...")
            
            # LARGE k for comprehensive results
            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 60}  # Get 60 chunks
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                """Format documents clearly for LLM"""
                if not docs or len(docs) == 0:
                    return "[NO DOCUMENTS PROVIDED - USE GENERAL KNOWLEDGE]"
                
                formatted_parts = []
                doc_count = 0
                
                for doc in docs:
                    doc_count += 1
                    source = doc.metadata.get('source_file', 'EWU Document')
                    content_type = doc.metadata.get('content_type', 'text')
                    
                    # Highlight faculty information
                    if any(keyword in doc.page_content.lower() for keyword in ['faculty', 'professor', 'lecturer', 'department of']):
                        formatted_parts.append(f"\n⭐ FACULTY/DEPARTMENT INFO (from {source}):\n{doc.page_content}\n")
                    else:
                        formatted_parts.append(f"\n📄 {content_type.upper()} (from {source}):\n{doc.page_content}\n")
                
                result = f"[TOTAL DOCUMENTS PROVIDED: {doc_count} chunks]\n" + "".join(formatted_parts)
                return result
            
            def get_chat_history(_):
                """Get full chat history"""
                if not self.chat_history:
                    return "[NO PREVIOUS MESSAGES]"
                
                history_lines = []
                for msg in self.chat_history[-10:]:  # Last 10 messages
                    role = msg['role'].upper()
                    content = msg['content']
                    if len(content) > 150:
                        content = content[:150] + "..."
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
            print(f"  - Retrieval k=60 (aggressive)")
            print(f"  - Temperature=0.05 (strict instruction following)")
            print(f"  - Max tokens=3000 (detailed answers)")
            return self.qa_chain
        except Exception as e:
            print(f"❌ Error creating chain: {e}")
            raise
    
    def query(self, question):
        """Query the system"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # Normalize the question
            normalized = self.normalize_query(question)
            print(f"🔍 Processing: {normalized[:100]}...")
            
            # Invoke chain
            answer = self.qa_chain.invoke(normalized)
            
            # Store in history (BEFORE returning)
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