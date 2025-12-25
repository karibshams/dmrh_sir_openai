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
            print("Loading embeddings...")
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
            print("Embeddings loaded")
            return self.embedding_model
        except Exception as e:
            print(f"Error: {e}")
            raise

class VectorStoreManager:
    def __init__(self, db_path="final_vectorstore/db_faiss", embedding_model=None):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.db = None
    
    def load(self):
        try:
            print("Loading vector database...")
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Vector store not found at {self.db_path}")
            self.db = FAISS.load_local(
                self.db_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print("Database loaded")
            return self.db
        except Exception as e:
            print(f"Error: {e}")
            raise

class LLMManager:
    def __init__(self, model="gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.llm = None
    
    def load(self):
        try:
            print("Loading OpenAI LLM...")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.0,
                max_tokens=4000
            )
            print(f"LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        self.system_prompt = """You are an AI Academic Assistant for East West University (EWU).

CRITICAL RULES - ABSOLUTE REQUIREMENTS:

1. DOCUMENT AUTHORITY FOR EWU QUESTIONS:
   - ALL EWU/East West University questions MUST be answered from the provided documents ONLY
   - You are FORBIDDEN from using general knowledge for EWU-specific information
   - If information is NOT in documents, explicitly state: "This information is not available in the provided EWU documents"

2. MANDATORY CITATION SYSTEM:
   - EVERY fact about EWU must cite: [Page X, Source: filename.pdf]
   - Example: "The CSE program requires 140 credits [Page 29, EWU_Bulletin_2024.pdf]"
   - Multiple sources: [Page X, Source1.pdf; Page Y, Source2.pdf]
   - NO citation = refuse to answer

3. COMPLETENESS REQUIREMENT (95%+ TARGET):
   - If document contains a TABLE → reproduce the COMPLETE table with all rows and columns
   - If document has DETAILED explanation → provide FULL explanation, not summary
   - If document lists REQUIREMENTS → list ALL requirements completely
   - If document shows FLOWCHART → describe complete sequence
   - NEVER truncate, summarize, or abbreviate document content
   - Partial answers are UNACCEPTABLE when full details exist

4. CONTENT TYPE HANDLING:
   - Tables/Matrices: Preserve exact structure, format as markdown table
   - Lists: Return complete list with all items
   - Numeric data: Include all figures, statistics, and values
   - Cross-references: Maintain all prerequisite/dependency information
   - Policies/Rules: State complete rules with all conditions

5. ANSWER QUALITY STANDARDS:
   - Professional, polite, and well-structured responses
   - Clear formatting with headers, bullet points when appropriate
   - Complete information before summarizing
   - Accurate representation of document content
   - May reorganize for clarity but MUST include all details

6. GENERAL KNOWLEDGE HANDLING:
   - Non-EWU questions: Answer normally using your knowledge
   - General academic advice: Provide helpful guidance
   - Mixed questions: Clearly separate EWU facts (cited) from general advice

7. CONVERSATION MEMORY:
   - Remember user context within conversation
   - Maintain consistency with previous answers
   - Reference earlier discussion when relevant
   - Build on previous responses

8. EXPLICIT REFUSAL PROTOCOL:
   - When EWU information not in documents, say: "This specific information about EWU is not available in the provided documents. I can only provide information that's documented in the official EWU bulletin and materials."
   - Never guess or assume EWU-specific information
   - Never use phrases like "typically", "usually", "generally" for EWU facts

9. CONFIDENCE INDICATORS:
   - [CONFIDENCE: HIGH] - Exact match in documents
   - [CONFIDENCE: MEDIUM] - Inferred from related document sections
   - [CONFIDENCE: LOW] - Partial information only

10. TABLE PRESERVATION RULES:
    - Keep all headers and columns
    - Include all rows (no "..." or "etc.")
    - Maintain alignment and structure
    - Add totals if present in original
    - Use markdown table format for clarity

11. COMPLETENESS FOR LIST QUERIES:
    - When user asks for "all", "complete", "entire" lists (faculty, courses, etc.)
    - Retrieve and present THE COMPLETE list from documents
    - Do NOT provide partial lists across multiple responses
    - Aggregate all relevant data before responding
    - Remove duplicates and organize alphabetically or by rank
    
12. FILTERING AND SEARCH PRECISION:
    - When user asks for specific roles (e.g., "professors only")
    - Filter results by exact designation/title
    - Do NOT include other roles unless explicitly asked
    - Clarify filter criteria used in response"""

        self.custom_prompt_template = """System Instructions:
{system_prompt}

═══════════════════════════════════════════════════════════════

DOCUMENT CONTEXT (AUTHORITATIVE SOURCE FOR EWU):
{{context}}

CONVERSATION HISTORY:
{{chat_history}}

STUDENT QUESTION:
{{question}}

═══════════════════════════════════════════════════════════════

RESPONSE PROTOCOL:

Step 1: Identify if question is about EWU/East West University
Step 2: Detect query type (comprehensive list vs specific fact vs filtered search)
Step 3: If comprehensive list → Search ALL relevant document sections and aggregate complete data
Step 4: If filtered search → Apply exact filter criteria and clarify in response
Step 5: If information found → Provide COMPLETE answer with citations
Step 6: If information NOT found → Search related sections before refusing
Step 7: If NO (general question) → Answer using your knowledge
Step 8: Format response professionally with complete details

MANDATORY FOR EWU QUESTIONS:
- Cite every EWU fact: [Page X, Source: filename.pdf]
- Include ALL details from documents (95%+ completeness)
- Reproduce tables/lists completely
- For "all/complete" queries: present entire dataset in single response
- For filtered queries: apply exact filters and state filter used
- Add [CONFIDENCE: level] tag
- Never truncate or summarize when full details exist
- Never provide partial lists across multiple responses

RESPONSE:[context]"""

    def get_template(self):
        template_str = self.custom_prompt_template.format(system_prompt=self.system_prompt)
        return PromptTemplate(
            template=template_str,
            input_variables=["chat_history", "context", "question"]
        )

class AcademicAssistant:
    def __init__(self, db_path="final_vectorstore/db_faiss", model="gpt-4o-mini"):
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
            print("Initializing EWU Academic Assistant")
            print("=" * 80 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\nAssistant Ready")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"Failed: {e}")
            raise
    
    def normalize_query(self, question):
        normalized = question.lower().strip()
        
        typo_fixes = {
            "facullty": "faculty",
            "coursse": "course",
            "cse": "computer science engineering cse",
            "feee": "fee",
            "creditt": "credit",
            "prerequisite": "prerequisite",
            "dept": "department",
            "bba": "business administration bba",
            "eng": "engineering eng",
        }
        
        for old, new in typo_fixes.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def create_qa_chain(self):
        try:
            print("Creating QA chain...")
            
            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 40}
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                if not docs or len(docs) == 0:
                    return "[NO DOCUMENTS FOUND - Cannot answer EWU-specific questions without document context]"
                
                formatted_parts = []
                
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    page = doc.metadata.get('page', '?')
                    section = doc.metadata.get('section', 'general')
                    content_type = doc.metadata.get('content_type', 'unknown')
                    
                    formatted_parts.append(
                        f"\n[DOCUMENT {i}]\n"
                        f"Page: {page} | Source: {source} | Section: {section} | Type: {content_type}\n"
                        f"Content:\n{doc.page_content}\n"
                        f"{'─' * 80}"
                    )
                
                return "".join(formatted_parts)
            
            def get_chat_history(_):
                if not self.chat_history:
                    return "[NO PREVIOUS CONVERSATION]"
                
                recent = self.chat_history[-10:]
                
                history_lines = []
                for msg in recent:
                    role = msg['role'].upper()
                    content = msg['content']
                    if len(content) > 300:
                        content = content[:300] + "..."
                    history_lines.append(f"{role}: {content}")
                
                return "\n".join(history_lines)
            
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
            
            print("QA chain created successfully")
            print(f"  - Retrieval k=40 (enhanced for comprehensive queries)")
            print(f"  - Temperature=0.0")
            print(f"  - Max tokens=4000")
            print(f"  - Complete answer mode enabled")
            return self.qa_chain
        except Exception as e:
            print(f"Error creating chain: {e}")
            raise
    
    def query(self, question):
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            normalized = self.normalize_query(question)
            print(f"Processing: {normalized[:100]}...")
            
            answer = self.qa_chain.invoke(normalized)
            
            self.chat_history.append({"role": "student", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {'answer': answer}
        
        except Exception as e:
            print(f"Error in query: {e}")
            raise
    
    def clear_memory(self):
        self.chat_history = []
        print("Chat history cleared")

if __name__ == "__main__":
    assistant = AcademicAssistant()
    assistant.initialize()
    print("Type 'quit' to exit | Type 'clear' to reset memory\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            if question.lower() == 'clear':
                assistant.clear_memory()
                continue
            if not question:
                continue
            
            result = assistant.query(question)
            print(f"\nAnswer:\n{result['answer']}\n")
            print("-" * 80 + "\n")
        except KeyboardInterrupt:
            print("\nEnded!")
            break
        except Exception as e:
            print(f"Error: {e}\n")