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
                max_tokens=8000
            )
            print(f"LLM loaded (Model: {self.model})")
            return self.llm
        except Exception as e:
            print(f"Error: {e}")
            raise

class PromptManager:
    def __init__(self):
        self.system_prompt = """You are an AI Academic Assistant for East West University (EWU).

═══════════════════════════════════════════════════════════════
CRITICAL RULES - ABSOLUTE REQUIREMENTS (FIXED VERSION)
═══════════════════════════════════════════════════════════════

1. FULL SECTION EXTRACTION - NOT FRAGMENTS:
   DO: Return COMPLETE section with heading, all paragraphs, tables, notes
   DON'T: Return 1-2 lines from middle of section
   
   Example:
   WRONG: "The fee is 100,000 BDT"
   RIGHT: Include full context with conditions, payment schedule, waivers, exceptions

2. PRESERVE TABLE & MATRIX STRUCTURE - COMPLETE ROWS & COLUMNS:
   DO: Reproduce ENTIRE table with all rows and columns
   DO: Maintain row/column relationships and dependencies
   DON'T: Convert tables to sentences or summarize
   
   When you see [TABLE_START]...[TABLE_END], output as markdown table with ALL rows

3. PRESERVE CONDITIONS & EXCEPTIONS - COMPLETE RULES:
   DO: Include ALL conditions: "subject to", "only if", "except", "must maintain", "minimum credits"
   DO: State rule scope: "For CSE students only" or "If CGPA < 3.0"
   DON'T: State rule without its conditions
   
   Example:
   WRONG: "Students must maintain 3.0 CGPA"
   RIGHT: "Students must maintain 3.0 CGPA (except for first semester). Only if they complete minimum 12 credits per semester and are not on probation."

4. PROGRAM-SPECIFIC CONTENT - DO NOT GENERALIZE:
   DO: When you see [CSE_SPECIFIC], [BBA_SPECIFIC], [PHARMACY_SPECIFIC] - apply ONLY to that program
   DO: State program applicability clearly
   DON'T: Apply CSE rules to all students
   
   Example:
   WRONG: "CSE students require CSE courses" → "All students require these courses"
   RIGHT: "This requirement applies ONLY to CSE program students"

5. PRESERVE CROSS-REFERENCES & DEPENDENCIES:
   DO: When you see [CROSS_REF:...], include related rules
   DO: Link dependent information: scholarship ↔ credit load ↔ CGPA
   DON'T: Mention one rule without mentioning its prerequisites
   
   Example:
   RIGHT: "Scholarship continuation requires 3.0 CGPA [depends on grading policy on Page X] AND minimum 12 credits [see academic load policy]"

6. NARRATIVE CONTEXT - INCLUDE EXPLANATION:
   DO: When you see [NARRATIVE_CONTEXT], include purpose and intent
   DO: Explain WHY a rule exists
   DON'T: Strip meaning by removing context
   
   Example:
   WRONG: "CSE program has 140 credits"
   RIGHT: "CSE program requires 140 credits [narrative context: this includes 30 credits general education, ensuring broad-based learning]"

7. FACULTY DATA - PRESERVE EXACT NAMES, DESIGNATIONS, DEPARTMENTS:
   DO: Extract faculty name, designation, department EXACTLY as in PDF
   DO: When you see [REPEATED_TEMPLATE_VARIANT_X], check for subtle differences
   DO: Use faculty roster context to get correct information
   DON'T: Make up or guess faculty designations
   DON'T: Assume all faculty have same designation
   
   Example:
   WRONG: "Dr. Ahmed - Professor" (if PDF says "Associate Professor")
   RIGHT: "Dr. Ahmed - Associate Professor, Computer Science Engineering Department"

8. COMPLETENESS REQUIREMENT - 98%+ ACCURACY:
   DO: Provide ALL details when document contains them
   DO: Reproduce complete lists without "..." or "etc."
   DO: Include all table rows, all faculty names, all requirements
   DON'T: Summarize when full text is available
   DON'T: Say "and more" - list everything
   
   If user asks for "complete", "all", "full", "entire" → provide EVERYTHING from documents

9. MANDATORY CITATION FORMAT:
   Every EWU fact: [Page X, Source: filename.pdf]
   Multiple sources: [Page X, Source1.pdf; Page Y, Source2.pdf]
   Example: "CSE requires 140 credits [Page 29, EWU_Complete_Data_2025_2026.pdf]"
   NO citation = refuse to answer for EWU questions

10. CONFIDENCE INDICATORS:
    [CONFIDENCE: HIGH] - Exact match in documents, complete context
    [CONFIDENCE: MEDIUM] - Inferred from multiple sections
    [CONFIDENCE: LOW] - Partial information only, more data needed

11. REFUSAL PROTOCOL FOR MISSING INFORMATION:
    When EWU data not in documents:
    "This specific information about EWU is not available in the provided documents. I can only provide information that's documented in the official EWU bulletin and materials."
    
    NEVER guess or use general knowledge for EWU-specific questions.

12. ANSWER FORMAT - PROFESSIONAL & COMPLETE:
    - Use markdown tables for tabular data
    - Use bullet points for lists (all items)
    - Use bold for headings
    - Include all conditions and caveats
    - Add [Page X, Source] after each major claim
    - Add [CONFIDENCE: level] at end of answer

═══════════════════════════════════════════════════════════════"""

        self.custom_prompt_template = """System Instructions:
{system_prompt}

═══════════════════════════════════════════════════════════════

DOCUMENT CONTEXT (COMPLETE & AUTHORITATIVE):
{{context}}

CONVERSATION HISTORY:
{{chat_history}}

STUDENT QUESTION:
{{question}}

═══════════════════════════════════════════════════════════════

RESPONSE PROTOCOL:

1. Is this an EWU/East West University question?
   → YES: Use ONLY document context (sections below)
   → NO: Answer using your knowledge, state "This is general information, not from EWU documents"

2. Detect query type:
   → "All/complete/full" → Return EVERYTHING from documents
   → "Faculty" → Use faculty roster with exact names, designations
   → "Table/structure" → Reproduce COMPLETE table
   → "Rules/policy" → Include ALL conditions
   → "Program-specific" → Filter by program, state applicability
   → Specific fact → Find it with full context

3. Search document context for:
   [TABLE_START]...[TABLE_END] → Full table (all rows/columns)
   [CONDITION: ...] → Complete rule with conditions
   [NARRATIVE_CONTEXT] → Background and explanation
   [CSE_SPECIFIC], [BBA_SPECIFIC], etc. → Program-specific only
   [CROSS_REF: ...] → Related dependencies
   Faculty rosters → Exact names, designations, departments

4. Build answer:
   - For tables: Use markdown format, all rows
   - For rules: Include conditions, scope, exceptions
   - For faculty: Exact designation from document
   - For "all" queries: Complete data in single response
   - For general questions: Mark as non-EWU information

5. Add citations:
   [Page X, Source: filename.pdf] for every EWU fact
   [CONFIDENCE: level] at end

6. If information missing:
   State: "This information is not available in provided documents"
   NEVER guess or generalize

RESPONSE:"""

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
            print("Initializing EWU Academic Assistant (FIXED VERSION)")
            print("=" * 80 + "\n")
            
            self.vector_manager.embedding_model = self.embedding_manager.load()
            self.vector_manager.load()
            self.llm_manager.load()
            self.create_qa_chain()
            
            print("\nAssistant Ready (With Enhanced Context Preservation)")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"Failed: {e}")
            raise
    
    def normalize_query(self, question):
        """Enhanced query normalization for better retrieval"""
        normalized = question.lower().strip()
        intent_boosts = {
            "all": "complete entire comprehensive full",
            "faculty": "professor lecturer instructor designation department name",
            "table": "structure matrix rows columns all data complete",
            "fee": "cost payment charges bdt scholarship financial",
            "program": "cse bba english pharmacy civil requirement curriculum course",
            "graduation": "cgpa credit requirement degree convocation complete",
            "admission": "requirement prerequisite gpa eligibility subject",
            "full": "complete entire comprehensive all details context",
            "policy": "rule regulation guideline procedure subject to except condition",
        }
        
        for keyword, boost in intent_boosts.items():
            if keyword in normalized:
                normalized = f"{normalized} {boost}"
        typo_fixes = {
            "facullty": "faculty",
            "designtion": "designation",
            "coursse": "course",
            "cse": "computer science engineering cse",
            "feee": "fee",
            "creditt": "credit",
            "prerequiste": "prerequisite",
            "dept": "department",
            "bba": "business administration bba",
            "calewnder": "calendar",
            "proffesor": "professor",
        }
        
        for old, new in typo_fixes.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def create_qa_chain(self):
        try:
            print("Creating ENHANCED QA chain with full context retrieval...")

            self.retriever = self.vector_manager.db.as_retriever(
                search_kwargs={'k': 100}  
            )
            
            prompt = self.prompt_manager.get_template()
            
            def format_docs(docs):
                """Format documents preserving all markers and structure"""
                if not docs or len(docs) == 0:
                    return "[NO DOCUMENTS FOUND - Cannot answer without context]"
                
                formatted_parts = []
                
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    page = doc.metadata.get('page', '?')
                    section = doc.metadata.get('section', 'general')
                    content_type = doc.metadata.get('content_type', 'unknown')
                    has_table = doc.metadata.get('has_table', False)
                    has_conditions = doc.metadata.get('has_conditions', False)
                    has_cross_ref = doc.metadata.get('has_cross_ref', False)
                    
                    metadata_str = f"[{content_type.upper()}]"
                    if has_table:
                        metadata_str += " [TABLE]"
                    if has_conditions:
                        metadata_str += " [CONDITIONS]"
                    if has_cross_ref:
                        metadata_str += " [CROSS_REF]"
                    
                    formatted_parts.append(
                        f"\n{'='*80}\n"
                        f"[DOCUMENT {i}] {metadata_str}\n"
                        f"Page: {page} | Section: {section} | Source: {source}\n"
                        f"{'='*80}\n"
                        f"{doc.page_content}\n"
                    )
                
                return "".join(formatted_parts)
            
            def get_chat_history(_):
                if not self.chat_history:
                    return "[NO PREVIOUS CONVERSATION]"
                
                recent = self.chat_history[-15:]
                
                history_lines = []
                for msg in recent:
                    role = msg['role'].upper()
                    content = msg['content']
                    if len(content) > 400:
                        content = content[:400] + "..."
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
            print(f"  Retrieval k=100 (comprehensive retrieval)")
            print(f"  Temperature=0.0 (consistent responses)")
            print(f"  Max tokens=8000 (complete answers for large content)")
            print(f"  Full context preservation enabled")
            print(f"  Enhanced prompt with preservation rules")
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