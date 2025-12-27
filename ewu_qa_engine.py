from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class EWUQAEngine:
    def __init__(self, db_path="ewu_db/faiss", model="gpt-4o-mini"):
        self.db_path = db_path
        self.model = model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.history = []
    
    def init(self):
        """Initialize QA engine"""
        print("\n" + "="*60)
        print("Initializing EWU QA Engine")
        print("="*60 + "\n")
        
        if not os.path.exists(self.db_path):
            print(f"❌ Vector store not found: {self.db_path}")
            return False
        
        # Load vector store
        self.vector_store = FAISS.load_local(
            self.db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded")
        
        # Setup retriever (k=80 optimal balance)
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 80})
        
        # Load LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return False
        
        llm = ChatOpenAI(api_key=api_key, model=self.model, temperature=0.0, max_tokens=5000)
        print("✅ LLM loaded")
        
        # Create prompt
        system = """You are EWU Academic Assistant. RULES:
1. Answer ONLY from document context
2. Include ALL details: complete tables, lists, conditions
3. Every EWU fact: [Page X, filename.pdf]
4. If [TABLE]...[/TABLE] → output complete markdown table
5. If [RULE: ...]  → include all conditions
6. Missing info → say "Not in documents"
7. Confidence level: [CONFIDENCE: HIGH/MEDIUM/LOW]

Context:
{context}

Chat History:
{history}

Question: {question}

Answer with COMPLETE information from documents:"""
        
        prompt = PromptTemplate(
            template=system,
            input_variables=["context", "history", "question"]
        )
        
        def format_docs(docs):
            if not docs:
                return "[NO CONTEXT]"
            return "\n\n---\n\n".join([
                f"[{doc.metadata.get('source_file')}:P{doc.metadata.get('page')}] {doc.page_content}"
                for doc in docs
            ])
        
        def get_history(_):
            if not self.history:
                return "[NO HISTORY]"
            recent = self.history[-10:]
            return "\n".join([f"{m['role'].upper()}: {m['content'][:200]}" for m in recent])
        
        # Build chain
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "history": RunnableLambda(get_history),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ Chain ready")
        print("="*60 + "\n")
        return True
    
    def query(self, question):
        """Ask question"""
        if not self.chain:
            raise ValueError("Engine not initialized")
        
        print(f"🤔 Processing: {question[:60]}...")
        answer = self.chain.invoke(question)
        
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        return answer
    
    def reset(self):
        """Clear history"""
        self.history = []
        print("✅ History cleared")

if __name__ == "__main__":
    qa = EWUQAEngine()
    if not qa.init():
        exit(1)
    
    print("Type 'quit' to exit | 'clear' to reset\n")
    
    while True:
        try:
            q = input("❓ Question: ").strip()
            if q.lower() in ['quit', 'q']:
                print("👋 Goodbye!")
                break
            if q.lower() == 'clear':
                qa.reset()
                continue
            if not q:
                continue
            
            answer = qa.query(q)
            print(f"\n✅ Answer:\n{answer}\n")
            print("-"*60 + "\n")
        except KeyboardInterrupt:
            print("\n👋 Ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")