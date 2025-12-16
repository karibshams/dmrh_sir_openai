import streamlit as st
import os
from dotenv import load_dotenv
from create_memory_for_llm import AcademicAssistant, EmbeddingManager, VectorStoreManager, LLMManager, PromptManager

load_dotenv()

class EWUAcademicUI:
    def __init__(self):
        self.db_path = "vectorstore/db_faiss"
        self.model = "gpt-4o-mini"
        self.assistant = None
        
        self.ewu_primary = "#003d7a"
        self.ewu_secondary = "#0066cc"
        self.ewu_accent = "#ff6600"
        self.ewu_light = "#f0f4f8"
    
    def configure_page(self):
        st.set_page_config(
            page_title="EWU Academic Assistant",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        self.apply_theme()
    
    def apply_theme(self):
        st.markdown(f"""
        <style>
        * {{
            margin: 0;
            padding: 0;
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, {self.ewu_light} 0%, #e8eef5 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        [data-testid="stChatMessage"] {{
            padding: 16px;
            border-radius: 12px;
            margin: 12px 0;
            backdrop-filter: blur(10px);
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
            color: #1a1a1a;
            line-height: 1.6;
        }}
        .user-message {{
            background: linear-gradient(135deg, {self.ewu_secondary} 0%, {self.ewu_primary} 100%);
            color: white;
            border-radius: 18px;
            margin-left: 20px;
            margin-right: 0;
            padding: 14px 18px;
            max-width: 85%;
            word-wrap: break-word;
        }}
        .assistant-message {{
            background: white;
            color: #333;
            border-left: 4px solid {self.ewu_accent};
            border-radius: 12px;
            margin-left: 0;
            margin-right: 20px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,61,122,0.08);
        }}
        .header-container {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, {self.ewu_primary} 0%, {self.ewu_secondary} 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,61,122,0.2);
        }}
        .header-title {{
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        .header-subtitle {{
            font-size: 1.1em;
            opacity: 0.95;
            font-weight: 300;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 20px;
        }}
        [data-testid="stChatInputContainer"] {{
            border-top: 1px solid rgba(0,61,122,0.1);
            padding: 20px;
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(10px);
        }}
        [data-testid="stChatInputContainer"] input {{
            border-radius: 24px;
            border: 2px solid {self.ewu_secondary};
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
        }}
        [data-testid="stChatInputContainer"] input:focus {{
            border-color: {self.ewu_accent};
            box-shadow: 0 0 0 3px rgba(255,102,0,0.1);
        }}
        .error-box {{
            background-color: #ff6b6b15;
            border-left: 4px solid #ff6b6b;
            padding: 16px;
            border-radius: 8px;
            margin: 20px 0;
            color: #d63031;
        }}
        .source-doc {{
            background: white;
            border-left: 3px solid {self.ewu_accent};
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            font-size: 0.95em;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def check_prerequisites(self):
        if not os.path.exists(self.db_path):
            st.markdown("""
            <div class="error-box">
            ⚠️ Vector Database Not Found<br>
            Run: python connect_memory_with_llm.py
            </div>
            """, unsafe_allow_html=True)
            return False
        if not os.getenv("OPENAI_API_KEY"):
            st.markdown("""
            <div class="error-box">
            ⚠️ OpenAI API Key Missing<br>
            Add OPENAI_API_KEY to .env file
            </div>
            """, unsafe_allow_html=True)
            return False
        return True
    
    @st.cache_resource
    def initialize_assistant(_self):
        try:
            assistant = AcademicAssistant(db_path=_self.db_path, model=_self.model)
            assistant.initialize()
            return assistant
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def render_header(self):
        st.markdown(f"""
        <div class="header-container">
            <div class="header-title">🎓 EWU Academic Assistant</div>
            <div class="header-subtitle">AI-Powered Q&A System | East West University</div>
        </div>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    
    def display_chat_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    def process_user_input(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching..."):
                    result = self.assistant.query(prompt)
                    answer = result['answer']
                    sources = result['sources']
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-doc">
                            <b>Document {i}:</b> {doc.metadata.get('source', 'Unknown')}<br>
                            {doc.page_content[:250]}...
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def run(self):
        self.configure_page()
        self.render_header()
        
        if not self.check_prerequisites():
            st.stop()
        
        self.assistant = self.initialize_assistant()
        if self.assistant is None:
            st.stop()
        
        self.initialize_session_state()
        self.display_chat_history()
        
        if prompt := st.chat_input("Ask about EWU programs, courses, or policies:"):
            self.process_user_input(prompt)

if __name__ == "__main__":
    app = EWUAcademicUI()
    app.run()