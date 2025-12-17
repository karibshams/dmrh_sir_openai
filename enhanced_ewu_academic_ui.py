import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from create_memory_for_llm import AcademicAssistant

load_dotenv()

class EnhancedEWUAcademicUI:
    def __init__(self):
        self.db_path = "vectorstore/db_faiss"
        self.model = "gpt-4o-mini"
        self.assistant = None
        self.history_file = "chat_history.json"
        
        self.ewu_primary = "#003d7a"
        self.ewu_secondary = "#0066cc"
        self.ewu_accent = "#ff6600"
        self.ewu_light = "#f0f4f8"
        self.success_green = "#27ae60"
        self.warning_yellow = "#f39c12"
    
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
            box-sizing: border-box;
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, {self.ewu_light} 0%, #e8eef5 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
        [data-testid="stChatMessageContainer"] {{
            padding: 20px;
        }}
        [data-testid="stChatMessage"] {{
            padding: 0;
            margin: 8px 0;
        }}
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
            color: #1a1a1a;
            line-height: 1.7;
            font-size: 0.95em;
        }}
        .stChatMessage {{
            border-radius: 12px !important;
        }}
        .stChatMessage.user {{
            background: linear-gradient(135deg, {self.ewu_secondary} 0%, {self.ewu_primary} 100%) !important;
            color: white !important;
            border-radius: 18px !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            width: fit-content;
            max-width: 85%;
            padding: 12px 16px !important;
        }}
        .stChatMessage.assistant {{
            background: white !important;
            border-left: 5px solid {self.ewu_accent} !important;
            border-radius: 12px !important;
            margin-left: 0 !important;
            margin-right: auto !important;
            padding: 14px 16px !important;
            box-shadow: 0 2px 10px rgba(0,61,122,0.1) !important;
        }}
        .source-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            margin-right: 8px;
            margin-top: 8px;
        }}
        .source-ewu {{
            background-color: {self.success_green}20;
            color: {self.success_green};
            border: 1px solid {self.success_green};
        }}
        .source-general {{
            background-color: {self.warning_yellow}20;
            color: {self.warning_yellow};
            border: 1px solid {self.warning_yellow};
        }}
        .confidence-high {{
            color: {self.success_green};
            font-weight: 600;
        }}
        .confidence-medium {{
            color: {self.warning_yellow};
            font-weight: 600;
        }}
        .confidence-low {{
            color: #e74c3c;
            font-weight: 600;
        }}
        .header-container {{
            text-align: center;
            padding: 25px 20px;
            background: linear-gradient(135deg, {self.ewu_primary} 0%, {self.ewu_secondary} 100%);
            border-radius: 16px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,61,122,0.2);
        }}
        .header-title {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 5px;
            letter-spacing: -0.5px;
        }}
        .header-subtitle {{
            font-size: 1em;
            opacity: 0.9;
            font-weight: 300;
        }}
        .info-box {{
            background-color: {self.ewu_light}80;
            border-left: 4px solid {self.ewu_secondary};
            padding: 16px;
            border-radius: 8px;
            margin: 10px 0;
            color: #1a1a1a;
        }}
        .error-box {{
            background-color: #ff6b6b15;
            border-left: 4px solid #ff6b6b;
            padding: 16px;
            border-radius: 8px;
            margin: 20px 0;
            color: #d63031;
        }}
        [data-testid="stChatInputContainer"] {{
            border-top: 1px solid rgba(0,61,122,0.1);
            padding: 16px 20px;
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
        }}
        [data-testid="stChatInputContainer"] textarea {{
            border-radius: 24px !important;
            border: 2px solid {self.ewu_secondary} !important;
            padding: 12px 16px !important;
            font-size: 0.95em !important;
            transition: all 0.3s ease;
        }}
        [data-testid="stChatInputContainer"] textarea:focus {{
            border-color: {self.ewu_accent} !important;
            box-shadow: 0 0 0 3px rgba(255,102,0,0.1) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def load_chat_history(self):
        """Load chat history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_chat_history(self, messages):
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(messages, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def check_prerequisites(self):
        if not os.path.exists(self.db_path):
            st.markdown("""
            <div class="error-box">
            ⚠️ Vector Database Not Found<br>
            Run: python create_vector_store.py
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
    
    def extract_source_info(self, answer):
        """Extract source and confidence from answer"""
        source = "GENERAL KNOWLEDGE"
        confidence = "MEDIUM"
        
        if "[FROM EWU DOCUMENTS]" in answer:
            source = "EWU DOCUMENTS"
            confidence = "HIGH"
        elif "[FROM GENERAL KNOWLEDGE]" in answer:
            source = "GENERAL KNOWLEDGE"
            confidence = "MEDIUM"
        
        if "[HIGH CONFIDENCE]" in answer:
            confidence = "HIGH"
        elif "[MEDIUM CONFIDENCE]" in answer:
            confidence = "MEDIUM"
        elif "[LOW CONFIDENCE]" in answer:
            confidence = "LOW"
        
        return source, confidence
    
    def render_header(self):
        st.markdown(f"""
        <div class="header-container">
            <div class="header-title">🎓 EWU Academic Assistant</div>
            <div class="header-subtitle">AI-Powered Q&A System | East West University</div>
        </div>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = self.load_chat_history()
    
    def display_chat_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # Extract and display source info
                if msg["role"] == "assistant":
                    source, confidence = self.extract_source_info(msg["content"])
                    
                    # Display source badge
                    source_class = "source-ewu" if "EWU DOCUMENTS" in source else "source-general"
                    confidence_class = f"confidence-{confidence.lower()}"
                    
                    st.markdown(
                        f"""
                        <span class="source-badge {source_class}">📍 {source}</span>
                        <span class="source-badge {confidence_class}">✓ {confidence} CONFIDENCE</span>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display answer
                    clean_content = msg["content"]
                    for tag in ["[FROM EWU DOCUMENTS]", "[FROM GENERAL KNOWLEDGE]", 
                               "[HIGH CONFIDENCE]", "[MEDIUM CONFIDENCE]", "[LOW CONFIDENCE]"]:
                        clean_content = clean_content.replace(tag, "")
                    
                    st.markdown(clean_content.strip())
                else:
                    st.markdown(msg["content"])
    
    def process_user_input(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching EWU documents..."):
                    result = self.assistant.query(prompt)
                    answer = result['answer']
                    
                    # Extract source info
                    source, confidence = self.extract_source_info(answer)
                    source_class = "source-ewu" if "EWU DOCUMENTS" in source else "source-general"
                    confidence_class = f"confidence-{confidence.lower()}"
                    
                    # Display source badge
                    st.markdown(
                        f"""
                        <span class="source-badge {source_class}">📍 {source}</span>
                        <span class="source-badge {confidence_class}">✓ {confidence} CONFIDENCE</span>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Clean and display answer
                    clean_content = answer
                    for tag in ["[FROM EWU DOCUMENTS]", "[FROM GENERAL KNOWLEDGE]", 
                               "[HIGH CONFIDENCE]", "[MEDIUM CONFIDENCE]", "[LOW CONFIDENCE]"]:
                        clean_content = clean_content.replace(tag, "")
                    
                    st.markdown(clean_content.strip())
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    self.save_chat_history(st.session_state.messages)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                self.save_chat_history(st.session_state.messages)
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px 0;">
                <h3 style="color: {self.ewu_primary};">📚 EWU Assistant</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.messages = []
                self.save_chat_history([])
                if self.assistant:
                    self.assistant.clear_memory()
                st.rerun()
            
            if st.button("💾 Export Chat", use_container_width=True):
                chat_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
                st.download_button(
                    "⬇️ Download Chat",
                    chat_text,
                    file_name=f"ewu_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
            
            st.markdown("---")
            
            st.markdown(f"""
            <div class="info-box">
                <b>ℹ️ How This Works</b><br>
                <small>
                • Answers from EWU documents get <span class="source-ewu" style="display: inline;">EWU DOCUMENTS</span> badge<br>
                • General knowledge answers get <span class="source-general" style="display: inline;">GENERAL KNOWLEDGE</span> badge<br>
                • Confidence levels show answer reliability
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="font-size: 0.85em; color: #666; margin-top: 20px;">
                <b>About</b><br>
                AI Assistant for East West University<br>
                Powered by OpenAI GPT-4o-mini<br>
                <br>
                <b>Features</b><br>
                ✓ Document-based answers<br>
                ✓ Hybrid knowledge system<br>
                ✓ Source transparency<br>
                ✓ Confidence scoring
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        self.configure_page()
        self.render_sidebar()
        self.render_header()
        
        if not self.check_prerequisites():
            st.stop()
        
        self.assistant = self.initialize_assistant()
        if self.assistant is None:
            st.stop()
        
        self.initialize_session_state()
        self.display_chat_history()
        
        if prompt := st.chat_input("Ask about EWU programs, courses, fees, departments, or policies:"):
            self.process_user_input(prompt)

if __name__ == "__main__":
    app = EnhancedEWUAcademicUI()
    app.run()