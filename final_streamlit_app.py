import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from final_memory_handler import AcademicAssistant

st.set_page_config(
    page_title="EWU Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        padding: 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c4a6d 50%, #1e3a5f 100%);
        border-right: 3px solid #2c4a6d;
    }
    
    [data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(44, 74, 109, 0.4) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(44, 74, 109, 0.6) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(44, 74, 109, 0.3) !important;
        margin: 1.5rem 0 !important;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        animation: slideIn 0.4s ease-out;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        color: #1e3a5f;
        margin-left: 0;
        margin-right: auto;
        max-width: 90%;
        border-left: 5px solid #2c4a6d;
        border-bottom-left-radius: 5px;
    }
    
    .message-timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.75rem;
        font-weight: 500;
    }
    
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c4a6d 100%);
        border-radius: 30px;
        color: white;
        box-shadow: 0 20px 60px rgba(30, 58, 95, 0.4);
        margin: 2rem 0;
        animation: fadeInScale 0.6s ease-out;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .welcome-container h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 800 !important;
    }
    
    .welcome-container p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2rem !important;
        line-height: 1.8 !important;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .feature-card h3 {
        color: white !important;
        margin-bottom: 0.5rem !important;
    }
    
    .feature-card p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.95rem !important;
    }
    
    input[type="text"],
    textarea {
        border: 2px solid #2c4a6d !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    input[type="text"]:focus,
    textarea:focus {
        border: 2px solid #3d5a7c !important;
        box-shadow: 0 0 0 4px rgba(44, 74, 109, 0.1) !important;
        transform: scale(1.01) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(44, 74, 109, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(44, 74, 109, 0.6) !important;
    }
    
    .citation-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(245, 87, 108, 0.3);
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(56, 239, 125, 0.3);
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(245, 87, 108, 0.3);
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(250, 112, 154, 0.3);
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    thead {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%);
        color: white;
    }
    
    th {
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    td {
        padding: 0.9rem 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    tbody tr:hover {
        background: rgba(44, 74, 109, 0.05);
        transition: all 0.2s ease;
    }
    
    code {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%) !important;
        color: white !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2c4a6d 0%, #3d5a7c 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3d5a7c 0%, #2c4a6d 100%);
    }
    
    .loading-animation {
        display: inline-flex;
        gap: 0.4rem;
    }
    
    .loading-animation span {
        width: 0.7rem;
        height: 0.7rem;
        border-radius: 50%;
        background: #2c4a6d;
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .loading-animation span:nth-child(1) { animation-delay: 0s; }
    .loading-animation span:nth-child(2) { animation-delay: 0.2s; }
    .loading-animation span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 80%, 100% { 
            transform: scale(0);
            opacity: 0.5;
        }
        40% { 
            transform: scale(1);
            opacity: 1;
        }
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a5f !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown a {
        color: #2c4a6d !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stMarkdown a:hover {
        color: #3d5a7c !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
        st.session_state.assistant_initialized = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_histories' not in st.session_state:
        st.session_state.chat_histories = load_chat_histories()
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    
    if 'current_chat_title' not in st.session_state:
        st.session_state.current_chat_title = "New Chat"

def load_chat_histories():
    history_dir = Path("chat_history")
    histories = []
    
    if history_dir.exists():
        for json_file in sorted(history_dir.glob("*.json"), reverse=True):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    histories.append(data)
            except Exception as e:
                st.warning(f"Error loading chat history: {e}")
    
    return histories

def save_chat_history(chat_id, title, messages):
    history_dir = Path("chat_history")
    history_dir.mkdir(exist_ok=True)
    
    chat_data = {
        'id': chat_id,
        'title': title,
        'created_at': datetime.now().isoformat(),
        'messages': messages
    }
    
    file_path = history_dir / f"{chat_id}.json"
    try:
        with open(file_path, 'w') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {e}")

def delete_chat_history(chat_id):
    history_dir = Path("chat_history")
    file_path = history_dir / f"{chat_id}.json"
    
    try:
        if file_path.exists():
            file_path.unlink()
            st.session_state.chat_histories = load_chat_histories()
    except Exception as e:
        st.error(f"Error deleting chat: {e}")

def format_timestamp(seconds=None):
    now = datetime.now()
    if seconds is None:
        seconds = now.timestamp()
    
    dt = datetime.fromtimestamp(seconds)
    today = now.date()
    
    if dt.date() == today:
        return dt.strftime("%I:%M %p")
    elif dt.date() == today.replace(day=today.day - 1):
        return "Yesterday"
    else:
        return dt.strftime("%m/%d/%Y")

def initialize_assistant():
    try:
        with st.spinner("Initializing EWU Academic Assistant..."):
            assistant = AcademicAssistant(
                db_path=os.getenv("VECTOR_STORE_PATH", "final_vectorstore/db_faiss"),
                model=os.getenv("LLM_MODEL", "gpt-4o-mini")
            )
            assistant.initialize()
            st.session_state.assistant = assistant
            st.session_state.assistant_initialized = True
            return True
    except Exception as e:
        st.error(f"Failed to initialize assistant: {e}")
        st.info("Make sure you've run final_vector_store.py first to create the vector store.")
        return False

def start_new_chat():
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.current_chat_title = "New Chat"

def load_chat(chat_id):
    try:
        chat_data = next((c for c in st.session_state.chat_histories if c['id'] == chat_id), None)
        if chat_data:
            st.session_state.messages = chat_data['messages']
            st.session_state.current_chat_id = chat_id
            st.session_state.current_chat_title = chat_data['title']
    except Exception as e:
        st.error(f"Error loading chat: {e}")

def extract_chat_title(message):
    if len(message) > 50:
        return message[:50] + "..."
    return message

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎓 EWU Academic Assistant")
        st.markdown("<small>East West University</small>", unsafe_allow_html=True)
        
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
        st.markdown("#### 📋 Chat History")
        
        if not st.session_state.chat_histories:
            st.info("No chat history yet. Start a conversation!")
        else:
            for chat in st.session_state.chat_histories[:15]:
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    if st.button(
                        f"💬 {chat['title']}",
                        use_container_width=True,
                        key=f"chat_{chat['id']}"
                    ):
                        load_chat(chat['id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete", use_container_width=True):
                        delete_chat_history(chat['id'])
                        if st.session_state.current_chat_id == chat['id']:
                            start_new_chat()
                        st.rerun()
        
        st.divider()
        
        if st.session_state.chat_histories:
            if st.button("🧹 Clear All History", use_container_width=True):
                history_dir = Path("chat_history")
                if history_dir.exists():
                    for file in history_dir.glob("*.json"):
                        file.unlink()
                st.session_state.chat_histories = []
                start_new_chat()
                st.rerun()
        
        st.divider()
        
        with st.expander("⚙️ System Info"):
            st.markdown("**Model:** gpt-4o-mini")
            st.markdown("**Temperature:** 0.0")
            st.markdown("**Max Tokens:** 4000")
            st.markdown("**Retrieval k:** 25")
            st.markdown("**Embeddings:** all-MiniLM-L6-v2")
            
            if st.button("🔄 Reinitialize", use_container_width=True):
                st.session_state.assistant_initialized = False
                st.rerun()

def render_chat_message(message):
    is_user = message['role'] == 'user'
    
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem;">
            <div class="chat-message user-message">
                {message['content']}
                <div class="message-timestamp">🕐 {message.get('timestamp', '')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="chat-message assistant-message">
                {message['content']}
                <div class="message-timestamp">🤖 {message.get('timestamp', '')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    render_sidebar()
    
    st.markdown("### 💬 Chat with EWU Assistant")
    
    if not st.session_state.assistant_initialized:
        if not initialize_assistant():
            st.stop()
    
    if st.session_state.current_chat_id:
        st.caption(f"📌 {st.session_state.current_chat_title}")
    
    messages_container = st.container()
    with messages_container:
        if not st.session_state.messages:
            st.markdown('<div class="welcome-container"><h1>👋 Welcome to EWU Academic Assistant</h1><p>Your intelligent guide to East West University academic information</p><div class="feature-grid"><div class="feature-card"><h3>📚 Complete Information</h3><p>Get comprehensive answers with full details from official documents</p></div><div class="feature-card"><h3>📖 Accurate Citations</h3><p>Every fact backed by page numbers and source references</p></div><div class="feature-card"><h3>📊 Tables & Data</h3><p>View complete course structures, requirements, and statistics</p></div><div class="feature-card"><h3>💡 Smart Assistance</h3><p>Ask about courses, faculty, fees, policies, and more</p></div></div></div>', unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_chat_message(message)
    
    st.markdown("---")
    
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([0.93, 0.07])
        
        with col1:
            user_input = st.text_input(
                "Type your question...",
                placeholder="Ask about courses, faculty, fees, programs, requirements...",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.form_submit_button("📤", help="Send", use_container_width=True)
    
    if send_button and user_input and user_input.strip():
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%I:%M %p")
        }
        st.session_state.messages.append(user_message)
        
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_title = extract_chat_title(user_input)
            st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with st.spinner(""):
            st.markdown('<div class="loading-animation"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            try:
                response = st.session_state.assistant.query(user_input)
                assistant_message = {
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': datetime.now().strftime("%I:%M %p")
                }
                st.session_state.messages.append(assistant_message)
                
                save_chat_history(
                    st.session_state.current_chat_id,
                    st.session_state.current_chat_title,
                    st.session_state.messages
                )
                
                st.session_state.chat_histories = load_chat_histories()
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.pop()
        
        st.rerun()

if __name__ == "__main__":
    main()