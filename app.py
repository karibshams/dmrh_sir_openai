"""
Streamlit UI for EWU Academic Assistant - Claude.ai Style Interface
Scalable and production-ready chat interface with chat history
EWU Branded Colors - Dark & Light Mode Compatible
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from enhanced_create_memory_for_llm import AcademicAssistant

# Configure Streamlit page
st.set_page_config(
    page_title="EWU Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Claude.ai-like appearance with EWU colors (Dark & Light Mode)
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main container */
    .main {
        background-color: transparent;
    }
    
    /* Sidebar styling - EWU Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #003d7a 0%, #00508c 100%) !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: linear-gradient(135deg, #003d7a 0%, #00508c 100%) !important;
    }
    
    /* Sidebar text - Always white */
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] button {
        color: #ffffff !important;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Chat history items */
    [data-testid="stSidebar"] .chat-history-btn {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.2s !important;
        word-wrap: break-word !important;
    }
    
    [data-testid="stSidebar"] .chat-history-btn:hover {
        background-color: rgba(255, 255, 255, 0.12) !important;
        transform: translateX(3px) !important;
    }
    
    /* User message styling */
    .user-message-box {
        background: linear-gradient(135deg, #0052a3 0%, #0066cc 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        max-width: 75%;
        margin-left: auto;
        margin-right: 0;
        box-shadow: 0 2px 8px rgba(0, 82, 163, 0.3);
        word-wrap: break-word;
    }
    
    .user-message-time {
        opacity: 0.85;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        display: block;
    }
    
    /* Assistant message styling - Light/Dark mode compatible */
    .assistant-message-box {
        background-color: var(--assistant-bg);
        color: var(--assistant-text);
        padding: 1rem 1.25rem;
        border-left: 4px solid #0052a3;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        max-width: 85%;
        margin-left: 0;
        word-wrap: break-word;
    }
    
    .assistant-message-time {
        opacity: 0.6;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        display: block;
    }
    
    /* Light mode variables */
    [data-theme="light"] {
        --assistant-bg: #f7f7f8;
        --assistant-text: #0d0d0d;
    }
    
    /* Dark mode variables */
    [data-theme="dark"] {
        --assistant-bg: #2a2a2a;
        --assistant-text: #e5e5e5;
    }
    
    /* Default fallback */
    :root {
        --assistant-bg: #f7f7f8;
        --assistant-text: #0d0d0d;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --assistant-bg: #2a2a2a;
            --assistant-text: #e5e5e5;
        }
    }
    
    /* Input styling */
    input[type="text"],
    textarea {
        border: 1.5px solid #0052a3 !important;
        border-radius: 0.75rem !important;
    }
    
    input[type="text"]:focus,
    textarea:focus {
        border: 2px solid #0066cc !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.15) !important;
    }
    
    /* Form styling */
    form {
        border-top: 1px solid #0052a3;
        padding-top: 1rem;
    }
    
    /* Welcome message */
    .welcome-box {
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 1rem;
        border: 2px dashed #0052a3;
        background: linear-gradient(135deg, rgba(0, 82, 163, 0.05) 0%, rgba(0, 102, 204, 0.05) 100%);
    }
    
    .welcome-box h2 {
        color: #0052a3 !important;
    }
    
    /* Source tags - Light/Dark compatible */
    .source-tag {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background-color: #e8f0f7;
        color: #0052a3;
        border-left: 3px solid #0052a3;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .source-ewu {
        background-color: #d4edda;
        color: #155724;
        border-left-color: #28a745;
    }
    
    .source-general {
        background-color: #fff3cd;
        color: #856404;
        border-left-color: #ffc107;
    }
    
    /* Code blocks */
    code {
        background-color: rgba(0, 82, 163, 0.1) !important;
        color: #0052a3 !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 0.25rem !important;
    }
    
    /* Tables */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    
    thead {
        background-color: #e8f0f7;
        border-bottom: 2px solid #0052a3;
    }
    
    th {
        color: #0052a3;
        font-weight: 600;
        padding: 0.75rem;
    }
    
    td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }
    
    tbody tr:hover {
        background-color: rgba(0, 82, 163, 0.05);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #0052a3;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0066cc;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-flex;
        gap: 0.3rem;
    }
    
    .loading-dots span {
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
        background: #0066cc;
        animation: bounce 1.4s infinite;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: 0s; }
    .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0052a3 !important;
    }
    
    /* Link styling */
    a {
        color: #0066cc !important;
    }
    
    a:hover {
        color: #0052a3 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
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
    """Load all saved chat histories from disk"""
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
    """Save chat history to disk"""
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
    """Delete a chat history file"""
    history_dir = Path("chat_history")
    file_path = history_dir / f"{chat_id}.json"
    
    try:
        if file_path.exists():
            file_path.unlink()
            st.session_state.chat_histories = load_chat_histories()
    except Exception as e:
        st.error(f"Error deleting chat: {e}")


def format_timestamp(seconds=None):
    """Format timestamp for display"""
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
    """Initialize the EWU Academic Assistant"""
    try:
        with st.spinner("🔧 Initializing EWU Academic Assistant..."):
            assistant = AcademicAssistant(
                db_path=os.getenv("VECTOR_STORE_PATH", "vectorstore/db_faiss"),
                model=os.getenv("LLM_MODEL", "gpt-4o-mini")
            )
            assistant.initialize()
            st.session_state.assistant = assistant
            st.session_state.assistant_initialized = True
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize assistant: {e}")
        st.info("Make sure you've run `enhanced_vector_store.py` first to create the vector store.")
        return False


def start_new_chat():
    """Start a new chat session"""
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.current_chat_title = "New Chat"


def load_chat(chat_id):
    """Load a saved chat from history"""
    try:
        chat_data = next((c for c in st.session_state.chat_histories if c['id'] == chat_id), None)
        if chat_data:
            st.session_state.messages = chat_data['messages']
            st.session_state.current_chat_id = chat_id
            st.session_state.current_chat_title = chat_data['title']
    except Exception as e:
        st.error(f"Error loading chat: {e}")


def extract_chat_title(message):
    """Extract a title from the first user message"""
    if len(message) > 50:
        return message[:50] + "..."
    return message


def render_sidebar():
    """Render the sidebar with chat history"""
    with st.sidebar:
        st.markdown("### 🎓 EWU Academic Assistant")
        st.markdown(f"<small>East West University</small>", unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
        # Chat History Section
        st.markdown("#### 📋 Chat History")
        
        if not st.session_state.chat_histories:
            st.info("No chat history yet. Start a new conversation!")
        else:
            for chat in st.session_state.chat_histories:
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    if st.button(
                        f"💬 {chat['title']}\n_{format_timestamp()}_",
                        use_container_width=True,
                        key=f"chat_{chat['id']}"
                    ):
                        load_chat(chat['id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat", use_container_width=True):
                        delete_chat_history(chat['id'])
                        if st.session_state.current_chat_id == chat['id']:
                            start_new_chat()
                        st.rerun()
        
        st.divider()
        
        # Clear All History
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
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.markdown("""
            <div style="color: #ffffff;">
                <p style="font-weight: 600; margin-bottom: 1rem; color: #ffffff;"><strong>LLM Configuration:</strong></p>
                <p style="margin: 0.5rem 0; color: #e5e5e5;">Model: <code style="background-color: rgba(255,255,255,0.1); color: #90caf9; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">gpt-4o-mini</code></p>
                <p style="margin: 0.5rem 0; color: #e5e5e5;">Temperature: <code style="background-color: rgba(255,255,255,0.1); color: #90caf9; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">0.05</code></p>
                <p style="margin: 0.5rem 0; color: #e5e5e5;">Max Tokens: <code style="background-color: rgba(255,255,255,0.1); color: #90caf9; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">3000</code></p>
                <p style="margin: 0.5rem 0; color: #e5e5e5;">Retrieval k: <code style="background-color: rgba(255,255,255,0.1); color: #90caf9; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">60</code></p>
                
                <p style="font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.75rem; color: #ffffff;"><strong>Embedding Model:</strong></p>
                <p style="margin: 0.5rem 0; color: #e5e5e5;">sentence-transformers/all-MiniLM-L6-v2</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Reinitialize Assistant", use_container_width=True):
                st.session_state.assistant_initialized = False
                st.rerun()


def render_chat_message(message):
    """Render a single chat message"""
    is_user = message['role'] == 'user'
    
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
            <div class="user-message-box">
                {message['content']}
                <span class="user-message-time">{message.get('timestamp', '')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div class="assistant-message-box">
                {message['content']}
                <span class="assistant-message-time">{message.get('timestamp', '')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat area
    st.markdown("### 💬 Chat")
    
    # Initialize assistant if not done
    if not st.session_state.assistant_initialized:
        if not initialize_assistant():
            st.stop()
    
    # Display current chat title
    if st.session_state.current_chat_id:
        st.caption(f"📌 Chat ID: {st.session_state.current_chat_id}")
    
    # Display chat messages
    messages_container = st.container()
    with messages_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-box">
                <h2>👋 Welcome to EWU Academic Assistant</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">Ask me anything about East West University academic programs, courses, faculty, and requirements.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_chat_message(message)
    
    # Input area
    st.markdown("---")
    
    # Create a form for proper Enter key handling
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([0.93, 0.07])
        
        with col1:
            user_input = st.text_input(
                "Type your question...",
                placeholder="Ask about courses, faculty, fees, requirements...",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.form_submit_button("📤", help="Send message", use_container_width=True)
    
    # Process user input
    if send_button and user_input and user_input.strip():
        # Add user message
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%I:%M %p")
        }
        st.session_state.messages.append(user_message)
        
        # Generate title for new chat
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_title = extract_chat_title(user_input)
            st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get assistant response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.assistant.query(user_input)
                assistant_message = {
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': datetime.now().strftime("%I:%M %p")
                }
                st.session_state.messages.append(assistant_message)
                
                # Save chat history
                save_chat_history(
                    st.session_state.current_chat_id,
                    st.session_state.current_chat_title,
                    st.session_state.messages
                )
                
                # Reload chat histories
                st.session_state.chat_histories = load_chat_histories()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                # Remove the user message if there was an error
                st.session_state.messages.pop()
        
        st.rerun()


if __name__ == "__main__":
    main()