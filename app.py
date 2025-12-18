"""
Streamlit UI for EWU Academic Assistant - Claude.ai Style Interface
Scalable and production-ready chat interface with chat history
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

# Custom CSS for Claude.ai-like appearance
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main container */
    .main {
        background-color: #ffffff;
        color: #0d0d0d;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        color: #ececf1;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #1a1a1a;
    }
    
    /* Chat messages styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    
    .user-message {
        background-color: #f0f0f0;
        justify-content: flex-end;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
    }
    
    .message-content {
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message .message-content {
        background-color: #0d66d0;
        color: white;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    
    .assistant-message .message-content {
        color: #0d0d0d;
    }
    
    /* Input area */
    .input-container {
        background-color: #ffffff;
        border-top: 1px solid #e5e5e5;
        padding: 1.5rem;
        position: sticky;
        bottom: 0;
    }
    
    /* Buttons */
    .new-chat-btn {
        background-color: #2a2a2a;
        color: #ececf1;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        width: 100%;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .new-chat-btn:hover {
        background-color: #3a3a3a;
    }
    
    .chat-history-item {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .chat-history-item:hover {
        background-color: #2a2a2a;
    }
    
    .chat-history-item.active {
        background-color: #444444;
    }
    
    /* Source tags */
    .source-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #e5e5e5;
        color: #0d0d0d;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .source-ewu {
        background-color: #d4edda;
        color: #155724;
    }
    
    .source-general {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-flex;
        gap: 0.25rem;
    }
    
    .loading-dots span {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background-color: #0d66d0;
        animation: bounce 1.4s infinite;
    }
    
    .loading-dots span:nth-child(1) {
        animation-delay: 0s;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes bounce {
        0%, 100% {
            opacity: 0.3;
        }
        50% {
            opacity: 1;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
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
            st.success("Chat deleted successfully")
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
                db_path=os.getenv("VECTOR_STORE_PATH", "vectorstore1/db_faiss"),
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
            # Display chat history items
            for chat in st.session_state.chat_histories:
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    is_active = chat['id'] == st.session_state.current_chat_id
                    if st.button(
                        f"💬 {chat['title']}\n_{format_timestamp()}_",
                        use_container_width=True,
                        key=f"chat_{chat['id']}"
                    ):
                        load_chat(chat['id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
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
            st.write("**LLM Configuration:**")
            st.write(f"- Model: `gpt-4o-mini`")
            st.write(f"- Temperature: `0.05`")
            st.write(f"- Max Tokens: `3000`")
            st.write(f"- Retrieval k: `60`")
            
            st.write("\n**Embedding Model:**")
            st.write("- sentence-transformers/all-MiniLM-L6-v2")
            
            if st.button("🔄 Reinitialize Assistant"):
                st.session_state.assistant_initialized = False
                st.rerun()


def render_chat_message(message):
    """Render a single chat message"""
    is_user = message['role'] == 'user'
    
    if is_user:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
            <div style="background-color: #0d66d0; color: white; padding: 1rem; border-radius: 0.75rem; max-width: 80%;">
                <p style="margin: 0;">{message['content']}</p>
                <small style="opacity: 0.7; font-size: 0.85rem;">{message.get('timestamp', '')}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="background-color: #f7f7f8; padding: 1rem; border-radius: 0.75rem; border: 1px solid #e5e5e5;">
                <p style="margin: 0 0 0.5rem 0;">{message['content']}</p>
                <small style="opacity: 0.6; font-size: 0.85rem;">{message.get('timestamp', '')}</small>
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
        st.caption(f"Chat ID: {st.session_state.current_chat_id}")
    
    # Display chat messages
    messages_container = st.container()
    with messages_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #8a8a8a;">
                <h2>👋 Welcome to EWU Academic Assistant</h2>
                <p>Ask me anything about East West University academic programs, courses, faculty, and requirements.</p>
                <p style="margin-top: 1rem; font-size: 0.9rem;">
                    <em>Example questions:</em><br>
                    • What are CSE course requirements?<br>
                    • Who are the faculty members in the Engineering department?<br>
                    • What is the fee structure for 2024?
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_chat_message(message)
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        user_input = st.text_input(
            "Type your question...",
            placeholder="Ask about courses, faculty, fees, requirements...",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("📤", help="Send message", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
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