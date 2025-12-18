"""
Streamlit UI for EWU Academic Assistant - Claude.ai Style Interface
Scalable and production-ready chat interface with chat history
EWU Branded Colors & Fixed Input Issues
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

# EWU Brand Colors
EWU_PRIMARY = "#003366"      # Dark Blue (EWU Primary)
EWU_SECONDARY = "#CC0000"    # Red (EWU Secondary)
EWU_LIGHT = "#E8F0F7"        # Light Blue
EWU_DARK = "#1a1a1a"         # Dark background
EWU_TEXT = "#0d0d0d"         # Dark text
EWU_ACCENT = "#0066CC"       # Accent Blue

# Custom CSS for Claude.ai-like appearance with EWU colors
st.markdown(f"""
<style>
    * {{
        margin: 0;
        padding: 0;
    }}
    
    /* Main container */
    .main {{
        background-color: #ffffff;
        color: {EWU_TEXT};
    }}
    
    /* Sidebar styling - EWU Dark Theme */
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {EWU_PRIMARY} 0%, #004d7a 100%);
        color: #ffffff;
    }}
    
    [data-testid="stSidebar"] > div {{
        background: linear-gradient(135deg, {EWU_PRIMARY} 0%, #004d7a 100%);
    }}
    
    /* Sidebar text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #ffffff !important;
    }}
    
    /* Chat messages styling */
    .chat-message {{
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }}
    
    .user-message {{
        background-color: {EWU_LIGHT};
        justify-content: flex-end;
    }}
    
    .assistant-message {{
        background-color: #f7f7f8;
        border-left: 4px solid {EWU_PRIMARY};
    }}
    
    .message-content {{
        max-width: 85%;
        word-wrap: break-word;
    }}
    
    .user-message .message-content {{
        background: linear-gradient(135deg, {EWU_PRIMARY} 0%, {EWU_ACCENT} 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 51, 102, 0.15);
    }}
    
    .assistant-message .message-content {{
        color: {EWU_TEXT};
        padding: 0.5rem 0;
    }}
    
    /* Input area */
    .input-container {{
        background-color: #ffffff;
        border-top: 2px solid {EWU_PRIMARY};
        padding: 1.5rem;
        position: sticky;
        bottom: 0;
    }}
    
    /* Buttons - EWU Primary Color */
    .new-chat-btn {{
        background: linear-gradient(135deg, {EWU_PRIMARY} 0%, #004d7a 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        width: 100%;
        margin-bottom: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .new-chat-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 51, 102, 0.3);
    }}
    
    .chat-history-item {{
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        color: #ffffff;
    }}
    
    .chat-history-item:hover {{
        background-color: rgba(255, 255, 255, 0.15);
        transform: translateX(5px);
    }}
    
    .chat-history-item.active {{
        background: linear-gradient(90deg, {EWU_SECONDARY} 0%, rgba(204, 0, 0, 0.7) 100%);
        border-left: 3px solid {EWU_SECONDARY};
    }}
    
    /* Source tags - EWU Themed */
    .source-tag {{
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background-color: {EWU_LIGHT};
        color: {EWU_PRIMARY};
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        border-left: 3px solid {EWU_PRIMARY};
    }}
    
    .source-ewu {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-left: 3px solid #28a745;
    }}
    
    .source-general {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404;
        border-left: 3px solid #ffc107;
    }}
    
    /* Loading animation */
    .loading-dots {{
        display: inline-flex;
        gap: 0.25rem;
    }}
    
    .loading-dots span {{
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: linear-gradient(135deg, {EWU_PRIMARY} 0%, {EWU_ACCENT} 100%);
        animation: bounce 1.4s infinite;
    }}
    
    .loading-dots span:nth-child(1) {{
        animation-delay: 0s;
    }}
    
    .loading-dots span:nth-child(2) {{
        animation-delay: 0.2s;
    }}
    
    .loading-dots span:nth-child(3) {{
        animation-delay: 0.4s;
    }}
    
    @keyframes bounce {{
        0%, 100% {{
            opacity: 0.3;
        }}
        50% {{
            opacity: 1;
        }}
    }}
    
    /* Text input styling */
    input[type="text"] {{
        border: 2px solid {EWU_LIGHT} !important;
        border-radius: 0.5rem !important;
    }}
    
    input[type="text"]:focus {{
        border: 2px solid {EWU_PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(0, 51, 102, 0.1) !important;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #f1f1f1;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {EWU_PRIMARY};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {EWU_SECONDARY};
    }}
    
    /* Header styling */
    h1, h2, h3 {{
        color: {EWU_PRIMARY} !important;
    }}
    
    /* Divider */
    hr {{
        border-top: 2px solid {EWU_PRIMARY} !important;
    }}
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
            <div style="background: linear-gradient(135deg, #003366 0%, #0066CC 100%); color: white; padding: 1rem; border-radius: 0.75rem; max-width: 80%; box-shadow: 0 2px 8px rgba(0, 51, 102, 0.15);">
                <p style="margin: 0;">{message['content']}</p>
                <small style="opacity: 0.8; font-size: 0.85rem;">{message.get('timestamp', '')}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format assistant message with source tags
        content = message['content']
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="background-color: #f7f7f8; padding: 1rem; border-radius: 0.75rem; border-left: 4px solid #003366;">
                <div style="color: #0d0d0d; line-height: 1.6;">{content}</div>
                <small style="opacity: 0.6; font-size: 0.85rem; margin-top: 0.5rem; display: block;">{message.get('timestamp', '')}</small>
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
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem 2rem; color: #666; background: linear-gradient(135deg, #E8F0F7 0%, #f0f4f8 100%); border-radius: 1rem; border: 2px dashed #003366;">
                <h2 style="color: #003366;">👋 Welcome to EWU Academic Assistant</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">Ask me anything about East West University academic programs, courses, faculty, and requirements.</p>
                <p style="margin-top: 1.5rem; font-size: 0.95rem; color: #555;">
                    <em><strong>Example questions:</strong></em><br>
                    • What are CSE course requirements?<br>
                    • Who are the faculty members in the Engineering department?<br>
                    • What is the fee structure for 2024?<br>
                    • Show me the 1st year courses for CSE
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_chat_message(message)
    
    # Input area - FIXED VERSION
    st.markdown("---")
    
    # Create a form for proper Enter key handling
    with st.form(key='chat_form', clear_on_submit=True):
        col1, col2 = st.columns([0.95, 0.05])
        
        with col1:
            user_input = st.text_input(
                "Type your question...",
                placeholder="Ask about courses, faculty, fees, requirements...",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.form_submit_button("📤", help="Send message")
    
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