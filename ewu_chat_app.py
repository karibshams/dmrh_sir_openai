import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
from ewu_qa_engine import EWUQAEngine

st.set_page_config(page_title="EWU Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #f0f4f8, #d9e2ec); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3a5f, #2c4a6d); }
[data-testid="stSidebar"] * { color: white !important; }
.user-msg { background: #2c4a6d; color: white; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; }
.bot-msg { background: #f0f4f8; color: #1e3a5f; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border-left: 4px solid #2c4a6d; }
h1, h2, h3 { color: #1e3a5f !important; }
table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }
thead { background: #2c4a6d; color: white; }
th, td { padding: 0.8rem; border-bottom: 1px solid #ddd; text-align: left; }
tbody tr:hover { background: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

def init_state():
    if 'qa' not in st.session_state:
        st.session_state.qa = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chats' not in st.session_state:
        st.session_state.chats = load_chats()

def load_chats():
    """Load saved chats"""
    path = Path("chats")
    if not path.exists():
        return []
    return sorted([json.loads(f.read_text()) for f in path.glob("*.json")], 
                  key=lambda x: x['created'], reverse=True)[:10]

def save_chat(title, messages):
    """Save chat"""
    path = Path("chats")
    path.mkdir(exist_ok=True)
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    (path / f"{chat_id}.json").write_text(json.dumps({
        'id': chat_id, 'title': title, 'created': chat_id, 'messages': messages
    }, indent=2))
    st.session_state.chats = load_chats()

def load_qa():
    """Initialize QA engine"""
    if st.session_state.qa is None:
        with st.spinner("Loading EWU Assistant..."):
            qa = EWUQAEngine()
            if qa.init():
                st.session_state.qa = qa
                return True
            return False
    return True

def sidebar():
    with st.sidebar:
        st.title("🎓 EWU Assistant")
        st.caption("v2 - LEAN")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.subheader("History")
        
        if st.session_state.chats:
            for chat in st.session_state.chats:
                if st.button(f"💬 {chat['title'][:30]}", use_container_width=True, key=chat['id']):
                    st.session_state.messages = chat['messages']
                    st.rerun()
        else:
            st.info("No chat history")
        
        st.divider()
        with st.expander("⚙️ Settings"):
            st.write("**Model:** gpt-4o-mini")
            st.write("**Retrieval k:** 80")
            st.write("**Temp:** 0.0")

def main():
    init_state()
    sidebar()
    
    st.header("💬 EWU Academic Assistant")
    
    if not load_qa():
        st.error("Failed to load assistant")
        st.stop()
    
    # Display messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"<div class='user-msg'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><b>Assistant:</b></div>", unsafe_allow_html=True)
            st.markdown(msg['content'])
    
    # Input
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        question = st.text_input(
            "Ask about EWU...",
            placeholder="Courses, faculty, fees, admission, calendar...",
            label_visibility="collapsed"
        )
    with col2:
        send = st.button("📤", use_container_width=True)
    
    if send and question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"<div class='user-msg'><b>You:</b> {question}</div>", unsafe_allow_html=True)
        
        # Get answer
        with st.spinner("Thinking..."):
            answer = st.session_state.qa.query(question)
        
        # Add bot message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"<div class='bot-msg'><b>Assistant:</b></div>", unsafe_allow_html=True)
        st.markdown(answer)
        
        # Save chat
        if len(st.session_state.messages) == 2:  # First exchange
            title = question[:40] + "..." if len(question) > 40 else question
            save_chat(title, st.session_state.messages)
        
        st.rerun()

if __name__ == "__main__":
    main()