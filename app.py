import streamlit as st
import uuid
from query_chats_wrapped import initialize_chat, run_chatbot_session, th_state

st.set_page_config(page_title="Sweet Robo AI Assistant", layout="centered")
st.title("🤖 Sweet Robo AI Assistant")

# === Session State Initialization ===
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
    st.session_state.chat_id = None
    st.session_state.machine_type = None
    st.session_state.history = []

# === Reset Button ===
if st.button("🔁 Reset Chat"):
    st.session_state.chat_initialized = False
    st.session_state.chat_id = None
    st.session_state.machine_type = None
    st.session_state.history = []
    th_state["thread_id"] = None
    th_state["machine_type"] = None
    th_state["used_matches_by_thread"] = {}
    th_state["conversation_history"] = []
    st.rerun()

# === New Chat Button and Machine Selector ===
st.markdown("### Start a New Chat")
col1, col2 = st.columns([2, 1])

with col1:
    selected_machine = st.selectbox("Select machine type:", [
        "Cotton Candy", "Ice Cream", "Balloon Bot", "Candy Monster",
        "Popcart", "Mr. Pop", "Marshmallow Spaceship"
    ])
with col2:
    if st.button("Start Chat"):
        init_result = initialize_chat(selected_machine)
        st.session_state.chat_id = init_result["thread_id"]
        st.session_state.machine_type = init_result["machine_type"]
        st.session_state.chat_initialized = True
        st.session_state.history = []

# === Show Current Chat ===
if st.session_state.chat_initialized:
    st.markdown(f"**Chat ID:** `{st.session_state.chat_id}`")
    st.markdown(f"**Machine Type:** `{st.session_state.machine_type}`")
    user_input = st.text_input("Describe your issue:")

    if st.button("Submit") and user_input:
        with st.spinner("Searching knowledge base..."):
            response = run_chatbot_session(user_input)
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Sweet Robo", response))

# === Chat History ===
if st.session_state.chat_initialized and st.session_state.history:
    st.markdown("---")
    st.markdown("### Chat History")
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")
