import streamlit as st
import uuid
import time
from query_chats_wrapped import initialize_chat, run_chatbot_session, th_state

st.set_page_config(page_title="Sweet Robo AI Assistant", layout="centered")
st.title("🤖 Sweet Robo AI Assistant")

# === Reset Chat Button (Improved Logic + th_state reset) ===
if st.button("🔁 Reset Chat"):
    keys_to_clear = [
        "chat_initialized", "chat_id", "machine_type", "history",
        "thread_id", "used_matches_by_thread", "match_pointer"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # ✅ Also clear global in-memory th_state
    th_state["thread_id"] = None
    th_state["machine_type"] = None
    th_state["used_matches_by_thread"] = {}
    th_state["conversation_history"] = []
    th_state["solution_attempts"] = {}
    th_state["embedding_cache"] = {}

    st.rerun()

# === Session State Init ===
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
    st.session_state.chat_id = None
    st.session_state.machine_type = None
    st.session_state.history = []

# === Start New Chat Section ===
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

# === Active Chat Mode ===
if st.session_state.chat_initialized:
    st.markdown(f"**Chat ID:** `{st.session_state.chat_id}`")
    st.markdown(f"**Machine Type:** `{st.session_state.machine_type}`")

    # === Render Chat History ===
    for speaker, message in st.session_state.history:
        with st.chat_message("user" if speaker == "You" else "assistant"):
            st.markdown(message)

    # === Message Input at Bottom ===
    if user_input := st.chat_input("Describe your issue..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Sweet Robo is thinking..."):
            response = run_chatbot_session(user_input.strip())
            time.sleep(0.5)
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save to history
        st.session_state.history.append(("You", user_input.strip()))
        st.session_state.history.append(("Sweet Robo", response))
