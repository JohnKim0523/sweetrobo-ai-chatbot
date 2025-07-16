import os
import streamlit as st
from dotenv import dotenv_values
import json
import uuid
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone
import numpy as np
import re
from difflib import SequenceMatcher

# === Load keys from secrets or .env ===
config = dotenv_values(".env")
openai_key = st.secrets.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY"))
pinecone_key = st.secrets.get("PINECONE_API_KEY", config.get("PINECONE_API_KEY"))

if not openai_key or not pinecone_key:
    raise ValueError("❌ Missing OPENAI_API_KEY or PINECONE_API_KEY")

# === Config ===
embedding_model = "text-embedding-3-small"
CONFIDENCE_THRESHOLD = 0.6

# === Init clients ===
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("sweetrobo-ai")

# === Global Chat State ===
th_state = {
    "thread_id": None,
    "machine_type": None,
    "used_matches_by_thread": {},
    "conversation_history": [],
    "solution_attempts": {},
    "email_collected": False,
}

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def track_solution_failure():
    thread_id = th_state["thread_id"]
    th_state["solution_attempts"][thread_id] = (
        th_state["solution_attempts"].get(thread_id, 0) + 1
    )
    return th_state["solution_attempts"][thread_id]

def handle_email_logic(user_question):
    if "@" in user_question:
        if is_valid_email(user_question.strip()):
            th_state["email_collected"] = True
            return "✅ Email received. Adding the employee to the chat session."
        else:
            return "⚠️ That doesn’t look like a valid email. Please check it and try again."

    if any(phrase in user_question.lower() for phrase in ["this didnt solve", "didn't work", "not fixed", "still broken"]):
        if not th_state["email_collected"]:
            return "Please provide the employee's email address so we can escalate this issue."

    return None

def initialize_chat(selected_machine: str):
    th_state["thread_id"] = str(uuid.uuid4())
    machine_mapping = {
        "cotton candy": "COTTON_CANDY",
        "ice cream": "ROBO_ICE_CREAM",
        "balloon bot": "BALLOON_BOT",
        "candy monster": "CANDY_MONSTER",
        "popcart": "POPCART",
        "mr. pop": "MR_POP",
        "marshmallow spaceship": "MARSHMALLOW_SPACESHIP",
    }
    machine_type = machine_mapping.get(selected_machine.strip().lower())
    if not machine_type:
        raise ValueError("Unknown machine type selected.")
    th_state["machine_type"] = machine_type
    return {"thread_id": th_state["thread_id"], "machine_type": machine_type}

def is_question_too_vague(user_q):
    specific_keywords = {
        'machine', 'candy', 'cotton', 'issue', 'sugar', 'after', 'settings', 'check',
        'error', 'please', 'stick', 'ensure', 'screen', 'support', 'video', 'power',
        'system', 'clean', 'sensor', 'provide', 'test', 'burner', 'further', 'team',
        'design', 'during', 'could', 'properly', 'furnace', 'issues', 'motor',
        'number', 'persists', 'time', 'correct', 'cleaning', 'nozzle', 'portal',
        'admin', 'update', 'sure', 'working', 'water', 'sticks', 'nayax', 'send',
        'correctly', 'showing', 'help', 'confirm', 'payment', 'heating', 'machines',
        'replacement', 'device', 'balloon', 'restart', 'problem', 'change', 'stuck',
        'through', 'using', 'menu', 'verify', 'connected', 'alert', 'inventory',
        'temperature', 'address', 'prevent', 'remove', 'resolve', 'software',
        'contact', 'robo', 'before', 'again', 'cable', 'data', 'access', 'down',
        'reset', 'card', 'setting', 'alerts', 'sync', 'process', 'call', 'print',
        'sweet', 'clear', 'causing', 'right', 'replace', 'internal', 'loose',
        'assistance', 'production', 'getting', 'inside'
    }
    if any(word in user_q.lower() for word in specific_keywords):
        return False

    prompt = f"""You are a support assistant. You will be given a user's message.

You must check if the message matches any known vague expressions from the list below. These expressions include messages that are too general, do not mention any part, symptom, or error code, and cannot be acted on without clarification.

- my machine is not working
- nothing is happening
- i need help
- it’s broken
- i’m having an issue
- something’s wrong
- please assist
- i can’t use it
- won’t start
- stopped working
- not working again
- what’s going on?
- can you fix this?
- having trouble
- doesn’t respond
- need support
- weird behavior
- won’t power on
- power issue
- broken again
- doesn’t run
- not behaving right
- crashes
- keeps glitching
- doesn’t load
- not responding
- app not working
- still not fixed
- confused
- won’t go
- machine failure
- nothing loads
- restart didn’t work
- not functional
- why isn’t it working
- how come it’s broken
- just doesn’t work
- problem with it
- general malfunction
- help me
- it’s acting weird
- glitching again
- it won’t do anything
- i give up
- issue persists
- same problem again
- problem still exists
- can’t figure it out
- doesn’t work right
- need some help

User message: \"{user_q}\"

Respond only with:
- \"yes\"
- \"no\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return "yes" in response.choices[0].message.content.lower()
    except Exception:
        return False

def is_similar_answer(ans1, ans2, threshold=0.85):
    return SequenceMatcher(None, ans1, ans2).ratio() >= threshold

def is_related(user_q, match_q, match_a):
    user_words = set(re.findall(r"\w+", user_q.lower()))
    match_words = set(re.findall(r"\w+", (match_q + " " + match_a).lower()))
    overlap = user_words.intersection(match_words)
    return len(overlap) / max(len(user_words), 1) > 0.3

def is_already_given(answer, history, threshold=0.85):
    for entry in history:
        if entry["role"] == "assistant":
            prev_ans = entry["content"]
            if SequenceMatcher(None, answer.strip(), prev_ans.strip()).ratio() >= threshold:
                return True
    return False

def fetch_valid_matches(query_embedding, previous_ids, error_code_filter, query_text):
    filter_query = {"machine_type": {"$eq": th_state["machine_type"]}}
    if error_code_filter:
        filter_query["error_codes"] = {"$in": [str(error_code_filter)]}
    else:
        filter_query["error_codes"] = {"$exists": False}

    all_valid = []
    seen_answers = set()
    score_cutoff = 10
    keyword_focus = "stick" if "stick" in query_text.lower() else None

    while len(all_valid) < 5 and score_cutoff >= 1:
        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True,
            include_values=True,
            filter=filter_query,
        )
        for match in results.matches:
            if match.id in previous_ids:
                continue
            meta = match.metadata
            usefulness_score = meta.get("usefulness_score", 0)
            confidence = meta.get("confidence", 0.0)
            answer = meta.get("a", "")
            tags = meta.get("tags", [])
            if usefulness_score == score_cutoff and confidence >= CONFIDENCE_THRESHOLD:
                if keyword_focus and not any(keyword_focus in t.lower() for t in tags):
                    continue
                if isinstance(match.values, list) and len(match.values) == 1536:
                    if answer not in seen_answers:
                        all_valid.append((match, usefulness_score, confidence))
                        seen_answers.add(answer)
        if len(all_valid) < 5:
            score_cutoff -= 1

    return all_valid[:100]  # Fetch up to 100 matches

def run_chatbot_session(user_question: str) -> str:
    email_response = handle_email_logic(user_question)
    if email_response:
        return email_response

    thread_id = th_state["thread_id"]
    used_matches_by_thread = th_state["used_matches_by_thread"]

    if re.fullmatch(r"\d{4,}", user_question):
        user_question = f"how to fix error {user_question}"

    if is_question_too_vague(user_question):
        return ("That’s a bit too general. Could you describe exactly what’s going wrong "
                "(e.g., error code, what part is malfunctioning, or what’s not working as expected)?")

    is_followup_phrases = ["didn't work", "didnt work", "didn't help", "didnt help",
                           "still broken", "not fixed", "didn’t resolve", "not working"]
    is_followup = any(p in user_question.lower() for p in is_followup_phrases)

    if is_followup and thread_id in used_matches_by_thread:
        previous = used_matches_by_thread[thread_id]
        query_embedding = previous["embedding"]
        previous_ids = previous["used_ids"]
        error_code_filter = previous["error_filter"]
        original_question = previous["original_question"]
        failure_count = track_solution_failure()
        if failure_count >= 2:
            return "This seems persistent. Escalating to a human agent now. Please wait..."
        top_matches = fetch_valid_matches(query_embedding, previous_ids, error_code_filter, original_question)
    else:
        response = client.embeddings.create(model=embedding_model, input=[user_question])
        query_embedding = response.data[0].embedding
        previous_ids = set()
        original_question = user_question
        match = re.search(r"(\d{4,})", user_question)
        error_code_filter = int(match.group(1)) if "error" in user_question.lower() and match else None
        top_matches = fetch_valid_matches(query_embedding, previous_ids, error_code_filter, original_question)
        if not top_matches:
            return "❌ No high-confidence, high-score matches found."
        used_ids = previous_ids.union({m[0].id for m in top_matches})
        used_matches_by_thread[thread_id] = {
            "embedding": query_embedding,
            "used_ids": used_ids,
            "error_filter": error_code_filter,
            "original_question": original_question,
        }

    # === Filter new answers only ===
    filtered_qas = []
    seen_ids = set()
    for match, usefulness, confidence in top_matches:
        if len(filtered_qas) >= 5:
            break
        answer = match.metadata.get("a", "[No A]").strip()
        if not answer or match.id in seen_ids:
            continue
        q = match.metadata.get("q", "")
        a = match.metadata.get("a", "")
        if is_related(original_question, q, a) and not is_already_given(a, th_state["conversation_history"]):
            filtered_qas.append(a)
            seen_ids.add(match.id)

    if not filtered_qas:
        failure_count = track_solution_failure()
        if failure_count >= 2:
            return "This seems persistent. Escalating to a human agent now. Please wait..."
        return "Sorry, I couldn’t find any new helpful info for this issue. Could you describe it in more detail or mention exactly what you tried?"

    combined_input = "\n\n".join(filtered_qas)
    first_match_answer = filtered_qas[0]

    simple_q_phrases = ["can i", "do you", "does it", "is there", "are there", "can we", "is it possible"]
    is_simple_question = any(original_question.lower().startswith(p) for p in simple_q_phrases)
    is_short_answer = len(first_match_answer.split()) <= 12 and first_match_answer.lower().startswith(("yes", "no", "sorry", "unfortunately"))

    try:
        if is_simple_question and is_short_answer:
            final_answer = first_match_answer
        else:
            gpt_prompt = f"""
You are a helpful AI assistant for customer support.{' The user said the initial solution didn\'t work.' if is_followup else ''}

You are given up to 5 detailed technical answers. Summarize them fully, combining steps as needed.

User Question:
{original_question}

Answer References:
{combined_input}

Final helpful answer:
"""
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": gpt_prompt}],
                temperature=0.3,
            )
            final_answer = gpt_response.choices[0].message.content.strip()
    except Exception:
        final_answer = first_match_answer or "[No answer found]"

    final_answer += "\n\nIf this didn’t resolve the issue, let me know."
    th_state["conversation_history"].append({"role": "user", "content": user_question})
    th_state["conversation_history"].append({"role": "assistant", "content": final_answer})
    return final_answer
