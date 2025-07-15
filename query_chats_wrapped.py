import os
import streamlit as st
from dotenv import dotenv_values
import json
import uuid
from datetime import datetime
from openai import OpenAI
import pinecone
import numpy as np
import re

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
pinecone.init(api_key=pinecone_key, environment="us-east1-gcp")
index = pinecone.Index("sweetrobo-ai")

# === Global Chat State ===
th_state = {
    "thread_id": None,
    "machine_type": None,
    "used_matches_by_thread": {},
    "conversation_history": []
}

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

    return {
        "thread_id": th_state["thread_id"],
        "machine_type": machine_type
    }

def is_question_too_vague(user_q):
    prompt = f"""You are a support assistant. You will be given a user's message.

You must check if the message matches any known vague expressions from the list below. These expressions include messages that are too general, do not mention any part, symptom, or error code, and cannot be acted on without clarification.

- my machine is not working
- nothing is happening
- i need help
- it’s broken
- i’m having an issue
- something’s wrong
- please assist
- it’s not turning on
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

User message: "{user_q}"

Respond only with:
- "yes"
- "no"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return "yes" in response.choices[0].message.content.lower()
    except Exception:
        return False

def is_related(user_q, match_q, match_a):
    user_words = set(re.findall(r"\w+", user_q.lower()))
    match_words = set(re.findall(r"\w+", (match_q + " " + match_a).lower()))
    overlap = user_words.intersection(match_words)
    return len(overlap) / max(len(user_words), 1) > 0.3

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
            filter=filter_query
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

    return all_valid[:5]

def run_chatbot_session(user_question: str) -> str:
    thread_id = th_state["thread_id"]
    used_matches_by_thread = th_state["used_matches_by_thread"]

    if re.fullmatch(r"\d{4,}", user_question):
        user_question = f"how to fix error {user_question}"

    if is_question_too_vague(user_question):
        return ("That’s a bit too general. Could you describe exactly what’s going wrong "
                "(e.g., error code, what part is malfunctioning, or what’s not working as expected)?")

    is_followup = user_question.lower() in {"it did not resolve", "didn't work", "still broken", "not fixed"}

    if is_followup and thread_id in used_matches_by_thread:
        previous = used_matches_by_thread[thread_id]
        query_embedding = previous["embedding"]
        previous_ids = previous["used_ids"]
        error_code_filter = previous["error_filter"]
        original_question = previous["original_question"]
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
        "original_question": original_question
    }

    filtered_qas = [
        m[0].metadata.get("a", "[No A]")
        for m in top_matches
        if is_related(original_question, m[0].metadata.get("q", ""), m[0].metadata.get("a", ""))
    ][:5]

    if not filtered_qas:
        filtered_qas = [top_matches[0][0].metadata.get("a", "[No A]")]

    combined_input = "\n\n".join(filtered_qas)
    first_match_answer = top_matches[0][0].metadata.get("a", "").strip()

    simple_q_phrases = ["can i", "do you", "does it", "is there", "are there", "can we", "is it possible"]
    is_simple_question = any(original_question.lower().startswith(p) for p in simple_q_phrases)
    is_short_answer = len(first_match_answer.split()) <= 12 and first_match_answer.lower().startswith(("yes", "no", "sorry", "unfortunately"))

    if is_simple_question and is_short_answer:
        final_answer = first_match_answer
    else:
        gpt_prompt = f"""
You are a helpful AI assistant summarizing customer support answers.

You are given up to 5 answers related to the same issue. Also examine the user's original question to determine if it can be answered simply with a short response. If the first answer is a high-confidence, clearly phrased yes/no or availability response and it directly addresses the question, return that answer directly with no summary.

Otherwise, summarize the full content of the answers. Do not reference brands, machines, or quote user questions. Combine similar advice, and if multiple paths exist, say: "If that doesn't work, you can also try..." Keep it precise and under 4 sentences.

User Question:
{original_question}

Answer References:
{combined_input}

Final helpful answer:
"""
        try:
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": gpt_prompt}],
                temperature=0.4
            )
            final_answer = gpt_response.choices[0].message.content.strip()
        except Exception:
            final_answer = first_match_answer or "[No answer found]"

    final_answer += "\n\nIf this didn’t resolve the issue, let me know."
    th_state["conversation_history"].append({"role": "user", "content": user_question})
    th_state["conversation_history"].append({"role": "assistant", "content": final_answer})
    return final_answer
