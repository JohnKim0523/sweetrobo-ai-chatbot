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
import math

config = dotenv_values(".env")
openai_key = st.secrets.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY"))
pinecone_key = st.secrets.get("PINECONE_API_KEY", config.get("PINECONE_API_KEY"))

if not openai_key or not pinecone_key:
    raise ValueError("❌ Missing OPENAI_API_KEY or PINECONE_API_KEY")

embedding_model = "text-embedding-3-small"
CONFIDENCE_THRESHOLD = 0.6
ESCALATION_RESPONSE = "I wasn’t able to find any additional information. I’m escalating this to our support team so they can follow up with you directly."

client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("sweetrobo-ai")

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
- "yes"
- "no"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return "yes" in response.choices[0].message.content.lower()
    except Exception:
        return False

def bulletify_if_long(answer):
    parts = re.split(r'(?<=\.)\s+', answer.strip())
    if len(parts) < 2:
        return answer
    bullets = [f"• {p.strip()}" for p in parts if p.strip()]
    return "\n\n".join(bullets)

def is_related(user_q, match_q, match_a):
    full_text = match_q + " " + match_a
    ratio = SequenceMatcher(None, user_q.lower(), full_text.lower()).ratio()
    
    # Debug output
    print(f"🔗 [is_related] Similarity between user_q and match Q+A:")
    print(f"   - User Q:    {user_q}")
    print(f"   - Match Q:   {match_q}")
    print(f"   - Match A:   {match_a[:100]}...") 
    print(f"   - Similarity Ratio: {ratio:.3f}")
    print(f"   - Pass? {'✅' if ratio > 0.3 else '❌'}\n")
    
    return ratio > 0.05  # Can be tuned

def is_already_given(answer, history, threshold=0.85):
    for entry in history:
        if entry["role"] == "assistant":
            prev_ans = entry["content"]
            if SequenceMatcher(None, answer.strip(), prev_ans.strip()).ratio() >= threshold:
                return True
    return False

def build_followup_query(user_q: str, original_q: str):
    """
    Returns enriched question and whether enrichment was applied.
    """
    cleaned = user_q.strip().lower()
    if len(cleaned.split()) <= 4:
        return f"{user_q.strip()} — referring to: {original_q.strip()}", True
    return user_q.strip(), False

def handle_followup_with_existing_matches(user_q, thread_id):
    match_pointer = th_state.get("match_pointer", 1)
    used_info = th_state["used_matches_by_thread"][thread_id]
    all_matches = used_info["all_matches"]
    original_question = used_info["original_question"]
    
    print(f"🔁 [DEBUG] Starting follow-up match filtering...")
    print(f"🔢 Match pointer: {match_pointer}")
    print(f"📊 Total matches available: {len(all_matches)}")
    
    # No matches left to try — escalate immediately
    if match_pointer >= len(all_matches):
        print("⚠️ All follow-up matches have been exhausted. Escalating.")
        return ESCALATION_RESPONSE

    # If follow up and vague, enrich question with original question
    query_for_related, used_enriched = build_followup_query(user_q, original_question)

    filtered_qas = []
    seen_ids = set()

    # Pass 1: try with chosen query (raw follow-up OR enriched, depending on heuristic)
    for i in range(match_pointer, len(all_matches)):
        match, usefulness, confidence = all_matches[i]
        answer = match.metadata.get("a", "[No A]").strip()
        if not answer:
            continue
        
        print(f"\n🔍 [DEBUG] Checking match index {i}")
        print(f"🆔 Match ID: {match.id}")
        print(f"❓ Q: {match.metadata.get('q', '')}")
        print(f"💬 A: {answer[:100]}...")

        q = match.metadata.get("q", "")
        a = match.metadata.get("a", "")

        related = is_related(query_for_related, q, a)
        already_given = is_already_given(a, th_state["conversation_history"])
        
        print(f"🔗 Related? {related}")
        print(f"🧠 Already Given? {already_given}")

        print(f"🧪 FOLLOW-UP FILTER (pass1 {'enriched' if used_enriched else 'raw'}):")
        print(f"- Related: {related}")
        print(f"- Already Given: {already_given}")
        print(f"- Q: {q}")
        print(f"- A: {a[:80]}...\n")

        if related and not already_given:
            th_state["match_pointer"] = i + 1  # advance pointer
            filtered_qas.append(a)
            seen_ids.add(match.id)

    # Pass 2 (fallback): if we *did not* enrich on pass1 AND got nothing, try enriched
    if not filtered_qas and not used_enriched:
        enriched_again, _ = build_followup_query("", original_question)  # force enrich
        for i in range(match_pointer, len(all_matches)):
            match, usefulness, confidence = all_matches[i]
            if match.id in seen_ids:
                continue
            a = match.metadata.get("a", "[No A]").strip()
            if not a:
                continue
            q = match.metadata.get("q", "")
            related = is_related(enriched_again, q, a)
            already_given = is_already_given(a, th_state["conversation_history"])

            print(f"🧪 FOLLOW-UP FILTER (pass2 enriched):")
            print(f"- Related: {related}")
            print(f"- Already Given: {already_given}")
            print(f"- Q: {q}")
            print(f"- A: {a[:80]}...\n")

            if related and not already_given:
                th_state["match_pointer"] = i + 1
                filtered_qas.append(a)
                seen_ids.add(match.id)

    # Nothing found after both passes -> clarification or escalation
    if not filtered_qas:
        print("⚠️ No valid follow-up matches found. Entering fallback mode.")
        failure_count = track_solution_failure()
        if failure_count == 1:
            return ("Thanks for letting me know. Could you describe exactly what didn’t work "
                    "or what’s still happening (e.g., error still shows, part stuck, no heat)?")
        if failure_count >= 2:
            return "This seems persistent. Escalating to a human agent now. Please wait..."
        # fallback safety
        return "Sorry, I couldn’t find any new helpful info for this issue. Escalating this to our support team. Please hold on."

    # Summarize / return one or more answers
    if len(filtered_qas) == 1:
        final_answer = filtered_qas[0]
    else:
        combined_input = "\n\n".join(filtered_qas)
        simple_q_phrases = ["can i", "do you", "does it", "is there", "are there", "can we", "is it possible"]
        is_simple_question = any(original_question.lower().startswith(p) for p in simple_q_phrases)
        is_short_answer = len(filtered_qas[0].split()) <= 12 and filtered_qas[0].lower().startswith(("yes", "no", "sorry", "unfortunately"))
        try:
            if is_simple_question and is_short_answer:
                final_answer = filtered_qas[0]
            else:
                gpt_prompt = f"""
You are a helpful AI assistant for customer support. The user said the initial solution didn’t work.

You are given up to 5 technical answers. Your job is to summarize only the most helpful 1–3 suggestions.

Instructions:
- Provide only the steps needed to address the issue (1 to 5 max).
- Use bullet points (•), not numbers.
- Be concise and do not repeat instructions.
- Do not say “if that doesn’t work” — that will be appended later.

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
                final_answer = final_answer.replace("•", "\n\n•")
        except Exception:
            final_answer = "Sorry, no answer was found. Escalating to our support team now."

    final_answer += "\n\nIf this didn’t resolve the issue, let me know."
    th_state["conversation_history"].append({"role": "user", "content": user_q})
    th_state["conversation_history"].append({"role": "assistant", "content": final_answer})
    return final_answer

def get_question_similarity_boost(user_q, candidate_q):
    ratio = SequenceMatcher(None, user_q.lower(), candidate_q.lower()).ratio()
    if ratio >= 0.95:
        return 0.5
    elif ratio >= 0.85:
        return 0.3
    elif ratio >= 0.7:
        return 0.1
    return 0

# Cosine similarity
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2 + 1e-10)

def is_same_topic(user_q, candidate_q):
    prompt = f"""You are helping match a user's support question with known help topics. Consider questions to be "the same" if they are about the same real-world issue, even if the wording is different.

Are these two questions about the same issue?

User: "{user_q}"
Candidate: "{candidate_q}"

Respond only with "yes" or "no".
"""    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        print(f"🧠 GPT Topic Check: '{candidate_q}' → {result}")
        return "yes" in result
    except Exception as e:
        print(f"⚠️ GPT topic check failed: {e}")
        return False

def fetch_valid_matches(query_embedding, previous_ids, error_code_filter, query_text):
    filter_query = {"machine_type": {"$eq": th_state["machine_type"]}}
    if error_code_filter:
        filter_query["error_codes"] = {"$in": [str(error_code_filter)]}
    else:
        filter_query["error_codes"] = {"$exists": False}

    all_valid = []
    seen_answers = set()
    keyword_focus = "stick" if "stick" in query_text.lower() else None

    results = index.query(
        vector=query_embedding,
        top_k=20,
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
        candidate_q = meta.get("q", "")

        if usefulness_score >= 7 and confidence >= CONFIDENCE_THRESHOLD:
            if keyword_focus and not any(keyword_focus in t.lower() for t in tags):
                continue
            if isinstance(match.values, list) and len(match.values) == 1536:
                if answer not in seen_answers:
                    # Compute embedding similarity to candidate_q
                    response = client.embeddings.create(model=embedding_model, input=[candidate_q])
                    candidate_q_embedding = response.data[0].embedding
                    similarity_score = cosine_similarity(query_embedding, candidate_q_embedding)

                    print(f"🎯 Cosine Sim = {similarity_score:.3f} between:\n→ User: {query_text}\n→ Q:    {candidate_q}\n")

                    if similarity_score < 0.50:
                        continue  # ❌ Skip irrelevant Qs

                    boost = get_question_similarity_boost(query_text, candidate_q)
                    boosted_score = usefulness_score + boost
                    all_valid.append((match, boosted_score, confidence))

                    # Debug Q
                    print(f"✅ Matched Q: {candidate_q}\n→ Boosted score: {boosted_score:.2f} (Usefulness: {usefulness_score}, Boost: {boost:.2f}, Confidence: {confidence:.2f})\n")

                    seen_answers.add(answer)

    all_valid.sort(key=lambda x: (-x[1], -x[2]))  # Sort by boosted_score, then confidence

    # Apply GPT topic check to top 10
    filtered_with_gpt = []
    for match, boosted_score, confidence in all_valid[:10]:  # Only check top 10 to keep it fast
        q = match.metadata.get("q", "")
        if is_same_topic(query_text, q):
            filtered_with_gpt.append((match, boosted_score, confidence))
        else:
            print(f"🚫 Rejected by GPT topic match: {q}")

    filtered_with_gpt.sort(key=lambda x: -x[0].score)  # Sort by cosine similarity DESC
    return filtered_with_gpt[:5]

def is_followup_message(user_q):
    followup_phrases = {
        "didn't work", "didnt work", "didn't help", "didnt help",
        "still broken", "not fixed", "didn’t resolve", "not working",
        "it did not resolve", "it did not work", "still not fixed",
        "that didn’t help", "that did not work", "this didn’t help",
        "not resolved", "didn’t fix", "didnt fix", "didn’t solve",
        "wasn’t fixed", "this didn’t fix", "that didn’t fix"
    }
    normalized = user_q.lower().replace("’", "'").strip()

    if any(p in normalized for p in followup_phrases):
        return True

    # Fallback: ask GPT if it's a follow-up intent
    prompt = f"""You're a support assistant. Determine if the following user message is a follow-up complaint — meaning that a previous solution attempt did not work and the user is still seeking help.

Respond ONLY with "yes" or "no".

User message:
"{user_q}"
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

def run_chatbot_session(user_question: str) -> str:
    email_response = handle_email_logic(user_question)
    if email_response:
        return email_response

    thread_id = th_state["thread_id"]
    used_matches_by_thread = th_state["used_matches_by_thread"]
    
    if not thread_id:
        return "⚠️ Please start a new support session by selecting the machine first."

    # Normalize short numeric input like "4012"
    if re.fullmatch(r"\d{4,}", user_question):
        user_question = f"how to fix error {user_question}"
        
    # Check for follow-up intent
    if len(th_state["conversation_history"]) == 0:
        is_followup = False  # First message of chat can never be follow-up
    else:
        is_followup = is_followup_message(user_question)
        
    # Check for vague question (clarification fallback)
    if not is_followup and is_question_too_vague(user_question):
        return ("That’s a bit too general. Could you describe exactly what’s going wrong "
                "(e.g., error code, what part is malfunctioning, or what’s not working as expected)?")

    print(f"🧵 thread_id: {th_state['thread_id']}")
    print(f"📁 has_all_matches: {'all_matches' in used_matches_by_thread.get(th_state['thread_id'], {})}")
    print(f"🔁 is_followup: {is_followup}")

    # If it's a follow-up, reuse prior results
    if is_followup:
        if thread_id in used_matches_by_thread and "all_matches" in used_matches_by_thread[thread_id]:
            return handle_followup_with_existing_matches(user_question, thread_id)
        else:
            return "I'm still trying to find the best solution. Could you restate the issue in more detail?"

    # First-time question: embed and fetch matches
    response = client.embeddings.create(model=embedding_model, input=[user_question])
    query_embedding = response.data[0].embedding
    previous_ids = set()
    original_question = user_question
    match = re.search(r"(\d{4,})", user_question)
    error_code_filter = int(match.group(1)) if "error" in user_question.lower() and match else None

    top_matches = fetch_valid_matches(query_embedding, previous_ids, error_code_filter, original_question)

    # Save matches for future follow-up
    th_state["used_matches_by_thread"][thread_id] = {
        "embedding": query_embedding,
        "error_filter": error_code_filter,
        "original_question": original_question,
        "all_matches": top_matches,
    }
    th_state["match_pointer"] = 1  # Start at second match for follow-up

    # Optional: filter very low-similarity matches
    filtered_top = []
    for match, boosted_score, confidence in top_matches:
        cosine_sim = match.score
        q_text = match.metadata.get("q", "")
        a_text = match.metadata.get("a", "[No A]").strip()

        print(f"🎯 Cosine Sim = {cosine_sim:.3f} between:")
        print(f"→ User: {user_question}")
        print(f"→ Q:    {q_text}")
        print(f"→ A:    {a_text[:120]}...\n")
        if cosine_sim < 0.4:
            break
        filtered_top.append((match, boosted_score, confidence))
    top_matches = filtered_top

    # No valid matches
    if not top_matches:
        print("❌ No top matches found after similarity filtering.")
        return "Sorry, I couldn’t find a helpful answer. Can you rephrase the question with more details?"

    # Present the best GPT-filtered match (highest cosine)
    if top_matches:
        best_match = top_matches[0]  # Already sorted in GPT filter
        raw_answer = best_match[0].metadata.get("a", "[No A]").strip()

    # Bulletify if long
    final_answer = bulletify_if_long(raw_answer)

    # ⬇Add visible spacing if the answer starts with bullet points
    if final_answer.strip().startswith("•"):
        final_answer = "\n\n" + final_answer
        print(f"✅ Bulletified answer selected after GPT topic filtering: {raw_answer}")
    else:
        print(f"✅ Short answer selected after GPT topic filtering: {raw_answer}")

    # Append to conversation history
    final_answer += "\n\nIf this didn’t resolve the issue, let me know."
    th_state["conversation_history"].append({"role": "user", "content": user_question})
    th_state["conversation_history"].append({"role": "assistant", "content": final_answer})
    return final_answer