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
import math
import logging
from collections import OrderedDict
import time

# Configure logging - use INFO level for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = dotenv_values(".env")
openai_key = st.secrets.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY"))
pinecone_key = st.secrets.get("PINECONE_API_KEY", config.get("PINECONE_API_KEY"))

if not openai_key or not pinecone_key:
    raise ValueError("❌ Missing OPENAI_API_KEY or PINECONE_API_KEY")

embedding_model = "text-embedding-3-small"
CONFIDENCE_THRESHOLD = 0.6
ESCALATION_RESPONSE = "I wasn't able to find any additional information. I'm escalating this to our support team so they can follow up with you directly."

client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("sweetrobo-ai")

# LRU Cache implementation for embeddings
class LRUCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()

# Initialize thread-safe session state
def get_session_state():
    """Get or initialize session state for thread safety"""
    if 'th_state' not in st.session_state:
        st.session_state.th_state = {
            "thread_id": None,
            "machine_type": None,
            "used_matches_by_thread": {},
            "conversation_history": [],
            "solution_attempts": {},
            "embedding_cache": LRUCache(max_size=100),
            "last_query_time": 0,
            "query_count": 0
        }
    return st.session_state.th_state

# For backward compatibility
th_state = get_session_state()

def get_or_create_embedding(text, cache_key=None):
    """Get embedding from cache or create new one. Reduces API calls significantly."""
    th_state = get_session_state()
    
    if cache_key:
        cached = th_state["embedding_cache"].get(cache_key)
        if cached is not None:
            return cached
    
    try:
        response = client.embeddings.create(model=embedding_model, input=[text])
        embedding = response.data[0].embedding
        
        if cache_key:
            th_state["embedding_cache"].put(cache_key, embedding)
        
        return embedding
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        return None

def track_solution_failure():
    th_state = get_session_state()
    thread_id = th_state["thread_id"]
    th_state["solution_attempts"][thread_id] = (
        th_state["solution_attempts"].get(thread_id, 0) + 1
    )
    return th_state["solution_attempts"][thread_id]

def initialize_chat(selected_machine: str):
    th_state = get_session_state()
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
    th_state["query_count"] = 0  # Reset query count for new session
    return {"thread_id": th_state["thread_id"], "machine_type": machine_type}

def is_question_too_vague(user_q_lower):
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
        'assistance', 'production', 'getting', 'inside', 'wifi', 'WiFi'
    }
    
    # Check for exact matches first (fastest)
    if any(word in user_q_lower for word in specific_keywords):
        return False
    
    # Check for fuzzy matches to handle typos
    from difflib import SequenceMatcher
    words_in_query = user_q_lower.split()
    
    for word in words_in_query:
        if len(word) < 3:  # Skip very short words like "a", "is", "to"
            continue
        for keyword in specific_keywords:
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, word, keyword).ratio()
            # Use 0.80 threshold for typo tolerance (e.g., "errro" matches "error" at 0.8)
            if similarity >= 0.80:
                logger.debug(f"Fuzzy match found: '{word}' matches '{keyword}' (similarity: {similarity:.2f})")
                return False
    
    return True  # No matches found, question is too vague

def is_already_given(answer, history):
    try:
        th_state = get_session_state()
        answer_cache_key = f"answer_{hash(answer)}"
        answer_emb = get_or_create_embedding(answer, answer_cache_key)
        
        if answer_emb is None:
            return False

        for entry in history:
            if entry["role"] == "assistant":
                hist_emb = entry.get("embedding")
                
                if hist_emb is None:
                    continue

                similarity = cosine_similarity(answer_emb, hist_emb)
                if similarity >= 0.90:
                    logger.debug(f"Answer already given (cosine sim: {similarity:.3f})")
                    return True
        return False
    except Exception as e:
        logger.error(f"is_already_given cosine check failed: {e}")
        return False


def build_followup_query(user_q: str, original_q: str):
    """Returns enriched question and whether enrichment was applied."""
    cleaned = user_q.strip()
    if len(cleaned.split()) <= 4:
        return f"{cleaned} — referring to: {original_q.strip()}", True
    return cleaned, False

def process_match_for_followup(match, user_q_lower, seen_ids):
    th_state = get_session_state()
    if match.id in seen_ids:
        return None, None

    metadata = match.metadata
    answer = metadata.get("a", "[No A]").strip()
    if not answer:
        return None, None

    q = metadata.get("q", "")

    # Check if answer was already given
    already_given = is_already_given(answer, th_state["conversation_history"])

    logger.debug(f"FOLLOW-UP FILTER - Q: {q[:50] if q else 'N/A'}... Already Given: {already_given}")

    if not already_given:
        return answer, match.id
    return None, None

def handle_followup_with_existing_matches(user_q, thread_id):
    th_state = get_session_state()
    match_pointer = th_state.get("match_pointer", 1)
    used_info = th_state["used_matches_by_thread"][thread_id]
    all_matches = used_info["all_matches"]
    original_question = used_info["original_question"]
    
    logger.info(f"Starting follow-up match filtering - Match pointer: {match_pointer}, Total matches: {len(all_matches)}")
    
    # No matches left to try — escalate immediately
    if match_pointer >= len(all_matches):
        logger.warning("All follow-up matches exhausted. Escalating.")
        return ESCALATION_RESPONSE

    user_q_lower = user_q.lower()
    query_for_related, used_enriched = build_followup_query(user_q, original_question)
    filtered_qas = []
    seen_ids = set()

    # Pass 1: try with chosen query (raw follow-up OR enriched)
    for i in range(match_pointer, len(all_matches)):
        match, usefulness, confidence = all_matches[i]
        answer, match_id = process_match_for_followup(match, user_q_lower, seen_ids)
        if answer:
            th_state["match_pointer"] = i + 1
            filtered_qas.append(answer)
            seen_ids.add(match_id)

    # Pass 2 removed - it was redundant and never found new matches

    # Nothing found after both passes -> clarification or escalation
    if not filtered_qas:
        logger.warning("No valid follow-up matches found. Entering fallback mode.")
        failure_count = track_solution_failure()
        if failure_count == 1:
            return ("Thanks for letting me know. Could you describe exactly what didn't work "
                    "or what's still happening (e.g., error still shows, part stuck, no heat)?")
        if failure_count >= 2:
            return "This seems persistent. Escalating to a human agent now. Please wait..."
        return "Sorry, I couldn't find any new helpful info for this issue. Escalating this to our support team. Please hold on."

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
You are a helpful AI assistant for customer support. The user said the initial solution didn't work.

You are given up to 5 technical answers. Your job is to summarize only the most helpful 1–3 suggestions.

Instructions:
- Provide only the steps needed to address the issue (1 to 5 max).
- Use bullet points (•), not numbers.
- Be concise and do not repeat instructions.
- Do not say "if that doesn't work" — that will be appended later.

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

    final_answer += "\n\nIf this didn't resolve the issue, let me know."
    
    # Cache embeddings with conversation history
    assistant_embedding = get_or_create_embedding(final_answer, f"hist_{thread_id}_{len(th_state['conversation_history'])}")
    
    th_state["conversation_history"].append({"role": "user", "content": user_q})
    th_state["conversation_history"].append({"role": "assistant", "content": final_answer, "embedding": assistant_embedding})
    return final_answer


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
        logger.debug(f"GPT Topic Check: '{candidate_q[:50]}...' → {result}")
        return "yes" in result
    except Exception as e:
        logger.error(f"GPT topic check failed: {e}")
        return False

def fetch_valid_matches(query_embedding, previous_ids, error_code_filter, query_text):
    th_state = get_session_state()
    filter_query = {"machine_type": {"$eq": th_state["machine_type"]}}
    
    # Only filter by error code if user mentioned a specific code
    # Otherwise, include ALL Q&As (with or without error codes)
    if error_code_filter:
        filter_query["error_codes"] = {"$in": [str(error_code_filter)]}

    all_valid = []
    seen_answers = set()
    query_text_lower = query_text.lower()

    results = index.query(
        namespace="sweetrobo-v2",
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        include_values=True,
        filter=filter_query,
    )
    
    logger.info(f"Pinecone query for: '{query_text[:60]}...'")
    logger.info(f"Pinecone returned {len(results.matches)} matches")
    if results.matches:
        logger.info(f"Top 3 Pinecone results:")
        for i, match in enumerate(results.matches[:3]):
            logger.info(f"  {i+1}. Score: {match.score:.3f}, Q: '{match.metadata.get('q', '')[:60]}...'")

    for match in results.matches:
        if match.id in previous_ids:
            continue

        if match.score < 0.25:
            continue  # ❌ Skip bad similarity early — don't even evaluate

        metadata = match.metadata
        usefulness_score = metadata.get("usefulness_score", 0)
        confidence = metadata.get("confidence", 0.0)
        answer = metadata.get("a", "")
        tags = metadata.get("tags", [])
        candidate_q = metadata.get("q", "")
        
        # Debug logging for filtering process
        if match.score >= 0.5:  # Log high-scoring matches to see why they're filtered
            logger.info(f"Checking match (score {match.score:.3f}): usefulness={usefulness_score}, confidence={confidence:.2f}, Q='{candidate_q[:50]}...'")

        if usefulness_score >= 7 and confidence >= CONFIDENCE_THRESHOLD:
            if isinstance(match.values, list) and len(match.values) == 1536:
                if answer not in seen_answers:
                    logger.info(f"✓ Match accepted - Cosine: {match.score:.3f}, Usefulness: {usefulness_score}, Confidence: {confidence}")
                    all_valid.append((match, usefulness_score, confidence))
                    seen_answers.add(answer)
            else:
                if match.score >= 0.5:
                    logger.warning(f"✗ Match rejected - Invalid embedding dimensions or format")
        else:
            if match.score >= 0.5:
                logger.warning(f"✗ Match rejected - Failed filters: usefulness={usefulness_score} (need >=7), confidence={confidence:.2f} (need >=0.6)")

    # Sort by cosine similarity (best semantic match first), not usefulness score
    # The match.score is the actual relevance to the user's query
    all_valid.sort(key=lambda x: -x[0].score)
    
    logger.info(f"Found {len(all_valid)} matches passing initial filters")
    if all_valid:
        for i, (match, score, conf) in enumerate(all_valid[:3]):
            logger.info(f"  Pre-GPT #{i+1}: '{match.metadata.get('q', '')[:60]}...'")

    # GPT topic match check — only top 5 to minimize cost and latency
    filtered_with_gpt = []
    for match, score, confidence in all_valid[:5]:  # Reduced from 10 to 5
        q = match.metadata.get("q", "")
        if is_same_topic(query_text, q):
            filtered_with_gpt.append((match, score, confidence))
            logger.info(f"✓ GPT approved: '{q[:50]}...'")
        else:
            logger.info(f"✗ GPT rejected: '{q[:50]}...'")

    # Calculate complexity penalty for each match
    # Simpler, more direct questions should rank higher
    def calculate_match_score(item):
        match, usefulness, confidence = item
        q = match.metadata.get("q", "").lower()
        user_q_lower = query_text.lower()
        
        # Base score is cosine similarity
        base_score = match.score
        
        # Penalize questions with extra conditions not in user query
        complexity_penalty = 0
        
        # Check for extra conditions (words in candidate but not in user query)
        user_words = set(user_q_lower.split())
        candidate_words = set(q.split())
        
        # Common connecting words to ignore
        ignore_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 
                       'what', 'how', 'when', 'where', 'why', 'should', 'i', 'my', 'on', 
                       'if', 'to', 'and', 'or', 'but', 'get', 'getting', 'have', 'has'}
        
        user_content_words = user_words - ignore_words
        candidate_content_words = candidate_words - ignore_words
        
        # Words in candidate but not in user query (extra conditions)
        extra_words = candidate_content_words - user_content_words
        
        # Dynamic penalty based on extra words count
        # Each significant extra word adds penalty
        significant_extras = {
            'hangs', 'hanging', 'frozen', 'freezes', 'stuck', 'crashes', 'screen',
            'making', 'spinning', 'loading', 'unresponsive', 'slow', 'delay'
        }
        
        penalty_words = extra_words & significant_extras
        if penalty_words:
            complexity_penalty += len(penalty_words) * 0.05  # 0.05 per extra condition word
        
        # General penalty for question length (prefer concise matches)
        if len(q) > len(user_q_lower) * 2:
            complexity_penalty += 0.05
        
        # Boost for very similar question length
        length_ratio = len(user_q_lower) / len(q) if q else 0
        if 0.7 <= length_ratio <= 1.3:
            complexity_penalty -= 0.05  # Bonus for similar length
        
        final_score = base_score - complexity_penalty
        logger.debug(f"Match scoring - Q: '{q[:60]}...' Base: {base_score:.3f}, Penalty: {complexity_penalty:.3f}, Final: {final_score:.3f}")
        
        return final_score

    # Sort by calculated match score (considers both similarity and complexity)
    filtered_with_gpt.sort(key=lambda x: -calculate_match_score(x))
    return filtered_with_gpt[:5]

def is_followup_message(user_q_lower):
    followup_phrases = {
        "didn't work", "didnt work", "didn't help", "didnt help",
        "still broken", "not fixed", "didn't resolve", "not working",
        "it did not resolve", "it did not work", "still not fixed",
        "that didn't help", "that did not work", "this didn't help",
        "not resolved", "didn't fix", "didnt fix", "didn't solve",
        "wasn't fixed", "this didn't fix", "that didn't fix", "what if", "still"
    }
    normalized = user_q_lower.replace("'", "'")

    if any(p in normalized for p in followup_phrases):
        return True

    # Fallback: ask GPT if it's a follow-up intent
    prompt = f"""You're a support assistant. Determine if the following user message is a follow-up complaint — meaning that a previous solution attempt did not work and the user is still seeking help.

Respond ONLY with "yes" or "no".

User message:
"{user_q_lower}"
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
    th_state = get_session_state()
    
    # Rate limiting - max 10 queries per minute
    current_time = time.time()
    if th_state.get("last_query_time", 0) > 0:
        time_diff = current_time - th_state["last_query_time"]
        if time_diff < 6:  # Less than 6 seconds between queries
            th_state["query_count"] += 1
            if th_state["query_count"] > 10:
                return "⚠️ Too many requests. Please wait a moment before asking another question."
        else:
            th_state["query_count"] = 1  # Reset counter after 6 seconds
    
    th_state["last_query_time"] = current_time
    
    thread_id = th_state["thread_id"]
    used_matches_by_thread = th_state["used_matches_by_thread"]
    
    if not thread_id:
        return "⚠️ Please start a new support session by selecting the machine first."

    # Normalize short numeric input like "4012"
    if re.fullmatch(r"\d{4,}", user_question):
        user_question = f"how to fix error {user_question}"
    
    user_question_lower = user_question.lower()
    
    # Check for follow-up intent
    is_followup = len(th_state["conversation_history"]) > 0 and is_followup_message(user_question_lower)
        
    # Check for vague question (clarification fallback)
    if not is_followup and is_question_too_vague(user_question_lower):
        return ("That's a bit too general. Could you describe exactly what's going wrong "
                "(e.g., error code, what part is malfunctioning, or what's not working as expected)?")

    logger.info(f"Processing query - Thread: {th_state['thread_id']}, Is follow-up: {is_followup}")

    # If it's a follow-up, reuse prior results
    if is_followup:
        if thread_id in used_matches_by_thread and "all_matches" in used_matches_by_thread[thread_id]:
            return handle_followup_with_existing_matches(user_question, thread_id)
        else:
            return "I'm still trying to find the best solution. Could you restate the issue in more detail?"

    # First-time question: embed and fetch matches with caching
    query_cache_key = f"query_{thread_id}_{hash(user_question)}"
    query_embedding = get_or_create_embedding(user_question, query_cache_key)
    
    if query_embedding is None:
        logger.error("Failed to create query embedding")
        return "Sorry, I encountered an error processing your question. Please try again."
    
    previous_ids = set()
    match = re.search(r"(\d{4,})", user_question)
    error_code_filter = int(match.group(1)) if "error" in user_question_lower and match else None

    top_matches = fetch_valid_matches(query_embedding, previous_ids, error_code_filter, user_question)

    # Save matches for future follow-up
    th_state["used_matches_by_thread"][thread_id] = {
        "embedding": query_embedding,
        "error_filter": error_code_filter,
        "original_question": user_question,
        "all_matches": top_matches,
    }
    th_state["match_pointer"] = 1  # Start at second match for follow-up


    # No valid matches
    if not top_matches:
        logger.warning("No top matches found after similarity filtering")
        return "Sorry, I couldn't find a helpful answer. Can you rephrase the question with more details?"

    # Present the best GPT-filtered match (highest cosine)
    if top_matches:
        best_match = top_matches[0]  # Already sorted in GPT filter
        raw_answer = best_match[0].metadata.get("a", "[No A]").strip()

    final_answer = raw_answer.strip()
    logger.info(f"Answer selected after GPT topic filtering: {final_answer[:80]}...")

    # Append to conversation history with embedding
    final_answer += "\n\nIf this didn't resolve the issue, let me know."
    
    # Cache assistant response embedding
    assistant_embedding = get_or_create_embedding(final_answer, f"hist_{thread_id}_{len(th_state['conversation_history'])}")
    
    th_state["conversation_history"].extend([
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": final_answer, "embedding": assistant_embedding}
    ])
    return final_answer