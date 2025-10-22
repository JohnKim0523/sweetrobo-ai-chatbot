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

# NEW: Import improvement modules
from context_manager import ConversationContextManager
from topic_matcher import is_same_topic_fast
from response_builder import ResponseBuilder
from analytics import get_analytics
from hybrid_search import get_hybrid_search

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

# Load video library
VIDEO_LIBRARY = []
try:
    with open('video_library.json', 'r', encoding='utf-8') as f:
        VIDEO_LIBRARY = json.load(f)
    logger.info(f"✓ Loaded {len(VIDEO_LIBRARY)} videos from library")
except FileNotFoundError:
    logger.warning("⚠️ video_library.json not found - video features disabled")
except Exception as e:
    logger.error(f"Failed to load video library: {e}")

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
            "query_count": 0,
            # NEW: Add improvement modules
            "context_manager": ConversationContextManager(),
            "response_builder": ResponseBuilder(),
            "analytics": get_analytics(),
            "hybrid_search": get_hybrid_search(),
            # NEW: Track suggested actions for solution-level deduplication
            "suggested_actions_by_thread": {}
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
    
    # PRODUCTION FIX 2: Retry logic for API failures
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=embedding_model, input=[text])
            embedding = response.data[0].embedding
            
            if cache_key:
                th_state["embedding_cache"].put(cache_key, embedding)
            
            return embedding
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Embedding creation failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

def track_solution_failure():
    th_state = get_session_state()
    thread_id = th_state["thread_id"]
    th_state["solution_attempts"][thread_id] = (
        th_state["solution_attempts"].get(thread_id, 0) + 1
    )
    return th_state["solution_attempts"][thread_id]

def extract_key_actions(answer_text):
    """
    Extract key troubleshooting actions from an answer.
    Returns a list of action phrases like ["clean nozzle", "check water level", "restart machine"]
    """
    prompt = f"""Extract the key troubleshooting actions from this technical support answer.

Answer:
"{answer_text}"

Extract ONLY the main action verbs + objects (e.g., "clean nozzle", "check water level", "restart machine", "replace part").

Return a comma-separated list of 2-6 key actions. If there are similar actions (like "clean nozzle with wire brush" and "clean nozzle with metal brush"), combine them into one action.

Format: action1, action2, action3

Examples:
- "clean nozzle, check water level, restart machine"
- "replace filter, test heating, verify power"
- "reset settings, clear cache, update firmware"

Key actions:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        actions_str = response.choices[0].message.content.strip()

        # Parse comma-separated actions
        actions = [a.strip().lower() for a in actions_str.split(',') if a.strip()]

        logger.info(f"Extracted actions: {actions}")
        return actions
    except Exception as e:
        logger.error(f"Failed to extract key actions: {e}")
        # Fallback: simple keyword extraction
        keywords = []
        answer_lower = answer_text.lower()
        if 'clean' in answer_lower or 'brush' in answer_lower:
            keywords.append('clean nozzle')
        if 'water' in answer_lower and 'level' in answer_lower:
            keywords.append('check water level')
        if 'restart' in answer_lower or 'reboot' in answer_lower:
            keywords.append('restart machine')
        if 'replace' in answer_lower:
            keywords.append('replace part')
        return keywords if keywords else ['troubleshoot']

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

def is_question_too_vague(user_q):
    """
    Use GPT-4 to intelligently determine if a query is too vague.
    Checks if the query specifies WHAT is wrong or WHAT the user needs.
    """
    prompt = f"""You are analyzing a user's support query to determine if it's specific enough to answer.

User's query: "{user_q}"

Determine if this query is VAGUE or SPECIFIC:

VAGUE = Query doesn't specify WHAT is wrong, WHAT error they're seeing, or WHAT they need help with
Examples of VAGUE:
- "machine is not working" → doesn't say what's wrong (too general)
- "it's broken" → no specifics
- "need help" → help with what?
- "something wrong" → what specifically?
- "having issues" → what kind of issues?

SPECIFIC = Query mentions a symptom, behavior, component, error, or specific procedure
Examples of SPECIFIC:
- "machine shows error 4012" → specific error code
- "machine not turning on" → specific symptom (power issue)
- "machine won't power on" → specific symptom (power issue)
- "no power lights" → specific symptom (power issue)
- "cotton candy won't spin" → specific symptom
- "sticks are getting stuck" → specific symptom (feeding issue)
- "sticks are jamming" → specific symptom (feeding issue)
- "cup didn't drop after payment" → specific issue
- "cups not dispensing" → specific symptom (dispensing issue)
- "how to clean the nozzle" → specific task (procedural)
- "how to connect to wifi" → specific task (procedural)
- "how to change settings" → specific task (procedural)
- "how to install card reader" → specific task (procedural)
- "machine making loud noise" → specific symptom
- "payment not working" → specific component
- "screen is black" → specific symptom
- "buttons not responding" → specific symptom

CRITICAL: ALL "how to" questions are SPECIFIC because they specify what procedure the user wants to learn.

Important: If the query mentions a specific symptom, behavior, component, OR asks "how to" do something, classify as SPECIFIC.

Respond with ONLY one word: VAGUE or SPECIFIC"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().upper()
        logger.info(f"Vagueness check: '{user_q}' → {result}")
        return result == "VAGUE"
    except Exception as e:
        logger.error(f"Vagueness check failed: {e}")
        # Fallback to simple check
        vague_phrases = ["not working", "broken", "help", "issue", "problem", "wrong"]
        return any(phrase in user_q.lower() for phrase in vague_phrases) and len(user_q.split()) <= 4

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

def calculate_action_overlap(new_actions, suggested_actions):
    """
    Calculate the overlap percentage between new actions and already-suggested actions.
    Returns a value from 0.0 to 1.0.
    """
    if not new_actions or not suggested_actions:
        return 0.0

    # Check how many new actions are similar to already-suggested actions
    overlap_count = 0
    for new_action in new_actions:
        for suggested_action in suggested_actions:
            # Check for similarity using word overlap
            new_words = set(new_action.split())
            suggested_words = set(suggested_action.split())

            # If actions share significant words, consider them overlapping
            common_words = new_words & suggested_words
            if len(common_words) >= 2 or (len(common_words) >= 1 and len(new_words) <= 2):
                overlap_count += 1
                break  # Don't double-count this new action

    overlap_ratio = overlap_count / len(new_actions)
    logger.debug(f"Action overlap: {overlap_count}/{len(new_actions)} = {overlap_ratio:.2f}")
    logger.debug(f"  New actions: {new_actions}")
    logger.debug(f"  Already suggested: {suggested_actions}")
    return overlap_ratio

def generate_contextual_no_match_response(user_question, machine_type=None):
    """
    Generate a contextual response when no matches are found.
    Uses GPT-4 to determine if the query is clear but lacks knowledge, or if it's too vague.
    Also searches for relevant videos before escalating.
    """
    prompt = f"""You are analyzing why a support chatbot couldn't find an answer for a user's query about Sweet Robo machines.

User's question: "{user_question}"

Determine the most appropriate response category:

1. CLEAR_NO_KNOWLEDGE - The question is specific and clear about WHAT they want to know, even if we don't have the answer
   Examples:
   - "how to access device testing mode" → clear procedural question
   - "how to calibrate temperature sensor" → clear technical question
   - "how to update firmware" → clear operation question
   These are CLEAR questions; we just don't have the information.

2. NEEDS_DETAILS - The question is vague and doesn't specify what the problem is or what they need
   Examples:
   - "it's not working" → vague, what specifically isn't working?
   - "help" → vague, help with what?
   - "something is broken" → vague, what is broken?
   These are UNCLEAR questions that need more context.

3. OUT_OF_SCOPE - The question is about something completely unrelated to machine operation or troubleshooting
   Example: "what's the weather today" → not about machines at all

Important: "How to" questions are CLEAR_NO_KNOWLEDGE if they specify what procedure/action the user wants.

Respond with ONLY one of these three categories: CLEAR_NO_KNOWLEDGE, NEEDS_DETAILS, or OUT_OF_SCOPE"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        category = response.choices[0].message.content.strip().upper()
        logger.info(f"No match response category: {category}")

        # Generate appropriate response based on category
        if category == "CLEAR_NO_KNOWLEDGE":
            # Check if there's a relevant video BEFORE escalating
            relevant_video = None
            if machine_type:
                # Search for video using empty string as final_answer since we don't have one
                relevant_video = find_relevant_video(user_question, "", machine_type)

            if relevant_video:
                # We have a video! Return a helpful response with the video
                logger.info(f"✓ No text answer but found video: '{relevant_video['title']}'")
                return (f"I don't have detailed written instructions for this, but I found a video guide that should help!\n\n"
                       f"📹 **Video Guide:** [{relevant_video['title']}]({relevant_video['url']})\n\n"
                       f"{relevant_video['description']}\n\n"
                       f"If you need additional help after watching, contact support@sweetrobo.com")

            # No video either - extract topic and escalate
            topic_prompt = f"""Extract the main topic or feature the user is asking about in 2-5 words.

User's question: "{user_question}"

Main topic (2-5 words):"""

            topic_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": topic_prompt}],
                temperature=0,
                max_tokens=20
            )
            topic = topic_response.choices[0].message.content.strip().lower()

            return (f"I don't currently have information about {topic} in my knowledge base. "
                   f"I'm escalating this to our support team who can help you with this specific question.")

        elif category == "NEEDS_DETAILS":
            return ("Could you provide more details about what you're experiencing? "
                   "For example:\n"
                   "• What specific error or issue are you seeing?\n"
                   "• Which part of the machine is affected?\n"
                   "• When does this problem occur?")

        else:  # OUT_OF_SCOPE
            return ("I specialize in troubleshooting Sweet Robo machines. "
                   "If you have a question about machine errors, maintenance, or operation, I'd be happy to help!")

    except Exception as e:
        logger.error(f"Failed to generate contextual no-match response: {e}")
        # Fallback to generic response
        return "Sorry, I couldn't find a helpful answer. Can you rephrase the question with more details?"

def user_wants_video(user_question):
    """
    Check if user is explicitly requesting visual guidance.
    Returns True if user's language indicates they want a video/demonstration.
    """
    user_q_lower = user_question.lower()

    video_request_phrases = [
        "how do i", "how to", "show me", "demonstrate",
        "where is", "where do i find", "video", "guide me",
        "can i see", "show video", "watch"
    ]

    is_video_request = any(phrase in user_q_lower for phrase in video_request_phrases)

    if is_video_request:
        logger.info(f"✓ User explicitly requested visual guidance: '{user_question[:60]}...'")

    return is_video_request


def find_relevant_video(user_question, final_answer, machine_type):
    """
    Find the most relevant video from the library for this query.
    Returns video dict if found, None otherwise.
    """
    if not VIDEO_LIBRARY:
        logger.info("No video library available")
        return None

    # Filter videos by machine type
    applicable_videos = [v for v in VIDEO_LIBRARY if machine_type in v.get("machine_types", [])]

    if not applicable_videos:
        logger.info(f"No videos available for machine type: {machine_type}")
        return None

    # Improved keyword matching - prioritize user question matches
    # Normalize WiFi variations for better matching
    user_q_lower = user_question.lower().replace("wi-fi", "wifi").replace("wi fi", "wifi")
    answer_lower = final_answer.lower().replace("wi-fi", "wifi").replace("wi fi", "wifi")

    # Extract error code from question if present
    import re
    error_code_in_question = None
    error_match = re.search(r'\b(\d{4})\b', user_question)
    if error_match:
        error_code_in_question = error_match.group(1)

    # Collect all scores for debugging
    video_scores = []

    for video in applicable_videos:
        score = 0
        keywords = video.get("keywords", [])
        related_errors = video.get("related_errors", [])

        # If user asks about specific error code, video MUST address that error
        if error_code_in_question:
            if related_errors and error_code_in_question in [str(e) for e in related_errors]:
                # Perfect match - specific error video for specific error question
                score += 10
            elif not related_errors:
                # Generic video (no error codes) for specific error question
                # Penalize heavily unless it's highly relevant
                score -= 5

        # Count keyword matches with weighted scoring
        for keyword in keywords:
            # Normalize keyword for better matching (handle WiFi variants)
            keyword_lower = keyword.lower().replace("wi-fi", "wifi").replace("wi fi", "wifi")

            # High priority: keyword in user's question (3x weight)
            if keyword_lower in user_q_lower:
                score += 3
            # Lower priority: keyword only in answer (1x weight)
            elif keyword_lower in answer_lower:
                score += 1

        video_scores.append((video, score))

    # Sort by score descending
    video_scores.sort(key=lambda x: x[1], reverse=True)

    # Log top 3 candidates for debugging
    if video_scores:
        logger.info("Top 3 video candidates:")
        for i, (vid, score) in enumerate(video_scores[:3], 1):
            logger.info(f"  {i}. '{vid['title']}' (score: {score})")

    # Determine minimum threshold based on context
    if error_code_in_question:
        # For error-specific questions, require high score (must have matching error code)
        min_threshold = 8
    else:
        # For general "how to" questions, lower threshold is OK
        min_threshold = 6

    # Return best match if score is good enough
    if video_scores and video_scores[0][1] >= min_threshold:
        best_match = video_scores[0][0]
        best_score = video_scores[0][1]
        logger.info(f"✓ Video matched: '{best_match['title']}' (score: {best_score}, threshold: {min_threshold})")
        return best_match
    else:
        best_score = video_scores[0][1] if video_scores else 0
        logger.info(f"No relevant video found (best score: {best_score}, needed: {min_threshold})")
        return None


def process_match_for_followup(match, user_q_lower, seen_ids):
    th_state = get_session_state()
    thread_id = th_state["thread_id"]

    if match.id in seen_ids:
        return None, None

    metadata = match.metadata
    answer = metadata.get("a", "[No A]").strip()
    if not answer:
        return None, None

    q = metadata.get("q", "")

    # Check if answer was already given (full answer embedding check)
    already_given = is_already_given(answer, th_state["conversation_history"])

    if already_given:
        logger.debug(f"FOLLOW-UP FILTER - Q: {q[:50] if q else 'N/A'}... Already Given: {already_given}")
        return None, None

    # NEW: Check for solution-level overlap (key actions)
    suggested_actions = th_state["suggested_actions_by_thread"].get(thread_id, [])

    if suggested_actions:
        # Extract key actions from this answer
        new_actions = extract_key_actions(answer)

        # Calculate overlap with already-suggested actions
        overlap = calculate_action_overlap(new_actions, suggested_actions)

        # If >70% of actions are already suggested, skip this entry
        if overlap >= 0.70:
            logger.info(f"✗ SKIPPING (action overlap {overlap:.0%}): {q[:60]}...")
            logger.info(f"  New actions: {new_actions}")
            logger.info(f"  Already suggested: {suggested_actions}")
            return None, None
        else:
            logger.info(f"✓ ACCEPTING (action overlap {overlap:.0%}): {q[:60]}...")
            logger.info(f"  New actions: {new_actions}")

    return answer, match.id

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
        # Handle both old 3-item and new 4-item tuples for backward compatibility
        if len(all_matches[i]) == 4:
            match, usefulness, confidence, _boosted = all_matches[i]
        else:
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
You are a technical support AI assistant for Sweet Robo machines. The user said the initial solution didn't work.

You are given up to 5 technical answers. Provide alternative troubleshooting steps in INDUSTRY-STANDARD format.

REQUIRED FORMAT:
1. Start with a brief acknowledgment (1 sentence)
   Example: "Let's try some additional steps."

2. Provide ONLY 2-3 alternative steps maximum (don't overwhelm)

3. Use numbered steps with BOLD headers (use **text** for bold in markdown)
   Format each step as:
   **1. [Action header]**
   Brief description of what to do.

   IMPORTANT: Put a blank line between the header and description!

4. Keep each step focused on ONE clear action

5. Only include USER-SAFE steps (no opening machines, inspecting circuit boards, etc.)

6. Do NOT add escalation text (that will be appended automatically)

EXAMPLE FORMAT:
Let's try some additional steps.

**1. Check the fuse**

Locate the fuse near the power switch and replace if blown.

**2. Test with different outlet**

Try plugging the machine into a different wall outlet.

**3. Check for visible damage**

Look for any frayed cables or loose connections.

User Question:
{original_question}

Answer References:
{combined_input}

NOW GENERATE THE ANSWER:
Final helpful answer:
"""
                gpt_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": gpt_prompt}],
                    temperature=0.3,
                )
                final_answer = gpt_response.choices[0].message.content.strip()
        except Exception:
            final_answer = "Sorry, no answer was found. Escalating to our support team now."

    final_answer += "\n\nNeed more help? Contact support@sweetrobo.com with your machine serial number."

    # NEW: Extract and track suggested actions for future deduplication
    actions_in_answer = extract_key_actions(final_answer)
    if thread_id not in th_state["suggested_actions_by_thread"]:
        th_state["suggested_actions_by_thread"][thread_id] = []

    # Add new actions to the list (avoid duplicates)
    for action in actions_in_answer:
        if action not in th_state["suggested_actions_by_thread"][thread_id]:
            th_state["suggested_actions_by_thread"][thread_id].append(action)

    logger.info(f"Updated suggested actions for thread {thread_id}: {th_state['suggested_actions_by_thread'][thread_id]}")

    # OPTION C (HYBRID): Check for video in follow-ups too
    if user_wants_video(user_q):
        logger.info("User requested visual guidance in follow-up - searching for relevant video")
        relevant_video = find_relevant_video(user_q, final_answer, th_state["machine_type"])

        if relevant_video:
            # Insert video BEFORE the escalation line
            escalation_line = "\n\nNeed more help? Contact support@sweetrobo.com with your machine serial number."
            if escalation_line in final_answer:
                final_answer = final_answer.replace(
                    escalation_line,
                    f"\n\n📹 **Video Guide:** [{relevant_video['title']}]({relevant_video['url']}){escalation_line}"
                )
            else:
                final_answer += f"\n\n📹 **Video Guide:** [{relevant_video['title']}]({relevant_video['url']})"
            logger.info(f"✓ Added video to follow-up response: {relevant_video['title']}")
        else:
            # User wanted video but none available - just skip it
            logger.info("User wanted video in follow-up but none found - not adding any message")

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

def is_same_topic(user_q, candidate_q, candidate_a):
    """
    Check if a Q&A entry is relevant to the user's query.
    NEW: Now considers BOTH question and answer content, not just question similarity.
    """
    prompt = f"""You are a technical support AI helping match user queries with relevant Q&A entries.

User's query: "{user_q}"

Q&A Entry:
Question: "{candidate_q}"
Answer: "{candidate_a[:300]}..."

Does this Q&A entry contain information related to the user's query?

Consider:
- Does the ANSWER mention relevant concepts, components, or procedures?
- Does it contain keywords or terminology related to the user's query?
- Even if it doesn't fully answer the question, is it partially relevant?
- Even if the questions seem different, the answer content might be related

Important: Answer "yes" if the content is even PARTIALLY relevant to the user's query.

Respond only with "yes" or "no".
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        logger.debug(f"GPT Content Check: Q: '{candidate_q[:40]}...' A: '{candidate_a[:40]}...' → {result}")
        return "yes" in result
    except Exception as e:
        logger.error(f"GPT content check failed: {e}")
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

        # ERROR CODE BOOST: If user query has error code AND entry has that exact error code, boost similarity
        error_codes_in_metadata = metadata.get("error_codes", [])
        boosted_score = match.score  # Start with original score

        if error_code_filter and error_codes_in_metadata:
            if str(error_code_filter) in [str(code) for code in error_codes_in_metadata]:
                boosted_score += 0.20  # Boost by 20% for exact error code match
                logger.info(f"🎯 ERROR CODE BOOST: Entry has exact error code {error_code_filter}, boosting score from {match.score:.3f} to {boosted_score:.3f}")

        # Debug logging for filtering process
        if match.score >= 0.5:  # Log high-scoring matches to see why they're filtered
            logger.info(f"Checking match (score {match.score:.3f}, boosted: {boosted_score:.3f}): usefulness={usefulness_score}, confidence={confidence:.2f}, Q='{candidate_q[:50]}...'")

        # Lower threshold for error code queries since they're critical
        is_error_entry = bool(error_codes_in_metadata) or error_code_filter
        usefulness_threshold = 4 if is_error_entry else 7  # Lower to 4 for error codes

        if usefulness_score >= usefulness_threshold and confidence >= CONFIDENCE_THRESHOLD:
            if isinstance(match.values, list) and len(match.values) == 1536:
                if answer not in seen_answers:
                    logger.info(f"✓ Match accepted - Cosine: {match.score:.3f}, Boosted: {boosted_score:.3f}, Usefulness: {usefulness_score}, Confidence: {confidence}")
                    all_valid.append((match, usefulness_score, confidence, boosted_score))
                    seen_answers.add(answer)
            else:
                if match.score >= 0.5:
                    logger.warning(f"✗ Match rejected - Invalid embedding dimensions or format")
        else:
            if match.score >= 0.5:
                logger.warning(f"✗ Match rejected - Failed filters: usefulness={usefulness_score} (need >=7), confidence={confidence:.2f} (need >=0.6)")

    # Sort by boosted similarity score (includes error code boost)
    # The boosted_score is at index 3 in the tuple (match, usefulness, confidence, boosted_score)
    all_valid.sort(key=lambda x: -x[3])

    logger.info(f"Found {len(all_valid)} matches passing initial filters")
    if all_valid:
        for i, (match, score, conf, boosted) in enumerate(all_valid[:3]):
            logger.info(f"  Pre-GPT #{i+1}: '{match.metadata.get('q', '')[:60]}...' (boosted score: {boosted:.3f})")

    # GPT content relevance check — only top 5 to minimize cost and latency
    # NEW: Now checks if Q&A CONTENT is relevant, not just question similarity
    filtered_with_gpt = []
    for match, score, confidence, boosted_score in all_valid[:5]:  # Reduced from 10 to 5
        q = match.metadata.get("q", "")
        a = match.metadata.get("a", "")
        if is_same_topic(query_text, q, a):  # Pass answer content too
            filtered_with_gpt.append((match, score, confidence, boosted_score))  # Include boosted_score
            logger.info(f"✓ GPT approved: '{q[:50]}...'")
        else:
            logger.info(f"✗ GPT rejected: '{q[:50]}...'")

    # Calculate complexity penalty for each match
    # Simpler, more direct questions should rank higher
    def calculate_match_score(item):
        match, usefulness, confidence, boosted_score = item  # Now includes boosted_score
        q = match.metadata.get("q", "").lower()
        user_q_lower = query_text.lower()

        # Base score is the boosted similarity score (includes error code boost)
        base_score = boosted_score  # Use boosted score instead of original
        
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
    
    # PRODUCTION FIX 1: Sanitize input
    user_question = user_question.strip()[:1000]  # Basic sanitization
    if not user_question:
        return "Please enter a valid question."
    
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
        
    # Check for vague question - generate dynamic clarification request
    if not is_followup and is_question_too_vague(user_question):
        # Check if we already asked for clarification recently (context-aware generation)
        previous_clarification = None
        if len(th_state["conversation_history"]) > 0:
            # Get the last assistant message
            for entry in reversed(th_state["conversation_history"]):
                if entry["role"] == "assistant":
                    last_msg = entry["content"]
                    # Check if last message was a clarification request
                    if "could you be more specific" in last_msg.lower() or "could you provide more details" in last_msg.lower():
                        previous_clarification = last_msg
                    break

        # Use GPT-4 to generate a contextual clarification request
        if previous_clarification:
            # Context-aware: avoid repeating the same opening
            clarification_prompt = f"""The user submitted another vague support query: "{user_question}"

You previously asked: "{previous_clarification}"

Now generate a DIFFERENT clarification request with varied wording (don't repeat the same opening).

Requirements:
- Be direct and technical (no empathy phrases)
- Use different opening than before (vary the phrasing)
- Provide 2-3 specific examples tailored to their NEW query
- Keep it to EXACTLY 2 sentences
- Use alternatives like: "What specifically is happening?", "Can you describe the issue in more detail?", "What exactly isn't working?"

Format:
Sentence 1: Ask what's happening (USE DIFFERENT WORDS than previous request)
Sentence 2: "For example, [specific example 1], [specific example 2], or [specific example 3]?"

Examples of VARIED openings:
- "What specifically is happening?"
- "Can you describe the issue in more detail?"
- "What exactly isn't working?"
- "Could you tell me more about what you're seeing?"

Clarification request:"""
        else:
            # First clarification request - use standard format
            clarification_prompt = f"""The user submitted a vague support query: "{user_question}"

Generate a concise clarification request that asks them to be more specific.

Requirements:
- Be direct and technical (no empathy phrases like "I'm sorry" or "I'd be happy to help")
- Provide 2-3 specific examples tailored to what they mentioned
- Keep it to EXACTLY 2 sentences: one asking for specifics, one with examples
- Start with "Could you be more specific about what's happening?"

Format:
Sentence 1: Ask what's happening
Sentence 2: "For example, [specific example 1], [specific example 2], or [specific example 3]?"

Examples:
User: "machine not working" → "Could you be more specific about what's happening? For example, is the machine showing an error code, not powering on, or behaving erratically?"
User: "having issues" → "Could you be more specific about what's happening? For example, are you seeing an error message, experiencing a hardware problem, or having trouble with a specific feature?"
User: "something broken" → "Could you be more specific about what's happening? For example, is a part physically broken, is something not functioning correctly, or is there an error displayed?"

Clarification request:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": clarification_prompt}],
                temperature=0.3,
                max_tokens=80
            )
            # Strip both whitespace and quotes from the response
            clarification = response.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(f"Generated {'varied' if previous_clarification else 'initial'} clarification: {clarification[:60]}...")
            return clarification
        except Exception as e:
            logger.error(f"Failed to generate dynamic clarification: {e}")
            # Fallback to generic message
            return ("Could you be more specific about what's happening? For example, is the machine showing an error code, not powering on, or behaving erratically?")

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
    # Accept error codes with or without the word "error"
    error_code_filter = int(match.group(1)) if match else None

    top_matches = fetch_valid_matches(query_embedding, previous_ids, error_code_filter, user_question)

    # Save matches for future follow-up
    th_state["used_matches_by_thread"][thread_id] = {
        "embedding": query_embedding,
        "error_filter": error_code_filter,
        "original_question": user_question,
        "all_matches": top_matches,
    }
    th_state["match_pointer"] = 1  # Start at second match for follow-up


    # No valid matches - generate contextual response
    if not top_matches:
        logger.warning("No top matches found after similarity filtering")
        return generate_contextual_no_match_response(user_question, th_state["machine_type"])

    # NEW: Check entry_type and similarity score for routing decision
    best_match = top_matches[0]  # Already sorted in GPT filter
    entry_type = best_match[0].metadata.get("entry_type", "direct_qa")
    original_score = best_match[0].score
    boosted_score = best_match[3]  # Get boosted score from tuple (match, usefulness, confidence, boosted_score)
    raw_answer = best_match[0].metadata.get("a", "[No A]").strip()

    logger.info(f"Routing decision - Entry type: {entry_type}, Original: {original_score:.3f}, Boosted: {boosted_score:.3f}")

    # Decision tree: Direct retrieval vs. GPT synthesis (0.70 threshold for boosted score)
    if entry_type == "direct_qa" and boosted_score >= 0.70:
        # HIGH CONFIDENCE DIRECT ANSWER: Format to match industry standard
        logger.info("Route: DIRECT RETRIEVAL (high confidence direct_qa) - formatting to industry standard")

        # Remove "Problem:" line if it exists to avoid redundancy
        lines = raw_answer.strip().split('\n')
        if lines and lines[0].startswith("Problem:"):
            raw_answer_cleaned = '\n'.join(lines[1:]).strip()
        else:
            raw_answer_cleaned = raw_answer.strip()

        # Format the direct answer to match industry standard
        format_prompt = f"""You are reformatting a technical support answer to match industry-standard formatting.

User's question: "{user_question}"

Current answer (plain text):
{raw_answer_cleaned}

Reformat this answer following this EXACT structure:

CRITICAL FORMATTING RULES:
1. Start with a brief intro (1 sentence)
2. Each step MUST have a number AND bold title (format: **1. Title**)
3. Add ONE BLANK LINE after the bold title
4. Put the description on the next line
5. Add ONE BLANK LINE between steps
6. End with: "Need more help? Contact support@sweetrobo.com with your machine serial number."

EXACT FORMAT TO FOLLOW:
[Intro sentence]

**1. [Step title]**

[Description for step 1]

**2. [Next step title]**

[Description for step 2]

**3. [Another step title]**

[Description for step 3]

Need more help? Contact support@sweetrobo.com with your machine serial number.

EXAMPLE:
Let's troubleshoot this issue.

**1. Check power cable**

Ensure the cable is firmly connected to both the machine and outlet.

**2. Verify outlet**

Plug another device into the outlet to confirm it's working.

**3. Inspect the fuse**

Check the fuse near the power switch and replace if blown.

Need more help? Contact support@sweetrobo.com with your machine serial number.

NOW REFORMAT (FOLLOW THE EXACT FORMAT ABOVE):"""

        try:
            format_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": format_prompt}],
                temperature=0.3,
            )
            final_answer = format_response.choices[0].message.content.strip()
            logger.info(f"Direct answer formatted: {final_answer[:80]}...")
        except Exception as e:
            logger.error(f"Direct answer formatting failed: {e}, using raw answer")
            final_answer = raw_answer_cleaned

    elif entry_type == "knowledge_base" or boosted_score < 0.70:
        # KNOWLEDGE-BASED SYNTHESIS: Use GPT-4 to reason with context
        # This handles: knowledge_base entries (always), direct_qa with boosted score < 0.70
        logger.info(f"Route: GPT SYNTHESIS (entry_type={entry_type}, boosted_score={boosted_score:.3f})")

        # Gather context from top matches (up to 3)
        context_entries = []
        for i, (match, usefulness, confidence, _boosted) in enumerate(top_matches[:3]):
            q = match.metadata.get("q", "")
            a = match.metadata.get("a", "")
            context_entries.append(f"Reference {i+1}:\nQ: {q}\nA: {a}")

        combined_context = "\n\n".join(context_entries)

        # GPT-4 synthesis prompt (INDUSTRY-STANDARD FORMAT)
        synthesis_prompt = f"""You are a technical support AI assistant for Sweet Robo machines.

The user asked: "{user_question}"

I found these relevant knowledge base entries:

{combined_context}

Based on these references, provide a clear, scannable answer following INDUSTRY-STANDARD formatting.

REQUIRED FORMAT:
1. Start with a brief intro (1 sentence) that acknowledges the issue
   Example: "Let's troubleshoot the power issue." or "Here's how to clean the nozzle."

2. Provide ONLY 3-4 steps maximum (don't overwhelm users)

3. Use numbered steps with BOLD headers (use **text** for bold in markdown)
   Format each step as:
   **1. [Action header]**
   Brief description of what to do.

   IMPORTANT: Put a blank line between the header and description!

4. Keep each step focused on ONE clear action (don't combine multiple actions)

5. Only include USER-SAFE steps (no opening machines, inspecting circuit boards, etc.)

6. End with a specific escalation:
   "Need more help? Contact support@sweetrobo.com with your machine serial number."

EXAMPLE FORMAT:
Let's troubleshoot the payment issue.

**1. Check power connection**

Ensure the payment terminal cable is firmly connected.

**2. Restart the terminal**

Unplug the terminal for 10 seconds, then plug it back in.

**3. Test a transaction**

Try processing a test payment to verify functionality.

Need more help? Contact support@sweetrobo.com with your machine serial number.

NOW GENERATE THE ANSWER:
Answer:"""

        try:
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
            )
            final_answer = gpt_response.choices[0].message.content.strip()
            logger.info(f"GPT synthesis successful: {final_answer[:80]}...")
        except Exception as e:
            logger.error(f"GPT synthesis failed: {e}")
            # Fallback to direct answer
            lines = raw_answer.strip().split('\n')
            if lines and lines[0].startswith("Problem:"):
                final_answer = '\n'.join(lines[1:]).strip()
            else:
                final_answer = raw_answer.strip()

    logger.info(f"Final answer selected: {final_answer[:80]}...")

    # NEW: Extract and track suggested actions for future deduplication
    actions_in_answer = extract_key_actions(final_answer)
    if thread_id not in th_state["suggested_actions_by_thread"]:
        th_state["suggested_actions_by_thread"][thread_id] = []

    # Add new actions to the list (avoid duplicates)
    for action in actions_in_answer:
        if action not in th_state["suggested_actions_by_thread"][thread_id]:
            th_state["suggested_actions_by_thread"][thread_id].append(action)

    logger.info(f"Initialized suggested actions for thread {thread_id}: {th_state['suggested_actions_by_thread'][thread_id]}")

    # OPTION C (HYBRID): Check for video only when user explicitly requests it
    if user_wants_video(user_question):
        logger.info("User requested visual guidance - searching for relevant video")
        relevant_video = find_relevant_video(user_question, final_answer, th_state["machine_type"])

        if relevant_video:
            # Video found - append it to response
            final_answer += f"\n\n📹 **Video Guide:** [{relevant_video['title']}]({relevant_video['url']})"
            logger.info(f"✓ Added video to response: {relevant_video['title']}")
        else:
            # User wanted video but none available - just skip it
            logger.info("User wanted video but none found - not adding any message")
    else:
        # User didn't explicitly request video - add subtle hint for complex answers
        # Only add hint if answer is complex (multi-step)
        if len(final_answer.split()) > 100:  # Complex answer with many steps
            final_answer += "\n\n💡 *Need visual guidance? Just ask for a video.*"
            logger.info("Added video hint to complex answer")

    # Cache assistant response embedding
    assistant_embedding = get_or_create_embedding(final_answer, f"hist_{thread_id}_{len(th_state['conversation_history'])}")

    th_state["conversation_history"].extend([
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": final_answer, "embedding": assistant_embedding}
    ])
    return final_answer