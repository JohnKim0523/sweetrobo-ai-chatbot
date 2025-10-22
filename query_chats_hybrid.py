"""
Hybrid Chatbot Implementation
Uses dataset as primary source, falls back to GPT-4 with dataset context
"""

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

# Import all functions from original
from query_chats_wrapped import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridChatbot:
    """
    Hybrid approach:
    1. Search dataset first (current logic)
    2. If low confidence, use GPT-4 WITH dataset context
    3. GPT-4 learns from your Q&As to answer new questions
    """
    
    def __init__(self):
        self.load_knowledge_base()
        self.confidence_threshold_generate = 0.5  # Below this, we generate
        self.confidence_threshold_enhance = 0.7   # Below this, we enhance
    
    def load_knowledge_base(self):
        """Load and index the dataset for GPT-4 context"""
        with open('cleaned_qa_JSON.json', 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # Build knowledge maps for quick context retrieval
        self.error_knowledge = {}
        self.symptom_solutions = {}
        self.machine_specific = {}
        
        for item in self.dataset:
            q = item.get('q', '').lower()
            a = item.get('a', '')
            machine = item.get('machine_type', '')
            usefulness = item.get('usefulness', {}).get('score', 0)
            
            # Only use high-quality answers for context
            if usefulness < 7:
                continue
            
            # Index by error codes
            error_matches = re.findall(r'(40\d{2})', q + ' ' + a)
            for error in error_matches:
                if error not in self.error_knowledge:
                    self.error_knowledge[error] = []
                self.error_knowledge[error].append({
                    'question': item.get('q', ''),
                    'answer': item.get('a', ''),
                    'usefulness': usefulness
                })
            
            # Index by symptoms
            symptoms = {
                'not_spinning': ['spin', 'rotate', 'turning'],
                'no_heat': ['heat', 'temperature', 'cold', 'warm'],
                'stuck': ['stuck', 'jam', 'block', 'clog'],
                'no_dispense': ['dispense', 'output', 'coming out'],
                'noise': ['noise', 'sound', 'loud', 'rattle']
            }
            
            for symptom, keywords in symptoms.items():
                if any(kw in q for kw in keywords):
                    if symptom not in self.symptom_solutions:
                        self.symptom_solutions[symptom] = []
                    self.symptom_solutions[symptom].append({
                        'question': item.get('q', ''),
                        'answer': item.get('a', ''),
                        'machine': machine
                    })
            
            # Index by machine type
            if machine:
                if machine not in self.machine_specific:
                    self.machine_specific[machine] = []
                self.machine_specific[machine].append({
                    'q': item.get('q', ''),
                    'a': item.get('a', '')[:300]  # Truncate for context
                })
    
    def get_relevant_context(self, question: str, machine_type: str, limit: int = 5) -> str:
        """
        Extract relevant Q&As from dataset to give GPT-4 context
        This is the KEY - GPT-4 uses YOUR data to answer
        """
        context_items = []
        q_lower = question.lower()
        
        # 1. Find error-specific context
        error_matches = re.findall(r'(40\d{2})', q_lower)
        for error in error_matches:
            if error in self.error_knowledge:
                for item in self.error_knowledge[error][:2]:  # Top 2 for each error
                    context_items.append({
                        'relevance': 'ERROR_MATCH',
                        'q': item['question'],
                        'a': item['answer'][:300]
                    })
        
        # 2. Find symptom-specific context
        for symptom, solutions in self.symptom_solutions.items():
            symptom_keywords = {
                'not_spinning': ['spin', 'rotate'],
                'no_heat': ['heat', 'cold'],
                'stuck': ['stuck', 'jam'],
            }
            
            if symptom in symptom_keywords:
                if any(kw in q_lower for kw in symptom_keywords[symptom]):
                    relevant = [s for s in solutions if s['machine'] == machine_type]
                    for sol in relevant[:2]:
                        context_items.append({
                            'relevance': 'SYMPTOM_MATCH',
                            'q': sol['question'],
                            'a': sol['answer'][:300]
                        })
        
        # 3. Find semantically similar questions using embeddings
        try:
            # Get embedding for user question
            query_embedding = get_or_create_embedding(question)
            if query_embedding:
                # Use existing Pinecone search
                matches = fetch_valid_matches(
                    query_embedding,
                    set(),
                    None,
                    question
                )
                
                for match, _, _ in matches[:3]:
                    metadata = match.metadata
                    context_items.append({
                        'relevance': f'SEMANTIC_MATCH (score: {match.score:.2f})',
                        'q': metadata.get('q', ''),
                        'a': metadata.get('a', '')[:300]
                    })
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
        
        # Format context for GPT-4
        if not context_items:
            # Fallback: get general machine-specific examples
            if machine_type in self.machine_specific:
                for item in self.machine_specific[machine_type][:3]:
                    context_items.append({
                        'relevance': 'MACHINE_GENERAL',
                        'q': item['q'],
                        'a': item['a']
                    })
        
        # Build context string
        context_str = "RELEVANT KNOWLEDGE FROM DATABASE:\n\n"
        for i, item in enumerate(context_items[:limit], 1):
            context_str += f"[{item['relevance']}]\n"
            context_str += f"Q: {item['q']}\n"
            context_str += f"A: {item['a']}\n\n"
        
        return context_str
    
    def generate_answer_with_context(self, question: str, machine_type: str, 
                                    dataset_matches: list = None) -> str:
        """
        Generate answer using GPT-4 WITH your dataset as context
        This ensures GPT-4 answers based on YOUR expertise, not general knowledge
        """
        
        # Get relevant context from dataset
        context = self.get_relevant_context(question, machine_type)
        
        # Build the prompt
        prompt = f"""You are a Sweet Robo technical support AI. You must answer based on the knowledge base provided.

MACHINE TYPE: {machine_type}
USER QUESTION: {question}

{context}

INSTRUCTIONS:
1. Use the knowledge base examples above to understand how this equipment works
2. If the exact question isn't in the examples, infer from similar cases
3. Be specific about error codes, part names, and procedures mentioned in the knowledge base
4. If the question is yes/no, start with "Yes" or "No" then explain
5. Format your answer like the examples: Problem, Why it's happening, What to do
6. If you're unsure, say "This issue isn't fully covered in my knowledge base. Contact support at..."

Generate a helpful answer based on the knowledge base:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Sweet Robo expert. Answer ONLY based on the provided knowledge base."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=500
            )
            
            generated_answer = response.choices[0].message.content
            
            # Add note about AI generation
            generated_answer += "\n\n*[AI-enhanced response based on knowledge base]*"
            
            # Log for monitoring
            logger.info(f"Generated answer for: '{question[:50]}...'")
            
            return generated_answer
            
        except Exception as e:
            logger.error(f"GPT-4 generation failed: {e}")
            return "I encountered an issue generating a response. Please contact support directly."
    
    def enhance_weak_answer(self, question: str, machine_type: str, 
                          weak_answer: str, confidence: float) -> str:
        """
        Enhance a low-confidence dataset answer using GPT-4
        Combines dataset answer with additional context
        """
        
        context = self.get_relevant_context(question, machine_type)
        
        prompt = f"""You are enhancing a support answer that has low confidence ({confidence:.2f}).

USER QUESTION: {question}
CURRENT ANSWER: {weak_answer}

{context}

Improve this answer by:
1. Adding missing details from the knowledge base
2. Clarifying any ambiguous points
3. Ensuring completeness
4. Maintaining accuracy to the knowledge base

Enhanced answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            enhanced = response.choices[0].message.content
            enhanced += f"\n\n*[Enhanced with confidence: {confidence:.2f}]*"
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return weak_answer  # Return original if enhancement fails


def run_hybrid_chatbot(user_question: str) -> str:
    """
    Main hybrid chatbot function
    Replaces run_chatbot_session with hybrid logic
    """
    
    # Initialize hybrid bot
    bot = HybridChatbot()
    
    # Get session state
    th_state = get_session_state()
    
    # Input validation and sanitization
    user_question = user_question.strip()[:1000]
    if not user_question:
        return "Please enter a valid question."
    
    # Check for session
    if not th_state.get("thread_id"):
        return "⚠️ Please start a new support session by selecting the machine first."
    
    machine_type = th_state.get("machine_type")
    
    # STEP 1: Try standard dataset search
    logger.info(f"Hybrid search for: '{user_question[:50]}...'")
    
    # Get embedding
    query_embedding = get_or_create_embedding(user_question)
    if not query_embedding:
        return bot.generate_answer_with_context(user_question, machine_type)
    
    # Search dataset
    error_code_filter = None
    match = re.search(r"(\d{4,})", user_question)
    if "error" in user_question.lower() and match:
        error_code_filter = int(match.group(1))
    
    matches = fetch_valid_matches(
        query_embedding,
        set(),
        error_code_filter,
        user_question
    )
    
    # STEP 2: Evaluate confidence and decide approach
    if matches:
        best_match = matches[0]
        confidence = best_match[0].score  # Cosine similarity score
        answer = best_match[0].metadata.get("a", "")
        matched_question = best_match[0].metadata.get("q", "")
        
        logger.info(f"Best match confidence: {confidence:.3f}")
        
        # Check for exact or near-exact match
        if user_question.lower().strip() in matched_question.lower() or confidence >= 0.85:
            logger.info("Using dataset answer (exact/high match)")
            return answer + "\n\nIf this didn't resolve the issue, let me know."
        
        if confidence >= 0.75:  # Lowered from 0.8
            # High confidence - use dataset answer directly
            logger.info("Using dataset answer (high confidence)")
            return answer + "\n\nIf this didn't resolve the issue, let me know."
        
        elif confidence >= 0.6:  # Raised from 0.5
            # Medium confidence - enhance with GPT-4
            logger.info("Enhancing dataset answer with GPT-4")
            return bot.enhance_weak_answer(
                user_question, 
                machine_type,
                answer,
                confidence
            )
        else:
            # Low confidence - generate new answer with context
            logger.info("Generating new answer with GPT-4 (low confidence match)")
            return bot.generate_answer_with_context(
                user_question,
                machine_type,
                matches
            )
    else:
        # No matches - pure generation with context
        logger.info("No dataset matches - generating with GPT-4")
        return bot.generate_answer_with_context(user_question, machine_type)


# Test the hybrid approach
if __name__ == "__main__":
    print("HYBRID CHATBOT TEST")
    print("="*60)
    
    # Initialize session for testing
    from query_chats_wrapped import initialize_chat
    initialize_chat("cotton candy")
    
    test_questions = [
        # Should find in dataset
        "error 4012",
        
        # Might be in dataset but need enhancement  
        "machine making weird noise",
        
        # Not in dataset - needs generation
        "can error 4012 damage the motor permanently?",
        "what's the cost of ignoring error 4011?",
        "should I buy spare heating elements?"
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        print("-"*40)
        answer = run_hybrid_chatbot(q)
        print(f"A: {answer[:300]}...")
        print()