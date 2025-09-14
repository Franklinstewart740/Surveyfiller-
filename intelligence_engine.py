"""
Intelligence & Reasoning Engine
Advanced AI system with short-term memory, reasoning logging, and survey intent inference.
"""

import asyncio
import logging
import json
import time
import hashlib
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import aiofiles

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Types of survey questions."""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"
    OPINION = "opinion"
    RATING = "rating"
    MULTIPLE_CHOICE = "multiple_choice"
    TEXT_INPUT = "text_input"
    TRAP = "trap"  # Questions designed to catch inconsistent responses
    ATTENTION_CHECK = "attention_check"

class SurveyIntent(Enum):
    """Inferred survey intents."""
    BRAND_LOYALTY = "brand_loyalty"
    INCOME_BRACKET = "income_bracket"
    SHOPPING_HABITS = "shopping_habits"
    MEDIA_CONSUMPTION = "media_consumption"
    HEALTH_WELLNESS = "health_wellness"
    TECHNOLOGY_USAGE = "technology_usage"
    LIFESTYLE = "lifestyle"
    POLITICAL_VIEWS = "political_views"
    PRODUCT_FEEDBACK = "product_feedback"
    MARKET_RESEARCH = "market_research"

@dataclass
class AnswerReasoning:
    """Represents the reasoning behind an answer."""
    question_id: str
    question_text: str
    question_type: QuestionType
    selected_answer: str
    reasoning: str
    confidence_score: float
    consistency_check: bool
    related_memories: List[str]
    timestamp: datetime

@dataclass
class SurveyMemory:
    """Represents a memory from the survey session."""
    memory_id: str
    survey_id: str
    question_id: str
    content: str
    memory_type: str  # "answer", "preference", "demographic", "behavior"
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0

@dataclass
class SurveyContext:
    """Context information about the current survey."""
    survey_id: str
    platform: str
    estimated_length: int
    complexity_score: float
    inferred_intents: List[SurveyIntent]
    target_demographics: Dict[str, Any]
    completion_time_estimate: int  # minutes
    risk_level: str  # "low", "medium", "high"

class IntelligenceEngine:
    """Advanced intelligence engine with memory and reasoning capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = None
        
        # Initialize OpenAI if API key is available
        openai_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
        
        # Memory systems
        self.short_term_memory: Dict[str, List[SurveyMemory]] = {}  # survey_id -> memories
        self.reasoning_log: List[AnswerReasoning] = []
        self.survey_contexts: Dict[str, SurveyContext] = {}
        
        # Semantic similarity for consistency checking
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.question_vectors = {}
        
        # Load persistent memory
        asyncio.create_task(self._load_persistent_memory())
        
    async def _load_persistent_memory(self):
        """Load persistent memory from disk."""
        memory_file = self.config.get('memory_file', 'survey_memory.pkl')
        if os.path.exists(memory_file):
            try:
                async with aiofiles.open(memory_file, 'rb') as f:
                    data = await f.read()
                memory_data = pickle.loads(data)
                
                # Restore memory structures
                self.short_term_memory = memory_data.get('short_term_memory', {})
                self.reasoning_log = [AnswerReasoning(**r) for r in memory_data.get('reasoning_log', [])]
                self.survey_contexts = {k: SurveyContext(**v) for k, v in memory_data.get('survey_contexts', {}).items()}
                
                logger.info(f"Loaded {len(self.reasoning_log)} reasoning entries and {sum(len(m) for m in self.short_term_memory.values())} memories")
                
            except Exception as e:
                logger.error(f"Failed to load persistent memory: {e}")
    
    async def _save_persistent_memory(self):
        """Save persistent memory to disk."""
        memory_file = self.config.get('memory_file', 'survey_memory.pkl')
        try:
            memory_data = {
                'short_term_memory': self.short_term_memory,
                'reasoning_log': [asdict(r) for r in self.reasoning_log],
                'survey_contexts': {k: asdict(v) for k, v in self.survey_contexts.items()}
            }
            
            data = pickle.dumps(memory_data)
            async with aiofiles.open(memory_file, 'wb') as f:
                await f.write(data)
                
        except Exception as e:
            logger.error(f"Failed to save persistent memory: {e}")
    
    def create_memory_id(self, survey_id: str, question_id: str, content: str) -> str:
        """Create unique memory ID."""
        return hashlib.sha256(f"{survey_id}:{question_id}:{content}".encode()).hexdigest()[:16]
    
    async def add_memory(self, survey_id: str, question_id: str, content: str, 
                        memory_type: str, importance_score: float = 0.5):
        """Add a new memory to short-term memory."""
        memory_id = self.create_memory_id(survey_id, question_id, content)
        
        memory = SurveyMemory(
            memory_id=memory_id,
            survey_id=survey_id,
            question_id=question_id,
            content=content,
            memory_type=memory_type,
            importance_score=importance_score,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        if survey_id not in self.short_term_memory:
            self.short_term_memory[survey_id] = []
        
        self.short_term_memory[survey_id].append(memory)
        
        # Keep only most important memories (limit to 50 per survey)
        if len(self.short_term_memory[survey_id]) > 50:
            self.short_term_memory[survey_id].sort(key=lambda m: m.importance_score, reverse=True)
            self.short_term_memory[survey_id] = self.short_term_memory[survey_id][:50]
        
        await self._save_persistent_memory()
    
    def get_related_memories(self, survey_id: str, question_text: str, limit: int = 5) -> List[SurveyMemory]:
        """Get memories related to the current question."""
        if survey_id not in self.short_term_memory:
            return []
        
        memories = self.short_term_memory[survey_id]
        
        # Simple keyword-based relevance for now
        # In production, would use more sophisticated semantic similarity
        question_words = set(question_text.lower().split())
        
        scored_memories = []
        for memory in memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(question_words.intersection(memory_words))
            relevance_score = overlap / max(len(question_words), 1)
            
            if relevance_score > 0:
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                scored_memories.append((memory, relevance_score))
        
        # Sort by relevance and importance
        scored_memories.sort(key=lambda x: x[1] * x[0].importance_score, reverse=True)
        
        return [memory for memory, score in scored_memories[:limit]]
    
    async def classify_question_type(self, question_text: str, options: List[str] = None) -> QuestionType:
        """Classify the type of survey question."""
        question_lower = question_text.lower()
        
        # Rule-based classification
        if any(word in question_lower for word in ['age', 'gender', 'income', 'education', 'location', 'occupation']):
            return QuestionType.DEMOGRAPHIC
        
        if any(word in question_lower for word in ['how often', 'frequency', 'usually', 'typically', 'behavior']):
            return QuestionType.BEHAVIORAL
        
        if any(word in question_lower for word in ['prefer', 'like', 'favorite', 'choose', 'select']):
            return QuestionType.PREFERENCE
        
        if any(word in question_lower for word in ['opinion', 'think', 'believe', 'feel', 'agree']):
            return QuestionType.OPINION
        
        if any(word in question_lower for word in ['rate', 'scale', 'score', 'satisfaction']):
            return QuestionType.RATING
        
        # Check for attention checks
        if any(phrase in question_lower for phrase in ['select the third option', 'choose option c', 'attention check']):
            return QuestionType.ATTENTION_CHECK
        
        # Check for trap questions (contradictory or consistency checks)
        if 'never' in question_lower and 'always' in question_lower:
            return QuestionType.TRAP
        
        # Default classification based on input type
        if options and len(options) > 1:
            return QuestionType.MULTIPLE_CHOICE
        else:
            return QuestionType.TEXT_INPUT
    
    async def infer_survey_intent(self, questions: List[str], survey_metadata: Dict[str, Any] = None) -> List[SurveyIntent]:
        """Infer the intent of the survey based on questions."""
        all_text = ' '.join(questions).lower()
        intents = []
        
        # Brand loyalty indicators
        if any(word in all_text for word in ['brand', 'loyalty', 'switch', 'recommend', 'trust']):
            intents.append(SurveyIntent.BRAND_LOYALTY)
        
        # Income bracket indicators
        if any(word in all_text for word in ['income', 'salary', 'earnings', 'household', 'financial']):
            intents.append(SurveyIntent.INCOME_BRACKET)
        
        # Shopping habits indicators
        if any(word in all_text for word in ['shop', 'buy', 'purchase', 'store', 'online', 'retail']):
            intents.append(SurveyIntent.SHOPPING_HABITS)
        
        # Media consumption indicators
        if any(word in all_text for word in ['watch', 'tv', 'streaming', 'social media', 'news', 'entertainment']):
            intents.append(SurveyIntent.MEDIA_CONSUMPTION)
        
        # Health and wellness indicators
        if any(word in all_text for word in ['health', 'wellness', 'exercise', 'diet', 'medical', 'fitness']):
            intents.append(SurveyIntent.HEALTH_WELLNESS)
        
        # Technology usage indicators
        if any(word in all_text for word in ['technology', 'smartphone', 'app', 'digital', 'internet', 'computer']):
            intents.append(SurveyIntent.TECHNOLOGY_USAGE)
        
        # Lifestyle indicators
        if any(word in all_text for word in ['lifestyle', 'hobbies', 'interests', 'activities', 'leisure']):
            intents.append(SurveyIntent.LIFESTYLE)
        
        # Product feedback indicators
        if any(word in all_text for word in ['product', 'service', 'feedback', 'review', 'experience', 'satisfaction']):
            intents.append(SurveyIntent.PRODUCT_FEEDBACK)
        
        # Default to market research if no specific intent found
        if not intents:
            intents.append(SurveyIntent.MARKET_RESEARCH)
        
        return intents
    
    async def check_answer_consistency(self, survey_id: str, question_text: str, 
                                     proposed_answer: str) -> Tuple[bool, float, List[str]]:
        """Check if proposed answer is consistent with previous responses."""
        related_memories = self.get_related_memories(survey_id, question_text)
        
        if not related_memories:
            return True, 1.0, []  # No previous context, assume consistent
        
        # Simple consistency check - in production would use more sophisticated NLP
        inconsistencies = []
        consistency_score = 1.0
        
        for memory in related_memories:
            # Check for direct contradictions
            if memory.memory_type == "answer":
                # Simple contradiction detection
                if "never" in memory.content.lower() and "always" in proposed_answer.lower():
                    inconsistencies.append(f"Contradicts previous answer: {memory.content}")
                    consistency_score *= 0.5
                
                if "love" in memory.content.lower() and "hate" in proposed_answer.lower():
                    inconsistencies.append(f"Contradicts previous preference: {memory.content}")
                    consistency_score *= 0.7
        
        is_consistent = consistency_score > 0.6
        return is_consistent, consistency_score, inconsistencies
    
    async def generate_reasoning(self, question_text: str, question_type: QuestionType,
                               options: List[str], selected_answer: str,
                               survey_context: SurveyContext, related_memories: List[SurveyMemory]) -> str:
        """Generate reasoning for why a particular answer was selected."""
        
        # Build context from memories
        memory_context = ""
        if related_memories:
            memory_context = "Previous responses: " + "; ".join([m.content for m in related_memories[:3]])
        
        # Generate reasoning based on question type and context
        if question_type == QuestionType.DEMOGRAPHIC:
            reasoning = f"Selected demographic answer '{selected_answer}' based on persona profile"
        
        elif question_type == QuestionType.BEHAVIORAL:
            reasoning = f"Selected '{selected_answer}' as it aligns with established behavioral patterns"
            if memory_context:
                reasoning += f" and is consistent with {memory_context}"
        
        elif question_type == QuestionType.PREFERENCE:
            reasoning = f"Chose '{selected_answer}' based on persona preferences"
            if any(intent in survey_context.inferred_intents for intent in [SurveyIntent.BRAND_LOYALTY, SurveyIntent.SHOPPING_HABITS]):
                reasoning += " and shopping behavior patterns"
        
        elif question_type == QuestionType.OPINION:
            reasoning = f"Selected opinion '{selected_answer}' to maintain consistency with persona values"
        
        elif question_type == QuestionType.RATING:
            reasoning = f"Rated '{selected_answer}' based on persona satisfaction patterns and previous ratings"
        
        elif question_type == QuestionType.ATTENTION_CHECK:
            reasoning = f"Correctly identified attention check and selected '{selected_answer}'"
        
        elif question_type == QuestionType.TRAP:
            reasoning = f"Detected potential trap question and selected '{selected_answer}' for consistency"
        
        else:
            reasoning = f"Selected '{selected_answer}' based on question context and persona alignment"
        
        return reasoning
    
    async def log_answer_reasoning(self, survey_id: str, question_id: str, question_text: str,
                                 question_type: QuestionType, selected_answer: str,
                                 confidence_score: float, consistency_check: bool,
                                 related_memories: List[SurveyMemory]):
        """Log the reasoning behind an answer selection."""
        
        survey_context = self.survey_contexts.get(survey_id)
        if not survey_context:
            # Create minimal context if none exists
            survey_context = SurveyContext(
                survey_id=survey_id,
                platform="unknown",
                estimated_length=0,
                complexity_score=0.5,
                inferred_intents=[SurveyIntent.MARKET_RESEARCH],
                target_demographics={},
                completion_time_estimate=10,
                risk_level="medium"
            )
        
        reasoning = await self.generate_reasoning(
            question_text, question_type, [], selected_answer,
            survey_context, related_memories
        )
        
        answer_reasoning = AnswerReasoning(
            question_id=question_id,
            question_text=question_text,
            question_type=question_type,
            selected_answer=selected_answer,
            reasoning=reasoning,
            confidence_score=confidence_score,
            consistency_check=consistency_check,
            related_memories=[m.memory_id for m in related_memories],
            timestamp=datetime.now()
        )
        
        self.reasoning_log.append(answer_reasoning)
        
        # Keep reasoning log manageable (last 1000 entries)
        if len(self.reasoning_log) > 1000:
            self.reasoning_log = self.reasoning_log[-1000:]
        
        await self._save_persistent_memory()
        
        logger.info(f"Logged reasoning for question {question_id}: {reasoning}")
    
    async def analyze_survey_complexity(self, questions: List[str], 
                                      estimated_length: int) -> Tuple[float, str]:
        """Analyze survey complexity and determine risk level."""
        complexity_factors = {
            'length': min(estimated_length / 50, 1.0),  # Normalize to 50 questions
            'text_complexity': 0.0,
            'question_variety': 0.0,
            'trap_questions': 0.0
        }
        
        # Analyze text complexity
        total_words = sum(len(q.split()) for q in questions)
        avg_words_per_question = total_words / max(len(questions), 1)
        complexity_factors['text_complexity'] = min(avg_words_per_question / 20, 1.0)
        
        # Analyze question variety
        question_types = set()
        trap_count = 0
        
        for question in questions:
            q_type = await self.classify_question_type(question)
            question_types.add(q_type)
            
            if q_type in [QuestionType.TRAP, QuestionType.ATTENTION_CHECK]:
                trap_count += 1
        
        complexity_factors['question_variety'] = len(question_types) / 8  # 8 total question types
        complexity_factors['trap_questions'] = min(trap_count / max(len(questions), 1), 0.3)
        
        # Calculate overall complexity score
        complexity_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        # Determine risk level
        if complexity_score < 0.3:
            risk_level = "low"
        elif complexity_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return complexity_score, risk_level
    
    async def create_survey_context(self, survey_id: str, platform: str, 
                                  questions: List[str] = None) -> SurveyContext:
        """Create comprehensive survey context."""
        if not questions:
            questions = []
        
        estimated_length = len(questions) if questions else 20  # Default estimate
        inferred_intents = await self.infer_survey_intent(questions) if questions else [SurveyIntent.MARKET_RESEARCH]
        complexity_score, risk_level = await self.analyze_survey_complexity(questions, estimated_length)
        
        # Estimate completion time (1-2 minutes per question on average)
        completion_time_estimate = max(estimated_length * 1.5, 5)  # Minimum 5 minutes
        
        context = SurveyContext(
            survey_id=survey_id,
            platform=platform,
            estimated_length=estimated_length,
            complexity_score=complexity_score,
            inferred_intents=inferred_intents,
            target_demographics={},  # Would be populated based on survey analysis
            completion_time_estimate=int(completion_time_estimate),
            risk_level=risk_level
        )
        
        self.survey_contexts[survey_id] = context
        await self._save_persistent_memory()
        
        logger.info(f"Created survey context for {survey_id}: {len(inferred_intents)} intents, {risk_level} risk")
        
        return context
    
    def get_reasoning_summary(self, survey_id: str) -> Dict[str, Any]:
        """Get summary of reasoning for a completed survey."""
        survey_reasoning = [r for r in self.reasoning_log if survey_id in r.question_id]
        
        if not survey_reasoning:
            return {}
        
        return {
            'total_questions': len(survey_reasoning),
            'avg_confidence': sum(r.confidence_score for r in survey_reasoning) / len(survey_reasoning),
            'consistency_rate': sum(1 for r in survey_reasoning if r.consistency_check) / len(survey_reasoning),
            'question_types': list(set(r.question_type.value for r in survey_reasoning)),
            'reasoning_samples': [
                {
                    'question': r.question_text[:100] + "..." if len(r.question_text) > 100 else r.question_text,
                    'answer': r.selected_answer,
                    'reasoning': r.reasoning
                }
                for r in survey_reasoning[:5]  # First 5 examples
            ]
        }
    
    async def cleanup_old_memories(self, days_old: int = 7):
        """Clean up old memories and reasoning logs."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean up reasoning log
        initial_reasoning_count = len(self.reasoning_log)
        self.reasoning_log = [r for r in self.reasoning_log if r.timestamp > cutoff_date]
        
        # Clean up memories
        initial_memory_count = sum(len(memories) for memories in self.short_term_memory.values())
        
        for survey_id in list(self.short_term_memory.keys()):
            self.short_term_memory[survey_id] = [
                m for m in self.short_term_memory[survey_id] 
                if m.created_at > cutoff_date
            ]
            
            # Remove empty survey memory lists
            if not self.short_term_memory[survey_id]:
                del self.short_term_memory[survey_id]
        
        final_memory_count = sum(len(memories) for memories in self.short_term_memory.values())
        
        removed_reasoning = initial_reasoning_count - len(self.reasoning_log)
        removed_memories = initial_memory_count - final_memory_count
        
        if removed_reasoning > 0 or removed_memories > 0:
            await self._save_persistent_memory()
            logger.info(f"Cleaned up {removed_reasoning} old reasoning entries and {removed_memories} old memories")