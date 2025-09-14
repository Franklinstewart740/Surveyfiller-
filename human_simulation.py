"""
Human Simulation & Answer Strategy System
Advanced behavioral modeling with persona-based responses, timing patterns, and anti-detection.
"""

import asyncio
import logging
import random
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import json

logger = logging.getLogger(__name__)

class PersonaProfile(Enum):
    """Detailed persona profiles for human simulation."""
    FRUGAL_SHOPPER = "frugal_shopper"
    TECH_SAVVY_MILLENNIAL = "tech_savvy_millennial"
    BUSY_PARENT = "busy_parent"
    HEALTH_CONSCIOUS = "health_conscious"
    LUXURY_SEEKER = "luxury_seeker"
    ENVIRONMENTALIST = "environmentalist"
    EARLY_ADOPTER = "early_adopter"
    TRADITIONAL_CONSERVATIVE = "traditional_conservative"

class QuestionCategory(Enum):
    """Categories for question classification."""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"
    OPINION = "opinion"
    RATING = "rating"
    TRAP_QUESTION = "trap_question"
    ATTENTION_CHECK = "attention_check"
    INCOME_SENSITIVE = "income_sensitive"
    BRAND_LOYALTY = "brand_loyalty"
    LIFESTYLE = "lifestyle"

class AnswerStrategy(Enum):
    """Answer strategy profiles."""
    AVOID_EXTREMES = "avoid_extremes"
    MID_RANGE_PREFERENCE = "mid_range_preference"
    CONSISTENT_PERSONA = "consistent_persona"
    SLIGHT_RANDOMIZATION = "slight_randomization"
    TRAP_AWARE = "trap_aware"
    BRAND_NEUTRAL = "brand_neutral"
    PRIVACY_CONSCIOUS = "privacy_conscious"

@dataclass
class TimingPattern:
    """Human-like timing patterns."""
    reading_speed_wpm: int  # Words per minute reading speed
    thinking_time_base: float  # Base thinking time in seconds
    thinking_time_variance: float  # Variance in thinking time
    typing_speed_cpm: int  # Characters per minute typing speed
    pause_probability: float  # Probability of pausing during typing
    idle_moments: List[Tuple[float, float]]  # (min_duration, max_duration) for idle moments
    mouse_movement_speed: float  # Pixels per second for mouse movement

@dataclass
class BehavioralFingerprint:
    """Unique behavioral fingerprint for each session."""
    session_id: str
    click_patterns: Dict[str, Any]
    scroll_patterns: Dict[str, Any]
    typing_patterns: Dict[str, Any]
    mouse_patterns: Dict[str, Any]
    interaction_sequences: List[str]
    timing_signature: Dict[str, float]
    created_at: datetime
    usage_count: int = 0

@dataclass
class PersonaCharacteristics:
    """Detailed characteristics for each persona."""
    name: str
    age_range: Tuple[int, int]
    income_bracket: str
    education_level: str
    tech_comfort: float  # 0-1 scale
    brand_loyalty: float  # 0-1 scale
    price_sensitivity: float  # 0-1 scale
    health_consciousness: float  # 0-1 scale
    environmental_concern: float  # 0-1 scale
    social_media_usage: float  # 0-1 scale
    shopping_frequency: str
    preferred_brands: List[str]
    avoided_brands: List[str]
    answer_tendencies: Dict[str, Any]
    timing_profile: TimingPattern

class HumanSimulation:
    """Advanced human behavior simulation system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_persona: Optional[PersonaCharacteristics] = None
        self.current_fingerprint: Optional[BehavioralFingerprint] = None
        self.answer_pools: Dict[str, List[str]] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Load persona definitions
        self.personas = self._load_persona_definitions()
        
        # Load answer pools
        self._load_answer_pools()
        
        # Behavioral fingerprint rotation
        self.fingerprint_pool: List[BehavioralFingerprint] = []
        self._generate_fingerprint_pool()
    
    def _load_persona_definitions(self) -> Dict[PersonaProfile, PersonaCharacteristics]:
        """Load detailed persona definitions."""
        return {
            PersonaProfile.FRUGAL_SHOPPER: PersonaCharacteristics(
                name="Frugal Shopper",
                age_range=(25, 55),
                income_bracket="$25,000-$50,000",
                education_level="High school or Some college",
                tech_comfort=0.6,
                brand_loyalty=0.3,
                price_sensitivity=0.9,
                health_consciousness=0.5,
                environmental_concern=0.4,
                social_media_usage=0.6,
                shopping_frequency="Weekly",
                preferred_brands=["Generic", "Store brand", "Walmart", "Target"],
                avoided_brands=["Luxury brands", "Premium brands"],
                answer_tendencies={
                    "income": ["$25,000-$35,000", "$35,000-$50,000"],
                    "shopping_frequency": ["Weekly", "Bi-weekly"],
                    "brand_importance": ["Not important", "Somewhat important"],
                    "price_importance": ["Very important", "Extremely important"]
                },
                timing_profile=TimingPattern(
                    reading_speed_wpm=200,
                    thinking_time_base=2.0,
                    thinking_time_variance=1.0,
                    typing_speed_cpm=180,
                    pause_probability=0.15,
                    idle_moments=[(1.0, 3.0), (0.5, 1.5)],
                    mouse_movement_speed=300
                )
            ),
            
            PersonaProfile.TECH_SAVVY_MILLENNIAL: PersonaCharacteristics(
                name="Tech-Savvy Millennial",
                age_range=(25, 40),
                income_bracket="$50,000-$100,000",
                education_level="Bachelor's degree or higher",
                tech_comfort=0.9,
                brand_loyalty=0.4,
                price_sensitivity=0.6,
                health_consciousness=0.7,
                environmental_concern=0.8,
                social_media_usage=0.9,
                shopping_frequency="Online frequently",
                preferred_brands=["Apple", "Google", "Tesla", "Patagonia", "Whole Foods"],
                avoided_brands=["Traditional brands", "Non-sustainable brands"],
                answer_tendencies={
                    "income": ["$50,000-$75,000", "$75,000-$100,000"],
                    "tech_usage": ["Daily", "Multiple times daily"],
                    "online_shopping": ["Weekly", "Multiple times per week"],
                    "sustainability": ["Very important", "Extremely important"]
                },
                timing_profile=TimingPattern(
                    reading_speed_wpm=250,
                    thinking_time_base=1.5,
                    thinking_time_variance=0.8,
                    typing_speed_cpm=220,
                    pause_probability=0.1,
                    idle_moments=[(0.5, 2.0), (0.3, 1.0)],
                    mouse_movement_speed=400
                )
            ),
            
            PersonaProfile.BUSY_PARENT: PersonaCharacteristics(
                name="Busy Parent",
                age_range=(30, 50),
                income_bracket="$40,000-$80,000",
                education_level="Some college or Bachelor's degree",
                tech_comfort=0.7,
                brand_loyalty=0.6,
                price_sensitivity=0.7,
                health_consciousness=0.8,
                environmental_concern=0.6,
                social_media_usage=0.5,
                shopping_frequency="Weekly with family focus",
                preferred_brands=["Target", "Costco", "Amazon", "Disney", "Gerber"],
                avoided_brands=["Time-consuming brands", "Complex products"],
                answer_tendencies={
                    "family_size": ["3-4 people", "4-5 people"],
                    "shopping_priority": ["Convenience", "Family-friendly"],
                    "time_availability": ["Limited", "Very limited"],
                    "health_focus": ["Important for family", "Very important"]
                },
                timing_profile=TimingPattern(
                    reading_speed_wpm=180,
                    thinking_time_base=2.5,
                    thinking_time_variance=1.5,
                    typing_speed_cpm=160,
                    pause_probability=0.2,
                    idle_moments=[(2.0, 5.0), (1.0, 3.0)],  # More interruptions
                    mouse_movement_speed=250
                )
            ),
            
            PersonaProfile.HEALTH_CONSCIOUS: PersonaCharacteristics(
                name="Health Conscious",
                age_range=(25, 65),
                income_bracket="$40,000-$100,000",
                education_level="Bachelor's degree or higher",
                tech_comfort=0.7,
                brand_loyalty=0.7,
                price_sensitivity=0.5,
                health_consciousness=0.95,
                environmental_concern=0.8,
                social_media_usage=0.6,
                shopping_frequency="Selective and research-based",
                preferred_brands=["Whole Foods", "Organic brands", "Fitbit", "Lululemon"],
                avoided_brands=["Fast food", "Processed food brands", "Unhealthy brands"],
                answer_tendencies={
                    "exercise_frequency": ["Daily", "5-6 times per week"],
                    "diet_type": ["Organic", "Plant-based", "Low-processed"],
                    "health_spending": ["Willing to pay more", "Premium for health"]
                },
                timing_profile=TimingPattern(
                    reading_speed_wpm=220,
                    thinking_time_base=2.0,
                    thinking_time_variance=1.0,
                    typing_speed_cpm=200,
                    pause_probability=0.12,
                    idle_moments=[(1.0, 2.5), (0.5, 1.5)],
                    mouse_movement_speed=320
                )
            )
        }
    
    def _load_answer_pools(self):
        """Load pools of answers for rotation."""
        self.answer_pools = {
            "age_ranges": [
                "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
            ],
            "income_ranges": [
                "Under $25,000", "$25,000-$35,000", "$35,000-$50,000",
                "$50,000-$75,000", "$75,000-$100,000", "$100,000-$150,000", "Over $150,000"
            ],
            "education_levels": [
                "High school", "Some college", "Associate degree",
                "Bachelor's degree", "Master's degree", "Doctoral degree"
            ],
            "employment_status": [
                "Full-time employed", "Part-time employed", "Self-employed",
                "Unemployed", "Student", "Retired", "Homemaker"
            ],
            "shopping_frequencies": [
                "Daily", "Several times a week", "Weekly", "Bi-weekly",
                "Monthly", "Several times a year", "Rarely"
            ],
            "satisfaction_levels": [
                "Very dissatisfied", "Dissatisfied", "Neutral",
                "Satisfied", "Very satisfied"
            ],
            "agreement_levels": [
                "Strongly disagree", "Disagree", "Neutral",
                "Agree", "Strongly agree"
            ],
            "frequency_levels": [
                "Never", "Rarely", "Sometimes", "Often", "Always"
            ],
            "importance_levels": [
                "Not important", "Slightly important", "Moderately important",
                "Very important", "Extremely important"
            ]
        }
    
    def _generate_fingerprint_pool(self, pool_size: int = 20):
        """Generate a pool of behavioral fingerprints for rotation."""
        for i in range(pool_size):
            fingerprint = BehavioralFingerprint(
                session_id=f"fp_{i:03d}",
                click_patterns={
                    "double_click_interval": random.uniform(0.1, 0.5),
                    "click_pressure_variance": random.uniform(0.8, 1.2),
                    "click_duration": random.uniform(0.05, 0.15)
                },
                scroll_patterns={
                    "scroll_speed": random.uniform(100, 500),
                    "scroll_acceleration": random.uniform(0.5, 2.0),
                    "scroll_direction_preference": random.choice(["smooth", "stepped"])
                },
                typing_patterns={
                    "keystroke_interval_mean": random.uniform(0.1, 0.3),
                    "keystroke_interval_std": random.uniform(0.02, 0.08),
                    "backspace_frequency": random.uniform(0.02, 0.1)
                },
                mouse_patterns={
                    "movement_smoothness": random.uniform(0.7, 1.0),
                    "acceleration_profile": random.choice(["linear", "curved", "stepped"]),
                    "hover_duration": random.uniform(0.5, 2.0)
                },
                interaction_sequences=[],
                timing_signature={
                    "page_load_wait": random.uniform(1.0, 3.0),
                    "element_focus_delay": random.uniform(0.2, 0.8),
                    "form_completion_pace": random.uniform(0.8, 2.0)
                },
                created_at=datetime.now()
            )
            self.fingerprint_pool.append(fingerprint)
    
    def set_persona(self, persona_profile: PersonaProfile):
        """Set the current persona for behavior simulation."""
        self.current_persona = self.personas[persona_profile]
        logger.info(f"Set persona to {self.current_persona.name}")
    
    def rotate_behavioral_fingerprint(self) -> BehavioralFingerprint:
        """Rotate to a new behavioral fingerprint."""
        # Select least used fingerprint
        available_fingerprints = [fp for fp in self.fingerprint_pool if fp.usage_count < 5]
        
        if not available_fingerprints:
            # Reset usage counts if all fingerprints are overused
            for fp in self.fingerprint_pool:
                fp.usage_count = 0
            available_fingerprints = self.fingerprint_pool
        
        # Select fingerprint with lowest usage
        selected_fingerprint = min(available_fingerprints, key=lambda fp: fp.usage_count)
        selected_fingerprint.usage_count += 1
        
        self.current_fingerprint = selected_fingerprint
        logger.info(f"Rotated to behavioral fingerprint {selected_fingerprint.session_id}")
        
        return selected_fingerprint
    
    async def classify_question(self, question_text: str, options: List[str] = None) -> QuestionCategory:
        """Classify question to determine appropriate response strategy."""
        question_lower = question_text.lower()
        
        # Demographic questions
        if any(word in question_lower for word in ['age', 'gender', 'income', 'education', 'occupation', 'location']):
            return QuestionCategory.DEMOGRAPHIC
        
        # Income-sensitive questions
        if any(word in question_lower for word in ['salary', 'earnings', 'household income', 'financial']):
            return QuestionCategory.INCOME_SENSITIVE
        
        # Brand loyalty questions
        if any(word in question_lower for word in ['brand', 'prefer', 'loyal', 'switch', 'recommend']):
            return QuestionCategory.BRAND_LOYALTY
        
        # Behavioral questions
        if any(word in question_lower for word in ['how often', 'frequency', 'usually', 'typically', 'behavior']):
            return QuestionCategory.BEHAVIORAL
        
        # Preference questions
        if any(word in question_lower for word in ['prefer', 'like', 'favorite', 'choose', 'select']):
            return QuestionCategory.PREFERENCE
        
        # Opinion questions
        if any(word in question_lower for word in ['opinion', 'think', 'believe', 'feel', 'agree']):
            return QuestionCategory.OPINION
        
        # Rating questions
        if any(word in question_lower for word in ['rate', 'rating', 'scale', 'score', 'satisfaction']):
            return QuestionCategory.RATING
        
        # Attention checks
        if any(phrase in question_lower for phrase in ['select the third option', 'choose option c', 'attention']):
            return QuestionCategory.ATTENTION_CHECK
        
        # Trap questions (look for contradictory or consistency-checking language)
        if any(phrase in question_lower for phrase in ['never said', 'previously answered', 'earlier you mentioned']):
            return QuestionCategory.TRAP_QUESTION
        
        # Lifestyle questions
        if any(word in question_lower for word in ['lifestyle', 'hobbies', 'interests', 'activities', 'leisure']):
            return QuestionCategory.LIFESTYLE
        
        return QuestionCategory.PREFERENCE  # Default category
    
    def apply_answer_strategy(self, question_category: QuestionCategory, 
                            options: List[str], question_text: str) -> Tuple[str, str]:
        """Apply persona-based answer strategy and return answer with reasoning."""
        if not self.current_persona:
            # Use default strategy if no persona set
            return self._apply_default_strategy(options, question_text)
        
        persona = self.current_persona
        
        if question_category == QuestionCategory.DEMOGRAPHIC:
            return self._handle_demographic_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.INCOME_SENSITIVE:
            return self._handle_income_question(options, persona)
        
        elif question_category == QuestionCategory.BRAND_LOYALTY:
            return self._handle_brand_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.BEHAVIORAL:
            return self._handle_behavioral_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.PREFERENCE:
            return self._handle_preference_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.OPINION:
            return self._handle_opinion_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.RATING:
            return self._handle_rating_question(question_text, options, persona)
        
        elif question_category == QuestionCategory.ATTENTION_CHECK:
            return self._handle_attention_check(question_text, options)
        
        elif question_category == QuestionCategory.TRAP_QUESTION:
            return self._handle_trap_question(question_text, options, persona)
        
        else:
            return self._apply_persona_strategy(options, persona)
    
    def _handle_demographic_question(self, question_text: str, options: List[str], 
                                   persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle demographic questions based on persona."""
        question_lower = question_text.lower()
        
        if 'age' in question_lower:
            # Select age range that fits persona
            age_min, age_max = persona.age_range
            for option in options:
                if any(str(age) in option for age in range(age_min, age_max + 1)):
                    return option, f"Selected age range {option} matching persona age range {persona.age_range}"
            
            # Fallback to closest match
            return random.choice(options), "Selected random age option as fallback"
        
        elif 'income' in question_lower:
            # Match persona income bracket
            for option in options:
                if persona.income_bracket.lower() in option.lower():
                    return option, f"Selected income {option} matching persona bracket {persona.income_bracket}"
            
            # Find closest income match
            return self._find_closest_income_match(options, persona.income_bracket)
        
        elif 'education' in question_lower:
            # Match persona education level
            for option in options:
                if persona.education_level.lower() in option.lower():
                    return option, f"Selected education {option} matching persona level {persona.education_level}"
            
            return random.choice(options), "Selected random education option as fallback"
        
        else:
            return self._apply_persona_strategy(options, persona)
    
    def _handle_income_question(self, options: List[str], persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle income-sensitive questions with privacy considerations."""
        # Apply privacy consciousness - sometimes refuse to answer income questions
        if random.random() < 0.2:  # 20% chance to prefer not to answer
            privacy_options = [opt for opt in options if 'prefer not' in opt.lower() or 'decline' in opt.lower()]
            if privacy_options:
                return random.choice(privacy_options), "Chose privacy option for income question"
        
        return self._find_closest_income_match(options, persona.income_bracket)
    
    def _find_closest_income_match(self, options: List[str], target_bracket: str) -> Tuple[str, str]:
        """Find the closest income match to the persona's bracket."""
        # Extract target income range
        target_numbers = [int(x.replace(',', '').replace('$', '')) for x in target_bracket.split('-') if x.replace(',', '').replace('$', '').isdigit()]
        
        if not target_numbers:
            return random.choice(options), "Could not parse target income, selected random option"
        
        target_mid = sum(target_numbers) / len(target_numbers)
        
        best_match = None
        best_score = float('inf')
        
        for option in options:
            option_numbers = [int(x.replace(',', '').replace('$', '')) for x in option.split() if x.replace(',', '').replace('$', '').isdigit()]
            if option_numbers:
                option_mid = sum(option_numbers) / len(option_numbers)
                score = abs(option_mid - target_mid)
                if score < best_score:
                    best_score = score
                    best_match = option
        
        if best_match:
            return best_match, f"Selected closest income match {best_match} to target {target_bracket}"
        
        return random.choice(options), "Could not find close income match, selected random option"
    
    def _handle_brand_question(self, question_text: str, options: List[str], 
                             persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle brand-related questions based on persona preferences."""
        question_lower = question_text.lower()
        
        # Check for preferred brands
        for option in options:
            for preferred_brand in persona.preferred_brands:
                if preferred_brand.lower() in option.lower():
                    return option, f"Selected preferred brand {option} matching persona preference for {preferred_brand}"
        
        # Avoid disliked brands
        safe_options = []
        for option in options:
            is_safe = True
            for avoided_brand in persona.avoided_brands:
                if avoided_brand.lower() in option.lower():
                    is_safe = False
                    break
            if is_safe:
                safe_options.append(option)
        
        if safe_options:
            # Apply brand loyalty factor
            if persona.brand_loyalty > 0.7 and len(safe_options) > 1:
                # High brand loyalty - prefer established options
                established_options = [opt for opt in safe_options if any(word in opt.lower() for word in ['well-known', 'established', 'trusted'])]
                if established_options:
                    return random.choice(established_options), "Selected established brand due to high brand loyalty"
            
            return random.choice(safe_options), "Selected brand option avoiding persona dislikes"
        
        # Fallback to neutral options
        neutral_options = [opt for opt in options if any(word in opt.lower() for word in ['no preference', 'other', 'neutral'])]
        if neutral_options:
            return random.choice(neutral_options), "Selected neutral brand option as fallback"
        
        return random.choice(options), "Selected random brand option as last resort"
    
    def _handle_behavioral_question(self, question_text: str, options: List[str], 
                                  persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle behavioral questions based on persona characteristics."""
        question_lower = question_text.lower()
        
        # Shopping frequency questions
        if any(word in question_lower for word in ['shop', 'buy', 'purchase']):
            shopping_freq = persona.shopping_frequency.lower()
            for option in options:
                if any(word in option.lower() for word in shopping_freq.split()):
                    return option, f"Selected shopping frequency {option} matching persona pattern {persona.shopping_frequency}"
        
        # Technology usage questions
        if any(word in question_lower for word in ['technology', 'internet', 'smartphone', 'app']):
            if persona.tech_comfort > 0.8:
                tech_options = [opt for opt in options if any(word in opt.lower() for word in ['daily', 'frequently', 'often', 'multiple times'])]
                if tech_options:
                    return random.choice(tech_options), f"Selected high-tech usage option due to persona tech comfort {persona.tech_comfort}"
            elif persona.tech_comfort < 0.4:
                low_tech_options = [opt for opt in options if any(word in opt.lower() for word in ['rarely', 'never', 'occasionally'])]
                if low_tech_options:
                    return random.choice(low_tech_options), f"Selected low-tech usage option due to persona tech comfort {persona.tech_comfort}"
        
        # Social media questions
        if any(word in question_lower for word in ['social media', 'facebook', 'instagram', 'twitter']):
            if persona.social_media_usage > 0.7:
                active_options = [opt for opt in options if any(word in opt.lower() for word in ['daily', 'multiple times', 'very active'])]
                if active_options:
                    return random.choice(active_options), f"Selected active social media option due to high usage score {persona.social_media_usage}"
        
        return self._apply_persona_strategy(options, persona)
    
    def _handle_preference_question(self, question_text: str, options: List[str], 
                                  persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle preference questions with persona-based logic."""
        question_lower = question_text.lower()
        
        # Health-related preferences
        if any(word in question_lower for word in ['health', 'organic', 'natural', 'wellness']):
            if persona.health_consciousness > 0.7:
                health_options = [opt for opt in options if any(word in opt.lower() for word in ['organic', 'natural', 'healthy', 'wellness'])]
                if health_options:
                    return random.choice(health_options), f"Selected health-conscious option due to persona health score {persona.health_consciousness}"
        
        # Environmental preferences
        if any(word in question_lower for word in ['environment', 'sustainable', 'eco', 'green']):
            if persona.environmental_concern > 0.7:
                eco_options = [opt for opt in options if any(word in opt.lower() for word in ['sustainable', 'eco', 'green', 'environmental'])]
                if eco_options:
                    return random.choice(eco_options), f"Selected eco-friendly option due to persona environmental concern {persona.environmental_concern}"
        
        # Price-sensitive preferences
        if any(word in question_lower for word in ['price', 'cost', 'expensive', 'cheap', 'budget']):
            if persona.price_sensitivity > 0.7:
                budget_options = [opt for opt in options if any(word in opt.lower() for word in ['affordable', 'budget', 'value', 'cheap', 'low cost'])]
                if budget_options:
                    return random.choice(budget_options), f"Selected budget option due to high price sensitivity {persona.price_sensitivity}"
        
        return self._apply_persona_strategy(options, persona)
    
    def _handle_opinion_question(self, question_text: str, options: List[str], 
                               persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle opinion questions with moderate responses to avoid extremes."""
        # Apply "avoid extremes" strategy
        if len(options) >= 5:
            # Remove extreme options (first and last)
            moderate_options = options[1:-1]
            if moderate_options:
                return random.choice(moderate_options), "Selected moderate opinion to avoid extremes"
        
        elif len(options) >= 3:
            # Select middle option if available
            middle_index = len(options) // 2
            return options[middle_index], "Selected middle opinion option"
        
        return random.choice(options), "Selected random opinion option"
    
    def _handle_rating_question(self, question_text: str, options: List[str], 
                              persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle rating questions with persona-based tendencies."""
        # Look for numeric ratings
        numeric_options = []
        for option in options:
            if any(char.isdigit() for char in option):
                numeric_options.append(option)
        
        if numeric_options:
            # Apply mid-range preference for ratings
            if len(numeric_options) >= 5:
                # Select from middle 60% of options
                start_idx = len(numeric_options) // 5
                end_idx = len(numeric_options) - start_idx
                mid_range_options = numeric_options[start_idx:end_idx]
                if mid_range_options:
                    return random.choice(mid_range_options), "Selected mid-range rating to appear balanced"
        
        # Handle satisfaction ratings
        if any(word in question_text.lower() for word in ['satisfied', 'satisfaction']):
            # Slightly positive bias for most personas
            positive_options = [opt for opt in options if any(word in opt.lower() for word in ['satisfied', 'good', 'positive'])]
            if positive_options and random.random() < 0.7:  # 70% chance for positive
                return random.choice(positive_options), "Selected positive satisfaction rating"
        
        return self._apply_persona_strategy(options, persona)
    
    def _handle_attention_check(self, question_text: str, options: List[str]) -> Tuple[str, str]:
        """Handle attention check questions correctly."""
        question_lower = question_text.lower()
        
        # Look for specific instructions
        if 'third option' in question_lower or 'option c' in question_lower:
            if len(options) >= 3:
                return options[2], "Correctly identified attention check for third option"
        
        if 'second option' in question_lower or 'option b' in question_lower:
            if len(options) >= 2:
                return options[1], "Correctly identified attention check for second option"
        
        if 'first option' in question_lower or 'option a' in question_lower:
            if len(options) >= 1:
                return options[0], "Correctly identified attention check for first option"
        
        # Look for specific text to select
        for option in options:
            if option.lower() in question_lower:
                return option, f"Correctly identified attention check for specific text: {option}"
        
        # Fallback - select a middle option
        if len(options) >= 3:
            return options[len(options) // 2], "Selected middle option for unrecognized attention check"
        
        return random.choice(options), "Could not identify attention check pattern, selected random option"
    
    def _handle_trap_question(self, question_text: str, options: List[str], 
                            persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Handle trap questions designed to catch inconsistent responses."""
        # Look for references to previous answers
        if any(phrase in question_text.lower() for phrase in ['previously', 'earlier', 'before']):
            # Try to maintain consistency with persona
            consistent_options = []
            
            # Check interaction history for related answers
            for interaction in self.interaction_history[-10:]:  # Check last 10 interactions
                if interaction.get('question_category') in ['demographic', 'preference', 'behavioral']:
                    previous_answer = interaction.get('selected_answer', '').lower()
                    
                    # Look for options that align with previous answers
                    for option in options:
                        if any(word in option.lower() for word in previous_answer.split()):
                            consistent_options.append(option)
            
            if consistent_options:
                return random.choice(consistent_options), "Selected option consistent with previous answers to avoid trap"
        
        # Default to persona-based strategy
        return self._apply_persona_strategy(options, persona)
    
    def _apply_persona_strategy(self, options: List[str], persona: PersonaCharacteristics) -> Tuple[str, str]:
        """Apply general persona-based strategy."""
        # Check persona answer tendencies first
        for tendency_key, tendency_values in persona.answer_tendencies.items():
            for option in options:
                if any(tendency.lower() in option.lower() for tendency in tendency_values):
                    return option, f"Selected option {option} matching persona tendency {tendency_key}"
        
        # Apply slight randomization to avoid patterns
        if random.random() < 0.1:  # 10% chance for random selection
            return random.choice(options), "Applied slight randomization to avoid detection patterns"
        
        # Default to middle options when available
        if len(options) >= 3:
            middle_options = options[1:-1] if len(options) > 3 else [options[len(options) // 2]]
            return random.choice(middle_options), "Selected middle option as default strategy"
        
        return random.choice(options), "Selected random option as fallback"
    
    def _apply_default_strategy(self, options: List[str], question_text: str) -> Tuple[str, str]:
        """Apply default strategy when no persona is set."""
        # Avoid extremes
        if len(options) >= 5:
            moderate_options = options[1:-1]
            return random.choice(moderate_options), "Applied default avoid-extremes strategy"
        
        # Select middle option
        if len(options) >= 3:
            middle_index = len(options) // 2
            return options[middle_index], "Applied default middle-option strategy"
        
        return random.choice(options), "Applied default random selection"
    
    async def calculate_human_timing(self, question_text: str, selected_answer: str) -> Dict[str, float]:
        """Calculate human-like timing for question interaction."""
        if not self.current_persona:
            # Default timing
            return {
                'reading_time': random.uniform(2.0, 5.0),
                'thinking_time': random.uniform(1.0, 3.0),
                'selection_time': random.uniform(0.5, 1.5),
                'total_time': 0.0
            }
        
        timing_profile = self.current_persona.timing_profile
        
        # Calculate reading time based on text length and reading speed
        word_count = len(question_text.split())
        reading_time = (word_count / timing_profile.reading_speed_wpm) * 60  # Convert to seconds
        reading_time += random.uniform(0.5, 2.0)  # Add base reading overhead
        
        # Calculate thinking time with variance
        thinking_time = timing_profile.thinking_time_base + random.gauss(0, timing_profile.thinking_time_variance)
        thinking_time = max(0.5, thinking_time)  # Minimum thinking time
        
        # Calculate selection time (mouse movement + click)
        selection_time = random.uniform(0.3, 1.0)
        
        # Add random idle moments
        idle_time = 0.0
        if random.random() < 0.3:  # 30% chance of idle moment
            idle_duration_range = random.choice(timing_profile.idle_moments)
            idle_time = random.uniform(idle_duration_range[0], idle_duration_range[1])
        
        total_time = reading_time + thinking_time + selection_time + idle_time
        
        return {
            'reading_time': reading_time,
            'thinking_time': thinking_time,
            'selection_time': selection_time,
            'idle_time': idle_time,
            'total_time': total_time
        }
    
    async def generate_curved_mouse_path(self, start_pos: Tuple[int, int], 
                                       end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate human-like curved mouse movement path."""
        if not self.current_fingerprint:
            # Generate simple straight path
            return [start_pos, end_pos]
        
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate distance and number of points
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        num_points = max(5, int(distance / 20))  # One point every 20 pixels
        
        path_points = []
        
        # Generate curved path using Bezier curve
        for i in range(num_points + 1):
            t = i / num_points
            
            # Add some randomness to create natural curve
            curve_offset_x = random.gauss(0, distance * 0.1) * math.sin(t * math.pi)
            curve_offset_y = random.gauss(0, distance * 0.1) * math.sin(t * math.pi)
            
            # Linear interpolation with curve offset
            x = start_x + (end_x - start_x) * t + curve_offset_x
            y = start_y + (end_y - start_y) * t + curve_offset_y
            
            path_points.append((int(x), int(y)))
        
        return path_points
    
    def log_interaction(self, question_text: str, question_category: QuestionCategory,
                       selected_answer: str, reasoning: str, timing: Dict[str, float]):
        """Log interaction for consistency tracking."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'question_text': question_text[:200],  # Truncate for storage
            'question_category': question_category.value,
            'selected_answer': selected_answer,
            'reasoning': reasoning,
            'timing': timing,
            'persona': self.current_persona.name if self.current_persona else None,
            'fingerprint_id': self.current_fingerprint.session_id if self.current_fingerprint else None
        }
        
        self.interaction_history.append(interaction)
        
        # Keep history manageable
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of recent interactions."""
        if not self.interaction_history:
            return {'total_interactions': 0}
        
        recent_interactions = self.interaction_history[-20:]  # Last 20 interactions
        
        category_counts = {}
        total_time = 0.0
        
        for interaction in recent_interactions:
            category = interaction['question_category']
            category_counts[category] = category_counts.get(category, 0) + 1
            total_time += interaction['timing']['total_time']
        
        return {
            'total_interactions': len(self.interaction_history),
            'recent_interactions': len(recent_interactions),
            'category_distribution': category_counts,
            'average_response_time': total_time / len(recent_interactions) if recent_interactions else 0,
            'current_persona': self.current_persona.name if self.current_persona else None,
            'current_fingerprint': self.current_fingerprint.session_id if self.current_fingerprint else None
        }