"""
Enhanced AI Response Generator
Advanced AI system for generating high-quality, contextual survey responses.
"""

import asyncio
import logging
import random
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from enum import Enum

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """Different persona types for response generation."""
    YOUNG_PROFESSIONAL = "young_professional"
    MIDDLE_AGED_PARENT = "middle_aged_parent"
    COLLEGE_STUDENT = "college_student"
    RETIREE = "retiree"
    TECH_ENTHUSIAST = "tech_enthusiast"
    BUDGET_CONSCIOUS = "budget_conscious"
    HEALTH_CONSCIOUS = "health_conscious"
    ENVIRONMENTALIST = "environmentalist"

@dataclass
class Persona:
    """Represents a user persona for consistent responses."""
    name: str
    age_range: Tuple[int, int]
    income_range: str
    education: str
    location_type: str
    interests: List[str]
    values: List[str]
    shopping_habits: List[str]
    media_consumption: List[str]
    personality_traits: List[str]
    response_patterns: Dict[str, Any]

@dataclass
class SurveyQuestion:
    """Enhanced survey question representation."""
    question_id: str
    question_text: str
    question_type: str
    options: List[str]
    category: str
    context: Dict[str, Any]
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None

@dataclass
class ResponseContext:
    """Context for generating responses."""
    previous_responses: List[Dict[str, Any]]
    survey_topic: str
    brand_context: Optional[str]
    time_spent_so_far: int
    question_number: int
    total_questions: int

class EnhancedAIGenerator:
    """Enhanced AI response generator with persona management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.current_persona = None
        self.personas = self._load_personas()
        self.response_history = []
        self.consistency_tracker = {}
        
        # Initialize AI models
        self.openai_client = None
        self.local_model = None
        self.tokenizer = None
        
        # Response quality settings
        self.quality_threshold = 0.8
        self.consistency_weight = 0.3
        self.creativity_weight = 0.4
        self.realism_weight = 0.3
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models based on configuration."""
        try:
            # Initialize OpenAI if API key provided
            if self.config.get("openai_api_key"):
                openai.api_key = self.config["openai_api_key"]
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            
            # Initialize local model as fallback
            model_name = self.config.get("local_model", "microsoft/DialoGPT-medium")
            if torch.cuda.is_available():
                self.local_model = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0,
                    torch_dtype=torch.float16
                )
            else:
                self.local_model = pipeline("text-generation", model=model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Local model {model_name} initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
    
    def _load_personas(self) -> Dict[PersonaType, Persona]:
        """Load predefined personas."""
        personas = {}
        
        # Young Professional
        personas[PersonaType.YOUNG_PROFESSIONAL] = Persona(
            name="Alex Chen",
            age_range=(25, 35),
            income_range="$50k-$80k",
            education="Bachelor's Degree",
            location_type="Urban",
            interests=["Technology", "Career Development", "Fitness", "Travel"],
            values=["Efficiency", "Innovation", "Work-life balance"],
            shopping_habits=["Online shopping", "Brand conscious", "Value for money"],
            media_consumption=["Social media", "Podcasts", "Streaming services"],
            personality_traits=["Ambitious", "Tech-savvy", "Social"],
            response_patterns={
                "length_preference": "medium",
                "detail_level": "moderate",
                "enthusiasm": "high",
                "brand_loyalty": "medium"
            }
        )
        
        # Middle-aged Parent
        personas[PersonaType.MIDDLE_AGED_PARENT] = Persona(
            name="Sarah Johnson",
            age_range=(35, 50),
            income_range="$60k-$100k",
            education="Some College",
            location_type="Suburban",
            interests=["Family", "Home improvement", "Cooking", "Local community"],
            values=["Family first", "Practicality", "Safety", "Value"],
            shopping_habits=["Bulk buying", "Coupon usage", "Family-focused"],
            media_consumption=["Facebook", "Local news", "Family shows"],
            personality_traits=["Practical", "Caring", "Budget-conscious"],
            response_patterns={
                "length_preference": "detailed",
                "detail_level": "high",
                "enthusiasm": "moderate",
                "brand_loyalty": "high"
            }
        )
        
        # College Student
        personas[PersonaType.COLLEGE_STUDENT] = Persona(
            name="Jordan Martinez",
            age_range=(18, 24),
            income_range="Under $25k",
            education="Some College",
            location_type="College Town",
            interests=["Gaming", "Social media", "Music", "Sports"],
            values=["Fun", "Affordability", "Trends", "Social acceptance"],
            shopping_habits=["Budget shopping", "Trend following", "Online deals"],
            media_consumption=["TikTok", "Instagram", "YouTube", "Gaming streams"],
            personality_traits=["Trendy", "Social", "Budget-conscious"],
            response_patterns={
                "length_preference": "short",
                "detail_level": "low",
                "enthusiasm": "high",
                "brand_loyalty": "low"
            }
        )
        
        # Add more personas...
        personas[PersonaType.RETIREE] = Persona(
            name="Robert Williams",
            age_range=(65, 80),
            income_range="$40k-$70k",
            education="High School",
            location_type="Suburban",
            interests=["Gardening", "Reading", "Grandchildren", "Travel"],
            values=["Tradition", "Quality", "Reliability", "Simplicity"],
            shopping_habits=["In-store shopping", "Brand loyalty", "Quality over price"],
            media_consumption=["Traditional TV", "Newspapers", "Radio"],
            personality_traits=["Traditional", "Loyal", "Cautious"],
            response_patterns={
                "length_preference": "detailed",
                "detail_level": "high",
                "enthusiasm": "moderate",
                "brand_loyalty": "very_high"
            }
        )
        
        return personas
    
    def set_persona(self, persona_type: PersonaType) -> bool:
        """Set the current persona for response generation."""
        if persona_type in self.personas:
            self.current_persona = self.personas[persona_type]
            self.consistency_tracker = {}
            logger.info(f"Set persona to {self.current_persona.name}")
            return True
        return False
    
    def get_available_personas(self) -> List[str]:
        """Get list of available persona names."""
        return [persona.value for persona in PersonaType]
    
    async def generate_response(self, question: SurveyQuestion, context: ResponseContext = None) -> Optional[str]:
        """Generate a high-quality response to a survey question."""
        if not self.current_persona:
            # Set default persona
            self.set_persona(PersonaType.YOUNG_PROFESSIONAL)
        
        try:
            # Analyze question to determine best approach
            question_analysis = self._analyze_question(question)
            
            # Generate multiple candidate responses
            candidates = await self._generate_candidate_responses(question, context, question_analysis)
            
            # Score and select best response
            best_response = self._select_best_response(candidates, question, context)
            
            # Post-process for consistency and quality
            final_response = self._post_process_response(best_response, question, context)
            
            # Update response history
            self._update_response_history(question, final_response, context)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(question)
    
    def _analyze_question(self, question: SurveyQuestion) -> Dict[str, Any]:
        """Analyze question to determine response strategy."""
        analysis = {
            "complexity": "medium",
            "requires_consistency": False,
            "demographic_related": False,
            "brand_related": False,
            "opinion_based": False,
            "factual": False
        }
        
        question_lower = question.question_text.lower()
        
        # Check for demographic questions
        demographic_keywords = ["age", "income", "education", "location", "gender", "occupation"]
        if any(keyword in question_lower for keyword in demographic_keywords):
            analysis["demographic_related"] = True
            analysis["requires_consistency"] = True
        
        # Check for brand-related questions
        brand_keywords = ["brand", "company", "product", "service", "prefer"]
        if any(keyword in question_lower for keyword in brand_keywords):
            analysis["brand_related"] = True
        
        # Check for opinion questions
        opinion_keywords = ["think", "feel", "opinion", "believe", "rate", "satisfaction"]
        if any(keyword in question_lower for keyword in opinion_keywords):
            analysis["opinion_based"] = True
        
        # Determine complexity
        if len(question.options) > 7 or len(question.question_text) > 200:
            analysis["complexity"] = "high"
        elif len(question.options) <= 3 or len(question.question_text) < 50:
            analysis["complexity"] = "low"
        
        return analysis
    
    async def _generate_candidate_responses(self, question: SurveyQuestion, context: ResponseContext, analysis: Dict[str, Any]) -> List[str]:
        """Generate multiple candidate responses."""
        candidates = []
        
        # Generate responses using different methods
        if self.openai_client:
            openai_response = await self._generate_openai_response(question, context, analysis)
            if openai_response:
                candidates.append(openai_response)
        
        if self.local_model:
            local_response = await self._generate_local_response(question, context, analysis)
            if local_response:
                candidates.append(local_response)
        
        # Generate rule-based response as fallback
        rule_based_response = self._generate_rule_based_response(question, analysis)
        if rule_based_response:
            candidates.append(rule_based_response)
        
        return candidates
    
    async def _generate_openai_response(self, question: SurveyQuestion, context: ResponseContext, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate response using OpenAI."""
        try:
            prompt = self._build_openai_prompt(question, context, analysis)
            
            response = await self.openai_client.ChatCompletion.acreate(
                model=self.config.get("openai_model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error with OpenAI generation: {e}")
            return None
    
    async def _generate_local_response(self, question: SurveyQuestion, context: ResponseContext, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate response using local model."""
        try:
            prompt = self._build_local_prompt(question, context, analysis)
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.local_model(
                    prompt,
                    max_length=len(prompt.split()) + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the new part
            new_text = generated_text[len(prompt):].strip()
            
            return new_text
            
        except Exception as e:
            logger.error(f"Error with local model generation: {e}")
            return None
    
    def _generate_rule_based_response(self, question: SurveyQuestion, analysis: Dict[str, Any]) -> str:
        """Generate response using rule-based approach."""
        if question.question_type == "single_choice" and question.options:
            return self._select_option_based_on_persona(question.options, analysis)
        
        elif question.question_type == "multiple_choice" and question.options:
            num_selections = random.randint(1, min(3, len(question.options)))
            selected = random.sample(question.options, num_selections)
            return ", ".join(selected)
        
        elif question.question_type in ["text_input", "textarea"]:
            return self._generate_text_response_by_persona(question, analysis)
        
        elif question.question_type == "rating_scale":
            return str(self._generate_rating_by_persona(question, analysis))
        
        else:
            return random.choice(question.options) if question.options else "Not sure"
    
    def _select_option_based_on_persona(self, options: List[str], analysis: Dict[str, Any]) -> str:
        """Select option based on current persona characteristics."""
        if not self.current_persona:
            return random.choice(options)
        
        # Demographic consistency
        if analysis["demographic_related"]:
            return self._get_consistent_demographic_response(options)
        
        # Brand preferences
        if analysis["brand_related"]:
            return self._get_brand_preference_response(options)
        
        # Opinion-based responses
        if analysis["opinion_based"]:
            return self._get_opinion_response(options)
        
        # Default weighted selection
        return self._weighted_option_selection(options)
    
    def _get_consistent_demographic_response(self, options: List[str]) -> str:
        """Get consistent demographic response."""
        persona = self.current_persona
        
        # Age-related consistency
        age_options = [opt for opt in options if any(age_term in opt.lower() for age_term in ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])]
        if age_options:
            target_age = sum(persona.age_range) // 2
            if target_age < 25:
                return next((opt for opt in age_options if "18-24" in opt), random.choice(age_options))
            elif target_age < 35:
                return next((opt for opt in age_options if "25-34" in opt), random.choice(age_options))
            elif target_age < 45:
                return next((opt for opt in age_options if "35-44" in opt), random.choice(age_options))
            elif target_age < 55:
                return next((opt for opt in age_options if "45-54" in opt), random.choice(age_options))
            elif target_age < 65:
                return next((opt for opt in age_options if "55-64" in opt), random.choice(age_options))
            else:
                return next((opt for opt in age_options if "65+" in opt), random.choice(age_options))
        
        # Income-related consistency
        income_options = [opt for opt in options if "$" in opt or "income" in opt.lower()]
        if income_options:
            return next((opt for opt in income_options if persona.income_range.replace("$", "").replace("k", "000") in opt), random.choice(income_options))
        
        # Education consistency
        education_options = [opt for opt in options if any(edu_term in opt.lower() for edu_term in ["high school", "college", "bachelor", "master", "phd"])]
        if education_options:
            return next((opt for opt in education_options if persona.education.lower() in opt.lower()), random.choice(education_options))
        
        return random.choice(options)
    
    def _get_brand_preference_response(self, options: List[str]) -> str:
        """Get brand preference based on persona."""
        persona = self.current_persona
        loyalty_level = persona.response_patterns.get("brand_loyalty", "medium")
        
        # High loyalty personas prefer established brands
        if loyalty_level == "very_high":
            established_brands = ["Apple", "Google", "Amazon", "Microsoft", "Coca-Cola", "Nike"]
            for option in options:
                if any(brand in option for brand in established_brands):
                    return option
        
        # Budget-conscious personas prefer value options
        if "Budget-conscious" in persona.personality_traits:
            budget_keywords = ["affordable", "cheap", "value", "discount", "budget"]
            for option in options:
                if any(keyword in option.lower() for keyword in budget_keywords):
                    return option
        
        return random.choice(options)
    
    def _get_opinion_response(self, options: List[str]) -> str:
        """Get opinion response based on persona traits."""
        persona = self.current_persona
        enthusiasm = persona.response_patterns.get("enthusiasm", "medium")
        
        # High enthusiasm personas prefer positive options
        if enthusiasm == "high":
            positive_keywords = ["excellent", "great", "love", "amazing", "fantastic", "very satisfied"]
            for option in options:
                if any(keyword in option.lower() for keyword in positive_keywords):
                    if random.random() < 0.7:  # 70% chance to select positive
                        return option
        
        # Moderate enthusiasm prefers middle options
        elif enthusiasm == "moderate":
            moderate_keywords = ["good", "okay", "satisfied", "neutral", "average"]
            for option in options:
                if any(keyword in option.lower() for keyword in moderate_keywords):
                    if random.random() < 0.6:  # 60% chance to select moderate
                        return option
        
        return random.choice(options)
    
    def _weighted_option_selection(self, options: List[str]) -> str:
        """Select option with weighted probability based on persona."""
        # Simple weighted selection - can be enhanced
        weights = [1.0] * len(options)
        
        # Adjust weights based on persona traits
        persona = self.current_persona
        for i, option in enumerate(options):
            option_lower = option.lower()
            
            # Boost options matching persona interests
            for interest in persona.interests:
                if interest.lower() in option_lower:
                    weights[i] *= 1.5
            
            # Boost options matching persona values
            for value in persona.values:
                if value.lower() in option_lower:
                    weights[i] *= 1.3
        
        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return options[i]
        
        return options[-1]  # Fallback
    
    def _generate_text_response_by_persona(self, question: SurveyQuestion, analysis: Dict[str, Any]) -> str:
        """Generate text response based on persona."""
        persona = self.current_persona
        length_pref = persona.response_patterns.get("length_preference", "medium")
        detail_level = persona.response_patterns.get("detail_level", "moderate")
        
        # Base responses by question category
        base_responses = {
            "satisfaction": [
                "I'm quite satisfied with the service overall.",
                "It meets my expectations most of the time.",
                "There's room for improvement but it's decent.",
                "I've had mostly positive experiences."
            ],
            "improvement": [
                "Better customer service would be great.",
                "More options and flexibility would help.",
                "Faster response times are needed.",
                "Lower prices would make it more attractive."
            ],
            "experience": [
                "My experience has been generally positive.",
                "I've encountered some issues but nothing major.",
                "It's been reliable and consistent.",
                "I would recommend it to others."
            ]
        }
        
        # Select appropriate base response
        question_lower = question.question_text.lower()
        if "satisfaction" in question_lower or "satisfied" in question_lower:
            responses = base_responses["satisfaction"]
        elif "improve" in question_lower or "better" in question_lower:
            responses = base_responses["improvement"]
        else:
            responses = base_responses["experience"]
        
        base_response = random.choice(responses)
        
        # Adjust based on persona
        if length_pref == "detailed" and detail_level == "high":
            # Add more detail
            additional_details = [
                " I've been using this for a while now and have noticed consistent quality.",
                " The value for money is reasonable considering what you get.",
                " I appreciate the attention to customer needs.",
                " It fits well with my lifestyle and requirements."
            ]
            base_response += random.choice(additional_details)
        
        elif length_pref == "short":
            # Shorten response
            base_response = base_response.split('.')[0] + "."
        
        return base_response
    
    def _generate_rating_by_persona(self, question: SurveyQuestion, analysis: Dict[str, Any]) -> int:
        """Generate rating based on persona characteristics."""
        persona = self.current_persona
        enthusiasm = persona.response_patterns.get("enthusiasm", "medium")
        
        # Determine rating scale
        scale_max = 10  # Default
        if "1-5" in question.question_text or "5-point" in question.question_text:
            scale_max = 5
        elif "1-7" in question.question_text or "7-point" in question.question_text:
            scale_max = 7
        
        # Generate rating based on enthusiasm
        if enthusiasm == "high":
            # Tend toward higher ratings
            return random.randint(max(1, scale_max - 2), scale_max)
        elif enthusiasm == "low":
            # Tend toward lower ratings
            return random.randint(1, max(1, scale_max // 2))
        else:
            # Moderate ratings
            return random.randint(max(1, scale_max // 3), max(1, scale_max * 2 // 3))
    
    def _build_openai_prompt(self, question: SurveyQuestion, context: ResponseContext, analysis: Dict[str, Any]) -> str:
        """Build prompt for OpenAI."""
        persona = self.current_persona
        
        prompt = f"""
You are {persona.name}, a {persona.age_range[0]}-{persona.age_range[1]} year old person with the following characteristics:
- Education: {persona.education}
- Income: {persona.income_range}
- Location: {persona.location_type}
- Interests: {', '.join(persona.interests)}
- Values: {', '.join(persona.values)}
- Personality: {', '.join(persona.personality_traits)}

Please answer this survey question as this person would, staying consistent with their background and previous responses:

Question: {question.question_text}
"""
        
        if question.options:
            prompt += f"\nOptions: {', '.join(question.options)}"
        
        if context and context.previous_responses:
            prompt += f"\nPrevious responses in this survey: {context.previous_responses[-3:]}"
        
        prompt += "\n\nProvide a natural, authentic response that this person would give:"
        
        return prompt
    
    def _build_local_prompt(self, question: SurveyQuestion, context: ResponseContext, analysis: Dict[str, Any]) -> str:
        """Build prompt for local model."""
        persona = self.current_persona
        
        prompt = f"As a {persona.age_range[0]}-year-old {persona.education} person who values {persona.values[0]}, "
        prompt += f"answer this survey question: {question.question_text}\n"
        
        if question.options:
            prompt += f"Choose from: {', '.join(question.options[:3])}\n"  # Limit options for local model
        
        prompt += "Answer:"
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for OpenAI."""
        return """You are a helpful assistant that generates realistic survey responses. 
        Always stay in character and provide responses that are consistent with the given persona. 
        Keep responses natural and authentic. If given multiple choice options, select one that fits the persona best.
        For text responses, keep them concise but genuine."""
    
    def _select_best_response(self, candidates: List[str], question: SurveyQuestion, context: ResponseContext) -> str:
        """Select the best response from candidates."""
        if not candidates:
            return self._generate_fallback_response(question)
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._score_response(candidate, question, context)
            scored_candidates.append((candidate, score))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]
    
    def _score_response(self, response: str, question: SurveyQuestion, context: ResponseContext) -> float:
        """Score a response for quality."""
        score = 0.0
        
        # Length appropriateness
        if question.question_type in ["text_input", "textarea"]:
            if 10 <= len(response) <= 200:
                score += 0.3
            elif len(response) < 5:
                score -= 0.2
        
        # Option validity for multiple choice
        if question.options and question.question_type in ["single_choice", "multiple_choice"]:
            if any(option in response for option in question.options):
                score += 0.4
            else:
                score -= 0.3
        
        # Consistency with persona
        if self.current_persona:
            persona_keywords = (self.current_persona.interests + 
                              self.current_persona.values + 
                              self.current_persona.personality_traits)
            
            response_lower = response.lower()
            matching_keywords = sum(1 for keyword in persona_keywords if keyword.lower() in response_lower)
            score += matching_keywords * 0.1
        
        # Avoid repetitive responses
        if response in [r["response"] for r in self.response_history[-5:]]:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _post_process_response(self, response: str, question: SurveyQuestion, context: ResponseContext) -> str:
        """Post-process response for quality and consistency."""
        if not response:
            return self._generate_fallback_response(question)
        
        # Clean up response
        response = response.strip()
        
        # Ensure response fits question type
        if question.question_type == "single_choice" and question.options:
            # Make sure only one option is selected
            matching_options = [opt for opt in question.options if opt in response]
            if matching_options:
                response = matching_options[0]
            elif not any(opt in response for opt in question.options):
                response = random.choice(question.options)
        
        # Length limits
        if question.max_length and len(response) > question.max_length:
            response = response[:question.max_length].rsplit(' ', 1)[0] + "..."
        
        if question.min_length and len(response) < question.min_length:
            # Pad with appropriate content
            padding = " I think this covers my main thoughts on the topic."
            response += padding[:question.min_length - len(response)]
        
        return response
    
    def _generate_fallback_response(self, question: SurveyQuestion) -> str:
        """Generate a simple fallback response."""
        if question.options:
            return random.choice(question.options)
        elif question.question_type == "rating_scale":
            return str(random.randint(1, 5))
        else:
            return "No strong opinion"
    
    def _update_response_history(self, question: SurveyQuestion, response: str, context: ResponseContext):
        """Update response history for consistency tracking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question.question_id,
            "question_text": question.question_text,
            "response": response,
            "question_type": question.question_type,
            "category": question.category
        }
        
        self.response_history.append(entry)
        
        # Keep only recent history
        if len(self.response_history) > 100:
            self.response_history = self.response_history[-100:]
        
        # Update consistency tracker for demographic questions
        if question.category in ["demographic", "personal"]:
            self.consistency_tracker[question.question_id] = response
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated responses."""
        if not self.response_history:
            return {"total_responses": 0}
        
        stats = {
            "total_responses": len(self.response_history),
            "current_persona": self.current_persona.name if self.current_persona else None,
            "response_types": {},
            "categories": {},
            "average_response_length": 0
        }
        
        # Calculate statistics
        total_length = 0
        for entry in self.response_history:
            # Response types
            q_type = entry["question_type"]
            stats["response_types"][q_type] = stats["response_types"].get(q_type, 0) + 1
            
            # Categories
            category = entry["category"]
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Length
            total_length += len(entry["response"])
        
        stats["average_response_length"] = total_length / len(self.response_history)
        
        return stats