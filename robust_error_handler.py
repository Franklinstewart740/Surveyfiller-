"""
Robust Error Handling System
Comprehensive error handling with exponential backoff, response validation, and semantic fallbacks.
"""

import asyncio
import logging
import json
import time
import random
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel, ValidationError
import aiohttp

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of errors that can occur."""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    BROWSER_ERROR = "browser_error"
    CAPTCHA_ERROR = "captcha_error"
    PARSING_ERROR = "parsing_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 2.0
    jitter: bool = True

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on: List[ErrorType] = None

class GeminiResponseSchema(BaseModel):
    """Schema for validating Gemini API responses."""
    text: Optional[str] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    
    def get_response_text(self) -> Optional[str]:
        """Extract response text from various response formats."""
        if self.text:
            return self.text
        
        if self.candidates and len(self.candidates) > 0:
            candidate = self.candidates[0]
            if 'content' in candidate:
                content = candidate['content']
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if parts and len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
                elif isinstance(content, str):
                    return content
        
        return None

class RobustErrorHandler:
    """Comprehensive error handling system with advanced recovery strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_history: List[ErrorContext] = []
        self.fallback_answers: Dict[str, List[str]] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.question_answer_cache: Dict[str, str] = {}
        
        # Load fallback answers
        self._load_fallback_answers()
        
        # Default retry configurations
        self.default_retry_configs = {
            ErrorType.NETWORK_ERROR: RetryConfig(max_retries=5, base_delay=2.0, backoff_factor=2.0),
            ErrorType.API_ERROR: RetryConfig(max_retries=3, base_delay=1.0, backoff_factor=1.5),
            ErrorType.RATE_LIMIT_ERROR: RetryConfig(max_retries=10, base_delay=5.0, backoff_factor=2.0, max_delay=300.0),
            ErrorType.TIMEOUT_ERROR: RetryConfig(max_retries=3, base_delay=3.0, backoff_factor=2.0),
            ErrorType.BROWSER_ERROR: RetryConfig(max_retries=2, base_delay=1.0, backoff_factor=1.5),
            ErrorType.CAPTCHA_ERROR: RetryConfig(max_retries=1, base_delay=10.0),  # Special handling for CAPTCHA
            ErrorType.PARSING_ERROR: RetryConfig(max_retries=2, base_delay=0.5, backoff_factor=1.2),
        }
    
    def _load_fallback_answers(self):
        """Load fallback answers for different question types."""
        self.fallback_answers = {
            'age': ['25-34', '35-44', '18-24', '45-54'],
            'gender': ['Male', 'Female', 'Prefer not to say'],
            'income': ['$50,000-$75,000', '$25,000-$50,000', '$75,000-$100,000'],
            'education': ['Bachelor\'s degree', 'High school', 'Some college'],
            'employment': ['Full-time employed', 'Part-time employed', 'Self-employed'],
            'location': ['Urban', 'Suburban', 'Rural'],
            'frequency': ['Sometimes', 'Often', 'Rarely', 'Never'],
            'satisfaction': ['Satisfied', 'Very satisfied', 'Neutral', 'Somewhat satisfied'],
            'agreement': ['Agree', 'Somewhat agree', 'Neutral', 'Disagree'],
            'rating': ['4', '3', '5', '2'],
            'yes_no': ['Yes', 'No'],
            'brand_preference': ['No preference', 'It depends', 'Other'],
            'shopping_frequency': ['Monthly', 'Weekly', 'A few times a year'],
            'media_consumption': ['Daily', 'Weekly', 'Monthly', 'Rarely'],
            'default': ['Other', 'Not sure', 'Prefer not to answer', 'It depends']
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Classify an error and determine its severity and type."""
        error_message = str(exception)
        error_details = context or {}
        
        # Network-related errors
        if isinstance(exception, (aiohttp.ClientError, ConnectionError, TimeoutError)):
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # API-specific errors
        elif isinstance(exception, openai.APIError):
            if "rate limit" in error_message.lower():
                error_type = ErrorType.RATE_LIMIT_ERROR
                severity = ErrorSeverity.HIGH
            elif "authentication" in error_message.lower():
                error_type = ErrorType.AUTHENTICATION_ERROR
                severity = ErrorSeverity.CRITICAL
            else:
                error_type = ErrorType.API_ERROR
                severity = ErrorSeverity.MEDIUM
        
        # Validation errors
        elif isinstance(exception, (ValidationError, ValueError, KeyError)):
            error_type = ErrorType.VALIDATION_ERROR
            severity = ErrorSeverity.LOW
        
        # Timeout errors
        elif isinstance(exception, asyncio.TimeoutError):
            error_type = ErrorType.TIMEOUT_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Browser-related errors
        elif "playwright" in error_message.lower() or "browser" in error_message.lower():
            error_type = ErrorType.BROWSER_ERROR
            severity = ErrorSeverity.HIGH
        
        # CAPTCHA-related errors
        elif "captcha" in error_message.lower():
            error_type = ErrorType.CAPTCHA_ERROR
            severity = ErrorSeverity.HIGH
        
        # Parsing errors
        elif "parse" in error_message.lower() or "json" in error_message.lower():
            error_type = ErrorType.PARSING_ERROR
            severity = ErrorSeverity.LOW
        
        else:
            error_type = ErrorType.UNKNOWN_ERROR
            severity = ErrorSeverity.MEDIUM
        
        return ErrorContext(
            error_type=error_type,
            severity=severity,
            message=error_message,
            details=error_details,
            timestamp=datetime.now()
        )
    
    async def calculate_backoff_delay(self, error_context: ErrorContext, retry_config: RetryConfig) -> float:
        """Calculate backoff delay with exponential backoff and jitter."""
        base_delay = retry_config.base_delay
        backoff_factor = retry_config.backoff_factor
        retry_count = error_context.retry_count
        
        # Exponential backoff
        delay = base_delay * (backoff_factor ** retry_count)
        
        # Apply maximum delay limit
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return delay
    
    async def should_retry(self, error_context: ErrorContext, retry_config: RetryConfig) -> bool:
        """Determine if an error should be retried."""
        # Check retry count
        if error_context.retry_count >= retry_config.max_retries:
            return False
        
        # Check if error type is retryable
        if retry_config.retry_on and error_context.error_type not in retry_config.retry_on:
            return False
        
        # Special cases
        if error_context.error_type == ErrorType.AUTHENTICATION_ERROR:
            return False  # Don't retry auth errors
        
        if error_context.error_type == ErrorType.CAPTCHA_ERROR:
            return error_context.retry_count == 0  # Only retry CAPTCHA once
        
        return True
    
    async def retry_with_backoff(self, func: Callable, error_context: ErrorContext, 
                               retry_config: RetryConfig = None, *args, **kwargs) -> Any:
        """Retry a function with exponential backoff."""
        if retry_config is None:
            retry_config = self.default_retry_configs.get(
                error_context.error_type, 
                RetryConfig()
            )
        
        while await self.should_retry(error_context, retry_config):
            try:
                # Calculate and apply backoff delay
                delay = await self.calculate_backoff_delay(error_context, retry_config)
                if delay > 0:
                    logger.info(f"Retrying after {delay:.2f}s (attempt {error_context.retry_count + 1}/{retry_config.max_retries})")
                    await asyncio.sleep(delay)
                
                # Attempt the function call
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                logger.info(f"Retry successful after {error_context.retry_count + 1} attempts")
                return result
                
            except Exception as e:
                error_context.retry_count += 1
                error_context.message = str(e)
                error_context.timestamp = datetime.now()
                
                logger.warning(f"Retry {error_context.retry_count} failed: {e}")
                
                if not await self.should_retry(error_context, retry_config):
                    break
        
        # All retries exhausted
        logger.error(f"All retries exhausted for {error_context.error_type.value}")
        raise Exception(f"Max retries exceeded: {error_context.message}")
    
    async def validate_gemini_response(self, response: Any) -> GeminiResponseSchema:
        """Validate Gemini API response against schema."""
        try:
            if hasattr(response, 'text') and response.text:
                return GeminiResponseSchema(text=response.text)
            
            # Handle different response formats
            if hasattr(response, 'candidates'):
                return GeminiResponseSchema(candidates=response.candidates)
            
            # Try to parse as dictionary
            if isinstance(response, dict):
                return GeminiResponseSchema(**response)
            
            # If response has a model_dump method (Pydantic model)
            if hasattr(response, 'model_dump'):
                return GeminiResponseSchema(**response.model_dump())
            
            raise ValidationError("Unable to parse response format")
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            raise ValidationError(f"Invalid response format: {e}")
    
    async def get_semantic_fallback_answer(self, question_text: str, options: List[str] = None) -> str:
        """Get fallback answer using semantic similarity."""
        question_lower = question_text.lower()
        
        # Try to find the best matching fallback category
        best_category = 'default'
        best_score = 0.0
        
        for category, keywords in {
            'age': ['age', 'old', 'born', 'year'],
            'gender': ['gender', 'male', 'female', 'sex'],
            'income': ['income', 'salary', 'earn', 'money', 'household'],
            'education': ['education', 'school', 'degree', 'college', 'university'],
            'employment': ['work', 'job', 'employ', 'career', 'occupation'],
            'location': ['live', 'location', 'area', 'city', 'state', 'country'],
            'frequency': ['often', 'frequency', 'how much', 'usually', 'typically'],
            'satisfaction': ['satisfied', 'satisfaction', 'happy', 'pleased'],
            'agreement': ['agree', 'disagree', 'opinion', 'think', 'believe'],
            'rating': ['rate', 'rating', 'scale', 'score', 'out of'],
            'yes_no': ['yes', 'no', 'do you', 'have you', 'will you'],
            'brand_preference': ['brand', 'prefer', 'favorite', 'like', 'choose'],
            'shopping_frequency': ['shop', 'buy', 'purchase', 'store'],
            'media_consumption': ['watch', 'read', 'listen', 'media', 'tv', 'news']
        }.items():
            
            # Calculate keyword overlap score
            question_words = set(question_lower.split())
            keyword_matches = sum(1 for keyword in keywords if keyword in question_lower)
            score = keyword_matches / len(keywords)
            
            if score > best_score:
                best_score = score
                best_category = category
        
        # Get fallback answers for the category
        fallback_options = self.fallback_answers.get(best_category, self.fallback_answers['default'])
        
        # If specific options are provided, try to match them
        if options:
            # Simple matching - look for options that contain fallback keywords
            for option in options:
                option_lower = option.lower()
                for fallback in fallback_options:
                    if fallback.lower() in option_lower or option_lower in fallback.lower():
                        return option
            
            # If no match found, return a middle option or random option
            if len(options) >= 3:
                return options[len(options) // 2]  # Middle option
            else:
                return random.choice(options)
        
        # Return random fallback answer
        return random.choice(fallback_options)
    
    async def score_survey_complexity(self, questions: List[str], layout_info: Dict[str, Any] = None) -> Dict[str, float]:
        """Score survey complexity to decide whether to skip."""
        scores = {
            'length_score': 0.0,
            'text_complexity_score': 0.0,
            'layout_clutter_score': 0.0,
            'input_variety_score': 0.0,
            'overall_complexity': 0.0
        }
        
        # Length score (0-1, higher = more complex)
        question_count = len(questions)
        scores['length_score'] = min(question_count / 100, 1.0)  # Normalize to 100 questions
        
        # Text complexity score
        if questions:
            total_words = sum(len(q.split()) for q in questions)
            avg_words_per_question = total_words / len(questions)
            scores['text_complexity_score'] = min(avg_words_per_question / 30, 1.0)  # Normalize to 30 words
        
        # Layout clutter score (if layout info provided)
        if layout_info:
            element_count = layout_info.get('element_count', 0)
            nested_depth = layout_info.get('nested_depth', 0)
            
            scores['layout_clutter_score'] = min((element_count / 100) + (nested_depth / 10), 1.0)
        
        # Input variety score
        input_types = set()
        for question in questions:
            if any(word in question.lower() for word in ['text', 'write', 'describe']):
                input_types.add('text')
            elif any(word in question.lower() for word in ['select', 'choose', 'pick']):
                input_types.add('select')
            elif any(word in question.lower() for word in ['rate', 'scale']):
                input_types.add('rating')
            elif any(word in question.lower() for word in ['yes', 'no']):
                input_types.add('boolean')
        
        scores['input_variety_score'] = len(input_types) / 4  # 4 main input types
        
        # Calculate overall complexity
        weights = {
            'length_score': 0.3,
            'text_complexity_score': 0.25,
            'layout_clutter_score': 0.25,
            'input_variety_score': 0.2
        }
        
        scores['overall_complexity'] = sum(
            scores[key] * weight for key, weight in weights.items()
        )
        
        return scores
    
    async def should_skip_survey(self, complexity_scores: Dict[str, float], 
                               error_history: List[ErrorContext] = None) -> Tuple[bool, str]:
        """Determine if survey should be skipped based on complexity and error history."""
        overall_complexity = complexity_scores['overall_complexity']
        
        # Skip if overall complexity is too high
        if overall_complexity > 0.8:
            return True, f"Survey complexity too high: {overall_complexity:.2f}"
        
        # Check error history for this session
        if error_history:
            recent_errors = [e for e in error_history if (datetime.now() - e.timestamp).seconds < 3600]  # Last hour
            
            # Skip if too many recent errors
            if len(recent_errors) > 5:
                return True, f"Too many recent errors: {len(recent_errors)}"
            
            # Skip if critical errors occurred
            critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            if critical_errors:
                return True, f"Critical errors detected: {len(critical_errors)}"
        
        return False, "Survey complexity acceptable"
    
    async def log_error_with_context(self, error_context: ErrorContext, additional_context: Dict[str, Any] = None):
        """Log error with comprehensive context information."""
        self.error_history.append(error_context)
        
        # Keep error history manageable (last 1000 errors)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        log_data = {
            'timestamp': error_context.timestamp.isoformat(),
            'error_type': error_context.error_type.value,
            'severity': error_context.severity.value,
            'message': error_context.message,
            'retry_count': error_context.retry_count,
            'details': error_context.details
        }
        
        if additional_context:
            log_data['additional_context'] = additional_context
        
        # Log with appropriate level based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {json.dumps(log_data, indent=2)}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {json.dumps(log_data, indent=2)}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"Low severity error: {json.dumps(log_data, indent=2)}")
    
    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        if not recent_errors:
            return {'total_errors': 0, 'error_rate': 0.0}
        
        # Count by type
        error_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_type = error.error_type.value
            severity = error.severity.value
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / time_window_hours,
            'error_types': error_counts,
            'severity_distribution': severity_counts,
            'most_common_error': max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }
    
    async def handle_gemini_api_call(self, api_call_func: Callable, *args, **kwargs) -> Any:
        """Handle Gemini API calls with comprehensive error handling."""
        error_context = None
        
        try:
            # Make the API call
            response = await api_call_func(*args, **kwargs) if asyncio.iscoroutinefunction(api_call_func) else api_call_func(*args, **kwargs)
            
            # Validate response
            validated_response = await self.validate_gemini_response(response)
            
            # Check if response text is available
            response_text = validated_response.get_response_text()
            if not response_text:
                raise ValueError("Response text is undefined or empty")
            
            return validated_response
            
        except Exception as e:
            error_context = self.classify_error(e, {'function': api_call_func.__name__, 'args': str(args)[:200]})
            await self.log_error_with_context(error_context)
            
            # Try to retry with backoff
            try:
                return await self.retry_with_backoff(api_call_func, error_context, None, *args, **kwargs)
            except Exception as retry_error:
                logger.error(f"All retries failed for Gemini API call: {retry_error}")
                
                # Return None to trigger fallback mechanisms
                return None