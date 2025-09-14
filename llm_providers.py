"""
LLM Providers Integration
Support for multiple LLM providers including Ollama, LM Studio, OpenAI, and DeepSeek.
"""

import asyncio
import logging
import json
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import openai
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    ANTHROPIC = "anthropic"

@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    text: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    reasoning_steps: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace('provider', '')
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_chain_of_thought(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with chain-of-thought reasoning."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('openai_api_key')
        self.model = config.get('openai_model', 'gpt-3.5-turbo')
        self.client = None
        
        if self.api_key:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.client is not None
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI."""
        if not self.is_available():
            raise ValueError("OpenAI provider not configured")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                provider="openai",
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_chain_of_thought(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with chain-of-thought reasoning."""
        cot_prompt = f"""Think step by step about this question and provide your reasoning:

{prompt}

Please structure your response as:
1. Analysis: [Your analysis of the question]
2. Reasoning: [Step-by-step reasoning]
3. Answer: [Your final answer]
"""
        
        response = await self.generate_response(cot_prompt, **kwargs)
        
        # Parse reasoning steps
        reasoning_steps = []
        lines = response.text.split('\n')
        current_step = ""
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', 'Analysis:', 'Reasoning:', 'Answer:')):
                if current_step:
                    reasoning_steps.append(current_step.strip())
                current_step = line.strip()
            else:
                current_step += " " + line.strip()
        
        if current_step:
            reasoning_steps.append(current_step.strip())
        
        response.reasoning_steps = reasoning_steps
        return response

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek R1 provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('deepseek_api_key')
        self.model = config.get('deepseek_model', 'deepseek-r1')
        self.base_url = config.get('deepseek_base_url', 'https://api.deepseek.com/v1')
        self.client = None
        
        if self.api_key:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.client is not None
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using DeepSeek."""
        if not self.is_available():
            raise ValueError("DeepSeek provider not configured")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                provider="deepseek",
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def generate_chain_of_thought(self, prompt: str, **kwargs) -> LLMResponse:
        """DeepSeek R1 has built-in reasoning capabilities."""
        reasoning_prompt = f"""<thinking>
Let me think about this survey question step by step:

{prompt}

I need to consider:
1. What type of question this is
2. What the survey is trying to learn
3. How my persona would respond
4. What answer would be most consistent
</thinking>

Please provide a thoughtful response to this survey question, explaining your reasoning."""
        
        response = await self.generate_response(reasoning_prompt, **kwargs)
        
        # DeepSeek R1 includes reasoning in <thinking> tags
        reasoning_steps = []
        if "<thinking>" in response.text and "</thinking>" in response.text:
            thinking_content = response.text.split("<thinking>")[1].split("</thinking>")[0]
            reasoning_steps = [step.strip() for step in thinking_content.split('\n') if step.strip()]
        
        response.reasoning_steps = reasoning_steps
        return response

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
        self.model = config.get('ollama_model', 'llama2')
        self.timeout = config.get('ollama_timeout', 60)
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama."""
        if not self.is_available():
            raise ValueError("Ollama not available at " + self.base_url)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "num_predict": kwargs.get('max_tokens', 500)
            }
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            text=result.get('response', ''),
                            model=self.model,
                            provider="ollama",
                            metadata={"eval_count": result.get('eval_count')}
                        )
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def generate_chain_of_thought(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with chain-of-thought reasoning."""
        cot_prompt = f"""You are answering a survey question. Think through this step by step:

Question: {prompt}

Please think through this systematically:
1. What type of question is this?
2. What information is being sought?
3. How should I respond based on my persona?
4. What would be the most appropriate answer?

Provide your step-by-step reasoning, then give your final answer."""
        
        response = await self.generate_response(cot_prompt, **kwargs)
        
        # Parse reasoning steps from response
        reasoning_steps = []
        lines = response.text.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', 'Step', 'First', 'Second', 'Third')):
                reasoning_steps.append(line.strip())
        
        response.reasoning_steps = reasoning_steps
        return response

class LMStudioProvider(BaseLLMProvider):
    """LM Studio local server provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
        self.model = config.get('lm_studio_model', 'local-model')
        self.timeout = config.get('lm_studio_timeout', 60)
    
    def is_available(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using LM Studio."""
        if not self.is_available():
            raise ValueError("LM Studio not available at " + self.base_url)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 500),
            "temperature": kwargs.get('temperature', 0.7),
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            text=result['choices'][0]['message']['content'],
                            model=self.model,
                            provider="lm_studio",
                            tokens_used=result.get('usage', {}).get('total_tokens')
                        )
                    else:
                        raise Exception(f"LM Studio API error: {response.status}")
        
        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            raise
    
    async def generate_chain_of_thought(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with chain-of-thought reasoning."""
        cot_prompt = f"""I need to answer this survey question thoughtfully. Let me think through this step by step:

{prompt}

My reasoning process:
1. Question Analysis: What is this question really asking?
2. Context Consideration: What context clues can I gather?
3. Persona Alignment: How would my persona respond?
4. Consistency Check: Is this consistent with my previous answers?
5. Final Decision: What is my best answer?

Please provide detailed reasoning for each step, then give the final answer."""
        
        response = await self.generate_response(cot_prompt, **kwargs)
        
        # Parse reasoning steps
        reasoning_steps = []
        lines = response.text.split('\n')
        current_step = ""
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['analysis:', 'consideration:', 'alignment:', 'check:', 'decision:']):
                if current_step:
                    reasoning_steps.append(current_step.strip())
                current_step = line.strip()
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_step:
                    reasoning_steps.append(current_step.strip())
                current_step = line.strip()
            else:
                current_step += " " + line.strip()
        
        if current_step:
            reasoning_steps.append(current_step.strip())
        
        response.reasoning_steps = reasoning_steps
        return response

class LLMManager:
    """Manager for multiple LLM providers with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.primary_provider = config.get('primary_llm_provider', LLMProvider.OPENAI.value)
        self.fallback_providers = config.get('fallback_llm_providers', [LLMProvider.OLLAMA.value])
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured LLM providers."""
        provider_classes = {
            LLMProvider.OPENAI.value: OpenAIProvider,
            LLMProvider.DEEPSEEK.value: DeepSeekProvider,
            LLMProvider.OLLAMA.value: OllamaProvider,
            LLMProvider.LM_STUDIO.value: LMStudioProvider
        }
        
        for provider_name, provider_class in provider_classes.items():
            try:
                provider = provider_class(self.config)
                if provider.is_available():
                    self.providers[provider_name] = provider
                    logger.info(f"✅ {provider_name.upper()} provider initialized")
                else:
                    logger.warning(f"⚠️  {provider_name.upper()} provider not available")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {provider_name} provider: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    async def generate_response(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using specified or primary provider with fallback."""
        providers_to_try = []
        
        if provider and provider in self.providers:
            providers_to_try.append(provider)
        elif self.primary_provider in self.providers:
            providers_to_try.append(self.primary_provider)
        
        # Add fallback providers
        for fallback in self.fallback_providers:
            if fallback in self.providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        # Try remaining providers
        for remaining_provider in self.providers:
            if remaining_provider not in providers_to_try:
                providers_to_try.append(remaining_provider)
        
        last_error = None
        for provider_name in providers_to_try:
            try:
                logger.info(f"🤖 Trying {provider_name.upper()} provider")
                response = await self.providers[provider_name].generate_response(prompt, **kwargs)
                logger.info(f"✅ {provider_name.upper()} response generated successfully")
                return response
            except Exception as e:
                logger.warning(f"⚠️  {provider_name.upper()} failed: {e}")
                last_error = e
                continue
        
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    async def generate_chain_of_thought(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate chain-of-thought response with fallback."""
        providers_to_try = []
        
        if provider and provider in self.providers:
            providers_to_try.append(provider)
        elif self.primary_provider in self.providers:
            providers_to_try.append(self.primary_provider)
        
        # Add fallback providers
        for fallback in self.fallback_providers:
            if fallback in self.providers and fallback not in providers_to_try:
                providers_to_try.append(fallback)
        
        last_error = None
        for provider_name in providers_to_try:
            try:
                logger.info(f"🧠 Generating chain-of-thought with {provider_name.upper()}")
                response = await self.providers[provider_name].generate_chain_of_thought(prompt, **kwargs)
                logger.info(f"✅ Chain-of-thought generated with {provider_name.upper()}")
                return response
            except Exception as e:
                logger.warning(f"⚠️  {provider_name.upper()} chain-of-thought failed: {e}")
                last_error = e
                continue
        
        raise Exception(f"All LLM providers failed for chain-of-thought. Last error: {last_error}")
    
    async def cross_validate_answer(self, prompt: str, num_agents: int = 3) -> Dict[str, Any]:
        """Use multiple agents to cross-validate an answer."""
        responses = []
        providers_used = []
        
        available_providers = list(self.providers.keys())
        
        for i in range(min(num_agents, len(available_providers))):
            provider = available_providers[i % len(available_providers)]
            try:
                response = await self.generate_chain_of_thought(prompt, provider=provider)
                responses.append(response)
                providers_used.append(provider)
            except Exception as e:
                logger.warning(f"Cross-validation failed for {provider}: {e}")
        
        if not responses:
            raise Exception("No providers available for cross-validation")
        
        # Analyze consensus
        answers = [resp.text for resp in responses]
        reasoning_steps = [resp.reasoning_steps or [] for resp in responses]
        
        # Simple consensus detection (could be enhanced with semantic similarity)
        answer_counts = {}
        for answer in answers:
            # Extract final answer (simple heuristic)
            final_answer = answer.split('\n')[-1].strip()
            answer_counts[final_answer] = answer_counts.get(final_answer, 0) + 1
        
        consensus_answer = max(answer_counts.items(), key=lambda x: x[1])
        consensus_strength = consensus_answer[1] / len(responses)
        
        return {
            'consensus_answer': consensus_answer[0],
            'consensus_strength': consensus_strength,
            'all_responses': responses,
            'providers_used': providers_used,
            'answer_distribution': answer_counts,
            'reasoning_steps': reasoning_steps
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider usage and availability."""
        stats = {
            'total_providers': len(self.providers),
            'available_providers': list(self.providers.keys()),
            'primary_provider': self.primary_provider,
            'fallback_providers': self.fallback_providers
        }
        
        # Check current availability
        for provider_name, provider in self.providers.items():
            stats[f'{provider_name}_available'] = provider.is_available()
        
        return stats