"""
Multi-Agent Collaboration System
Cooperative agents for cross-validation, specialization, and improved survey completion.
"""

import asyncio
import logging
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Different agent specialization roles."""
    SPEED_RUNNER = "speed_runner"
    REALIST = "realist"
    CONTRARIAN = "contrarian"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"

class AgentPersonality(Enum):
    """Agent personality traits affecting decision making."""
    CAUTIOUS = "cautious"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"

@dataclass
class AgentDecision:
    """Represents a decision made by an agent."""
    agent_id: str
    question_id: str
    selected_answer: str
    confidence: float
    reasoning: str
    reasoning_steps: List[str]
    decision_time: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ConsensusResult:
    """Result of multi-agent consensus building."""
    final_answer: str
    consensus_strength: float
    participating_agents: List[str]
    individual_decisions: List[AgentDecision]
    reasoning_synthesis: str
    confidence_score: float
    dissenting_opinions: List[str]
    decision_method: str

@dataclass
class AgentProfile:
    """Profile defining an agent's characteristics."""
    agent_id: str
    role: AgentRole
    personality: AgentPersonality
    specializations: List[str]
    experience_level: float
    success_rate: float
    response_time_preference: str  # "fast", "medium", "thorough"
    risk_tolerance: float
    consistency_weight: float
    innovation_tendency: float

class BaseAgent(ABC):
    """Base class for all survey agents."""
    
    def __init__(self, profile: AgentProfile, config: Dict[str, Any]):
        self.profile = profile
        self.config = config
        self.decision_history: List[AgentDecision] = []
        self.performance_metrics = {
            'decisions_made': 0,
            'avg_confidence': 0.0,
            'consistency_score': 0.0,
            'response_time': 0.0
        }
    
    @abstractmethod
    async def make_decision(self, question: str, options: List[str], context: Dict[str, Any]) -> AgentDecision:
        """Make a decision for a survey question."""
        pass
    
    @abstractmethod
    async def validate_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate another agent's decision."""
        pass
    
    def update_performance_metrics(self, decision: AgentDecision):
        """Update agent's performance metrics."""
        self.performance_metrics['decisions_made'] += 1
        
        # Update average confidence
        total_decisions = self.performance_metrics['decisions_made']
        current_avg = self.performance_metrics['avg_confidence']
        self.performance_metrics['avg_confidence'] = (
            (current_avg * (total_decisions - 1) + decision.confidence) / total_decisions
        )
        
        # Update response time
        current_time_avg = self.performance_metrics['response_time']
        self.performance_metrics['response_time'] = (
            (current_time_avg * (total_decisions - 1) + decision.decision_time) / total_decisions
        )

class SpeedRunnerAgent(BaseAgent):
    """Agent optimized for fast survey completion."""
    
    async def make_decision(self, question: str, options: List[str], context: Dict[str, Any]) -> AgentDecision:
        """Make quick decisions with simple heuristics."""
        start_time = asyncio.get_event_loop().time()
        
        # Simple decision logic for speed
        selected_answer = self._quick_select(question, options)
        confidence = 0.7  # Moderate confidence for speed
        reasoning = f"Quick selection based on pattern matching: {selected_answer}"
        reasoning_steps = [
            "1. Analyzed question type",
            "2. Applied speed heuristics",
            "3. Selected most likely answer"
        ]
        
        decision_time = asyncio.get_event_loop().time() - start_time
        
        decision = AgentDecision(
            agent_id=self.profile.agent_id,
            question_id=context.get('question_id', 'unknown'),
            selected_answer=selected_answer,
            confidence=confidence,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            decision_time=decision_time,
            metadata={'strategy': 'speed_heuristics'},
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        self.update_performance_metrics(decision)
        
        return decision
    
    def _quick_select(self, question: str, options: List[str]) -> str:
        """Quick selection using simple heuristics."""
        question_lower = question.lower()
        
        # Demographic questions - use middle options
        if any(word in question_lower for word in ['age', 'income', 'education']):
            return options[len(options) // 2] if options else ""
        
        # Yes/No questions - slight bias toward positive
        if len(options) == 2 and any(opt.lower() in ['yes', 'no'] for opt in options):
            return 'Yes' if 'Yes' in options else options[0]
        
        # Rating scales - avoid extremes
        if any(word in question_lower for word in ['rate', 'scale', 'satisfaction']):
            if len(options) >= 5:
                return options[len(options) // 2 + random.choice([-1, 0, 1])]
        
        # Default: select middle option or random
        return options[len(options) // 2] if options else ""
    
    async def validate_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Quick validation focused on obvious errors."""
        # Check for consistency with previous decisions
        if self.decision_history:
            recent_decisions = self.decision_history[-5:]  # Last 5 decisions
            
            # Simple contradiction check
            for prev_decision in recent_decisions:
                if self._check_contradiction(decision, prev_decision):
                    return False, "Contradicts recent decision"
        
        return True, "Quick validation passed"
    
    def _check_contradiction(self, decision1: AgentDecision, decision2: AgentDecision) -> bool:
        """Check for obvious contradictions between decisions."""
        # Simple contradiction detection
        if "never" in decision1.selected_answer.lower() and "always" in decision2.selected_answer.lower():
            return True
        if "love" in decision1.selected_answer.lower() and "hate" in decision2.selected_answer.lower():
            return True
        return False

class RealistAgent(BaseAgent):
    """Agent focused on realistic, human-like responses."""
    
    async def make_decision(self, question: str, options: List[str], context: Dict[str, Any]) -> AgentDecision:
        """Make realistic decisions based on human behavior patterns."""
        start_time = asyncio.get_event_loop().time()
        
        # Analyze question for realistic response
        selected_answer = await self._realistic_select(question, options, context)
        confidence = self._calculate_realistic_confidence(question, selected_answer, options)
        reasoning = f"Selected '{selected_answer}' based on realistic human behavior patterns"
        reasoning_steps = [
            "1. Analyzed question context and type",
            "2. Considered typical human response patterns",
            "3. Applied demographic consistency",
            "4. Selected most realistic answer"
        ]
        
        decision_time = asyncio.get_event_loop().time() - start_time
        
        decision = AgentDecision(
            agent_id=self.profile.agent_id,
            question_id=context.get('question_id', 'unknown'),
            selected_answer=selected_answer,
            confidence=confidence,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            decision_time=decision_time,
            metadata={'strategy': 'realistic_behavior'},
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        self.update_performance_metrics(decision)
        
        return decision
    
    async def _realistic_select(self, question: str, options: List[str], context: Dict[str, Any]) -> str:
        """Select answer based on realistic human behavior."""
        question_lower = question.lower()
        
        # Income questions - people often underreport
        if 'income' in question_lower or 'salary' in question_lower:
            # Bias toward lower-middle options
            if len(options) >= 4:
                return options[len(options) // 3]
        
        # Age questions - realistic distribution
        if 'age' in question_lower:
            # Most survey takers are 25-45
            age_options = ['25-34', '35-44', '18-24', '45-54']
            for option in options:
                if any(age_range in option for age_range in age_options):
                    return option
        
        # Shopping frequency - realistic patterns
        if any(word in question_lower for word in ['shop', 'buy', 'purchase']):
            realistic_freq = ['weekly', 'monthly', 'few times a month']
            for option in options:
                if any(freq in option.lower() for freq in realistic_freq):
                    return option
        
        # Brand loyalty - most people are somewhat loyal
        if 'brand' in question_lower or 'loyal' in question_lower:
            moderate_options = [opt for opt in options if any(word in opt.lower() for word in ['sometimes', 'somewhat', 'occasionally'])]
            if moderate_options:
                return random.choice(moderate_options)
        
        # Default: avoid extremes, prefer middle-ground answers
        if len(options) >= 5:
            # Remove extreme options
            middle_options = options[1:-1]
            return random.choice(middle_options) if middle_options else options[len(options) // 2]
        elif len(options) >= 3:
            return options[1]  # Second option
        
        return random.choice(options) if options else ""
    
    def _calculate_realistic_confidence(self, question: str, answer: str, options: List[str]) -> float:
        """Calculate confidence based on realism of the answer."""
        base_confidence = 0.8
        
        # Higher confidence for demographic questions
        if any(word in question.lower() for word in ['age', 'gender', 'location']):
            base_confidence += 0.1
        
        # Lower confidence for opinion questions
        if any(word in question.lower() for word in ['opinion', 'think', 'believe']):
            base_confidence -= 0.1
        
        # Adjust based on answer position (middle answers more confident)
        if answer in options:
            position = options.index(answer)
            middle_position = len(options) // 2
            distance_from_middle = abs(position - middle_position)
            confidence_adjustment = -0.05 * distance_from_middle
            base_confidence += confidence_adjustment
        
        return max(0.3, min(1.0, base_confidence))
    
    async def validate_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate decision for realism."""
        # Check if answer seems realistic
        question = context.get('question', '')
        answer = decision.selected_answer
        
        # Flag unrealistic combinations
        if 'income' in question.lower() and any(word in answer.lower() for word in ['million', 'billionaire']):
            return False, "Unrealistic income level"
        
        if 'age' in question.lower() and any(word in answer.lower() for word in ['under 10', 'over 100']):
            return False, "Unrealistic age range"
        
        # Check consistency with persona
        if hasattr(self, 'persona_data'):
            if not self._check_persona_consistency(decision, self.persona_data):
                return False, "Inconsistent with established persona"
        
        return True, "Realistic decision"
    
    def _check_persona_consistency(self, decision: AgentDecision, persona_data: Dict[str, Any]) -> bool:
        """Check if decision is consistent with persona."""
        # Simple consistency checks
        answer_lower = decision.selected_answer.lower()
        
        # Income consistency
        if 'income_bracket' in persona_data:
            expected_income = persona_data['income_bracket'].lower()
            if 'income' in decision.question_id.lower():
                return expected_income in answer_lower or answer_lower in expected_income
        
        return True  # Default to consistent

class ContrarianAgent(BaseAgent):
    """Agent that provides contrarian viewpoints for better coverage."""
    
    async def make_decision(self, question: str, options: List[str], context: Dict[str, Any]) -> AgentDecision:
        """Make contrarian decisions to explore different answer spaces."""
        start_time = asyncio.get_event_loop().time()
        
        # Get contrarian answer
        selected_answer = self._contrarian_select(question, options, context)
        confidence = 0.6  # Lower confidence for contrarian views
        reasoning = f"Contrarian perspective: {selected_answer} to provide alternative viewpoint"
        reasoning_steps = [
            "1. Identified conventional answer",
            "2. Explored alternative perspectives",
            "3. Selected contrarian but plausible option",
            "4. Validated for reasonableness"
        ]
        
        decision_time = asyncio.get_event_loop().time() - start_time
        
        decision = AgentDecision(
            agent_id=self.profile.agent_id,
            question_id=context.get('question_id', 'unknown'),
            selected_answer=selected_answer,
            confidence=confidence,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            decision_time=decision_time,
            metadata={'strategy': 'contrarian_perspective'},
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        self.update_performance_metrics(decision)
        
        return decision
    
    def _contrarian_select(self, question: str, options: List[str], context: Dict[str, Any]) -> str:
        """Select contrarian but reasonable answers."""
        question_lower = question.lower()
        
        # For satisfaction questions, be more critical
        if any(word in question_lower for word in ['satisfied', 'satisfaction', 'happy']):
            critical_options = [opt for opt in options if any(word in opt.lower() for word in ['dissatisfied', 'poor', 'bad', 'unsatisfied'])]
            if critical_options:
                return random.choice(critical_options)
        
        # For frequency questions, choose less common frequencies
        if any(word in question_lower for word in ['often', 'frequency', 'how much']):
            uncommon_options = [opt for opt in options if any(word in opt.lower() for word in ['rarely', 'never', 'seldom'])]
            if uncommon_options:
                return random.choice(uncommon_options)
        
        # For brand questions, show less loyalty
        if 'brand' in question_lower or 'loyal' in question_lower:
            disloyal_options = [opt for opt in options if any(word in opt.lower() for word in ['switch', 'try different', 'not loyal'])]
            if disloyal_options:
                return random.choice(disloyal_options)
        
        # For agreement questions, be more disagreeable
        if any(word in question_lower for word in ['agree', 'opinion', 'believe']):
            disagreement_options = [opt for opt in options if any(word in opt.lower() for word in ['disagree', 'oppose', 'against'])]
            if disagreement_options:
                return random.choice(disagreement_options)
        
        # Default: choose less popular options (first or last)
        if len(options) >= 3:
            return random.choice([options[0], options[-1]])
        
        return random.choice(options) if options else ""
    
    async def validate_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that contrarian decision is still reasonable."""
        # Check if the contrarian view is too extreme
        answer_lower = decision.selected_answer.lower()
        
        # Flag completely unreasonable answers
        unreasonable_patterns = ['impossible', 'never existed', 'completely false']
        if any(pattern in answer_lower for pattern in unreasonable_patterns):
            return False, "Contrarian view too extreme"
        
        return True, "Reasonable contrarian perspective"

class ValidatorAgent(BaseAgent):
    """Agent specialized in validating and cross-checking decisions."""
    
    async def make_decision(self, question: str, options: List[str], context: Dict[str, Any]) -> AgentDecision:
        """Make decisions focused on validation and consistency."""
        start_time = asyncio.get_event_loop().time()
        
        # Analyze for consistency and validation
        selected_answer = await self._validated_select(question, options, context)
        confidence = self._calculate_validation_confidence(question, selected_answer, context)
        reasoning = f"Validated selection: {selected_answer} after consistency checks"
        reasoning_steps = [
            "1. Analyzed question for validation requirements",
            "2. Checked consistency with previous answers",
            "3. Validated against logical constraints",
            "4. Selected most consistent option"
        ]
        
        decision_time = asyncio.get_event_loop().time() - start_time
        
        decision = AgentDecision(
            agent_id=self.profile.agent_id,
            question_id=context.get('question_id', 'unknown'),
            selected_answer=selected_answer,
            confidence=confidence,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            decision_time=decision_time,
            metadata={'strategy': 'validation_focused'},
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        self.update_performance_metrics(decision)
        
        return decision
    
    async def _validated_select(self, question: str, options: List[str], context: Dict[str, Any]) -> str:
        """Select answer with maximum validation."""
        # Get previous decisions for consistency checking
        previous_decisions = context.get('previous_decisions', [])
        
        # Score each option for consistency
        option_scores = {}
        
        for option in options:
            score = 0.5  # Base score
            
            # Check consistency with previous decisions
            for prev_decision in previous_decisions:
                if self._check_consistency(question, option, prev_decision):
                    score += 0.2
                else:
                    score -= 0.3
            
            # Check logical consistency
            if self._check_logical_consistency(question, option):
                score += 0.1
            
            option_scores[option] = score
        
        # Select option with highest score
        if option_scores:
            best_option = max(option_scores.items(), key=lambda x: x[1])
            return best_option[0]
        
        return options[0] if options else ""
    
    def _check_consistency(self, question: str, answer: str, previous_decision: AgentDecision) -> bool:
        """Check consistency with previous decision."""
        # Simple consistency rules
        question_lower = question.lower()
        answer_lower = answer.lower()
        prev_answer_lower = previous_decision.selected_answer.lower()
        
        # Income consistency
        if 'income' in question_lower and 'income' in previous_decision.question_id.lower():
            return answer_lower == prev_answer_lower
        
        # Age consistency
        if 'age' in question_lower and 'age' in previous_decision.question_id.lower():
            return answer_lower == prev_answer_lower
        
        # Behavioral consistency
        if 'never' in prev_answer_lower and 'always' in answer_lower:
            return False
        
        return True  # Default to consistent
    
    def _check_logical_consistency(self, question: str, answer: str) -> bool:
        """Check logical consistency of answer."""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Age logic
        if 'age' in question_lower:
            if 'under 18' in answer_lower and any(word in question_lower for word in ['job', 'work', 'career']):
                return False  # Unlikely to have career under 18
        
        # Income logic
        if 'income' in question_lower:
            if 'student' in answer_lower and 'high income' in answer_lower:
                return False  # Students typically don't have high income
        
        return True  # Default to logically consistent
    
    def _calculate_validation_confidence(self, question: str, answer: str, context: Dict[str, Any]) -> float:
        """Calculate confidence based on validation strength."""
        base_confidence = 0.7
        
        # Higher confidence if answer passed all consistency checks
        previous_decisions = context.get('previous_decisions', [])
        consistency_score = 0
        
        for prev_decision in previous_decisions:
            if self._check_consistency(question, answer, prev_decision):
                consistency_score += 1
        
        if previous_decisions:
            consistency_ratio = consistency_score / len(previous_decisions)
            base_confidence += consistency_ratio * 0.2
        
        return min(1.0, base_confidence)
    
    async def validate_decision(self, decision: AgentDecision, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Comprehensive validation of decision."""
        validation_issues = []
        
        # Check consistency
        previous_decisions = context.get('previous_decisions', [])
        for prev_decision in previous_decisions:
            if not self._check_consistency(decision.question_id, decision.selected_answer, prev_decision):
                validation_issues.append(f"Inconsistent with previous decision: {prev_decision.question_id}")
        
        # Check logical consistency
        if not self._check_logical_consistency(decision.question_id, decision.selected_answer):
            validation_issues.append("Logically inconsistent answer")
        
        # Check confidence threshold
        if decision.confidence < 0.3:
            validation_issues.append("Confidence too low")
        
        if validation_issues:
            return False, "; ".join(validation_issues)
        
        return True, "All validation checks passed"

class MultiAgentSystem:
    """Coordinates multiple agents for collaborative survey completion."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.decision_history: List[ConsensusResult] = []
        self.performance_tracker = {
            'total_decisions': 0,
            'consensus_rate': 0.0,
            'avg_confidence': 0.0,
            'agent_performance': {}
        }
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize the multi-agent system."""
        agent_configs = [
            {
                'role': AgentRole.SPEED_RUNNER,
                'personality': AgentPersonality.AGGRESSIVE,
                'specializations': ['quick_decisions', 'pattern_matching'],
                'response_time_preference': 'fast'
            },
            {
                'role': AgentRole.REALIST,
                'personality': AgentPersonality.BALANCED,
                'specializations': ['human_behavior', 'demographic_consistency'],
                'response_time_preference': 'medium'
            },
            {
                'role': AgentRole.CONTRARIAN,
                'personality': AgentPersonality.ANALYTICAL,
                'specializations': ['alternative_perspectives', 'edge_cases'],
                'response_time_preference': 'medium'
            },
            {
                'role': AgentRole.VALIDATOR,
                'personality': AgentPersonality.CAUTIOUS,
                'specializations': ['consistency_checking', 'validation'],
                'response_time_preference': 'thorough'
            }
        ]
        
        for i, agent_config in enumerate(agent_configs):
            profile = AgentProfile(
                agent_id=f"agent_{agent_config['role'].value}_{i}",
                role=agent_config['role'],
                personality=agent_config['personality'],
                specializations=agent_config['specializations'],
                experience_level=0.8,
                success_rate=0.85,
                response_time_preference=agent_config['response_time_preference'],
                risk_tolerance=0.5,
                consistency_weight=0.7,
                innovation_tendency=0.3
            )
            
            # Create appropriate agent instance
            if agent_config['role'] == AgentRole.SPEED_RUNNER:
                agent = SpeedRunnerAgent(profile, self.config)
            elif agent_config['role'] == AgentRole.REALIST:
                agent = RealistAgent(profile, self.config)
            elif agent_config['role'] == AgentRole.CONTRARIAN:
                agent = ContrarianAgent(profile, self.config)
            elif agent_config['role'] == AgentRole.VALIDATOR:
                agent = ValidatorAgent(profile, self.config)
            else:
                continue
            
            self.agents[profile.agent_id] = agent
            self.performance_tracker['agent_performance'][profile.agent_id] = {
                'decisions_made': 0,
                'avg_confidence': 0.0,
                'validation_success_rate': 0.0
            }
        
        logger.info(f"Initialized multi-agent system with {len(self.agents)} agents")
    
    async def collaborative_decision(self, question: str, options: List[str], 
                                   context: Dict[str, Any] = None) -> ConsensusResult:
        """Make collaborative decision using multiple agents."""
        if context is None:
            context = {}
        
        context['previous_decisions'] = self.decision_history
        
        # Get decisions from all agents
        agent_decisions = []
        
        for agent_id, agent in self.agents.items():
            try:
                decision = await agent.make_decision(question, options, context)
                agent_decisions.append(decision)
                logger.debug(f"Agent {agent_id} decided: {decision.selected_answer}")
            except Exception as e:
                logger.error(f"Agent {agent_id} failed to make decision: {e}")
        
        if not agent_decisions:
            raise Exception("No agents were able to make decisions")
        
        # Build consensus
        consensus = await self._build_consensus(agent_decisions, question, options, context)
        
        # Update performance tracking
        self._update_performance_tracking(consensus)
        
        # Store decision
        self.decision_history.append(consensus)
        
        logger.info(f"Collaborative decision: {consensus.final_answer} (confidence: {consensus.confidence_score:.2f})")
        
        return consensus
    
    async def _build_consensus(self, decisions: List[AgentDecision], question: str, 
                             options: List[str], context: Dict[str, Any]) -> ConsensusResult:
        """Build consensus from multiple agent decisions."""
        # Count votes for each answer
        answer_votes = Counter(decision.selected_answer for decision in decisions)
        
        # Calculate weighted votes (by confidence)
        weighted_votes = {}
        total_weight = 0
        
        for decision in decisions:
            answer = decision.selected_answer
            weight = decision.confidence
            weighted_votes[answer] = weighted_votes.get(answer, 0) + weight
            total_weight += weight
        
        # Determine consensus method
        if len(answer_votes) == 1:
            # Unanimous decision
            final_answer = list(answer_votes.keys())[0]
            consensus_strength = 1.0
            decision_method = "unanimous"
        elif answer_votes.most_common(1)[0][1] >= len(decisions) * 0.6:
            # Strong majority (60%+)
            final_answer = answer_votes.most_common(1)[0][0]
            consensus_strength = answer_votes.most_common(1)[0][1] / len(decisions)
            decision_method = "majority"
        else:
            # Use weighted voting
            final_answer = max(weighted_votes.items(), key=lambda x: x[1])[0]
            consensus_strength = weighted_votes[final_answer] / total_weight
            decision_method = "weighted"
        
        # Calculate overall confidence
        relevant_decisions = [d for d in decisions if d.selected_answer == final_answer]
        confidence_score = sum(d.confidence for d in relevant_decisions) / len(relevant_decisions)
        
        # Synthesize reasoning
        reasoning_synthesis = self._synthesize_reasoning(decisions, final_answer)
        
        # Identify dissenting opinions
        dissenting_opinions = []
        for answer, count in answer_votes.items():
            if answer != final_answer:
                dissenting_opinions.append(f"{answer} ({count} votes)")
        
        return ConsensusResult(
            final_answer=final_answer,
            consensus_strength=consensus_strength,
            participating_agents=[d.agent_id for d in decisions],
            individual_decisions=decisions,
            reasoning_synthesis=reasoning_synthesis,
            confidence_score=confidence_score,
            dissenting_opinions=dissenting_opinions,
            decision_method=decision_method
        )
    
    def _synthesize_reasoning(self, decisions: List[AgentDecision], final_answer: str) -> str:
        """Synthesize reasoning from multiple agents."""
        relevant_decisions = [d for d in decisions if d.selected_answer == final_answer]
        
        if not relevant_decisions:
            return "No supporting reasoning available"
        
        # Combine reasoning from agents that chose the final answer
        reasoning_points = []
        
        for decision in relevant_decisions:
            agent_role = decision.agent_id.split('_')[1]  # Extract role from agent_id
            reasoning_points.append(f"{agent_role.title()}: {decision.reasoning}")
        
        synthesis = "Multi-agent consensus reasoning:\n" + "\n".join(reasoning_points)
        
        # Add dissenting views if any
        dissenting_decisions = [d for d in decisions if d.selected_answer != final_answer]
        if dissenting_decisions:
            synthesis += "\n\nDissenting views considered:\n"
            for decision in dissenting_decisions:
                agent_role = decision.agent_id.split('_')[1]
                synthesis += f"{agent_role.title()}: {decision.reasoning}\n"
        
        return synthesis
    
    def _update_performance_tracking(self, consensus: ConsensusResult):
        """Update performance tracking metrics."""
        self.performance_tracker['total_decisions'] += 1
        
        # Update consensus rate
        total_decisions = self.performance_tracker['total_decisions']
        current_consensus_rate = self.performance_tracker['consensus_rate']
        
        # Consider strong consensus (>0.8) as successful consensus
        consensus_success = 1 if consensus.consensus_strength > 0.8 else 0
        self.performance_tracker['consensus_rate'] = (
            (current_consensus_rate * (total_decisions - 1) + consensus_success) / total_decisions
        )
        
        # Update average confidence
        current_avg_confidence = self.performance_tracker['avg_confidence']
        self.performance_tracker['avg_confidence'] = (
            (current_avg_confidence * (total_decisions - 1) + consensus.confidence_score) / total_decisions
        )
        
        # Update individual agent performance
        for decision in consensus.individual_decisions:
            agent_id = decision.agent_id
            if agent_id in self.performance_tracker['agent_performance']:
                agent_perf = self.performance_tracker['agent_performance'][agent_id]
                agent_perf['decisions_made'] += 1
                
                # Update agent's average confidence
                decisions_made = agent_perf['decisions_made']
                current_avg = agent_perf['avg_confidence']
                agent_perf['avg_confidence'] = (
                    (current_avg * (decisions_made - 1) + decision.confidence) / decisions_made
                )
    
    async def cross_validate_answer(self, question: str, proposed_answer: str, 
                                  options: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Cross-validate a proposed answer using multiple agents."""
        if context is None:
            context = {}
        
        validation_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                # Create a mock decision for validation
                mock_decision = AgentDecision(
                    agent_id="proposed",
                    question_id=context.get('question_id', 'validation'),
                    selected_answer=proposed_answer,
                    confidence=0.8,
                    reasoning="Proposed answer for validation",
                    reasoning_steps=[],
                    decision_time=0.0,
                    metadata={},
                    timestamp=datetime.now()
                )
                
                is_valid, reason = await agent.validate_decision(mock_decision, context)
                validation_results[agent_id] = {
                    'valid': is_valid,
                    'reason': reason,
                    'agent_role': agent.profile.role.value
                }
                
            except Exception as e:
                logger.error(f"Validation failed for agent {agent_id}: {e}")
                validation_results[agent_id] = {
                    'valid': False,
                    'reason': f"Validation error: {e}",
                    'agent_role': agent.profile.role.value
                }
        
        # Calculate overall validation score
        valid_count = sum(1 for result in validation_results.values() if result['valid'])
        validation_score = valid_count / len(validation_results) if validation_results else 0
        
        return {
            'validation_score': validation_score,
            'individual_validations': validation_results,
            'overall_valid': validation_score >= 0.6,  # 60% threshold
            'consensus_strength': validation_score
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about agent performance."""
        stats = {
            'system_performance': self.performance_tracker.copy(),
            'agent_details': {}
        }
        
        for agent_id, agent in self.agents.items():
            agent_stats = {
                'profile': asdict(agent.profile),
                'performance_metrics': agent.performance_metrics.copy(),
                'decision_count': len(agent.decision_history),
                'recent_decisions': [
                    {
                        'question_id': d.question_id,
                        'answer': d.selected_answer,
                        'confidence': d.confidence,
                        'timestamp': d.timestamp.isoformat()
                    }
                    for d in agent.decision_history[-5:]  # Last 5 decisions
                ]
            }
            stats['agent_details'][agent_id] = agent_stats
        
        return stats
    
    async def adapt_agent_weights(self, performance_feedback: Dict[str, float]):
        """Adapt agent weights based on performance feedback."""
        for agent_id, performance_score in performance_feedback.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Adjust agent's confidence based on performance
                if performance_score > 0.8:
                    agent.profile.experience_level = min(1.0, agent.profile.experience_level + 0.05)
                elif performance_score < 0.4:
                    agent.profile.experience_level = max(0.1, agent.profile.experience_level - 0.05)
                
                logger.info(f"Adapted agent {agent_id} experience level to {agent.profile.experience_level}")
    
    def get_consensus_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent consensus decision history."""
        recent_decisions = self.decision_history[-limit:]
        
        return [
            {
                'final_answer': decision.final_answer,
                'consensus_strength': decision.consensus_strength,
                'confidence_score': decision.confidence_score,
                'decision_method': decision.decision_method,
                'participating_agents': decision.participating_agents,
                'dissenting_opinions': decision.dissenting_opinions
            }
            for decision in recent_decisions
        ]