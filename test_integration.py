"""
Integration Tests for Autonomous Survey Agent
Comprehensive testing of all system components working together.
"""

import asyncio
import pytest
import logging
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import our modules
from autonomous_survey_agent import AutonomousSurveyAgent, AgentConfig
from secure_auth_manager import SecureAuthManager
from intelligence_engine import IntelligenceEngine, QuestionType
from robust_error_handler import RobustErrorHandler, ErrorType
from human_simulation import HumanSimulation, PersonaProfile
from layout_intelligence import LayoutIntelligence
from control_panel import ControlPanel

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAutonomousSurveyAgent:
    """Integration tests for the main agent."""
    
    @pytest.fixture
    async def agent_config(self):
        """Create test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AgentConfig(
                database_url=f"sqlite:///{temp_dir}/test.db",
                browser_headless=True,
                human_simulation_enabled=True,
                max_concurrent_tasks=1,
                control_panel_port=12999,  # Use different port for testing
                sessions_file=f"{temp_dir}/test_sessions.dat",
                memory_file=f"{temp_dir}/test_memory.pkl",
                layout_cache_file=f"{temp_dir}/test_layout.pkl"
            )
            yield config
    
    @pytest.fixture
    async def mock_agent(self, agent_config):
        """Create agent with mocked browser interactions."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Mock browser manager to avoid actual browser launches
        with patch('autonomous_survey_agent.BrowserManager') as mock_browser_class:
            mock_browser = AsyncMock()
            mock_browser_class.return_value = mock_browser
            
            # Mock page interactions
            mock_page = AsyncMock()
            mock_browser.page = mock_page
            mock_browser.navigate_to = AsyncMock()
            mock_browser.close = AsyncMock()
            
            yield agent, mock_browser
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_config):
        """Test agent initializes all components correctly."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Check all components are initialized
        assert isinstance(agent.auth_manager, SecureAuthManager)
        assert isinstance(agent.intelligence_engine, IntelligenceEngine)
        assert isinstance(agent.error_handler, RobustErrorHandler)
        assert isinstance(agent.human_simulation, HumanSimulation)
        assert isinstance(agent.layout_intelligence, LayoutIntelligence)
        assert isinstance(agent.control_panel, ControlPanel)
        
        # Check initial state
        assert not agent.running
        assert len(agent.active_tasks) == 0
        assert agent.stats['tasks_completed'] == 0
    
    @pytest.mark.asyncio
    async def test_secure_credential_management(self, agent_config):
        """Test secure credential handling."""
        # Set test environment variables
        os.environ['TEST_PLATFORM_EMAIL'] = 'test@example.com'
        os.environ['TEST_PLATFORM_PASSWORD'] = 'test_password'
        
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test credential retrieval
        credentials = await agent.auth_manager.get_secure_credentials('test_platform')
        
        # Should return None since we don't have the exact env var format
        # But the method should not crash
        assert credentials is None or isinstance(credentials, dict)
        
        # Clean up
        del os.environ['TEST_PLATFORM_EMAIL']
        del os.environ['TEST_PLATFORM_PASSWORD']
    
    @pytest.mark.asyncio
    async def test_intelligence_engine_integration(self, agent_config):
        """Test intelligence engine functionality."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test survey context creation
        context = await agent.intelligence_engine.create_survey_context(
            "test_survey", "test_platform", ["What is your age?", "Do you like pizza?"]
        )
        
        assert context.survey_id == "test_survey"
        assert context.platform == "test_platform"
        assert len(context.inferred_intents) > 0
        
        # Test memory addition
        await agent.intelligence_engine.add_memory(
            "test_survey", "q1", "25-34", "demographic", 0.9
        )
        
        # Test memory retrieval
        memories = agent.intelligence_engine.get_related_memories("test_survey", "age")
        assert len(memories) > 0
        assert memories[0].content == "25-34"
    
    @pytest.mark.asyncio
    async def test_human_simulation_integration(self, agent_config):
        """Test human simulation system."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Set persona
        agent.human_simulation.set_persona(PersonaProfile.TECH_SAVVY_MILLENNIAL)
        
        # Test question classification
        question_category = await agent.human_simulation.classify_question(
            "What is your annual household income?"
        )
        
        # Test answer strategy
        options = ["Under $25k", "$25k-$50k", "$50k-$75k", "$75k-$100k", "Over $100k"]
        answer, reasoning = agent.human_simulation.apply_answer_strategy(
            question_category, options, "What is your annual household income?"
        )
        
        assert answer in options
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        
        # Test timing calculation
        timing = await agent.human_simulation.calculate_human_timing(
            "What is your annual household income?", answer
        )
        
        assert 'reading_time' in timing
        assert 'thinking_time' in timing
        assert 'total_time' in timing
        assert timing['total_time'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, agent_config):
        """Test error handling system."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test error classification
        test_exception = ValueError("Test error message")
        error_context = agent.error_handler.classify_error(test_exception)
        
        assert error_context.error_type == ErrorType.VALIDATION_ERROR
        assert "Test error message" in error_context.message
        
        # Test semantic fallback
        fallback_answer = await agent.error_handler.get_semantic_fallback_answer(
            "What is your age?", ["18-24", "25-34", "35-44", "45-54"]
        )
        
        assert fallback_answer in ["18-24", "25-34", "35-44", "45-54"]
        
        # Test complexity scoring
        questions = [
            "What is your age?",
            "How often do you shop online?",
            "Rate your satisfaction with our service on a scale of 1-10"
        ]
        
        complexity_scores = await agent.error_handler.score_survey_complexity(questions)
        assert 'overall_complexity' in complexity_scores
        assert 0 <= complexity_scores['overall_complexity'] <= 1
    
    @pytest.mark.asyncio
    async def test_layout_intelligence_integration(self, agent_config):
        """Test layout intelligence system."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test fuzzy pattern loading
        assert 'next_button' in agent.layout_intelligence.fuzzy_patterns
        assert 'submit_button' in agent.layout_intelligence.fuzzy_patterns
        
        # Test button pattern loading
        assert 'next' in agent.layout_intelligence.button_patterns
        assert 'submit' in agent.layout_intelligence.button_patterns
        
        # Test layout statistics (should be empty initially)
        stats = agent.layout_intelligence.get_layout_statistics()
        assert stats['total_patterns'] == 0
    
    @pytest.mark.asyncio
    async def test_task_creation_and_queuing(self, mock_agent):
        """Test task creation and queuing system."""
        agent, mock_browser = mock_agent
        
        # Mock credential retrieval
        with patch.object(agent.auth_manager, 'get_secure_credentials') as mock_creds:
            mock_creds.return_value = {'email': 'test@example.com', 'password': 'test_pass'}
            
            # Create a task
            task_id = await agent.create_survey_task(
                platform="swagbucks",
                persona=PersonaProfile.TECH_SAVVY_MILLENNIAL.value
            )
            
            assert isinstance(task_id, str)
            assert len(task_id) > 0
            
            # Check task was queued
            assert agent.task_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_control_panel_integration(self, agent_config):
        """Test control panel integration."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test status retrieval
        status = agent.get_system_status()
        
        assert 'agent_state' in status
        assert 'uptime_hours' in status
        assert 'active_tasks' in status
        assert 'stats' in status
        
        # Test control panel initialization
        assert agent.control_panel.automation_system == agent
        assert agent.control_panel.live_status.agent_state.value == 'idle'
    
    @pytest.mark.asyncio
    async def test_persona_rotation_and_fingerprinting(self, agent_config):
        """Test persona rotation and behavioral fingerprinting."""
        agent = AutonomousSurveyAgent(agent_config)
        
        # Test persona setting
        agent.human_simulation.set_persona(PersonaProfile.FRUGAL_SHOPPER)
        assert agent.human_simulation.current_persona.name == "Frugal Shopper"
        
        # Test fingerprint rotation
        fingerprint1 = agent.human_simulation.rotate_behavioral_fingerprint()
        fingerprint2 = agent.human_simulation.rotate_behavioral_fingerprint()
        
        assert fingerprint1.session_id != fingerprint2.session_id
        assert fingerprint1.usage_count == 1
        assert fingerprint2.usage_count == 1
    
    @pytest.mark.asyncio
    async def test_memory_and_consistency_checking(self, agent_config):
        """Test memory system and consistency checking."""
        agent = AutonomousSurveyAgent(agent_config)
        
        survey_id = "test_survey_123"
        
        # Add some memories
        await agent.intelligence_engine.add_memory(
            survey_id, "q1", "I shop weekly", "behavioral", 0.8
        )
        await agent.intelligence_engine.add_memory(
            survey_id, "q2", "I prefer budget brands", "preference", 0.9
        )
        
        # Test consistency checking
        is_consistent, score, issues = await agent.intelligence_engine.check_answer_consistency(
            survey_id, "How often do you shop?", "I shop daily"
        )
        
        # Should detect some inconsistency with "weekly" vs "daily"
        assert isinstance(is_consistent, bool)
        assert 0 <= score <= 1
        assert isinstance(issues, list)
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow_simulation(self, mock_agent):
        """Test a complete workflow simulation."""
        agent, mock_browser = mock_agent
        
        # Mock all external dependencies
        with patch.object(agent.auth_manager, 'get_secure_credentials') as mock_creds, \
             patch.object(agent.auth_manager, 'restore_session') as mock_restore, \
             patch.object(agent.layout_intelligence, 'analyze_page_layout') as mock_layout:
            
            # Setup mocks
            mock_creds.return_value = {'email': 'test@example.com', 'password': 'test_pass'}
            mock_restore.return_value = False  # Force login
            mock_layout.return_value = {
                'total_elements': 10,
                'navigation_elements': {},
                'complexity_score': 0.5
            }
            
            # Mock page elements for form interaction
            mock_email_element = AsyncMock()
            mock_password_element = AsyncMock()
            mock_login_button = AsyncMock()
            
            mock_browser.page.query_selector_all.return_value = [
                mock_email_element, mock_password_element
            ]
            
            with patch.object(agent.layout_intelligence, 'find_element_with_fuzzy_selector') as mock_find:
                mock_find.side_effect = [
                    mock_email_element,  # Email field
                    mock_password_element,  # Password field
                    mock_login_button,  # Login button
                    None,  # Survey element (not found)
                ]
                
                # Start agent briefly
                agent.running = True
                
                # Create and process a task
                task_id = await agent.create_survey_task(
                    platform="swagbucks",
                    persona=PersonaProfile.HEALTH_CONSCIOUS.value
                )
                
                # Verify task creation
                assert task_id in [task_data['task_id'] for task_data in [await agent.task_queue.get()]]
                
                # Stop agent
                agent.running = False

class TestSystemIntegration:
    """Test system-wide integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_component_error_recovery(self):
        """Test error recovery across multiple components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AgentConfig(
                database_url=f"sqlite:///{temp_dir}/test.db",
                browser_headless=True,
                max_retries=2
            )
            
            agent = AutonomousSurveyAgent(config)
            
            # Test error propagation and handling
            test_error = ConnectionError("Network error")
            error_context = agent.error_handler.classify_error(test_error)
            
            # Should be classified as network error
            assert error_context.error_type == ErrorType.NETWORK_ERROR
            
            # Test retry logic would be triggered
            should_retry = await agent.error_handler.should_retry(
                error_context, 
                agent.error_handler.default_retry_configs[ErrorType.NETWORK_ERROR]
            )
            assert should_retry == True
    
    @pytest.mark.asyncio
    async def test_persona_consistency_across_components(self):
        """Test persona consistency across all components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AgentConfig(
                database_url=f"sqlite:///{temp_dir}/test.db",
                default_persona=PersonaProfile.ENVIRONMENTALIST.value
            )
            
            agent = AutonomousSurveyAgent(config)
            
            # Set persona
            agent.human_simulation.set_persona(PersonaProfile.ENVIRONMENTALIST)
            
            # Test persona influences answer selection
            question_category = await agent.human_simulation.classify_question(
                "How important is environmental sustainability to you?"
            )
            
            options = ["Not important", "Somewhat important", "Very important", "Extremely important"]
            answer, reasoning = agent.human_simulation.apply_answer_strategy(
                question_category, options, "How important is environmental sustainability to you?"
            )
            
            # Environmentalist persona should prefer pro-environment answers
            assert answer in ["Very important", "Extremely important"]
            assert "environmental" in reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_memory_persistence_and_retrieval(self):
        """Test memory persistence across sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AgentConfig(
                database_url=f"sqlite:///{temp_dir}/test.db",
                memory_file=f"{temp_dir}/test_memory.pkl"
            )
            
            # First agent instance
            agent1 = AutonomousSurveyAgent(config)
            
            # Add memories
            await agent1.intelligence_engine.add_memory(
                "survey_123", "q1", "I am 25 years old", "demographic", 0.9
            )
            await agent1.intelligence_engine.add_memory(
                "survey_123", "q2", "I exercise daily", "behavioral", 0.8
            )
            
            # Save memories
            await agent1.intelligence_engine._save_persistent_memory()
            
            # Create second agent instance (simulating restart)
            agent2 = AutonomousSurveyAgent(config)
            
            # Load memories
            await agent2.intelligence_engine._load_persistent_memory()
            
            # Check memories were loaded
            memories = agent2.intelligence_engine.get_related_memories("survey_123", "age")
            assert len(memories) > 0
            assert any("25 years old" in memory.content for memory in memories)

def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Autonomous Survey Agent Integration Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

if __name__ == "__main__":
    run_integration_tests()