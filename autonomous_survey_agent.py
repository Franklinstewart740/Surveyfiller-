"""
Autonomous Survey Agent - Main Application
Comprehensive survey automation system with all advanced features integrated.
"""

import asyncio
import logging
import json
import os
import sys
import time
import threading
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from flask import Flask
import signal

# Import all our enhanced modules
from secure_auth_manager import SecureAuthManager
from intelligence_engine import IntelligenceEngine, QuestionType, SurveyIntent
from robust_error_handler import RobustErrorHandler, ErrorType, ErrorSeverity
from control_panel import ControlPanel, AgentState
from human_simulation import HumanSimulation, PersonaProfile, QuestionCategory
from layout_intelligence import LayoutIntelligence, ElementRole
from browser_manager import BrowserManager
from database_models import DatabaseManager, Platform, Account, Task, TaskStatus

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_survey_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for the autonomous survey agent."""
    # Database
    database_url: str = "sqlite:///autonomous_survey_agent.db"
    
    # AI Configuration
    openai_api_key: Optional[str] = None
    ai_model: str = "gpt-3.5-turbo"
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-r1:8b"
    
    # Security
    encryption_key_file: str = ".encryption_key"
    sessions_file: str = "encrypted_sessions.dat"
    
    # System Configuration
    max_concurrent_tasks: int = 3
    browser_headless: bool = True
    anti_detection_enabled: bool = True
    proxy_rotation_enabled: bool = False
    
    # Timing and Behavior
    human_simulation_enabled: bool = True
    default_persona: str = PersonaProfile.TECH_SAVVY_MILLENNIAL.value
    
    # Error Handling
    max_retries: int = 3
    retry_delay: int = 5
    
    # Control Panel
    control_panel_host: str = "0.0.0.0"
    control_panel_port: int = 12000
    
    # Logging and Monitoring
    screenshot_logging: bool = True
    performance_tracking: bool = True
    memory_file: str = "survey_memory.pkl"
    layout_cache_file: str = "layout_cache.pkl"

class AutonomousSurveyAgent:
    """Main autonomous survey agent with all advanced features."""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self._load_config_from_env()
        
        # Initialize core components
        self.db_manager = DatabaseManager(self.config.database_url)
        self.auth_manager = SecureAuthManager(asdict(self.config))
        self.intelligence_engine = IntelligenceEngine(asdict(self.config))
        self.error_handler = RobustErrorHandler(asdict(self.config))
        self.human_simulation = HumanSimulation(asdict(self.config))
        self.layout_intelligence = LayoutIntelligence(asdict(self.config))
        
        # Task management
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue = asyncio.Queue()
        self.running = False
        
        # Performance tracking
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_questions_answered': 0,
            'total_captchas_solved': 0,
            'total_surveys_completed': 0,
            'uptime_start': datetime.now(),
            'agent_performance_score': 1.0
        }
        
        # Control panel
        self.control_panel = ControlPanel(self, asdict(self.config))
        
        # Initialize database
        self._initialize_database()
        
        logger.info("Autonomous Survey Agent initialized successfully")
    
    def _load_config_from_env(self):
        """Load configuration from environment variables."""
        # API Keys
        self.config.openai_api_key = os.getenv('OPENAI_API_KEY', self.config.openai_api_key)
        self.config.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', self.config.deepseek_api_key)
        
        # Database
        self.config.database_url = os.getenv('DATABASE_URL', self.config.database_url)
        
        # System settings
        self.config.max_concurrent_tasks = int(os.getenv('MAX_CONCURRENT_TASKS', str(self.config.max_concurrent_tasks)))
        self.config.browser_headless = os.getenv('BROWSER_HEADLESS', 'true').lower() == 'true'
        self.config.anti_detection_enabled = os.getenv('ANTI_DETECTION_ENABLED', 'true').lower() == 'true'
        
        # Control panel
        self.config.control_panel_port = int(os.getenv('CONTROL_PANEL_PORT', str(self.config.control_panel_port)))
        
        logger.info("Configuration loaded from environment variables")
    
    def _initialize_database(self):
        """Initialize database with required tables and data."""
        try:
            self.db_manager.create_tables()
            self.db_manager.init_default_data()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def start_agent(self):
        """Start the autonomous survey agent."""
        logger.info("🤖 Starting Autonomous Survey Agent")
        self.running = True
        
        # Start background services
        background_tasks = []
        
        # Start authentication manager cleanup
        background_tasks.append(
            asyncio.create_task(self.auth_manager.start_periodic_cleanup())
        )
        
        # Start task processor
        background_tasks.append(
            asyncio.create_task(self._process_task_queue())
        )
        
        # Start performance monitoring
        background_tasks.append(
            asyncio.create_task(self._performance_monitor())
        )
        
        # Start control panel in separate thread
        control_panel_thread = threading.Thread(
            target=self.control_panel.start_server,
            args=(self.config.control_panel_host, self.config.control_panel_port),
            daemon=True
        )
        control_panel_thread.start()
        
        logger.info(f"🌐 Control panel started on http://{self.config.control_panel_host}:{self.config.control_panel_port}")
        logger.info("✅ All systems operational")
        
        # Wait for tasks or shutdown signal
        try:
            await asyncio.gather(*background_tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            await self.stop_agent()
    
    async def stop_agent(self):
        """Stop the agent gracefully."""
        logger.info("🛑 Stopping Autonomous Survey Agent")
        self.running = False
        
        # Cancel active tasks
        for task_id, task_data in self.active_tasks.items():
            if 'browser_manager' in task_data:
                try:
                    await task_data['browser_manager'].close()
                except Exception as e:
                    logger.error(f"Error closing browser for task {task_id}: {e}")
        
        # Save final state
        await self.intelligence_engine._save_persistent_memory()
        await self.layout_intelligence._save_layout_cache()
        
        logger.info("✅ Agent stopped successfully")
    
    async def create_survey_task(self, platform: str, survey_id: Optional[str] = None,
                               persona: Optional[str] = None, config: Dict[str, Any] = None) -> str:
        """Create a new survey automation task."""
        try:
            # Get secure credentials
            credentials = await self.auth_manager.get_secure_credentials(platform)
            if not credentials:
                raise ValueError(f"No secure credentials found for platform: {platform}")
            
            # Set persona for human simulation
            if persona:
                persona_profile = PersonaProfile(persona)
                self.human_simulation.set_persona(persona_profile)
            
            # Create task in database
            session = self.db_manager.get_session()
            try:
                platform_obj = session.query(Platform).filter_by(name=platform.lower()).first()
                if not platform_obj:
                    raise ValueError(f"Unsupported platform: {platform}")
                
                # Get or create account
                account = session.query(Account).filter_by(
                    platform_id=platform_obj.id,
                    email=credentials['email']
                ).first()
                
                if not account:
                    account = Account(
                        platform_id=platform_obj.id,
                        email=credentials['email'],
                        password_hash=self.auth_manager.encrypt_credentials(credentials)
                    )
                    session.add(account)
                    session.commit()
                
                # Create task
                task = Task(
                    platform_id=platform_obj.id,
                    account_id=account.id,
                    survey_id=survey_id,
                    config=config or {}
                )
                session.add(task)
                session.commit()
                
                # Add to queue
                task_data = {
                    'task_id': task.id,
                    'platform': platform,
                    'credentials': credentials,
                    'survey_id': survey_id,
                    'persona': persona,
                    'config': config or {}
                }
                
                await self.task_queue.put(task_data)
                
                logger.info(f"✅ Created survey task {task.id} for {platform}")
                return task.id
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Failed to create survey task: {e}")
            raise
    
    async def _process_task_queue(self):
        """Process tasks from the queue."""
        logger.info("🔄 Task queue processor started")
        
        while self.running:
            try:
                # Check concurrent task limit
                if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                try:
                    task_data = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Start task execution
                task_id = task_data['task_id']
                self.active_tasks[task_id] = task_data
                
                # Execute task in background
                asyncio.create_task(self._execute_survey_task(task_data))
                
            except Exception as e:
                logger.error(f"❌ Error in task queue processor: {e}")
                await asyncio.sleep(5)
    
    async def _execute_survey_task(self, task_data: Dict[str, Any]):
        """Execute a complete survey automation task."""
        task_id = task_data['task_id']
        platform = task_data['platform']
        credentials = task_data['credentials']
        survey_id = task_data.get('survey_id')
        persona = task_data.get('persona')
        
        session = self.db_manager.get_session()
        browser_manager = None
        
        try:
            logger.info(f"🚀 Starting survey task {task_id} on {platform}")
            
            # Update task status
            task = session.query(Task).filter_by(id=task_id).first()
            task.status = TaskStatus.RUNNING.value
            task.start_time = datetime.utcnow()
            task.current_step = "Initializing"
            session.commit()
            
            # Initialize browser with anti-detection
            browser_manager = BrowserManager()
            await browser_manager.start_browser(
                headless=self.config.browser_headless,
                browser_type='chromium'
            )
            
            task_data['browser_manager'] = browser_manager
            
            # Set persona if specified
            if persona:
                persona_profile = PersonaProfile(persona)
                self.human_simulation.set_persona(persona_profile)
                logger.info(f"🎭 Set persona to {persona}")
            
            # Rotate behavioral fingerprint
            fingerprint = self.human_simulation.rotate_behavioral_fingerprint()
            logger.info(f"🔄 Using behavioral fingerprint {fingerprint.session_id}")
            
            # Try to restore session
            session_restored = await self.auth_manager.restore_session(
                browser_manager, platform, credentials['email']
            )
            
            if not session_restored:
                # Perform login
                await self._perform_secure_login(browser_manager, platform, credentials, task, session)
            else:
                logger.info(f"✅ Restored session for {credentials['email']} on {platform}")
            
            # Navigate to surveys page
            await self._navigate_to_surveys(browser_manager, platform, task, session)
            
            # Find and start survey
            survey_url = await self._find_and_start_survey(browser_manager, platform, survey_id, task, session)
            
            # Complete survey with intelligence
            questions_answered = await self._complete_survey_intelligently(
                browser_manager, platform, survey_url, task, session
            )
            
            # Save session for future use
            await self._save_browser_session(browser_manager, platform, credentials['email'])
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED.value
            task.end_time = datetime.utcnow()
            task.current_step = "Completed"
            task.progress = 100.0
            task.questions_answered = questions_answered
            
            if task.start_time:
                duration = (task.end_time - task.start_time).total_seconds()
                task.actual_duration = int(duration)
            
            session.commit()
            
            # Update statistics
            self.stats['tasks_completed'] += 1
            self.stats['total_questions_answered'] += questions_answered
            self.stats['total_surveys_completed'] += 1
            
            logger.info(f"✅ Task {task_id} completed successfully - {questions_answered} questions answered")
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")
            
            # Handle error with comprehensive error handling
            error_context = self.error_handler.classify_error(e, {
                'task_id': task_id,
                'platform': platform,
                'step': task.current_step if 'task' in locals() else 'unknown'
            })
            
            await self.error_handler.log_error_with_context(error_context)
            
            # Update task with error
            task = session.query(Task).filter_by(id=task_id).first()
            task.status = TaskStatus.FAILED.value
            task.error_message = str(e)
            task.end_time = datetime.utcnow()
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count <= self.config.max_retries:
                logger.info(f"🔄 Retrying task {task_id} (attempt {task.retry_count}/{self.config.max_retries})")
                task.status = TaskStatus.PENDING.value
                task.current_step = "Retrying"
                
                # Re-queue task with delay
                await asyncio.sleep(self.config.retry_delay * task.retry_count)
                await self.task_queue.put(task_data)
            else:
                self.stats['tasks_failed'] += 1
            
            session.commit()
            
        finally:
            # Cleanup
            if browser_manager:
                try:
                    await browser_manager.close()
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")
            
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
            session.close()
    
    async def _perform_secure_login(self, browser_manager: BrowserManager, platform: str,
                                  credentials: Dict[str, str], task: Task, session):
        """Perform secure login with CAPTCHA handling."""
        task.current_step = "Logging in"
        task.progress = 20.0
        session.commit()
        
        # Navigate to login page
        login_url = self._get_platform_login_url(platform)
        await browser_manager.navigate_to(login_url)
        
        # Analyze page layout
        layout_analysis = await self.layout_intelligence.analyze_page_layout(
            browser_manager.page, platform
        )
        
        # Find login form elements
        email_element = await self.layout_intelligence.find_element_with_fuzzy_selector(
            browser_manager.page, 'text_input'
        )
        
        password_element = await self.layout_intelligence.find_element_with_fuzzy_selector(
            browser_manager.page, 'text_input'
        )
        
        if not email_element or not password_element:
            # Try OCR-based element finding
            email_element = await self.layout_intelligence.find_element_by_ocr_text(
                browser_manager.page, 'email'
            )
            password_element = await self.layout_intelligence.find_element_by_ocr_text(
                browser_manager.page, 'password'
            )
        
        if not email_element or not password_element:
            raise Exception("Could not find login form elements")
        
        # Fill credentials with human-like timing
        await self._human_like_form_fill(email_element, credentials['email'])
        await self._human_like_form_fill(password_element, credentials['password'])
        
        # Handle CAPTCHA if present
        await self._handle_captcha_if_present(browser_manager, platform, task.id)
        
        # Find and click login button
        login_button = await self.layout_intelligence.find_element_with_fuzzy_selector(
            browser_manager.page, 'submit_button'
        )
        
        if login_button:
            await self._human_like_click(login_button)
        else:
            # Try OCR-based button finding
            login_button = await self.layout_intelligence.find_element_by_ocr_text(
                browser_manager.page, 'login'
            )
            if login_button:
                await self._human_like_click(login_button)
            else:
                raise Exception("Could not find login button")
        
        # Wait for login to complete
        await asyncio.sleep(3)
        
        # Verify login success
        current_url = browser_manager.page.url
        if 'login' in current_url.lower():
            raise Exception("Login appears to have failed")
        
        logger.info(f"✅ Successfully logged in to {platform}")
    
    async def _navigate_to_surveys(self, browser_manager: BrowserManager, platform: str,
                                 task: Task, session):
        """Navigate to surveys page."""
        task.current_step = "Navigating to surveys"
        task.progress = 30.0
        session.commit()
        
        surveys_url = self._get_platform_surveys_url(platform)
        await browser_manager.navigate_to(surveys_url)
        
        # Wait for page to load
        await asyncio.sleep(2)
        
        logger.info(f"📋 Navigated to surveys page for {platform}")
    
    async def _find_and_start_survey(self, browser_manager: BrowserManager, platform: str,
                                   survey_id: Optional[str], task: Task, session) -> str:
        """Find and start a survey."""
        task.current_step = "Finding survey"
        task.progress = 40.0
        session.commit()
        
        # Analyze page layout to find survey links
        layout_analysis = await self.layout_intelligence.analyze_page_layout(
            browser_manager.page, platform
        )
        
        # Look for survey links or buttons
        survey_elements = await browser_manager.page.query_selector_all(
            'a[href*="survey"], button:has-text("Start"), .survey-link'
        )
        
        if not survey_elements:
            # Try OCR-based finding
            survey_element = await self.layout_intelligence.find_element_by_ocr_text(
                browser_manager.page, 'survey'
            )
            if survey_element:
                survey_elements = [survey_element]
        
        if not survey_elements:
            raise Exception("No available surveys found")
        
        # Select first available survey
        selected_survey = survey_elements[0]
        survey_url = await selected_survey.get_attribute('href') or browser_manager.page.url
        
        # Click to start survey
        await self._human_like_click(selected_survey)
        
        # Wait for survey to load
        await asyncio.sleep(3)
        
        logger.info(f"🎯 Started survey: {survey_url}")
        return survey_url
    
    async def _complete_survey_intelligently(self, browser_manager: BrowserManager, platform: str,
                                           survey_url: str, task: Task, session) -> int:
        """Complete survey using intelligence engine and human simulation."""
        task.current_step = "Completing survey"
        task.progress = 50.0
        session.commit()
        
        # Create survey context
        survey_context = await self.intelligence_engine.create_survey_context(
            task.id, platform
        )
        
        questions_answered = 0
        max_questions = 100  # Safety limit
        
        while questions_answered < max_questions:
            try:
                # Analyze current page layout
                layout_analysis = await self.layout_intelligence.analyze_page_layout(
                    browser_manager.page, platform
                )
                
                # Look for question text
                question_text = await self._extract_question_text(browser_manager.page)
                if not question_text:
                    logger.info("No more questions found, survey may be complete")
                    break
                
                # Classify question
                question_category = await self.human_simulation.classify_question(question_text)
                question_type = await self.intelligence_engine.classify_question_type(question_text)
                
                # Get answer options
                answer_options = await self._extract_answer_options(browser_manager.page)
                
                # Check for consistency with previous answers
                is_consistent, consistency_score, inconsistencies = await self.intelligence_engine.check_answer_consistency(
                    task.id, question_text, ""
                )
                
                # Generate answer using human simulation
                selected_answer, reasoning = self.human_simulation.apply_answer_strategy(
                    question_category, answer_options, question_text
                )
                
                # Calculate human-like timing
                timing = await self.human_simulation.calculate_human_timing(
                    question_text, selected_answer
                )
                
                # Wait for reading and thinking time
                await asyncio.sleep(timing['reading_time'])
                await asyncio.sleep(timing['thinking_time'])
                
                # Find and interact with answer element
                await self._select_answer_with_human_behavior(
                    browser_manager.page, selected_answer, answer_options
                )
                
                # Add to memory
                await self.intelligence_engine.add_memory(
                    task.id, f"q_{questions_answered}", selected_answer, "answer", 0.8
                )
                
                # Log reasoning
                await self.intelligence_engine.log_answer_reasoning(
                    task.id, f"q_{questions_answered}", question_text,
                    question_type, selected_answer, consistency_score,
                    is_consistent, []
                )
                
                # Log interaction for human simulation
                self.human_simulation.log_interaction(
                    question_text, question_category, selected_answer, reasoning, timing
                )
                
                # Look for next button
                next_button = await self.layout_intelligence.find_element_with_fuzzy_selector(
                    browser_manager.page, 'next_button'
                )
                
                if not next_button:
                    # Try submit button
                    next_button = await self.layout_intelligence.find_element_with_fuzzy_selector(
                        browser_manager.page, 'submit_button'
                    )
                
                if next_button:
                    await asyncio.sleep(timing['selection_time'])
                    await self._human_like_click(next_button)
                    
                    # Wait for page transition
                    await asyncio.sleep(2)
                else:
                    logger.info("No next/submit button found, survey may be complete")
                    break
                
                questions_answered += 1
                
                # Update progress
                progress = 50.0 + (questions_answered / max_questions) * 40.0
                task.progress = min(progress, 90.0)
                session.commit()
                
                # Update control panel
                self.control_panel.update_survey_progress(
                    task.id, task.id, platform, questions_answered,
                    max_questions, f"Question {questions_answered}", []
                )
                
                logger.info(f"📝 Answered question {questions_answered}: {selected_answer}")
                
            except Exception as e:
                logger.error(f"Error answering question {questions_answered + 1}: {e}")
                
                # Try to skip question or continue
                skip_button = await self.layout_intelligence.find_element_with_fuzzy_selector(
                    browser_manager.page, 'skip_button'
                )
                
                if skip_button:
                    await self._human_like_click(skip_button)
                    await asyncio.sleep(2)
                    continue
                else:
                    break
        
        logger.info(f"✅ Survey completed - {questions_answered} questions answered")
        return questions_answered
    
    async def _extract_question_text(self, page) -> Optional[str]:
        """Extract question text from the page."""
        # Try common question selectors
        question_selectors = [
            '.question-text', '.question', '[class*="question"]',
            'h1', 'h2', 'h3', '.survey-question', 'p:has-text("?")'
        ]
        
        for selector in question_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and len(text.strip()) > 10:  # Reasonable question length
                        return text.strip()
            except:
                continue
        
        return None
    
    async def _extract_answer_options(self, page) -> List[str]:
        """Extract answer options from the page."""
        options = []
        
        # Try radio buttons
        radio_elements = await page.query_selector_all('input[type="radio"]')
        for radio in radio_elements:
            try:
                label = await page.query_selector(f'label[for="{await radio.get_attribute("id")}"]')
                if label:
                    text = await label.text_content()
                    if text:
                        options.append(text.strip())
            except:
                continue
        
        # Try checkboxes
        if not options:
            checkbox_elements = await page.query_selector_all('input[type="checkbox"]')
            for checkbox in checkbox_elements:
                try:
                    label = await page.query_selector(f'label[for="{await checkbox.get_attribute("id")}"]')
                    if label:
                        text = await label.text_content()
                        if text:
                            options.append(text.strip())
                except:
                    continue
        
        # Try select options
        if not options:
            select_elements = await page.query_selector_all('select option')
            for option in select_elements:
                try:
                    text = await option.text_content()
                    if text and text.strip() != "":
                        options.append(text.strip())
                except:
                    continue
        
        # Try button options
        if not options:
            button_elements = await page.query_selector_all('button:not([type="submit"]):not([type="button"])')
            for button in button_elements:
                try:
                    text = await button.text_content()
                    if text and len(text.strip()) > 1:
                        options.append(text.strip())
                except:
                    continue
        
        return options
    
    async def _select_answer_with_human_behavior(self, page, selected_answer: str, 
                                               answer_options: List[str]):
        """Select answer with human-like behavior."""
        # Find the matching option element
        for option_text in answer_options:
            if selected_answer.lower() in option_text.lower() or option_text.lower() in selected_answer.lower():
                # Try to find the corresponding input element
                elements = await page.query_selector_all(f'input, button, select')
                
                for element in elements:
                    try:
                        # Check if this element corresponds to our answer
                        element_text = await element.text_content() or ""
                        element_value = await element.get_attribute('value') or ""
                        
                        if option_text.lower() in element_text.lower() or \
                           option_text.lower() in element_value.lower():
                            
                            await self._human_like_click(element)
                            return
                    except:
                        continue
                
                # Try finding by label
                label_element = await page.query_selector(f'label:has-text("{option_text}")')
                if label_element:
                    await self._human_like_click(label_element)
                    return
        
        # Fallback: click first available option
        first_input = await page.query_selector('input[type="radio"], input[type="checkbox"], button')
        if first_input:
            await self._human_like_click(first_input)
    
    async def _human_like_click(self, element):
        """Perform human-like click with timing and movement."""
        # Get element position
        bounding_box = await element.bounding_box()
        if not bounding_box:
            await element.click()
            return
        
        # Calculate click position with slight randomness
        click_x = bounding_box['x'] + bounding_box['width'] / 2 + random.uniform(-5, 5)
        click_y = bounding_box['y'] + bounding_box['height'] / 2 + random.uniform(-5, 5)
        
        # Generate curved mouse path if human simulation is enabled
        if self.config.human_simulation_enabled:
            current_pos = (100, 100)  # Approximate current position
            target_pos = (int(click_x), int(click_y))
            
            mouse_path = await self.human_simulation.generate_curved_mouse_path(
                current_pos, target_pos
            )
            
            # Simulate mouse movement along path
            for i, (x, y) in enumerate(mouse_path):
                if i > 0:  # Skip first point
                    await element.page.mouse.move(x, y)
                    await asyncio.sleep(0.01)  # Small delay between movements
        
        # Perform click with human-like timing
        await element.click()
        
        # Small delay after click
        await asyncio.sleep(random.uniform(0.1, 0.3))
    
    async def _human_like_form_fill(self, element, text: str):
        """Fill form field with human-like typing."""
        await element.click()
        await element.clear()
        
        if self.config.human_simulation_enabled and self.human_simulation.current_persona:
            # Type with human-like delays
            timing_profile = self.human_simulation.current_persona.timing_profile
            
            for char in text:
                await element.type(char)
                
                # Add typing delay with variance
                base_delay = 60 / timing_profile.typing_speed_cpm  # Convert CPM to delay
                delay = base_delay + random.gauss(0, base_delay * 0.3)
                delay = max(0.01, delay)  # Minimum delay
                
                await asyncio.sleep(delay)
                
                # Occasional pause
                if random.random() < timing_profile.pause_probability:
                    await asyncio.sleep(random.uniform(0.2, 0.8))
        else:
            # Simple typing
            await element.type(text, delay=random.uniform(50, 150))
    
    async def _handle_captcha_if_present(self, browser_manager: BrowserManager, 
                                       platform: str, task_id: str):
        """Handle CAPTCHA if present on the page."""
        # Check for CAPTCHA elements
        captcha_elements = await browser_manager.page.query_selector_all(
            '.captcha, #captcha, [class*="captcha"], iframe[src*="captcha"]'
        )
        
        if captcha_elements:
            logger.info("🔐 CAPTCHA detected, adding to queue")
            
            # Take screenshot for CAPTCHA solving
            screenshot = await browser_manager.page.screenshot()
            
            # Add to CAPTCHA queue
            challenge_id = await self.auth_manager.add_captcha_challenge(
                platform, task_id, "image_captcha", screenshot
            )
            
            # For now, just wait and hope it resolves or times out
            # In production, this would integrate with CAPTCHA solving services
            await asyncio.sleep(30)
            
            # Mark as resolved (placeholder)
            await self.auth_manager.resolve_captcha_challenge(challenge_id, False)
    
    async def _save_browser_session(self, browser_manager: BrowserManager, 
                                  platform: str, account_email: str):
        """Save browser session for future use."""
        try:
            # Get cookies
            cookies = await browser_manager.page.context.cookies()
            
            # Get local storage
            local_storage = await browser_manager.page.evaluate('''() => {
                const storage = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    storage[key] = localStorage.getItem(key);
                }
                return storage;
            }''')
            
            # Get session storage
            session_storage = await browser_manager.page.evaluate('''() => {
                const storage = {};
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    storage[key] = sessionStorage.getItem(key);
                }
                return storage;
            }''')
            
            # Get user agent
            user_agent = await browser_manager.page.evaluate('() => navigator.userAgent')
            
            # Save session
            await self.auth_manager.save_session(
                platform, account_email, 
                {'cookies': cookies}, local_storage, session_storage,
                user_agent, {}
            )
            
        except Exception as e:
            logger.error(f"Failed to save browser session: {e}")
    
    async def _performance_monitor(self):
        """Monitor system performance and update statistics."""
        while self.running:
            try:
                # Calculate agent performance score
                total_tasks = self.stats['tasks_completed'] + self.stats['tasks_failed']
                if total_tasks > 0:
                    success_rate = self.stats['tasks_completed'] / total_tasks
                    self.stats['agent_performance_score'] = success_rate
                
                # Clean up old memories and caches
                await self.intelligence_engine.cleanup_old_memories(7)
                
                # Update control panel with current stats
                # (This would be handled by the control panel's background tasks)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
    
    def _get_platform_login_url(self, platform: str) -> str:
        """Get login URL for platform."""
        urls = {
            'swagbucks': 'https://www.swagbucks.com/login',
            'inboxdollars': 'https://www.inboxdollars.com/login',
        }
        return urls.get(platform.lower(), f'https://{platform}.com/login')
    
    def _get_platform_surveys_url(self, platform: str) -> str:
        """Get surveys URL for platform."""
        urls = {
            'swagbucks': 'https://www.swagbucks.com/surveys',
            'inboxdollars': 'https://www.inboxdollars.com/surveys',
        }
        return urls.get(platform.lower(), f'https://{platform}.com/surveys')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'agent_state': 'active' if self.running else 'stopped',
            'uptime_hours': uptime.total_seconds() / 3600,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'stats': self.stats,
            'error_statistics': self.error_handler.get_error_statistics(),
            'layout_statistics': self.layout_intelligence.get_layout_statistics(),
            'interaction_summary': self.human_simulation.get_interaction_summary(),
            'captcha_queue_size': len(self.auth_manager.get_captcha_queue())
        }

# Main execution
async def main():
    """Main entry point for the autonomous survey agent."""
    # Load configuration
    config = AgentConfig()
    
    # Create and start agent
    agent = AutonomousSurveyAgent(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(agent.stop_agent())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await agent.start_agent()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await agent.stop_agent()

if __name__ == "__main__":
    asyncio.run(main())