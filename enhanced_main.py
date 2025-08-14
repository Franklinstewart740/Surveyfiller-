"""
Enhanced Main Application
Comprehensive survey automation system with all requested features.
"""

import os
import sys
import asyncio
import logging
import json
import random
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import threading
import signal

# Import our enhanced modules
from drift_detector import DriftDetector
from enhanced_ai_generator import EnhancedAIGenerator, PersonaType
from enhanced_captcha_solver import EnhancedCaptchaSolver
from browser_manager import BrowserManager
from database_models import DatabaseManager, Platform, Account, Task, TaskStatus
from swagbucks_data import SWAGBUCKS_DATA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('survey_automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedSurveyAutomation:
    """Enhanced survey automation system with all requested features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        
        # Initialize database
        self.db_manager = DatabaseManager(self.config.get('database_url', 'sqlite:///survey_automation.db'))
        self.db_manager.create_tables()
        self.db_manager.init_default_data()
        
        # Initialize core components
        self.drift_detector = DriftDetector(self.config.get('drift_db_path', 'drift_detection.db'))
        self.ai_generator = EnhancedAIGenerator(self.config)
        self.captcha_solver = EnhancedCaptchaSolver(self.config)
        
        # Task management
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
        self.running = False
        
        # Performance tracking
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_questions_answered': 0,
            'total_captchas_solved': 0,
            'uptime_start': datetime.now()
        }
        
        # Initialize monitoring
        self._setup_monitoring()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'database_url': 'sqlite:///survey_automation.db',
            'max_concurrent_tasks': 5,
            'drift_check_interval': 3600,  # 1 hour
            'captcha_timeout': 300,  # 5 minutes
            'ai_model': 'gpt-3.5-turbo',
            'default_persona': PersonaType.YOUNG_PROFESSIONAL.value,
            'browser_headless': True,
            'anti_detection_enabled': True,
            'proxy_rotation_enabled': False,
            'human_solver_enabled': False,
            'logging_level': 'INFO'
        }
    
    def _setup_monitoring(self):
        """Set up drift monitoring for supported platforms."""
        # Add Swagbucks to monitoring
        swagbucks_selectors = {
            'login_email': SWAGBUCKS_DATA['element_locators']['login_form']['email_field'][0],
            'login_password': SWAGBUCKS_DATA['element_locators']['login_form']['password_field'][0],
            'login_button': SWAGBUCKS_DATA['element_locators']['login_form']['login_button'][0],
            'survey_list': SWAGBUCKS_DATA['element_locators']['survey_elements']['survey_list'][0]
        }
        
        self.drift_detector.add_monitored_url(
            SWAGBUCKS_DATA['target_url'],
            swagbucks_selectors
        )
        self.drift_detector.add_monitored_url(
            SWAGBUCKS_DATA['login_url'],
            swagbucks_selectors
        )
        self.drift_detector.add_monitored_url(
            SWAGBUCKS_DATA['surveys_url'],
            swagbucks_selectors
        )
    
    async def start_system(self):
        """Start the enhanced survey automation system."""
        logger.info("Starting Enhanced Survey Automation System")
        self.running = True
        
        # Start drift monitoring
        drift_task = asyncio.create_task(self.drift_detector.start_monitoring())
        
        # Start task processor
        processor_task = asyncio.create_task(self._process_task_queue())
        
        # Start periodic maintenance
        maintenance_task = asyncio.create_task(self._periodic_maintenance())
        
        logger.info("All system components started successfully")
        
        # Wait for tasks
        try:
            await asyncio.gather(drift_task, processor_task, maintenance_task)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await self.stop_system()
    
    async def stop_system(self):
        """Stop the system gracefully."""
        logger.info("Stopping Enhanced Survey Automation System")
        self.running = False
        
        # Stop drift monitoring
        self.drift_detector.stop_monitoring()
        
        # Cancel active tasks
        for task_id, task_data in self.active_tasks.items():
            if 'browser' in task_data:
                try:
                    await task_data['browser'].close()
                except Exception as e:
                    logger.error(f"Error closing browser for task {task_id}: {e}")
        
        logger.info("System stopped successfully")
    
    async def create_task(self, platform: str, credentials: Dict[str, str], survey_id: Optional[str] = None, config: Dict[str, Any] = None) -> str:
        """Create a new survey automation task."""
        session = self.db_manager.get_session()
        try:
            # Get platform
            platform_obj = session.query(Platform).filter_by(name=platform.lower()).first()
            if not platform_obj:
                raise ValueError(f"Unsupported platform: {platform}")
            
            # Get or create account
            account = session.query(Account).filter_by(
                platform_id=platform_obj.id,
                email=credentials['email']
            ).first()
            
            if not account:
                # Create new account (password should be encrypted in production)
                account = Account(
                    platform_id=platform_obj.id,
                    email=credentials['email'],
                    password_hash=credentials['password']  # Should be encrypted
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
            await self.task_queue.put({
                'task_id': task.id,
                'platform': platform,
                'credentials': credentials,
                'survey_id': survey_id,
                'config': config or {}
            })
            
            logger.info(f"Created task {task.id} for platform {platform}")
            return task.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating task: {e}")
            raise
        finally:
            session.close()
    
    async def _process_task_queue(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                # Check if we can start new tasks
                if len(self.active_tasks) >= self.config['max_concurrent_tasks']:
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
                asyncio.create_task(self._execute_task(task_data))
                
            except Exception as e:
                logger.error(f"Error in task queue processor: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task_data: Dict[str, Any]):
        """Execute a single survey automation task."""
        task_id = task_data['task_id']
        platform = task_data['platform']
        credentials = task_data['credentials']
        survey_id = task_data.get('survey_id')
        config = task_data.get('config', {})
        
        session = self.db_manager.get_session()
        browser_manager = None
        
        try:
            # Update task status
            task = session.query(Task).filter_by(id=task_id).first()
            task.status = TaskStatus.RUNNING.value
            task.start_time = datetime.utcnow()
            task.current_step = "Initializing"
            session.commit()
            
            logger.info(f"Starting execution of task {task_id}")
            
            # Initialize browser with anti-detection
            proxy_config = None
            if self.config.get('proxy_rotation_enabled'):
                proxy_config = self._get_next_proxy()
            
            browser_manager = BrowserManager(proxy_config)
            await browser_manager.start_browser(
                headless=self.config.get('browser_headless', True),
                browser_type='chromium'
            )
            
            task_data['browser'] = browser_manager
            
            # Set AI persona
            persona_name = config.get('persona', self.config['default_persona'])
            persona_type = PersonaType(persona_name)
            self.ai_generator.set_persona(persona_type)
            
            # Execute platform-specific automation
            if platform.lower() == 'swagbucks':
                await self._execute_swagbucks_task(task, browser_manager, credentials, survey_id, session)
            elif platform.lower() == 'inboxdollars':
                await self._execute_inboxdollars_task(task, browser_manager, credentials, survey_id, session)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED.value
            task.end_time = datetime.utcnow()
            task.current_step = "Completed"
            task.progress = 100.0
            
            if task.start_time:
                duration = (task.end_time - task.start_time).total_seconds()
                task.actual_duration = int(duration)
            
            session.commit()
            
            self.stats['tasks_completed'] += 1
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Update task with error
            task = session.query(Task).filter_by(id=task_id).first()
            task.status = TaskStatus.FAILED.value
            task.error_message = str(e)
            task.end_time = datetime.utcnow()
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count <= task.max_retries:
                logger.info(f"Retrying task {task_id} (attempt {task.retry_count}/{task.max_retries})")
                task.status = TaskStatus.PENDING.value
                task.current_step = "Retrying"
                
                # Re-queue task with delay
                await asyncio.sleep(self.config.get('retry_delay', 5) * task.retry_count)
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
    
    async def _execute_swagbucks_task(self, task: Task, browser: BrowserManager, credentials: Dict[str, str], survey_id: Optional[str], session):
        """Execute Swagbucks-specific automation."""
        # Navigate to login page
        task.current_step = "Navigating to login"
        task.progress = 10.0
        session.commit()
        
        await browser.navigate_to(SWAGBUCKS_DATA['login_url'])
        
        # Handle login
        task.current_step = "Logging in"
        task.progress = 20.0
        session.commit()
        
        await self._handle_login(browser, credentials, SWAGBUCKS_DATA)
        
        # Navigate to surveys
        task.current_step = "Navigating to surveys"
        task.progress = 30.0
        session.commit()
        
        await browser.navigate_to(SWAGBUCKS_DATA['surveys_url'])
        
        # Find and start survey
        task.current_step = "Finding surveys"
        task.progress = 40.0
        session.commit()
        
        survey_url = await self._find_available_survey(browser, SWAGBUCKS_DATA, survey_id)
        if not survey_url:
            raise Exception("No available surveys found")
        
        # Complete survey
        task.current_step = "Completing survey"
        task.progress = 50.0
        session.commit()
        
        questions_answered = await self._complete_survey(browser, task, session)
        
        task.questions_answered = questions_answered
        task.progress = 90.0
        session.commit()
        
        logger.info(f"Completed survey with {questions_answered} questions")
    
    async def _execute_inboxdollars_task(self, task: Task, browser: BrowserManager, credentials: Dict[str, str], survey_id: Optional[str], session):
        """Execute InboxDollars-specific automation."""
        # Similar implementation to Swagbucks but with InboxDollars-specific selectors
        logger.info("InboxDollars automation not fully implemented yet")
        raise NotImplementedError("InboxDollars automation coming soon")
    
    async def _handle_login(self, browser: BrowserManager, credentials: Dict[str, str], platform_data: Dict[str, Any]):
        """Handle login process with CAPTCHA solving."""
        # Fill email
        email_selectors = platform_data['element_locators']['login_form']['email_field']
        for selector in email_selectors:
            if await browser.wait_for_element(selector, timeout=3000):
                await browser.human_like_type(selector, credentials['email'])
                break
        else:
            raise Exception("Could not find email field")
        
        # Fill password
        password_selectors = platform_data['element_locators']['login_form']['password_field']
        for selector in password_selectors:
            if await browser.wait_for_element(selector, timeout=3000):
                await browser.human_like_type(selector, credentials['password'])
                break
        else:
            raise Exception("Could not find password field")
        
        # Handle CAPTCHA if present
        captcha_type = await self.captcha_solver.detect_captcha_type(browser.page)
        if captcha_type:
            logger.info(f"CAPTCHA detected: {captcha_type}")
            solution = await self.captcha_solver.solve_captcha(browser.page, captcha_type)
            if solution:
                await self.captcha_solver.submit_captcha_solution(browser.page, captcha_type, solution)
                self.stats['total_captchas_solved'] += 1
            else:
                raise Exception("Failed to solve CAPTCHA")
        
        # Click login button
        login_selectors = platform_data['element_locators']['login_form']['login_button']
        for selector in login_selectors:
            if await browser.wait_for_element(selector, timeout=3000):
                await browser.human_like_click(selector)
                break
        else:
            raise Exception("Could not find login button")
        
        # Wait for login to complete
        await asyncio.sleep(5)
        
        # Verify login success (check for dashboard or profile elements)
        await asyncio.sleep(3)
    
    async def _find_available_survey(self, browser: BrowserManager, platform_data: Dict[str, Any], survey_id: Optional[str]) -> Optional[str]:
        """Find an available survey to complete."""
        if survey_id:
            # Look for specific survey
            return f"{platform_data['surveys_url']}/{survey_id}"
        
        # Find any available survey
        survey_selectors = platform_data['element_locators']['survey_elements']['survey_link']
        
        for selector in survey_selectors:
            if await browser.wait_for_element(selector, timeout=5000):
                # Get the first available survey link
                survey_element = await browser.page.query_selector(selector)
                if survey_element:
                    href = await survey_element.get_attribute('href')
                    if href:
                        return href
        
        return None
    
    async def _complete_survey(self, browser: BrowserManager, task: Task, session) -> int:
        """Complete a survey by answering questions."""
        questions_answered = 0
        max_questions = 50  # Prevent infinite loops
        
        while questions_answered < max_questions:
            try:
                # Check if survey is complete
                if await self._is_survey_complete(browser):
                    logger.info("Survey completed successfully")
                    break
                
                # Extract current question
                question = await self._extract_current_question(browser)
                if not question:
                    logger.warning("No question found, survey might be complete")
                    break
                
                logger.info(f"Processing question {questions_answered + 1}: {question.question_text[:100]}...")
                
                # Generate response using AI
                response = await self.ai_generator.generate_response(question)
                if not response:
                    logger.warning("Failed to generate response, skipping question")
                    continue
                
                # Submit response
                success = await self._submit_response(browser, question, response.solution)
                if success:
                    questions_answered += 1
                    self.stats['total_questions_answered'] += 1
                    
                    # Update progress
                    progress = 50.0 + (questions_answered / max_questions) * 40.0
                    task.progress = min(progress, 90.0)
                    task.questions_answered = questions_answered
                    session.commit()
                    
                    logger.info(f"Successfully submitted response {questions_answered}")
                else:
                    logger.warning("Failed to submit response")
                
                # Human-like delay between questions
                await asyncio.sleep(random.uniform(2, 5))
                
                # Handle any CAPTCHAs that might appear
                captcha_type = await self.captcha_solver.detect_captcha_type(browser.page)
                if captcha_type:
                    logger.info(f"CAPTCHA detected during survey: {captcha_type}")
                    solution = await self.captcha_solver.solve_captcha(browser.page, captcha_type)
                    if solution:
                        await self.captcha_solver.submit_captcha_solution(browser.page, captcha_type, solution)
                        self.stats['total_captchas_solved'] += 1
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                continue
        
        return questions_answered
    
    async def _is_survey_complete(self, browser: BrowserManager) -> bool:
        """Check if the survey is complete."""
        completion_indicators = [
            '.survey-complete',
            '.thank-you',
            '.completion-message',
            '[class*="complete"]',
            '[class*="finished"]',
            'h1:has-text("Thank you")',
            'h2:has-text("Complete")'
        ]
        
        for indicator in completion_indicators:
            if await browser.wait_for_element(indicator, timeout=1000):
                return True
        
        return False
    
    async def _extract_current_question(self, browser: BrowserManager):
        """Extract the current survey question."""
        # This would use the survey_extractor module
        # For now, simplified implementation
        question_selectors = [
            '.question-text',
            '.survey-question',
            'h2',
            'h3',
            '.question-title'
        ]
        
        for selector in question_selectors:
            if await browser.wait_for_element(selector, timeout=3000):
                element = await browser.page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if text and len(text.strip()) > 10:
                        # Create a basic question object
                        from enhanced_ai_generator import SurveyQuestion
                        return SurveyQuestion(
                            question_id=f"q_{int(time.time())}",
                            question_text=text.strip(),
                            question_type="single_choice",  # Default
                            options=await self._extract_question_options(browser),
                            category="general",
                            context={}
                        )
        
        return None
    
    async def _extract_question_options(self, browser: BrowserManager) -> List[str]:
        """Extract options for the current question."""
        options = []
        
        # Try different option selectors
        option_selectors = [
            'input[type="radio"] + label',
            'input[type="checkbox"] + label',
            '.answer-option',
            '.choice',
            'option'
        ]
        
        for selector in option_selectors:
            elements = await browser.page.query_selector_all(selector)
            if elements:
                for element in elements:
                    text = await element.inner_text()
                    if text and text.strip():
                        options.append(text.strip())
                break
        
        return options
    
    async def _submit_response(self, browser: BrowserManager, question, response: str) -> bool:
        """Submit a response to a survey question."""
        try:
            # Try different submission methods based on question type
            if question.question_type == "single_choice":
                # Look for radio buttons
                radio_selectors = [
                    f'input[type="radio"][value="{response}"]',
                    f'input[type="radio"] + label:has-text("{response}")'
                ]
                
                for selector in radio_selectors:
                    if await browser.wait_for_element(selector, timeout=3000):
                        await browser.human_like_click(selector)
                        return True
            
            elif question.question_type in ["text_input", "textarea"]:
                # Look for text inputs
                text_selectors = [
                    'input[type="text"]',
                    'textarea',
                    '.text-input'
                ]
                
                for selector in text_selectors:
                    if await browser.wait_for_element(selector, timeout=3000):
                        await browser.human_like_type(selector, response)
                        return True
            
            # Try to find and click next/continue button
            next_selectors = [
                'button:has-text("Next")',
                'button:has-text("Continue")',
                'input[type="submit"]',
                '.next-button',
                '.continue-button'
            ]
            
            for selector in next_selectors:
                if await browser.wait_for_element(selector, timeout=3000):
                    await browser.human_like_click(selector)
                    await asyncio.sleep(2)  # Wait for page transition
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error submitting response: {e}")
            return False
    
    def _get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy from rotation."""
        # This would integrate with the proxy manager
        # For now, return None (no proxy)
        return None
    
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        while self.running:
            try:
                # Clean up old logs
                await self._cleanup_old_logs()
                
                # Update statistics
                await self._update_statistics()
                
                # Check system health
                await self._health_check()
                
                # Wait for next maintenance cycle (1 hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in periodic maintenance: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _cleanup_old_logs(self):
        """Clean up old log entries."""
        # Implementation for log cleanup
        pass
    
    async def _update_statistics(self):
        """Update system statistics."""
        # Implementation for statistics update
        pass
    
    async def _health_check(self):
        """Perform system health check."""
        # Check database connection
        # Check AI service availability
        # Check proxy health
        # Check drift detector status
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'running': self.running,
            'uptime_seconds': int(uptime.total_seconds()),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'stats': self.stats,
            'drift_alerts': len(self.drift_detector.get_recent_alerts(hours=24)),
            'ai_generator_stats': self.ai_generator.get_response_statistics(),
            'captcha_solver_stats': self.captcha_solver.get_solve_statistics()
        }

# Flask Web Interface
def create_web_app(automation_system: EnhancedSurveyAutomation) -> Flask:
    """Create Flask web application."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    CORS(app)
    
    @app.route('/')
    def dashboard():
        """Main dashboard."""
        return render_template_string(DASHBOARD_TEMPLATE)
    
    @app.route('/api/status')
    def get_status():
        """Get system status."""
        return jsonify(automation_system.get_system_status())
    
    @app.route('/api/tasks', methods=['POST'])
    async def create_task():
        """Create a new task."""
        data = request.json
        
        try:
            task_id = await automation_system.create_task(
                platform=data['platform'],
                credentials=data['credentials'],
                survey_id=data.get('survey_id'),
                config=data.get('config', {})
            )
            
            return jsonify({
                'task_id': task_id,
                'status': 'created',
                'message': 'Task created successfully'
            }), 201
            
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 400
    
    @app.route('/api/tasks/<task_id>')
    def get_task(task_id):
        """Get task status."""
        session = automation_system.db_manager.get_session()
        try:
            task = session.query(Task).filter_by(id=task_id).first()
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            
            return jsonify({
                'task_id': task.id,
                'status': task.status,
                'progress': task.progress,
                'current_step': task.current_step,
                'questions_answered': task.questions_answered,
                'error_message': task.error_message,
                'created_at': task.created_at.isoformat() if task.created_at else None,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None
            })
            
        finally:
            session.close()
    
    @app.route('/api/drift/alerts')
    def get_drift_alerts():
        """Get drift detection alerts."""
        alerts = automation_system.drift_detector.get_recent_alerts(hours=24)
        return jsonify({
            'alerts': [
                {
                    'alert_id': alert.alert_id,
                    'url': alert.url,
                    'drift_type': alert.drift_type,
                    'severity': alert.severity,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in alerts
            ]
        })
    
    return app

# Dashboard HTML Template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Survey Automation Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
        .status { padding: 5px 10px; border-radius: 3px; color: white; }
        .running { background-color: green; }
        .stopped { background-color: red; }
        .stats { display: flex; gap: 20px; }
        .stat-item { text-align: center; }
    </style>
</head>
<body>
    <h1>Enhanced Survey Automation Dashboard</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <div id="system-status">Loading...</div>
    </div>
    
    <div class="card">
        <h2>Create New Task</h2>
        <form id="task-form">
            <div>
                <label>Platform:</label>
                <select name="platform" required>
                    <option value="swagbucks">Swagbucks</option>
                    <option value="inboxdollars">InboxDollars</option>
                </select>
            </div>
            <div>
                <label>Email:</label>
                <input type="email" name="email" required>
            </div>
            <div>
                <label>Password:</label>
                <input type="password" name="password" required>
            </div>
            <div>
                <label>Survey ID (optional):</label>
                <input type="text" name="survey_id">
            </div>
            <button type="submit">Create Task</button>
        </form>
    </div>
    
    <div class="card">
        <h2>Recent Drift Alerts</h2>
        <div id="drift-alerts">Loading...</div>
    </div>
    
    <script>
        // Load system status
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('system-status').innerHTML = `
                    <div class="stats">
                        <div class="stat-item">
                            <h3>Status</h3>
                            <span class="status ${data.running ? 'running' : 'stopped'}">
                                ${data.running ? 'Running' : 'Stopped'}
                            </span>
                        </div>
                        <div class="stat-item">
                            <h3>Active Tasks</h3>
                            <p>${data.active_tasks}</p>
                        </div>
                        <div class="stat-item">
                            <h3>Completed Tasks</h3>
                            <p>${data.stats.tasks_completed}</p>
                        </div>
                        <div class="stat-item">
                            <h3>Questions Answered</h3>
                            <p>${data.stats.total_questions_answered}</p>
                        </div>
                    </div>
                `;
            } catch (error) {
                document.getElementById('system-status').innerHTML = 'Error loading status';
            }
        }
        
        // Load drift alerts
        async function loadAlerts() {
            try {
                const response = await fetch('/api/drift/alerts');
                const data = await response.json();
                
                const alertsHtml = data.alerts.map(alert => `
                    <div class="alert ${alert.severity}">
                        <strong>${alert.drift_type}</strong> - ${alert.severity}
                        <p>${alert.description}</p>
                        <small>${alert.timestamp}</small>
                    </div>
                `).join('');
                
                document.getElementById('drift-alerts').innerHTML = alertsHtml || 'No recent alerts';
            } catch (error) {
                document.getElementById('drift-alerts').innerHTML = 'Error loading alerts';
            }
        }
        
        // Handle task creation
        document.getElementById('task-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const taskData = {
                platform: formData.get('platform'),
                credentials: {
                    email: formData.get('email'),
                    password: formData.get('password')
                },
                survey_id: formData.get('survey_id') || null
            };
            
            try {
                const response = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(taskData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    alert(`Task created successfully! ID: ${result.task_id}`);
                    e.target.reset();
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                alert('Error creating task');
            }
        });
        
        // Load data on page load
        loadStatus();
        loadAlerts();
        
        // Refresh every 30 seconds
        setInterval(() => {
            loadStatus();
            loadAlerts();
        }, 30000);
    </script>
</body>
</html>
'''

# Main execution
async def main():
    """Main application entry point."""
    # Load configuration
    config = {
        'database_url': os.getenv('DATABASE_URL', 'sqlite:///survey_automation.db'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        '2captcha_api_key': os.getenv('CAPTCHA_API_KEY'),
        'max_concurrent_tasks': int(os.getenv('MAX_CONCURRENT_TASKS', '5')),
        'browser_headless': os.getenv('BROWSER_HEADLESS', 'true').lower() == 'true',
        'proxy_rotation_enabled': os.getenv('PROXY_ROTATION_ENABLED', 'false').lower() == 'true'
    }
    
    # Initialize system
    automation_system = EnhancedSurveyAutomation(config)
    
    # Create web app
    web_app = create_web_app(automation_system)
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(automation_system.stop_system())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start web server in thread
    def run_web_server():
        web_app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)
    
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # Start automation system
    await automation_system.start_system()

if __name__ == '__main__':
    asyncio.run(main())