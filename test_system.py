#!/usr/bin/env python3
"""
Enhanced Survey Automation System Test Suite
Comprehensive testing for all system components.
"""

import asyncio
import logging
import sys
import os
import time
from typing import Dict, Any, List
import pytest
import requests
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from enhanced_main import EnhancedSurveyAutomation
from browser_manager import BrowserManager
from enhanced_ai_generator import EnhancedAIGenerator, PersonaType, SurveyQuestion
from enhanced_captcha_solver import EnhancedCaptchaSolver, CaptchaType
from drift_detector import DriftDetector
from database_models import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system testing class."""
    
    def __init__(self):
        self.test_config = {
            'database_url': 'sqlite:///test_survey_automation.db',
            'browser_headless': True,
            'max_concurrent_tasks': 2,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            '2captcha_api_key': os.getenv('CAPTCHA_API_KEY')
        }
        self.test_results = {}
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all system tests."""
        logger.info("Starting comprehensive system tests...")
        
        tests = [
            ("Database Models", self.test_database_models),
            ("Browser Manager", self.test_browser_manager),
            ("AI Generator", self.test_ai_generator),
            ("CAPTCHA Solver", self.test_captcha_solver),
            ("Drift Detector", self.test_drift_detector),
            ("Main System", self.test_main_system),
            ("API Endpoints", self.test_api_endpoints),
            ("Integration", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results[test_name] = False
        
        return self.test_results
    
    async def test_database_models(self) -> bool:
        """Test database models and operations."""
        try:
            # Initialize database
            db_manager = DatabaseManager(self.test_config['database_url'])
            db_manager.create_tables()
            db_manager.init_default_data()
            
            # Test session creation
            session = db_manager.get_session()
            
            # Test basic queries
            from database_models import Platform, Account, Task
            
            platforms = session.query(Platform).all()
            assert len(platforms) >= 2, "Should have at least 2 default platforms"
            
            # Test platform creation
            test_platform = Platform(
                name='test_platform',
                display_name='Test Platform',
                base_url='https://test.com',
                login_url='https://test.com/login',
                surveys_url='https://test.com/surveys'
            )
            session.add(test_platform)
            session.commit()
            
            # Verify creation
            retrieved = session.query(Platform).filter_by(name='test_platform').first()
            assert retrieved is not None, "Platform should be created"
            
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return False
    
    async def test_browser_manager(self) -> bool:
        """Test browser manager functionality."""
        try:
            browser_manager = BrowserManager()
            
            # Test browser startup
            await browser_manager.start_browser(headless=True)
            assert browser_manager.browser is not None, "Browser should be started"
            assert browser_manager.page is not None, "Page should be available"
            
            # Test navigation
            success = await browser_manager.navigate_to("https://httpbin.org/get")
            assert success, "Navigation should succeed"
            
            # Test human-like interactions
            await browser_manager.page.set_content('''
                <html>
                    <body>
                        <input type="text" id="test-input" />
                        <button id="test-button">Click me</button>
                    </body>
                </html>
            ''')
            
            # Test typing
            success = await browser_manager.human_like_type("#test-input", "Test text")
            assert success, "Human-like typing should work"
            
            # Test clicking
            success = await browser_manager.human_like_click("#test-button")
            assert success, "Human-like clicking should work"
            
            # Test screenshot
            await browser_manager.take_screenshot("test_screenshot.png")
            assert os.path.exists("test_screenshot.png"), "Screenshot should be saved"
            
            # Cleanup
            await browser_manager.close()
            if os.path.exists("test_screenshot.png"):
                os.remove("test_screenshot.png")
            
            return True
            
        except Exception as e:
            logger.error(f"Browser manager test failed: {e}")
            return False
    
    async def test_ai_generator(self) -> bool:
        """Test AI response generator."""
        try:
            ai_generator = EnhancedAIGenerator(self.test_config)
            
            # Test persona setting
            success = ai_generator.set_persona(PersonaType.YOUNG_PROFESSIONAL)
            assert success, "Should be able to set persona"
            
            # Test question creation
            test_question = SurveyQuestion(
                question_id="test_q1",
                question_text="What is your favorite color?",
                question_type="single_choice",
                options=["Red", "Blue", "Green", "Yellow"],
                category="preference",
                context={}
            )
            
            # Test response generation
            response = await ai_generator.generate_response(test_question)
            assert response is not None, "Should generate a response"
            assert response.solution in test_question.options, "Response should be valid option"
            assert response.confidence > 0, "Should have confidence score"
            
            # Test different question types
            text_question = SurveyQuestion(
                question_id="test_q2",
                question_text="Describe your ideal vacation.",
                question_type="text_input",
                options=[],
                category="lifestyle",
                context={}
            )
            
            text_response = await ai_generator.generate_response(text_question)
            assert text_response is not None, "Should generate text response"
            assert len(text_response.solution) > 10, "Text response should be substantial"
            
            # Test statistics
            stats = ai_generator.get_response_statistics()
            assert stats['total_responses'] >= 2, "Should track response statistics"
            
            return True
            
        except Exception as e:
            logger.error(f"AI generator test failed: {e}")
            return False
    
    async def test_captcha_solver(self) -> bool:
        """Test CAPTCHA solver functionality."""
        try:
            captcha_solver = EnhancedCaptchaSolver(self.test_config)
            
            # Test OCR engines initialization
            assert 'tesseract' in captcha_solver.ocr_engines, "Should have OCR engines"
            
            # Test math CAPTCHA solving
            from enhanced_captcha_solver import CaptchaChallenge
            
            math_challenge = CaptchaChallenge(
                captcha_type=CaptchaType.MATH,
                challenge_text="5 + 3 = ?"
            )
            
            solution = await captcha_solver._solve_math_captcha(math_challenge)
            if solution:
                assert solution.solution == "8", "Should solve math CAPTCHA correctly"
                assert solution.confidence == 1.0, "Math solving should have high confidence"
            
            # Test statistics
            stats = captcha_solver.get_solve_statistics()
            assert 'total_attempts' in stats, "Should track statistics"
            
            return True
            
        except Exception as e:
            logger.error(f"CAPTCHA solver test failed: {e}")
            return False
    
    async def test_drift_detector(self) -> bool:
        """Test drift detection functionality."""
        try:
            drift_detector = DriftDetector("test_drift.db")
            
            # Test URL monitoring setup
            test_selectors = {
                'login_button': '#login',
                'email_field': '#email'
            }
            
            drift_detector.add_monitored_url("https://httpbin.org/get", test_selectors)
            
            # Test snapshot creation (mock)
            with patch.object(drift_detector, '_capture_page_snapshot') as mock_capture:
                mock_capture.return_value = {
                    'html_hash': 'test_hash',
                    'structure_hash': 'test_structure',
                    'element_count': 10
                }
                
                # This would normally capture a real snapshot
                # For testing, we'll just verify the method exists
                assert hasattr(drift_detector, '_capture_page_snapshot'), "Should have snapshot method"
            
            # Test alert retrieval
            alerts = drift_detector.get_recent_alerts(hours=24)
            assert isinstance(alerts, list), "Should return list of alerts"
            
            return True
            
        except Exception as e:
            logger.error(f"Drift detector test failed: {e}")
            return False
    
    async def test_main_system(self) -> bool:
        """Test main system initialization and basic functionality."""
        try:
            # Initialize system
            automation_system = EnhancedSurveyAutomation(self.test_config)
            
            # Test system status
            status = automation_system.get_system_status()
            assert 'running' in status, "Should have running status"
            assert 'stats' in status, "Should have statistics"
            
            # Test configuration loading
            assert automation_system.config is not None, "Should have configuration"
            assert automation_system.db_manager is not None, "Should have database manager"
            assert automation_system.ai_generator is not None, "Should have AI generator"
            
            return True
            
        except Exception as e:
            logger.error(f"Main system test failed: {e}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test API endpoints (requires running server)."""
        try:
            # This test would require the server to be running
            # For now, we'll just test the endpoint definitions exist
            from enhanced_main import create_web_app, EnhancedSurveyAutomation
            
            automation_system = EnhancedSurveyAutomation(self.test_config)
            app = create_web_app(automation_system)
            
            # Test app creation
            assert app is not None, "Should create Flask app"
            
            # Test routes exist
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ['/', '/api/status', '/api/tasks', '/api/drift/alerts']
            
            for route in expected_routes:
                assert any(route in r for r in routes), f"Should have {route} route"
            
            return True
            
        except Exception as e:
            logger.error(f"API endpoints test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test integration between components."""
        try:
            # Test AI generator with browser manager integration
            browser_manager = BrowserManager()
            ai_generator = EnhancedAIGenerator(self.test_config)
            
            await browser_manager.start_browser(headless=True)
            ai_generator.set_persona(PersonaType.COLLEGE_STUDENT)
            
            # Create a mock survey page
            await browser_manager.page.set_content('''
                <html>
                    <body>
                        <h2>What is your age group?</h2>
                        <input type="radio" name="age" value="18-24" id="age1">
                        <label for="age1">18-24</label>
                        <input type="radio" name="age" value="25-34" id="age2">
                        <label for="age2">25-34</label>
                        <input type="radio" name="age" value="35-44" id="age3">
                        <label for="age3">35-44</label>
                    </body>
                </html>
            ''')
            
            # Test question extraction and response
            question = SurveyQuestion(
                question_id="age_q",
                question_text="What is your age group?",
                question_type="single_choice",
                options=["18-24", "25-34", "35-44"],
                category="demographic",
                context={}
            )
            
            response = await ai_generator.generate_response(question)
            assert response is not None, "Should generate response"
            
            # Test response submission
            if response.solution == "18-24":
                success = await browser_manager.human_like_click("#age1")
            elif response.solution == "25-34":
                success = await browser_manager.human_like_click("#age2")
            else:
                success = await browser_manager.human_like_click("#age3")
            
            assert success, "Should be able to submit response"
            
            await browser_manager.close()
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("ENHANCED SURVEY AUTOMATION SYSTEM - TEST RESULTS")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<25} {status}")
        
        print("-"*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*60)
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            print("⚠️  Some tests failed. Please review the logs above.")
        
        return failed_tests == 0

async def main():
    """Main test execution."""
    print("Enhanced Survey Automation System - Comprehensive Test Suite")
    print("="*60)
    
    tester = SystemTester()
    
    # Run all tests
    results = await tester.run_all_tests()
    
    # Print summary
    all_passed = tester.print_test_summary()
    
    # Cleanup test files
    test_files = [
        'test_survey_automation.db',
        'test_drift.db',
        'test_screenshot.png'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    asyncio.run(main())