#!/usr/bin/env python3
"""
Swagbucks Data Gathering Script
Comprehensive data collection for Swagbucks platform automation.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from browser_manager import BrowserManager
from swagbucks_data import SWAGBUCKS_DATA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwagbucksDataGatherer:
    """Comprehensive Swagbucks data gathering system."""
    
    def __init__(self):
        self.browser_manager = None
        self.collected_data = {
            'target_urls': [],
            'interaction_steps': [],
            'element_locators': {},
            'waiting_conditions': [],
            'ids_and_selectors': {},
            'data_points': [],
            'navigation_links': [],
            'dynamic_content_indicators': [],
            'error_blocking_signals': [],
            'form_fields_buttons': [],
            'timestamp': datetime.now().isoformat(),
            'platform': 'swagbucks'
        }
    
    async def gather_all_data(self) -> Dict[str, Any]:
        """Gather comprehensive Swagbucks data."""
        logger.info("Starting comprehensive Swagbucks data gathering...")
        
        try:
            # Initialize browser
            self.browser_manager = BrowserManager()
            await self.browser_manager.start_browser(headless=False)  # Use visible browser for data gathering
            
            # Gather data from different pages
            await self._gather_homepage_data()
            await self._gather_login_page_data()
            await self._gather_surveys_page_data()
            await self._gather_survey_example_data()
            await self._gather_profile_data()
            await self._gather_rewards_data()
            
            # Analyze dynamic content
            await self._analyze_dynamic_content()
            
            # Test anti-bot detection
            await self._test_anti_bot_measures()
            
            logger.info("Data gathering completed successfully")
            return self.collected_data
            
        except Exception as e:
            logger.error(f"Error during data gathering: {e}")
            raise
        finally:
            if self.browser_manager:
                await self.browser_manager.close()
    
    async def _gather_homepage_data(self):
        """Gather data from Swagbucks homepage."""
        logger.info("Gathering homepage data...")
        
        url = SWAGBUCKS_DATA['target_url']
        await self.browser_manager.navigate_to(url)
        
        # Add to target URLs
        self.collected_data['target_urls'].append({
            'url': url,
            'type': 'homepage',
            'description': 'Main Swagbucks homepage'
        })
        
        # Gather element locators
        homepage_elements = await self._extract_page_elements()
        self.collected_data['element_locators']['homepage'] = homepage_elements
        
        # Look for navigation links
        nav_links = await self._extract_navigation_links()
        self.collected_data['navigation_links'].extend(nav_links)
        
        # Check for dynamic content
        dynamic_indicators = await self._check_dynamic_content()
        self.collected_data['dynamic_content_indicators'].extend(dynamic_indicators)
    
    async def _gather_login_page_data(self):
        """Gather data from login page."""
        logger.info("Gathering login page data...")
        
        url = SWAGBUCKS_DATA['login_url']
        await self.browser_manager.navigate_to(url)
        
        self.collected_data['target_urls'].append({
            'url': url,
            'type': 'login',
            'description': 'Swagbucks login page'
        })
        
        # Extract login form elements
        login_elements = await self._extract_login_form_elements()
        self.collected_data['element_locators']['login_form'] = login_elements
        
        # Add interaction steps for login
        login_steps = [
            {
                'step': 1,
                'action': 'navigate',
                'target': url,
                'description': 'Navigate to login page'
            },
            {
                'step': 2,
                'action': 'fill',
                'target': login_elements.get('email_field', ['#email'])[0],
                'description': 'Fill email field'
            },
            {
                'step': 3,
                'action': 'fill',
                'target': login_elements.get('password_field', ['#password'])[0],
                'description': 'Fill password field'
            },
            {
                'step': 4,
                'action': 'click',
                'target': login_elements.get('login_button', ['#login-btn'])[0],
                'description': 'Click login button'
            }
        ]
        self.collected_data['interaction_steps'].extend(login_steps)
        
        # Check for CAPTCHA elements
        captcha_elements = await self._check_captcha_elements()
        if captcha_elements:
            self.collected_data['element_locators']['captcha'] = captcha_elements
    
    async def _gather_surveys_page_data(self):
        """Gather data from surveys page."""
        logger.info("Gathering surveys page data...")
        
        url = SWAGBUCKS_DATA['surveys_url']
        await self.browser_manager.navigate_to(url)
        
        self.collected_data['target_urls'].append({
            'url': url,
            'type': 'surveys',
            'description': 'Swagbucks surveys listing page'
        })
        
        # Extract survey elements
        survey_elements = await self._extract_survey_elements()
        self.collected_data['element_locators']['survey_elements'] = survey_elements
        
        # Extract data points from survey listings
        survey_data_points = await self._extract_survey_data_points()
        self.collected_data['data_points'].extend(survey_data_points)
        
        # Add survey interaction steps
        survey_steps = [
            {
                'step': 1,
                'action': 'navigate',
                'target': url,
                'description': 'Navigate to surveys page'
            },
            {
                'step': 2,
                'action': 'wait',
                'target': survey_elements.get('survey_list', ['.survey-list'])[0],
                'description': 'Wait for survey list to load'
            },
            {
                'step': 3,
                'action': 'click',
                'target': survey_elements.get('survey_link', ['.survey-item a'])[0],
                'description': 'Click on first available survey'
            }
        ]
        self.collected_data['interaction_steps'].extend(survey_steps)
    
    async def _gather_survey_example_data(self):
        """Gather data from an example survey."""
        logger.info("Gathering survey example data...")
        
        # Try to find and enter a survey
        survey_links = await self.browser_manager.page.query_selector_all('.survey-item a, .survey-link, [href*="survey"]')
        
        if survey_links:
            # Click on first survey link
            await survey_links[0].click()
            await asyncio.sleep(3)
            
            current_url = self.browser_manager.page.url
            self.collected_data['target_urls'].append({
                'url': current_url,
                'type': 'survey_example',
                'description': 'Example survey page'
            })
            
            # Extract survey question elements
            question_elements = await self._extract_question_elements()
            self.collected_data['element_locators']['survey_questions'] = question_elements
            
            # Extract form fields and buttons
            form_elements = await self._extract_form_elements()
            self.collected_data['form_fields_buttons'].extend(form_elements)
            
            # Add survey completion steps
            survey_completion_steps = [
                {
                    'step': 1,
                    'action': 'wait',
                    'target': question_elements.get('question_text', ['.question'])[0],
                    'description': 'Wait for question to load'
                },
                {
                    'step': 2,
                    'action': 'analyze',
                    'target': 'question_type',
                    'description': 'Analyze question type and options'
                },
                {
                    'step': 3,
                    'action': 'respond',
                    'target': 'appropriate_option',
                    'description': 'Select or input appropriate response'
                },
                {
                    'step': 4,
                    'action': 'click',
                    'target': question_elements.get('next_button', ['.next-btn', '#next'])[0],
                    'description': 'Click next/continue button'
                }
            ]
            self.collected_data['interaction_steps'].extend(survey_completion_steps)
    
    async def _gather_profile_data(self):
        """Gather data from profile/account pages."""
        logger.info("Gathering profile data...")
        
        profile_urls = [
            f"{SWAGBUCKS_DATA['target_url']}/profile",
            f"{SWAGBUCKS_DATA['target_url']}/account",
            f"{SWAGBUCKS_DATA['target_url']}/settings"
        ]
        
        for url in profile_urls:
            try:
                await self.browser_manager.navigate_to(url)
                await asyncio.sleep(2)
                
                if self.browser_manager.page.url != url:
                    continue  # Page redirected, might not exist
                
                self.collected_data['target_urls'].append({
                    'url': url,
                    'type': 'profile',
                    'description': f'Profile/account page: {url}'
                })
                
                # Extract profile elements
                profile_elements = await self._extract_page_elements()
                self.collected_data['element_locators'][f'profile_{url.split("/")[-1]}'] = profile_elements
                
            except Exception as e:
                logger.warning(f"Could not access {url}: {e}")
    
    async def _gather_rewards_data(self):
        """Gather data from rewards/redemption pages."""
        logger.info("Gathering rewards data...")
        
        rewards_url = f"{SWAGBUCKS_DATA['target_url']}/rewards"
        
        try:
            await self.browser_manager.navigate_to(rewards_url)
            await asyncio.sleep(2)
            
            self.collected_data['target_urls'].append({
                'url': rewards_url,
                'type': 'rewards',
                'description': 'Rewards/redemption page'
            })
            
            # Extract rewards elements
            rewards_elements = await self._extract_page_elements()
            self.collected_data['element_locators']['rewards'] = rewards_elements
            
        except Exception as e:
            logger.warning(f"Could not access rewards page: {e}")
    
    async def _extract_page_elements(self) -> Dict[str, List[str]]:
        """Extract all relevant elements from current page."""
        elements = {
            'buttons': [],
            'inputs': [],
            'links': [],
            'forms': [],
            'divs': [],
            'spans': [],
            'images': [],
            'iframes': []
        }
        
        # Extract buttons
        buttons = await self.browser_manager.page.query_selector_all('button, input[type="button"], input[type="submit"], .btn, [role="button"]')
        for button in buttons:
            selector = await self._get_element_selector(button)
            if selector:
                elements['buttons'].append(selector)
        
        # Extract inputs
        inputs = await self.browser_manager.page.query_selector_all('input, textarea, select')
        for input_elem in inputs:
            selector = await self._get_element_selector(input_elem)
            if selector:
                elements['inputs'].append(selector)
        
        # Extract links
        links = await self.browser_manager.page.query_selector_all('a[href]')
        for link in links:
            selector = await self._get_element_selector(link)
            if selector:
                elements['links'].append(selector)
        
        # Extract forms
        forms = await self.browser_manager.page.query_selector_all('form')
        for form in forms:
            selector = await self._get_element_selector(form)
            if selector:
                elements['forms'].append(selector)
        
        # Extract other important elements
        other_selectors = [
            ('divs', 'div[class*="survey"], div[class*="question"], div[class*="answer"]'),
            ('spans', 'span[class*="error"], span[class*="message"], span[class*="alert"]'),
            ('images', 'img[src*="captcha"], img[alt*="captcha"]'),
            ('iframes', 'iframe')
        ]
        
        for element_type, selector in other_selectors:
            elems = await self.browser_manager.page.query_selector_all(selector)
            for elem in elems:
                elem_selector = await self._get_element_selector(elem)
                if elem_selector:
                    elements[element_type].append(elem_selector)
        
        return elements
    
    async def _extract_login_form_elements(self) -> Dict[str, List[str]]:
        """Extract login form specific elements."""
        elements = {
            'email_field': [],
            'password_field': [],
            'login_button': [],
            'remember_me': [],
            'forgot_password': [],
            'signup_link': []
        }
        
        # Email field selectors
        email_selectors = [
            'input[type="email"]',
            'input[name*="email"]',
            'input[id*="email"]',
            'input[placeholder*="email"]',
            '#email',
            '#username',
            '.email-input'
        ]
        
        for selector in email_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['email_field'].append(selector)
        
        # Password field selectors
        password_selectors = [
            'input[type="password"]',
            'input[name*="password"]',
            'input[id*="password"]',
            '#password',
            '.password-input'
        ]
        
        for selector in password_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['password_field'].append(selector)
        
        # Login button selectors
        login_button_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Sign In")',
            'button:has-text("Log In")',
            'button:has-text("Login")',
            '.login-btn',
            '#login-button',
            '[data-testid="login-button"]'
        ]
        
        for selector in login_button_selectors:
            try:
                if await self.browser_manager.page.query_selector(selector):
                    elements['login_button'].append(selector)
            except:
                pass
        
        return elements
    
    async def _extract_survey_elements(self) -> Dict[str, List[str]]:
        """Extract survey-specific elements."""
        elements = {
            'survey_list': [],
            'survey_item': [],
            'survey_link': [],
            'survey_title': [],
            'survey_reward': [],
            'survey_time': [],
            'survey_status': []
        }
        
        # Survey list containers
        list_selectors = [
            '.survey-list',
            '.surveys-container',
            '#surveys',
            '[data-testid="survey-list"]',
            '.offers-list'
        ]
        
        for selector in list_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['survey_list'].append(selector)
        
        # Individual survey items
        item_selectors = [
            '.survey-item',
            '.survey-card',
            '.offer-item',
            '[data-testid="survey-item"]',
            '.survey-row'
        ]
        
        for selector in item_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['survey_item'].append(selector)
        
        # Survey links
        link_selectors = [
            '.survey-item a',
            '.survey-link',
            'a[href*="survey"]',
            'a[href*="offer"]',
            '.start-survey'
        ]
        
        for selector in link_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['survey_link'].append(selector)
        
        return elements
    
    async def _extract_question_elements(self) -> Dict[str, List[str]]:
        """Extract survey question elements."""
        elements = {
            'question_text': [],
            'question_title': [],
            'answer_options': [],
            'radio_buttons': [],
            'checkboxes': [],
            'text_inputs': [],
            'dropdowns': [],
            'next_button': [],
            'previous_button': [],
            'submit_button': []
        }
        
        # Question text selectors
        question_selectors = [
            '.question-text',
            '.question-title',
            'h2',
            'h3',
            '.survey-question',
            '[data-testid="question"]',
            '.question-content'
        ]
        
        for selector in question_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['question_text'].append(selector)
        
        # Answer option selectors
        option_selectors = [
            'input[type="radio"]',
            'input[type="checkbox"]',
            '.answer-option',
            '.choice',
            'label[for*="answer"]'
        ]
        
        for selector in option_selectors:
            if await self.browser_manager.page.query_selector(selector):
                elements['answer_options'].append(selector)
        
        # Navigation button selectors
        next_selectors = [
            'button:has-text("Next")',
            'button:has-text("Continue")',
            '.next-btn',
            '#next',
            '[data-testid="next-button"]'
        ]
        
        for selector in next_selectors:
            try:
                if await self.browser_manager.page.query_selector(selector):
                    elements['next_button'].append(selector)
            except:
                pass
        
        return elements
    
    async def _extract_form_elements(self) -> List[Dict[str, str]]:
        """Extract form fields and buttons."""
        form_elements = []
        
        # Get all form elements
        inputs = await self.browser_manager.page.query_selector_all('input, textarea, select, button')
        
        for element in inputs:
            try:
                tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                element_type = await element.get_attribute('type') or 'text'
                element_id = await element.get_attribute('id')
                element_name = await element.get_attribute('name')
                element_class = await element.get_attribute('class')
                
                form_elements.append({
                    'tag': tag_name,
                    'type': element_type,
                    'id': element_id,
                    'name': element_name,
                    'class': element_class,
                    'selector': await self._get_element_selector(element)
                })
            except:
                continue
        
        return form_elements
    
    async def _extract_navigation_links(self) -> List[Dict[str, str]]:
        """Extract navigation links."""
        nav_links = []
        
        # Get all navigation links
        links = await self.browser_manager.page.query_selector_all('nav a, .nav a, .navigation a, .menu a')
        
        for link in links:
            try:
                href = await link.get_attribute('href')
                text = await link.inner_text()
                
                if href and text:
                    nav_links.append({
                        'url': href,
                        'text': text.strip(),
                        'selector': await self._get_element_selector(link)
                    })
            except:
                continue
        
        return nav_links
    
    async def _extract_survey_data_points(self) -> List[Dict[str, Any]]:
        """Extract data points from survey listings."""
        data_points = []
        
        # Look for survey information
        survey_items = await self.browser_manager.page.query_selector_all('.survey-item, .survey-card, .offer-item')
        
        for item in survey_items:
            try:
                # Extract survey details
                title_elem = await item.query_selector('.title, .survey-title, h3, h4')
                reward_elem = await item.query_selector('.reward, .points, .sb, [class*="reward"]')
                time_elem = await item.query_selector('.time, .duration, [class*="time"]')
                
                title = await title_elem.inner_text() if title_elem else None
                reward = await reward_elem.inner_text() if reward_elem else None
                time = await time_elem.inner_text() if time_elem else None
                
                if title or reward or time:
                    data_points.append({
                        'type': 'survey_listing',
                        'title': title,
                        'reward': reward,
                        'time': time,
                        'selector': await self._get_element_selector(item)
                    })
            except:
                continue
        
        return data_points
    
    async def _check_dynamic_content(self) -> List[Dict[str, str]]:
        """Check for dynamic content indicators."""
        indicators = []
        
        # Look for loading indicators
        loading_selectors = [
            '.loading',
            '.spinner',
            '.loader',
            '[data-loading]',
            '.progress'
        ]
        
        for selector in loading_selectors:
            if await self.browser_manager.page.query_selector(selector):
                indicators.append({
                    'type': 'loading_indicator',
                    'selector': selector,
                    'description': 'Loading/spinner element'
                })
        
        # Look for AJAX containers
        ajax_selectors = [
            '[data-ajax]',
            '.ajax-content',
            '.dynamic-content',
            '[data-url]'
        ]
        
        for selector in ajax_selectors:
            if await self.browser_manager.page.query_selector(selector):
                indicators.append({
                    'type': 'ajax_container',
                    'selector': selector,
                    'description': 'AJAX/dynamic content container'
                })
        
        return indicators
    
    async def _check_captcha_elements(self) -> Dict[str, List[str]]:
        """Check for CAPTCHA elements."""
        captcha_elements = {
            'recaptcha': [],
            'hcaptcha': [],
            'image_captcha': [],
            'text_captcha': []
        }
        
        # reCAPTCHA
        recaptcha_selectors = [
            '.g-recaptcha',
            'iframe[src*="recaptcha"]',
            '[data-sitekey]'
        ]
        
        for selector in recaptcha_selectors:
            if await self.browser_manager.page.query_selector(selector):
                captcha_elements['recaptcha'].append(selector)
        
        # hCaptcha
        hcaptcha_selectors = [
            '.h-captcha',
            'iframe[src*="hcaptcha"]'
        ]
        
        for selector in hcaptcha_selectors:
            if await self.browser_manager.page.query_selector(selector):
                captcha_elements['hcaptcha'].append(selector)
        
        # Image CAPTCHA
        image_captcha_selectors = [
            'img[src*="captcha"]',
            'img[alt*="captcha"]',
            '.captcha-image'
        ]
        
        for selector in image_captcha_selectors:
            if await self.browser_manager.page.query_selector(selector):
                captcha_elements['image_captcha'].append(selector)
        
        return captcha_elements
    
    async def _analyze_dynamic_content(self):
        """Analyze dynamic content loading patterns."""
        logger.info("Analyzing dynamic content patterns...")
        
        # Check for JavaScript frameworks
        js_frameworks = await self.browser_manager.page.evaluate('''
            () => {
                const frameworks = [];
                if (window.React) frameworks.push('React');
                if (window.Vue) frameworks.push('Vue');
                if (window.angular) frameworks.push('Angular');
                if (window.jQuery) frameworks.push('jQuery');
                return frameworks;
            }
        ''')
        
        if js_frameworks:
            self.collected_data['dynamic_content_indicators'].append({
                'type': 'javascript_frameworks',
                'frameworks': js_frameworks,
                'description': 'Detected JavaScript frameworks'
            })
        
        # Check for AJAX requests
        await self.browser_manager.page.route('**/*', self._track_requests)
        
        # Wait and observe
        await asyncio.sleep(5)
    
    async def _track_requests(self, route, request):
        """Track AJAX requests."""
        if request.resource_type in ['xhr', 'fetch']:
            self.collected_data['dynamic_content_indicators'].append({
                'type': 'ajax_request',
                'url': request.url,
                'method': request.method,
                'description': f'AJAX request to {request.url}'
            })
        
        await route.continue_()
    
    async def _test_anti_bot_measures(self):
        """Test for anti-bot detection measures."""
        logger.info("Testing anti-bot measures...")
        
        # Check for common anti-bot scripts
        anti_bot_scripts = await self.browser_manager.page.evaluate('''
            () => {
                const scripts = Array.from(document.scripts);
                const antiBot = [];
                
                scripts.forEach(script => {
                    const src = script.src || '';
                    const content = script.innerHTML || '';
                    
                    if (src.includes('cloudflare') || content.includes('cloudflare')) {
                        antiBot.push('Cloudflare');
                    }
                    if (src.includes('recaptcha') || content.includes('recaptcha')) {
                        antiBot.push('reCAPTCHA');
                    }
                    if (content.includes('webdriver') || content.includes('automation')) {
                        antiBot.push('WebDriver Detection');
                    }
                });
                
                return [...new Set(antiBot)];
            }
        ''')
        
        if anti_bot_scripts:
            self.collected_data['error_blocking_signals'].extend([
                {
                    'type': 'anti_bot_script',
                    'detection': script,
                    'description': f'Detected {script} anti-bot measure'
                }
                for script in anti_bot_scripts
            ])
        
        # Check for rate limiting indicators
        rate_limit_indicators = [
            '.rate-limit',
            '.too-many-requests',
            '.blocked',
            '[class*="limit"]'
        ]
        
        for selector in rate_limit_indicators:
            if await self.browser_manager.page.query_selector(selector):
                self.collected_data['error_blocking_signals'].append({
                    'type': 'rate_limit_indicator',
                    'selector': selector,
                    'description': 'Rate limiting indicator'
                })
    
    async def _get_element_selector(self, element) -> Optional[str]:
        """Get the best selector for an element."""
        try:
            # Try ID first
            element_id = await element.get_attribute('id')
            if element_id:
                return f'#{element_id}'
            
            # Try name attribute
            name = await element.get_attribute('name')
            if name:
                tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                return f'{tag_name}[name="{name}"]'
            
            # Try class
            class_name = await element.get_attribute('class')
            if class_name:
                classes = class_name.split()
                if classes:
                    return f'.{classes[0]}'
            
            # Try data attributes
            data_testid = await element.get_attribute('data-testid')
            if data_testid:
                return f'[data-testid="{data_testid}"]'
            
            # Fallback to tag name
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            return tag_name
            
        except:
            return None
    
    def save_data_to_file(self, filename: str = None):
        """Save collected data to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"swagbucks_data_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return filename

async def main():
    """Main execution function."""
    print("Swagbucks Data Gathering Tool")
    print("=" * 50)
    
    gatherer = SwagbucksDataGatherer()
    
    try:
        # Gather all data
        data = await gatherer.gather_all_data()
        
        # Save to file
        filename = gatherer.save_data_to_file()
        
        # Print summary
        print("\nData Gathering Summary:")
        print(f"Target URLs: {len(data['target_urls'])}")
        print(f"Interaction Steps: {len(data['interaction_steps'])}")
        print(f"Element Locators: {len(data['element_locators'])}")
        print(f"Data Points: {len(data['data_points'])}")
        print(f"Navigation Links: {len(data['navigation_links'])}")
        print(f"Dynamic Content Indicators: {len(data['dynamic_content_indicators'])}")
        print(f"Error/Blocking Signals: {len(data['error_blocking_signals'])}")
        print(f"Form Fields/Buttons: {len(data['form_fields_buttons'])}")
        print(f"\nData saved to: {filename}")
        
        # Print some key findings
        print("\nKey Findings:")
        for url_data in data['target_urls']:
            print(f"- {url_data['type']}: {url_data['url']}")
        
        if data['error_blocking_signals']:
            print("\nAnti-Bot Measures Detected:")
            for signal in data['error_blocking_signals']:
                print(f"- {signal['type']}: {signal['description']}")
        
    except Exception as e:
        logger.error(f"Data gathering failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())