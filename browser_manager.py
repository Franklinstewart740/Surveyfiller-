"""
Enhanced Browser Manager with Advanced Anti-Detection Features
Handles browser automation with comprehensive stealth capabilities and proxy rotation.
"""

import asyncio
import random
import time
import json
import os
import math
from typing import Optional, Dict, Any, List, Tuple
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import stealth
import logging
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class MouseMovement:
    """Represents a natural mouse movement."""
    x: int
    y: int
    duration: int

@dataclass
class TypingPattern:
    """Represents natural typing patterns."""
    text: str
    delays: List[float]
    mistakes: List[Tuple[int, str]]  # (position, wrong_char)

class BrowserManager:
    """Enhanced browser manager with advanced anti-detection capabilities."""
    
    def __init__(self, proxy_config: Optional[Dict[str, str]] = None):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.proxy_config = proxy_config
        self.user_agents = self._load_user_agents()
        self.fingerprints = self._load_fingerprints()
        self.session_data = {}
        self.mouse_position = {"x": 0, "y": 0}
    
    def _load_user_agents(self) -> List[str]:
        """Load realistic user agents."""
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
    
    def _load_fingerprints(self) -> List[Dict[str, Any]]:
        """Load browser fingerprints for different devices."""
        return [
            {
                "viewport": {"width": 1920, "height": 1080},
                "screen": {"width": 1920, "height": 1080},
                "device_scale_factor": 1,
                "is_mobile": False,
                "has_touch": False,
                "timezone": "America/New_York"
            },
            {
                "viewport": {"width": 1366, "height": 768},
                "screen": {"width": 1366, "height": 768},
                "device_scale_factor": 1,
                "is_mobile": False,
                "has_touch": False,
                "timezone": "America/Los_Angeles"
            },
            {
                "viewport": {"width": 1440, "height": 900},
                "screen": {"width": 1440, "height": 900},
                "device_scale_factor": 2,
                "is_mobile": False,
                "has_touch": False,
                "timezone": "America/Chicago"
            }
        ]
        
    async def start_browser(self, headless: bool = True, browser_type: str = "chromium") -> None:
        """Start browser with enhanced anti-detection features."""
        try:
            self.playwright = await async_playwright().start()
            
            # Select random fingerprint
            fingerprint = random.choice(self.fingerprints)
            user_agent = random.choice(self.user_agents)
            
            # Enhanced browser launch options with anti-detection
            launch_options = {
                "headless": headless,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-accelerated-2d-canvas",
                    "--no-first-run",
                    "--no-zygote",
                    "--disable-gpu",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",
                    "--disable-javascript-harmony-shipping",
                    "--disable-client-side-phishing-detection",
                    "--disable-sync",
                    "--disable-default-apps",
                    "--hide-scrollbars",
                    "--mute-audio",
                    "--no-default-browser-check",
                    "--no-first-run",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection"
                ]
            }
            
            # Add proxy if configured
            if self.proxy_config:
                launch_options["proxy"] = self.proxy_config
            
            # Launch browser
            if browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(**launch_options)
            elif browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(**launch_options)
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")
            
            # Enhanced context options with fingerprint
            context_options = {
                "viewport": fingerprint["viewport"],
                "user_agent": user_agent,
                "locale": "en-US",
                "timezone_id": fingerprint["timezone"],
                "permissions": ["geolocation", "notifications"],
                "geolocation": {"latitude": 40.7128 + random.uniform(-0.1, 0.1), 
                              "longitude": -74.0060 + random.uniform(-0.1, 0.1)},
                "device_scale_factor": fingerprint["device_scale_factor"],
                "is_mobile": fingerprint["is_mobile"],
                "has_touch": fingerprint["has_touch"],
                "extra_http_headers": {
                    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0"
                }
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Create page and apply enhanced stealth
            self.page = await self.context.new_page()
            await stealth(self.page)
            
            # Enhanced navigator properties override
            await self.page.add_init_script(f"""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {{
                    get: () => undefined,
                }});
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {{
                    get: () => [
                        {{name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'}},
                        {{name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'}},
                        {{name: 'Native Client', filename: 'internal-nacl-plugin'}}
                    ],
                }});
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {{
                    get: () => ['en-US', 'en', 'es'],
                }});
                
                // Override chrome object
                window.chrome = {{
                    runtime: {{}},
                    loadTimes: function() {{
                        return {{
                            requestTime: Date.now() / 1000 - Math.random() * 2,
                            startLoadTime: Date.now() / 1000 - Math.random() * 2,
                            commitLoadTime: Date.now() / 1000 - Math.random() * 1,
                            finishDocumentLoadTime: Date.now() / 1000 - Math.random() * 0.5,
                            finishLoadTime: Date.now() / 1000,
                            firstPaintTime: Date.now() / 1000 - Math.random() * 0.5,
                            firstPaintAfterLoadTime: 0,
                            navigationType: 'Other',
                            wasFetchedViaSpdy: false,
                            wasNpnNegotiated: false,
                            npnNegotiatedProtocol: 'unknown',
                            wasAlternateProtocolAvailable: false,
                            connectionInfo: 'http/1.1'
                        }};
                    }},
                    csi: function() {{
                        return {{}};
                    }}
                }};
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({{ state: Notification.permission }}) :
                        originalQuery(parameters)
                );
                
                // Override screen properties
                Object.defineProperty(screen, 'availWidth', {{
                    get: () => {fingerprint['screen']['width']}
                }});
                Object.defineProperty(screen, 'availHeight', {{
                    get: () => {fingerprint['screen']['height']}
                }});
                
                // Add realistic timing
                const originalSetTimeout = window.setTimeout;
                window.setTimeout = function(callback, delay) {{
                    const jitter = Math.random() * 10 - 5; // ±5ms jitter
                    return originalSetTimeout(callback, delay + jitter);
                }};
            """)
            
            # Set up request interception for additional stealth
            await self.page.route("**/*", self._handle_request)
            
            logger.info(f"Browser started with fingerprint: {fingerprint['viewport']}")
            
        except Exception as e:
            logger.error(f"Error starting browser: {e}")
            raise
    
    async def _handle_request(self, route, request):
        """Handle requests for additional stealth."""
        # Add random delays to requests
        if random.random() < 0.1:  # 10% of requests get delayed
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Modify headers for stealth
        headers = dict(request.headers)
        
        # Remove automation headers
        headers.pop('sec-ch-ua-mobile', None)
        headers.pop('sec-ch-ua-platform', None)
        
        # Add realistic headers
        if 'referer' not in headers and request.url != self.page.url:
            headers['referer'] = self.page.url
        
        await route.continue_(headers=headers)
    
    async def human_like_click(self, selector: str, delay_range: Tuple[float, float] = (0.1, 0.3)) -> bool:
        """Perform a human-like click with natural mouse movement."""
        try:
            element = await self.page.wait_for_selector(selector, timeout=5000)
            if not element:
                return False
            
            # Get element position
            box = await element.bounding_box()
            if not box:
                return False
            
            # Calculate click position with some randomness
            click_x = box['x'] + box['width'] / 2 + random.uniform(-5, 5)
            click_y = box['y'] + box['height'] / 2 + random.uniform(-5, 5)
            
            # Move mouse naturally to the element
            await self._move_mouse_naturally(click_x, click_y)
            
            # Random delay before click
            await asyncio.sleep(random.uniform(*delay_range))
            
            # Perform click
            await self.page.mouse.click(click_x, click_y)
            
            # Small delay after click
            await asyncio.sleep(random.uniform(0.05, 0.15))
            
            return True
            
        except Exception as e:
            logger.error(f"Error in human-like click: {e}")
            return False
    
    async def _move_mouse_naturally(self, target_x: float, target_y: float):
        """Move mouse in a natural, curved path."""
        current_x = self.mouse_position["x"]
        current_y = self.mouse_position["y"]
        
        # Calculate distance
        distance = ((target_x - current_x) ** 2 + (target_y - current_y) ** 2) ** 0.5
        
        # Number of steps based on distance
        steps = max(3, int(distance / 50))
        
        # Generate curved path
        for i in range(steps):
            progress = (i + 1) / steps
            
            # Add some curve to the movement
            curve_offset = 20 * math.sin(progress * math.pi) * random.uniform(0.5, 1.5)
            
            # Calculate intermediate position
            x = current_x + (target_x - current_x) * progress
            y = current_y + (target_y - current_y) * progress + curve_offset
            
            # Move mouse
            await self.page.mouse.move(x, y)
            
            # Variable delay between movements
            await asyncio.sleep(random.uniform(0.01, 0.03))
        
        # Update mouse position
        self.mouse_position = {"x": target_x, "y": target_y}
    
    async def human_like_type(self, selector: str, text: str, mistake_probability: float = 0.02) -> bool:
        """Type text with human-like patterns including occasional mistakes."""
        try:
            element = await self.page.wait_for_selector(selector, timeout=5000)
            if not element:
                return False
            
            # Click on the element first
            await self.human_like_click(selector)
            
            # Clear existing text
            await element.fill("")
            
            # Generate typing pattern
            typing_pattern = self._generate_typing_pattern(text, mistake_probability)
            
            # Type with pattern
            for char, delay in zip(typing_pattern.text, typing_pattern.delays):
                await self.page.keyboard.type(char)
                await asyncio.sleep(delay)
            
            # Handle mistakes
            for position, wrong_char in typing_pattern.mistakes:
                # Go to mistake position
                for _ in range(len(text) - position):
                    await self.page.keyboard.press('ArrowLeft')
                
                # Delete wrong character and type correct one
                await self.page.keyboard.press('Delete')
                await asyncio.sleep(random.uniform(0.1, 0.3))
                await self.page.keyboard.type(text[position])
                
                # Go back to end
                await self.page.keyboard.press('End')
            
            return True
            
        except Exception as e:
            logger.error(f"Error in human-like typing: {e}")
            return False
    
    def _generate_typing_pattern(self, text: str, mistake_probability: float) -> TypingPattern:
        """Generate realistic typing pattern with delays and mistakes."""
        delays = []
        mistakes = []
        
        for i, char in enumerate(text):
            # Base delay varies by character type
            if char == ' ':
                base_delay = random.uniform(0.1, 0.3)
            elif char in '.,!?;:':
                base_delay = random.uniform(0.2, 0.4)
            elif char.isupper():
                base_delay = random.uniform(0.08, 0.15)
            else:
                base_delay = random.uniform(0.05, 0.12)
            
            # Add randomness
            delay = base_delay * random.uniform(0.7, 1.3)
            delays.append(delay)
            
            # Occasional mistakes
            if random.random() < mistake_probability and char.isalpha():
                # Generate a nearby key mistake
                wrong_char = self._get_nearby_key(char)
                mistakes.append((i, wrong_char))
        
        return TypingPattern(text=text, delays=delays, mistakes=mistakes)
    
    def _get_nearby_key(self, char: str) -> str:
        """Get a nearby key for realistic typing mistakes."""
        keyboard_layout = {
            'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
            'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
            'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
            'p': ['o', 'l'], 'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x'],
            'd': ['s', 'e', 'f', 'c'], 'f': ['d', 'r', 'g', 'v'],
            'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'],
            'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l'], 'l': ['k', 'o', 'p'],
            'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'],
            'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k']
        }
        
        nearby_keys = keyboard_layout.get(char.lower(), [char])
        return random.choice(nearby_keys)
    
    async def wait_for_element(self, selector: str, timeout: int = 5000) -> bool:
        """Wait for an element to appear."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False
    
    async def navigate_to(self, url: str, wait_until: str = 'networkidle') -> bool:
        """Navigate to a URL with human-like behavior."""
        try:
            # Random delay before navigation
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Navigate
            await self.page.goto(url, wait_until=wait_until, timeout=30000)
            
            # Random delay after navigation
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Simulate reading the page
            await self._simulate_reading_behavior()
            
            return True
            
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return False
    
    async def _simulate_reading_behavior(self):
        """Simulate human reading behavior with random scrolling."""
        # Random small scrolls to simulate reading
        for _ in range(random.randint(1, 3)):
            scroll_amount = random.randint(100, 300)
            await self.page.mouse.wheel(0, scroll_amount)
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Scroll back to top
        await self.page.mouse.wheel(0, -1000)
        await asyncio.sleep(random.uniform(0.3, 0.8))
    
    async def close(self) -> None:
        """Close browser and cleanup resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("Browser closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    async def get_page_content(self) -> str:
        """Get the current page content."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        return await self.page.content()
    
    async def take_screenshot(self, path: str) -> None:
        """Take a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        await self.page.screenshot(path=path, full_page=True)
        logger.info(f"Screenshot saved to: {path}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

