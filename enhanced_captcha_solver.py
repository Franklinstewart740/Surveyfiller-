"""
Enhanced CAPTCHA Solver
Comprehensive CAPTCHA solving system with multiple methods and fallbacks.
"""

import asyncio
import logging
import base64
import io
import time
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from playwright.async_api import Page
import requests

logger = logging.getLogger(__name__)

class CaptchaType(Enum):
    """Types of CAPTCHAs that can be solved."""
    TEXT_BASED = "text_based"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE_SELECTION = "image_selection"
    SLIDER = "slider"
    ROTATE = "rotate"
    PUZZLE = "puzzle"
    MATH = "math"
    AUDIO = "audio"

@dataclass
class CaptchaChallenge:
    """Represents a CAPTCHA challenge."""
    captcha_type: CaptchaType
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    site_key: Optional[str] = None
    page_url: Optional[str] = None
    challenge_text: Optional[str] = None
    audio_url: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class CaptchaSolution:
    """Represents a CAPTCHA solution."""
    solution: str
    confidence: float
    method_used: str
    solve_time: float
    additional_data: Optional[Dict[str, Any]] = None

class EnhancedCaptchaSolver:
    """Enhanced CAPTCHA solver with multiple solving methods."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.solving_services = self._initialize_services()
        self.ocr_engines = self._initialize_ocr_engines()
        self.solve_history = []
        
        # Performance tracking
        self.success_rates = {}
        self.average_solve_times = {}
        
    def _initialize_services(self) -> Dict[str, Dict[str, Any]]:
        """Initialize third-party solving services."""
        services = {}
        
        # 2captcha
        if self.config.get("2captcha_api_key"):
            services["2captcha"] = {
                "api_key": self.config["2captcha_api_key"],
                "base_url": "http://2captcha.com",
                "supported_types": [CaptchaType.TEXT_BASED, CaptchaType.RECAPTCHA_V2, 
                                  CaptchaType.RECAPTCHA_V3, CaptchaType.HCAPTCHA]
            }
        
        # AntiCaptcha
        if self.config.get("anticaptcha_api_key"):
            services["anticaptcha"] = {
                "api_key": self.config["anticaptcha_api_key"],
                "base_url": "https://api.anti-captcha.com",
                "supported_types": [CaptchaType.TEXT_BASED, CaptchaType.RECAPTCHA_V2,
                                  CaptchaType.HCAPTCHA, CaptchaType.IMAGE_SELECTION]
            }
        
        # CapMonster
        if self.config.get("capmonster_api_key"):
            services["capmonster"] = {
                "api_key": self.config["capmonster_api_key"],
                "base_url": "https://api.capmonster.cloud",
                "supported_types": [CaptchaType.TEXT_BASED, CaptchaType.RECAPTCHA_V2,
                                  CaptchaType.RECAPTCHA_V3, CaptchaType.HCAPTCHA]
            }
        
        return services
    
    def _initialize_ocr_engines(self) -> Dict[str, Any]:
        """Initialize OCR engines for text-based CAPTCHAs."""
        engines = {}
        
        # Tesseract OCR
        try:
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            engines["tesseract"] = {
                "available": True,
                "configs": [
                    "--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                    "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                    "--psm 6"
                ]
            }
        except Exception:
            engines["tesseract"] = {"available": False}
        
        return engines
    
    async def detect_captcha_type(self, page: Page) -> Optional[CaptchaType]:
        """Detect the type of CAPTCHA present on the page."""
        try:
            # Check for reCAPTCHA v2
            recaptcha_v2 = await page.query_selector('.g-recaptcha, iframe[src*="recaptcha"], [data-sitekey]')
            if recaptcha_v2:
                return CaptchaType.RECAPTCHA_V2
            
            # Check for reCAPTCHA v3
            recaptcha_v3_script = await page.query_selector('script[src*="recaptcha/releases/"]')
            if recaptcha_v3_script:
                return CaptchaType.RECAPTCHA_V3
            
            # Check for hCaptcha
            hcaptcha = await page.query_selector('.h-captcha, iframe[src*="hcaptcha"]')
            if hcaptcha:
                return CaptchaType.HCAPTCHA
            
            # Check for text-based CAPTCHA
            captcha_images = await page.query_selector_all('img[src*="captcha"], img[alt*="captcha"], .captcha img')
            if captcha_images:
                return CaptchaType.TEXT_BASED
            
            # Check for slider CAPTCHA
            slider = await page.query_selector('.slider-captcha, .slide-verify, [class*="slider"]')
            if slider:
                return CaptchaType.SLIDER
            
            # Check for image selection CAPTCHA
            image_grid = await page.query_selector('.captcha-grid, .image-selection, [class*="image-captcha"]')
            if image_grid:
                return CaptchaType.IMAGE_SELECTION
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting CAPTCHA type: {e}")
            return None
    
    async def solve_captcha(self, page: Page, captcha_type: CaptchaType = None) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA using the best available method."""
        start_time = time.time()
        
        try:
            # Auto-detect if type not provided
            if not captcha_type:
                captcha_type = await self.detect_captcha_type(page)
                if not captcha_type:
                    logger.warning("No CAPTCHA detected on page")
                    return None
            
            logger.info(f"Attempting to solve {captcha_type.value} CAPTCHA")
            
            # Extract CAPTCHA challenge
            challenge = await self._extract_captcha_challenge(page, captcha_type)
            if not challenge:
                logger.error("Failed to extract CAPTCHA challenge")
                return None
            
            # Try different solving methods in order of preference
            solution = None
            
            # Method 1: Local OCR for text-based CAPTCHAs
            if captcha_type == CaptchaType.TEXT_BASED and challenge.image_data:
                solution = await self._solve_with_ocr(challenge)
                if solution and solution.confidence > 0.7:
                    logger.info(f"Solved with OCR (confidence: {solution.confidence})")
                else:
                    solution = None
            
            # Method 2: Third-party services
            if not solution:
                solution = await self._solve_with_service(challenge)
                if solution:
                    logger.info(f"Solved with service: {solution.method_used}")
            
            # Method 3: Specialized solvers
            if not solution:
                if captcha_type == CaptchaType.SLIDER:
                    solution = await self._solve_slider_captcha(page, challenge)
                elif captcha_type == CaptchaType.MATH:
                    solution = await self._solve_math_captcha(challenge)
            
            # Method 4: Human-in-the-loop (if configured)
            if not solution and self.config.get("human_solver_enabled"):
                solution = await self._request_human_solve(challenge)
            
            if solution:
                solve_time = time.time() - start_time
                solution.solve_time = solve_time
                
                # Update statistics
                self._update_solve_statistics(captcha_type, solution.method_used, True, solve_time)
                
                # Store in history
                self.solve_history.append({
                    "timestamp": time.time(),
                    "captcha_type": captcha_type.value,
                    "method": solution.method_used,
                    "success": True,
                    "solve_time": solve_time,
                    "confidence": solution.confidence
                })
                
                return solution
            else:
                logger.error(f"Failed to solve {captcha_type.value} CAPTCHA")
                self._update_solve_statistics(captcha_type, "failed", False, time.time() - start_time)
                return None
                
        except Exception as e:
            logger.error(f"Error solving CAPTCHA: {e}")
            return None
    
    async def _extract_captcha_challenge(self, page: Page, captcha_type: CaptchaType) -> Optional[CaptchaChallenge]:
        """Extract CAPTCHA challenge data from the page."""
        try:
            challenge = CaptchaChallenge(captcha_type=captcha_type)
            
            if captcha_type == CaptchaType.TEXT_BASED:
                # Find CAPTCHA image
                captcha_img = await page.query_selector('img[src*="captcha"], img[alt*="captcha"], .captcha img')
                if captcha_img:
                    # Get image source
                    src = await captcha_img.get_attribute('src')
                    if src:
                        if src.startswith('data:image'):
                            # Base64 encoded image
                            image_data = base64.b64decode(src.split(',')[1])
                            challenge.image_data = image_data
                        else:
                            # URL to image
                            challenge.image_url = src
                            # Download image
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(src) as response:
                                        if response.status == 200:
                                            challenge.image_data = await response.read()
                            except Exception as e:
                                logger.error(f"Failed to download CAPTCHA image: {e}")
            
            elif captcha_type in [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3]:
                # Extract site key
                site_key_element = await page.query_selector('[data-sitekey]')
                if site_key_element:
                    challenge.site_key = await site_key_element.get_attribute('data-sitekey')
                challenge.page_url = page.url
            
            elif captcha_type == CaptchaType.HCAPTCHA:
                # Extract hCaptcha site key
                hcaptcha_element = await page.query_selector('.h-captcha[data-sitekey]')
                if hcaptcha_element:
                    challenge.site_key = await hcaptcha_element.get_attribute('data-sitekey')
                challenge.page_url = page.url
            
            elif captcha_type == CaptchaType.SLIDER:
                # Extract slider challenge image
                slider_img = await page.query_selector('.slider-captcha img, .slide-verify img')
                if slider_img:
                    src = await slider_img.get_attribute('src')
                    if src and src.startswith('data:image'):
                        challenge.image_data = base64.b64decode(src.split(',')[1])
            
            return challenge
            
        except Exception as e:
            logger.error(f"Error extracting CAPTCHA challenge: {e}")
            return None
    
    async def _solve_with_ocr(self, challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve text-based CAPTCHA using OCR."""
        if not challenge.image_data or not self.ocr_engines.get("tesseract", {}).get("available"):
            return None
        
        try:
            # Load image
            image = Image.open(io.BytesIO(challenge.image_data))
            
            # Preprocess image for better OCR
            processed_images = self._preprocess_captcha_image(image)
            
            best_result = None
            best_confidence = 0
            
            # Try different preprocessing and OCR configurations
            for processed_img in processed_images:
                for config in self.ocr_engines["tesseract"]["configs"]:
                    try:
                        # Convert PIL image to numpy array for OpenCV
                        img_array = np.array(processed_img)
                        
                        # OCR with confidence
                        data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
                        
                        # Extract text and calculate confidence
                        text_parts = []
                        confidences = []
                        
                        for i, word in enumerate(data['text']):
                            if word.strip():
                                text_parts.append(word)
                                confidences.append(int(data['conf'][i]))
                        
                        if text_parts and confidences:
                            result_text = ''.join(text_parts)
                            avg_confidence = sum(confidences) / len(confidences) / 100.0
                            
                            if avg_confidence > best_confidence:
                                best_result = result_text
                                best_confidence = avg_confidence
                    
                    except Exception as e:
                        logger.debug(f"OCR attempt failed: {e}")
                        continue
            
            if best_result and best_confidence > 0.3:
                return CaptchaSolution(
                    solution=best_result,
                    confidence=best_confidence,
                    method_used="tesseract_ocr",
                    solve_time=0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in OCR solving: {e}")
            return None
    
    def _preprocess_captcha_image(self, image: Image.Image) -> List[Image.Image]:
        """Preprocess CAPTCHA image for better OCR accuracy."""
        processed_images = []
        
        try:
            # Convert to grayscale
            gray = image.convert('L')
            processed_images.append(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(2.0)
            processed_images.append(high_contrast)
            
            # Apply threshold
            threshold_img = gray.point(lambda x: 0 if x < 128 else 255, '1')
            processed_images.append(threshold_img)
            
            # Noise reduction
            denoised = gray.filter(ImageFilter.MedianFilter(size=3))
            processed_images.append(denoised)
            
            # Sharpen
            sharpened = gray.filter(ImageFilter.SHARPEN)
            processed_images.append(sharpened)
            
            # Resize for better OCR
            width, height = gray.size
            if width < 200 or height < 50:
                scale_factor = max(200 / width, 50 / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                resized = gray.resize(new_size, Image.LANCZOS)
                processed_images.append(resized)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            processed_images = [image]  # Fallback to original
        
        return processed_images
    
    async def _solve_with_service(self, challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA using third-party services."""
        # Try services in order of preference
        service_order = ["2captcha", "anticaptcha", "capmonster"]
        
        for service_name in service_order:
            if service_name not in self.solving_services:
                continue
            
            service = self.solving_services[service_name]
            if challenge.captcha_type not in service["supported_types"]:
                continue
            
            try:
                solution = await self._solve_with_specific_service(service_name, service, challenge)
                if solution:
                    return solution
            except Exception as e:
                logger.error(f"Error with {service_name}: {e}")
                continue
        
        return None
    
    async def _solve_with_specific_service(self, service_name: str, service: Dict[str, Any], challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA with a specific service."""
        if service_name == "2captcha":
            return await self._solve_with_2captcha(service, challenge)
        elif service_name == "anticaptcha":
            return await self._solve_with_anticaptcha(service, challenge)
        elif service_name == "capmonster":
            return await self._solve_with_capmonster(service, challenge)
        
        return None
    
    async def _solve_with_2captcha(self, service: Dict[str, Any], challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA using 2captcha service."""
        try:
            api_key = service["api_key"]
            base_url = service["base_url"]
            
            # Submit CAPTCHA
            submit_data = {"key": api_key, "method": "post"}
            
            if challenge.captcha_type == CaptchaType.TEXT_BASED:
                if challenge.image_data:
                    submit_data["file"] = base64.b64encode(challenge.image_data).decode()
                elif challenge.image_url:
                    submit_data["url"] = challenge.image_url
                else:
                    return None
            
            elif challenge.captcha_type == CaptchaType.RECAPTCHA_V2:
                submit_data.update({
                    "method": "userrecaptcha",
                    "googlekey": challenge.site_key,
                    "pageurl": challenge.page_url
                })
            
            elif challenge.captcha_type == CaptchaType.HCAPTCHA:
                submit_data.update({
                    "method": "hcaptcha",
                    "sitekey": challenge.site_key,
                    "pageurl": challenge.page_url
                })
            
            # Submit request
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{base_url}/in.php", data=submit_data) as response:
                    result = await response.text()
                    
                    if result.startswith("OK|"):
                        captcha_id = result.split("|")[1]
                    else:
                        logger.error(f"2captcha submit error: {result}")
                        return None
                
                # Poll for result
                for _ in range(60):  # Wait up to 5 minutes
                    await asyncio.sleep(5)
                    
                    async with session.get(f"{base_url}/res.php", params={"key": api_key, "action": "get", "id": captcha_id}) as response:
                        result = await response.text()
                        
                        if result == "CAPCHA_NOT_READY":
                            continue
                        elif result.startswith("OK|"):
                            solution_text = result.split("|")[1]
                            return CaptchaSolution(
                                solution=solution_text,
                                confidence=0.9,  # Assume high confidence for paid service
                                method_used="2captcha",
                                solve_time=0
                            )
                        else:
                            logger.error(f"2captcha result error: {result}")
                            return None
                
                logger.error("2captcha timeout")
                return None
                
        except Exception as e:
            logger.error(f"Error with 2captcha: {e}")
            return None
    
    async def _solve_with_anticaptcha(self, service: Dict[str, Any], challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA using AntiCaptcha service."""
        # Similar implementation to 2captcha but with AntiCaptcha API
        # Implementation details would follow AntiCaptcha's API documentation
        logger.info("AntiCaptcha solving not implemented yet")
        return None
    
    async def _solve_with_capmonster(self, service: Dict[str, Any], challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve CAPTCHA using CapMonster service."""
        # Similar implementation to 2captcha but with CapMonster API
        # Implementation details would follow CapMonster's API documentation
        logger.info("CapMonster solving not implemented yet")
        return None
    
    async def _solve_slider_captcha(self, page: Page, challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve slider CAPTCHA by simulating drag movement."""
        try:
            # Find slider elements
            slider_track = await page.query_selector('.slider-track, .slide-track')
            slider_button = await page.query_selector('.slider-button, .slide-button')
            
            if not slider_track or not slider_button:
                return None
            
            # Get positions
            track_box = await slider_track.bounding_box()
            button_box = await slider_button.bounding_box()
            
            if not track_box or not button_box:
                return None
            
            # Calculate drag distance
            drag_distance = track_box['width'] - button_box['width']
            
            # Perform drag with human-like movement
            start_x = button_box['x'] + button_box['width'] / 2
            start_y = button_box['y'] + button_box['height'] / 2
            end_x = start_x + drag_distance
            
            # Simulate human-like drag
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            
            # Move in small steps with slight variations
            steps = 20
            for i in range(steps):
                progress = (i + 1) / steps
                current_x = start_x + (drag_distance * progress)
                
                # Add slight vertical variation
                variation_y = start_y + random.uniform(-2, 2)
                
                await page.mouse.move(current_x, variation_y)
                await asyncio.sleep(random.uniform(0.01, 0.03))
            
            await page.mouse.up()
            
            # Wait for validation
            await asyncio.sleep(2)
            
            # Check if successful (this would need to be customized per site)
            success_indicator = await page.query_selector('.success, .verified, [class*="success"]')
            
            if success_indicator:
                return CaptchaSolution(
                    solution="slider_completed",
                    confidence=0.8,
                    method_used="slider_automation",
                    solve_time=0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error solving slider CAPTCHA: {e}")
            return None
    
    async def _solve_math_captcha(self, challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Solve simple math CAPTCHA."""
        if not challenge.challenge_text:
            return None
        
        try:
            text = challenge.challenge_text.lower()
            
            # Simple math operations
            if "+" in text:
                parts = text.split("+")
                if len(parts) == 2:
                    num1 = int(parts[0].strip())
                    num2 = int(parts[1].strip())
                    result = num1 + num2
                    
                    return CaptchaSolution(
                        solution=str(result),
                        confidence=1.0,
                        method_used="math_solver",
                        solve_time=0
                    )
            
            elif "-" in text:
                parts = text.split("-")
                if len(parts) == 2:
                    num1 = int(parts[0].strip())
                    num2 = int(parts[1].strip())
                    result = num1 - num2
                    
                    return CaptchaSolution(
                        solution=str(result),
                        confidence=1.0,
                        method_used="math_solver",
                        solve_time=0
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error solving math CAPTCHA: {e}")
            return None
    
    async def _request_human_solve(self, challenge: CaptchaChallenge) -> Optional[CaptchaSolution]:
        """Request human intervention for CAPTCHA solving."""
        # This would integrate with a human solver interface
        # For now, just log the request
        logger.info("Human solver requested but not implemented")
        return None
    
    async def submit_captcha_solution(self, page: Page, captcha_type: CaptchaType, solution: CaptchaSolution) -> bool:
        """Submit the CAPTCHA solution to the page."""
        try:
            if captcha_type == CaptchaType.TEXT_BASED:
                # Find input field and enter solution
                captcha_input = await page.query_selector('input[name*="captcha"], input[id*="captcha"], .captcha input')
                if captcha_input:
                    await captcha_input.fill(solution.solution)
                    return True
            
            elif captcha_type in [CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3]:
                # For reCAPTCHA, the solution is usually handled automatically
                # by the service, but we might need to trigger a callback
                await page.evaluate(f"""
                    if (window.grecaptcha && window.grecaptcha.getResponse) {{
                        window.grecaptcha.execute();
                    }}
                """)
                return True
            
            elif captcha_type == CaptchaType.HCAPTCHA:
                # Similar to reCAPTCHA
                await page.evaluate(f"""
                    if (window.hcaptcha) {{
                        window.hcaptcha.execute();
                    }}
                """)
                return True
            
            elif captcha_type == CaptchaType.SLIDER:
                # Slider solution is already applied during solving
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error submitting CAPTCHA solution: {e}")
            return False
    
    def _update_solve_statistics(self, captcha_type: CaptchaType, method: str, success: bool, solve_time: float):
        """Update solving statistics."""
        key = f"{captcha_type.value}_{method}"
        
        if key not in self.success_rates:
            self.success_rates[key] = {"successes": 0, "attempts": 0}
            self.average_solve_times[key] = {"total_time": 0, "count": 0}
        
        self.success_rates[key]["attempts"] += 1
        if success:
            self.success_rates[key]["successes"] += 1
        
        self.average_solve_times[key]["total_time"] += solve_time
        self.average_solve_times[key]["count"] += 1
    
    def get_solve_statistics(self) -> Dict[str, Any]:
        """Get CAPTCHA solving statistics."""
        stats = {
            "total_attempts": len(self.solve_history),
            "success_rates": {},
            "average_solve_times": {},
            "recent_performance": []
        }
        
        # Calculate success rates
        for key, data in self.success_rates.items():
            if data["attempts"] > 0:
                stats["success_rates"][key] = data["successes"] / data["attempts"]
        
        # Calculate average solve times
        for key, data in self.average_solve_times.items():
            if data["count"] > 0:
                stats["average_solve_times"][key] = data["total_time"] / data["count"]
        
        # Recent performance (last 10 attempts)
        stats["recent_performance"] = self.solve_history[-10:]
        
        return stats