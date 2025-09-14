"""
Advanced Computer Vision System
Enhanced visual processing for survey automation with multimodal AI integration.
"""

import asyncio
import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import aiofiles
from playwright.async_api import Page, ElementHandle
from sklearn.cluster import DBSCAN
from scipy import ndimage
import pickle

logger = logging.getLogger(__name__)

class ElementVisualType(Enum):
    """Visual types of UI elements."""
    BUTTON = "button"
    INPUT_FIELD = "input_field"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    SLIDER = "slider"
    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    CAPTCHA = "captcha"
    PROGRESS_BAR = "progress_bar"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"

@dataclass
class VisualElement:
    """Visual element detected by computer vision."""
    element_id: str
    element_type: ElementVisualType
    bounding_box: Dict[str, int]  # x, y, width, height
    confidence: float
    text_content: str
    visual_features: Dict[str, Any]
    interaction_point: Tuple[int, int]  # Optimal click point
    accessibility_score: float
    prominence_score: float
    ocr_confidence: Optional[float] = None

@dataclass
class LayoutAnalysis:
    """Complete visual layout analysis."""
    elements: List[VisualElement]
    layout_type: str
    complexity_score: float
    visual_hierarchy: List[str]  # Element IDs in visual importance order
    interaction_flow: List[str]  # Suggested interaction sequence
    screenshot_hash: str
    analysis_timestamp: datetime
    heatmap_data: Optional[np.ndarray] = None

@dataclass
class CaptchaAnalysis:
    """CAPTCHA detection and analysis."""
    captcha_type: str
    bounding_box: Dict[str, int]
    confidence: float
    image_data: bytes
    text_content: Optional[str] = None
    complexity_level: str = "medium"  # low, medium, high
    solving_strategy: str = "ocr"  # ocr, third_party, human

class ComputerVisionSystem:
    """Advanced computer vision system for survey automation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.element_cache: Dict[str, LayoutAnalysis] = {}
        self.visual_patterns: Dict[str, Any] = {}
        self.heatmap_model = None
        
        # Load pre-trained models and patterns
        self._load_visual_patterns()
        self._initialize_models()
    
    def _load_visual_patterns(self):
        """Load visual patterns for element recognition."""
        self.visual_patterns = {
            'button_patterns': {
                'color_ranges': [
                    ([100, 150, 0], [140, 255, 255]),  # Blue buttons
                    ([25, 150, 0], [35, 255, 255]),   # Green buttons
                    ([0, 150, 0], [10, 255, 255]),    # Red buttons
                ],
                'shape_features': ['rectangular', 'rounded_corners', 'elevated'],
                'text_patterns': ['submit', 'next', 'continue', 'send', 'complete']
            },
            'input_patterns': {
                'border_styles': ['solid', 'inset', 'outset'],
                'background_colors': ['white', 'light_gray', 'transparent'],
                'placeholder_indicators': ['gray_text', 'italic_text']
            },
            'captcha_patterns': {
                'size_ranges': [(100, 50), (400, 200)],
                'text_indicators': ['captcha', 'verification', 'security'],
                'visual_features': ['distorted_text', 'background_noise', 'grid_overlay']
            }
        }
    
    def _initialize_models(self):
        """Initialize computer vision models."""
        try:
            # Initialize Tesseract with custom config
            self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR initialized")
            
        except Exception as e:
            logger.warning(f"⚠️  Tesseract OCR not available: {e}")
    
    async def analyze_page_visually(self, page: Page, screenshot_path: Optional[str] = None) -> LayoutAnalysis:
        """Perform comprehensive visual analysis of a page."""
        # Take screenshot
        screenshot_bytes = await page.screenshot(full_page=True)
        screenshot_hash = hashlib.sha256(screenshot_bytes).hexdigest()[:16]
        
        # Check cache
        if screenshot_hash in self.element_cache:
            logger.info(f"Using cached visual analysis for {screenshot_hash}")
            return self.element_cache[screenshot_hash]
        
        # Convert to OpenCV format
        image = self._bytes_to_cv2(screenshot_bytes)
        
        # Perform visual analysis
        elements = await self._detect_visual_elements(image, page)
        layout_type = self._classify_layout_type(elements, image)
        complexity_score = self._calculate_visual_complexity(elements, image)
        visual_hierarchy = self._determine_visual_hierarchy(elements)
        interaction_flow = self._suggest_interaction_flow(elements)
        heatmap_data = self._generate_attention_heatmap(image, elements)
        
        analysis = LayoutAnalysis(
            elements=elements,
            layout_type=layout_type,
            complexity_score=complexity_score,
            visual_hierarchy=visual_hierarchy,
            interaction_flow=interaction_flow,
            screenshot_hash=screenshot_hash,
            analysis_timestamp=datetime.now(),
            heatmap_data=heatmap_data
        )
        
        # Cache analysis
        self.element_cache[screenshot_hash] = analysis
        
        # Save screenshot if path provided
        if screenshot_path:
            await self._save_annotated_screenshot(image, elements, screenshot_path)
        
        logger.info(f"Visual analysis complete: {len(elements)} elements detected")
        return analysis
    
    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image."""
        image = Image.open(io.BytesIO(image_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    async def _detect_visual_elements(self, image: np.ndarray, page: Page) -> List[VisualElement]:
        """Detect and classify visual elements in the image."""
        elements = []
        
        # Detect buttons
        button_elements = await self._detect_buttons(image, page)
        elements.extend(button_elements)
        
        # Detect input fields
        input_elements = await self._detect_input_fields(image, page)
        elements.extend(input_elements)
        
        # Detect checkboxes and radio buttons
        choice_elements = await self._detect_choice_elements(image, page)
        elements.extend(choice_elements)
        
        # Detect text blocks
        text_elements = await self._detect_text_blocks(image)
        elements.extend(text_elements)
        
        # Detect CAPTCHAs
        captcha_elements = await self._detect_captchas(image)
        elements.extend(captcha_elements)
        
        # Remove duplicates and overlapping elements
        elements = self._remove_duplicate_elements(elements)
        
        return elements
    
    async def _detect_buttons(self, image: np.ndarray, page: Page) -> List[VisualElement]:
        """Detect button elements using visual features."""
        buttons = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect buttons by color and shape
        for i, (lower, upper) in enumerate(self.visual_patterns['button_patterns']['color_ranges']):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Reasonable button size
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (buttons are usually wider than tall)
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 10:
                    continue
                
                # Extract text from button area
                button_roi = image[y:y+h, x:x+w]
                text_content = self._extract_text_from_roi(button_roi)
                
                # Calculate confidence based on visual features
                confidence = self._calculate_button_confidence(contour, text_content)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    element = VisualElement(
                        element_id=f"button_{i}_{x}_{y}",
                        element_type=ElementVisualType.BUTTON,
                        bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                        confidence=confidence,
                        text_content=text_content,
                        visual_features={
                            'color_index': i,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour_points': len(contour)
                        },
                        interaction_point=(x + w//2, y + h//2),
                        accessibility_score=self._calculate_accessibility_score(x, y, w, h, text_content),
                        prominence_score=self._calculate_prominence_score(x, y, w, h, image.shape)
                    )
                    buttons.append(element)
        
        return buttons
    
    async def _detect_input_fields(self, image: np.ndarray, page: Page) -> List[VisualElement]:
        """Detect input field elements."""
        inputs = []
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find rectangular contours (typical for input fields)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (input fields have specific size ranges)
                if w < 50 or h < 20 or w > 800 or h > 100:
                    continue
                
                # Check aspect ratio (input fields are usually wider than tall)
                aspect_ratio = w / h
                if aspect_ratio < 2 or aspect_ratio > 20:
                    continue
                
                # Check if it looks like an input field (light background)
                roi = image[y:y+h, x:x+w]
                avg_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                
                if avg_brightness > 200:  # Light background typical for input fields
                    # Extract any placeholder or label text
                    text_content = self._extract_text_from_roi(roi)
                    
                    confidence = 0.6 if text_content else 0.4
                    
                    element = VisualElement(
                        element_id=f"input_{i}_{x}_{y}",
                        element_type=ElementVisualType.INPUT_FIELD,
                        bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                        confidence=confidence,
                        text_content=text_content,
                        visual_features={
                            'brightness': avg_brightness,
                            'aspect_ratio': aspect_ratio,
                            'corners': len(approx)
                        },
                        interaction_point=(x + w//2, y + h//2),
                        accessibility_score=self._calculate_accessibility_score(x, y, w, h, text_content),
                        prominence_score=self._calculate_prominence_score(x, y, w, h, image.shape)
                    )
                    inputs.append(element)
        
        return inputs
    
    async def _detect_choice_elements(self, image: np.ndarray, page: Page) -> List[VisualElement]:
        """Detect checkbox and radio button elements."""
        choices = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles (radio buttons)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles):
                # Extract surrounding area for context
                roi_size = r * 4
                roi_x = max(0, x - roi_size)
                roi_y = max(0, y - roi_size)
                roi_w = min(image.shape[1] - roi_x, roi_size * 2)
                roi_h = min(image.shape[0] - roi_y, roi_size * 2)
                
                roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                text_content = self._extract_text_from_roi(roi)
                
                element = VisualElement(
                    element_id=f"radio_{i}_{x}_{y}",
                    element_type=ElementVisualType.RADIO_BUTTON,
                    bounding_box={'x': x-r, 'y': y-r, 'width': r*2, 'height': r*2},
                    confidence=0.7,
                    text_content=text_content,
                    visual_features={'radius': r, 'center': (x, y)},
                    interaction_point=(x, y),
                    accessibility_score=self._calculate_accessibility_score(x-r, y-r, r*2, r*2, text_content),
                    prominence_score=self._calculate_prominence_score(x-r, y-r, r*2, r*2, image.shape)
                )
                choices.append(element)
        
        # Detect squares (checkboxes)
        # Use template matching for common checkbox patterns
        checkbox_templates = self._generate_checkbox_templates()
        
        for i, template in enumerate(checkbox_templates):
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.6)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = template.shape[::-1]
                
                # Extract surrounding area for context
                roi_size = max(w, h) * 3
                roi_x = max(0, x - roi_size//2)
                roi_y = max(0, y - roi_size//2)
                roi_w = min(image.shape[1] - roi_x, roi_size)
                roi_h = min(image.shape[0] - roi_y, roi_size)
                
                roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                text_content = self._extract_text_from_roi(roi)
                
                element = VisualElement(
                    element_id=f"checkbox_{i}_{x}_{y}",
                    element_type=ElementVisualType.CHECKBOX,
                    bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                    confidence=result[y, x],
                    text_content=text_content,
                    visual_features={'template_match': result[y, x]},
                    interaction_point=(x + w//2, y + h//2),
                    accessibility_score=self._calculate_accessibility_score(x, y, w, h, text_content),
                    prominence_score=self._calculate_prominence_score(x, y, w, h, image.shape)
                )
                choices.append(element)
        
        return choices
    
    async def _detect_text_blocks(self, image: np.ndarray) -> List[VisualElement]:
        """Detect text blocks using OCR."""
        text_elements = []
        
        try:
            # Use Tesseract to get detailed text information
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=self.tesseract_config)
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                if text and confidence > 30:  # Filter low-confidence detections
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Skip very small text
                    if w < 10 or h < 10:
                        continue
                    
                    element = VisualElement(
                        element_id=f"text_{i}_{x}_{y}",
                        element_type=ElementVisualType.TEXT_BLOCK,
                        bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                        confidence=confidence / 100.0,  # Normalize to 0-1
                        text_content=text,
                        visual_features={
                            'word_count': len(text.split()),
                            'font_size_estimate': h
                        },
                        interaction_point=(x + w//2, y + h//2),
                        accessibility_score=1.0,  # Text is always accessible
                        prominence_score=self._calculate_text_prominence(text, h, image.shape),
                        ocr_confidence=confidence
                    )
                    text_elements.append(element)
        
        except Exception as e:
            logger.warning(f"OCR text detection failed: {e}")
        
        return text_elements
    
    async def _detect_captchas(self, image: np.ndarray) -> List[VisualElement]:
        """Detect CAPTCHA elements."""
        captchas = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for typical CAPTCHA characteristics
        # 1. Distorted text patterns
        # 2. Specific size ranges
        # 3. Background noise patterns
        
        # Use edge detection to find potential CAPTCHA regions
        edges = cv2.Canny(gray, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if size matches typical CAPTCHA dimensions
            if (100 <= w <= 400) and (30 <= h <= 150):
                roi = image[y:y+h, x:x+w]
                
                # Analyze visual characteristics
                noise_level = self._calculate_noise_level(roi)
                text_distortion = self._calculate_text_distortion(roi)
                
                # Check for CAPTCHA-like features
                if noise_level > 0.3 or text_distortion > 0.4:
                    # Try to extract text
                    text_content = self._extract_text_from_roi(roi, captcha_mode=True)
                    
                    confidence = (noise_level + text_distortion) / 2
                    
                    element = VisualElement(
                        element_id=f"captcha_{i}_{x}_{y}",
                        element_type=ElementVisualType.CAPTCHA,
                        bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                        confidence=confidence,
                        text_content=text_content,
                        visual_features={
                            'noise_level': noise_level,
                            'text_distortion': text_distortion,
                            'area': w * h
                        },
                        interaction_point=(x + w//2, y + h//2),
                        accessibility_score=0.1,  # CAPTCHAs are intentionally hard to access
                        prominence_score=0.8  # CAPTCHAs are usually prominent
                    )
                    captchas.append(element)
        
        return captchas
    
    def _extract_text_from_roi(self, roi: np.ndarray, captcha_mode: bool = False) -> str:
        """Extract text from a region of interest."""
        try:
            if captcha_mode:
                # Special preprocessing for CAPTCHAs
                roi = self._preprocess_captcha_image(roi)
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            else:
                config = self.tesseract_config
            
            text = pytesseract.image_to_string(roi, config=config).strip()
            return text
        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            return ""
    
    def _preprocess_captcha_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess CAPTCHA image for better OCR."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _generate_checkbox_templates(self) -> List[np.ndarray]:
        """Generate checkbox templates for template matching."""
        templates = []
        
        # Empty checkbox template
        empty_checkbox = np.zeros((20, 20), dtype=np.uint8)
        cv2.rectangle(empty_checkbox, (2, 2), (17, 17), 255, 2)
        templates.append(empty_checkbox)
        
        # Checked checkbox template
        checked_checkbox = empty_checkbox.copy()
        cv2.line(checked_checkbox, (5, 10), (9, 14), 255, 2)
        cv2.line(checked_checkbox, (9, 14), (15, 6), 255, 2)
        templates.append(checked_checkbox)
        
        return templates
    
    def _calculate_button_confidence(self, contour: np.ndarray, text_content: str) -> float:
        """Calculate confidence score for button detection."""
        confidence = 0.3  # Base confidence
        
        # Boost confidence if text contains button-like words
        button_words = ['submit', 'next', 'continue', 'send', 'complete', 'finish', 'start', 'click']
        if any(word in text_content.lower() for word in button_words):
            confidence += 0.4
        
        # Boost confidence based on shape regularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area > 0:
            solidity = contour_area / hull_area
            confidence += solidity * 0.3
        
        return min(confidence, 1.0)
    
    def _calculate_accessibility_score(self, x: int, y: int, w: int, h: int, text_content: str) -> float:
        """Calculate accessibility score for an element."""
        score = 0.5  # Base score
        
        # Boost score if element has text
        if text_content:
            score += 0.3
        
        # Boost score if element is reasonably sized
        area = w * h
        if 1000 <= area <= 50000:  # Reasonable interaction size
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_prominence_score(self, x: int, y: int, w: int, h: int, image_shape: Tuple[int, int, int]) -> float:
        """Calculate visual prominence score."""
        img_height, img_width = image_shape[:2]
        
        # Size factor
        area = w * h
        size_factor = min(area / (img_width * img_height), 0.5)
        
        # Position factor (elements higher on page are more prominent)
        position_factor = max(0, 1 - (y / img_height))
        
        # Center bias (elements near center are more prominent)
        center_x, center_y = img_width // 2, img_height // 2
        element_center_x, element_center_y = x + w // 2, y + h // 2
        
        distance_from_center = ((element_center_x - center_x) ** 2 + (element_center_y - center_y) ** 2) ** 0.5
        max_distance = (center_x ** 2 + center_y ** 2) ** 0.5
        center_factor = 1 - (distance_from_center / max_distance)
        
        prominence = (size_factor * 0.4 + position_factor * 0.3 + center_factor * 0.3)
        return min(prominence, 1.0)
    
    def _calculate_text_prominence(self, text: str, font_size: int, image_shape: Tuple[int, int, int]) -> float:
        """Calculate prominence score for text elements."""
        # Base score from font size
        size_score = min(font_size / 50, 1.0)  # Normalize to typical font sizes
        
        # Boost for important words
        important_words = ['question', 'required', 'error', 'warning', 'important']
        importance_boost = 0.3 if any(word in text.lower() for word in important_words) else 0
        
        # Boost for longer text (likely to be questions)
        length_boost = min(len(text) / 100, 0.2)
        
        return min(size_score + importance_boost + length_boost, 1.0)
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate noise level in an image region."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (measure of blur/noise)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        return min(laplacian_var / 1000, 1.0)
    
    def _calculate_text_distortion(self, image: np.ndarray) -> float:
        """Calculate text distortion level."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        # High edge density suggests distorted text
        edge_density = edge_pixels / total_pixels
        
        return min(edge_density * 2, 1.0)  # Scale and cap at 1.0
    
    def _remove_duplicate_elements(self, elements: List[VisualElement]) -> List[VisualElement]:
        """Remove duplicate and overlapping elements."""
        if not elements:
            return elements
        
        # Sort by confidence (highest first)
        elements.sort(key=lambda e: e.confidence, reverse=True)
        
        unique_elements = []
        
        for element in elements:
            is_duplicate = False
            
            for existing in unique_elements:
                # Calculate overlap
                overlap = self._calculate_bounding_box_overlap(
                    element.bounding_box, existing.bounding_box
                )
                
                # If significant overlap and same type, consider duplicate
                if overlap > 0.7 and element.element_type == existing.element_type:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(element)
        
        return unique_elements
    
    def _calculate_bounding_box_overlap(self, box1: Dict[str, int], box2: Dict[str, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        intersection_area = x_overlap * y_overlap
        
        # Calculate union
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _classify_layout_type(self, elements: List[VisualElement], image: np.ndarray) -> str:
        """Classify the overall layout type."""
        button_count = len([e for e in elements if e.element_type == ElementVisualType.BUTTON])
        input_count = len([e for e in elements if e.element_type == ElementVisualType.INPUT_FIELD])
        choice_count = len([e for e in elements if e.element_type in [ElementVisualType.CHECKBOX, ElementVisualType.RADIO_BUTTON]])
        text_count = len([e for e in elements if e.element_type == ElementVisualType.TEXT_BLOCK])
        
        # Simple classification logic
        if choice_count > 5:
            return "multiple_choice_survey"
        elif input_count > 3:
            return "form_heavy"
        elif button_count > 3:
            return "navigation_heavy"
        elif text_count > 10:
            return "text_heavy"
        else:
            return "balanced_layout"
    
    def _calculate_visual_complexity(self, elements: List[VisualElement], image: np.ndarray) -> float:
        """Calculate overall visual complexity score."""
        # Element density
        total_elements = len(elements)
        image_area = image.shape[0] * image.shape[1]
        element_density = total_elements / (image_area / 10000)  # Normalize per 10k pixels
        
        # Element type diversity
        unique_types = len(set(e.element_type for e in elements))
        type_diversity = unique_types / len(ElementVisualType)
        
        # Visual clutter (overlapping elements)
        overlap_count = 0
        for i, elem1 in enumerate(elements):
            for elem2 in elements[i+1:]:
                if self._calculate_bounding_box_overlap(elem1.bounding_box, elem2.bounding_box) > 0.1:
                    overlap_count += 1
        
        clutter_score = min(overlap_count / max(total_elements, 1), 1.0)
        
        # Combine factors
        complexity = (element_density * 0.4 + type_diversity * 0.3 + clutter_score * 0.3)
        return min(complexity, 1.0)
    
    def _determine_visual_hierarchy(self, elements: List[VisualElement]) -> List[str]:
        """Determine visual hierarchy of elements."""
        # Sort by prominence score
        sorted_elements = sorted(elements, key=lambda e: e.prominence_score, reverse=True)
        return [e.element_id for e in sorted_elements]
    
    def _suggest_interaction_flow(self, elements: List[VisualElement]) -> List[str]:
        """Suggest optimal interaction flow."""
        # Group elements by type and position
        buttons = [e for e in elements if e.element_type == ElementVisualType.BUTTON]
        inputs = [e for e in elements if e.element_type == ElementVisualType.INPUT_FIELD]
        choices = [e for e in elements if e.element_type in [ElementVisualType.CHECKBOX, ElementVisualType.RADIO_BUTTON]]
        
        flow = []
        
        # Sort by vertical position (top to bottom)
        all_interactive = inputs + choices + buttons
        all_interactive.sort(key=lambda e: e.bounding_box['y'])
        
        # Add elements in reading order
        for element in all_interactive:
            flow.append(element.element_id)
        
        return flow
    
    def _generate_attention_heatmap(self, image: np.ndarray, elements: List[VisualElement]) -> np.ndarray:
        """Generate attention heatmap based on element prominence."""
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for element in elements:
            x, y, w, h = element.bounding_box['x'], element.bounding_box['y'], element.bounding_box['width'], element.bounding_box['height']
            
            # Create Gaussian blob for element
            center_x, center_y = x + w // 2, y + h // 2
            
            # Create coordinate grids
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            
            # Calculate Gaussian
            sigma = max(w, h) / 4  # Spread based on element size
            gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
            
            # Weight by prominence score
            weighted_gaussian = gaussian * element.prominence_score
            
            # Add to heatmap
            heatmap += weighted_gaussian
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    async def _save_annotated_screenshot(self, image: np.ndarray, elements: List[VisualElement], path: str):
        """Save screenshot with element annotations."""
        annotated = image.copy()
        
        # Color map for different element types
        color_map = {
            ElementVisualType.BUTTON: (0, 255, 0),      # Green
            ElementVisualType.INPUT_FIELD: (255, 0, 0),  # Blue
            ElementVisualType.CHECKBOX: (0, 255, 255),   # Yellow
            ElementVisualType.RADIO_BUTTON: (255, 0, 255), # Magenta
            ElementVisualType.TEXT_BLOCK: (128, 128, 128), # Gray
            ElementVisualType.CAPTCHA: (0, 0, 255),      # Red
        }
        
        for element in elements:
            x, y, w, h = element.bounding_box['x'], element.bounding_box['y'], element.bounding_box['width'], element.bounding_box['height']
            color = color_map.get(element.element_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw element ID
            cv2.putText(annotated, element.element_id, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw confidence score
            confidence_text = f"{element.confidence:.2f}"
            cv2.putText(annotated, confidence_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save annotated image
        cv2.imwrite(path, annotated)
        logger.info(f"Saved annotated screenshot to {path}")
    
    async def detect_captcha_advanced(self, page: Page) -> Optional[CaptchaAnalysis]:
        """Advanced CAPTCHA detection and analysis."""
        screenshot_bytes = await page.screenshot()
        image = self._bytes_to_cv2(screenshot_bytes)
        
        # Detect potential CAPTCHA regions
        captcha_elements = await self._detect_captchas(image)
        
        if not captcha_elements:
            return None
        
        # Analyze the most confident CAPTCHA
        best_captcha = max(captcha_elements, key=lambda e: e.confidence)
        
        x, y, w, h = best_captcha.bounding_box['x'], best_captcha.bounding_box['y'], best_captcha.bounding_box['width'], best_captcha.bounding_box['height']
        captcha_roi = image[y:y+h, x:x+w]
        
        # Determine CAPTCHA type and complexity
        captcha_type = self._classify_captcha_type(captcha_roi)
        complexity_level = self._assess_captcha_complexity(captcha_roi)
        solving_strategy = self._determine_solving_strategy(captcha_type, complexity_level)
        
        # Extract image data
        _, buffer = cv2.imencode('.png', captcha_roi)
        image_data = buffer.tobytes()
        
        return CaptchaAnalysis(
            captcha_type=captcha_type,
            bounding_box=best_captcha.bounding_box,
            confidence=best_captcha.confidence,
            image_data=image_data,
            text_content=best_captcha.text_content,
            complexity_level=complexity_level,
            solving_strategy=solving_strategy
        )
    
    def _classify_captcha_type(self, captcha_image: np.ndarray) -> str:
        """Classify the type of CAPTCHA."""
        # Simple classification based on visual features
        height, width = captcha_image.shape[:2]
        
        # Check for text-based CAPTCHA
        text_content = self._extract_text_from_roi(captcha_image, captcha_mode=True)
        if text_content and len(text_content) > 2:
            return "text_captcha"
        
        # Check for image-based CAPTCHA (multiple objects)
        gray = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY) if len(captcha_image.shape) == 3 else captcha_image
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 5:
            return "image_selection_captcha"
        
        # Check for checkbox CAPTCHA
        if width > height * 2:  # Wide aspect ratio typical for "I'm not a robot"
            return "checkbox_captcha"
        
        return "unknown_captcha"
    
    def _assess_captcha_complexity(self, captcha_image: np.ndarray) -> str:
        """Assess CAPTCHA complexity level."""
        noise_level = self._calculate_noise_level(captcha_image)
        distortion_level = self._calculate_text_distortion(captcha_image)
        
        complexity_score = (noise_level + distortion_level) / 2
        
        if complexity_score < 0.3:
            return "low"
        elif complexity_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _determine_solving_strategy(self, captcha_type: str, complexity_level: str) -> str:
        """Determine the best strategy for solving the CAPTCHA."""
        if captcha_type == "checkbox_captcha":
            return "simple_click"
        elif captcha_type == "text_captcha" and complexity_level == "low":
            return "ocr"
        elif captcha_type == "text_captcha" and complexity_level in ["medium", "high"]:
            return "third_party"
        elif captcha_type == "image_selection_captcha":
            return "third_party"
        else:
            return "human"
    
    async def find_element_by_visual_similarity(self, page: Page, target_description: str) -> Optional[VisualElement]:
        """Find element by visual similarity to description."""
        analysis = await self.analyze_page_visually(page)
        
        # Simple text matching for now (could be enhanced with semantic similarity)
        target_words = target_description.lower().split()
        
        best_match = None
        best_score = 0
        
        for element in analysis.elements:
            if not element.text_content:
                continue
            
            element_words = element.text_content.lower().split()
            
            # Calculate word overlap score
            common_words = set(target_words).intersection(set(element_words))
            score = len(common_words) / max(len(target_words), 1)
            
            # Boost score for exact matches
            if target_description.lower() in element.text_content.lower():
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_match = element
        
        return best_match if best_score > 0.3 else None
    
    def get_visual_statistics(self) -> Dict[str, Any]:
        """Get statistics about visual analysis performance."""
        total_analyses = len(self.element_cache)
        
        if total_analyses == 0:
            return {'total_analyses': 0}
        
        # Aggregate statistics from cached analyses
        total_elements = 0
        element_type_counts = {}
        complexity_scores = []
        
        for analysis in self.element_cache.values():
            total_elements += len(analysis.elements)
            complexity_scores.append(analysis.complexity_score)
            
            for element in analysis.elements:
                element_type = element.element_type.value
                element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1
        
        avg_elements_per_page = total_elements / total_analyses
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        return {
            'total_analyses': total_analyses,
            'total_elements_detected': total_elements,
            'avg_elements_per_page': avg_elements_per_page,
            'avg_complexity_score': avg_complexity,
            'element_type_distribution': element_type_counts,
            'cache_hit_rate': 0.0  # Would need to track cache hits vs misses
        }