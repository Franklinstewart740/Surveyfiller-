"""
Layout Awareness & UI Mapping System
Advanced DOM intelligence with fuzzy selectors, OCR integration, and layout caching.
"""

import asyncio
import logging
import json
import hashlib
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import base64
from playwright.async_api import Page, ElementHandle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import aiofiles

logger = logging.getLogger(__name__)

class ElementType(Enum):
    """Types of UI elements."""
    BUTTON = "button"
    INPUT = "input"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTAREA = "textarea"
    LINK = "link"
    TEXT = "text"
    IMAGE = "image"
    FORM = "form"
    CONTAINER = "container"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"

class ElementRole(Enum):
    """Semantic roles of elements."""
    SUBMIT = "submit"
    NEXT = "next"
    PREVIOUS = "previous"
    SKIP = "skip"
    CANCEL = "cancel"
    QUESTION = "question"
    ANSWER_OPTION = "answer_option"
    REQUIRED_FIELD = "required_field"
    OPTIONAL_FIELD = "optional_field"
    NAVIGATION = "navigation"
    CONTENT = "content"
    DECORATION = "decoration"

@dataclass
class ElementInfo:
    """Comprehensive element information."""
    selector: str
    element_type: ElementType
    role: ElementRole
    text_content: str
    attributes: Dict[str, str]
    bounding_box: Dict[str, float]  # x, y, width, height
    z_index: int
    is_visible: bool
    is_enabled: bool
    prominence_score: float
    confidence_score: float
    ocr_text: Optional[str] = None
    parent_selector: Optional[str] = None
    children_selectors: List[str] = None

@dataclass
class LayoutPattern:
    """Common layout patterns for caching."""
    pattern_id: str
    pattern_name: str
    selectors: Dict[str, str]  # role -> selector mapping
    confidence: float
    usage_count: int
    last_used: datetime
    platform: str
    survey_type: str

@dataclass
class OCRResult:
    """OCR text extraction result."""
    text: str
    confidence: float
    bounding_box: Dict[str, float]
    language: str

class LayoutIntelligence:
    """Advanced layout awareness and UI mapping system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.layout_cache: Dict[str, LayoutPattern] = {}
        self.element_cache: Dict[str, List[ElementInfo]] = {}
        self.ocr_cache: Dict[str, List[OCRResult]] = {}
        
        # Fuzzy selector patterns
        self.fuzzy_patterns = self._load_fuzzy_patterns()
        
        # Common button text patterns
        self.button_patterns = self._load_button_patterns()
        
        # Load cached layouts
        asyncio.create_task(self._load_layout_cache())
    
    def _load_fuzzy_patterns(self) -> Dict[str, List[str]]:
        """Load fuzzy selector patterns for common elements."""
        return {
            'next_button': [
                'button:has-text("Next")',
                'button:has-text("Continue")',
                'button:has-text("Proceed")',
                'input[type="submit"][value*="Next"]',
                'input[type="button"][value*="Next"]',
                'a:has-text("Next")',
                '[class*="next"]:has-text("Next")',
                '[id*="next"]',
                'button[class*="next"]',
                'button[id*="next"]'
            ],
            'submit_button': [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Send")',
                'button:has-text("Complete")',
                'button:has-text("Finish")',
                '[class*="submit"]',
                '[id*="submit"]'
            ],
            'skip_button': [
                'button:has-text("Skip")',
                'a:has-text("Skip")',
                'button:has-text("Skip this question")',
                '[class*="skip"]',
                '[id*="skip"]'
            ],
            'previous_button': [
                'button:has-text("Previous")',
                'button:has-text("Back")',
                'a:has-text("Previous")',
                'a:has-text("Back")',
                '[class*="prev"]',
                '[class*="back"]',
                '[id*="prev"]',
                '[id*="back"]'
            ],
            'radio_options': [
                'input[type="radio"]',
                '[role="radio"]',
                '.radio-option',
                '.answer-option input[type="radio"]'
            ],
            'checkbox_options': [
                'input[type="checkbox"]',
                '[role="checkbox"]',
                '.checkbox-option',
                '.answer-option input[type="checkbox"]'
            ],
            'text_input': [
                'input[type="text"]',
                'input[type="email"]',
                'input[type="tel"]',
                'textarea',
                '[contenteditable="true"]'
            ],
            'select_dropdown': [
                'select',
                '[role="combobox"]',
                '.dropdown',
                '.select-wrapper select'
            ]
        }
    
    def _load_button_patterns(self) -> Dict[str, List[str]]:
        """Load common button text patterns for different languages."""
        return {
            'next': ['Next', 'Continue', 'Proceed', 'Forward', 'Siguiente', 'Continuer', 'Weiter'],
            'submit': ['Submit', 'Send', 'Complete', 'Finish', 'Done', 'Enviar', 'Soumettre', 'Senden'],
            'skip': ['Skip', 'Skip this question', 'Pass', 'Omitir', 'Passer', 'Überspringen'],
            'previous': ['Previous', 'Back', 'Return', 'Anterior', 'Précédent', 'Zurück'],
            'cancel': ['Cancel', 'Close', 'Exit', 'Cancelar', 'Annuler', 'Abbrechen']
        }
    
    async def _load_layout_cache(self):
        """Load cached layout patterns from disk."""
        cache_file = self.config.get('layout_cache_file', 'layout_cache.pkl')
        if os.path.exists(cache_file):
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    data = await f.read()
                cache_data = pickle.loads(data)
                
                self.layout_cache = {k: LayoutPattern(**v) for k, v in cache_data.items()}
                logger.info(f"Loaded {len(self.layout_cache)} cached layout patterns")
                
            except Exception as e:
                logger.error(f"Failed to load layout cache: {e}")
    
    async def _save_layout_cache(self):
        """Save layout patterns to disk."""
        cache_file = self.config.get('layout_cache_file', 'layout_cache.pkl')
        try:
            cache_data = {k: asdict(v) for k, v in self.layout_cache.items()}
            data = pickle.dumps(cache_data)
            
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(data)
                
        except Exception as e:
            logger.error(f"Failed to save layout cache: {e}")
    
    async def analyze_page_layout(self, page: Page, platform: str = "unknown") -> Dict[str, Any]:
        """Analyze page layout and identify key elements."""
        page_url = page.url
        page_hash = hashlib.sha256(page_url.encode()).hexdigest()[:16]
        
        # Check if we have cached analysis
        if page_hash in self.element_cache:
            cached_elements = self.element_cache[page_hash]
            # Check if cache is still valid (elements still exist)
            if await self._validate_cached_elements(page, cached_elements):
                logger.info(f"Using cached layout analysis for {page_url}")
                return self._build_layout_analysis(cached_elements)
        
        logger.info(f"Analyzing page layout for {page_url}")
        
        # Get page screenshot for OCR
        screenshot = await page.screenshot(full_page=True)
        ocr_results = await self._perform_ocr_analysis(screenshot)
        
        # Analyze DOM elements
        elements = await self._analyze_dom_elements(page)
        
        # Score element prominence
        for element in elements:
            element.prominence_score = await self._calculate_prominence_score(page, element)
        
        # Predict element roles using multiple methods
        for element in elements:
            element.role = await self._predict_element_role(element, ocr_results)
        
        # Cache the analysis
        self.element_cache[page_hash] = elements
        self.ocr_cache[page_hash] = ocr_results
        
        # Try to match against known layout patterns
        matched_pattern = await self._match_layout_pattern(elements, platform)
        
        analysis = self._build_layout_analysis(elements)
        analysis['matched_pattern'] = matched_pattern
        analysis['ocr_results'] = [asdict(ocr) for ocr in ocr_results]
        
        return analysis
    
    async def _perform_ocr_analysis(self, screenshot_bytes: bytes) -> List[OCRResult]:
        """Perform OCR analysis on page screenshot."""
        try:
            # Convert screenshot to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract to extract text with bounding boxes
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text and len(text) > 1:  # Filter out single characters and empty strings
                    confidence = float(data['conf'][i])
                    
                    if confidence > 30:  # Only include results with reasonable confidence
                        bounding_box = {
                            'x': float(data['left'][i]),
                            'y': float(data['top'][i]),
                            'width': float(data['width'][i]),
                            'height': float(data['height'][i])
                        }
                        
                        ocr_result = OCRResult(
                            text=text,
                            confidence=confidence,
                            bounding_box=bounding_box,
                            language='en'  # Could be detected automatically
                        )
                        ocr_results.append(ocr_result)
            
            logger.info(f"OCR extracted {len(ocr_results)} text elements")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            return []
    
    async def _analyze_dom_elements(self, page: Page) -> List[ElementInfo]:
        """Analyze DOM elements and extract information."""
        elements = []
        
        # Get all interactive elements
        selectors_to_check = [
            'button', 'input', 'select', 'textarea', 'a[href]',
            '[role="button"]', '[role="link"]', '[role="textbox"]',
            '[onclick]', '[class*="btn"]', '[class*="button"]'
        ]
        
        for selector in selectors_to_check:
            try:
                element_handles = await page.query_selector_all(selector)
                
                for handle in element_handles:
                    element_info = await self._extract_element_info(page, handle, selector)
                    if element_info:
                        elements.append(element_info)
                        
            except Exception as e:
                logger.debug(f"Error analyzing selector {selector}: {e}")
        
        # Remove duplicates based on bounding box
        unique_elements = self._remove_duplicate_elements(elements)
        
        logger.info(f"Analyzed {len(unique_elements)} unique DOM elements")
        return unique_elements
    
    async def _extract_element_info(self, page: Page, handle: ElementHandle, selector: str) -> Optional[ElementInfo]:
        """Extract comprehensive information about a DOM element."""
        try:
            # Get element properties
            bounding_box = await handle.bounding_box()
            if not bounding_box:
                return None
            
            # Get element attributes
            tag_name = await handle.evaluate('el => el.tagName.toLowerCase()')
            text_content = await handle.text_content() or ""
            inner_text = await handle.inner_text() or ""
            
            # Get all attributes
            attributes = await handle.evaluate('''el => {
                const attrs = {};
                for (let attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }''')
            
            # Check visibility and enabled state
            is_visible = await handle.is_visible()
            is_enabled = await handle.is_enabled()
            
            # Get z-index
            z_index = await handle.evaluate('el => parseInt(getComputedStyle(el).zIndex) || 0')
            
            # Determine element type
            element_type = self._determine_element_type(tag_name, attributes)
            
            element_info = ElementInfo(
                selector=selector,
                element_type=element_type,
                role=ElementRole.UNKNOWN,  # Will be determined later
                text_content=text_content or inner_text,
                attributes=attributes,
                bounding_box=bounding_box,
                z_index=z_index,
                is_visible=is_visible,
                is_enabled=is_enabled,
                prominence_score=0.0,  # Will be calculated later
                confidence_score=0.0
            )
            
            return element_info
            
        except Exception as e:
            logger.debug(f"Error extracting element info: {e}")
            return None
    
    def _determine_element_type(self, tag_name: str, attributes: Dict[str, str]) -> ElementType:
        """Determine element type based on tag and attributes."""
        if tag_name == 'button':
            return ElementType.BUTTON
        elif tag_name == 'input':
            input_type = attributes.get('type', 'text').lower()
            if input_type in ['button', 'submit']:
                return ElementType.BUTTON
            elif input_type == 'checkbox':
                return ElementType.CHECKBOX
            elif input_type == 'radio':
                return ElementType.RADIO
            else:
                return ElementType.INPUT
        elif tag_name == 'select':
            return ElementType.SELECT
        elif tag_name == 'textarea':
            return ElementType.TEXTAREA
        elif tag_name == 'a':
            return ElementType.LINK
        elif tag_name == 'img':
            return ElementType.IMAGE
        elif tag_name == 'form':
            return ElementType.FORM
        elif tag_name in ['div', 'span', 'section', 'article']:
            # Check for button-like behavior
            if any(attr in attributes for attr in ['onclick', 'role']) or \
               any(cls in attributes.get('class', '').lower() for cls in ['btn', 'button']):
                return ElementType.BUTTON
            return ElementType.CONTAINER
        else:
            return ElementType.UNKNOWN
    
    def _remove_duplicate_elements(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """Remove duplicate elements based on bounding box overlap."""
        unique_elements = []
        
        for element in elements:
            is_duplicate = False
            
            for existing in unique_elements:
                if self._calculate_overlap(element.bounding_box, existing.bounding_box) > 0.8:
                    # Elements overlap significantly, keep the one with higher prominence
                    if element.prominence_score > existing.prominence_score:
                        unique_elements.remove(existing)
                        unique_elements.append(element)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(element)
        
        return unique_elements
    
    def _calculate_overlap(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
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
    
    async def _calculate_prominence_score(self, page: Page, element: ElementInfo) -> float:
        """Calculate element prominence score based on size, position, and styling."""
        score = 0.0
        
        # Size factor (larger elements are more prominent)
        area = element.bounding_box['width'] * element.bounding_box['height']
        page_area = await page.evaluate('() => document.body.scrollWidth * document.body.scrollHeight')
        size_factor = min(area / page_area * 100, 1.0)  # Normalize to 0-1
        score += size_factor * 0.3
        
        # Position factor (elements higher on page are more prominent)
        viewport_height = await page.evaluate('() => window.innerHeight')
        y_position = element.bounding_box['y']
        position_factor = max(0, 1 - (y_position / viewport_height))
        score += position_factor * 0.2
        
        # Z-index factor
        z_index_factor = min(element.z_index / 1000, 1.0) if element.z_index > 0 else 0
        score += z_index_factor * 0.1
        
        # Text content factor (elements with clear action text are more prominent)
        text_lower = element.text_content.lower()
        action_words = ['next', 'submit', 'continue', 'send', 'complete', 'finish']
        if any(word in text_lower for word in action_words):
            score += 0.3
        
        # Element type factor
        if element.element_type == ElementType.BUTTON:
            score += 0.2
        elif element.element_type in [ElementType.INPUT, ElementType.SELECT]:
            score += 0.1
        
        # Visibility and enabled state
        if element.is_visible and element.is_enabled:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _predict_element_role(self, element: ElementInfo, ocr_results: List[OCRResult]) -> ElementRole:
        """Predict element role using multiple methods."""
        text_lower = element.text_content.lower()
        
        # Rule-based role prediction
        if element.element_type == ElementType.BUTTON:
            # Check button text patterns
            for role, patterns in self.button_patterns.items():
                if any(pattern.lower() in text_lower for pattern in patterns):
                    if role == 'next':
                        return ElementRole.NEXT
                    elif role == 'submit':
                        return ElementRole.SUBMIT
                    elif role == 'skip':
                        return ElementRole.SKIP
                    elif role == 'previous':
                        return ElementRole.PREVIOUS
                    elif role == 'cancel':
                        return ElementRole.CANCEL
        
        # Check OCR results near the element
        nearby_ocr_text = self._get_nearby_ocr_text(element, ocr_results)
        if nearby_ocr_text:
            nearby_text_lower = nearby_ocr_text.lower()
            
            # Check for question indicators
            if any(word in nearby_text_lower for word in ['question', '?', 'q.', 'q:']):
                return ElementRole.QUESTION
            
            # Check for navigation indicators
            if any(word in nearby_text_lower for word in ['next', 'continue', 'proceed']):
                return ElementRole.NEXT
        
        # Check element attributes for role hints
        attributes = element.attributes
        
        # Check class names and IDs
        class_name = attributes.get('class', '').lower()
        element_id = attributes.get('id', '').lower()
        
        if any(word in class_name or word in element_id for word in ['next', 'continue']):
            return ElementRole.NEXT
        elif any(word in class_name or word in element_id for word in ['submit', 'send']):
            return ElementRole.SUBMIT
        elif any(word in class_name or word in element_id for word in ['skip']):
            return ElementRole.SKIP
        elif any(word in class_name or word in element_id for word in ['prev', 'back']):
            return ElementRole.PREVIOUS
        
        # Check for form elements
        if element.element_type in [ElementType.INPUT, ElementType.SELECT, ElementType.TEXTAREA]:
            required = attributes.get('required') is not None or 'required' in class_name
            return ElementRole.REQUIRED_FIELD if required else ElementRole.OPTIONAL_FIELD
        
        # Check for answer options
        if element.element_type in [ElementType.RADIO, ElementType.CHECKBOX]:
            return ElementRole.ANSWER_OPTION
        
        return ElementRole.UNKNOWN
    
    def _get_nearby_ocr_text(self, element: ElementInfo, ocr_results: List[OCRResult], 
                           proximity_threshold: float = 50.0) -> str:
        """Get OCR text near the element."""
        element_center_x = element.bounding_box['x'] + element.bounding_box['width'] / 2
        element_center_y = element.bounding_box['y'] + element.bounding_box['height'] / 2
        
        nearby_texts = []
        
        for ocr_result in ocr_results:
            ocr_center_x = ocr_result.bounding_box['x'] + ocr_result.bounding_box['width'] / 2
            ocr_center_y = ocr_result.bounding_box['y'] + ocr_result.bounding_box['height'] / 2
            
            distance = ((element_center_x - ocr_center_x) ** 2 + (element_center_y - ocr_center_y) ** 2) ** 0.5
            
            if distance <= proximity_threshold:
                nearby_texts.append(ocr_result.text)
        
        return ' '.join(nearby_texts)
    
    async def _validate_cached_elements(self, page: Page, cached_elements: List[ElementInfo]) -> bool:
        """Validate that cached elements still exist and are valid."""
        try:
            # Check a sample of cached elements
            sample_size = min(5, len(cached_elements))
            sample_elements = cached_elements[:sample_size]
            
            for element in sample_elements:
                # Try to find the element using its selector
                handle = await page.query_selector(element.selector)
                if not handle:
                    return False
                
                # Check if bounding box is similar
                current_box = await handle.bounding_box()
                if not current_box:
                    return False
                
                # Allow some tolerance for layout changes
                if abs(current_box['x'] - element.bounding_box['x']) > 50 or \
                   abs(current_box['y'] - element.bounding_box['y']) > 50:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False
    
    async def _match_layout_pattern(self, elements: List[ElementInfo], platform: str) -> Optional[str]:
        """Try to match current layout against known patterns."""
        # Create signature of current layout
        current_signature = self._create_layout_signature(elements)
        
        best_match = None
        best_score = 0.0
        
        for pattern_id, pattern in self.layout_cache.items():
            if pattern.platform != platform and pattern.platform != "generic":
                continue
            
            # Calculate similarity score
            score = self._calculate_pattern_similarity(current_signature, pattern)
            
            if score > best_score and score > 0.7:  # Minimum similarity threshold
                best_score = score
                best_match = pattern_id
        
        if best_match:
            # Update usage statistics
            self.layout_cache[best_match].usage_count += 1
            self.layout_cache[best_match].last_used = datetime.now()
            await self._save_layout_cache()
            
            logger.info(f"Matched layout pattern {best_match} with score {best_score:.2f}")
        
        return best_match
    
    def _create_layout_signature(self, elements: List[ElementInfo]) -> Dict[str, Any]:
        """Create a signature of the current layout."""
        signature = {
            'element_count': len(elements),
            'button_count': len([e for e in elements if e.element_type == ElementType.BUTTON]),
            'input_count': len([e for e in elements if e.element_type == ElementType.INPUT]),
            'roles': [e.role.value for e in elements if e.role != ElementRole.UNKNOWN],
            'prominent_elements': [e.text_content for e in elements if e.prominence_score > 0.7]
        }
        return signature
    
    def _calculate_pattern_similarity(self, signature: Dict[str, Any], pattern: LayoutPattern) -> float:
        """Calculate similarity between current layout and a pattern."""
        score = 0.0
        
        # Compare element counts (with tolerance)
        if abs(signature['element_count'] - pattern.selectors.get('element_count', 0)) <= 5:
            score += 0.2
        
        # Compare button counts
        if abs(signature['button_count'] - pattern.selectors.get('button_count', 0)) <= 2:
            score += 0.2
        
        # Compare roles
        pattern_roles = pattern.selectors.get('roles', [])
        common_roles = set(signature['roles']).intersection(set(pattern_roles))
        if pattern_roles:
            role_similarity = len(common_roles) / len(pattern_roles)
            score += role_similarity * 0.4
        
        # Compare prominent elements text
        pattern_prominent = pattern.selectors.get('prominent_elements', [])
        if pattern_prominent:
            text_similarity = self._calculate_text_similarity(
                signature['prominent_elements'], 
                pattern_prominent
            )
            score += text_similarity * 0.2
        
        return score
    
    def _calculate_text_similarity(self, texts1: List[str], texts2: List[str]) -> float:
        """Calculate text similarity between two lists of strings."""
        if not texts1 or not texts2:
            return 0.0
        
        try:
            # Use TF-IDF vectorization for text similarity
            all_texts = texts1 + texts2
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between the two groups
            similarity_matrix = cosine_similarity(
                tfidf_matrix[:len(texts1)], 
                tfidf_matrix[len(texts1):]
            )
            
            return float(np.mean(similarity_matrix))
            
        except Exception as e:
            logger.debug(f"Text similarity calculation failed: {e}")
            return 0.0
    
    def _build_layout_analysis(self, elements: List[ElementInfo]) -> Dict[str, Any]:
        """Build comprehensive layout analysis result."""
        # Group elements by role
        elements_by_role = {}
        for element in elements:
            role = element.role.value
            if role not in elements_by_role:
                elements_by_role[role] = []
            elements_by_role[role].append(asdict(element))
        
        # Find most prominent elements
        prominent_elements = sorted(elements, key=lambda e: e.prominence_score, reverse=True)[:10]
        
        # Identify key navigation elements
        navigation_elements = {
            'next_button': self._find_best_element_by_role(elements, ElementRole.NEXT),
            'submit_button': self._find_best_element_by_role(elements, ElementRole.SUBMIT),
            'skip_button': self._find_best_element_by_role(elements, ElementRole.SKIP),
            'previous_button': self._find_best_element_by_role(elements, ElementRole.PREVIOUS)
        }
        
        # Calculate layout complexity
        complexity_score = self._calculate_layout_complexity(elements)
        
        return {
            'total_elements': len(elements),
            'elements_by_role': elements_by_role,
            'prominent_elements': [asdict(e) for e in prominent_elements],
            'navigation_elements': {k: asdict(v) if v else None for k, v in navigation_elements.items()},
            'complexity_score': complexity_score,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _find_best_element_by_role(self, elements: List[ElementInfo], role: ElementRole) -> Optional[ElementInfo]:
        """Find the best element for a specific role."""
        candidates = [e for e in elements if e.role == role and e.is_visible and e.is_enabled]
        
        if not candidates:
            return None
        
        # Return the most prominent candidate
        return max(candidates, key=lambda e: e.prominence_score)
    
    def _calculate_layout_complexity(self, elements: List[ElementInfo]) -> float:
        """Calculate layout complexity score."""
        complexity_factors = {
            'element_count': min(len(elements) / 50, 1.0),  # Normalize to 50 elements
            'element_variety': len(set(e.element_type for e in elements)) / len(ElementType),
            'z_index_layers': len(set(e.z_index for e in elements)) / 10,  # Normalize to 10 layers
            'invisible_elements': len([e for e in elements if not e.is_visible]) / max(len(elements), 1)
        }
        
        return sum(complexity_factors.values()) / len(complexity_factors)
    
    async def find_element_with_fuzzy_selector(self, page: Page, element_role: str) -> Optional[ElementHandle]:
        """Find element using fuzzy selectors."""
        if element_role not in self.fuzzy_patterns:
            logger.warning(f"No fuzzy patterns defined for role: {element_role}")
            return None
        
        patterns = self.fuzzy_patterns[element_role]
        
        for pattern in patterns:
            try:
                element = await page.query_selector(pattern)
                if element:
                    # Verify element is visible and enabled
                    is_visible = await element.is_visible()
                    is_enabled = await element.is_enabled()
                    
                    if is_visible and is_enabled:
                        logger.info(f"Found element for {element_role} using pattern: {pattern}")
                        return element
                        
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
                continue
        
        logger.warning(f"Could not find element for role {element_role} using fuzzy selectors")
        return None
    
    async def find_element_by_ocr_text(self, page: Page, target_text: str, 
                                     proximity_threshold: float = 100.0) -> Optional[ElementHandle]:
        """Find clickable element near OCR-detected text."""
        # Get page screenshot and perform OCR
        screenshot = await page.screenshot(full_page=True)
        ocr_results = await self._perform_ocr_analysis(screenshot)
        
        # Find OCR result matching target text
        target_ocr = None
        for ocr_result in ocr_results:
            if target_text.lower() in ocr_result.text.lower():
                target_ocr = ocr_result
                break
        
        if not target_ocr:
            logger.warning(f"OCR text '{target_text}' not found on page")
            return None
        
        # Find clickable elements near the OCR text
        all_elements = await page.query_selector_all('button, input, a, [onclick], [role="button"]')
        
        best_element = None
        best_distance = float('inf')
        
        for element in all_elements:
            try:
                bounding_box = await element.bounding_box()
                if not bounding_box:
                    continue
                
                # Calculate distance from OCR text center
                element_center_x = bounding_box['x'] + bounding_box['width'] / 2
                element_center_y = bounding_box['y'] + bounding_box['height'] / 2
                
                ocr_center_x = target_ocr.bounding_box['x'] + target_ocr.bounding_box['width'] / 2
                ocr_center_y = target_ocr.bounding_box['y'] + target_ocr.bounding_box['height'] / 2
                
                distance = ((element_center_x - ocr_center_x) ** 2 + (element_center_y - ocr_center_y) ** 2) ** 0.5
                
                if distance <= proximity_threshold and distance < best_distance:
                    # Verify element is clickable
                    is_visible = await element.is_visible()
                    is_enabled = await element.is_enabled()
                    
                    if is_visible and is_enabled:
                        best_distance = distance
                        best_element = element
                        
            except Exception as e:
                logger.debug(f"Error checking element near OCR text: {e}")
                continue
        
        if best_element:
            logger.info(f"Found clickable element near OCR text '{target_text}' at distance {best_distance:.1f}px")
        else:
            logger.warning(f"No clickable element found near OCR text '{target_text}'")
        
        return best_element
    
    async def cache_layout_pattern(self, elements: List[ElementInfo], platform: str, 
                                 survey_type: str, pattern_name: str):
        """Cache a successful layout pattern for future use."""
        pattern_id = hashlib.sha256(f"{platform}:{survey_type}:{pattern_name}".encode()).hexdigest()[:16]
        
        # Create pattern from current elements
        selectors = {}
        for element in elements:
            if element.role != ElementRole.UNKNOWN:
                role_key = element.role.value
                if role_key not in selectors:
                    selectors[role_key] = element.selector
        
        # Add layout signature
        signature = self._create_layout_signature(elements)
        selectors.update(signature)
        
        pattern = LayoutPattern(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            selectors=selectors,
            confidence=0.8,  # Initial confidence
            usage_count=1,
            last_used=datetime.now(),
            platform=platform,
            survey_type=survey_type
        )
        
        self.layout_cache[pattern_id] = pattern
        await self._save_layout_cache()
        
        logger.info(f"Cached layout pattern {pattern_name} for {platform}")
    
    def get_layout_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached layouts and analysis performance."""
        if not self.layout_cache:
            return {'total_patterns': 0}
        
        patterns_by_platform = {}
        total_usage = 0
        
        for pattern in self.layout_cache.values():
            platform = pattern.platform
            patterns_by_platform[platform] = patterns_by_platform.get(platform, 0) + 1
            total_usage += pattern.usage_count
        
        most_used_pattern = max(self.layout_cache.values(), key=lambda p: p.usage_count)
        
        return {
            'total_patterns': len(self.layout_cache),
            'patterns_by_platform': patterns_by_platform,
            'total_usage': total_usage,
            'average_usage': total_usage / len(self.layout_cache),
            'most_used_pattern': {
                'name': most_used_pattern.pattern_name,
                'platform': most_used_pattern.platform,
                'usage_count': most_used_pattern.usage_count
            },
            'cache_hit_rate': len(self.element_cache) / max(len(self.element_cache) + 1, 1)  # Placeholder calculation
        }