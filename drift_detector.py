"""
Drift Detection System
Monitors website changes and detects when automation needs to be updated.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
import difflib
from playwright.async_api import async_playwright
import sqlite3
import os

logger = logging.getLogger(__name__)

@dataclass
class PageSnapshot:
    """Represents a snapshot of a webpage at a specific time."""
    url: str
    timestamp: datetime
    html_hash: str
    structure_hash: str
    element_count: int
    form_elements: Dict[str, List[str]]
    key_selectors: Dict[str, bool]
    page_title: str
    meta_description: str
    scripts_count: int
    styles_count: int

@dataclass
class DriftAlert:
    """Represents a drift detection alert."""
    alert_id: str
    url: str
    timestamp: datetime
    drift_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_elements: List[str]
    suggested_actions: List[str]
    confidence_score: float

class DriftDetector:
    """Detects changes in website structure that could break automation."""
    
    def __init__(self, db_path: str = "drift_detection.db"):
        self.db_path = db_path
        self.init_database()
        self.monitored_urls = set()
        self.check_interval = 3600  # 1 hour
        self.running = False
        
    def init_database(self):
        """Initialize the drift detection database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                html_hash TEXT NOT NULL,
                structure_hash TEXT NOT NULL,
                element_count INTEGER,
                form_elements TEXT,
                key_selectors TEXT,
                page_title TEXT,
                meta_description TEXT,
                scripts_count INTEGER,
                styles_count INTEGER
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                drift_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                affected_elements TEXT,
                suggested_actions TEXT,
                confidence_score REAL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_monitored_url(self, url: str, key_selectors: Dict[str, str] = None):
        """Add a URL to the monitoring list."""
        self.monitored_urls.add(url)
        if key_selectors:
            # Store key selectors for this URL
            self._store_key_selectors(url, key_selectors)
        logger.info(f"Added {url} to drift monitoring")
        
    def _store_key_selectors(self, url: str, selectors: Dict[str, str]):
        """Store key selectors for a URL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO key_selectors (url, selectors)
            VALUES (?, ?)
        ''', (url, json.dumps(selectors)))
        
        conn.commit()
        conn.close()
        
    async def take_snapshot(self, url: str) -> PageSnapshot:
        """Take a snapshot of a webpage."""
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Navigate to page
            await page.goto(url, wait_until='networkidle')
            
            # Get page content
            html_content = await page.content()
            page_title = await page.title()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Calculate hashes
            html_hash = hashlib.md5(html_content.encode()).hexdigest()
            
            # Create structure hash (without content, just tags and attributes)
            structure_content = self._extract_structure(soup)
            structure_hash = hashlib.md5(structure_content.encode()).hexdigest()
            
            # Count elements
            element_count = len(soup.find_all())
            
            # Extract form elements
            form_elements = self._extract_form_elements(soup)
            
            # Check key selectors
            key_selectors = await self._check_key_selectors(page, url)
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ''
            
            # Count scripts and styles
            scripts_count = len(soup.find_all('script'))
            styles_count = len(soup.find_all(['style', 'link'], {'rel': 'stylesheet'}))
            
            await browser.close()
            await playwright.stop()
            
            snapshot = PageSnapshot(
                url=url,
                timestamp=datetime.now(),
                html_hash=html_hash,
                structure_hash=structure_hash,
                element_count=element_count,
                form_elements=form_elements,
                key_selectors=key_selectors,
                page_title=page_title,
                meta_description=meta_description,
                scripts_count=scripts_count,
                styles_count=styles_count
            )
            
            # Store snapshot in database
            self._store_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking snapshot of {url}: {e}")
            raise
            
    def _extract_structure(self, soup: BeautifulSoup) -> str:
        """Extract structural information from HTML."""
        structure_parts = []
        
        for element in soup.find_all():
            # Get tag name and important attributes
            tag_info = element.name
            
            # Include important attributes
            important_attrs = ['id', 'class', 'name', 'type', 'role', 'data-*']
            attrs = []
            for attr, value in element.attrs.items():
                if attr in important_attrs or attr.startswith('data-'):
                    if isinstance(value, list):
                        value = ' '.join(value)
                    attrs.append(f"{attr}='{value}'")
            
            if attrs:
                tag_info += f"[{','.join(attrs)}]"
            
            structure_parts.append(tag_info)
        
        return '\n'.join(structure_parts)
        
    def _extract_form_elements(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract form elements and their properties."""
        form_elements = {
            'forms': [],
            'inputs': [],
            'buttons': [],
            'selects': [],
            'textareas': []
        }
        
        # Extract forms
        for form in soup.find_all('form'):
            form_info = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'id': form.get('id', ''),
                'class': form.get('class', [])
            }
            form_elements['forms'].append(str(form_info))
        
        # Extract inputs
        for input_elem in soup.find_all('input'):
            input_info = {
                'type': input_elem.get('type', 'text'),
                'name': input_elem.get('name', ''),
                'id': input_elem.get('id', ''),
                'class': input_elem.get('class', [])
            }
            form_elements['inputs'].append(str(input_info))
        
        # Extract buttons
        for button in soup.find_all(['button', 'input'], {'type': ['button', 'submit']}):
            button_info = {
                'type': button.get('type', 'button'),
                'id': button.get('id', ''),
                'class': button.get('class', []),
                'text': button.get_text(strip=True)
            }
            form_elements['buttons'].append(str(button_info))
        
        # Extract selects
        for select in soup.find_all('select'):
            select_info = {
                'name': select.get('name', ''),
                'id': select.get('id', ''),
                'class': select.get('class', []),
                'options_count': len(select.find_all('option'))
            }
            form_elements['selects'].append(str(select_info))
        
        # Extract textareas
        for textarea in soup.find_all('textarea'):
            textarea_info = {
                'name': textarea.get('name', ''),
                'id': textarea.get('id', ''),
                'class': textarea.get('class', [])
            }
            form_elements['textareas'].append(str(textarea_info))
        
        return form_elements
        
    async def _check_key_selectors(self, page, url: str) -> Dict[str, bool]:
        """Check if key selectors still exist on the page."""
        key_selectors = self._get_stored_key_selectors(url)
        results = {}
        
        for selector_name, selector in key_selectors.items():
            try:
                element = await page.query_selector(selector)
                results[selector_name] = element is not None
            except Exception:
                results[selector_name] = False
        
        return results
        
    def _get_stored_key_selectors(self, url: str) -> Dict[str, str]:
        """Get stored key selectors for a URL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT selectors FROM key_selectors WHERE url = ?', (url,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}
        
    def _store_snapshot(self, snapshot: PageSnapshot):
        """Store a snapshot in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO snapshots (
                url, timestamp, html_hash, structure_hash, element_count,
                form_elements, key_selectors, page_title, meta_description,
                scripts_count, styles_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.url,
            snapshot.timestamp,
            snapshot.html_hash,
            snapshot.structure_hash,
            snapshot.element_count,
            json.dumps(snapshot.form_elements),
            json.dumps(snapshot.key_selectors),
            snapshot.page_title,
            snapshot.meta_description,
            snapshot.scripts_count,
            snapshot.styles_count
        ))
        
        conn.commit()
        conn.close()
        
    def compare_snapshots(self, url: str, limit: int = 2) -> List[DriftAlert]:
        """Compare recent snapshots and detect drift."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM snapshots 
            WHERE url = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (url, limit))
        
        snapshots = cursor.fetchall()
        conn.close()
        
        if len(snapshots) < 2:
            return []
        
        alerts = []
        current = snapshots[0]
        previous = snapshots[1]
        
        # Compare structure hash
        if current[3] != previous[3]:  # structure_hash
            alert = self._create_structure_drift_alert(url, current, previous)
            alerts.append(alert)
        
        # Compare form elements
        current_forms = json.loads(current[6])  # form_elements
        previous_forms = json.loads(previous[6])
        
        if current_forms != previous_forms:
            alert = self._create_form_drift_alert(url, current_forms, previous_forms)
            alerts.append(alert)
        
        # Compare key selectors
        current_selectors = json.loads(current[7])  # key_selectors
        previous_selectors = json.loads(previous[7])
        
        if current_selectors != previous_selectors:
            alert = self._create_selector_drift_alert(url, current_selectors, previous_selectors)
            alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
        
        return alerts
        
    def _create_structure_drift_alert(self, url: str, current: tuple, previous: tuple) -> DriftAlert:
        """Create an alert for structure drift."""
        element_diff = current[5] - previous[5]  # element_count
        
        severity = "low"
        if abs(element_diff) > 50:
            severity = "high"
        elif abs(element_diff) > 20:
            severity = "medium"
        
        description = f"Page structure changed. Element count changed by {element_diff}"
        
        return DriftAlert(
            alert_id=f"struct_{url}_{int(time.time())}",
            url=url,
            timestamp=datetime.now(),
            drift_type="structure_change",
            severity=severity,
            description=description,
            affected_elements=["page_structure"],
            suggested_actions=[
                "Review page structure changes",
                "Update element selectors",
                "Test automation scripts"
            ],
            confidence_score=0.8
        )
        
    def _create_form_drift_alert(self, url: str, current_forms: Dict, previous_forms: Dict) -> DriftAlert:
        """Create an alert for form drift."""
        affected_elements = []
        
        for form_type in current_forms:
            if current_forms[form_type] != previous_forms.get(form_type, []):
                affected_elements.append(form_type)
        
        severity = "critical" if "forms" in affected_elements else "medium"
        
        description = f"Form elements changed: {', '.join(affected_elements)}"
        
        return DriftAlert(
            alert_id=f"form_{url}_{int(time.time())}",
            url=url,
            timestamp=datetime.now(),
            drift_type="form_change",
            severity=severity,
            description=description,
            affected_elements=affected_elements,
            suggested_actions=[
                "Update form selectors",
                "Test form submission",
                "Verify input field mappings"
            ],
            confidence_score=0.9
        )
        
    def _create_selector_drift_alert(self, url: str, current_selectors: Dict, previous_selectors: Dict) -> DriftAlert:
        """Create an alert for selector drift."""
        affected_elements = []
        
        for selector_name, is_present in current_selectors.items():
            previous_present = previous_selectors.get(selector_name, True)
            if is_present != previous_present:
                affected_elements.append(selector_name)
        
        severity = "critical" if len(affected_elements) > 2 else "high"
        
        description = f"Key selectors changed: {', '.join(affected_elements)}"
        
        return DriftAlert(
            alert_id=f"selector_{url}_{int(time.time())}",
            url=url,
            timestamp=datetime.now(),
            drift_type="selector_change",
            severity=severity,
            description=description,
            affected_elements=affected_elements,
            suggested_actions=[
                "Update affected selectors",
                "Test automation workflows",
                "Implement fallback selectors"
            ],
            confidence_score=0.95
        )
        
    def _store_alert(self, alert: DriftAlert):
        """Store an alert in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO drift_alerts (
                    alert_id, url, timestamp, drift_type, severity,
                    description, affected_elements, suggested_actions, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.url,
                alert.timestamp,
                alert.drift_type,
                alert.severity,
                alert.description,
                json.dumps(alert.affected_elements),
                json.dumps(alert.suggested_actions),
                alert.confidence_score
            ))
            
            conn.commit()
        except sqlite3.IntegrityError:
            # Alert already exists
            pass
        
        conn.close()
        
    async def start_monitoring(self):
        """Start the drift monitoring process."""
        self.running = True
        logger.info("Starting drift monitoring")
        
        while self.running:
            try:
                for url in self.monitored_urls:
                    logger.info(f"Checking drift for {url}")
                    
                    # Take new snapshot
                    await self.take_snapshot(url)
                    
                    # Compare with previous snapshots
                    alerts = self.compare_snapshots(url)
                    
                    if alerts:
                        logger.warning(f"Drift detected for {url}: {len(alerts)} alerts")
                        for alert in alerts:
                            logger.warning(f"Alert: {alert.description}")
                    
                    # Wait between URLs to avoid rate limiting
                    await asyncio.sleep(5)
                
                # Wait for next check cycle
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
                
    def stop_monitoring(self):
        """Stop the drift monitoring process."""
        self.running = False
        logger.info("Stopping drift monitoring")
        
    def get_recent_alerts(self, hours: int = 24, severity: str = None) -> List[DriftAlert]:
        """Get recent drift alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM drift_alerts 
            WHERE timestamp > ? AND resolved = FALSE
        '''
        params = [since]
        
        if severity:
            query += ' AND severity = ?'
            params.append(severity)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        alerts = []
        for row in results:
            alert = DriftAlert(
                alert_id=row[1],
                url=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                drift_type=row[4],
                severity=row[5],
                description=row[6],
                affected_elements=json.loads(row[7]),
                suggested_actions=json.loads(row[8]),
                confidence_score=row[9]
            )
            alerts.append(alert)
        
        return alerts
        
    def mark_alert_resolved(self, alert_id: str):
        """Mark an alert as resolved."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE drift_alerts 
            SET resolved = TRUE 
            WHERE alert_id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked alert {alert_id} as resolved")
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get the current monitoring status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count snapshots per URL
        cursor.execute('''
            SELECT url, COUNT(*) as snapshot_count,
                   MAX(timestamp) as last_snapshot
            FROM snapshots 
            GROUP BY url
        ''')
        snapshots_info = cursor.fetchall()
        
        # Count unresolved alerts
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM drift_alerts 
            WHERE resolved = FALSE
            GROUP BY severity
        ''')
        alerts_info = cursor.fetchall()
        
        conn.close()
        
        return {
            "running": self.running,
            "monitored_urls": list(self.monitored_urls),
            "check_interval": self.check_interval,
            "snapshots": {row[0]: {"count": row[1], "last_snapshot": row[2]} for row in snapshots_info},
            "unresolved_alerts": {row[0]: row[1] for row in alerts_info}
        }