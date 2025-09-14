"""
UX Control Panel & Live Status Dashboard
Real-time monitoring, manual overrides, and comprehensive survey management interface.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from flask import Flask, request, jsonify, render_template_string, websocket
from flask_cors import CORS
import threading
import uuid

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states."""
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING = "waiting"
    ERROR = "error"
    IDLE = "idle"
    MAINTENANCE = "maintenance"

class TaskAction(Enum):
    """Available task actions."""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    SKIP_QUESTION = "skip_question"
    OVERRIDE_ANSWER = "override_answer"
    RETRY_CAPTCHA = "retry_captcha"

@dataclass
class LiveStatus:
    """Real-time system status."""
    agent_state: AgentState
    current_task_id: Optional[str]
    current_survey_id: Optional[str]
    current_question: Optional[str]
    last_action: Optional[str]
    last_action_time: datetime
    active_tasks: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    captcha_queue_size: int
    uptime: timedelta
    health_score: float

@dataclass
class SurveyProgress:
    """Survey completion progress."""
    survey_id: str
    task_id: str
    platform: str
    current_question: int
    total_questions: int
    progress_percentage: float
    estimated_time_remaining: int  # minutes
    questions_answered: List[Dict[str, Any]]
    current_step: str
    start_time: datetime
    last_activity: datetime

@dataclass
class ManualOverride:
    """Manual intervention request."""
    override_id: str
    task_id: str
    override_type: str  # "answer", "skip", "pause", "custom"
    question_id: str
    original_answer: Optional[str]
    override_answer: Optional[str]
    reason: str
    created_by: str
    created_at: datetime
    applied: bool = False

class ControlPanel:
    """Comprehensive control panel for survey automation system."""
    
    def __init__(self, automation_system, config: Dict[str, Any]):
        self.automation_system = automation_system
        self.config = config
        
        # Flask app for web interface
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Real-time data
        self.live_status = LiveStatus(
            agent_state=AgentState.IDLE,
            current_task_id=None,
            current_survey_id=None,
            current_question=None,
            last_action=None,
            last_action_time=datetime.now(),
            active_tasks=0,
            queued_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            captcha_queue_size=0,
            uptime=timedelta(0),
            health_score=1.0
        )
        
        # Survey progress tracking
        self.survey_progress: Dict[str, SurveyProgress] = {}
        
        # Manual overrides
        self.manual_overrides: List[ManualOverride] = []
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # System start time
        self.start_time = datetime.now()
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self.background_tasks = []
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup Flask routes for the control panel."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status."""
            self._update_live_status()
            return jsonify(asdict(self.live_status))
        
        @self.app.route('/api/tasks')
        def get_tasks():
            """Get all tasks with their status."""
            tasks = []
            
            # Get tasks from automation system
            if hasattr(self.automation_system, 'active_tasks'):
                for task_id, task_data in self.automation_system.active_tasks.items():
                    tasks.append({
                        'task_id': task_id,
                        'platform': task_data.get('platform', 'unknown'),
                        'status': 'active',
                        'progress': self.survey_progress.get(task_id, {}).get('progress_percentage', 0)
                    })
            
            return jsonify({'tasks': tasks})
        
        @self.app.route('/api/survey-progress/<task_id>')
        def get_survey_progress(task_id):
            """Get detailed progress for a specific survey."""
            progress = self.survey_progress.get(task_id)
            if progress:
                return jsonify(asdict(progress))
            else:
                return jsonify({'error': 'Survey not found'}), 404
        
        @self.app.route('/api/captcha-queue')
        def get_captcha_queue():
            """Get current CAPTCHA queue."""
            if hasattr(self.automation_system, 'auth_manager'):
                queue = self.automation_system.auth_manager.get_captcha_queue()
                return jsonify({'captcha_queue': queue})
            return jsonify({'captcha_queue': []})
        
        @self.app.route('/api/task-action', methods=['POST'])
        def task_action():
            """Execute task action (pause, resume, cancel, etc.)."""
            data = request.get_json()
            task_id = data.get('task_id')
            action = data.get('action')
            
            if not task_id or not action:
                return jsonify({'error': 'Missing task_id or action'}), 400
            
            try:
                result = self._execute_task_action(task_id, TaskAction(action), data)
                return jsonify({'success': True, 'result': result})
            except Exception as e:
                logger.error(f"Task action failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/manual-override', methods=['POST'])
        def create_manual_override():
            """Create manual override for a question."""
            data = request.get_json()
            
            override = ManualOverride(
                override_id=str(uuid.uuid4()),
                task_id=data.get('task_id'),
                override_type=data.get('override_type'),
                question_id=data.get('question_id'),
                original_answer=data.get('original_answer'),
                override_answer=data.get('override_answer'),
                reason=data.get('reason', ''),
                created_by=data.get('created_by', 'user'),
                created_at=datetime.now()
            )
            
            self.manual_overrides.append(override)
            
            # Apply override immediately if task is active
            self._apply_manual_override(override)
            
            return jsonify({'success': True, 'override_id': override.override_id})
        
        @self.app.route('/api/health-check')
        def health_check():
            """System health check endpoint."""
            health_data = self._calculate_health_metrics()
            return jsonify(health_data)
        
        @self.app.route('/api/system-logs')
        def get_system_logs():
            """Get recent system logs."""
            # This would integrate with the logging system
            logs = []
            
            # Get error history if available
            if hasattr(self.automation_system, 'error_handler'):
                error_stats = self.automation_system.error_handler.get_error_statistics()
                logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'message': f"Error statistics: {error_stats}"
                })
            
            return jsonify({'logs': logs})
        
        @self.app.websocket('/ws/live-updates')
        def websocket_live_updates():
            """WebSocket endpoint for real-time updates."""
            self.websocket_connections.add(websocket)
            try:
                while True:
                    # Send periodic updates
                    self._update_live_status()
                    update_data = {
                        'type': 'status_update',
                        'data': asdict(self.live_status)
                    }
                    websocket.send(json.dumps(update_data))
                    time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for the dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Automation Control Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #2c3e50; margin-bottom: 1rem; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background: #27ae60; }
        .status-paused { background: #f39c12; }
        .status-error { background: #e74c3c; }
        .status-idle { background: #95a5a6; }
        .progress-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; margin: 0.5rem 0; }
        .progress-fill { height: 100%; background: #3498db; transition: width 0.3s ease; }
        .btn { padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; margin: 0.25rem; }
        .btn-primary { background: #3498db; color: white; }
        .btn-warning { background: #f39c12; color: white; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .metric { text-align: center; padding: 1rem; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 0.9rem; }
        .log-entry { padding: 0.5rem; border-left: 3px solid #3498db; margin: 0.5rem 0; background: #f8f9fa; }
        .captcha-item { padding: 1rem; border: 1px solid #ddd; border-radius: 4px; margin: 0.5rem 0; }
        .survey-map { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
        .question-dot { width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; color: white; }
        .question-completed { background: #27ae60; }
        .question-current { background: #3498db; }
        .question-pending { background: #95a5a6; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Survey Automation Control Panel</h1>
        <p>Real-time monitoring and control for autonomous survey completion</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h3>🔄 System Status</h3>
                <div id="system-status">
                    <p><span class="status-indicator status-idle"></span><span id="agent-state">Loading...</span></p>
                    <p><strong>Current Task:</strong> <span id="current-task">None</span></p>
                    <p><strong>Last Action:</strong> <span id="last-action">None</span></p>
                    <p><strong>Uptime:</strong> <span id="uptime">0h 0m</span></p>
                    <p><strong>Health Score:</strong> <span id="health-score">100%</span></p>
                </div>
            </div>
            
            <!-- Task Metrics -->
            <div class="card">
                <h3>📊 Task Metrics</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div class="metric">
                        <div class="metric-value" id="active-tasks">0</div>
                        <div class="metric-label">Active</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="queued-tasks">0</div>
                        <div class="metric-label">Queued</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="completed-tasks">0</div>
                        <div class="metric-label">Completed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="failed-tasks">0</div>
                        <div class="metric-label">Failed</div>
                    </div>
                </div>
            </div>
            
            <!-- Survey Progress -->
            <div class="card">
                <h3>📋 Current Survey Progress</h3>
                <div id="survey-progress">
                    <p>No active survey</p>
                </div>
            </div>
            
            <!-- CAPTCHA Queue -->
            <div class="card">
                <h3>🔐 CAPTCHA Queue</h3>
                <div id="captcha-queue">
                    <p>No pending CAPTCHAs</p>
                </div>
            </div>
            
            <!-- Manual Controls -->
            <div class="card">
                <h3>🎮 Manual Controls</h3>
                <div>
                    <button class="btn btn-primary" onclick="pauseSystem()">⏸️ Pause System</button>
                    <button class="btn btn-success" onclick="resumeSystem()">▶️ Resume System</button>
                    <button class="btn btn-warning" onclick="skipQuestion()">⏭️ Skip Question</button>
                    <button class="btn btn-danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
                </div>
                <div style="margin-top: 1rem;">
                    <h4>Override Answer</h4>
                    <input type="text" id="override-answer" placeholder="Enter override answer" style="width: 100%; padding: 0.5rem; margin: 0.5rem 0;">
                    <button class="btn btn-primary" onclick="overrideAnswer()">Apply Override</button>
                </div>
            </div>
            
            <!-- System Logs -->
            <div class="card">
                <h3>📝 Recent Logs</h3>
                <div id="system-logs" style="max-height: 300px; overflow-y: auto;">
                    <p>Loading logs...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let currentTaskId = null;
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/live-updates`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateDashboard(data.data);
                }
            };
            
            ws.onclose = function() {
                setTimeout(initWebSocket, 5000); // Reconnect after 5 seconds
            };
        }
        
        function updateDashboard(status) {
            // Update system status
            document.getElementById('agent-state').textContent = status.agent_state;
            document.getElementById('current-task').textContent = status.current_task_id || 'None';
            document.getElementById('last-action').textContent = status.last_action || 'None';
            document.getElementById('health-score').textContent = Math.round(status.health_score * 100) + '%';
            
            // Update task metrics
            document.getElementById('active-tasks').textContent = status.active_tasks;
            document.getElementById('queued-tasks').textContent = status.queued_tasks;
            document.getElementById('completed-tasks').textContent = status.completed_tasks;
            document.getElementById('failed-tasks').textContent = status.failed_tasks;
            
            currentTaskId = status.current_task_id;
            
            // Update status indicator
            const indicator = document.querySelector('.status-indicator');
            indicator.className = 'status-indicator status-' + status.agent_state;
        }
        
        function pauseSystem() {
            if (currentTaskId) {
                fetch('/api/task-action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task_id: currentTaskId, action: 'pause'})
                });
            }
        }
        
        function resumeSystem() {
            if (currentTaskId) {
                fetch('/api/task-action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task_id: currentTaskId, action: 'resume'})
                });
            }
        }
        
        function skipQuestion() {
            if (currentTaskId) {
                fetch('/api/task-action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task_id: currentTaskId, action: 'skip_question'})
                });
            }
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to stop all tasks?')) {
                if (currentTaskId) {
                    fetch('/api/task-action', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({task_id: currentTaskId, action: 'cancel'})
                    });
                }
            }
        }
        
        function overrideAnswer() {
            const answer = document.getElementById('override-answer').value;
            if (answer && currentTaskId) {
                fetch('/api/manual-override', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        task_id: currentTaskId,
                        override_type: 'answer',
                        question_id: 'current',
                        override_answer: answer,
                        reason: 'Manual override from control panel'
                    })
                });
                document.getElementById('override-answer').value = '';
            }
        }
        
        // Load initial data
        function loadInitialData() {
            fetch('/api/status').then(r => r.json()).then(updateDashboard);
            
            // Load CAPTCHA queue
            fetch('/api/captcha-queue').then(r => r.json()).then(data => {
                const container = document.getElementById('captcha-queue');
                if (data.captcha_queue.length === 0) {
                    container.innerHTML = '<p>No pending CAPTCHAs</p>';
                } else {
                    container.innerHTML = data.captcha_queue.map(captcha => 
                        `<div class="captcha-item">
                            <strong>${captcha.captcha_type}</strong> - ${captcha.platform}
                            <br>Created: ${new Date(captcha.created_at).toLocaleString()}
                            <br>Retries: ${captcha.retry_count}
                        </div>`
                    ).join('');
                }
            });
            
            // Load system logs
            fetch('/api/system-logs').then(r => r.json()).then(data => {
                const container = document.getElementById('system-logs');
                if (data.logs.length === 0) {
                    container.innerHTML = '<p>No recent logs</p>';
                } else {
                    container.innerHTML = data.logs.map(log => 
                        `<div class="log-entry">
                            <strong>${log.level}</strong> ${new Date(log.timestamp).toLocaleTimeString()}: ${log.message}
                        </div>`
                    ).join('');
                }
            });
        }
        
        // Initialize
        initWebSocket();
        loadInitialData();
        setInterval(loadInitialData, 10000); // Refresh every 10 seconds
    </script>
</body>
</html>
        """
    
    def _update_live_status(self):
        """Update live status with current system information."""
        # Update uptime
        self.live_status.uptime = datetime.now() - self.start_time
        
        # Update task counts
        if hasattr(self.automation_system, 'active_tasks'):
            self.live_status.active_tasks = len(self.automation_system.active_tasks)
        
        if hasattr(self.automation_system, 'task_queue'):
            self.live_status.queued_tasks = self.automation_system.task_queue.qsize()
        
        if hasattr(self.automation_system, 'stats'):
            stats = self.automation_system.stats
            self.live_status.completed_tasks = stats.get('tasks_completed', 0)
            self.live_status.failed_tasks = stats.get('tasks_failed', 0)
        
        # Update CAPTCHA queue size
        if hasattr(self.automation_system, 'auth_manager'):
            captcha_queue = self.automation_system.auth_manager.get_captcha_queue()
            self.live_status.captcha_queue_size = len(captcha_queue)
        
        # Calculate health score
        self.live_status.health_score = self._calculate_health_score()
        
        # Update agent state based on system status
        if self.live_status.active_tasks > 0:
            self.live_status.agent_state = AgentState.ACTIVE
        elif self.live_status.queued_tasks > 0:
            self.live_status.agent_state = AgentState.WAITING
        else:
            self.live_status.agent_state = AgentState.IDLE
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score."""
        health_factors = []
        
        # Error rate factor
        if hasattr(self.automation_system, 'error_handler'):
            error_stats = self.automation_system.error_handler.get_error_statistics(24)
            error_rate = error_stats.get('error_rate', 0)
            error_factor = max(0, 1 - (error_rate / 10))  # Normalize to 10 errors per hour
            health_factors.append(error_factor)
        
        # Task success rate factor
        total_tasks = self.live_status.completed_tasks + self.live_status.failed_tasks
        if total_tasks > 0:
            success_rate = self.live_status.completed_tasks / total_tasks
            health_factors.append(success_rate)
        else:
            health_factors.append(1.0)  # No tasks yet, assume healthy
        
        # CAPTCHA queue factor
        captcha_factor = max(0, 1 - (self.live_status.captcha_queue_size / 10))  # Normalize to 10 CAPTCHAs
        health_factors.append(captcha_factor)
        
        # System responsiveness factor (placeholder)
        health_factors.append(1.0)  # Would measure actual response times
        
        return sum(health_factors) / len(health_factors) if health_factors else 1.0
    
    def _calculate_health_metrics(self) -> Dict[str, Any]:
        """Calculate detailed health metrics."""
        metrics = {
            'overall_health': self._calculate_health_score(),
            'uptime_hours': self.live_status.uptime.total_seconds() / 3600,
            'active_tasks': self.live_status.active_tasks,
            'error_rate': 0.0,
            'success_rate': 0.0,
            'captcha_queue_size': self.live_status.captcha_queue_size,
            'memory_usage': 0.0,  # Would implement actual memory monitoring
            'cpu_usage': 0.0      # Would implement actual CPU monitoring
        }
        
        # Calculate success rate
        total_tasks = self.live_status.completed_tasks + self.live_status.failed_tasks
        if total_tasks > 0:
            metrics['success_rate'] = self.live_status.completed_tasks / total_tasks
        
        # Get error rate
        if hasattr(self.automation_system, 'error_handler'):
            error_stats = self.automation_system.error_handler.get_error_statistics(24)
            metrics['error_rate'] = error_stats.get('error_rate', 0)
        
        return metrics
    
    def _execute_task_action(self, task_id: str, action: TaskAction, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task action."""
        result = {'action': action.value, 'task_id': task_id, 'success': False}
        
        try:
            if action == TaskAction.PAUSE:
                # Implement pause logic
                result['message'] = f"Task {task_id} paused"
                result['success'] = True
                
            elif action == TaskAction.RESUME:
                # Implement resume logic
                result['message'] = f"Task {task_id} resumed"
                result['success'] = True
                
            elif action == TaskAction.CANCEL:
                # Implement cancel logic
                if hasattr(self.automation_system, 'active_tasks') and task_id in self.automation_system.active_tasks:
                    # Would cancel the actual task
                    result['message'] = f"Task {task_id} cancelled"
                    result['success'] = True
                
            elif action == TaskAction.SKIP_QUESTION:
                # Implement skip question logic
                result['message'] = f"Skipped current question for task {task_id}"
                result['success'] = True
                
            elif action == TaskAction.RETRY_CAPTCHA:
                # Implement CAPTCHA retry logic
                captcha_id = data.get('captcha_id')
                if captcha_id and hasattr(self.automation_system, 'auth_manager'):
                    # Would retry the CAPTCHA
                    result['message'] = f"Retrying CAPTCHA {captcha_id}"
                    result['success'] = True
            
            # Update last action
            self.live_status.last_action = f"{action.value} on {task_id}"
            self.live_status.last_action_time = datetime.now()
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Task action failed: {e}")
        
        return result
    
    def _apply_manual_override(self, override: ManualOverride):
        """Apply a manual override to an active task."""
        try:
            # This would integrate with the actual task execution
            # For now, just mark as applied
            override.applied = True
            
            logger.info(f"Applied manual override {override.override_id} for task {override.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to apply manual override: {e}")
    
    def update_survey_progress(self, task_id: str, survey_id: str, platform: str,
                             current_question: int, total_questions: int,
                             current_step: str, questions_answered: List[Dict[str, Any]] = None):
        """Update survey progress information."""
        progress_percentage = (current_question / max(total_questions, 1)) * 100
        
        # Estimate time remaining based on average time per question
        avg_time_per_question = 1.5  # minutes
        remaining_questions = max(0, total_questions - current_question)
        estimated_time_remaining = int(remaining_questions * avg_time_per_question)
        
        progress = SurveyProgress(
            survey_id=survey_id,
            task_id=task_id,
            platform=platform,
            current_question=current_question,
            total_questions=total_questions,
            progress_percentage=progress_percentage,
            estimated_time_remaining=estimated_time_remaining,
            questions_answered=questions_answered or [],
            current_step=current_step,
            start_time=self.survey_progress.get(task_id, {}).get('start_time', datetime.now()),
            last_activity=datetime.now()
        )
        
        self.survey_progress[task_id] = progress
        
        # Update live status
        self.live_status.current_task_id = task_id
        self.live_status.current_survey_id = survey_id
        self.live_status.current_question = f"Question {current_question} of {total_questions}"
        
        # Broadcast update to WebSocket connections
        self._broadcast_update('survey_progress', asdict(progress))
    
    def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to all WebSocket connections."""
        message = json.dumps({
            'type': update_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Remove closed connections
        closed_connections = set()
        for conn in self.websocket_connections:
            try:
                conn.send(message)
            except Exception:
                closed_connections.add(conn)
        
        self.websocket_connections -= closed_connections
    
    def _start_background_tasks(self):
        """Start background tasks for the control panel."""
        def status_updater():
            while True:
                try:
                    self._update_live_status()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Status updater error: {e}")
                    time.sleep(10)
        
        def cleanup_old_data():
            while True:
                try:
                    # Clean up old survey progress data
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    
                    old_surveys = [
                        task_id for task_id, progress in self.survey_progress.items()
                        if progress.last_activity < cutoff_time
                    ]
                    
                    for task_id in old_surveys:
                        del self.survey_progress[task_id]
                    
                    # Clean up old manual overrides
                    cutoff_time = datetime.now() - timedelta(hours=6)
                    self.manual_overrides = [
                        override for override in self.manual_overrides
                        if override.created_at > cutoff_time
                    ]
                    
                    time.sleep(3600)  # Clean up every hour
                    
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    time.sleep(1800)  # Wait 30 minutes on error
        
        # Start background threads
        status_thread = threading.Thread(target=status_updater, daemon=True)
        cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
        
        status_thread.start()
        cleanup_thread.start()
        
        self.background_tasks = [status_thread, cleanup_thread]
    
    def start_server(self, host: str = '0.0.0.0', port: int = 12000):
        """Start the control panel web server."""
        logger.info(f"Starting control panel server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False, threaded=True)
    
    def get_manual_overrides(self, task_id: str = None) -> List[ManualOverride]:
        """Get manual overrides, optionally filtered by task ID."""
        if task_id:
            return [override for override in self.manual_overrides if override.task_id == task_id]
        return self.manual_overrides.copy()
    
    def get_survey_map(self, task_id: str) -> Dict[str, Any]:
        """Get survey map with progress indicators."""
        progress = self.survey_progress.get(task_id)
        if not progress:
            return {}
        
        questions = []
        for i in range(1, progress.total_questions + 1):
            if i < progress.current_question:
                status = 'completed'
            elif i == progress.current_question:
                status = 'current'
            else:
                status = 'pending'
            
            questions.append({
                'question_number': i,
                'status': status,
                'answered': i <= len(progress.questions_answered)
            })
        
        return {
            'survey_id': progress.survey_id,
            'total_questions': progress.total_questions,
            'current_question': progress.current_question,
            'progress_percentage': progress.progress_percentage,
            'questions': questions
        }