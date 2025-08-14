"""
Comprehensive Database Models
Enhanced database models for survey automation system.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import json
from typing import Dict, List, Any, Optional
from enum import Enum

Base = declarative_base()

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class Platform(Base):
    """Platform information and configuration."""
    __tablename__ = 'platforms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    base_url = Column(String(500), nullable=False)
    login_url = Column(String(500), nullable=False)
    surveys_url = Column(String(500), nullable=False)
    
    # Configuration
    config = Column(JSON, default={})
    selectors = Column(JSON, default={})
    
    # Status
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    accounts = relationship("Account", back_populates="platform")
    tasks = relationship("Task", back_populates="platform")
    drift_snapshots = relationship("DriftSnapshot", back_populates="platform")
    
    def __repr__(self):
        return f"<Platform(name='{self.name}', display_name='{self.display_name}')>"

class Account(Base):
    """User accounts for different platforms."""
    __tablename__ = 'accounts'
    
    id = Column(Integer, primary_key=True)
    platform_id = Column(Integer, ForeignKey('platforms.id'), nullable=False)
    
    # Credentials (encrypted)
    email = Column(String(255), nullable=False)
    password_hash = Column(String(500), nullable=False)  # Encrypted
    
    # Account info
    username = Column(String(100))
    display_name = Column(String(200))
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime)
    login_attempts = Column(Integer, default=0)
    
    # Profile data
    profile_data = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    platform = relationship("Platform", back_populates="accounts")
    tasks = relationship("Task", back_populates="account")
    
    # Indexes
    __table_args__ = (
        Index('idx_account_platform_email', 'platform_id', 'email'),
    )
    
    def __repr__(self):
        return f"<Account(email='{self.email}', platform='{self.platform.name if self.platform else None}')>"

class Task(Base):
    """Survey automation tasks."""
    __tablename__ = 'tasks'
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    platform_id = Column(Integer, ForeignKey('platforms.id'), nullable=False)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False)
    
    # Task details
    task_type = Column(String(50), default='survey_completion')
    survey_id = Column(String(100))
    survey_url = Column(String(1000))
    
    # Status
    status = Column(String(20), default=TaskStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    current_step = Column(String(200))
    error_message = Column(Text)
    
    # Execution details
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    estimated_duration = Column(Integer)  # seconds
    actual_duration = Column(Integer)  # seconds
    
    # Results
    questions_answered = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    completion_rate = Column(Float, default=0.0)
    
    # Retry information
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Configuration
    config = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    platform = relationship("Platform", back_populates="tasks")
    account = relationship("Account", back_populates="tasks")
    responses = relationship("SurveyResponse", back_populates="task")
    logs = relationship("TaskLog", back_populates="task")
    
    # Indexes
    __table_args__ = (
        Index('idx_task_status', 'status'),
        Index('idx_task_platform', 'platform_id'),
        Index('idx_task_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Task(id='{self.id}', status='{self.status}', platform='{self.platform.name if self.platform else None}')>"

class Survey(Base):
    """Survey metadata and information."""
    __tablename__ = 'surveys'
    
    id = Column(Integer, primary_key=True)
    platform_id = Column(Integer, ForeignKey('platforms.id'), nullable=False)
    
    # Survey details
    external_id = Column(String(200))  # Platform's survey ID
    title = Column(String(500))
    description = Column(Text)
    category = Column(String(100))
    
    # Reward information
    reward_amount = Column(Float)
    reward_currency = Column(String(10), default='USD')
    reward_type = Column(String(50))  # points, cash, gift_card
    
    # Survey characteristics
    estimated_time = Column(Integer)  # minutes
    difficulty_level = Column(String(20))
    target_audience = Column(JSON, default={})
    
    # URLs
    survey_url = Column(String(1000))
    preview_url = Column(String(1000))
    
    # Status
    is_active = Column(Boolean, default=True)
    is_available = Column(Boolean, default=True)
    
    # Statistics
    completion_rate = Column(Float, default=0.0)
    average_time = Column(Integer)  # actual completion time in minutes
    total_attempts = Column(Integer, default=0)
    successful_completions = Column(Integer, default=0)
    
    # Timestamps
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    platform = relationship("Platform")
    questions = relationship("SurveyQuestion", back_populates="survey")
    responses = relationship("SurveyResponse", back_populates="survey")
    
    # Indexes
    __table_args__ = (
        Index('idx_survey_platform_external', 'platform_id', 'external_id'),
        Index('idx_survey_active', 'is_active', 'is_available'),
    )
    
    def __repr__(self):
        return f"<Survey(title='{self.title}', platform='{self.platform.name if self.platform else None}')>"

class SurveyQuestion(Base):
    """Individual survey questions."""
    __tablename__ = 'survey_questions'
    
    id = Column(Integer, primary_key=True)
    survey_id = Column(Integer, ForeignKey('surveys.id'), nullable=False)
    
    # Question details
    question_id = Column(String(100))  # Platform's question ID
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)
    
    # Question configuration
    is_required = Column(Boolean, default=True)
    order_index = Column(Integer, default=0)
    
    # Options and validation
    options = Column(JSON, default=[])
    validation_rules = Column(JSON, default={})
    min_length = Column(Integer)
    max_length = Column(Integer)
    
    # Categorization
    category = Column(String(100))
    tags = Column(JSON, default=[])
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    survey = relationship("Survey", back_populates="questions")
    responses = relationship("SurveyResponse", back_populates="question")
    
    def __repr__(self):
        return f"<SurveyQuestion(id='{self.question_id}', type='{self.question_type}')>"

class SurveyResponse(Base):
    """Survey responses generated by the AI."""
    __tablename__ = 'survey_responses'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(50), ForeignKey('tasks.id'), nullable=False)
    survey_id = Column(Integer, ForeignKey('surveys.id'), nullable=False)
    question_id = Column(Integer, ForeignKey('survey_questions.id'), nullable=False)
    
    # Response data
    response_value = Column(Text, nullable=False)
    response_type = Column(String(50))
    
    # Generation details
    ai_model_used = Column(String(100))
    persona_used = Column(String(100))
    confidence_score = Column(Float)
    generation_method = Column(String(100))
    
    # Timing
    response_time = Column(Float)  # Time taken to generate response
    submission_time = Column(Float)  # Time taken to submit
    
    # Quality metrics
    quality_score = Column(Float)
    consistency_score = Column(Float)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)
    
    # Relationships
    task = relationship("Task", back_populates="responses")
    survey = relationship("Survey", back_populates="responses")
    question = relationship("SurveyQuestion", back_populates="responses")
    
    # Indexes
    __table_args__ = (
        Index('idx_response_task', 'task_id'),
        Index('idx_response_survey', 'survey_id'),
    )
    
    def __repr__(self):
        return f"<SurveyResponse(task_id='{self.task_id}', question_id='{self.question_id}')>"

class AIPersona(Base):
    """AI personas for response generation."""
    __tablename__ = 'ai_personas'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    
    # Persona characteristics
    age_range_min = Column(Integer)
    age_range_max = Column(Integer)
    gender = Column(String(20))
    income_range = Column(String(50))
    education_level = Column(String(100))
    location_type = Column(String(50))
    
    # Interests and values
    interests = Column(JSON, default=[])
    values = Column(JSON, default=[])
    personality_traits = Column(JSON, default=[])
    
    # Behavioral patterns
    response_patterns = Column(JSON, default={})
    shopping_habits = Column(JSON, default=[])
    media_consumption = Column(JSON, default=[])
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIPersona(name='{self.name}', display_name='{self.display_name}')>"

class ProxyServer(Base):
    """Proxy server configurations."""
    __tablename__ = 'proxy_servers'
    
    id = Column(Integer, primary_key=True)
    
    # Proxy details
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    protocol = Column(String(20), default='http')  # http, https, socks4, socks5
    
    # Authentication
    username = Column(String(100))
    password_hash = Column(String(500))  # Encrypted
    
    # Location
    country = Column(String(100))
    city = Column(String(100))
    
    # Performance metrics
    response_time = Column(Float)  # Average response time in seconds
    success_rate = Column(Float, default=0.0)
    uptime_percentage = Column(Float, default=0.0)
    
    # Usage tracking
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_working = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_checked = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_proxy_active', 'is_active', 'is_working'),
        Index('idx_proxy_performance', 'success_rate', 'response_time'),
    )
    
    def __repr__(self):
        return f"<ProxyServer(host='{self.host}', port={self.port}, protocol='{self.protocol}')>"

class DriftSnapshot(Base):
    """Website drift detection snapshots."""
    __tablename__ = 'drift_snapshots'
    
    id = Column(Integer, primary_key=True)
    platform_id = Column(Integer, ForeignKey('platforms.id'), nullable=False)
    
    # Snapshot details
    url = Column(String(1000), nullable=False)
    page_title = Column(String(500))
    
    # Hashes for change detection
    html_hash = Column(String(64), nullable=False)
    structure_hash = Column(String(64), nullable=False)
    content_hash = Column(String(64))
    
    # Page metrics
    element_count = Column(Integer)
    form_count = Column(Integer)
    input_count = Column(Integer)
    button_count = Column(Integer)
    
    # Element data
    form_elements = Column(JSON, default={})
    key_selectors = Column(JSON, default={})
    
    # Performance metrics
    load_time = Column(Float)
    page_size = Column(Integer)  # bytes
    
    # Timestamps
    captured_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    platform = relationship("Platform", back_populates="drift_snapshots")
    alerts = relationship("DriftAlert", back_populates="snapshot")
    
    # Indexes
    __table_args__ = (
        Index('idx_snapshot_platform_url', 'platform_id', 'url'),
        Index('idx_snapshot_captured', 'captured_at'),
    )
    
    def __repr__(self):
        return f"<DriftSnapshot(url='{self.url}', captured_at='{self.captured_at}')>"

class DriftAlert(Base):
    """Drift detection alerts."""
    __tablename__ = 'drift_alerts'
    
    id = Column(Integer, primary_key=True)
    snapshot_id = Column(Integer, ForeignKey('drift_snapshots.id'), nullable=False)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # structure_change, form_change, selector_change
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    
    # Description
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Affected elements
    affected_elements = Column(JSON, default=[])
    suggested_actions = Column(JSON, default=[])
    
    # Confidence and impact
    confidence_score = Column(Float, default=0.0)
    impact_score = Column(Float, default=0.0)
    
    # Status
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    snapshot = relationship("DriftSnapshot", back_populates="alerts")
    
    # Indexes
    __table_args__ = (
        Index('idx_alert_severity', 'severity', 'is_resolved'),
        Index('idx_alert_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<DriftAlert(type='{self.alert_type}', severity='{self.severity}', resolved={self.is_resolved})>"

class TaskLog(Base):
    """Detailed logs for task execution."""
    __tablename__ = 'task_logs'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(50), ForeignKey('tasks.id'), nullable=False)
    
    # Log details
    log_level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    
    # Context
    step = Column(String(200))
    component = Column(String(100))
    
    # Additional data
    metadata = Column(JSON, default={})
    stack_trace = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_log_task_level', 'task_id', 'log_level'),
        Index('idx_log_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TaskLog(task_id='{self.task_id}', level='{self.log_level}', message='{self.message[:50]}...')>"

class CaptchaSolve(Base):
    """CAPTCHA solving attempts and results."""
    __tablename__ = 'captcha_solves'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(50), ForeignKey('tasks.id'))
    
    # CAPTCHA details
    captcha_type = Column(String(50), nullable=False)
    site_key = Column(String(200))
    page_url = Column(String(1000))
    
    # Solution details
    solution = Column(Text)
    confidence_score = Column(Float)
    method_used = Column(String(100))
    
    # Timing
    solve_time = Column(Float)  # seconds
    
    # Result
    was_successful = Column(Boolean)
    error_message = Column(Text)
    
    # Service details (if used)
    service_name = Column(String(50))
    service_cost = Column(Float)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    task = relationship("Task")
    
    # Indexes
    __table_args__ = (
        Index('idx_captcha_type_success', 'captcha_type', 'was_successful'),
        Index('idx_captcha_started', 'started_at'),
    )
    
    def __repr__(self):
        return f"<CaptchaSolve(type='{self.captcha_type}', method='{self.method_used}', success={self.was_successful})>"

class SystemConfig(Base):
    """System configuration settings."""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    value_type = Column(String(20), default='string')  # string, int, float, bool, json
    
    # Metadata
    description = Column(Text)
    category = Column(String(50))
    is_sensitive = Column(Boolean, default=False)  # For passwords, API keys, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.key}', category='{self.category}')>"

# Database utility functions
class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def init_default_data(self):
        """Initialize default data."""
        session = self.get_session()
        try:
            # Add default platforms
            if not session.query(Platform).filter_by(name='swagbucks').first():
                swagbucks = Platform(
                    name='swagbucks',
                    display_name='Swagbucks',
                    base_url='https://www.swagbucks.com',
                    login_url='https://www.swagbucks.com/p/login',
                    surveys_url='https://www.swagbucks.com/surveys',
                    config={
                        'max_concurrent_tasks': 3,
                        'retry_delay': 5,
                        'timeout_seconds': 300
                    }
                )
                session.add(swagbucks)
            
            if not session.query(Platform).filter_by(name='inboxdollars').first():
                inboxdollars = Platform(
                    name='inboxdollars',
                    display_name='InboxDollars',
                    base_url='https://www.inboxdollars.com',
                    login_url='https://www.inboxdollars.com/account/signin',
                    surveys_url='https://www.inboxdollars.com/surveys',
                    config={
                        'max_concurrent_tasks': 2,
                        'retry_delay': 10,
                        'timeout_seconds': 600
                    }
                )
                session.add(inboxdollars)
            
            # Add default personas
            if not session.query(AIPersona).filter_by(name='young_professional').first():
                young_prof = AIPersona(
                    name='young_professional',
                    display_name='Young Professional',
                    age_range_min=25,
                    age_range_max=35,
                    gender='mixed',
                    income_range='$50k-$80k',
                    education_level='Bachelor\'s Degree',
                    location_type='Urban',
                    interests=['Technology', 'Career Development', 'Fitness', 'Travel'],
                    values=['Efficiency', 'Innovation', 'Work-life balance'],
                    personality_traits=['Ambitious', 'Tech-savvy', 'Social'],
                    response_patterns={
                        'length_preference': 'medium',
                        'detail_level': 'moderate',
                        'enthusiasm': 'high',
                        'brand_loyalty': 'medium'
                    }
                )
                session.add(young_prof)
            
            # Add default system config
            default_configs = [
                ('max_concurrent_tasks', '5', 'int', 'Maximum number of concurrent tasks'),
                ('default_retry_count', '3', 'int', 'Default number of retries for failed tasks'),
                ('drift_check_interval', '3600', 'int', 'Drift detection check interval in seconds'),
                ('captcha_timeout', '300', 'int', 'CAPTCHA solving timeout in seconds'),
                ('ai_model_default', 'gpt-3.5-turbo', 'string', 'Default AI model for response generation')
            ]
            
            for key, value, value_type, description in default_configs:
                if not session.query(SystemConfig).filter_by(key=key).first():
                    config = SystemConfig(
                        key=key,
                        value=value,
                        value_type=value_type,
                        description=description,
                        category='system'
                    )
                    session.add(config)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()