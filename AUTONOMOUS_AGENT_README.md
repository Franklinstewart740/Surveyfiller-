# 🤖 Autonomous Survey Agent

A comprehensive, enterprise-grade autonomous survey completion system with advanced AI-powered reasoning, human-like behavior simulation, and robust error handling.

## 🚀 Key Features

### 🔐 Secure Authentication & Session Management
- **Environment Variable Injection**: Secure credential management via environment variables and secrets managers
- **Session Persistence**: Encrypted session storage with automatic restoration
- **CAPTCHA Queue Management**: Intelligent CAPTCHA detection and queuing system with retry logic
- **Multi-Platform Support**: Extensible authentication for various survey platforms

### 🧠 Advanced Intelligence & Reasoning
- **Short-Term Memory**: Contextual memory system that tracks previous answers for consistency
- **Answer Reasoning**: Detailed logging of decision-making process for each response
- **Survey Intent Inference**: AI-powered analysis to understand survey goals and tailor responses
- **Question Classification**: Automatic categorization of question types (demographic, behavioral, etc.)

### 🛡️ Robust Error Handling
- **Exponential Backoff**: Intelligent retry mechanisms with configurable backoff strategies
- **Response Validation**: Schema-based validation of AI responses with fallback mechanisms
- **Semantic Similarity Fallbacks**: Context-aware fallback answers using semantic analysis
- **Complexity Scoring**: Automatic survey complexity assessment to determine skip strategies

### 🎮 Live Control Panel & Dashboard
- **Real-Time Monitoring**: Live status updates with WebSocket connections
- **Manual Overrides**: Mid-survey answer adjustments and question skipping
- **CAPTCHA Management**: Visual CAPTCHA queue with retry controls
- **Survey Progress Mapping**: Interactive progress visualization with question-by-question tracking

### 🧩 Human Simulation & Behavioral Modeling
- **Persona-Based Responses**: 8 detailed personas with consistent behavioral patterns
- **Question Strategy Profiles**: Intelligent answer strategies (avoid extremes, mid-range preference, etc.)
- **Human-Like Timing**: Realistic reading, thinking, and interaction timing patterns
- **Behavioral Fingerprinting**: Rotating interaction patterns to avoid detection

### 🧭 Layout Awareness & UI Intelligence
- **Fuzzy Selectors**: Robust element detection with multiple fallback strategies
- **OCR Integration**: Text recognition for element identification when DOM parsing fails
- **Layout Caching**: Pattern recognition and caching for faster page analysis
- **DOM Prominence Scoring**: Intelligent element prioritization based on visual importance

### 🧱 Advanced DOM Intelligence
- **Semantic Element Mapping**: Automatic role detection for buttons, inputs, and navigation
- **Visibility & Z-Index Checking**: Smart element interaction based on actual visibility
- **LLM-Powered Role Prediction**: AI assistance for ambiguous element classification

### 🧼 Form Hygiene & Flow Control
- **Auto-Scroll**: Automatic scrolling to active elements before interaction
- **Retry Mechanisms**: Failed interaction recovery with alternative selectors
- **Field Validation**: Required field detection and validation before submission
- **Completion Confirmation**: Survey completion verification via redirect detection

### 📊 Comprehensive Logging & Analytics
- **Screenshot Logging**: Before/after screenshots for debugging and training
- **Performance Tracking**: Detailed metrics on speed, accuracy, and consistency
- **Agent Scoring**: Self-assessment and improvement recommendations
- **Health Monitoring**: System health checks with alerting

### 🧰 Modular Plugin Architecture
- **Survey Type Plugins**: Extensible support for new survey platforms
- **LLM Integration**: Support for multiple AI models (OpenAI, DeepSeek-R1:8b, Ollama)
- **Queue Management**: Priority-based task scheduling and execution
- **Session Recording**: Full interaction recording for debugging and training

### 🧠 Cognitive Layer Enhancements
- **Multi-Modal Analysis**: OCR + DOM + LLM triangulation for element detection
- **Curved Mouse Paths**: Natural mouse movement simulation
- **Behavioral Fingerprint Rotation**: Dynamic behavior pattern changes across sessions

## 📋 Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js (for Playwright browser installation)
- Tesseract OCR
- OpenAI API key (optional)
- DeepSeek API key (optional)

### Quick Start

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd Surveyfiller-
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Install Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **Set Environment Variables**
   ```bash
   export SWAGBUCKS_EMAIL="your_email@example.com"
   export SWAGBUCKS_PASSWORD="your_secure_password"
   export OPENAI_API_KEY="sk-your_openai_key_here"
   export DEEPSEEK_API_KEY="your_deepseek_key_here"
   ```

4. **Run the Agent**
   ```bash
   python autonomous_survey_agent.py
   ```

5. **Access Control Panel**
   - Open http://localhost:12000 in your browser
   - Monitor real-time progress and control the agent

## 🎯 Usage Examples

### Basic Usage
```python
from autonomous_survey_agent import AutonomousSurveyAgent, AgentConfig
from human_simulation import PersonaProfile

# Create configuration
config = AgentConfig(
    browser_headless=False,
    human_simulation_enabled=True,
    max_concurrent_tasks=2
)

# Initialize agent
agent = AutonomousSurveyAgent(config)

# Start agent
await agent.start_agent()

# Create survey task
task_id = await agent.create_survey_task(
    platform="swagbucks",
    persona=PersonaProfile.TECH_SAVVY_MILLENNIAL.value
)
```

### Advanced Multi-Persona Usage
```python
# Create multiple tasks with different personas
personas = [
    PersonaProfile.FRUGAL_SHOPPER.value,
    PersonaProfile.HEALTH_CONSCIOUS.value,
    PersonaProfile.ENVIRONMENTALIST.value
]

for persona in personas:
    task_id = await agent.create_survey_task(
        platform="swagbucks",
        persona=persona,
        config={
            "max_questions": 50,
            "timeout_minutes": 30
        }
    )
```

## 🎭 Available Personas

1. **Frugal Shopper**: Price-sensitive, budget-conscious, prefers value brands
2. **Tech-Savvy Millennial**: High tech comfort, environmentally conscious, premium brands
3. **Busy Parent**: Family-focused, convenience-oriented, time-constrained
4. **Health Conscious**: Wellness-focused, organic preferences, premium health products
5. **Luxury Seeker**: High income, premium brands, quality over price
6. **Environmentalist**: Sustainability-focused, eco-friendly choices
7. **Early Adopter**: Innovation-focused, new technology enthusiast
8. **Traditional Conservative**: Established brands, traditional values

## 🌐 Control Panel Features

### Real-Time Dashboard
- **System Status**: Agent state, current task, health score
- **Task Metrics**: Active, queued, completed, and failed tasks
- **Survey Progress**: Question-by-question progress with visual map
- **Performance Analytics**: Success rates, timing statistics

### Manual Controls
- **Pause/Resume**: Stop and restart survey execution
- **Answer Override**: Manually specify answers for specific questions
- **Question Skip**: Skip problematic or sensitive questions
- **Emergency Stop**: Immediate halt of all operations

### CAPTCHA Management
- **Queue Visualization**: See pending CAPTCHA challenges
- **Retry Controls**: Manual retry with cooldown management
- **Success Tracking**: CAPTCHA solving statistics

### System Monitoring
- **Error Logs**: Real-time error tracking and classification
- **Health Metrics**: System performance and reliability scores
- **Memory Usage**: Intelligence engine memory and cache statistics

## 🔧 Configuration Options

### Environment Variables
```bash
# Platform Credentials
SWAGBUCKS_EMAIL=your_email@example.com
SWAGBUCKS_PASSWORD=your_secure_password
INBOXDOLLARS_EMAIL=your_email@example.com
INBOXDOLLARS_PASSWORD=your_secure_password

# AI Configuration
OPENAI_API_KEY=sk-your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Security
SURVEY_ENCRYPTION_KEY=base64_encoded_key
SURVEY_MASTER_PASSWORD=your_master_password

# System Settings
MAX_CONCURRENT_TASKS=3
BROWSER_HEADLESS=true
ANTI_DETECTION_ENABLED=true
CONTROL_PANEL_PORT=12000

# Database
DATABASE_URL=sqlite:///surveys.db
```

### AgentConfig Options
```python
config = AgentConfig(
    # Database
    database_url="sqlite:///surveys.db",
    
    # AI Models
    openai_api_key="your_key",
    ai_model="gpt-3.5-turbo",
    deepseek_api_key="your_key",
    deepseek_model="deepseek-r1:8b",
    
    # Behavior
    human_simulation_enabled=True,
    default_persona=PersonaProfile.TECH_SAVVY_MILLENNIAL.value,
    
    # System
    max_concurrent_tasks=3,
    browser_headless=True,
    anti_detection_enabled=True,
    
    # Error Handling
    max_retries=3,
    retry_delay=5,
    
    # Monitoring
    screenshot_logging=True,
    performance_tracking=True
)
```

## 🛡️ Security Features

### Credential Protection
- **AES-256 Encryption**: All stored credentials are encrypted
- **Environment Variable Injection**: No hardcoded credentials
- **Secrets Manager Integration**: Support for AWS Secrets Manager, Azure Key Vault
- **Session Encryption**: Encrypted browser session storage

### Anti-Detection Measures
- **Browser Fingerprint Randomization**: Rotating device fingerprints
- **Human-Like Behavior**: Natural timing, mouse movements, typing patterns
- **Behavioral Fingerprint Rotation**: Changing interaction patterns
- **Proxy Support**: Optional proxy rotation for IP anonymity

### Privacy Protection
- **Local Processing**: All AI reasoning happens locally when possible
- **Minimal Data Retention**: Automatic cleanup of old data
- **Anonymized Logging**: Personal information scrubbed from logs

## 📊 Monitoring & Analytics

### Performance Metrics
- **Task Success Rate**: Percentage of successfully completed surveys
- **Average Completion Time**: Time per survey and per question
- **Error Rates**: Categorized error statistics
- **CAPTCHA Success Rate**: CAPTCHA solving effectiveness

### Intelligence Analytics
- **Answer Consistency**: Measure of response coherence
- **Persona Adherence**: How well responses match selected persona
- **Question Classification Accuracy**: AI question categorization success
- **Memory Utilization**: Short-term memory effectiveness

### System Health
- **Uptime Tracking**: System availability metrics
- **Resource Usage**: Memory and CPU utilization
- **Cache Hit Rates**: Layout and element detection cache effectiveness
- **Error Recovery**: Automatic recovery success rates

## 🔍 Troubleshooting

### Common Issues

1. **Browser Launch Failures**
   ```bash
   playwright install-deps chromium
   ```

2. **OCR Not Working**
   - Ensure Tesseract is installed and in PATH
   - Check image preprocessing settings

3. **CAPTCHA Handling**
   - Verify CAPTCHA service API keys
   - Check queue management settings

4. **Memory Issues**
   - Adjust memory cleanup intervals
   - Reduce concurrent task limits

### Debug Mode
```python
config = AgentConfig(
    browser_headless=False,  # See browser actions
    screenshot_logging=True,  # Capture screenshots
    performance_tracking=True  # Detailed metrics
)
```

### Logging Configuration
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,  # Verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🚀 Advanced Features

### Custom Personas
```python
from human_simulation import PersonaCharacteristics, TimingPattern

custom_persona = PersonaCharacteristics(
    name="Custom Persona",
    age_range=(25, 35),
    income_bracket="$50,000-$75,000",
    # ... other characteristics
)
```

### Plugin Development
```python
# Create custom survey platform plugin
class CustomPlatformHandler:
    def __init__(self, config):
        self.config = config
    
    async def login(self, credentials):
        # Custom login logic
        pass
    
    async def find_surveys(self):
        # Custom survey finding logic
        pass
```

### LLM Integration
```python
# Add custom LLM provider
from intelligence_engine import IntelligenceEngine

class CustomLLMProvider:
    async def generate_response(self, prompt):
        # Custom LLM integration
        pass
```

## 📄 API Reference

### Main Agent Class
```python
class AutonomousSurveyAgent:
    async def start_agent()
    async def stop_agent()
    async def create_survey_task(platform, persona=None, config=None)
    def get_system_status()
```

### Control Panel API
```python
# REST API Endpoints
GET /api/status              # System status
GET /api/tasks               # Task list
GET /api/captcha-queue       # CAPTCHA queue
POST /api/task-action        # Task control
POST /api/manual-override    # Answer override
```

### Intelligence Engine
```python
class IntelligenceEngine:
    async def add_memory(survey_id, question_id, content, memory_type)
    async def classify_question_type(question_text)
    async def check_answer_consistency(survey_id, question_text, answer)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### Development Setup
```bash
pip install -r requirements.txt
pip install -e .
pytest tests/
black .
flake8 .
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Users are responsible for complying with all applicable laws and terms of service of the platforms they interact with. The developers are not responsible for any misuse of this software.

## 🆘 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions for help and ideas
- **Control Panel**: Use the built-in monitoring dashboard for real-time debugging

## 🎯 Roadmap

### Upcoming Features
- [ ] Mobile survey platform support
- [ ] Advanced CAPTCHA solving integration
- [ ] Machine learning model training from interaction data
- [ ] Multi-language survey support
- [ ] Advanced proxy rotation
- [ ] Cloud deployment templates
- [ ] Performance optimization dashboard
- [ ] A/B testing framework for personas

### Long-term Goals
- [ ] Fully autonomous survey discovery
- [ ] Cross-platform session sharing
- [ ] Advanced behavioral analysis
- [ ] Real-time persona adaptation
- [ ] Distributed agent coordination
- [ ] Enterprise SSO integration