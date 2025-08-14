# Enhanced Survey Automation System

A comprehensive, enterprise-grade survey automation system with advanced AI-powered response generation, drift detection, CAPTCHA solving, and anti-detection capabilities.

## 🚀 Features

### Core Automation Features
- **Multi-Platform Support**: Swagbucks, InboxDollars, and extensible to other platforms
- **AI-Powered Responses**: Advanced AI with multiple personas for realistic survey responses
- **CAPTCHA Solving**: Multiple solving methods including OCR, third-party services, and human-in-the-loop
- **Anti-Detection**: Advanced browser fingerprinting, proxy rotation, and human-like behavior simulation
- **Drift Detection**: Automatic monitoring of website changes with alerting system

### Advanced Capabilities
- **Concurrency & Scalability**: Multi-task execution with queue management
- **Error Handling & Retries**: Comprehensive error handling with exponential backoff
- **Database Integration**: Full PostgreSQL support with comprehensive data models
- **Real-time Monitoring**: Prometheus metrics, Grafana dashboards, and ELK stack logging
- **Web Dashboard**: Modern web interface for task management and monitoring

### Security & Reliability
- **Encrypted Credentials**: Secure storage of user credentials
- **Proxy Support**: Built-in proxy rotation for anonymity
- **Rate Limiting**: Intelligent request throttling to avoid detection
- **Health Monitoring**: Comprehensive system health checks and alerts

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (optional, for enhanced AI responses)
- CAPTCHA solving service API key (optional, for automatic CAPTCHA solving)

## 🛠 Installation

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Surveyfiller-
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the system**
   ```bash
   docker-compose up -d
   ```

4. **Access the dashboard**
   - Main Dashboard: http://localhost:5000
   - Grafana Monitoring: http://localhost:3000 (admin/admin123)
   - Kibana Logs: http://localhost:5601
   - Portainer Management: http://localhost:9000

### Local Development Setup

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers**
   ```bash
   playwright install chromium
   ```

3. **Install Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. **Set up database**
   ```bash
   # Using PostgreSQL
   createdb survey_automation
   
   # Or use SQLite for development
   export DATABASE_URL=sqlite:///survey_automation.db
   ```

5. **Run the application**
   ```bash
   python enhanced_main.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/survey_automation

# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
AI_MODEL=gpt-3.5-turbo

# CAPTCHA Solving
CAPTCHA_API_KEY=your_2captcha_api_key_here
CAPTCHA_SERVICE=2captcha

# System Configuration
MAX_CONCURRENT_TASKS=5
BROWSER_HEADLESS=true
PROXY_ROTATION_ENABLED=false
DRIFT_CHECK_INTERVAL=3600

# Security
SECRET_KEY=your_secret_key_here
ENCRYPT_CREDENTIALS=true

# Monitoring
PROMETHEUS_ENABLED=true
LOGGING_LEVEL=INFO
```

## 📊 Usage

### Web Dashboard

1. **Access the dashboard** at http://localhost:5000
2. **Create a new task**:
   - Select platform (Swagbucks/InboxDollars)
   - Enter credentials
   - Optionally specify survey ID
   - Configure persona and settings
3. **Monitor progress** in real-time
4. **View drift alerts** and system status

### API Usage

#### Create a Task
```bash
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "swagbucks",
    "credentials": {
      "email": "your_email@example.com",
      "password": "your_password"
    },
    "survey_id": "optional_survey_id",
    "config": {
      "persona": "young_professional",
      "max_questions": 50
    }
  }'
```

#### Check Task Status
```bash
curl http://localhost:5000/api/tasks/{task_id}
```

#### Get System Status
```bash
curl http://localhost:5000/api/status
```

## 🤖 AI Personas

The system includes multiple AI personas for realistic responses:

- **Young Professional**: Tech-savvy, ambitious, moderate income
- **Middle-aged Parent**: Family-focused, practical, budget-conscious
- **College Student**: Trendy, social, limited budget
- **Retiree**: Traditional, loyal, quality-focused
- **Tech Enthusiast**: Innovation-focused, early adopter
- **Budget Conscious**: Value-focused, deal-seeking
- **Health Conscious**: Wellness-focused, organic preferences
- **Environmentalist**: Sustainability-focused, eco-friendly

## 🔍 Monitoring & Logging

### Grafana Dashboards
- System performance metrics
- Task completion rates
- Error rates and types
- Response quality scores
- CAPTCHA solving statistics

### Prometheus Metrics
- `survey_tasks_total`: Total tasks processed
- `survey_tasks_duration`: Task completion time
- `survey_questions_answered`: Questions answered
- `captcha_solve_rate`: CAPTCHA solving success rate
- `drift_alerts_total`: Drift detection alerts

## 🛡 Security Features

### Credential Protection
- AES-256 encryption for stored credentials
- Secure key management
- No plaintext password storage

### Anti-Detection
- Browser fingerprint randomization
- Human-like mouse movements and typing
- Request timing randomization
- Proxy rotation support
- User-agent rotation

## 🔧 Troubleshooting

### Common Issues

1. **Browser fails to start**
   ```bash
   # Install missing dependencies
   playwright install-deps chromium
   ```

2. **CAPTCHA solving fails**
   - Check API key configuration
   - Verify service credits
   - Enable fallback methods

3. **Database connection errors**
   - Verify DATABASE_URL format
   - Check database server status
   - Ensure proper permissions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Users are responsible for complying with all applicable laws and terms of service of the platforms they interact with. The developers are not responsible for any misuse of this software.
