#!/usr/bin/env python3
"""
Startup Script for Autonomous Survey Agent
Easy-to-use launcher with configuration validation and setup assistance.
"""

import asyncio
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_survey_agent import AutonomousSurveyAgent, AgentConfig
from human_simulation import PersonaProfile

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('autonomous_survey_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_environment() -> Dict[str, Any]:
    """Check environment setup and return status."""
    status = {
        'credentials': {},
        'api_keys': {},
        'system': {},
        'warnings': [],
        'errors': []
    }
    
    # Check platform credentials
    platforms = ['SWAGBUCKS', 'INBOXDOLLARS']
    for platform in platforms:
        email = os.getenv(f'{platform}_EMAIL')
        password = os.getenv(f'{platform}_PASSWORD')
        
        if email and password:
            status['credentials'][platform.lower()] = '✅ Configured'
        else:
            status['credentials'][platform.lower()] = '❌ Missing'
            status['warnings'].append(f"No credentials found for {platform}")
    
    # Check API keys
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI GPT',
        'DEEPSEEK_API_KEY': 'DeepSeek R1',
        'CAPTCHA_API_KEY': '2CAPTCHA Service'
    }
    
    for key, service in api_keys.items():
        if os.getenv(key):
            status['api_keys'][service] = '✅ Configured'
        else:
            status['api_keys'][service] = '⚠️  Optional'
    
    # Check system requirements
    try:
        import playwright
        status['system']['Playwright'] = '✅ Installed'
    except ImportError:
        status['system']['Playwright'] = '❌ Missing'
        status['errors'].append("Playwright not installed. Run: pip install playwright && playwright install chromium")
    
    try:
        import pytesseract
        status['system']['Tesseract OCR'] = '✅ Installed'
    except ImportError:
        status['system']['Tesseract OCR'] = '❌ Missing'
        status['errors'].append("Tesseract OCR not installed. See installation instructions in README")
    
    try:
        import cv2
        status['system']['OpenCV'] = '✅ Installed'
    except ImportError:
        status['system']['OpenCV'] = '❌ Missing'
        status['errors'].append("OpenCV not installed. Run: pip install opencv-python")
    
    return status

def print_status_report(status: Dict[str, Any]):
    """Print environment status report."""
    print("\n🔍 Environment Status Report")
    print("=" * 50)
    
    print("\n📧 Platform Credentials:")
    for platform, stat in status['credentials'].items():
        print(f"   {platform.title()}: {stat}")
    
    print("\n🔑 API Keys:")
    for service, stat in status['api_keys'].items():
        print(f"   {service}: {stat}")
    
    print("\n🛠️  System Requirements:")
    for requirement, stat in status['system'].items():
        print(f"   {requirement}: {stat}")
    
    if status['warnings']:
        print("\n⚠️  Warnings:")
        for warning in status['warnings']:
            print(f"   • {warning}")
    
    if status['errors']:
        print("\n❌ Errors:")
        for error in status['errors']:
            print(f"   • {error}")
        print("\n   Please fix these errors before starting the agent.")
        return False
    
    return True

def create_sample_env_file():
    """Create a sample .env file."""
    env_content = """# Autonomous Survey Agent Configuration

# Platform Credentials (Required)
SWAGBUCKS_EMAIL=your_email@example.com
SWAGBUCKS_PASSWORD=your_secure_password
INBOXDOLLARS_EMAIL=your_email@example.com
INBOXDOLLARS_PASSWORD=your_secure_password

# AI API Keys (Optional but recommended)
OPENAI_API_KEY=sk-your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Security (Optional - will be generated if not provided)
SURVEY_ENCRYPTION_KEY=base64_encoded_encryption_key
SURVEY_MASTER_PASSWORD=your_master_password_here
SURVEY_SALT=your_salt_here

# System Configuration
MAX_CONCURRENT_TASKS=3
BROWSER_HEADLESS=true
ANTI_DETECTION_ENABLED=true
HUMAN_SIMULATION_ENABLED=true
CONTROL_PANEL_PORT=12000

# Database
DATABASE_URL=sqlite:///autonomous_survey_agent.db

# Logging
LOGGING_LEVEL=INFO
SCREENSHOT_LOGGING=true
PERFORMANCE_TRACKING=true

# CAPTCHA Solving (Optional)
CAPTCHA_API_KEY=your_2captcha_api_key_here
CAPTCHA_SERVICE=2captcha
"""
    
    env_file = Path('.env.example')
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"📝 Created sample environment file: {env_file}")
    print("   Copy this to .env and fill in your credentials")

def interactive_setup():
    """Interactive setup wizard."""
    print("\n🧙 Interactive Setup Wizard")
    print("=" * 30)
    
    config = {}
    
    # Platform selection
    print("\n📋 Which survey platforms do you want to use?")
    platforms = ['swagbucks', 'inboxdollars']
    
    for platform in platforms:
        use_platform = input(f"Use {platform.title()}? (y/n): ").lower().startswith('y')
        if use_platform:
            email = input(f"Enter {platform.title()} email: ")
            password = input(f"Enter {platform.title()} password: ")
            
            os.environ[f'{platform.upper()}_EMAIL'] = email
            os.environ[f'{platform.upper()}_PASSWORD'] = password
            
            config[f'{platform}_credentials'] = True
    
    # Persona selection
    print("\n🎭 Select default persona:")
    personas = list(PersonaProfile)
    for i, persona in enumerate(personas, 1):
        print(f"   {i}. {persona.value.replace('_', ' ').title()}")
    
    try:
        choice = int(input("Enter persona number (1-8): ")) - 1
        if 0 <= choice < len(personas):
            config['default_persona'] = personas[choice].value
        else:
            config['default_persona'] = PersonaProfile.TECH_SAVVY_MILLENNIAL.value
    except ValueError:
        config['default_persona'] = PersonaProfile.TECH_SAVVY_MILLENNIAL.value
    
    # Browser mode
    headless = input("\nRun browser in headless mode? (y/n): ").lower().startswith('y')
    config['browser_headless'] = headless
    
    # Concurrent tasks
    try:
        max_tasks = int(input("Maximum concurrent tasks (1-5): "))
        config['max_concurrent_tasks'] = max(1, min(5, max_tasks))
    except ValueError:
        config['max_concurrent_tasks'] = 3
    
    return config

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(
        description="Autonomous Survey Agent - Intelligent Survey Completion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_agent.py                    # Interactive startup
  python start_agent.py --headless        # Run in headless mode
  python start_agent.py --setup           # Run setup wizard
  python start_agent.py --check-env       # Check environment only
  python start_agent.py --create-env      # Create sample .env file
        """
    )
    
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    parser.add_argument('--setup', action='store_true',
                       help='Run interactive setup wizard')
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment and exit')
    parser.add_argument('--create-env', action='store_true',
                       help='Create sample .env file and exit')
    parser.add_argument('--persona', choices=[p.value for p in PersonaProfile],
                       default=PersonaProfile.TECH_SAVVY_MILLENNIAL.value,
                       help='Default persona to use')
    parser.add_argument('--max-tasks', type=int, default=3,
                       help='Maximum concurrent tasks (1-5)')
    parser.add_argument('--port', type=int, default=12000,
                       help='Control panel port')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--database-url', default='sqlite:///autonomous_survey_agent.db',
                       help='Database URL')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🤖 Autonomous Survey Agent")
    print("=" * 50)
    print("Comprehensive AI-powered survey automation system")
    print("with human-like behavior and intelligent reasoning")
    print("=" * 50)
    
    # Handle special commands
    if args.create_env:
        create_sample_env_file()
        return
    
    # Check environment
    status = check_environment()
    
    if args.check_env:
        print_status_report(status)
        return
    
    # Print status report
    env_ok = print_status_report(status)
    
    if not env_ok:
        print("\n❌ Environment check failed. Please fix the errors above.")
        return
    
    # Interactive setup if requested
    interactive_config = {}
    if args.setup:
        interactive_config = interactive_setup()
    
    # Create configuration
    config = AgentConfig(
        database_url=args.database_url,
        browser_headless=args.headless or interactive_config.get('browser_headless', True),
        default_persona=interactive_config.get('default_persona', args.persona),
        max_concurrent_tasks=interactive_config.get('max_concurrent_tasks', args.max_tasks),
        control_panel_port=args.port,
        human_simulation_enabled=True,
        anti_detection_enabled=True,
        screenshot_logging=True,
        performance_tracking=True
    )
    
    print(f"\n🚀 Starting Autonomous Survey Agent")
    print(f"   Database: {config.database_url}")
    print(f"   Browser Mode: {'Headless' if config.browser_headless else 'Visible'}")
    print(f"   Default Persona: {config.default_persona.replace('_', ' ').title()}")
    print(f"   Max Concurrent Tasks: {config.max_concurrent_tasks}")
    print(f"   Control Panel: http://localhost:{config.control_panel_port}")
    
    # Start the agent
    async def run_agent():
        agent = AutonomousSurveyAgent(config)
        
        try:
            await agent.start_agent()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"\n❌ Agent failed: {e}")
            logging.exception("Agent startup failed")
        finally:
            await agent.stop_agent()
    
    # Run the agent
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Startup failed: {e}")
        logging.exception("Startup failed")

if __name__ == "__main__":
    main()