"""
Example Usage of Autonomous Survey Agent
Demonstrates how to use the comprehensive survey automation system.
"""

import asyncio
import os
import logging
from autonomous_survey_agent import AutonomousSurveyAgent, AgentConfig
from human_simulation import PersonaProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_usage():
    """Basic usage example."""
    print("🤖 Autonomous Survey Agent - Basic Usage Example")
    
    # Create configuration
    config = AgentConfig(
        database_url="sqlite:///example_surveys.db",
        browser_headless=False,  # Set to True for headless operation
        human_simulation_enabled=True,
        max_concurrent_tasks=2,
        control_panel_port=12000
    )
    
    # Create agent
    agent = AutonomousSurveyAgent(config)
    
    try:
        # Start agent in background
        agent_task = asyncio.create_task(agent.start_agent())
        
        # Wait a moment for agent to initialize
        await asyncio.sleep(2)
        
        # Create a survey task
        task_id = await agent.create_survey_task(
            platform="swagbucks",
            persona=PersonaProfile.TECH_SAVVY_MILLENNIAL.value,
            config={
                "max_questions": 50,
                "timeout_minutes": 30
            }
        )
        
        print(f"✅ Created survey task: {task_id}")
        print(f"🌐 Control panel available at: http://localhost:{config.control_panel_port}")
        
        # Let it run for a while
        await asyncio.sleep(60)  # Run for 1 minute in this example
        
    except KeyboardInterrupt:
        print("🛑 Stopping agent...")
    finally:
        await agent.stop_agent()

async def example_advanced_usage():
    """Advanced usage with multiple personas and error handling."""
    print("🚀 Autonomous Survey Agent - Advanced Usage Example")
    
    # Set environment variables for secure credentials
    os.environ['SWAGBUCKS_EMAIL'] = 'your_email@example.com'
    os.environ['SWAGBUCKS_PASSWORD'] = 'your_secure_password'
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
    
    config = AgentConfig(
        database_url="sqlite:///advanced_surveys.db",
        browser_headless=True,
        human_simulation_enabled=True,
        max_concurrent_tasks=3,
        control_panel_port=12001,
        screenshot_logging=True,
        performance_tracking=True
    )
    
    agent = AutonomousSurveyAgent(config)
    
    try:
        # Start agent
        agent_task = asyncio.create_task(agent.start_agent())
        await asyncio.sleep(3)
        
        # Create multiple tasks with different personas
        personas = [
            PersonaProfile.FRUGAL_SHOPPER.value,
            PersonaProfile.TECH_SAVVY_MILLENNIAL.value,
            PersonaProfile.HEALTH_CONSCIOUS.value
        ]
        
        tasks = []
        for i, persona in enumerate(personas):
            task_id = await agent.create_survey_task(
                platform="swagbucks",
                persona=persona,
                config={
                    "max_questions": 30,
                    "timeout_minutes": 20,
                    "priority": i + 1
                }
            )
            tasks.append(task_id)
            print(f"✅ Created task {task_id} with persona {persona}")
        
        print(f"🌐 Advanced control panel: http://localhost:{config.control_panel_port}")
        print("📊 Monitor progress, handle CAPTCHAs, and override answers in real-time")
        
        # Monitor system status
        for i in range(10):  # Check status 10 times
            await asyncio.sleep(30)  # Wait 30 seconds between checks
            
            status = agent.get_system_status()
            print(f"📈 Status Update {i+1}:")
            print(f"   Active Tasks: {status['active_tasks']}")
            print(f"   Completed: {status['stats']['tasks_completed']}")
            print(f"   Failed: {status['stats']['tasks_failed']}")
            print(f"   Questions Answered: {status['stats']['total_questions_answered']}")
            print(f"   Performance Score: {status['stats']['agent_performance_score']:.2f}")
            
            if status['active_tasks'] == 0 and status['queued_tasks'] == 0:
                print("✅ All tasks completed!")
                break
        
    except Exception as e:
        logger.error(f"Error in advanced example: {e}")
    finally:
        await agent.stop_agent()

async def example_control_panel_integration():
    """Example showing control panel integration."""
    print("🎮 Control Panel Integration Example")
    
    config = AgentConfig(
        control_panel_port=12002,
        browser_headless=False  # So you can see the automation
    )
    
    agent = AutonomousSurveyAgent(config)
    
    try:
        # Start agent
        agent_task = asyncio.create_task(agent.start_agent())
        await asyncio.sleep(2)
        
        # Create a task
        task_id = await agent.create_survey_task(
            platform="swagbucks",
            persona=PersonaProfile.BUSY_PARENT.value
        )
        
        print(f"🎯 Task created: {task_id}")
        print(f"🌐 Control Panel: http://localhost:{config.control_panel_port}")
        print("\n📋 Available Control Panel Features:")
        print("   • Real-time task monitoring")
        print("   • Live survey progress with question map")
        print("   • Manual answer overrides")
        print("   • CAPTCHA queue management")
        print("   • System health monitoring")
        print("   • Error logs and statistics")
        print("   • Pause/Resume/Cancel controls")
        
        print("\n🎮 Try these actions in the control panel:")
        print("   1. Monitor the survey progress in real-time")
        print("   2. Override an answer when prompted")
        print("   3. Pause and resume the task")
        print("   4. View system logs and performance metrics")
        
        # Keep running to allow control panel interaction
        await asyncio.sleep(300)  # 5 minutes
        
    except KeyboardInterrupt:
        print("🛑 Stopping for control panel demo...")
    finally:
        await agent.stop_agent()

async def example_error_handling_and_recovery():
    """Example demonstrating error handling and recovery."""
    print("🛡️ Error Handling and Recovery Example")
    
    config = AgentConfig(
        max_retries=5,
        retry_delay=10,
        browser_headless=True
    )
    
    agent = AutonomousSurveyAgent(config)
    
    try:
        agent_task = asyncio.create_task(agent.start_agent())
        await asyncio.sleep(2)
        
        # Create task that might encounter errors
        task_id = await agent.create_survey_task(
            platform="swagbucks",  # This might fail if no credentials
            persona=PersonaProfile.ENVIRONMENTALIST.value
        )
        
        print(f"🎯 Created potentially problematic task: {task_id}")
        print("🔍 Monitoring error handling and recovery...")
        
        # Monitor for errors and recovery
        for i in range(20):  # Monitor for 10 minutes
            await asyncio.sleep(30)
            
            status = agent.get_system_status()
            error_stats = status.get('error_statistics', {})
            
            if error_stats.get('total_errors', 0) > 0:
                print(f"⚠️  Errors detected: {error_stats}")
                print("🔄 System should be handling retries automatically...")
            
            if status['stats']['tasks_failed'] > 0:
                print("❌ Task failed after all retries")
                break
            elif status['stats']['tasks_completed'] > 0:
                print("✅ Task completed successfully despite errors")
                break
        
    except Exception as e:
        logger.error(f"Error in recovery example: {e}")
    finally:
        await agent.stop_agent()

def setup_environment_example():
    """Show how to set up environment variables."""
    print("🔧 Environment Setup Example")
    print("\n📝 Required Environment Variables:")
    
    env_vars = {
        'SWAGBUCKS_EMAIL': 'your_swagbucks_email@example.com',
        'SWAGBUCKS_PASSWORD': 'your_secure_password',
        'INBOXDOLLARS_EMAIL': 'your_inboxdollars_email@example.com',
        'INBOXDOLLARS_PASSWORD': 'your_secure_password',
        'OPENAI_API_KEY': 'sk-your_openai_api_key_here',
        'DEEPSEEK_API_KEY': 'your_deepseek_api_key_here',
        'SURVEY_ENCRYPTION_KEY': 'base64_encoded_encryption_key',
        'DATABASE_URL': 'sqlite:///surveys.db',
        'MAX_CONCURRENT_TASKS': '3',
        'BROWSER_HEADLESS': 'true',
        'CONTROL_PANEL_PORT': '12000'
    }
    
    print("\n🔐 Security Best Practices:")
    print("   • Store credentials in environment variables or secrets manager")
    print("   • Use encrypted session storage")
    print("   • Rotate behavioral fingerprints")
    print("   • Enable anti-detection features")
    
    print("\n💡 Example .env file:")
    for key, value in env_vars.items():
        print(f"{key}={value}")
    
    print("\n🚀 To run with environment variables:")
    print("   export SWAGBUCKS_EMAIL='your_email@example.com'")
    print("   export SWAGBUCKS_PASSWORD='your_password'")
    print("   python autonomous_survey_agent.py")

async def main():
    """Main example runner."""
    print("🤖 Autonomous Survey Agent - Examples")
    print("=====================================")
    
    examples = {
        '1': ('Basic Usage', example_basic_usage),
        '2': ('Advanced Multi-Persona', example_advanced_usage),
        '3': ('Control Panel Integration', example_control_panel_integration),
        '4': ('Error Handling & Recovery', example_error_handling_and_recovery),
        '5': ('Environment Setup Guide', lambda: setup_environment_example())
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"   {key}. {name}")
    
    choice = input("\nSelect example (1-5) or 'all' to run all: ").strip()
    
    if choice == 'all':
        for key, (name, func) in examples.items():
            print(f"\n{'='*50}")
            print(f"Running Example {key}: {name}")
            print('='*50)
            
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            
            if key != '5':  # Don't wait after setup guide
                input("\nPress Enter to continue to next example...")
    
    elif choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        
        if asyncio.iscoroutinefunction(func):
            await func()
        else:
            func()
    
    else:
        print("Invalid choice. Running basic example...")
        await example_basic_usage()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")