#!/usr/bin/env python3
"""
Basic usage example for the Autonomous Agent System
"""

import asyncio
import logging
from datetime import datetime, timedelta

from agent_manager import AgentManager, Task, TaskPriority
from config_loader import ConfigLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_example():
    """Basic usage example"""
    
    # Load configuration
    config = ConfigLoader.load_config("config.yaml")
    
    # Create agent manager
    agent_manager = AgentManager(config)
    await agent_manager.initialize()
    
    # Create some example tasks
    tasks = [
        Task(
            id="example_001",
            description="Research the latest trends in artificial intelligence",
            priority=TaskPriority.HIGH,
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(hours=1)
        ),
        Task(
            id="example_002",
            description="Analyze the performance metrics of our current system",
            priority=TaskPriority.MEDIUM,
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(hours=2)
        ),
        Task(
            id="example_003",
            description="Create a summary report of completed tasks",
            priority=TaskPriority.LOW,
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(hours=4)
        )
    ]
    
    # Add tasks to the system
    for task in tasks:
        await agent_manager.add_task(task)
        logger.info(f"Added task: {task.description}")
    
    # Run for a limited time (for demo purposes)
    logger.info("Starting autonomous processing...")
    
    # Start the system in background
    system_task = asyncio.create_task(agent_manager.start_autonomous_mode())
    
    # Monitor for 60 seconds
    for i in range(12):  # 12 * 5 seconds = 60 seconds
        await asyncio.sleep(5)
        status = await agent_manager.get_status()
        logger.info(f"Status check {i+1}: {status['completed_tasks']} completed, {status['active_tasks']} active")
    
    # Shutdown
    await agent_manager.shutdown()
    system_task.cancel()
    
    logger.info("Example completed!")

if __name__ == "__main__":
    asyncio.run(basic_example())