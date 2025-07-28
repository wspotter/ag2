#!/usr/bin/env python3
"""
Autonomous Agent System - Main Entry Point
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from agent_manager import AgentManager
from config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AutonomousAgentSystem:
    def __init__(self):
        self.config = None
        self.agent_manager = None
        self.running = False
        
    async def initialize(self):
        """Initialize the autonomous agent system"""
        try:
            # Load configuration
            config_path = Path("config.yaml")
            self.config = ConfigLoader.load_config(config_path)
            logger.info("Configuration loaded successfully")
            
            # Initialize agent manager
            self.agent_manager = AgentManager(self.config)
            await self.agent_manager.initialize()
            logger.info("Agent manager initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def start(self):
        """Start the autonomous agent system"""
        if not await self.initialize():
            logger.error("System initialization failed")
            return
            
        self.running = True
        logger.info("Autonomous Agent System starting...")
        
        try:
            # Start the main execution loop
            await self.agent_manager.start_autonomous_mode()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down Autonomous Agent System...")
        self.running = False
        
        if self.agent_manager:
            await self.agent_manager.shutdown()
        
        logger.info("System shutdown complete")

def signal_handler(signum, frame):
    """Handle system signals"""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the system
    system = AutonomousAgentSystem()
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())