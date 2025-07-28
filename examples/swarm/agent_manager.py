"""
Agent Manager - Orchestrates autonomous agent operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.swarm_agent import initiate_swarm_chat, AfterWorkOption

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    description: str
    priority: TaskPriority
    created_at: datetime
    deadline: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None

class AgentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_completion_time': 0,
            'agent_utilization': {}
        }
    
    async def initialize(self):
        """Initialize all agents and systems"""
        try:
            await self._create_agents()
            await self._setup_monitoring()
            logger.info("Agent manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {e}")
            raise
    
    async def _create_agents(self):
        """Create and configure all agents"""
        llm_config = self.config.get('llm_config', {})
        
        # Task Coordinator Agent
        self.agents['coordinator'] = ConversableAgent(
            name="TaskCoordinator",
            system_message="""You are a Task Coordinator responsible for:
            1. Analyzing incoming tasks and requirements
            2. Determining task priority and complexity
            3. Assigning tasks to appropriate specialist agents
            4. Monitoring task progress and quality
            5. Coordinating between agents when needed
            
            Always be efficient and prioritize high-impact tasks.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Research Agent
        self.agents['researcher'] = ConversableAgent(
            name="ResearchAgent",
            system_message="""You are a Research Specialist responsible for:
            1. Gathering information from various sources
            2. Analyzing data and trends
            3. Providing comprehensive research reports
            4. Fact-checking and verification
            5. Staying updated on relevant topics
            
            Be thorough, accurate, and cite your sources.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Analysis Agent
        self.agents['analyst'] = ConversableAgent(
            name="AnalysisAgent",
            system_message="""You are a Data Analysis Specialist responsible for:
            1. Processing and analyzing complex data
            2. Identifying patterns and insights
            3. Creating visualizations and reports
            4. Making data-driven recommendations
            5. Statistical analysis and modeling
            
            Focus on actionable insights and clear communication.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Execution Agent
        self.agents['executor'] = ConversableAgent(
            name="ExecutionAgent",
            system_message="""You are an Execution Specialist responsible for:
            1. Implementing solutions and recommendations
            2. Automating routine tasks
            3. Managing workflows and processes
            4. Quality assurance and testing
            5. Deployment and monitoring
            
            Be precise, reliable, and always verify your work.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Human Proxy (for oversight)
        self.agents['human'] = UserProxyAgent(
            name="HumanOversight",
            system_message="Human oversight agent for critical decisions",
            human_input_mode="TERMINATE",
            code_execution_config=False
        )
        
        logger.info(f"Created {len(self.agents)} agents")
    
    async def _setup_monitoring(self):
        """Setup system monitoring and health checks"""
        # Initialize agent utilization tracking
        for agent_name in self.agents.keys():
            self.metrics['agent_utilization'][agent_name] = {
                'tasks_handled': 0,
                'total_time': 0,
                'success_rate': 0
            }
    
    async def start_autonomous_mode(self):
        """Start the autonomous operation mode"""
        self.running = True
        logger.info("Starting autonomous mode...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._performance_tracker()),
            asyncio.create_task(self._generate_initial_tasks())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in autonomous mode: {e}")
            raise
    
    async def _generate_initial_tasks(self):
        """Generate initial tasks for the system to work on"""
        initial_tasks = [
            Task(
                id="task_001",
                description="Analyze current market trends in AI and automation",
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                deadline=datetime.now() + timedelta(hours=2)
            ),
            Task(
                id="task_002", 
                description="Research best practices for autonomous agent systems",
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now(),
                deadline=datetime.now() + timedelta(hours=4)
            ),
            Task(
                id="task_003",
                description="Create performance optimization recommendations",
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                deadline=datetime.now() + timedelta(hours=3)
            )
        ]
        
        for task in initial_tasks:
            await self.add_task(task)
        
        logger.info(f"Generated {len(initial_tasks)} initial tasks")
    
    async def _task_processor(self):
        """Main task processing loop"""
        while self.running:
            try:
                if self.task_queue:
                    # Get highest priority task
                    task = max(self.task_queue, key=lambda t: t.priority.value)
                    self.task_queue.remove(task)
                    
                    # Process the task
                    await self._process_task(task)
                
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, task: Task):
        """Process a single task using swarm orchestration"""
        try:
            logger.info(f"Processing task: {task.id} - {task.description}")
            task.status = "in_progress"
            self.active_tasks[task.id] = task
            
            start_time = datetime.now()
            
            # Determine which agents to use based on task type
            selected_agents = self._select_agents_for_task(task)
            
            # Use swarm orchestration to process the task
            chat_result, context_variables, last_speaker = initiate_swarm_chat(
                initial_agent=self.agents['coordinator'],
                agents=selected_agents,
                messages=f"Task: {task.description}\nPriority: {task.priority.name}\nDeadline: {task.deadline}",
                max_rounds=20,
                after_work=AfterWorkOption.TERMINATE
            )
            
            # Extract result from chat
            task.result = self._extract_task_result(chat_result)
            task.status = "completed"
            
            # Update metrics
            completion_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(task, completion_time, True)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task.id]
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process task {task.id}: {e}")
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            self._update_metrics(task, 0, False)
    
    def _select_agents_for_task(self, task: Task) -> List[ConversableAgent]:
        """Select appropriate agents based on task requirements"""
        # Always include coordinator
        selected = [self.agents['coordinator']]
        
        # Add specialist agents based on task description
        description_lower = task.description.lower()
        
        if any(keyword in description_lower for keyword in ['research', 'analyze', 'study', 'investigate']):
            selected.append(self.agents['researcher'])
        
        if any(keyword in description_lower for keyword in ['data', 'analysis', 'pattern', 'insight']):
            selected.append(self.agents['analyst'])
        
        if any(keyword in description_lower for keyword in ['implement', 'execute', 'deploy', 'create']):
            selected.append(self.agents['executor'])
        
        # Add human oversight for critical tasks
        if task.priority == TaskPriority.CRITICAL:
            selected.append(self.agents['human'])
        
        return selected
    
    def _extract_task_result(self, chat_result) -> str:
        """Extract meaningful result from chat history"""
        if not chat_result or not chat_result.chat_history:
            return "No result generated"
        
        # Get the last meaningful message
        for message in reversed(chat_result.chat_history):
            if message.get('content') and len(message['content']) > 50:
                return message['content']
        
        return "Task completed but no detailed result available"
    
    def _update_metrics(self, task: Task, completion_time: float, success: bool):
        """Update performance metrics"""
        if success:
            self.metrics['tasks_completed'] += 1
        else:
            self.metrics['tasks_failed'] += 1
        
        # Update average completion time
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        if total_tasks > 0:
            current_avg = self.metrics['average_completion_time']
            self.metrics['average_completion_time'] = (
                (current_avg * (total_tasks - 1) + completion_time) / total_tasks
            )
    
    async def _health_monitor(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                # Check agent health
                healthy_agents = 0
                for agent_name, agent in self.agents.items():
                    if hasattr(agent, 'llm_config') and agent.llm_config:
                        healthy_agents += 1
                
                # Log health status
                if healthy_agents == len(self.agents):
                    logger.debug("All agents healthy")
                else:
                    logger.warning(f"Only {healthy_agents}/{len(self.agents)} agents healthy")
                
                # Check task queue health
                if len(self.task_queue) > 100:
                    logger.warning(f"Task queue getting large: {len(self.task_queue)} tasks")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracker(self):
        """Track and log performance metrics"""
        while self.running:
            try:
                logger.info(f"Performance Metrics: {self.metrics}")
                await asyncio.sleep(300)  # Log every 5 minutes
            except Exception as e:
                logger.error(f"Error in performance tracker: {e}")
                await asyncio.sleep(300)
    
    async def add_task(self, task: Task):
        """Add a new task to the queue"""
        self.task_queue.append(task)
        logger.info(f"Added task {task.id} to queue (Priority: {task.priority.name})")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'agents_count': len(self.agents),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'metrics': self.metrics
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent manager"""
        logger.info("Shutting down agent manager...")
        self.running = False
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30
        while self.active_tasks and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        if self.active_tasks:
            logger.warning(f"Shutdown with {len(self.active_tasks)} active tasks")
        
        logger.info("Agent manager shutdown complete")