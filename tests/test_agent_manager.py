"""
Tests for Agent Manager
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agent_manager import AgentManager, Task, TaskPriority
from config_loader import ConfigLoader

@pytest.fixture
def config():
    """Test configuration"""
    return ConfigLoader._get_default_config()

@pytest.fixture
async def agent_manager(config):
    """Create agent manager for testing"""
    manager = AgentManager(config)
    await manager.initialize()
    return manager

@pytest.mark.asyncio
async def test_agent_manager_initialization(config):
    """Test agent manager initialization"""
    manager = AgentManager(config)
    await manager.initialize()
    
    assert len(manager.agents) == 5  # coordinator, researcher, analyst, executor, human
    assert 'coordinator' in manager.agents
    assert 'researcher' in manager.agents
    assert 'analyst' in manager.agents
    assert 'executor' in manager.agents
    assert 'human' in manager.agents

@pytest.mark.asyncio
async def test_task_creation():
    """Test task creation"""
    task = Task(
        id="test_001",
        description="Test task",
        priority=TaskPriority.HIGH,
        created_at=datetime.now()
    )
    
    assert task.id == "test_001"
    assert task.priority == TaskPriority.HIGH
    assert task.status == "pending"

@pytest.mark.asyncio
async def test_add_task(agent_manager):
    """Test adding tasks to the queue"""
    task = Task(
        id="test_002",
        description="Test task 2",
        priority=TaskPriority.MEDIUM,
        created_at=datetime.now()
    )
    
    await agent_manager.add_task(task)
    assert len(agent_manager.task_queue) == 1
    assert agent_manager.task_queue[0].id == "test_002"

@pytest.mark.asyncio
async def test_agent_selection(agent_manager):
    """Test agent selection for different task types"""
    research_task = Task(
        id="research_001",
        description="Research market trends",
        priority=TaskPriority.HIGH,
        created_at=datetime.now()
    )
    
    selected_agents = agent_manager._select_agents_for_task(research_task)
    agent_names = [agent.name for agent in selected_agents]
    
    assert "TaskCoordinator" in agent_names
    assert "ResearchAgent" in agent_names

@pytest.mark.asyncio
async def test_status_reporting(agent_manager):
    """Test status reporting"""
    status = await agent_manager.get_status()
    
    assert 'running' in status
    assert 'agents_count' in status
    assert 'queued_tasks' in status
    assert 'active_tasks' in status
    assert 'completed_tasks' in status
    assert 'metrics' in status

@pytest.mark.asyncio
async def test_metrics_initialization(agent_manager):
    """Test metrics initialization"""
    metrics = agent_manager.metrics
    
    assert metrics['tasks_completed'] == 0
    assert metrics['tasks_failed'] == 0
    assert metrics['average_completion_time'] == 0
    assert 'agent_utilization' in metrics

if __name__ == "__main__":
    pytest.main([__file__])