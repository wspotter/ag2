# Autonomous Agent System

A sophisticated autonomous agent system built on AG2 (AutoGen) that can operate independently, manage tasks, and coordinate multiple specialized agents using swarm orchestration.

## Features

- **Autonomous Operation**: Runs independently with minimal human intervention
- **Swarm Orchestration**: Uses AG2's swarm pattern for dynamic agent coordination
- **Task Management**: Intelligent task queuing, prioritization, and execution
- **Multi-Agent Collaboration**: Specialized agents for research, analysis, and execution
- **Performance Monitoring**: Real-time metrics and health monitoring
- **Configurable**: YAML-based configuration for easy customization

## Architecture

### Core Components

1. **Agent Manager** (`agent_manager.py`): Orchestrates all agent operations
2. **Task System**: Manages task lifecycle from creation to completion
3. **Specialized Agents**:
   - **Coordinator**: Task assignment and workflow management
   - **Researcher**: Information gathering and analysis
   - **Analyst**: Data processing and insights
   - **Executor**: Implementation and deployment

### Agent Workflow

```
Task Creation → Coordinator → Specialist Agents → Execution → Results
```

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM service)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/wspotter/ag2.git
cd ag2/swarm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your LLM settings:
```bash
# Create OAI_CONFIG_LIST file or set environment variables
export OPENAI_API_KEY="your-api-key-here"
```

4. Customize configuration (optional):
```bash
# Edit config.yaml to match your needs
nano config.yaml
```

### Running the System

```bash
python main.py
```

The system will:
1. Initialize all agents
2. Start autonomous operation mode
3. Generate initial tasks
4. Begin processing tasks using swarm coordination
5. Monitor performance and health

## Configuration

### Basic Configuration (`config.yaml`)

```yaml
system:
  max_concurrent_tasks: 5
  task_timeout: 3600

llm_config:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 2000

agents:
  coordinator:
    enabled: true
    max_rounds: 20
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `CONFIG_PATH`: Path to configuration file (default: config.yaml)

## Usage Examples

### Adding Custom Tasks

```python
from agent_manager import AgentManager, Task, TaskPriority
from datetime import datetime, timedelta

# Create a custom task
task = Task(
    id="custom_001",
    description="Analyze competitor pricing strategies",
    priority=TaskPriority.HIGH,
    created_at=datetime.now(),
    deadline=datetime.now() + timedelta(hours=2)
)

# Add to the system
await agent_manager.add_task(task)
```

### Monitoring System Status

```python
# Get current system status
status = await agent_manager.get_status()
print(f"Active tasks: {status['active_tasks']}")
print(f"Completed tasks: {status['completed_tasks']}")
```

## Advanced Features

### Custom Agent Creation

You can extend the system with custom agents:

```python
custom_agent = ConversableAgent(
    name="CustomSpecialist",
    system_message="Your specialized system message here",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Add to agent manager
agent_manager.agents['custom'] = custom_agent
```

### Task Prioritization

The system supports four priority levels:
- `CRITICAL`: Immediate attention required
- `HIGH`: Important tasks with tight deadlines
- `MEDIUM`: Standard priority tasks
- `LOW`: Background tasks

### Performance Monitoring

The system tracks:
- Task completion rates
- Average completion times
- Agent utilization
- System health metrics

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Configuration Errors**:
   - Check YAML syntax in `config.yaml`
   - Verify all required fields are present

3. **Agent Initialization Failures**:
   - Ensure LLM configuration is correct
   - Check network connectivity

### Logging

Logs are written to:
- Console (stdout)
- `autonomous_agent.log` file

Adjust log level in `config.yaml`:
```yaml
system:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Development

### Project Structure

```
swarm/
├── main.py              # Entry point
├── agent_manager.py     # Core orchestration
├── config_loader.py     # Configuration handling
├── config.yaml          # System configuration
├── requirements.txt     # Dependencies
├── README.md           # This file
└── autonomous_agent.log # Runtime logs
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the AG2 ecosystem. See the main AG2 repository for license information.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review AG2 documentation: https://docs.ag2.ai/
3. Open an issue in the main AG2 repository

## Roadmap

- [ ] Web-based dashboard for monitoring
- [ ] Integration with external APIs
- [ ] Advanced task scheduling
- [ ] Machine learning for task optimization
- [ ] Multi-tenant support
- [ ] Enhanced security features