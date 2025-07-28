#!/bin/bash

# Navigate to the swarm directory
cd swarm

# Initialize git if not already done
git init

# Add all files
git add .

# Commit the changes
git commit -m "Add autonomous agent system with swarm architecture

- Core agent manager with task orchestration
- Configuration loader with YAML support
- Multi-agent system (coordinator, researcher, analyst, executor)
- Task prioritization and monitoring
- Complete test suite
- Documentation and examples"

# Add remote origin (replace with your actual repo URL)
git remote add origin https://github.com/wspotter/ag2.git

# Push to main branch
git push -u origin main