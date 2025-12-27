# Autonomous Agent Extension for draagon-ai

This extension provides autonomous background agent capabilities for draagon-ai applications.

## Features

- **Background Cognitive Processes**: Agent runs during idle time, making decisions about what to research, verify, or learn
- **Multi-Layer Guardrails**: Tier classification, harm checks, rate limiting, semantic safety
- **Self-Monitoring**: Agent reviews its own actions for issues and improvements
- **Transparency**: All actions logged with full reasoning for dashboard visibility
- **Configurable**: Cycle interval, daily budget, active hours, shadow mode

## Installation

```bash
pip install draagon-ai-ext-autonomous
```

Or install in editable mode:

```bash
cd extensions/autonomous
pip install -e .
```

## Configuration

Add to your `draagon.yaml`:

```yaml
extensions:
  autonomous:
    enabled: true
    config:
      cycle_minutes: 30
      daily_budget: 20
      shadow_mode: false
      self_monitoring: true
      active_hours_start: 8
      active_hours_end: 22
```

## Usage

```python
from draagon_ai.extensions import get_extension_manager

# Extensions are auto-discovered and loaded
manager = get_extension_manager()

# Get the autonomous agent service
autonomous_service = manager.get_service("autonomous_agent")

# Start the agent (usually done by your application)
await autonomous_service.start()
```

## Action Tiers

| Tier | Risk Level | Actions | Approval |
|------|------------|---------|----------|
| **0** | Always Safe | Research, reflect, update beliefs | Autonomous |
| **1** | Low Risk | Prepare suggestions, note questions | Autonomous + Log |
| **2** | Medium | Proactive reminders, calendar suggestions | Notify User |
| **3** | High | Send messages, control devices | Requires Approval |
| **4** | Forbidden | Financial, security, impersonation | Never |

## Guardrail Layers

1. **Tier Classification** - Is action within allowed tier?
2. **Harm Check** (LLM) - Could this harm anyone?
3. **Rate Limiting** - Not too many consecutive same actions?
4. **Semantic Safety** (LLM) - Final sanity check

## Self-Monitoring

After each cycle, the agent reviews its actions to detect:
- Unexpected results
- Contradictions in learned information
- Low-value activities
- Important findings needing user attention

High-severity findings are persisted for dashboard review.

## License

AGPL-3.0-or-later
