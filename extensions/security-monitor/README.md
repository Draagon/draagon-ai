# Security Monitor Extension for Draagon-AI

Autonomous, AI-powered security monitoring with agentic investigation capabilities.

## Features

- **Multi-Source Monitoring**: Suricata IDS, syslogs, system health
- **Groq 70B Analysis**: Fast, intelligent threat classification
- **Agentic Investigation**: ReAct-style deep analysis for uncertain threats
- **Memory Integration**: Learns from past issues, remembers false positives
- **Voice Announcements**: Speaks alerts through Home Assistant voice devices

## Installation

```bash
cd ~/Development/draagon-ai/extensions/security-monitor
pip install -e .

# With Groq support
pip install -e ".[groq]"
```

## Configuration

Add to your `draagon.yaml`:

```yaml
extensions:
  security-monitor:
    enabled: true
    check_interval_seconds: 300

    llm:
      provider: groq
      model: llama-3.3-70b-versatile
      api_key: ${GROQ_API_KEY}

    monitors:
      suricata:
        enabled: true
        eve_log: /var/log/suricata/eve.json

      syslog:
        enabled: true
        paths:
          - /var/log/auth.log

      system_health:
        enabled: true
        nvme_warning_celsius: 60

    notifications:
      voice:
        enabled: true
        entity_id: assist_satellite.home_assistant_voice_...
        min_severity: high
```

## Usage

### As Background Service

The monitor runs automatically when the extension is loaded.

### Query via Roxy

```
"Roxy, any security issues?"
"Tell me about the SSH attempts yesterday"
"Investigate the suspicious domain"
```

### Programmatic Access

```python
from draagon_ai_ext_security import SecurityMonitorExtension

ext = SecurityMonitorExtension()
ext.initialize(config)

service = ext.get_services()["security_monitor"]
status = service.get_status()
```

## Architecture

See [DESIGN.md](./DESIGN.md) for detailed architecture documentation.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT
