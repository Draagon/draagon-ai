# Home Assistant Extension for draagon-ai

This extension provides Home Assistant smart home integration for draagon-ai agents.

## Features

- **Entity State Queries**: Get state of lights, sensors, switches, etc.
- **Device Control**: Turn on/off, set brightness, color, temperature
- **Entity Search**: Find entities by name or type
- **Weather**: Get current weather from Home Assistant
- **Location Awareness**: Know which room the voice device is in

## Installation

```bash
pip install draagon-ai-ext-home-assistant
```

Or install in development mode:

```bash
cd extensions/home-assistant
pip install -e .
```

## Configuration

In your `draagon.yaml`:

```yaml
extensions:
  home_assistant:
    enabled: true
    config:
      url: "http://192.168.168.206:8123"
      token: "${HA_TOKEN}"  # From environment variable
      cache_ttl_seconds: 300
```

## Usage

The extension is auto-discovered via entry points:

```python
from draagon_ai.extensions.discovery import discover_extensions

extensions = discover_extensions()
ha_ext = extensions['home_assistant']()
ha_ext.initialize({
    'url': 'http://your-ha-instance:8123',
    'token': 'your-long-lived-access-token',
})

tools = ha_ext.get_tools()
```

## Tools Provided

| Tool | Description |
|------|-------------|
| `get_entity` | Get state of a specific entity |
| `search_entities` | Search for entities by name/type |
| `call_service` | Control a device (turn on/off, etc.) |
| `get_weather` | Get current weather |
| `get_location` | Get assistant's room location |

## License

MIT
