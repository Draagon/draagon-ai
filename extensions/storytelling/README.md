# Draagon-AI Storytelling Extension

Interactive storytelling extension for draagon-ai that provides text adventure / interactive fiction capabilities.

## Features

- **Story Teller Behavior**: Narrates interactive stories with rich descriptions
- **Story Character Behavior**: NPCs with personalities, goals, and secrets
- **Drama Manager**: Paces the narrative for engaging experiences
- **Theme Discovery**: Personalized story generation based on user preferences

## Installation

```bash
pip install draagon-ai-ext-storytelling
```

Or for development:
```bash
pip install -e path/to/draagon-ai/extensions/storytelling
```

## Usage

### With Extension Manager

```python
from draagon_ai.extensions import get_extension_manager

manager = get_extension_manager()
behaviors = manager.get_all_behaviors()

# Find story teller
story_teller = next(b for b in behaviors if b.behavior_id == "story_teller")
```

### Direct Import

```python
from draagon_ai_ext_storytelling.behavior import (
    STORY_TELLER_TEMPLATE,
    create_story_character,
)
```

## Configuration

In your `draagon.yaml`:

```yaml
extensions:
  storytelling:
    enabled: true
    config:
      drama_intensity: 0.7
      default_narrator: "warm"  # warm, mysterious, dramatic, sardonic
```

## License

MIT
