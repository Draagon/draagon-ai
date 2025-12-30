"""Security Monitor Extension for draagon-ai.

Provides autonomous, AI-powered security monitoring with:
- Multi-source monitoring (Suricata, syslogs, system health)
- Groq 70B analysis for intelligent threat classification
- Agentic investigation for deep analysis
- Memory integration for learning from past issues
- Voice announcements through Home Assistant
"""

from draagon_ai_ext_security.extension import SecurityMonitorExtension
from draagon_ai_ext_security.models import Alert, AnalysisResult, Severity

__version__ = "0.1.0"
__all__ = [
    "SecurityMonitorExtension",
    "Alert",
    "AnalysisResult",
    "Severity",
]
