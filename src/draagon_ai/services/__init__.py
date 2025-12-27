"""Services for draagon-ai.

Services contain the core logic for various capabilities like
behavior creation, evolution, quality validation, execution, and research.
"""

from .behavior_architect import (
    BehaviorArchitectService,
    BehaviorDesign,
    MutationPrompt,
)
from .behavior_quality import (
    BehaviorQualityValidator,
    BehaviorQualityReport,
    QualityLevel,
    QualityIssue,
)
from .behavior_executor import (
    BehaviorExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionDecision,
    ToolRegistry,
    ToolResult,
    create_mock_tool_registry,
)

# Feedback collection
from .feedback import (
    FeedbackCollector,
    InteractionRecord,
    FeedbackStats,
)

# Evolution
from .evolution import (
    EvolutionService,
    EvolutionServiceConfig,
    EvolutionServiceResult,
    VoiceAssistantFitnessEvaluator,
)

# Evolution scheduling
from .evolution_scheduler import (
    EvolutionScheduler,
    EvolutionSchedulerConfig,
    EvolutionRun,
)

__all__ = [
    # Architect
    "BehaviorArchitectService",
    "BehaviorDesign",
    "MutationPrompt",
    # Quality
    "BehaviorQualityValidator",
    "BehaviorQualityReport",
    "QualityLevel",
    "QualityIssue",
    # Executor
    "BehaviorExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionDecision",
    "ToolRegistry",
    "ToolResult",
    "create_mock_tool_registry",
    # Feedback
    "FeedbackCollector",
    "InteractionRecord",
    "FeedbackStats",
    # Evolution
    "EvolutionService",
    "EvolutionServiceConfig",
    "EvolutionServiceResult",
    "VoiceAssistantFitnessEvaluator",
    # Scheduler
    "EvolutionScheduler",
    "EvolutionSchedulerConfig",
    "EvolutionRun",
]
