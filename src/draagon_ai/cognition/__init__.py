"""Cognition module for Draagon AI.

This module contains the cognitive services that power AI agents:
- Belief reconciliation (observations → beliefs)
- Opinion formation
- Curiosity engine
- Identity management

These services depend on LLM and Memory providers supplied by the host application.
"""

from draagon_ai.cognition.beliefs import (
    BeliefReconciliationService,
    ReconciliationResult,
    CredibilityProvider,
    BELIEF_FORMATION_PROMPT,
    CONFLICT_RESOLUTION_PROMPT,
    OBSERVATION_EXTRACTION_PROMPT,
)

from draagon_ai.cognition.opinions import (
    OpinionFormationService,
    OpinionRequest,
    FormedOpinion,
    OpinionBasis,
    OpinionStrength,
    IdentityManager,  # Protocol
    OPINION_FORMATION_PROMPT,
    PREFERENCE_FORMATION_PROMPT,
    OPINION_UPDATE_PROMPT,
)

from draagon_ai.cognition.curiosity import (
    CuriosityEngine,
    CuriousQuestion,
    KnowledgeGap,
    QuestionType,
    QuestionPriority,
    QuestionPurpose,
    TraitProvider,
    CURIOSITY_DETECTION_PROMPT,
    ANSWER_PROCESSING_PROMPT,
)

from draagon_ai.cognition.identity import (
    IdentityManager as IdentityManagerImpl,  # Implementation
    IdentityStorage,
    serialize_identity,
    deserialize_identity,
    serialize_user_prefs,
    deserialize_user_prefs,
)

from draagon_ai.cognition.proactive_questions import (
    ProactiveQuestionTimingService,
    QuestionOpportunity,
    ConversationMoment,
    UserSentiment,
    SentimentResult,
    UserPreferencesProvider,
    TraitProvider as QuestionTraitProvider,  # Avoid conflict with curiosity.TraitProvider
    TIMING_ASSESSMENT_PROMPT,
    QUESTION_SELECTION_PROMPT,
)

from draagon_ai.cognition.learning import (
    LearningService,
    LearningResult,
    LearningCandidate,
    LearningType,
    MemoryAction,
    FailureType,
    VerificationResult,
    SkillConfidence,
    SearchProvider,
    CredibilityProvider as LearningCredibilityProvider,  # Avoid conflict
    UserProvider,
    LearningExtension,
    LEARNING_DETECTION_PROMPT,
    LEARNING_EXTRACTION_PROMPT,
    MODE_LEARNING_GUIDANCE,
)

from draagon_ai.cognition.decomposition import (
    # Types
    EntityType as DecompositionEntityType,  # Avoid potential conflicts
    DecompositionResult,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    SemanticRole,
    Presupposition,
    CommonsenseInference,
    TemporalInfo,
    ModalityInfo,
    InterpretationBranch,
    # Service
    DecompositionService,
)

from draagon_ai.cognition.decomposition.memory_integration import (
    MemoryIntegration,
    DecompositionMemoryService,
    IntegrationConfig,
    IntegrationResult,
)

__all__ = [
    # Beliefs
    "BeliefReconciliationService",
    "ReconciliationResult",
    "CredibilityProvider",
    "BELIEF_FORMATION_PROMPT",
    "CONFLICT_RESOLUTION_PROMPT",
    "OBSERVATION_EXTRACTION_PROMPT",
    # Opinions
    "OpinionFormationService",
    "OpinionRequest",
    "FormedOpinion",
    "OpinionBasis",
    "OpinionStrength",
    "IdentityManager",  # Protocol
    "OPINION_FORMATION_PROMPT",
    "PREFERENCE_FORMATION_PROMPT",
    "OPINION_UPDATE_PROMPT",
    # Curiosity
    "CuriosityEngine",
    "CuriousQuestion",
    "KnowledgeGap",
    "QuestionType",
    "QuestionPriority",
    "QuestionPurpose",
    "TraitProvider",
    "CURIOSITY_DETECTION_PROMPT",
    "ANSWER_PROCESSING_PROMPT",
    # Identity
    "IdentityManagerImpl",  # Implementation
    "IdentityStorage",
    "serialize_identity",
    "deserialize_identity",
    "serialize_user_prefs",
    "deserialize_user_prefs",
    # Proactive Questions
    "ProactiveQuestionTimingService",
    "QuestionOpportunity",
    "ConversationMoment",
    "UserSentiment",
    "SentimentResult",
    "UserPreferencesProvider",
    "QuestionTraitProvider",
    "TIMING_ASSESSMENT_PROMPT",
    "QUESTION_SELECTION_PROMPT",
    # Learning
    "LearningService",
    "LearningResult",
    "LearningCandidate",
    "LearningType",
    "MemoryAction",
    "FailureType",
    "VerificationResult",
    "SkillConfidence",
    "SearchProvider",
    "LearningCredibilityProvider",
    "UserProvider",
    "LearningExtension",
    "LEARNING_DETECTION_PROMPT",
    "LEARNING_EXTRACTION_PROMPT",
    "MODE_LEARNING_GUIDANCE",
    # Decomposition
    "DecompositionEntityType",
    "DecompositionResult",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "SemanticRole",
    "Presupposition",
    "CommonsenseInference",
    "TemporalInfo",
    "ModalityInfo",
    "InterpretationBranch",
    "DecompositionService",
    "MemoryIntegration",
    "DecompositionMemoryService",
    "IntegrationConfig",
    "IntegrationResult",
]
