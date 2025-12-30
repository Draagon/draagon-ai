"""Behavior Quality Validation Framework.

Provides comprehensive quality assessment for generated behaviors,
including prompt quality, action coverage, test robustness, and
production readiness scoring.

This is the "god-level" quality gate that ensures behaviors
created by the Architect are actually production-ready.

ARCHITECTURE NOTE:
This validator uses LLM-based semantic evaluation for assessing prompt quality.
We do NOT use regex patterns for semantic understanding - the LLM handles all
quality judgments about whether prompts are clear, complete, and well-structured.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
import json

from ..behaviors.types import (
    Action,
    Behavior,
    BehaviorPrompts,
    BehaviorStatus,
    BehaviorTestCase,
    TestResults,
    Trigger,
)


class LLMProvider(Protocol):
    """Protocol for LLM inference."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        ...


class QualityLevel(str, Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"  # 90%+ Ready for production
    GOOD = "good"  # 75-90% Minor improvements suggested
    ACCEPTABLE = "acceptable"  # 60-75% Usable but needs work
    POOR = "poor"  # 40-60% Significant issues
    FAILING = "failing"  # <40% Not usable


@dataclass
class QualityIssue:
    """A specific quality issue found during validation."""

    category: str  # "prompt", "action", "test", "structure", "coverage"
    severity: str  # "critical", "major", "minor", "info"
    message: str
    field: str | None = None  # Which field has the issue
    suggestion: str | None = None  # How to fix


@dataclass
class PromptQualityScore:
    """Quality assessment for prompts."""

    decision_prompt_score: float = 0.0  # 0-1
    synthesis_prompt_score: float = 0.0  # 0-1

    # Detailed metrics
    has_clear_role: bool = False
    has_action_list: bool = False
    has_decision_criteria: bool = False
    has_output_format: bool = False
    has_examples: bool = False
    has_edge_case_handling: bool = False
    has_domain_context: bool = False

    # Issues found
    issues: list[QualityIssue] = field(default_factory=list)


@dataclass
class ActionQualityScore:
    """Quality assessment for actions."""

    overall_score: float = 0.0  # 0-1
    per_action_scores: dict[str, float] = field(default_factory=dict)

    # Metrics
    has_descriptions: bool = False
    has_parameters: bool = False
    has_examples: bool = False
    has_triggers: bool = False
    parameter_coverage: float = 0.0  # % of actions with full params

    issues: list[QualityIssue] = field(default_factory=list)


@dataclass
class TestQualityScore:
    """Quality assessment for test cases."""

    overall_score: float = 0.0  # 0-1

    # Coverage metrics
    action_coverage: float = 0.0  # % of actions with tests
    positive_test_count: int = 0
    negative_test_count: int = 0
    edge_case_count: int = 0

    # Quality metrics
    has_expected_actions: bool = False
    has_expected_responses: bool = False
    has_forbidden_actions: bool = False
    tests_per_action: float = 0.0

    issues: list[QualityIssue] = field(default_factory=list)


@dataclass
class BehaviorQualityReport:
    """Complete quality assessment report for a behavior."""

    behavior_id: str
    assessed_at: datetime = field(default_factory=datetime.now)

    # Overall
    overall_score: float = 0.0  # 0-1
    quality_level: QualityLevel = QualityLevel.FAILING
    production_ready: bool = False

    # Component scores
    prompt_quality: PromptQualityScore = field(default_factory=PromptQualityScore)
    action_quality: ActionQualityScore = field(default_factory=ActionQualityScore)
    test_quality: TestQualityScore = field(default_factory=TestQualityScore)

    # Structure validation
    structure_score: float = 0.0
    has_required_fields: bool = False

    # All issues
    all_issues: list[QualityIssue] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "behavior_id": self.behavior_id,
            "assessed_at": self.assessed_at.isoformat(),
            "overall_score": self.overall_score,
            "quality_level": self.quality_level.value,
            "production_ready": self.production_ready,
            "prompt_quality": {
                "decision_score": self.prompt_quality.decision_prompt_score,
                "synthesis_score": self.prompt_quality.synthesis_prompt_score,
            },
            "action_quality": {
                "score": self.action_quality.overall_score,
                "per_action": self.action_quality.per_action_scores,
            },
            "test_quality": {
                "score": self.test_quality.overall_score,
                "coverage": self.test_quality.action_coverage,
            },
            "issue_count": len(self.all_issues),
            "critical_issues": len([i for i in self.all_issues if i.severity == "critical"]),
            "recommendations": self.recommendations,
        }


# =============================================================================
# LLM Prompts for Semantic Quality Evaluation
# =============================================================================

DECISION_PROMPT_EVALUATION = '''Evaluate the quality of this AI behavior decision prompt.

PROMPT TO EVALUATE:
{prompt}

Assess the prompt on these criteria (score each 0-10):

1. ROLE_CLARITY: Does the prompt clearly define who/what the AI is and its purpose?
   - Look for: Identity statement, purpose, responsibilities
   - Score 0 if no role definition, 10 if crystal clear

2. ACTION_DOCUMENTATION: Are available actions clearly listed and described?
   - Look for: List of actions, what each does, when to use each
   - Score 0 if no actions listed, 10 if comprehensive

3. DECISION_GUIDANCE: Does it explain how to choose between actions?
   - Look for: Criteria for selection, conditions, priorities
   - Score 0 if no guidance, 10 if clear decision logic

4. OUTPUT_FORMAT: Is the expected response format specified?
   - Look for: Format specification (XML, JSON, etc.), structure
   - Score 0 if no format, 10 if clearly specified

5. EXAMPLES: Are there examples of inputs and expected outputs?
   - Look for: Sample queries, example responses
   - Score 0 if no examples, 10 if helpful examples

6. COMPLETENESS: Is the prompt thorough enough to guide good decisions?
   - Consider: Length, detail, coverage of edge cases
   - Score 0 if too sparse, 10 if comprehensive

Respond with XML only:
<evaluation>
    <role_clarity>0-10</role_clarity>
    <action_documentation>0-10</action_documentation>
    <decision_guidance>0-10</decision_guidance>
    <output_format>0-10</output_format>
    <examples>0-10</examples>
    <completeness>0-10</completeness>
    <issues>
        <issue>specific problem found</issue>
    </issues>
    <suggestions>
        <suggestion>specific improvement</suggestion>
    </suggestions>
</evaluation>
'''

SYNTHESIS_PROMPT_EVALUATION = '''Evaluate the quality of this AI behavior synthesis prompt.

PROMPT TO EVALUATE:
{prompt}

Assess the prompt on these criteria (score each 0-10):

1. STYLE_GUIDANCE: Does it define the tone and style of responses?
   - Look for: Tone (formal/casual), verbosity, personality
   - Score 0 if no style guidance, 10 if well-defined

2. FORMAT_GUIDANCE: Does it specify how to format responses?
   - Look for: Structure, length, formatting rules
   - Score 0 if no format guidance, 10 if clear

3. ERROR_HANDLING: Does it explain how to handle failures gracefully?
   - Look for: Error messages, fallback responses, edge cases
   - Score 0 if no error handling, 10 if comprehensive

Respond with XML only:
<evaluation>
    <style_guidance>0-10</style_guidance>
    <format_guidance>0-10</format_guidance>
    <error_handling>0-10</error_handling>
    <issues>
        <issue>specific problem found</issue>
    </issues>
    <suggestions>
        <suggestion>specific improvement</suggestion>
    </suggestions>
</evaluation>
'''


class BehaviorQualityValidator:
    """Validates behavior quality for production readiness.

    This validator performs comprehensive quality assessment including:
    - Prompt quality (LLM-based semantic evaluation)
    - Action quality (parameters, descriptions, examples)
    - Test quality (coverage, edge cases, assertions)
    - Structure validation (required fields, consistency)

    ARCHITECTURE:
    Uses LLM-based semantic evaluation for prompt quality assessment.
    We do NOT use regex patterns for understanding meaning - the LLM
    handles all quality judgments about clarity, completeness, and structure.

    Usage:
        # With LLM (recommended for accurate evaluation)
        validator = BehaviorQualityValidator(llm=my_llm)
        report = await validator.validate_async(behavior)

        # Without LLM (fast but less accurate - uses heuristics)
        validator = BehaviorQualityValidator()
        report = validator.validate(behavior)

        if report.production_ready:
            print("Ready for deployment!")
        else:
            for issue in report.all_issues:
                print(f"{issue.severity}: {issue.message}")
    """

    # Minimum thresholds for production readiness
    MIN_PROMPT_SCORE = 0.7
    MIN_ACTION_SCORE = 0.7
    MIN_TEST_SCORE = 0.6
    MIN_TEST_COVERAGE = 0.8  # 80% action coverage
    MIN_OVERALL_SCORE = 0.7

    def __init__(self, llm: LLMProvider | None = None):
        """Initialize validator.

        Args:
            llm: Optional LLM provider for semantic evaluation.
                 If not provided, uses fast heuristic-based evaluation.
        """
        self._llm = llm

    async def validate_async(self, behavior: Behavior) -> BehaviorQualityReport:
        """Perform comprehensive quality validation with LLM-based evaluation.

        This is the preferred method when an LLM is available, as it provides
        accurate semantic evaluation of prompt quality.

        Args:
            behavior: The behavior to validate

        Returns:
            Complete quality report
        """
        report = BehaviorQualityReport(behavior_id=behavior.behavior_id)

        # Validate structure first
        report.structure_score, structure_issues = self._validate_structure(behavior)
        report.all_issues.extend(structure_issues)
        report.has_required_fields = report.structure_score >= 0.8

        # Validate prompts using LLM semantic evaluation
        if self._llm and behavior.prompts:
            report.prompt_quality = await self._validate_prompts_with_llm(behavior.prompts)
        else:
            report.prompt_quality = self._validate_prompts_heuristic(behavior.prompts)
        report.all_issues.extend(report.prompt_quality.issues)

        # Validate actions
        report.action_quality = self._validate_actions(behavior.actions)
        report.all_issues.extend(report.action_quality.issues)

        # Validate tests
        report.test_quality = self._validate_tests(
            behavior.test_cases,
            behavior.actions,
        )
        report.all_issues.extend(report.test_quality.issues)

        # Calculate overall score (weighted)
        report.overall_score = self._calculate_overall_score(report)

        # Determine quality level
        report.quality_level = self._determine_quality_level(report.overall_score)

        # Check production readiness
        report.production_ready = self._check_production_ready(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def validate(self, behavior: Behavior) -> BehaviorQualityReport:
        """Perform comprehensive quality validation (sync, heuristic-based).

        This is the fast synchronous method that uses heuristic evaluation.
        For accurate semantic evaluation, use validate_async() with an LLM.

        Args:
            behavior: The behavior to validate

        Returns:
            Complete quality report
        """
        report = BehaviorQualityReport(behavior_id=behavior.behavior_id)

        # Validate structure first
        report.structure_score, structure_issues = self._validate_structure(behavior)
        report.all_issues.extend(structure_issues)
        report.has_required_fields = report.structure_score >= 0.8

        # Validate prompts using heuristics (fast but less accurate)
        report.prompt_quality = self._validate_prompts_heuristic(behavior.prompts)
        report.all_issues.extend(report.prompt_quality.issues)

        # Validate actions
        report.action_quality = self._validate_actions(behavior.actions)
        report.all_issues.extend(report.action_quality.issues)

        # Validate tests
        report.test_quality = self._validate_tests(
            behavior.test_cases,
            behavior.actions,
        )
        report.all_issues.extend(report.test_quality.issues)

        # Calculate overall score (weighted)
        report.overall_score = self._calculate_overall_score(report)

        # Determine quality level
        report.quality_level = self._determine_quality_level(report.overall_score)

        # Check production readiness
        report.production_ready = self._check_production_ready(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    # =========================================================================
    # LLM-Based Semantic Evaluation (Preferred)
    # =========================================================================

    async def _validate_prompts_with_llm(
        self,
        prompts: BehaviorPrompts,
    ) -> PromptQualityScore:
        """Validate prompt quality using LLM semantic evaluation.

        This is the preferred method as it understands meaning, not just keywords.
        """
        score = PromptQualityScore()

        # Evaluate decision prompt
        decision_score, decision_issues = await self._evaluate_decision_prompt_llm(
            prompts.decision_prompt
        )
        score.decision_prompt_score = decision_score
        score.issues.extend(decision_issues)

        # Evaluate synthesis prompt
        synthesis_score, synthesis_issues = await self._evaluate_synthesis_prompt_llm(
            prompts.synthesis_prompt
        )
        score.synthesis_prompt_score = synthesis_score
        score.issues.extend(synthesis_issues)

        return score

    async def _evaluate_decision_prompt_llm(
        self,
        prompt: str,
    ) -> tuple[float, list[QualityIssue]]:
        """Evaluate decision prompt quality using LLM."""
        issues = []

        if not prompt:
            issues.append(QualityIssue(
                category="prompt",
                severity="critical",
                message="Decision prompt is empty",
                field="decision_prompt",
            ))
            return 0.0, issues

        try:
            # Ask LLM to evaluate the prompt
            eval_prompt = DECISION_PROMPT_EVALUATION.format(prompt=prompt)
            response = await self._llm.generate(
                eval_prompt,
                system_prompt="You are an expert at evaluating AI prompts. Be objective and thorough.",
                temperature=0.3,
            )

            # Parse JSON response
            result = self._parse_json_response(response)
            if not result:
                # Fallback to heuristic if LLM response parsing fails
                return self._validate_decision_prompt_heuristic(prompt)

            # Calculate score from criteria (each 0-10, convert to 0-1)
            criteria_weights = {
                "role_clarity": 0.15,
                "action_documentation": 0.20,
                "decision_guidance": 0.15,
                "output_format": 0.20,
                "examples": 0.15,
                "completeness": 0.15,
            }

            total_score = 0.0
            for criterion, weight in criteria_weights.items():
                criterion_score = result.get(criterion, 5) / 10.0
                total_score += criterion_score * weight

                # Generate issues for low scores
                if result.get(criterion, 5) < 5:
                    severity = "major" if result.get(criterion, 5) < 3 else "minor"
                    issues.append(QualityIssue(
                        category="prompt",
                        severity=severity,
                        message=f"Decision prompt has low {criterion.replace('_', ' ')} ({result.get(criterion, 5)}/10)",
                        field="decision_prompt",
                    ))

            # Add specific issues from LLM
            for issue_text in result.get("issues", []):
                if issue_text and len(issue_text) > 5:
                    issues.append(QualityIssue(
                        category="prompt",
                        severity="minor",
                        message=issue_text,
                        field="decision_prompt",
                    ))

            # Store suggestions for recommendations
            score = min(1.0, max(0.0, total_score))
            return score, issues

        except Exception:
            # Fallback to heuristic on any error
            return self._validate_decision_prompt_heuristic(prompt)

    async def _evaluate_synthesis_prompt_llm(
        self,
        prompt: str,
    ) -> tuple[float, list[QualityIssue]]:
        """Evaluate synthesis prompt quality using LLM."""
        issues = []

        if not prompt:
            issues.append(QualityIssue(
                category="prompt",
                severity="major",
                message="Synthesis prompt is empty",
                field="synthesis_prompt",
            ))
            return 0.0, issues

        try:
            # Ask LLM to evaluate the prompt
            eval_prompt = SYNTHESIS_PROMPT_EVALUATION.format(prompt=prompt)
            response = await self._llm.generate(
                eval_prompt,
                system_prompt="You are an expert at evaluating AI prompts. Be objective and thorough.",
                temperature=0.3,
            )

            # Parse JSON response
            result = self._parse_json_response(response)
            if not result:
                # Fallback to heuristic if LLM response parsing fails
                return self._validate_synthesis_prompt_heuristic(prompt)

            # Calculate score from criteria
            criteria_weights = {
                "style_guidance": 0.3,
                "format_guidance": 0.4,
                "error_handling": 0.3,
            }

            total_score = 0.0
            for criterion, weight in criteria_weights.items():
                criterion_score = result.get(criterion, 5) / 10.0
                total_score += criterion_score * weight

                # Generate issues for low scores
                if result.get(criterion, 5) < 5:
                    severity = "major" if result.get(criterion, 5) < 3 else "minor"
                    issues.append(QualityIssue(
                        category="prompt",
                        severity=severity,
                        message=f"Synthesis prompt has low {criterion.replace('_', ' ')} ({result.get(criterion, 5)}/10)",
                        field="synthesis_prompt",
                    ))

            # Add specific issues from LLM
            for issue_text in result.get("issues", []):
                if issue_text and len(issue_text) > 5:
                    issues.append(QualityIssue(
                        category="prompt",
                        severity="minor",
                        message=issue_text,
                        field="synthesis_prompt",
                    ))

            score = min(1.0, max(0.0, total_score))
            return score, issues

        except Exception:
            # Fallback to heuristic on any error
            return self._validate_synthesis_prompt_heuristic(prompt)

    def _parse_json_response(self, response: str) -> dict | None:
        """Parse JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # =========================================================================
    # Heuristic-Based Evaluation (Fast Fallback)
    # =========================================================================

    def _validate_prompts_heuristic(
        self,
        prompts: BehaviorPrompts | None,
    ) -> PromptQualityScore:
        """Validate prompt quality using fast heuristics.

        This uses simple length/structure checks rather than semantic analysis.
        Less accurate but much faster than LLM evaluation.
        """
        score = PromptQualityScore()

        if not prompts:
            score.issues.append(QualityIssue(
                category="prompt",
                severity="critical",
                message="No prompts defined",
            ))
            return score

        # Validate decision prompt
        score.decision_prompt_score, decision_issues = self._validate_decision_prompt_heuristic(
            prompts.decision_prompt
        )
        score.issues.extend(decision_issues)

        # Validate synthesis prompt
        score.synthesis_prompt_score, synthesis_issues = self._validate_synthesis_prompt_heuristic(
            prompts.synthesis_prompt
        )
        score.issues.extend(synthesis_issues)

        return score

    def _validate_decision_prompt_heuristic(
        self,
        prompt: str,
    ) -> tuple[float, list[QualityIssue]]:
        """Validate decision prompt using simple heuristics.

        NOTE: This is a fast fallback. It uses length and basic structure checks,
        NOT regex patterns for semantic understanding. For accurate evaluation,
        use _evaluate_decision_prompt_llm() instead.
        """
        issues = []

        if not prompt:
            issues.append(QualityIssue(
                category="prompt",
                severity="critical",
                message="Decision prompt is empty",
                field="decision_prompt",
            ))
            return 0.0, issues

        score = 0.0
        word_count = len(prompt.split())

        # Length-based scoring (simple heuristic)
        # A good decision prompt is typically 100-500 words
        if word_count >= 100:
            score += 0.4  # Has substantial content
        elif word_count >= 50:
            score += 0.2
        else:
            issues.append(QualityIssue(
                category="prompt",
                severity="major",
                message=f"Decision prompt too short ({word_count} words)",
                field="decision_prompt",
                suggestion="Expand prompt with more guidance (aim for 100+ words)",
            ))

        # Structure indicators (just checking presence, not meaning)
        # These are very loose checks - the LLM evaluation is more accurate
        if word_count > 1000:
            issues.append(QualityIssue(
                category="prompt",
                severity="minor",
                message=f"Decision prompt very long ({word_count} words)",
                field="decision_prompt",
                suggestion="Consider condensing to improve focus",
            ))

        # Line count as structure indicator
        line_count = len(prompt.strip().split('\n'))
        if line_count >= 10:
            score += 0.3  # Has structured content
        elif line_count >= 5:
            score += 0.15

        # Paragraph indicator (multiple blank lines = structure)
        if '\n\n' in prompt:
            score += 0.3  # Has paragraphs/sections

        return min(1.0, max(0.0, score)), issues

    def _validate_synthesis_prompt_heuristic(
        self,
        prompt: str,
    ) -> tuple[float, list[QualityIssue]]:
        """Validate synthesis prompt using simple heuristics."""
        issues = []

        if not prompt:
            issues.append(QualityIssue(
                category="prompt",
                severity="major",
                message="Synthesis prompt is empty",
                field="synthesis_prompt",
            ))
            return 0.0, issues

        score = 0.0
        word_count = len(prompt.split())

        # Length-based scoring
        if word_count >= 50:
            score += 0.5
        elif word_count >= 25:
            score += 0.25

        # Structure indicator
        line_count = len(prompt.strip().split('\n'))
        if line_count >= 5:
            score += 0.3
        elif line_count >= 3:
            score += 0.15

        # Has some structure
        if '\n\n' in prompt or ':' in prompt:
            score += 0.2

        return min(1.0, max(0.0, score)), issues

    def _validate_structure(self, behavior: Behavior) -> tuple[float, list[QualityIssue]]:
        """Validate behavior structure completeness."""
        issues = []
        score = 1.0

        # Required fields
        if not behavior.behavior_id:
            issues.append(QualityIssue(
                category="structure",
                severity="critical",
                message="Behavior ID is missing",
                field="behavior_id",
            ))
            score -= 0.3

        if not behavior.name:
            issues.append(QualityIssue(
                category="structure",
                severity="critical",
                message="Behavior name is missing",
                field="name",
            ))
            score -= 0.2

        if not behavior.description:
            issues.append(QualityIssue(
                category="structure",
                severity="major",
                message="Behavior description is missing",
                field="description",
                suggestion="Add a clear description of what this behavior does",
            ))
            score -= 0.1

        if not behavior.actions:
            issues.append(QualityIssue(
                category="structure",
                severity="critical",
                message="Behavior has no actions",
                field="actions",
                suggestion="Add at least one action",
            ))
            score -= 0.3

        if not behavior.prompts:
            issues.append(QualityIssue(
                category="structure",
                severity="critical",
                message="Behavior has no prompts",
                field="prompts",
                suggestion="Generate decision and synthesis prompts",
            ))
            score -= 0.3

        return max(0.0, score), issues

    def _validate_actions(self, actions: list[Action]) -> ActionQualityScore:
        """Validate action quality."""
        score = ActionQualityScore()

        if not actions:
            score.issues.append(QualityIssue(
                category="action",
                severity="critical",
                message="No actions defined",
            ))
            return score

        total_score = 0.0
        actions_with_params = 0
        actions_with_descriptions = 0
        actions_with_examples = 0

        for action in actions:
            action_score = 0.0

            # Check description
            if action.description and len(action.description) > 10:
                action_score += 0.3
                actions_with_descriptions += 1
            else:
                score.issues.append(QualityIssue(
                    category="action",
                    severity="major",
                    message=f"Action '{action.name}' lacks description",
                    field=f"actions.{action.name}.description",
                ))

            # Check parameters
            if action.parameters:
                action_score += 0.3
                actions_with_params += 1

                # Check parameter completeness
                for param_name, param in action.parameters.items():
                    if not param.description:
                        score.issues.append(QualityIssue(
                            category="action",
                            severity="minor",
                            message=f"Parameter '{param_name}' in '{action.name}' lacks description",
                            field=f"actions.{action.name}.parameters.{param_name}",
                        ))

            # Check examples
            if action.examples:
                action_score += 0.2
                actions_with_examples += 1
            else:
                score.issues.append(QualityIssue(
                    category="action",
                    severity="minor",
                    message=f"Action '{action.name}' lacks examples",
                    field=f"actions.{action.name}.examples",
                    suggestion="Add example queries that trigger this action",
                ))

            # Check triggers
            if action.triggers:
                action_score += 0.2
            else:
                score.issues.append(QualityIssue(
                    category="action",
                    severity="minor",
                    message=f"Action '{action.name}' has no triggers",
                    field=f"actions.{action.name}.triggers",
                ))

            score.per_action_scores[action.name] = action_score
            total_score += action_score

        # Calculate overall
        score.overall_score = total_score / len(actions) if actions else 0.0
        score.has_descriptions = actions_with_descriptions == len(actions)
        score.has_parameters = actions_with_params > 0
        score.has_examples = actions_with_examples > 0
        score.parameter_coverage = actions_with_params / len(actions) if actions else 0.0

        return score

    def _validate_tests(
        self,
        tests: list[BehaviorTestCase],
        actions: list[Action],
    ) -> TestQualityScore:
        """Validate test case quality."""
        score = TestQualityScore()

        if not tests:
            score.issues.append(QualityIssue(
                category="test",
                severity="major",
                message="No test cases defined",
                suggestion="Generate test cases for all actions",
            ))
            return score

        action_names = {a.name for a in actions}
        covered_actions = set()

        positive_count = 0
        negative_count = 0
        edge_case_count = 0
        has_expected_actions = 0
        has_expected_responses = 0
        has_forbidden = 0

        for test in tests:
            # Track coverage
            if test.expected_actions:
                has_expected_actions += 1
                for action in test.expected_actions:
                    covered_actions.add(action)

            if test.expected_response_contains:
                has_expected_responses += 1

            if test.forbidden_actions:
                has_forbidden += 1
                negative_count += 1

            # Classify test type
            if test.expected_actions:
                positive_count += 1

            # Check for edge case indicators
            name_lower = test.name.lower()
            query_lower = test.user_query.lower() if test.user_query else ""
            if any(term in name_lower or term in query_lower
                   for term in ["edge", "boundary", "empty", "invalid", "ambiguous"]):
                edge_case_count += 1

        # Calculate metrics
        score.action_coverage = (
            len(covered_actions & action_names) / len(action_names)
            if action_names else 0.0
        )
        score.positive_test_count = positive_count
        score.negative_test_count = negative_count
        score.edge_case_count = edge_case_count
        score.has_expected_actions = has_expected_actions > 0
        score.has_expected_responses = has_expected_responses > 0
        score.has_forbidden_actions = has_forbidden > 0
        score.tests_per_action = len(tests) / len(actions) if actions else 0.0

        # Generate issues
        if score.action_coverage < 0.8:
            uncovered = action_names - covered_actions
            score.issues.append(QualityIssue(
                category="test",
                severity="major",
                message=f"Test coverage only {score.action_coverage:.0%}",
                suggestion=f"Add tests for: {', '.join(uncovered)}",
            ))

        if negative_count == 0:
            score.issues.append(QualityIssue(
                category="test",
                severity="major",
                message="No negative test cases",
                suggestion="Add tests for invalid inputs and forbidden actions",
            ))

        if edge_case_count == 0:
            score.issues.append(QualityIssue(
                category="test",
                severity="minor",
                message="No edge case tests",
                suggestion="Add tests for boundary conditions and ambiguous inputs",
            ))

        if len(tests) < len(actions):
            score.issues.append(QualityIssue(
                category="test",
                severity="minor",
                message=f"Only {len(tests)} tests for {len(actions)} actions",
                suggestion="Add more tests (aim for 2-3 per action)",
            ))

        # Calculate overall score
        coverage_weight = 0.4
        variety_weight = 0.3
        completeness_weight = 0.3

        coverage_score = score.action_coverage

        variety_score = min(1.0, (
            (0.5 if positive_count > 0 else 0.0) +
            (0.3 if negative_count > 0 else 0.0) +
            (0.2 if edge_case_count > 0 else 0.0)
        ))

        completeness_score = min(1.0, (
            (0.4 if has_expected_actions > 0 else 0.0) +
            (0.3 if has_expected_responses > 0 else 0.0) +
            (0.3 if has_forbidden > 0 else 0.0)
        ))

        score.overall_score = (
            coverage_score * coverage_weight +
            variety_score * variety_weight +
            completeness_score * completeness_weight
        )

        return score

    def _calculate_overall_score(self, report: BehaviorQualityReport) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "structure": 0.15,
            "prompts": 0.35,
            "actions": 0.25,
            "tests": 0.25,
        }

        prompt_score = (
            report.prompt_quality.decision_prompt_score * 0.7 +
            report.prompt_quality.synthesis_prompt_score * 0.3
        )

        weighted_sum = (
            report.structure_score * weights["structure"] +
            prompt_score * weights["prompts"] +
            report.action_quality.overall_score * weights["actions"] +
            report.test_quality.overall_score * weights["tests"]
        )

        return weighted_sum

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILING

    def _check_production_ready(self, report: BehaviorQualityReport) -> bool:
        """Check if behavior meets production readiness criteria."""
        # Check for critical issues
        critical_issues = [i for i in report.all_issues if i.severity == "critical"]
        if critical_issues:
            return False

        # Check minimum scores
        prompt_score = (
            report.prompt_quality.decision_prompt_score * 0.7 +
            report.prompt_quality.synthesis_prompt_score * 0.3
        )

        return (
            report.overall_score >= self.MIN_OVERALL_SCORE and
            prompt_score >= self.MIN_PROMPT_SCORE and
            report.action_quality.overall_score >= self.MIN_ACTION_SCORE and
            report.test_quality.overall_score >= self.MIN_TEST_SCORE and
            report.test_quality.action_coverage >= self.MIN_TEST_COVERAGE
        )

    def _generate_recommendations(
        self,
        report: BehaviorQualityReport,
    ) -> list[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Critical issues first
        critical = [i for i in report.all_issues if i.severity == "critical"]
        for issue in critical[:3]:
            rec = f"[CRITICAL] {issue.message}"
            if issue.suggestion:
                rec += f" - {issue.suggestion}"
            recommendations.append(rec)

        # Major issues
        major = [i for i in report.all_issues if i.severity == "major"]
        for issue in major[:3]:
            rec = f"[MAJOR] {issue.message}"
            if issue.suggestion:
                rec += f" - {issue.suggestion}"
            recommendations.append(rec)

        # Score-based recommendations
        if report.prompt_quality.decision_prompt_score < 0.7:
            recommendations.append(
                "Improve decision prompt: Add clear role, action list, and output format"
            )

        if report.test_quality.action_coverage < 0.8:
            recommendations.append(
                f"Increase test coverage from {report.test_quality.action_coverage:.0%} to 80%+"
            )

        if report.test_quality.edge_case_count == 0:
            recommendations.append(
                "Add edge case tests for robustness"
            )

        return recommendations[:10]  # Top 10 recommendations
