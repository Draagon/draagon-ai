"""Content Type Analyzer and Semantic Extractor.

LLM-driven analysis of content to determine:
1. What type of content it is (prose, code, data, config, mixed)
2. Which portions contain natural language worth WSD processing
3. What structural/relational knowledge to extract
4. How to store for optimal context retrieval

Key Principle: Use LLM for semantic understanding, not regex heuristics.

Example:
    >>> from content_analyzer import ContentAnalyzer, ContentAnalysis
    >>>
    >>> analyzer = ContentAnalyzer(llm=llm_provider)
    >>>
    >>> # Analyze mixed content
    >>> analysis = await analyzer.analyze(code_with_docstrings)
    >>> print(analysis.content_type)  # "mixed"
    >>> print(analysis.natural_language_portions)  # [docstrings, comments]
    >>>
    >>> # Get just the NL for WSD processing
    >>> nl_text = analysis.get_natural_language_text()
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any: ...


def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ContentType(str, Enum):
    """Primary content type classification."""
    PROSE = "prose"           # Natural language documents
    CODE = "code"             # Source code
    DATA = "data"             # CSV, JSON data
    CONFIG = "config"         # Configuration files
    LOGS = "logs"             # Log files
    MIXED = "mixed"           # Multiple types combined
    UNKNOWN = "unknown"


class ProcessingStrategy(str, Enum):
    """How to process content for semantic extraction."""
    FULL_WSD = "full_wsd"                    # Full WSD pipeline
    EXTRACT_NL_ONLY = "extract_nl_only"      # Extract NL, WSD on that
    SCHEMA_EXTRACTION = "schema_extraction"   # Extract structure/schema
    PATTERN_ANALYSIS = "pattern_analysis"     # Find patterns (logs)
    SKIP = "skip"                            # Don't process semantically
    HYBRID = "hybrid"                        # Multiple strategies


@dataclass
class ContentComponent:
    """A portion of content with specific type."""
    component_type: str  # "docstring", "comment", "code", "data_row", etc.
    content: str
    start_line: int = 0
    end_line: int = 0
    language: str = ""  # Programming language if code
    is_natural_language: bool = False
    processing_strategy: ProcessingStrategy = ProcessingStrategy.SKIP

    def __post_init__(self):
        # Auto-detect NL for common types
        if self.component_type in ("docstring", "comment", "description", "error_message"):
            self.is_natural_language = True
            self.processing_strategy = ProcessingStrategy.FULL_WSD


@dataclass
class StructuralKnowledge:
    """Structural/relational knowledge extracted from content."""
    relationship_type: str  # "is_type", "returns", "contains", "raises", etc.
    subject: str
    object: str
    confidence: float = 1.0
    source_lines: tuple[int, int] = (0, 0)


@dataclass
class ContentAnalysis:
    """Complete analysis of content."""
    content_type: ContentType
    components: list[ContentComponent] = field(default_factory=list)
    structural_knowledge: list[StructuralKnowledge] = field(default_factory=list)
    detected_language: str = ""  # Programming language
    detected_format: str = ""    # Data format (csv, json, yaml)
    processing_recommendation: ProcessingStrategy = ProcessingStrategy.SKIP
    raw_content: str = ""
    analysis_confidence: float = 0.0

    def get_natural_language_text(self) -> str:
        """Extract all natural language portions combined."""
        nl_parts = []
        for component in self.components:
            if component.is_natural_language:
                nl_parts.append(component.content)
        return "\n\n".join(nl_parts)

    def get_natural_language_components(self) -> list[ContentComponent]:
        """Get only the NL components."""
        return [c for c in self.components if c.is_natural_language]

    def get_code_components(self) -> list[ContentComponent]:
        """Get only code components."""
        return [c for c in self.components if c.component_type == "code"]

    def has_natural_language(self) -> bool:
        """Check if any NL was found."""
        return any(c.is_natural_language for c in self.components)

    def get_structural_summary(self) -> str:
        """Get a text summary of structural knowledge."""
        if not self.structural_knowledge:
            return ""
        lines = []
        for sk in self.structural_knowledge:
            lines.append(f"- {sk.subject} {sk.relationship_type} {sk.object}")
        return "\n".join(lines)


# =============================================================================
# Content Analyzer
# =============================================================================


class ContentAnalyzer:
    """LLM-driven content type analyzer and semantic extractor.

    Uses the LLM to understand what type of content is provided and
    extract the portions that are meaningful for semantic processing.

    This is the entry point before WSD - we first understand WHAT we're
    looking at, then decide HOW to process it.
    """

    # Prompt for content analysis
    ANALYSIS_PROMPT = """Analyze this content and classify it for semantic processing.

Content to analyze:
```
{content}
```

Determine:
1. What type of content is this? (prose, code, data, config, logs, mixed)
2. If code, what programming language?
3. If data, what format (csv, json, yaml, etc.)?
4. What portions contain natural language that would benefit from semantic analysis?
   - Docstrings, comments, string literals with user-facing text
   - Error messages, log messages
   - Descriptions, annotations
5. What structural/relational knowledge can be extracted?
   - Type relationships, function contracts
   - Data schemas, column meanings
   - Configuration patterns

Output XML format:
<analysis>
    <content_type>prose|code|data|config|logs|mixed</content_type>
    <language>python|javascript|go|none</language>
    <format>csv|json|yaml|none</format>
    <confidence>0.0-1.0</confidence>
    <processing_recommendation>full_wsd|extract_nl_only|schema_extraction|pattern_analysis|skip</processing_recommendation>

    <components>
        <component type="docstring" lines="2-5" is_nl="true">
            <text>The actual docstring text here</text>
        </component>
        <component type="code" lines="6-20" is_nl="false">
            <text>Optional: key code summary</text>
        </component>
        <component type="comment" lines="10" is_nl="true">
            <text>The comment text</text>
        </component>
    </components>

    <structural_knowledge>
        <relationship type="has_parameter" subject="function_name" object="param_type"/>
        <relationship type="returns" subject="function_name" object="return_type"/>
        <relationship type="column_type" subject="column_name" object="data_type"/>
    </structural_knowledge>
</analysis>

Focus on extracting semantic value. Skip boilerplate, syntax, and non-meaningful content.
If the content is pure prose, set content_type to "prose" and include the full text as a single component.
"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        max_content_length: int = 8000,
    ):
        """Initialize the content analyzer.

        Args:
            llm: LLM provider for analysis
            max_content_length: Maximum content length to analyze
        """
        self._llm = llm
        self.max_content_length = max_content_length

        # Metrics
        self.metrics = {
            "analyses_performed": 0,
            "prose_detected": 0,
            "code_detected": 0,
            "data_detected": 0,
            "mixed_detected": 0,
            "nl_components_extracted": 0,
            "llm_calls": 0,
            "llm_calls_avoided": 0,
            "ambiguous_detected": 0,
        }

    async def analyze(self, content: str) -> ContentAnalysis:
        """Analyze content to determine type and extract semantic portions.

        Uses smart deferral: runs fast heuristics first, only calls LLM
        when content is ambiguous (mixed signals from multiple types).

        Args:
            content: The content to analyze

        Returns:
            ContentAnalysis with type, components, and recommendations
        """
        if not content or not content.strip():
            return ContentAnalysis(
                content_type=ContentType.UNKNOWN,
                raw_content=content,
            )

        self.metrics["analyses_performed"] += 1

        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n... (truncated)"

        # Step 1: Always run fast heuristic analysis first
        heuristic_result, is_ambiguous = self._heuristic_analysis_with_ambiguity(content)

        # Step 2: If no LLM available, return heuristic result
        if not self._llm:
            return heuristic_result

        # Step 3: If content is unambiguous, skip LLM (save cost/latency)
        if not is_ambiguous:
            self.metrics["llm_calls_avoided"] += 1
            logger.debug(f"Skipping LLM - unambiguous content type: {heuristic_result.content_type}")
            return heuristic_result

        # Step 4: Ambiguous content - use LLM for better classification
        self.metrics["ambiguous_detected"] += 1
        logger.debug(f"Ambiguous content detected, calling LLM (heuristic: {heuristic_result.content_type})")

        try:
            self.metrics["llm_calls"] += 1
            prompt = self.ANALYSIS_PROMPT.format(content=content)
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low for consistent classification
            )
            response_text = _extract_llm_content(response)
            return self._parse_analysis_response(response_text, content)
        except Exception as e:
            logger.warning(f"LLM analysis failed, using heuristic: {e}")
            return heuristic_result

    def _parse_analysis_response(
        self,
        response: str,
        original_content: str,
    ) -> ContentAnalysis:
        """Parse LLM analysis response."""
        analysis = ContentAnalysis(
            content_type=ContentType.UNKNOWN,
            raw_content=original_content,
        )

        # Extract content_type
        type_match = re.search(r"<content_type>(\w+)</content_type>", response)
        if type_match:
            type_str = type_match.group(1).lower()
            try:
                analysis.content_type = ContentType(type_str)
                self._update_type_metrics(analysis.content_type)
            except ValueError:
                analysis.content_type = ContentType.UNKNOWN

        # Extract language
        lang_match = re.search(r"<language>(\w+)</language>", response)
        if lang_match and lang_match.group(1).lower() != "none":
            analysis.detected_language = lang_match.group(1).lower()

        # Extract format
        fmt_match = re.search(r"<format>(\w+)</format>", response)
        if fmt_match and fmt_match.group(1).lower() != "none":
            analysis.detected_format = fmt_match.group(1).lower()

        # Extract confidence
        conf_match = re.search(r"<confidence>([0-9.]+)</confidence>", response)
        if conf_match:
            try:
                analysis.analysis_confidence = float(conf_match.group(1))
            except ValueError:
                analysis.analysis_confidence = 0.5

        # Extract processing recommendation
        rec_match = re.search(
            r"<processing_recommendation>(\w+)</processing_recommendation>",
            response,
        )
        if rec_match:
            try:
                analysis.processing_recommendation = ProcessingStrategy(rec_match.group(1))
            except ValueError:
                pass

        # Extract components
        analysis.components = self._extract_components(response, original_content)
        self.metrics["nl_components_extracted"] += len(
            [c for c in analysis.components if c.is_natural_language]
        )

        # Extract structural knowledge
        analysis.structural_knowledge = self._extract_structural_knowledge(response)

        return analysis

    def _extract_components(
        self,
        response: str,
        original_content: str,
    ) -> list[ContentComponent]:
        """Extract component definitions from response."""
        components = []

        # Find all component tags
        component_pattern = re.compile(
            r'<component\s+type="(\w+)"[^>]*is_nl="(true|false)"[^>]*>\s*'
            r'<text>(.*?)</text>\s*</component>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in component_pattern.finditer(response):
            comp_type = match.group(1)
            is_nl = match.group(2).lower() == "true"
            text = match.group(3).strip()

            if text:
                components.append(ContentComponent(
                    component_type=comp_type,
                    content=text,
                    is_natural_language=is_nl,
                    processing_strategy=(
                        ProcessingStrategy.FULL_WSD if is_nl
                        else ProcessingStrategy.SKIP
                    ),
                ))

        # If no components found but content is prose, treat whole thing as NL
        if not components and "prose" in response.lower():
            components.append(ContentComponent(
                component_type="prose",
                content=original_content,
                is_natural_language=True,
                processing_strategy=ProcessingStrategy.FULL_WSD,
            ))

        return components

    def _extract_structural_knowledge(
        self,
        response: str,
    ) -> list[StructuralKnowledge]:
        """Extract structural relationships from response."""
        knowledge = []

        rel_pattern = re.compile(
            r'<relationship\s+type="([^"]+)"\s+subject="([^"]+)"\s+object="([^"]+)"',
            re.IGNORECASE,
        )

        for match in rel_pattern.finditer(response):
            knowledge.append(StructuralKnowledge(
                relationship_type=match.group(1),
                subject=match.group(2),
                object=match.group(3),
            ))

        return knowledge

    def _heuristic_analysis(self, content: str) -> ContentAnalysis:
        """Fallback heuristic analysis when LLM unavailable.

        This is a simple fallback - the LLM version is preferred.
        """
        analysis, _ = self._heuristic_analysis_with_ambiguity(content)
        return analysis

    def _heuristic_analysis_with_ambiguity(
        self,
        content: str,
    ) -> tuple[ContentAnalysis, bool]:
        """Heuristic analysis that also detects ambiguous content.

        Returns:
            Tuple of (ContentAnalysis, is_ambiguous)
            - is_ambiguous is True when content has mixed signals suggesting
              LLM analysis would give better results
        """
        analysis = ContentAnalysis(
            content_type=ContentType.UNKNOWN,
            raw_content=content,
            analysis_confidence=0.5,
        )

        lines = content.split("\n")
        first_lines = "\n".join(lines[:20])

        # Detect content type signals
        signals = self._detect_content_signals(content, first_lines)

        # Count how many different content types are signaled
        signal_count = sum([
            signals["has_code_patterns"],
            signals["has_yaml_patterns"],
            signals["has_json_patterns"],
            signals["has_csv_patterns"],
            signals["has_markdown_headers"],
            signals["has_code_blocks"],
            signals["has_prose"],
        ])

        # Specific ambiguity patterns that heuristics get wrong
        is_ambiguous = False

        # Pattern 1: YAML frontmatter + markdown headers
        # Heuristic sees YAML, but it's actually markdown with frontmatter
        if signals["has_yaml_frontmatter"] and signals["has_markdown_headers"]:
            is_ambiguous = True

        # Pattern 2: Markdown with code blocks
        # Could be documentation (PROSE/MIXED) or code examples
        if signals["has_markdown_headers"] and signals["has_code_blocks"]:
            is_ambiguous = True

        # Pattern 3: Multiple distinct content types
        # When 3+ different signals are present, it's likely MIXED
        if signal_count >= 3:
            is_ambiguous = True

        # Pattern 4: Prose discussing code (without being code itself)
        # e.g., "To create a function, use def foo():"
        # But NOT actual code files with docstrings (that's just code)
        # Only flag as ambiguous if it looks like prose BUT has code snippets inline
        if signals["has_prose"] and signals["has_code_patterns"]:
            # If it looks primarily like code (def/class at line start), it's code
            looks_like_code_file = self._looks_like_code("\n".join(lines[:20]))
            if not looks_like_code_file and not signals["has_code_blocks"]:
                # Prose with inline code mentions - ambiguous
                code_line_ratio = signals["code_line_count"] / max(len(lines), 1)
                if 0.1 < code_line_ratio < 0.5:  # Prose with some code discussion
                    is_ambiguous = True

        # Now classify using simple heuristics (same logic as before)
        if self._looks_like_code(first_lines):
            analysis.content_type = ContentType.CODE
            analysis.processing_recommendation = ProcessingStrategy.EXTRACT_NL_ONLY
            analysis.components = self._extract_code_nl_heuristic(content)
            # Higher confidence if unambiguous
            analysis.analysis_confidence = 0.6 if is_ambiguous else 0.85
        elif self._looks_like_csv(first_lines):
            analysis.content_type = ContentType.DATA
            analysis.detected_format = "csv"
            analysis.processing_recommendation = ProcessingStrategy.SCHEMA_EXTRACTION
            analysis.analysis_confidence = 0.9  # CSV is usually unambiguous
        elif self._looks_like_json(first_lines):
            analysis.content_type = ContentType.DATA
            analysis.detected_format = "json"
            analysis.processing_recommendation = ProcessingStrategy.SCHEMA_EXTRACTION
            analysis.analysis_confidence = 0.9  # JSON is usually unambiguous
        elif self._looks_like_yaml(first_lines):
            analysis.content_type = ContentType.CONFIG
            analysis.detected_format = "yaml"
            analysis.processing_recommendation = ProcessingStrategy.SCHEMA_EXTRACTION
            # Lower confidence if we also see markdown
            analysis.analysis_confidence = 0.5 if signals["has_markdown_headers"] else 0.85
        else:
            # Assume prose
            analysis.content_type = ContentType.PROSE
            analysis.processing_recommendation = ProcessingStrategy.FULL_WSD
            analysis.components.append(ContentComponent(
                component_type="prose",
                content=content,
                is_natural_language=True,
                processing_strategy=ProcessingStrategy.FULL_WSD,
            ))
            analysis.analysis_confidence = 0.7 if is_ambiguous else 0.9

        self._update_type_metrics(analysis.content_type)
        return analysis, is_ambiguous

    def _detect_content_signals(
        self,
        content: str,
        first_lines: str,
    ) -> dict[str, bool | int]:
        """Detect various content type signals in the text.

        Returns dict of signals that indicate different content types.
        """
        lines = content.split("\n")

        # Code patterns
        code_indicators = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+[:\(]',
            r'\bfunction\s+\w+\s*\(',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\bconst\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=',
            r'\bfunc\s+\w+\(',
            r'\bpackage\s+\w+',
        ]
        code_matches = sum(1 for p in code_indicators if re.search(p, content))
        code_line_count = sum(
            1 for line in lines
            if any(re.search(p, line) for p in code_indicators)
        )

        # Markdown patterns
        has_markdown_headers = bool(re.search(r'^#{1,6}\s+\w', content, re.MULTILINE))
        has_code_blocks = "```" in content

        # YAML frontmatter: starts with ---, has key: value, ends with ---
        has_yaml_frontmatter = (
            content.strip().startswith("---") and
            content.count("---") >= 2 and
            bool(re.search(r'^\w+:\s*\S', content, re.MULTILINE))
        )

        # General YAML patterns (not frontmatter)
        yaml_line_count = sum(
            1 for line in lines[:15]
            if re.match(r'^\s*\w+:\s*', line) or line.strip().startswith("- ")
        )
        has_yaml_patterns = yaml_line_count >= 3

        # JSON patterns
        stripped = content.strip()
        has_json_patterns = (
            (stripped.startswith("{") or stripped.startswith("[")) and
            '"' in content and ":" in content
        )

        # CSV patterns
        if len(lines) >= 2:
            comma_counts = [line.count(",") for line in lines[:5] if line.strip()]
            has_csv_patterns = (
                len(set(comma_counts)) == 1 and
                comma_counts and comma_counts[0] >= 1
            )
        else:
            has_csv_patterns = False

        # Prose detection (significant natural language)
        # Count words that aren't code-like
        word_count = len(content.split())
        has_prose = word_count > 30 and not has_json_patterns and not has_csv_patterns

        return {
            "has_code_patterns": code_matches >= 2,
            "code_line_count": code_line_count,
            "has_markdown_headers": has_markdown_headers,
            "has_code_blocks": has_code_blocks,
            "has_yaml_frontmatter": has_yaml_frontmatter,
            "has_yaml_patterns": has_yaml_patterns,
            "has_json_patterns": has_json_patterns,
            "has_csv_patterns": has_csv_patterns,
            "has_prose": has_prose,
        }

    def _looks_like_code(self, text: str) -> bool:
        """Quick heuristic for code detection."""
        code_indicators = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+[:\(]',
            r'\bfunction\s+\w+\s*\(',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\bconst\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=',
            r'=>',
            r'\bfunc\s+\w+\(',
            r'\bpackage\s+\w+',
        ]
        matches = sum(1 for p in code_indicators if re.search(p, text))
        return matches >= 2

    def _looks_like_csv(self, text: str) -> bool:
        """Quick heuristic for CSV detection."""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False
        # Check if lines have consistent comma counts
        comma_counts = [line.count(",") for line in lines[:5]]
        return len(set(comma_counts)) == 1 and comma_counts[0] >= 1

    def _looks_like_json(self, text: str) -> bool:
        """Quick heuristic for JSON detection."""
        text = text.strip()
        return (text.startswith("{") or text.startswith("[")) and (
            '"' in text and ":" in text
        )

    def _looks_like_yaml(self, text: str) -> bool:
        """Quick heuristic for YAML detection."""
        lines = text.strip().split("\n")
        yaml_indicators = sum(
            1 for line in lines[:10]
            if re.match(r"^\s*\w+:\s*", line) or line.strip().startswith("- ")
        )
        return yaml_indicators >= 3

    def _extract_code_nl_heuristic(self, code: str) -> list[ContentComponent]:
        """Extract NL from code using simple patterns."""
        components = []

        # Python docstrings
        docstring_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
        for match in docstring_pattern.finditer(code):
            components.append(ContentComponent(
                component_type="docstring",
                content=match.group(1).strip(),
                is_natural_language=True,
                processing_strategy=ProcessingStrategy.FULL_WSD,
            ))

        # Single-quoted docstrings
        docstring_pattern2 = re.compile(r"'''(.*?)'''", re.DOTALL)
        for match in docstring_pattern2.finditer(code):
            components.append(ContentComponent(
                component_type="docstring",
                content=match.group(1).strip(),
                is_natural_language=True,
                processing_strategy=ProcessingStrategy.FULL_WSD,
            ))

        # Python/JS/etc comments
        comment_pattern = re.compile(r"#\s*(.+)$", re.MULTILINE)
        for match in comment_pattern.finditer(code):
            comment = match.group(1).strip()
            # Skip shebangs and pragma comments
            if not comment.startswith("!") and not comment.startswith("type:"):
                components.append(ContentComponent(
                    component_type="comment",
                    content=comment,
                    is_natural_language=True,
                    processing_strategy=ProcessingStrategy.FULL_WSD,
                ))

        # C-style comments
        c_comment_pattern = re.compile(r"//\s*(.+)$", re.MULTILINE)
        for match in c_comment_pattern.finditer(code):
            components.append(ContentComponent(
                component_type="comment",
                content=match.group(1).strip(),
                is_natural_language=True,
                processing_strategy=ProcessingStrategy.FULL_WSD,
            ))

        return components

    def _update_type_metrics(self, content_type: ContentType) -> None:
        """Update metrics based on detected type."""
        if content_type == ContentType.PROSE:
            self.metrics["prose_detected"] += 1
        elif content_type == ContentType.CODE:
            self.metrics["code_detected"] += 1
        elif content_type == ContentType.DATA:
            self.metrics["data_detected"] += 1
        elif content_type == ContentType.MIXED:
            self.metrics["mixed_detected"] += 1

    def get_metrics(self) -> dict[str, int]:
        """Get analysis metrics."""
        return dict(self.metrics)


# =============================================================================
# Convenience Functions
# =============================================================================


async def analyze_content(
    content: str,
    llm: LLMProvider | None = None,
) -> ContentAnalysis:
    """Analyze content type and extract semantic portions.

    Convenience function that creates an analyzer and runs analysis.

    Args:
        content: Content to analyze
        llm: Optional LLM provider

    Returns:
        ContentAnalysis with type and components
    """
    analyzer = ContentAnalyzer(llm=llm)
    return await analyzer.analyze(content)


def extract_natural_language(content: str) -> str:
    """Quick extraction of NL from code (heuristic, no LLM).

    Use this when you just need to pull out comments/docstrings
    without full analysis.

    Args:
        content: Code or mixed content

    Returns:
        Combined natural language text
    """
    analyzer = ContentAnalyzer()
    analysis = analyzer._heuristic_analysis(content)
    return analysis.get_natural_language_text()
