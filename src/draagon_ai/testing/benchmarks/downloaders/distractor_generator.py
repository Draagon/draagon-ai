"""Synthetic distractor generator for benchmark corpus.

Generates synthetic distractor documents using LLM or templates
to test retrieval robustness. Distractors can be:
- Very different (random topics)
- Somewhat similar (related domain, different content)
- Very similar (hard negatives - same keywords, different meaning)
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from ..corpus import BenchmarkDocument, DocumentCategory, DocumentSource

logger = logging.getLogger(__name__)


class SimilarityLevel(str, Enum):
    """Similarity level for generated distractors."""

    VERY_DIFFERENT = "very_different"  # Random unrelated topics
    SOMEWHAT_SIMILAR = "somewhat_similar"  # Related domain, different content
    VERY_SIMILAR = "very_similar"  # Hard negatives with keyword overlap


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by distractor generator."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs,
    ) -> dict[str, str]:
        """Generate chat completion."""
        ...


@dataclass
class DistractorConfig:
    """Configuration for distractor generation."""

    category: DocumentCategory
    domain: str
    similarity_level: SimilarityLevel
    keywords: list[str] = field(default_factory=list)
    target_length: int = 500  # Approximate word count


# Topic pools for each category
CATEGORY_TOPICS = {
    DocumentCategory.TECHNICAL: [
        ("database", ["SELECT", "JOIN", "index", "query", "table", "schema"]),
        ("networking", ["TCP", "HTTP", "socket", "packet", "protocol", "latency"]),
        ("security", ["encryption", "authentication", "certificate", "firewall", "vulnerability"]),
        ("cloud", ["container", "kubernetes", "deployment", "scaling", "instance"]),
        ("compiler", ["parser", "lexer", "AST", "optimization", "bytecode"]),
    ],
    DocumentCategory.NARRATIVE: [
        ("adventure", ["journey", "quest", "hero", "discovery", "danger"]),
        ("mystery", ["detective", "clue", "suspect", "investigation", "crime"]),
        ("romance", ["love", "heart", "relationship", "passion", "destiny"]),
        ("scifi", ["spaceship", "alien", "future", "technology", "galaxy"]),
        ("fantasy", ["magic", "wizard", "dragon", "kingdom", "prophecy"]),
    ],
    DocumentCategory.LEGAL: [
        ("contract", ["party", "agreement", "terms", "obligation", "liability"]),
        ("copyright", ["intellectual property", "license", "infringement", "rights"]),
        ("employment", ["employee", "termination", "compensation", "benefits"]),
        ("privacy", ["data", "consent", "disclosure", "protection", "GDPR"]),
        ("litigation", ["plaintiff", "defendant", "court", "damages", "judgment"]),
    ],
    DocumentCategory.ACADEMIC: [
        ("physics", ["quantum", "particle", "energy", "mass", "momentum"]),
        ("biology", ["cell", "DNA", "protein", "organism", "evolution"]),
        ("chemistry", ["molecule", "reaction", "compound", "catalyst", "bond"]),
        ("mathematics", ["theorem", "proof", "equation", "function", "set"]),
        ("linguistics", ["syntax", "morphology", "phoneme", "semantics", "grammar"]),
    ],
    DocumentCategory.CONVERSATIONAL: [
        ("support", ["help", "issue", "ticket", "resolved", "thanks"]),
        ("chat", ["hey", "cool", "yeah", "lol", "btw"]),
        ("email", ["regards", "attached", "meeting", "follow-up", "FYI"]),
        ("forum", ["thread", "reply", "bump", "solved", "off-topic"]),
        ("interview", ["experience", "skills", "team", "project", "role"]),
    ],
    DocumentCategory.KNOWLEDGE_BASE: [
        ("tutorial", ["step", "guide", "how-to", "learn", "beginner"]),
        ("faq", ["question", "answer", "common", "frequently", "asked"]),
        ("troubleshooting", ["error", "fix", "solution", "problem", "workaround"]),
        ("reference", ["documentation", "API", "parameter", "method", "example"]),
        ("howto", ["configure", "setup", "install", "enable", "disable"]),
    ],
    DocumentCategory.NEWS_BLOG: [
        ("tech_news", ["announced", "launch", "update", "release", "feature"]),
        ("opinion", ["think", "believe", "argue", "perspective", "view"]),
        ("review", ["rating", "pros", "cons", "recommend", "verdict"]),
        ("tutorial_blog", ["today", "show", "learn", "simple", "quick"]),
        ("analysis", ["market", "trend", "growth", "impact", "industry"]),
    ],
}

# Templates for generating distractor content
TEMPLATES = {
    SimilarityLevel.VERY_DIFFERENT: [
        "The history of {topic} dates back to ancient times when people first discovered {keyword1}. "
        "Over centuries, the understanding of {keyword2} evolved significantly. "
        "Modern research has shown that {keyword3} plays a crucial role in {topic}. "
        "Experts agree that without {keyword4}, the field would not have progressed as far.",
        "In a remote village, the tradition of {topic} has been passed down for generations. "
        "The elders speak of {keyword1} with reverence, while {keyword2} remains a mystery. "
        "Young apprentices learn the secrets of {keyword3} through careful observation. "
        "The ritual of {keyword4} is performed during the harvest moon.",
    ],
    SimilarityLevel.SOMEWHAT_SIMILAR: [
        "When working with {topic}, it's important to understand {keyword1}. "
        "Many practitioners overlook {keyword2}, which can lead to issues. "
        "Best practices suggest that {keyword3} should be prioritized. "
        "The relationship between {keyword1} and {keyword4} is often misunderstood.",
        "This guide covers the fundamentals of {topic} for intermediate users. "
        "We'll explore how {keyword1} interacts with {keyword2}. "
        "Pay special attention to {keyword3} as it's frequently a source of errors. "
        "Advanced techniques involving {keyword4} will be covered in the next section.",
    ],
    SimilarityLevel.VERY_SIMILAR: [
        "The {keyword1} system provides comprehensive support for {keyword2}. "
        "Users can configure {keyword3} through the standard interface. "
        "This enables {keyword4} without requiring manual intervention. "
        "The {keyword1} approach is widely adopted in production environments.",
        "Implementing {keyword2} requires careful consideration of {keyword1}. "
        "The {keyword3} pattern ensures proper handling of edge cases. "
        "When {keyword4} is enabled, performance may be affected. "
        "We recommend testing {keyword2} thoroughly before deployment.",
    ],
}


class DistractorGenerator:
    """Generates synthetic distractor documents for benchmarking.

    Can generate distractors using:
    1. Templates with keyword substitution (fast, no API required)
    2. LLM generation (higher quality, requires API)

    Example:
        generator = DistractorGenerator()

        # Generate template-based distractors
        distractors = generator.generate(
            count=25,
            categories=[DocumentCategory.TECHNICAL, DocumentCategory.NARRATIVE],
            similarity_distribution={"very_different": 0.5, "somewhat_similar": 0.3, "very_similar": 0.2},
        )

        # Generate LLM-based distractors (higher quality)
        generator_with_llm = DistractorGenerator(llm_provider=my_llm)
        distractors = await generator_with_llm.generate_with_llm(
            count=25,
            reference_docs=corpus.documents[:10],  # Use real docs as reference
        )
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            llm_provider: Optional LLM for high-quality generation
            seed: Random seed for reproducibility
        """
        self.llm_provider = llm_provider
        self.rng = random.Random(seed)

    def generate(
        self,
        count: int,
        categories: list[DocumentCategory] | None = None,
        similarity_distribution: dict[str, float] | None = None,
    ) -> list[BenchmarkDocument]:
        """Generate template-based distractor documents.

        Args:
            count: Number of distractors to generate
            categories: Categories to sample from (default: all except SYNTHETIC)
            similarity_distribution: Distribution of similarity levels
                (default: {"very_different": 0.5, "somewhat_similar": 0.3, "very_similar": 0.2})

        Returns:
            List of BenchmarkDocument instances marked as distractors
        """
        if categories is None:
            categories = [cat for cat in DocumentCategory if cat != DocumentCategory.SYNTHETIC]

        if similarity_distribution is None:
            similarity_distribution = {
                "very_different": 0.5,
                "somewhat_similar": 0.3,
                "very_similar": 0.2,
            }

        # Calculate counts per similarity level
        level_counts = self._calculate_distribution(count, similarity_distribution)

        documents: list[BenchmarkDocument] = []
        doc_index = 0

        for level_name, level_count in level_counts.items():
            level = SimilarityLevel(level_name)
            for _ in range(level_count):
                category = self.rng.choice(categories)
                doc = self._generate_single(
                    doc_index=doc_index,
                    category=category,
                    similarity_level=level,
                )
                documents.append(doc)
                doc_index += 1

        logger.info(f"Generated {len(documents)} distractor documents")
        return documents

    async def generate_with_llm(
        self,
        count: int,
        reference_docs: list[BenchmarkDocument] | None = None,
        categories: list[DocumentCategory] | None = None,
        similarity_distribution: dict[str, float] | None = None,
    ) -> list[BenchmarkDocument]:
        """Generate LLM-based distractor documents.

        Args:
            count: Number of distractors to generate
            reference_docs: Reference documents for context-aware generation
            categories: Categories to sample from
            similarity_distribution: Distribution of similarity levels

        Returns:
            List of BenchmarkDocument instances marked as distractors
        """
        if self.llm_provider is None:
            raise ValueError("LLM provider required for generate_with_llm")

        if categories is None:
            categories = [cat for cat in DocumentCategory if cat != DocumentCategory.SYNTHETIC]

        if similarity_distribution is None:
            similarity_distribution = {
                "very_different": 0.5,
                "somewhat_similar": 0.3,
                "very_similar": 0.2,
            }

        level_counts = self._calculate_distribution(count, similarity_distribution)

        documents: list[BenchmarkDocument] = []
        doc_index = 0

        for level_name, level_count in level_counts.items():
            level = SimilarityLevel(level_name)
            for _ in range(level_count):
                category = self.rng.choice(categories)

                # Get reference doc if available
                reference = None
                if reference_docs and level != SimilarityLevel.VERY_DIFFERENT:
                    # For similar distractors, use reference from same category if possible
                    same_category = [d for d in reference_docs if d.category == category]
                    if same_category:
                        reference = self.rng.choice(same_category)
                    else:
                        reference = self.rng.choice(reference_docs)

                doc = await self._generate_with_llm_single(
                    doc_index=doc_index,
                    category=category,
                    similarity_level=level,
                    reference_doc=reference,
                )
                documents.append(doc)
                doc_index += 1

        logger.info(f"Generated {len(documents)} LLM-based distractor documents")
        return documents

    def _calculate_distribution(
        self,
        count: int,
        distribution: dict[str, float],
    ) -> dict[str, int]:
        """Calculate counts for each similarity level.

        Args:
            count: Total documents to generate
            distribution: Probability distribution

        Returns:
            Dict of level -> count
        """
        # Normalize distribution
        total = sum(distribution.values())
        normalized = {k: v / total for k, v in distribution.items()}

        # Calculate counts
        counts = {}
        remaining = count
        for level, prob in normalized.items():
            level_count = int(count * prob)
            counts[level] = level_count
            remaining -= level_count

        # Distribute remaining to first levels
        for level in counts:
            if remaining <= 0:
                break
            counts[level] += 1
            remaining -= 1

        return counts

    def _generate_single(
        self,
        doc_index: int,
        category: DocumentCategory,
        similarity_level: SimilarityLevel,
    ) -> BenchmarkDocument:
        """Generate a single template-based distractor.

        Args:
            doc_index: Index for unique ID
            category: Document category
            similarity_level: Similarity level

        Returns:
            BenchmarkDocument instance
        """
        # Get topic and keywords for category
        if category in CATEGORY_TOPICS:
            topic, keywords = self.rng.choice(CATEGORY_TOPICS[category])
        else:
            # Fallback for SYNTHETIC or unknown categories
            topic = "general topic"
            keywords = ["concept", "idea", "method", "approach", "technique"]

        # Extend keywords if needed
        while len(keywords) < 4:
            keywords = keywords + keywords

        # Select template
        templates = TEMPLATES[similarity_level]
        template = self.rng.choice(templates)

        # Generate content
        content = template.format(
            topic=topic,
            keyword1=keywords[0],
            keyword2=keywords[1],
            keyword3=keywords[2],
            keyword4=keywords[3],
        )

        # Add more content for realistic document length
        for _ in range(self.rng.randint(2, 5)):
            additional_template = self.rng.choice(templates)
            shuffled_keywords = keywords.copy()
            self.rng.shuffle(shuffled_keywords)
            content += " " + additional_template.format(
                topic=topic,
                keyword1=shuffled_keywords[0],
                keyword2=shuffled_keywords[1],
                keyword3=shuffled_keywords[2] if len(shuffled_keywords) > 2 else shuffled_keywords[0],
                keyword4=shuffled_keywords[3] if len(shuffled_keywords) > 3 else shuffled_keywords[1],
            )

        doc_id = f"distractor_{doc_index:04d}"
        tags = [
            "distractor",
            similarity_level.value,
            topic.replace(" ", "_"),
        ]

        return BenchmarkDocument(
            doc_id=doc_id,
            source=DocumentSource.SYNTHETIC,
            category=DocumentCategory.SYNTHETIC,
            domain=f"synthetic_{category.value}",
            file_path=f"synthetic://distractor/{doc_id}",
            content=content,
            is_distractor=True,
            semantic_tags=tags,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "similarity_level": similarity_level.value,
                "base_category": category.value,
                "topic": topic,
                "generator": "template",
            },
        )

    async def _generate_with_llm_single(
        self,
        doc_index: int,
        category: DocumentCategory,
        similarity_level: SimilarityLevel,
        reference_doc: BenchmarkDocument | None = None,
    ) -> BenchmarkDocument:
        """Generate a single LLM-based distractor.

        Args:
            doc_index: Index for unique ID
            category: Document category
            similarity_level: Similarity level
            reference_doc: Optional reference document for context

        Returns:
            BenchmarkDocument instance
        """
        prompt = self._build_llm_prompt(category, similarity_level, reference_doc)

        messages = [
            {"role": "system", "content": "You are a content generator creating diverse document samples for benchmarking retrieval systems."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm_provider.chat(messages)
            content = response.get("content", "")

            # Clean up response if it has extra formatting
            content = content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to template")
            return self._generate_single(doc_index, category, similarity_level)

        doc_id = f"distractor_{doc_index:04d}"
        tags = [
            "distractor",
            similarity_level.value,
            "llm_generated",
        ]

        return BenchmarkDocument(
            doc_id=doc_id,
            source=DocumentSource.SYNTHETIC,
            category=DocumentCategory.SYNTHETIC,
            domain=f"synthetic_{category.value}",
            file_path=f"synthetic://distractor/{doc_id}",
            content=content,
            is_distractor=True,
            semantic_tags=tags,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "similarity_level": similarity_level.value,
                "base_category": category.value,
                "generator": "llm",
                "reference_doc_id": reference_doc.doc_id if reference_doc else None,
            },
        )

    def _build_llm_prompt(
        self,
        category: DocumentCategory,
        similarity_level: SimilarityLevel,
        reference_doc: BenchmarkDocument | None,
    ) -> str:
        """Build prompt for LLM generation.

        Args:
            category: Document category
            similarity_level: Similarity level
            reference_doc: Optional reference document

        Returns:
            Prompt string
        """
        category_desc = {
            DocumentCategory.TECHNICAL: "technical documentation about software, APIs, or programming",
            DocumentCategory.NARRATIVE: "creative writing, stories, or narrative prose",
            DocumentCategory.LEGAL: "legal documents, contracts, or terms of service",
            DocumentCategory.ACADEMIC: "academic or research-style writing with formal tone",
            DocumentCategory.CONVERSATIONAL: "informal chat, email, or support ticket style",
            DocumentCategory.KNOWLEDGE_BASE: "FAQ, how-to, or troubleshooting content",
            DocumentCategory.NEWS_BLOG: "blog post or news article style",
        }

        level_desc = {
            SimilarityLevel.VERY_DIFFERENT: (
                "completely unrelated topic with different vocabulary and domain. "
                "This should be clearly irrelevant to the reference content."
            ),
            SimilarityLevel.SOMEWHAT_SIMILAR: (
                "similar domain but different specific topic. "
                "It should share some general themes but discuss different specifics."
            ),
            SimilarityLevel.VERY_SIMILAR: (
                "very similar keywords and style but discussing something different. "
                "This is a 'hard negative' - it looks relevant but isn't about the same thing."
            ),
        }

        desc = category_desc.get(category, "general content")
        level = level_desc[similarity_level]

        if reference_doc:
            reference_snippet = reference_doc.content[:500]
            prompt = f"""Generate a document that is {category.value} content ({desc}).

The document should be {level}

For reference, here is an example of content in the corpus:
---
{reference_snippet}
---

Generate a document of approximately 300-500 words. Do not include any meta-commentary or explanations, just the document content."""
        else:
            prompt = f"""Generate a document that is {category.value} content ({desc}).

The document should be {level}

Generate a document of approximately 300-500 words. Do not include any meta-commentary or explanations, just the document content."""

        return prompt
