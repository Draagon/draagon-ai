"""Word Sense Disambiguation (WSD) System.

Provides comprehensive word sense disambiguation using:
1. WordNet synsets for canonical identifiers
2. Lesk algorithm for gloss-based disambiguation
3. Extended Lesk with hypernym/hyponym glosses
4. LLM-based disambiguation for complex/ambiguous cases
5. Hybrid pipeline with evolvable thresholds

Key Principles:
- WSD is FOUNDATIONAL - must happen before any other processing
- Hybrid approach: fast algorithms first, LLM fallback when uncertain
- All thresholds and parameters are evolvable
- XML output format for LLM responses

Example:
    >>> from wsd import WordSenseDisambiguator, WSDConfig
    >>>
    >>> config = WSDConfig()
    >>> wsd = WordSenseDisambiguator(config)
    >>>
    >>> # Disambiguate a word in context
    >>> result = await wsd.disambiguate("bank", "I deposited money in the bank")
    >>> print(result.synset_id)  # "bank.n.01"
    >>> print(result.confidence)  # 0.92

Based on research from:
- WordNet (https://wordnet.princeton.edu/)
- Lesk algorithm (dictionary-based WSD)
- Extended Lesk (Banerjee & Pedersen, 2002)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from identifiers import SynsetInfo, UniversalSemanticIdentifier, EntityType

if TYPE_CHECKING:
    pass


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
    ) -> Any:
        """Send chat messages and get response.

        Returns:
            Either a string directly, or a ChatResponse object with .content attribute.
        """
        ...


def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response.

    Handles both direct string returns (from simple/mock providers)
    and ChatResponse objects (from real providers like GroqLLM).
    """
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)


# =============================================================================
# Evolvable Configuration
# =============================================================================


@dataclass
class WSDConfig:
    """Evolvable configuration for Word Sense Disambiguation.

    All parameters can be evolved to optimize WSD accuracy vs cost.

    Attributes:
        lesk_context_window: Number of words around target to use as context
        lesk_extended_depth: Depth of hypernym/hyponym traversal for Extended Lesk
        lesk_use_examples: Whether to include synset examples in gloss
        lesk_high_confidence: Threshold for accepting Lesk result without LLM
        embedding_agreement_boost: Confidence boost when methods agree
        llm_fallback_threshold: Threshold below which to use LLM
        wsd_prompt_template: The prompt template for LLM disambiguation
        wsd_temperature: Temperature for LLM calls
        max_synset_candidates: Maximum synsets to consider
    """

    # Lesk parameters
    lesk_context_window: int = 10
    lesk_extended_depth: int = 1
    lesk_use_examples: bool = True

    # Threshold parameters
    lesk_high_confidence: float = 0.8
    embedding_agreement_boost: float = 0.1
    llm_fallback_threshold: float = 0.5

    # LLM parameters
    wsd_prompt_template: str = """Determine which meaning of '{word}' is used in this sentence:

Sentence: "{sentence}"

Possible meanings:
{options}

Consider the context carefully. Choose the sense that best fits how the word is used.

Respond in XML:
<disambiguation>
    <synset_id>The WordNet synset ID (e.g., bank.n.01)</synset_id>
    <confidence>0.0-1.0 based on how certain you are</confidence>
    <reasoning>Brief explanation of why this sense fits</reasoning>
</disambiguation>"""
    wsd_temperature: float = 0.1
    max_synset_candidates: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for evolution/storage."""
        return {
            "lesk_context_window": self.lesk_context_window,
            "lesk_extended_depth": self.lesk_extended_depth,
            "lesk_use_examples": self.lesk_use_examples,
            "lesk_high_confidence": self.lesk_high_confidence,
            "embedding_agreement_boost": self.embedding_agreement_boost,
            "llm_fallback_threshold": self.llm_fallback_threshold,
            "wsd_prompt_template": self.wsd_prompt_template,
            "wsd_temperature": self.wsd_temperature,
            "max_synset_candidates": self.max_synset_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WSDConfig:
        """Deserialize config from dict."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# Disambiguation Result
# =============================================================================


@dataclass
class DisambiguationResult:
    """Result of word sense disambiguation.

    Attributes:
        word: The original word (surface form)
        lemma: The lemmatized form
        pos: Part of speech
        synset_id: The disambiguated WordNet synset ID
        definition: The synset definition
        confidence: Confidence in the disambiguation (0.0-1.0)
        method: The method used (unambiguous, lesk, extended_lesk, llm)
        alternatives: Other synset IDs considered
        llm_used: Whether LLM was invoked
        reasoning: Explanation (if from LLM)
    """

    word: str
    lemma: str
    pos: str
    synset_id: str
    definition: str = ""
    confidence: float = 1.0
    method: str = "unambiguous"
    alternatives: list[str] = field(default_factory=list)
    llm_used: bool = False
    reasoning: str = ""

    def to_identifier(
        self,
        domain: str | None = None,
        hypernym_chain: list[str] | None = None,
    ) -> UniversalSemanticIdentifier:
        """Convert to a UniversalSemanticIdentifier.

        Args:
            domain: Optional semantic domain
            hypernym_chain: Optional hypernym chain

        Returns:
            A new UniversalSemanticIdentifier for this sense
        """
        from identifiers import create_class_identifier

        return create_class_identifier(
            synset_id=self.synset_id,
            definition=self.definition,
            domain=domain,
            hypernym_chain=hypernym_chain,
            confidence=self.confidence,
        )


# =============================================================================
# Mock Synset Database (for when NLTK is unavailable)
# =============================================================================


@dataclass
class MockSynset:
    """Mock synset for when NLTK is not available."""

    name: str
    pos: str
    lemmas: list[str]
    _definition: str = ""
    _examples: list[str] = field(default_factory=list)
    _hypernyms: list[str] = field(default_factory=list)
    _hyponyms: list[str] = field(default_factory=list)

    def definition(self) -> str:
        return self._definition

    def examples(self) -> list[str]:
        return self._examples

    def lemma_names(self) -> list[str]:
        return self.lemmas


# Common synsets for testing without NLTK
COMMON_SYNSETS: dict[str, list[MockSynset]] = {
    "bank": [
        MockSynset(
            name="bank.n.01",
            pos="n",
            lemmas=["bank", "banking_company", "banking_concern", "depository_financial_institution"],
            _definition="a financial institution that accepts deposits and channels the money into lending activities",
            _examples=["he cashed a check at the bank", "that bank holds the mortgage on my home"],
            _hypernyms=["financial_institution.n.01"],
        ),
        MockSynset(
            name="bank.n.02",
            pos="n",
            lemmas=["bank"],
            _definition="sloping land (especially the slope beside a body of water)",
            _examples=["they pulled the canoe up on the bank", "he sat on the bank of the river and watched the currents"],
            _hypernyms=["slope.n.01"],
        ),
        MockSynset(
            name="bank.n.03",
            pos="n",
            lemmas=["bank", "coin_bank", "money_box", "savings_bank"],
            _definition="a container (usually with a slot in the top) for keeping money at home",
            _examples=["the child had a piggy bank"],
            _hypernyms=["container.n.01"],
        ),
        MockSynset(
            name="bank.n.04",
            pos="n",
            lemmas=["bank"],
            _definition="a supply or stock held in reserve for future use (especially in emergencies)",
            _examples=["a blood bank"],
            _hypernyms=["reserve.n.01"],
        ),
        MockSynset(
            name="bank.v.01",
            pos="v",
            lemmas=["bank"],
            _definition="tip laterally",
            _examples=["the pilot had to bank the aircraft"],
            _hypernyms=["tip.v.01"],
        ),
    ],
    "bass": [
        MockSynset(
            name="bass.n.01",
            pos="n",
            lemmas=["bass", "freshwater_bass"],
            _definition="the lean flesh of a saltwater fish of the family Serranidae",
            _examples=["he caught a large bass"],
            _hypernyms=["fish.n.01"],
        ),
        MockSynset(
            name="bass.n.02",
            pos="n",
            lemmas=["bass"],
            _definition="any of various North American freshwater fish with lean flesh",
            _examples=["largemouth bass"],
            _hypernyms=["freshwater_fish.n.01"],
        ),
        MockSynset(
            name="bass.n.07",
            pos="n",
            lemmas=["bass", "bass_voice", "basso"],
            _definition="the lowest part of the musical range",
            _examples=["the bass line holds the song together"],
            _hypernyms=["voice.n.01"],
        ),
        MockSynset(
            name="bass.n.08",
            pos="n",
            lemmas=["bass", "bass_part"],
            _definition="the lowest part in polyphonic music",
            _examples=["he sang the bass"],
            _hypernyms=["part.n.01"],
        ),
    ],
    "apple": [
        MockSynset(
            name="apple.n.01",
            pos="n",
            lemmas=["apple"],
            _definition="fruit with red or yellow or green skin and sweet to tart crisp whitish flesh",
            _examples=["I ate an apple for lunch"],
            _hypernyms=["fruit.n.01"],
        ),
        MockSynset(
            name="apple.n.02",
            pos="n",
            lemmas=["apple", "apple_tree", "Malus_pumila"],
            _definition="native Eurasian tree widely cultivated in many varieties for its firm rounded edible fruits",
            _examples=["the apple tree in our backyard"],
            _hypernyms=["tree.n.01"],
        ),
    ],
    "tea": [
        MockSynset(
            name="tea.n.01",
            pos="n",
            lemmas=["tea"],
            _definition="a beverage made by steeping tea leaves in water",
            _examples=["iced tea is a cooling drink"],
            _hypernyms=["beverage.n.01"],
        ),
        MockSynset(
            name="tea.n.02",
            pos="n",
            lemmas=["tea", "afternoon_tea", "teatime"],
            _definition="a light midafternoon meal of tea and sandwiches or cakes",
            _examples=["an Englishman would interrupt a war to have his afternoon tea"],
            _hypernyms=["meal.n.01"],
        ),
    ],
    "morning": [
        MockSynset(
            name="morning.n.01",
            pos="n",
            lemmas=["morning", "morn", "forenoon"],
            _definition="the time period between dawn and noon",
            _examples=["I spent the morning jogging on the beach"],
            _hypernyms=["time_period.n.01"],
        ),
    ],
    "money": [
        MockSynset(
            name="money.n.01",
            pos="n",
            lemmas=["money"],
            _definition="the most common medium of exchange; functions as legal tender",
            _examples=["we tried to collect the money he owed us"],
            _hypernyms=["currency.n.01"],
        ),
    ],
    "deposit": [
        MockSynset(
            name="deposit.v.01",
            pos="v",
            lemmas=["deposit", "lodge", "stick", "wedge"],
            _definition="put into a bank account",
            _examples=["She deposited her paycheck every month"],
            _hypernyms=["put.v.01"],
        ),
        MockSynset(
            name="deposit.v.02",
            pos="v",
            lemmas=["deposit"],
            _definition="put, fix, force, or implant",
            _examples=["lodge a bullet in the table"],
            _hypernyms=["fasten.v.01"],
        ),
    ],
    "river": [
        MockSynset(
            name="river.n.01",
            pos="n",
            lemmas=["river"],
            _definition="a large natural stream of water (larger than a creek)",
            _examples=["the river was navigable for 50 miles"],
            _hypernyms=["stream.n.01"],
        ),
    ],
    "lake": [
        MockSynset(
            name="lake.n.01",
            pos="n",
            lemmas=["lake"],
            _definition="a body of (usually fresh) water surrounded by land",
            _examples=["we went swimming in the lake"],
            _hypernyms=["body_of_water.n.01"],
        ),
    ],
    "song": [
        MockSynset(
            name="song.n.01",
            pos="n",
            lemmas=["song", "vocal"],
            _definition="a short musical composition with words",
            _examples=["she was humming a song"],
            _hypernyms=["musical_composition.n.01"],
        ),
    ],
    "line": [
        MockSynset(
            name="line.n.01",
            pos="n",
            lemmas=["line"],
            _definition="a mark that is long relative to its width",
            _examples=["he drew a line on the chart"],
            _hypernyms=["mark.n.01"],
        ),
        MockSynset(
            name="line.n.05",
            pos="n",
            lemmas=["line"],
            _definition="a formation of people or things one beside another",
            _examples=["the line of soldiers awaited the order"],
            _hypernyms=["formation.n.01"],
        ),
    ],
    "catch": [
        MockSynset(
            name="catch.v.01",
            pos="v",
            lemmas=["catch", "get"],
            _definition="discover or come upon accidentally, suddenly, or unexpectedly",
            _examples=["he caught his son smoking"],
            _hypernyms=["discover.v.01"],
        ),
        MockSynset(
            name="catch.v.02",
            pos="v",
            lemmas=["catch"],
            _definition="perceive with the senses quickly, suddenly, or momentarily",
            _examples=["I caught the aroma of coffee"],
            _hypernyms=["perceive.v.01"],
        ),
        MockSynset(
            name="catch.v.10",
            pos="v",
            lemmas=["catch", "capture"],
            _definition="capture as if by hunting, snaring, or trapping",
            _examples=["I caught a rabbit in the trap today"],
            _hypernyms=["seize.v.01"],
        ),
    ],
    "walk": [
        MockSynset(
            name="walk.v.01",
            pos="v",
            lemmas=["walk"],
            _definition="use one's feet to advance; advance by steps",
            _examples=["we walked through the park"],
            _hypernyms=["travel.v.01"],
        ),
    ],
}


# =============================================================================
# WordNet Interface
# =============================================================================


class WordNetInterface:
    """Interface for WordNet synset operations.

    Provides a unified API for WordNet access, falling back to mock data
    when NLTK WordNet is unavailable.
    """

    def __init__(self, use_nltk: bool = True):
        """Initialize the WordNet interface.

        Args:
            use_nltk: Whether to attempt using NLTK WordNet
        """
        self.use_nltk = use_nltk
        self._wordnet = None

        if use_nltk:
            try:
                from nltk.corpus import wordnet
                # Test that the corpus is actually available
                wordnet.synsets("test")
                self._wordnet = wordnet
            except (ImportError, LookupError):
                self.use_nltk = False

    def get_synsets(self, word: str, pos: str | None = None) -> list[SynsetInfo]:
        """Get synsets for a word.

        Args:
            word: The word to look up (lemmatized)
            pos: Part of speech filter (n, v, a, r)

        Returns:
            List of SynsetInfo objects
        """
        if self.use_nltk and self._wordnet:
            synsets = self._wordnet.synsets(word, pos=pos)
            return [self._synset_to_info(s) for s in synsets]
        else:
            # Use mock synsets
            mock_synsets = COMMON_SYNSETS.get(word.lower(), [])
            if pos:
                mock_synsets = [s for s in mock_synsets if s.pos == pos]
            return [self._mock_synset_to_info(s) for s in mock_synsets]

    def get_synset_by_id(self, synset_id: str) -> SynsetInfo | None:
        """Get a specific synset by ID.

        Args:
            synset_id: The synset ID (e.g., "bank.n.01")

        Returns:
            SynsetInfo or None if not found
        """
        if self.use_nltk and self._wordnet:
            try:
                synset = self._wordnet.synset(synset_id)
                return self._synset_to_info(synset)
            except Exception:
                return None
        else:
            # Search mock synsets
            word = synset_id.split(".")[0] if "." in synset_id else synset_id
            for mock in COMMON_SYNSETS.get(word.lower(), []):
                if mock.name == synset_id:
                    return self._mock_synset_to_info(mock)
            return None

    def get_hypernym_chain(self, synset_id: str, max_depth: int = 10) -> list[str]:
        """Get the hypernym chain for a synset.

        Args:
            synset_id: The synset ID
            max_depth: Maximum depth to traverse

        Returns:
            List of hypernym synset IDs from most specific to most general
        """
        chain = []

        if self.use_nltk and self._wordnet:
            try:
                synset = self._wordnet.synset(synset_id)
                current = synset
                for _ in range(max_depth):
                    hypernyms = current.hypernyms()
                    if not hypernyms:
                        break
                    # Take first hypernym (most common)
                    current = hypernyms[0]
                    chain.append(current.name())
            except Exception:
                pass
        else:
            # Use mock hypernyms
            word = synset_id.split(".")[0] if "." in synset_id else synset_id
            for mock in COMMON_SYNSETS.get(word.lower(), []):
                if mock.name == synset_id:
                    chain.extend(mock._hypernyms)
                    break

        return chain

    def _synset_to_info(self, synset: Any) -> SynsetInfo:
        """Convert NLTK synset to SynsetInfo."""
        return SynsetInfo(
            synset_id=synset.name(),
            pos=synset.pos(),
            lemmas=[l.name() for l in synset.lemmas()],
            definition=synset.definition(),
            examples=synset.examples(),
            hypernyms=[h.name() for h in synset.hypernyms()],
            hyponyms=[h.name() for h in synset.hyponyms()],
        )

    def _mock_synset_to_info(self, mock: MockSynset) -> SynsetInfo:
        """Convert mock synset to SynsetInfo."""
        return SynsetInfo(
            synset_id=mock.name,
            pos=mock.pos,
            lemmas=mock.lemmas,
            definition=mock._definition,
            examples=mock._examples,
            hypernyms=mock._hypernyms,
            hyponyms=mock._hyponyms,
        )


# =============================================================================
# Lesk Disambiguator
# =============================================================================


class LeskDisambiguator:
    """Word sense disambiguation using the Lesk algorithm.

    The Lesk algorithm disambiguates by comparing:
    - The context words around the target word
    - The gloss (definition + examples) of each candidate synset

    The synset with the most word overlap wins.

    Extended Lesk also includes glosses of related synsets (hypernyms, hyponyms).
    """

    # Stopwords to filter from context and glosses
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "or", "and", "that", "this", "it", "as", "not", "but", "so",
        "if", "when", "then", "than", "too", "very", "just", "only",
    }

    def __init__(self, wordnet: WordNetInterface, config: WSDConfig):
        """Initialize the Lesk disambiguator.

        Args:
            wordnet: WordNet interface for synset access
            config: WSD configuration
        """
        self.wordnet = wordnet
        self.config = config

    def disambiguate(
        self,
        word: str,
        context: list[str],
        pos: str | None = None,
    ) -> DisambiguationResult | None:
        """Disambiguate a word using the basic Lesk algorithm.

        Args:
            word: The word to disambiguate (lemmatized)
            context: Context words from the sentence
            pos: Part of speech

        Returns:
            DisambiguationResult or None if no synsets found
        """
        synsets = self.wordnet.get_synsets(word, pos)

        if not synsets:
            return None

        if len(synsets) == 1:
            # Unambiguous
            syn = synsets[0]
            return DisambiguationResult(
                word=word,
                lemma=word,
                pos=pos or syn.pos,
                synset_id=syn.synset_id,
                definition=syn.definition,
                confidence=1.0,
                method="unambiguous",
            )

        # Score each synset by context overlap
        context_set = set(w.lower() for w in context if w.lower() not in self.STOPWORDS)
        best_synset = None
        best_score = -1
        scores: dict[str, int] = {}

        for syn in synsets:
            # Get gloss words
            gloss_words = self._get_gloss_words(syn)
            overlap = len(context_set & gloss_words)
            scores[syn.synset_id] = overlap

            if overlap > best_score:
                best_score = overlap
                best_synset = syn

        if best_synset is None:
            best_synset = synsets[0]  # Fall back to first sense

        # Compute confidence based on score margin
        if best_score == 0:
            confidence = 0.3  # No overlap, low confidence
        else:
            # Check margin over second best
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                margin = best_score - sorted_scores[1]
                confidence = min(1.0, 0.5 + (margin * 0.15))
            else:
                confidence = min(1.0, 0.4 + (best_score * 0.1))

        return DisambiguationResult(
            word=word,
            lemma=word,
            pos=pos or best_synset.pos,
            synset_id=best_synset.synset_id,
            definition=best_synset.definition,
            confidence=confidence,
            method="lesk",
            alternatives=[s.synset_id for s in synsets if s.synset_id != best_synset.synset_id],
        )

    def extended_disambiguate(
        self,
        word: str,
        context: list[str],
        pos: str | None = None,
    ) -> DisambiguationResult | None:
        """Disambiguate using Extended Lesk algorithm.

        Extends basic Lesk by including glosses of related synsets
        (hypernyms and hyponyms up to configured depth).

        Args:
            word: The word to disambiguate
            context: Context words from the sentence
            pos: Part of speech

        Returns:
            DisambiguationResult or None if no synsets found
        """
        synsets = self.wordnet.get_synsets(word, pos)

        if not synsets:
            return None

        if len(synsets) == 1:
            syn = synsets[0]
            return DisambiguationResult(
                word=word,
                lemma=word,
                pos=pos or syn.pos,
                synset_id=syn.synset_id,
                definition=syn.definition,
                confidence=1.0,
                method="unambiguous",
            )

        # Score each synset by extended context overlap
        context_set = set(w.lower() for w in context if w.lower() not in self.STOPWORDS)
        best_synset = None
        best_score = -1
        scores: dict[str, float] = {}

        for syn in synsets:
            # Get extended gloss words (including related synsets)
            gloss_words = self._get_extended_gloss_words(syn)
            overlap = len(context_set & gloss_words)

            # Weight hypernym/hyponym matches slightly less
            base_overlap = len(context_set & self._get_gloss_words(syn))
            extended_overlap = overlap - base_overlap
            weighted_score = base_overlap + (extended_overlap * 0.7)

            scores[syn.synset_id] = weighted_score

            if weighted_score > best_score:
                best_score = weighted_score
                best_synset = syn

        if best_synset is None:
            best_synset = synsets[0]

        # Compute confidence with better margins
        if best_score == 0:
            confidence = 0.35
        else:
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                margin = best_score - sorted_scores[1]
                relative_margin = margin / best_score if best_score > 0 else 0
                confidence = min(1.0, 0.5 + (relative_margin * 0.4) + (best_score * 0.05))
            else:
                confidence = min(1.0, 0.45 + (best_score * 0.1))

        return DisambiguationResult(
            word=word,
            lemma=word,
            pos=pos or best_synset.pos,
            synset_id=best_synset.synset_id,
            definition=best_synset.definition,
            confidence=confidence,
            method="extended_lesk",
            alternatives=[s.synset_id for s in synsets if s.synset_id != best_synset.synset_id],
        )

    def _get_gloss_words(self, synset: SynsetInfo) -> set[str]:
        """Get words from synset definition and examples."""
        words = set(self._tokenize(synset.definition))

        if self.config.lesk_use_examples:
            for example in synset.examples:
                words.update(self._tokenize(example))

        return words

    def _get_extended_gloss_words(self, synset: SynsetInfo) -> set[str]:
        """Get words from synset and related synsets."""
        words = self._get_gloss_words(synset)

        # Add hypernym glosses
        for hypernym_id in synset.hypernyms[:self.config.lesk_extended_depth]:
            hyp_info = self.wordnet.get_synset_by_id(hypernym_id)
            if hyp_info:
                words.update(self._get_gloss_words(hyp_info))

        # Add hyponym glosses
        for hyponym_id in synset.hyponyms[:self.config.lesk_extended_depth]:
            hypo_info = self.wordnet.get_synset_by_id(hyponym_id)
            if hypo_info:
                words.update(self._get_gloss_words(hypo_info))

        return words

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize and filter text."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()
        return [w for w in words if w not in self.STOPWORDS and len(w) > 2]


# =============================================================================
# LLM Disambiguator
# =============================================================================


class LLMDisambiguator:
    """LLM-based word sense disambiguation.

    Uses an LLM to disambiguate words when algorithmic methods are uncertain.
    Follows draagon-ai's XML output format pattern.
    """

    def __init__(
        self,
        llm: LLMProvider,
        wordnet: WordNetInterface,
        config: WSDConfig,
    ):
        """Initialize the LLM disambiguator.

        Args:
            llm: LLM provider for chat completions
            wordnet: WordNet interface for synset data
            config: WSD configuration
        """
        self.llm = llm
        self.wordnet = wordnet
        self.config = config

    async def disambiguate(
        self,
        word: str,
        sentence: str,
        candidates: list[SynsetInfo] | None = None,
        pos: str | None = None,
    ) -> DisambiguationResult | None:
        """Disambiguate a word using LLM.

        Args:
            word: The word to disambiguate
            sentence: The full sentence for context
            candidates: Optional pre-filtered candidate synsets
            pos: Part of speech

        Returns:
            DisambiguationResult or None if failed
        """
        if candidates is None:
            candidates = self.wordnet.get_synsets(word, pos)

        if not candidates:
            return None

        if len(candidates) == 1:
            syn = candidates[0]
            return DisambiguationResult(
                word=word,
                lemma=word,
                pos=pos or syn.pos,
                synset_id=syn.synset_id,
                definition=syn.definition,
                confidence=1.0,
                method="unambiguous",
            )

        # Build options for prompt
        candidates = candidates[:self.config.max_synset_candidates]
        options = "\n".join([
            f"- {syn.synset_id}: {syn.definition}"
            for syn in candidates
        ])

        # Build prompt from template
        prompt = self.config.wsd_prompt_template.format(
            word=word,
            sentence=sentence,
            options=options,
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.wsd_temperature,
            )

            content = _extract_llm_content(response)
            return self._parse_response(content, word, pos, candidates)

        except Exception:
            return None

    def _parse_response(
        self,
        response: str,
        word: str,
        pos: str | None,
        candidates: list[SynsetInfo],
    ) -> DisambiguationResult | None:
        """Parse LLM disambiguation response."""
        # Extract synset_id
        synset_match = re.search(r"<synset_id>([^<]+)</synset_id>", response)
        if not synset_match:
            return None

        synset_id = synset_match.group(1).strip()

        # Validate synset_id
        valid_ids = {s.synset_id for s in candidates}
        if synset_id not in valid_ids:
            # Try to match partial
            for valid_id in valid_ids:
                if synset_id in valid_id or valid_id in synset_id:
                    synset_id = valid_id
                    break
            else:
                # Default to first candidate
                synset_id = candidates[0].synset_id

        # Extract confidence
        conf_match = re.search(r"<confidence>([0-9.]+)</confidence>", response)
        confidence = float(conf_match.group(1)) if conf_match else 0.8

        # Extract reasoning
        reason_match = re.search(r"<reasoning>([^<]+)</reasoning>", response)
        reasoning = reason_match.group(1).strip() if reason_match else ""

        # Find matching synset for definition
        definition = ""
        for syn in candidates:
            if syn.synset_id == synset_id:
                definition = syn.definition
                break

        return DisambiguationResult(
            word=word,
            lemma=word,
            pos=pos or "",
            synset_id=synset_id,
            definition=definition,
            confidence=confidence,
            method="llm",
            alternatives=[s.synset_id for s in candidates if s.synset_id != synset_id],
            llm_used=True,
            reasoning=reasoning,
        )


# =============================================================================
# Hybrid Disambiguator (Main Class)
# =============================================================================


class WordSenseDisambiguator:
    """Hybrid Word Sense Disambiguation system.

    Combines multiple methods in an efficient pipeline:
    1. Single synset → Return immediately (confidence=1.0)
    2. Extended Lesk → High confidence? → Return
    3. LLM fallback → Return LLM result

    All thresholds are evolvable via WSDConfig.

    Example:
        >>> config = WSDConfig(llm_fallback_threshold=0.6)
        >>> wsd = WordSenseDisambiguator(config, llm=my_llm)
        >>>
        >>> result = await wsd.disambiguate("bank", "I deposited money at the bank")
        >>> print(result.synset_id)  # "bank.n.01"
        >>> print(result.method)  # "extended_lesk" or "llm"
    """

    # Map common POS tags to WordNet POS
    POS_MAP = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
        "PROPN": "n",
        "n": "n",
        "v": "v",
        "a": "a",
        "r": "r",
    }

    def __init__(
        self,
        config: WSDConfig | None = None,
        llm: LLMProvider | None = None,
        use_nltk: bool = True,
    ):
        """Initialize the hybrid disambiguator.

        Args:
            config: WSD configuration (uses defaults if not provided)
            llm: LLM provider for fallback disambiguation
            use_nltk: Whether to use NLTK WordNet
        """
        self.config = config or WSDConfig()
        self.llm = llm
        self.wordnet = WordNetInterface(use_nltk=use_nltk)
        self.lesk = LeskDisambiguator(self.wordnet, self.config)

        if llm:
            self.llm_disambiguator = LLMDisambiguator(llm, self.wordnet, self.config)
        else:
            self.llm_disambiguator = None

        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "unambiguous": 0,
            "lesk_accepted": 0,
            "llm_calls": 0,
            "failures": 0,
        }

    async def disambiguate(
        self,
        word: str,
        sentence: str,
        pos: str | None = None,
    ) -> DisambiguationResult | None:
        """Disambiguate a word in context using the hybrid pipeline.

        Args:
            word: The word to disambiguate
            sentence: The full sentence for context
            pos: Part of speech (optional)

        Returns:
            DisambiguationResult or None if cannot disambiguate
        """
        self.metrics["total_calls"] += 1

        # Normalize POS
        wn_pos = self.POS_MAP.get(pos) if pos else None

        # Get synsets
        synsets = self.wordnet.get_synsets(word.lower(), wn_pos)

        if not synsets:
            self.metrics["failures"] += 1
            return None

        # Step 1: Single synset = unambiguous
        if len(synsets) == 1:
            self.metrics["unambiguous"] += 1
            syn = synsets[0]
            return DisambiguationResult(
                word=word,
                lemma=word.lower(),
                pos=wn_pos or syn.pos,
                synset_id=syn.synset_id,
                definition=syn.definition,
                confidence=1.0,
                method="unambiguous",
            )

        # Step 2: Try Extended Lesk
        context = self._extract_context(word, sentence)
        lesk_result = self.lesk.extended_disambiguate(word.lower(), context, wn_pos)

        if lesk_result and lesk_result.confidence >= self.config.lesk_high_confidence:
            self.metrics["lesk_accepted"] += 1
            return lesk_result

        # Step 3: LLM fallback if available and confidence is low
        if (
            self.llm_disambiguator
            and lesk_result
            and lesk_result.confidence < self.config.llm_fallback_threshold
        ):
            self.metrics["llm_calls"] += 1
            llm_result = await self.llm_disambiguator.disambiguate(
                word.lower(),
                sentence,
                candidates=synsets,
                pos=wn_pos,
            )
            if llm_result and llm_result.confidence > 0.5:
                return llm_result

        # Return Lesk result (even if not high confidence)
        if lesk_result:
            self.metrics["lesk_accepted"] += 1
            return lesk_result

        self.metrics["failures"] += 1
        return None

    async def disambiguate_all(
        self,
        sentence: str,
        words: list[str] | None = None,
    ) -> dict[str, DisambiguationResult]:
        """Disambiguate all content words in a sentence.

        Args:
            sentence: The sentence to analyze
            words: Optional list of words to disambiguate (defaults to all content words)

        Returns:
            Dict mapping words to their disambiguation results
        """
        if words is None:
            words = self._extract_content_words(sentence)

        results = {}
        for word in words:
            result = await self.disambiguate(word, sentence)
            if result:
                results[word] = result

        return results

    def get_synsets(self, word: str, pos: str | None = None) -> list[SynsetInfo]:
        """Get synsets for a word (convenience method).

        Args:
            word: The word to look up
            pos: Part of speech filter

        Returns:
            List of SynsetInfo objects
        """
        wn_pos = self.POS_MAP.get(pos) if pos else None
        return self.wordnet.get_synsets(word.lower(), wn_pos)

    def get_hypernym_chain(self, synset_id: str) -> list[str]:
        """Get hypernym chain for a synset (convenience method).

        Args:
            synset_id: The synset ID

        Returns:
            List of hypernym synset IDs
        """
        return self.wordnet.get_hypernym_chain(synset_id)

    def get_metrics(self) -> dict[str, int]:
        """Get disambiguation metrics.

        Returns:
            Dictionary of metric counts
        """
        return dict(self.metrics)

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0

    def _extract_context(self, word: str, sentence: str) -> list[str]:
        """Extract context words around target word."""
        words = re.findall(r"\b\w+\b", sentence.lower())
        word_lower = word.lower()

        # Find target word position
        try:
            pos = words.index(word_lower)
        except ValueError:
            # Word not found exactly, return all words
            return [w for w in words if w != word_lower]

        # Get context window
        window = self.config.lesk_context_window
        start = max(0, pos - window)
        end = min(len(words), pos + window + 1)

        # Exclude target word
        context = words[start:pos] + words[pos + 1:end]
        return context

    def _extract_content_words(self, sentence: str) -> list[str]:
        """Extract content words (nouns, verbs, adjectives) from sentence."""
        words = re.findall(r"\b\w+\b", sentence.lower())

        # Filter out common function words
        function_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "or", "and", "that", "this", "it", "as", "not", "but", "so",
            "if", "when", "then", "than", "too", "very", "just", "only",
            "i", "you", "he", "she", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
        }

        return [w for w in words if w not in function_words and len(w) > 2]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_synset_id(word: str, pos: str = "n", sense_num: int = 1) -> str:
    """Create a WordNet synset ID string.

    Args:
        word: The word
        pos: Part of speech (n, v, a, r)
        sense_num: Sense number (1-based)

    Returns:
        Synset ID string (e.g., "bank.n.01")
    """
    return f"{word}.{pos}.{sense_num:02d}"


def synset_ids_match(id_a: str, id_b: str) -> bool:
    """Check if two synset IDs are identical."""
    return id_a == id_b


def are_same_word_different_sense(id_a: str, id_b: str) -> bool:
    """Check if two synset IDs are different senses of the same word."""
    if not id_a or not id_b:
        return False

    parts_a = id_a.rsplit(".", 2)
    parts_b = id_b.rsplit(".", 2)

    if len(parts_a) < 3 or len(parts_b) < 3:
        return False

    return (
        parts_a[0] == parts_b[0]
        and parts_a[1] == parts_b[1]
        and parts_a[2] != parts_b[2]
    )
