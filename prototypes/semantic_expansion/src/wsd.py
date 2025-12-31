"""Word Sense Disambiguation (WSD) System.

Provides disambiguation of word senses using:
1. WordNet synsets for canonical identifiers
2. Lesk algorithm for gloss-based disambiguation
3. LLM-based disambiguation for complex cases
4. spaCy for lemmatization and POS tagging

Based on research from:
- WordNet (https://wordnet.princeton.edu/)
- Lesk algorithm (dictionary-based WSD)
- Modern LLM-based WSD approaches

Usage:
    >>> wsd = WordSenseDisambiguator()
    >>> senses = await wsd.disambiguate_sentence("I went to the bank")
    >>> print(senses["4:bank"].synset_id)  # "bank.n.01" (financial institution)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from semantic_types import WordSense

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


class NLPPipeline(Protocol):
    """Protocol for NLP pipeline (spaCy-like)."""

    def __call__(self, text: str) -> Any:
        """Process text and return doc object."""
        ...


# =============================================================================
# Synset Database (In-Memory Mock for when NLTK not available)
# =============================================================================


@dataclass
class MockSynset:
    """Mock synset for when NLTK is not available."""

    name: str
    pos: str
    lemmas: list[str]
    _definition: str = ""
    _examples: list[str] = field(default_factory=list)

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
            lemmas=["bank", "banking_company"],
            _definition="a financial institution that accepts deposits and channels the money into lending activities",
            _examples=["he went to the bank to deposit his check"],
        ),
        MockSynset(
            name="bank.n.02",
            pos="n",
            lemmas=["bank"],
            _definition="sloping land (especially the slope beside a body of water)",
            _examples=["they pulled the canoe up on the bank"],
        ),
        MockSynset(
            name="bank.n.03",
            pos="n",
            lemmas=["bank", "reserve"],
            _definition="a supply or stock held in reserve for future use",
            _examples=["a blood bank"],
        ),
    ],
    "tea": [
        MockSynset(
            name="tea.n.01",
            pos="n",
            lemmas=["tea"],
            _definition="a beverage made by steeping tea leaves in water",
            _examples=["iced tea is a cooling drink"],
        ),
        MockSynset(
            name="tea.n.02",
            pos="n",
            lemmas=["tea", "afternoon_tea"],
            _definition="a light midafternoon meal of tea and sandwiches or cakes",
            _examples=["an Englishman would interrupt a war to have his afternoon tea"],
        ),
    ],
    "coffee": [
        MockSynset(
            name="coffee.n.01",
            pos="n",
            lemmas=["coffee", "java"],
            _definition="a beverage consisting of an infusion of ground coffee beans",
            _examples=["he ordered a cup of coffee"],
        ),
    ],
    "morning": [
        MockSynset(
            name="morning.n.01",
            pos="n",
            lemmas=["morning", "morn", "forenoon"],
            _definition="the time period between dawn and noon",
            _examples=["I spent the morning jogging on the beach"],
        ),
    ],
    "prefer": [
        MockSynset(
            name="prefer.v.01",
            pos="v",
            lemmas=["prefer"],
            _definition="like better; value more highly",
            _examples=["Some people prefer camping to staying in hotels"],
        ),
        MockSynset(
            name="prefer.v.02",
            pos="v",
            lemmas=["prefer", "choose"],
            _definition="select as an alternative over another",
            _examples=["I always choose the cake over the pie"],
        ),
    ],
    "like": [
        MockSynset(
            name="like.v.01",
            pos="v",
            lemmas=["wish", "care", "like"],
            _definition="prefer or wish to do something",
            _examples=["Do you care to try this dish?"],
        ),
        MockSynset(
            name="like.v.02",
            pos="v",
            lemmas=["like"],
            _definition="find enjoyable or agreeable",
            _examples=["I like jogging"],
        ),
    ],
    "go": [
        MockSynset(
            name="go.v.01",
            pos="v",
            lemmas=["travel", "go", "move", "locomote"],
            _definition="change location; move, travel, or proceed",
            _examples=["How fast does your new car go?"],
        ),
    ],
    "person": [
        MockSynset(
            name="person.n.01",
            pos="n",
            lemmas=["person", "individual", "someone"],
            _definition="a human being",
            _examples=["there was too much for one person to do"],
        ),
    ],
}


# =============================================================================
# Lesk Disambiguator
# =============================================================================


class LeskDisambiguator:
    """Word sense disambiguation using the Lesk algorithm.

    The Lesk algorithm disambiguates by comparing:
    - The context words around the target word
    - The gloss (definition) of each candidate synset

    The synset with the most word overlap wins.
    """

    def __init__(self, use_nltk: bool = True):
        """Initialize the Lesk disambiguator.

        Args:
            use_nltk: Whether to use NLTK WordNet (requires nltk package)
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
                # NLTK not installed or wordnet corpus not downloaded
                self.use_nltk = False

    def get_synsets(self, lemma: str, pos: str | None = None) -> list[Any]:
        """Get candidate synsets for a lemma.

        Args:
            lemma: The lemmatized word
            pos: Part of speech (n, v, a, r)

        Returns:
            List of synset objects
        """
        if self.use_nltk and self._wordnet:
            return list(self._wordnet.synsets(lemma, pos=pos))
        else:
            # Use mock synsets
            return COMMON_SYNSETS.get(lemma.lower(), [])

    def disambiguate(
        self,
        word: str,
        context: list[str],
        pos: str | None = None,
    ) -> WordSense | None:
        """Disambiguate a word using the Lesk algorithm.

        Args:
            word: The word to disambiguate (lemmatized)
            context: Context words from the sentence
            pos: Part of speech

        Returns:
            The best WordSense, or None if no synsets found
        """
        synsets = self.get_synsets(word, pos)

        if not synsets:
            return None

        if len(synsets) == 1:
            # Unambiguous
            syn = synsets[0]
            synset_name = syn.name() if callable(getattr(syn, "name", None)) else syn.name
            synset_def = syn.definition() if callable(getattr(syn, "definition", None)) else getattr(syn, "_definition", "")
            return WordSense(
                surface_form=word,
                lemma=word,
                pos=pos or "UNKNOWN",
                synset_id=synset_name,
                definition=synset_def,
                confidence=1.0,
                disambiguation_method="unambiguous",
            )

        # Score each synset by context overlap
        context_set = set(w.lower() for w in context)
        best_synset = None
        best_score = -1

        for syn in synsets:
            # Get gloss words
            definition = syn.definition() if callable(syn.definition) else syn.definition
            gloss_words = set(self._tokenize(definition))

            # Add example words
            examples = syn.examples() if callable(syn.examples) else syn.examples
            for example in examples:
                gloss_words.update(self._tokenize(example))

            # Count overlap
            overlap = len(context_set & gloss_words)

            if overlap > best_score:
                best_score = overlap
                best_synset = syn

        if best_synset is None:
            best_synset = synsets[0]  # Fall back to first sense

        synset_name = best_synset.name() if callable(getattr(best_synset, "name", None)) else best_synset.name
        definition = best_synset.definition() if callable(getattr(best_synset, "definition", None)) else getattr(best_synset, "_definition", "")

        # Compute confidence based on score margin
        confidence = min(1.0, 0.5 + (best_score * 0.1))

        def get_synset_name(s: Any) -> str:
            return s.name() if callable(getattr(s, "name", None)) else s.name

        return WordSense(
            surface_form=word,
            lemma=word,
            pos=pos or "UNKNOWN",
            synset_id=synset_name,
            definition=definition,
            confidence=confidence,
            disambiguation_method="lesk",
            alternatives=[
                get_synset_name(s)
                for s in synsets
                if s != best_synset
            ],
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for gloss comparison."""
        # Remove punctuation and lowercase
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()
        # Remove stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "or", "and", "that", "this", "it", "as",
        }
        return [w for w in words if w not in stopwords and len(w) > 2]


# =============================================================================
# Main Word Sense Disambiguator
# =============================================================================


class WordSenseDisambiguator:
    """Word Sense Disambiguation using multiple methods.

    Methods (in order of preference):
    1. LLM-based WSD (most accurate, requires LLM provider)
    2. Lesk algorithm (dictionary-based, always available)

    Usage:
        >>> wsd = WordSenseDisambiguator()
        >>> senses = await wsd.disambiguate_sentence("I went to the bank")
        >>> print(senses["4:bank"].synset_id)
        "bank.n.01"
    """

    # Stopword POS tags (skip these for WSD)
    SKIP_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PUNCT", "SPACE", "SYM"}

    # Map spaCy POS to WordNet POS
    POS_MAP = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
        "PROPN": "n",  # Proper nouns treated as nouns
    }

    def __init__(
        self,
        nlp: NLPPipeline | None = None,
        llm: LLMProvider | None = None,
        use_nltk: bool = True,
    ):
        """Initialize the WSD system.

        Args:
            nlp: spaCy NLP pipeline (optional, uses simple tokenization if not provided)
            llm: LLM provider for advanced disambiguation (optional)
            use_nltk: Whether to use NLTK WordNet
        """
        self.nlp = nlp
        self.llm = llm
        self.lesk = LeskDisambiguator(use_nltk=use_nltk)

    async def disambiguate_sentence(
        self,
        sentence: str,
    ) -> dict[str, WordSense]:
        """Disambiguate all content words in a sentence.

        Args:
            sentence: The sentence to analyze

        Returns:
            Dict mapping "position:word" to WordSense
        """
        # Parse with spaCy if available
        if self.nlp:
            doc = self.nlp(sentence)
            tokens = [
                {
                    "i": token.i,
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                }
                for token in doc
            ]
        else:
            # Simple tokenization fallback
            tokens = self._simple_tokenize(sentence)

        # Get context words for Lesk
        context = [t["lemma"] for t in tokens if t["pos"] not in self.SKIP_POS]

        results: dict[str, WordSense] = {}

        for token in tokens:
            # Skip function words
            if token["pos"] in self.SKIP_POS:
                continue

            # Get WordNet POS
            wn_pos = self.POS_MAP.get(token["pos"])
            if not wn_pos:
                continue

            # Try LLM disambiguation for ambiguous words
            synsets = self.lesk.get_synsets(token["lemma"], wn_pos)

            if not synsets:
                continue

            if len(synsets) == 1:
                # Unambiguous
                sense = self.lesk.disambiguate(token["lemma"], context, wn_pos)
            elif self.llm and len(synsets) > 2:
                # Use LLM for highly ambiguous words
                sense = await self._llm_disambiguate(token, synsets, sentence)
                if not sense or sense.confidence < 0.7:
                    # Fall back to Lesk
                    sense = self.lesk.disambiguate(token["lemma"], context, wn_pos)
            else:
                # Use Lesk
                sense = self.lesk.disambiguate(token["lemma"], context, wn_pos)

            if sense:
                key = f"{token['i']}:{token['text']}"
                results[key] = sense

        return results

    async def disambiguate_word(
        self,
        word: str,
        context_sentence: str,
        pos: str | None = None,
    ) -> WordSense | None:
        """Disambiguate a single word in context.

        Args:
            word: The word to disambiguate
            context_sentence: The sentence containing the word
            pos: Part of speech (optional)

        Returns:
            WordSense or None
        """
        # Get lemma
        if self.nlp:
            doc = self.nlp(word)
            lemma = doc[0].lemma_ if doc else word.lower()
            pos = pos or doc[0].pos_ if doc else None
        else:
            lemma = word.lower()

        wn_pos = self.POS_MAP.get(pos) if pos else None

        # Get context
        context = self._simple_tokenize(context_sentence)
        context_words = [t["lemma"] for t in context]

        # Check synset count
        synsets = self.lesk.get_synsets(lemma, wn_pos)

        if not synsets:
            return None

        if len(synsets) > 2 and self.llm:
            # Use LLM for ambiguous words
            token = {"i": 0, "text": word, "lemma": lemma, "pos": pos or "NOUN"}
            sense = await self._llm_disambiguate(token, synsets, context_sentence)
            if sense and sense.confidence >= 0.7:
                return sense

        # Fall back to Lesk
        return self.lesk.disambiguate(lemma, context_words, wn_pos)

    async def _llm_disambiguate(
        self,
        token: dict[str, Any],
        synsets: list[Any],
        sentence: str,
    ) -> WordSense | None:
        """Use LLM for word sense disambiguation.

        Args:
            token: Token info dict
            synsets: Candidate synsets
            sentence: Full sentence for context

        Returns:
            WordSense or None
        """
        if not self.llm:
            return None

        # Build options list
        options = []
        for syn in synsets[:5]:  # Limit to 5 options
            name = syn.name() if hasattr(syn, "name") and callable(syn.name) else syn.name
            definition = syn.definition() if callable(syn.definition) else syn.definition
            options.append(f"- {name}: {definition}")

        options_text = "\n".join(options)

        prompt = f"""Determine which meaning of '{token['lemma']}' is used in this sentence:

Sentence: "{sentence}"

Possible meanings:
{options_text}

Respond in XML:
<disambiguation>
    <synset_id>The synset ID (e.g., bank.n.01)</synset_id>
    <confidence>0.0-1.0</confidence>
    <reasoning>Brief explanation</reasoning>
</disambiguation>
"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            # Parse response (handle both str and ChatResponse)
            return self._parse_llm_response(_extract_llm_content(response), token, synsets)

        except Exception:
            return None

    def _parse_llm_response(
        self,
        response: str,
        token: dict[str, Any],
        synsets: list[Any],
    ) -> WordSense | None:
        """Parse LLM disambiguation response."""
        # Extract synset_id
        synset_match = re.search(r"<synset_id>([^<]+)</synset_id>", response)
        if not synset_match:
            return None

        synset_id = synset_match.group(1).strip()

        # Extract confidence
        conf_match = re.search(r"<confidence>([0-9.]+)</confidence>", response)
        confidence = float(conf_match.group(1)) if conf_match else 0.8

        # Find matching synset for definition
        definition = ""
        for syn in synsets:
            name = syn.name() if hasattr(syn, "name") and callable(syn.name) else syn.name
            if name == synset_id:
                definition = syn.definition() if callable(syn.definition) else syn.definition
                break

        return WordSense(
            surface_form=token["text"],
            lemma=token["lemma"],
            pos=token["pos"],
            synset_id=synset_id,
            definition=definition,
            confidence=confidence,
            disambiguation_method="llm",
            alternatives=[
                (s.name() if hasattr(s, "name") and callable(s.name) else s.name)
                for s in synsets
                if (s.name() if hasattr(s, "name") and callable(s.name) else s.name) != synset_id
            ],
        )

    def _simple_tokenize(self, text: str) -> list[dict[str, Any]]:
        """Simple tokenization when spaCy is not available."""
        # Basic word tokenization
        words = re.findall(r"\b\w+\b", text.lower())

        tokens = []
        for i, word in enumerate(words):
            # Simple POS guessing based on common patterns
            if word in {"the", "a", "an"}:
                pos = "DET"
            elif word in {"in", "on", "at", "to", "from", "with", "by", "for", "of"}:
                pos = "ADP"
            elif word in {"and", "or", "but"}:
                pos = "CCONJ"
            elif word in {"is", "are", "was", "were", "be", "been", "being",
                          "have", "has", "had", "do", "does", "did",
                          "go", "went", "gone", "come", "came"}:
                pos = "VERB"
            elif word.endswith("ly"):
                pos = "ADV"
            elif word.endswith(("ing", "ed", "es", "s")) and len(word) > 3:
                # Could be verb or noun plural
                pos = "VERB" if word.endswith(("ing", "ed")) else "NOUN"
            else:
                pos = "NOUN"  # Default to noun

            tokens.append({
                "i": i,
                "text": word,
                "lemma": word,  # Without spaCy, just use the word
                "pos": pos,
            })

        return tokens


# =============================================================================
# Convenience Functions
# =============================================================================


def get_synset_id(word: str, pos: str = "n", sense_num: int = 1) -> str:
    """Get a WordNet synset ID string.

    Args:
        word: The word
        pos: Part of speech (n, v, a, r)
        sense_num: Sense number (1-based)

    Returns:
        Synset ID string (e.g., "bank.n.01")
    """
    return f"{word}.{pos}.{sense_num:02d}"


def synset_ids_match(id_a: str, id_b: str) -> bool:
    """Check if two synset IDs represent the same concept.

    Args:
        id_a: First synset ID
        id_b: Second synset ID

    Returns:
        True if they match
    """
    return id_a == id_b


def are_same_word_different_sense(id_a: str, id_b: str) -> bool:
    """Check if two synset IDs are different senses of the same word.

    Args:
        id_a: First synset ID (e.g., "bank.n.01")
        id_b: Second synset ID (e.g., "bank.n.02")

    Returns:
        True if same word but different sense
    """
    if not id_a or not id_b:
        return False

    # Extract word and POS
    parts_a = id_a.rsplit(".", 2)
    parts_b = id_b.rsplit(".", 2)

    if len(parts_a) < 3 or len(parts_b) < 3:
        return False

    # Same word and POS, different sense number
    return (
        parts_a[0] == parts_b[0] and
        parts_a[1] == parts_b[1] and
        parts_a[2] != parts_b[2]
    )
