"""Evolving Synset Database.

A self-learning database of synsets that extends WordNet with:
- Technology terms (kubernetes, docker, terraform, etc.)
- Domain-specific jargon not in WordNet
- User-defined terms and corrections
- LLM-generated definitions

The database loads from JSON files and can learn new terms over time.
It integrates with WordNetInterface to provide merged results.

Example:
    >>> from evolving_synsets import EvolvingSynsetDatabase
    >>> from identifiers import LearnedSynset, SynsetSource
    >>>
    >>> # Load existing synsets
    >>> db = EvolvingSynsetDatabase()
    >>> db.load_from_directory("data/synsets")
    >>>
    >>> # Get synsets (merges learned + WordNet)
    >>> synsets = db.get_synsets("kubernetes")
    >>> print(synsets[0].definition)  # "Container orchestration platform..."
    >>>
    >>> # Add a new synset
    >>> db.add_synset(LearnedSynset(
    ...     synset_id="newterm.tech.01",
    ...     word="newterm",
    ...     pos="n",
    ...     definition="A new technology term",
    ...     domain="TECHNOLOGY",
    ... ))
    >>>
    >>> # Record usage outcome
    >>> db.reinforce("kubernetes.tech.01", success=True)
    >>>
    >>> # Save changes
    >>> db.save()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from identifiers import LearnedSynset, SynsetInfo, SynsetSource

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvolvingDBConfig:
    """Configuration for the evolving synset database.

    Attributes:
        data_directory: Path to the synset JSON files
        auto_save: Whether to auto-save after modifications
        prefer_learned: Whether to prefer learned synsets over WordNet
        min_confidence_threshold: Minimum confidence to include in results
        success_rate_threshold: Minimum success rate to include in results
    """

    data_directory: str = ""
    auto_save: bool = False
    prefer_learned: bool = True
    min_confidence_threshold: float = 0.3
    success_rate_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "data_directory": self.data_directory,
            "auto_save": self.auto_save,
            "prefer_learned": self.prefer_learned,
            "min_confidence_threshold": self.min_confidence_threshold,
            "success_rate_threshold": self.success_rate_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolvingDBConfig":
        """Deserialize from dictionary."""
        return cls(
            data_directory=data.get("data_directory", ""),
            auto_save=data.get("auto_save", False),
            prefer_learned=data.get("prefer_learned", True),
            min_confidence_threshold=data.get("min_confidence_threshold", 0.3),
            success_rate_threshold=data.get("success_rate_threshold", 0.5),
        )


# =============================================================================
# Evolving Synset Database
# =============================================================================


class EvolvingSynsetDatabase:
    """Self-learning database of synsets that extends WordNet.

    Features:
    - Loads synsets from JSON files organized by domain
    - Supports alias resolution (k8s -> kubernetes)
    - Tracks usage statistics for reinforcement learning
    - Integrates with WordNet for merged results
    - Persists learned synsets back to JSON files

    Priority System:
    1. User-provided synsets (highest confidence)
    2. Bootstrap synsets (pre-loaded vocabulary)
    3. LLM-verified synsets
    4. WordNet synsets
    5. LLM-unverified synsets (lowest priority)
    """

    def __init__(
        self,
        config: EvolvingDBConfig | None = None,
        data_directory: str | Path | None = None,
    ):
        """Initialize the evolving synset database.

        Args:
            config: Configuration options
            data_directory: Path to synset JSON files (overrides config)
        """
        self.config = config or EvolvingDBConfig()
        if data_directory:
            self.config.data_directory = str(data_directory)

        # Synsets indexed by synset_id
        self._synsets: dict[str, LearnedSynset] = {}

        # Word to synset_id index (includes aliases)
        self._word_index: dict[str, list[str]] = {}

        # Domain to synset_id index
        self._domain_index: dict[str, list[str]] = {}

        # Track which files synsets came from
        self._file_index: dict[str, str] = {}

        # Dirty flag for auto-save
        self._dirty = False

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_from_directory(self, directory: str | Path | None = None) -> int:
        """Load all synset files from a directory.

        Args:
            directory: Path to directory containing JSON files.
                      Uses config.data_directory if not specified.

        Returns:
            Number of synsets loaded
        """
        dir_path = Path(directory or self.config.data_directory)
        if not dir_path.exists():
            logger.warning(f"Synset directory does not exist: {dir_path}")
            return 0

        count = 0
        for json_file in dir_path.glob("*.json"):
            loaded = self.load_from_file(json_file)
            count += loaded
            logger.info(f"Loaded {loaded} synsets from {json_file.name}")

        logger.info(f"Total synsets loaded: {count}")
        return count

    def load_from_file(self, filepath: str | Path) -> int:
        """Load synsets from a single JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Number of synsets loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Synset file does not exist: {filepath}")
            return 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return 0

        # Handle both list format and dict with "synsets" key
        synsets_data = data if isinstance(data, list) else data.get("synsets", [])

        count = 0
        for synset_data in synsets_data:
            try:
                synset = LearnedSynset.from_dict(synset_data)
                self._add_synset_internal(synset, str(filepath))
                count += 1
            except Exception as e:
                logger.error(f"Failed to load synset: {e}")

        return count

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def get_synsets(
        self,
        word: str,
        pos: str | None = None,
        domain: str | None = None,
    ) -> list[LearnedSynset]:
        """Get synsets matching a word.

        Args:
            word: The word to look up (case-insensitive)
            pos: Optional part of speech filter
            domain: Optional domain filter

        Returns:
            List of matching synsets, sorted by priority
        """
        word_lower = word.lower()

        # Get synset IDs from word index
        synset_ids = self._word_index.get(word_lower, [])

        # Filter and collect synsets
        results = []
        for sid in synset_ids:
            synset = self._synsets.get(sid)
            if not synset:
                continue

            # Apply filters
            if pos and synset.pos != pos:
                continue
            if domain and synset.domain != domain:
                continue

            # Apply quality thresholds
            if synset.confidence < self.config.min_confidence_threshold:
                continue
            if synset.success_rate < self.config.success_rate_threshold:
                continue

            results.append(synset)

        # Sort by priority (user > bootstrap > llm_verified > wordnet > llm_unverified)
        results.sort(key=self._synset_priority, reverse=True)

        return results

    def get_synset_by_id(self, synset_id: str) -> LearnedSynset | None:
        """Get a specific synset by ID.

        Args:
            synset_id: The synset ID

        Returns:
            The synset or None if not found
        """
        return self._synsets.get(synset_id)

    def get_synsets_by_domain(self, domain: str) -> list[LearnedSynset]:
        """Get all synsets in a domain.

        Args:
            domain: The domain name

        Returns:
            List of synsets in the domain
        """
        synset_ids = self._domain_index.get(domain, [])
        return [self._synsets[sid] for sid in synset_ids if sid in self._synsets]

    def has_word(self, word: str) -> bool:
        """Check if a word is in the database.

        Args:
            word: The word to check (case-insensitive)

        Returns:
            True if the word exists
        """
        return word.lower() in self._word_index

    def resolve_alias(self, alias: str) -> str | None:
        """Resolve an alias to its canonical word.

        Args:
            alias: The alias to resolve (e.g., "k8s")

        Returns:
            The canonical word (e.g., "kubernetes") or None
        """
        alias_lower = alias.lower()
        synset_ids = self._word_index.get(alias_lower, [])
        if synset_ids:
            synset = self._synsets.get(synset_ids[0])
            if synset:
                return synset.word
        return None

    def get_all_synsets(self) -> list[LearnedSynset]:
        """Get all synsets in the database.

        Returns:
            List of all synsets
        """
        return list(self._synsets.values())

    def count(self) -> int:
        """Get the total number of synsets.

        Returns:
            Number of synsets in the database
        """
        return len(self._synsets)

    # -------------------------------------------------------------------------
    # Modification
    # -------------------------------------------------------------------------

    def add_synset(
        self,
        synset: LearnedSynset,
        domain_file: str | None = None,
    ) -> None:
        """Add a new synset to the database.

        Args:
            synset: The synset to add
            domain_file: Optional filename to associate (for saving)
        """
        if not domain_file:
            # Determine file based on domain
            domain_file = self._domain_to_filename(synset.domain)

        self._add_synset_internal(synset, domain_file)
        self._dirty = True

        if self.config.auto_save:
            self.save()

    def remove_synset(self, synset_id: str) -> bool:
        """Remove a synset from the database.

        Args:
            synset_id: The synset ID to remove

        Returns:
            True if removed, False if not found
        """
        synset = self._synsets.get(synset_id)
        if not synset:
            return False

        # Remove from main index
        del self._synsets[synset_id]

        # Remove from word index
        for word in [synset.word.lower()] + [a.lower() for a in synset.aliases]:
            if word in self._word_index:
                self._word_index[word] = [
                    sid for sid in self._word_index[word] if sid != synset_id
                ]
                if not self._word_index[word]:
                    del self._word_index[word]

        # Remove from domain index
        if synset.domain in self._domain_index:
            self._domain_index[synset.domain] = [
                sid for sid in self._domain_index[synset.domain] if sid != synset_id
            ]

        # Remove from file index
        if synset_id in self._file_index:
            del self._file_index[synset_id]

        self._dirty = True

        if self.config.auto_save:
            self.save()

        return True

    def update_synset(
        self,
        synset_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update fields of an existing synset.

        Args:
            synset_id: The synset ID to update
            updates: Dictionary of field updates

        Returns:
            True if updated, False if not found
        """
        synset = self._synsets.get(synset_id)
        if not synset:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(synset, key):
                setattr(synset, key, value)

        # Re-index if word or aliases changed
        if "word" in updates or "aliases" in updates:
            self._reindex_synset(synset)

        # Re-index if domain changed
        if "domain" in updates:
            self._reindex_domain(synset)

        self._dirty = True

        if self.config.auto_save:
            self.save()

        return True

    # -------------------------------------------------------------------------
    # Reinforcement Learning
    # -------------------------------------------------------------------------

    def reinforce(self, synset_id: str, success: bool = True) -> bool:
        """Record a usage outcome for reinforcement learning.

        Args:
            synset_id: The synset that was used
            success: Whether the disambiguation was successful

        Returns:
            True if recorded, False if synset not found
        """
        synset = self._synsets.get(synset_id)
        if not synset:
            return False

        synset.record_usage(success)
        self._dirty = True

        if self.config.auto_save:
            self.save()

        return True

    def boost_synset(self, synset_id: str, amount: float = 0.1) -> bool:
        """Boost a synset's confidence.

        Args:
            synset_id: The synset to boost
            amount: Amount to add to confidence (capped at 1.0)

        Returns:
            True if boosted, False if synset not found
        """
        synset = self._synsets.get(synset_id)
        if not synset:
            return False

        synset.confidence = min(1.0, synset.confidence + amount)
        self._dirty = True

        if self.config.auto_save:
            self.save()

        return True

    def demote_synset(self, synset_id: str, amount: float = 0.1) -> bool:
        """Demote a synset's confidence.

        Args:
            synset_id: The synset to demote
            amount: Amount to subtract from confidence (floored at 0.0)

        Returns:
            True if demoted, False if synset not found
        """
        synset = self._synsets.get(synset_id)
        if not synset:
            return False

        synset.confidence = max(0.0, synset.confidence - amount)
        self._dirty = True

        if self.config.auto_save:
            self.save()

        return True

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> int:
        """Save all modified synsets to their respective files.

        Args:
            directory: Optional output directory (uses config if not specified)

        Returns:
            Number of files written
        """
        dir_path = Path(directory or self.config.data_directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Group synsets by file
        files: dict[str, list[LearnedSynset]] = {}
        for synset_id, synset in self._synsets.items():
            filepath = self._file_index.get(synset_id, "")
            if not filepath:
                filepath = str(dir_path / self._domain_to_filename(synset.domain))

            if filepath not in files:
                files[filepath] = []
            files[filepath].append(synset)

        # Write each file
        count = 0
        for filepath, synsets in files.items():
            filepath = Path(filepath)
            if not filepath.is_absolute():
                filepath = dir_path / filepath

            # Sort synsets by ID for stable output
            synsets.sort(key=lambda s: s.synset_id)

            data = {
                "domain": synsets[0].domain if synsets else "",
                "synsets": [s.to_dict() for s in synsets],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            count += 1
            logger.info(f"Saved {len(synsets)} synsets to {filepath.name}")

        self._dirty = False
        return count

    def export_to_file(self, filepath: str | Path) -> int:
        """Export all synsets to a single JSON file.

        Args:
            filepath: Output file path

        Returns:
            Number of synsets exported
        """
        synsets = list(self._synsets.values())
        synsets.sort(key=lambda s: s.synset_id)

        data = {
            "total": len(synsets),
            "synsets": [s.to_dict() for s in synsets],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return len(synsets)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics
        """
        sources: dict[str, int] = {}
        domains: dict[str, int] = {}

        for synset in self._synsets.values():
            source = synset.source.value if isinstance(synset.source, SynsetSource) else synset.source
            sources[source] = sources.get(source, 0) + 1
            domains[synset.domain] = domains.get(synset.domain, 0) + 1

        return {
            "total_synsets": len(self._synsets),
            "total_words": len(self._word_index),
            "by_source": sources,
            "by_domain": domains,
            "dirty": self._dirty,
        }

    @property
    def size(self) -> int:
        """Get the number of synsets in the database."""
        return len(self._synsets)

    @property
    def words(self) -> list[str]:
        """Get all indexed words."""
        return list(self._word_index.keys())

    @property
    def domains(self) -> list[str]:
        """Get all domains."""
        return list(self._domain_index.keys())

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _add_synset_internal(self, synset: LearnedSynset, filepath: str) -> None:
        """Internal method to add a synset with indexing.

        Args:
            synset: The synset to add
            filepath: The file this synset came from
        """
        # Add to main index
        self._synsets[synset.synset_id] = synset

        # Index by word and aliases
        for word in [synset.word.lower()] + [a.lower() for a in synset.aliases]:
            if word not in self._word_index:
                self._word_index[word] = []
            if synset.synset_id not in self._word_index[word]:
                self._word_index[word].append(synset.synset_id)

        # Index by domain
        if synset.domain:
            if synset.domain not in self._domain_index:
                self._domain_index[synset.domain] = []
            if synset.synset_id not in self._domain_index[synset.domain]:
                self._domain_index[synset.domain].append(synset.synset_id)

        # Track file
        self._file_index[synset.synset_id] = filepath

    def _reindex_synset(self, synset: LearnedSynset) -> None:
        """Re-index a synset after word/alias changes."""
        # Remove old entries
        for word, sids in list(self._word_index.items()):
            if synset.synset_id in sids:
                self._word_index[word] = [
                    sid for sid in sids if sid != synset.synset_id
                ]
                if not self._word_index[word]:
                    del self._word_index[word]

        # Add new entries
        for word in [synset.word.lower()] + [a.lower() for a in synset.aliases]:
            if word not in self._word_index:
                self._word_index[word] = []
            if synset.synset_id not in self._word_index[word]:
                self._word_index[word].append(synset.synset_id)

    def _reindex_domain(self, synset: LearnedSynset) -> None:
        """Re-index a synset after domain change."""
        # Remove from old domains
        for domain, sids in list(self._domain_index.items()):
            if synset.synset_id in sids:
                self._domain_index[domain] = [
                    sid for sid in sids if sid != synset.synset_id
                ]

        # Add to new domain
        if synset.domain:
            if synset.domain not in self._domain_index:
                self._domain_index[synset.domain] = []
            if synset.synset_id not in self._domain_index[synset.domain]:
                self._domain_index[synset.domain].append(synset.synset_id)

    def _synset_priority(self, synset: LearnedSynset) -> tuple[int, float, float]:
        """Calculate priority score for sorting.

        Returns:
            Tuple of (source_priority, confidence, success_rate)
        """
        source_priorities = {
            SynsetSource.USER: 5,
            SynsetSource.BOOTSTRAP: 4,
            SynsetSource.WORDNET: 3,
            SynsetSource.LLM: 2,
        }
        source = synset.source if isinstance(synset.source, SynsetSource) else SynsetSource(synset.source)
        source_priority = source_priorities.get(source, 1)

        return (source_priority, synset.confidence, synset.success_rate)

    def _domain_to_filename(self, domain: str) -> str:
        """Convert a domain name to a filename.

        Args:
            domain: Domain name (e.g., "CLOUD_INFRASTRUCTURE")

        Returns:
            Filename (e.g., "cloud_infrastructure.json")
        """
        if not domain:
            return "general.json"
        return domain.lower().replace(" ", "_") + ".json"


# =============================================================================
# Factory Functions
# =============================================================================


def create_evolving_database(
    data_directory: str | Path,
    auto_load: bool = True,
    **config_kwargs: Any,
) -> EvolvingSynsetDatabase:
    """Create an evolving synset database.

    Args:
        data_directory: Path to synset JSON files
        auto_load: Whether to load synsets on creation
        **config_kwargs: Additional configuration options

    Returns:
        Configured database instance
    """
    config = EvolvingDBConfig(
        data_directory=str(data_directory),
        **config_kwargs,
    )
    db = EvolvingSynsetDatabase(config)

    if auto_load:
        db.load_from_directory()

    return db
