"""Tests for presupposition extraction."""

import pytest

from decomposition.presuppositions import (
    PresuppositionExtractor,
    TriggerDetector,
    ContentGenerator,
    detect_triggers,
)
from decomposition.models import PresuppositionTrigger, Presupposition
from decomposition.config import PresuppositionConfig


# =============================================================================
# TriggerDetector Tests
# =============================================================================


class TestTriggerDetector:
    """Test trigger detection patterns."""

    @pytest.fixture
    def detector(self):
        """Create a detector with default config."""
        config = PresuppositionConfig()
        return TriggerDetector(config)

    # Definite Descriptions
    def test_detect_definite_description(self, detector):
        """Test detecting 'the X' patterns."""
        triggers = detector._detect_definite_descriptions("Doug forgot the meeting")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.DEFINITE_DESC
        assert triggers[0].trigger_text == "the meeting"

    def test_skip_common_definite_articles(self, detector):
        """Test that common phrases are skipped."""
        triggers = detector._detect_definite_descriptions("by the way")
        assert len(triggers) == 0

    def test_multiple_definites(self, detector):
        """Test detecting multiple definite descriptions."""
        triggers = detector._detect_definite_descriptions(
            "The president of the company met the board"
        )
        assert len(triggers) >= 2

    # Factive Verbs
    def test_detect_factive_realize(self, detector):
        """Test detecting 'realize' factive verb."""
        triggers = detector._detect_factive_verbs("Doug realized he was late")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.FACTIVE_VERB
        assert "realize" in triggers[0].trigger_text.lower()

    def test_detect_factive_know(self, detector):
        """Test detecting 'know' factive verb."""
        triggers = detector._detect_factive_verbs("She knew the answer was wrong")
        assert len(triggers) >= 1
        factive_triggers = [t for t in triggers if t.trigger_type == PresuppositionTrigger.FACTIVE_VERB]
        assert len(factive_triggers) >= 1

    def test_detect_factive_regret(self, detector):
        """Test detecting 'regret' factive verb."""
        triggers = detector._detect_factive_verbs("Sarah regrets selling her car")
        assert len(triggers) >= 1

    # Change of State
    def test_detect_stop(self, detector):
        """Test detecting 'stop' change-of-state."""
        triggers = detector._detect_change_of_state("Doug stopped smoking")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.CHANGE_OF_STATE
        assert "stop" in triggers[0].trigger_text.lower()

    def test_detect_start(self, detector):
        """Test detecting 'start' change-of-state."""
        triggers = detector._detect_change_of_state("She started working from home")
        assert len(triggers) == 1
        assert "start" in triggers[0].trigger_text.lower()

    def test_detect_continue(self, detector):
        """Test detecting 'continue' change-of-state."""
        triggers = detector._detect_change_of_state("The company continues to grow")
        assert len(triggers) == 1
        assert "continue" in triggers[0].trigger_text.lower()

    # Iteratives
    def test_detect_again(self, detector):
        """Test detecting 'again' iterative."""
        triggers = detector._detect_iteratives("Doug forgot again")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.ITERATIVE
        assert triggers[0].trigger_text.lower() == "again"
        assert triggers[0].confidence >= 0.9

    def test_detect_another(self, detector):
        """Test detecting 'another' iterative."""
        triggers = detector._detect_iteratives("She won another award")
        assert len(triggers) == 1
        assert triggers[0].trigger_text.lower() == "another"

    def test_detect_still(self, detector):
        """Test detecting 'still' iterative."""
        triggers = detector._detect_iteratives("He is still waiting")
        assert len(triggers) == 1
        assert triggers[0].trigger_text.lower() == "still"

    # Temporal Clauses
    def test_detect_before(self, detector):
        """Test detecting 'before' temporal clause."""
        triggers = detector._detect_temporal_clauses("Before Doug left, he locked the door")
        assert len(triggers) >= 1
        assert any(t.trigger_type == PresuppositionTrigger.TEMPORAL_CLAUSE for t in triggers)

    def test_detect_after(self, detector):
        """Test detecting 'after' temporal clause."""
        triggers = detector._detect_temporal_clauses("After the meeting ended, we went to lunch")
        assert len(triggers) >= 1

    def test_detect_when(self, detector):
        """Test detecting 'when' temporal clause."""
        triggers = detector._detect_temporal_clauses("When she arrived, everyone applauded")
        assert len(triggers) >= 1

    # Possessives
    def test_detect_possessive(self, detector):
        """Test detecting possessive 'X's Y'."""
        triggers = detector._detect_possessives("Doug's car is red")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.POSSESSIVE
        assert "Doug's car" in triggers[0].trigger_text

    def test_detect_possessive_multiple(self, detector):
        """Test detecting multiple possessives."""
        triggers = detector._detect_possessives("John's wife is Mary's sister")
        assert len(triggers) >= 2

    def test_skip_it_possessive(self, detector):
        """Test that 'it's' is skipped (contraction, not possessive)."""
        triggers = detector._detect_possessives("it's raining")
        assert len(triggers) == 0

    # Implicatives
    def test_detect_manage(self, detector):
        """Test detecting 'manage to' implicative."""
        triggers = detector._detect_implicatives("Doug managed to finish the project")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.IMPLICATIVE
        assert "manage" in triggers[0].trigger_text.lower()

    def test_detect_forget_to(self, detector):
        """Test detecting 'forget to' implicative."""
        triggers = detector._detect_implicatives("He forgot to lock the door")
        assert len(triggers) >= 1

    def test_detect_remember_to(self, detector):
        """Test detecting 'remember to' implicative."""
        triggers = detector._detect_implicatives("She remembered to call her mom")
        assert len(triggers) >= 1

    # Comparatives
    def test_detect_comparative_er(self, detector):
        """Test detecting '-er than' comparative."""
        triggers = detector._detect_comparatives("Doug is taller than Sarah")
        assert len(triggers) >= 1
        assert any(t.trigger_type == PresuppositionTrigger.COMPARATIVE for t in triggers)

    def test_detect_more_than(self, detector):
        """Test detecting 'more X than Y' comparative."""
        triggers = detector._detect_comparatives("This year's sales are more impressive than last year's")
        assert len(triggers) >= 1

    def test_detect_as_as(self, detector):
        """Test detecting 'as X as Y' comparative."""
        triggers = detector._detect_comparatives("She speaks French as well as her mother")
        assert len(triggers) >= 1

    # Clefts
    def test_detect_it_was_who(self, detector):
        """Test detecting 'It was X who...' cleft."""
        triggers = detector._detect_clefts("It was Doug who called")
        assert len(triggers) == 1
        assert triggers[0].trigger_type == PresuppositionTrigger.CLEFT

    def test_detect_it_is_who(self, detector):
        """Test detecting 'It is X who...' cleft."""
        triggers = detector._detect_clefts("It is Mary who manages the project")
        assert len(triggers) >= 1

    # Counterfactuals
    def test_detect_if_had(self, detector):
        """Test detecting 'if X had...' counterfactual."""
        triggers = detector._detect_counterfactuals("If Doug had known, he would have helped")
        assert len(triggers) >= 1
        assert any(t.trigger_type == PresuppositionTrigger.COUNTERFACTUAL for t in triggers)

    def test_detect_wish_had(self, detector):
        """Test detecting 'wish I had' counterfactual."""
        triggers = detector._detect_counterfactuals("I wish I had studied harder")
        assert len(triggers) >= 1

    # Full detection
    def test_detect_all_triggers(self, detector):
        """Test detecting multiple trigger types in one sentence."""
        triggers = detector.detect_triggers("Doug forgot the meeting again")
        trigger_types = {t.trigger_type for t in triggers}

        assert PresuppositionTrigger.DEFINITE_DESC in trigger_types
        assert PresuppositionTrigger.ITERATIVE in trigger_types

    def test_confidence_filtering(self):
        """Test that weak triggers are filtered when configured."""
        config = PresuppositionConfig(include_weak_triggers=False)
        detector = TriggerDetector(config)

        # All triggers should have confidence >= 0.7
        triggers = detector.detect_triggers("Doug forgot the meeting again")
        for t in triggers:
            assert t.confidence >= 0.7


# =============================================================================
# ContentGenerator Tests
# =============================================================================


class TestContentGenerator:
    """Test presupposition content generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator without LLM (template-only)."""
        config = PresuppositionConfig(use_llm_for_content=False)
        return ContentGenerator(config, llm=None)

    def test_template_definite_desc(self, generator):
        """Test template generation for definite descriptions."""
        from decomposition.presuppositions import DetectedTrigger

        trigger = DetectedTrigger(
            trigger_type=PresuppositionTrigger.DEFINITE_DESC,
            trigger_text="the meeting",
            span=(15, 26),
            context="Doug forgot the meeting again",
        )

        content = generator._generate_with_template(trigger, "Doug forgot the meeting again")
        assert "meeting" in content.lower()
        assert "exists" in content.lower()

    def test_template_iterative(self, generator):
        """Test template generation for iteratives."""
        from decomposition.presuppositions import DetectedTrigger

        trigger = DetectedTrigger(
            trigger_type=PresuppositionTrigger.ITERATIVE,
            trigger_text="again",
            span=(25, 30),
            context="Doug forgot the meeting again",
        )

        content = generator._generate_with_template(trigger, "Doug forgot the meeting again")
        assert "before" in content.lower() or "happened" in content.lower()

    def test_template_change_of_state_stop(self, generator):
        """Test template for 'stop' change-of-state."""
        from decomposition.presuppositions import DetectedTrigger

        trigger = DetectedTrigger(
            trigger_type=PresuppositionTrigger.CHANGE_OF_STATE,
            trigger_text="stopped",
            span=(5, 12),
            context="smoking",
        )

        content = generator._generate_with_template(trigger, "Doug stopped smoking")
        assert "doing" in content.lower() or "smoking" in content.lower()

    def test_template_possessive(self, generator):
        """Test template for possessives."""
        from decomposition.presuppositions import DetectedTrigger

        trigger = DetectedTrigger(
            trigger_type=PresuppositionTrigger.POSSESSIVE,
            trigger_text="Doug's car",
            span=(0, 10),
            context="Doug has car",
        )

        content = generator._generate_with_template(trigger, "Doug's car is red")
        assert "doug" in content.lower()
        assert "car" in content.lower()


# =============================================================================
# PresuppositionExtractor Tests
# =============================================================================


class TestPresuppositionExtractor:
    """Test the main extractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor without LLM."""
        config = PresuppositionConfig(use_llm_for_content=False)
        return PresuppositionExtractor(config, llm=None)

    def test_extract_sync_basic(self, extractor):
        """Test synchronous extraction."""
        presups = extractor.extract_sync("Doug forgot the meeting again")

        assert len(presups) >= 2
        trigger_types = {p.trigger_type for p in presups}
        assert PresuppositionTrigger.DEFINITE_DESC in trigger_types
        assert PresuppositionTrigger.ITERATIVE in trigger_types

    def test_extract_sync_no_triggers(self, extractor):
        """Test extraction with no triggers."""
        presups = extractor.extract_sync("Hello world")
        # May have some false positives, but should be minimal
        assert len(presups) <= 1

    def test_extract_sync_change_of_state(self, extractor):
        """Test extracting change-of-state presuppositions."""
        presups = extractor.extract_sync("Doug stopped smoking")

        change_presups = [
            p for p in presups
            if p.trigger_type == PresuppositionTrigger.CHANGE_OF_STATE
        ]
        assert len(change_presups) >= 1

    def test_extract_sync_multiple_triggers(self, extractor):
        """Test extracting multiple presuppositions."""
        presups = extractor.extract_sync(
            "Doug's wife realized that he had stopped drinking again"
        )

        assert len(presups) >= 2  # Should have multiple triggers

    def test_confidence_threshold(self):
        """Test that confidence threshold filters results."""
        config = PresuppositionConfig(
            use_llm_for_content=False,
            confidence_threshold=0.95,  # Very high threshold
        )
        extractor = PresuppositionExtractor(config)

        presups = extractor.extract_sync("Doug forgot the meeting again")
        # Only high-confidence triggers should pass
        for p in presups:
            assert p.confidence >= 0.95

    def test_max_presuppositions(self):
        """Test that max_presuppositions limits output."""
        config = PresuppositionConfig(
            use_llm_for_content=False,
            max_presuppositions=2,
        )
        extractor = PresuppositionExtractor(config)

        presups = extractor.extract_sync(
            "Doug's wife realized that he had stopped drinking again and the doctor was pleased"
        )
        assert len(presups) <= 2

    def test_entity_ids_passed_through(self, extractor):
        """Test that entity IDs are included in presuppositions."""
        entity_ids = ["doug-123", "meeting-456"]
        presups = extractor.extract_sync(
            "Doug forgot the meeting again",
            entity_ids=entity_ids
        )

        for p in presups:
            assert p.entity_ids == entity_ids

    def test_cancellable_attribute(self, extractor):
        """Test that cancellable attribute is set correctly."""
        presups = extractor.extract_sync("Doug stopped smoking")

        # Most presuppositions should be cancellable
        for p in presups:
            assert isinstance(p.cancellable, bool)

    def test_metrics_tracking(self, extractor):
        """Test that metrics are tracked."""
        extractor.reset_metrics()

        extractor.extract_sync("Doug forgot the meeting again")
        extractor.extract_sync("Sarah stopped working")

        metrics = extractor.get_metrics()
        assert metrics["total_extractions"] == 2
        assert metrics["triggers_detected"] > 0

    @pytest.mark.asyncio
    async def test_extract_async(self, extractor):
        """Test async extraction (without LLM falls back to sync)."""
        presups = await extractor.extract("Doug forgot the meeting again")

        assert len(presups) >= 2
        trigger_types = {p.trigger_type for p in presups}
        assert PresuppositionTrigger.ITERATIVE in trigger_types


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_triggers_function(self):
        """Test the detect_triggers convenience function."""
        triggers = detect_triggers("Doug forgot the meeting again")

        assert len(triggers) >= 2
        # Returns (text, type, span) tuples
        for text, trigger_type, span in triggers:
            assert isinstance(text, str)
            assert isinstance(trigger_type, PresuppositionTrigger)
            assert isinstance(span, tuple)
            assert len(span) == 2


# =============================================================================
# Presupposition Content Quality Tests
# =============================================================================


class TestPresuppositionQuality:
    """Test the quality of extracted presuppositions."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor without LLM."""
        config = PresuppositionConfig(use_llm_for_content=False)
        return PresuppositionExtractor(config, llm=None)

    def test_iterative_content_quality(self, extractor):
        """Test that iterative presupposition content is meaningful."""
        presups = extractor.extract_sync("She won another award")

        iterative_presups = [
            p for p in presups
            if p.trigger_type == PresuppositionTrigger.ITERATIVE
        ]
        assert len(iterative_presups) >= 1

        # Content should mention prior occurrence
        content = iterative_presups[0].content.lower()
        assert "before" in content or "prior" in content or "happened" in content

    def test_possessive_content_quality(self, extractor):
        """Test that possessive presupposition content is meaningful."""
        presups = extractor.extract_sync("John's car is fast")

        poss_presups = [
            p for p in presups
            if p.trigger_type == PresuppositionTrigger.POSSESSIVE
        ]
        assert len(poss_presups) >= 1

        # Content should mention possession relationship
        content = poss_presups[0].content.lower()
        assert "john" in content and "car" in content

    def test_presupposition_span_accuracy(self, extractor):
        """Test that trigger spans are accurate."""
        text = "Doug forgot the meeting again"
        presups = extractor.extract_sync(text)

        for p in presups:
            if p.trigger_span:
                start, end = p.trigger_span
                extracted = text[start:end]
                # The extracted text should match or contain the trigger text
                assert p.trigger_text in extracted or extracted in p.trigger_text
