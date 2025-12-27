"""Tests for Story Teller behavior templates.

These tests verify that the story teller and NPC character behaviors
are correctly defined and can be used to create interactive fiction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.behaviors import (
    # Templates
    STORY_TELLER_TEMPLATE,
    STORY_TELLER_CHARACTER_TEMPLATE,
    # Factory functions
    create_story_character,
    create_story_teller,
    # State types
    StoryState,
    CharacterProfile,
    # Registry
    BehaviorRegistry,
    BehaviorTier,
    BehaviorStatus,
)


# =============================================================================
# Story Teller Template Tests
# =============================================================================


class TestStoryTellerTemplate:
    """Tests for the STORY_TELLER_TEMPLATE."""

    def test_template_exists(self):
        """Verify template is defined."""
        assert STORY_TELLER_TEMPLATE is not None
        assert STORY_TELLER_TEMPLATE.behavior_id == "story_teller"
        assert STORY_TELLER_TEMPLATE.name == "Story Teller"

    def test_template_has_narrative_actions(self):
        """Verify core narrative actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        # Core narrative actions
        assert "describe_scene" in action_names
        assert "present_choice" in action_names
        assert "advance_story" in action_names
        assert "reveal_information" in action_names

    def test_template_has_theme_discovery_actions(self):
        """Verify theme discovery phase actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        # Theme discovery actions (used at story start)
        assert "ask_theme_preference" in action_names
        assert "list_theme_options" in action_names
        assert "set_story_theme" in action_names
        assert "generate_personalized_theme" in action_names

    def test_template_has_character_actions(self):
        """Verify character-related actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        assert "introduce_character" in action_names
        assert "npc_action" in action_names

    def test_template_has_state_actions(self):
        """Verify state management actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        assert "update_inventory" in action_names
        assert "check_inventory" in action_names
        assert "update_relationship" in action_names

    def test_template_has_pacing_actions(self):
        """Verify pacing control actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        assert "create_tension" in action_names
        assert "provide_relief" in action_names
        assert "end_chapter" in action_names

    def test_template_has_meta_actions(self):
        """Verify meta/game actions are present."""
        action_names = [a.name for a in STORY_TELLER_TEMPLATE.actions]

        assert "start_story" in action_names
        assert "player_speaks" in action_names

    def test_template_has_triggers(self):
        """Verify template has activation triggers."""
        assert len(STORY_TELLER_TEMPLATE.triggers) > 0
        # Check keyword patterns contain expected terms
        trigger = STORY_TELLER_TEMPLATE.triggers[0]
        assert any("story" in p or "adventure" in p for p in trigger.keyword_patterns)

    def test_template_has_test_cases(self):
        """Verify template includes test cases for both phases."""
        assert len(STORY_TELLER_TEMPLATE.test_cases) >= 8  # Theme discovery + active story tests

        test_ids = [t.test_id for t in STORY_TELLER_TEMPLATE.test_cases]

        # Theme discovery phase tests
        assert "ask_for_theme" in test_ids
        assert "user_chooses_theme" in test_ids
        assert "surprise_me_theme" in test_ids

        # Active story phase tests
        assert "describe_location" in test_ids
        assert "present_choices" in test_ids

    def test_template_has_prompts(self):
        """Verify decision and synthesis prompts are defined."""
        assert STORY_TELLER_TEMPLATE.prompts is not None
        assert STORY_TELLER_TEMPLATE.prompts.decision_prompt is not None
        assert STORY_TELLER_TEMPLATE.prompts.synthesis_prompt is not None

        # Decision prompt should reference story state
        assert "story_state" in STORY_TELLER_TEMPLATE.prompts.decision_prompt.lower()

        # Decision prompt should be phase-aware
        assert "theme_discovery" in STORY_TELLER_TEMPLATE.prompts.decision_prompt.lower()
        assert "active_story" in STORY_TELLER_TEMPLATE.prompts.decision_prompt.lower()

        # Synthesis prompt should reference narrative
        assert "narrative" in STORY_TELLER_TEMPLATE.prompts.synthesis_prompt.lower()

    def test_template_is_addon_tier(self):
        """Verify template is marked as addon tier."""
        assert STORY_TELLER_TEMPLATE.tier == BehaviorTier.ADDON

    def test_template_is_active(self):
        """Verify template is active."""
        assert STORY_TELLER_TEMPLATE.status == BehaviorStatus.ACTIVE


# =============================================================================
# Story Teller Character Template Tests
# =============================================================================


class TestStoryCharacterTemplate:
    """Tests for the STORY_TELLER_CHARACTER_TEMPLATE."""

    def test_template_exists(self):
        """Verify NPC template is defined."""
        assert STORY_TELLER_CHARACTER_TEMPLATE is not None
        assert STORY_TELLER_CHARACTER_TEMPLATE.behavior_id == "story_character"

    def test_template_has_dialogue_actions(self):
        """Verify dialogue actions are present."""
        action_names = [a.name for a in STORY_TELLER_CHARACTER_TEMPLATE.actions]

        assert "speak" in action_names
        assert "react" in action_names

    def test_template_has_relationship_actions(self):
        """Verify relationship actions are present."""
        action_names = [a.name for a in STORY_TELLER_CHARACTER_TEMPLATE.actions]

        assert "offer_help" in action_names
        assert "withhold" in action_names
        assert "reveal_secret" in action_names

    def test_template_has_autonomy_actions(self):
        """Verify NPCs can act autonomously."""
        action_names = [a.name for a in STORY_TELLER_CHARACTER_TEMPLATE.actions]

        assert "pursue_goal" in action_names
        assert "leave_scene" in action_names

    def test_template_has_no_triggers(self):
        """NPCs should not have direct triggers - they're invoked by story teller."""
        assert len(STORY_TELLER_CHARACTER_TEMPLATE.triggers) == 0

    def test_template_prompts_reference_character(self):
        """Verify prompts reference character personality."""
        assert STORY_TELLER_CHARACTER_TEMPLATE.prompts is not None
        assert "character" in STORY_TELLER_CHARACTER_TEMPLATE.prompts.decision_prompt.lower()
        assert "personality" in STORY_TELLER_CHARACTER_TEMPLATE.prompts.decision_prompt.lower()


# =============================================================================
# Story State Tests
# =============================================================================


class TestStoryState:
    """Tests for StoryState dataclass."""

    def test_default_state(self):
        """Test default story state initialization."""
        state = StoryState()

        # Phase defaults to theme_discovery
        assert state.phase == "theme_discovery"
        assert state.theme == ""
        assert state.setting == ""
        assert state.tone == ""

        # World state defaults
        assert state.current_location == "unknown"
        assert state.chapter == 1
        assert state.player_name == "Adventurer"
        assert state.health == "healthy"
        assert len(state.inventory) == 0
        assert len(state.plot_flags) == 0

    def test_theme_discovery_state(self):
        """Test story state during theme discovery phase."""
        state = StoryState(
            phase="theme_discovery",
            user_preferences=["fantasy", "mystery"],
            user_interests=["dragons", "detective stories"],
        )

        assert state.phase == "theme_discovery"
        assert "fantasy" in state.user_preferences
        assert "dragons" in state.user_interests

    def test_active_story_state(self):
        """Test story state after theme is set."""
        state = StoryState(
            phase="active_story",
            theme="mystery",
            setting="Victorian London",
            tone="gritty",
            central_conflict="A series of impossible murders",
            theme_source="user_choice",
        )

        assert state.phase == "active_story"
        assert state.theme == "mystery"
        assert state.setting == "Victorian London"
        assert state.theme_source == "user_choice"

    def test_custom_state(self):
        """Test custom story state."""
        state = StoryState(
            current_location="Ancient Library",
            time_of_day="night",
            chapter=3,
            player_name="Kira",
            inventory=["torch", "ancient map", "dagger"],
            plot_flags={"met_the_wizard": True, "found_secret_door": False},
            character_relationships={"aldric": 45, "shadow_thief": -20},
        )

        assert state.current_location == "Ancient Library"
        assert state.time_of_day == "night"
        assert state.chapter == 3
        assert len(state.inventory) == 3
        assert "torch" in state.inventory
        assert state.plot_flags["met_the_wizard"] is True
        assert state.character_relationships["aldric"] == 45


# =============================================================================
# Character Profile Tests
# =============================================================================


class TestCharacterProfile:
    """Tests for CharacterProfile dataclass."""

    def test_minimal_profile(self):
        """Test minimal character profile."""
        profile = CharacterProfile(
            character_id="guard_01",
            name="Town Guard",
            personality="Suspicious, duty-bound",
            backstory="A local who joined the guard to support his family",
        )

        assert profile.character_id == "guard_01"
        assert profile.name == "Town Guard"
        assert profile.trust_level == 0
        assert profile.met_player is False

    def test_full_profile(self):
        """Test fully detailed character profile."""
        profile = CharacterProfile(
            character_id="aldric",
            name="Aldric the Wise",
            personality="Mysterious, speaks in riddles, secretly caring",
            backstory="Former court wizard, now in exile",
            appearance="Tall, silver beard, deep blue robes",
            speech_style="Formal, archaic, trails off mysteriously...",
            goals=["Find a worthy successor", "Protect the ancient secrets"],
            secrets=["Knows the true heir", "Cursed by the shadow lord"],
            current_mood="contemplative",
            trust_level=30,
            met_player=True,
            interactions=["Gave cryptic warning", "Offered a quest"],
        )

        assert profile.name == "Aldric the Wise"
        assert len(profile.goals) == 2
        assert len(profile.secrets) == 2
        assert profile.trust_level == 30
        assert len(profile.interactions) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateStoryCharacter:
    """Tests for create_story_character factory."""

    def test_create_basic_character(self):
        """Test creating a basic character."""
        behavior, profile = create_story_character(
            character_id="merchant",
            name="Gregor the Merchant",
            personality="Greedy, clever, pragmatic",
            backstory="A traveling trader with connections everywhere",
        )

        # Check behavior
        assert behavior.behavior_id == "character_merchant"
        assert "Gregor" in behavior.name
        assert behavior.tier == BehaviorTier.GENERATED

        # Check profile
        assert profile.character_id == "merchant"
        assert profile.name == "Gregor the Merchant"
        assert profile.trust_level == 0

    def test_create_detailed_character(self):
        """Test creating a character with all details."""
        behavior, profile = create_story_character(
            character_id="witch",
            name="Elara the Witch",
            personality="Mischievous, wise, eccentric",
            backstory="Lives in the swamp, knows ancient herbalism",
            appearance="Wild gray hair, piercing green eyes, patched robes",
            speech_style="Cackling, rhyming sometimes, never gives straight answers",
            goals=["Collect rare ingredients", "Test the worthy"],
            secrets=["Was once a princess", "Guards a portal"],
            initial_mood="amused",
            initial_trust=-10,
        )

        assert profile.current_mood == "amused"
        assert profile.trust_level == -10
        assert len(profile.goals) == 2
        assert len(profile.secrets) == 2
        # Character name is included in domain_context
        assert "Elara the Witch" in behavior.domain_context

    def test_character_behavior_has_npc_actions(self):
        """Verify created character has NPC actions."""
        behavior, _ = create_story_character(
            character_id="test",
            name="Test NPC",
            personality="Test personality",
            backstory="Test backstory",
        )

        action_names = [a.name for a in behavior.actions]
        assert "speak" in action_names
        assert "react" in action_names
        assert "offer_help" in action_names

    def test_character_speech_style_default(self):
        """Test default speech style generation."""
        _, profile = create_story_character(
            character_id="noble",
            name="Lord Ashworth",
            personality="Arrogant, powerful, cunning",
            backstory="Duke of the eastern lands",
        )

        # Should generate speech style from personality
        assert profile.speech_style is not None
        assert len(profile.speech_style) > 0
        assert "arrogant" in profile.speech_style.lower()


class TestCreateStoryTeller:
    """Tests for create_story_teller factory."""

    def test_create_default_story_teller(self):
        """Test creating default story teller."""
        narrator = create_story_teller()

        assert "story_teller" in narrator.behavior_id
        assert narrator.tier == BehaviorTier.GENERATED

    def test_create_genre_story_teller(self):
        """Test creating genre-specific story teller."""
        horror_narrator = create_story_teller(
            genre="horror",
            tone="dark",
        )

        assert "horror" in horror_narrator.behavior_id
        # Genre and tone are included in domain_context
        assert "horror" in horror_narrator.domain_context.lower()
        assert "dark" in horror_narrator.domain_context.lower()

    def test_story_teller_has_all_actions(self):
        """Verify created story teller has all standard actions."""
        narrator = create_story_teller(genre="mystery")

        action_names = [a.name for a in narrator.actions]

        # Should have all narrative actions
        assert "describe_scene" in action_names
        assert "present_choice" in action_names
        assert "advance_story" in action_names


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for registering story behaviors."""

    def test_register_story_teller(self):
        """Test registering story teller in registry."""
        registry = BehaviorRegistry()
        registry.register(STORY_TELLER_TEMPLATE)

        behavior = registry.get("story_teller")
        assert behavior is not None
        assert behavior.name == "Story Teller"

    def test_register_multiple_characters(self):
        """Test registering multiple NPCs."""
        registry = BehaviorRegistry()

        # Create and register several characters
        wizard_behavior, _ = create_story_character(
            character_id="wizard",
            name="Aldric",
            personality="Mysterious",
            backstory="Old wizard",
        )

        thief_behavior, _ = create_story_character(
            character_id="thief",
            name="Shadow",
            personality="Cunning",
            backstory="Skilled thief",
        )

        registry.register(wizard_behavior)
        registry.register(thief_behavior)

        assert registry.get("character_wizard") is not None
        assert registry.get("character_thief") is not None

    def test_list_by_tier(self):
        """Test listing behaviors by tier."""
        registry = BehaviorRegistry()
        registry.register(STORY_TELLER_TEMPLATE)

        # Create a generated character
        char_behavior, _ = create_story_character(
            character_id="test",
            name="Test",
            personality="Test",
            backstory="Test",
        )
        registry.register(char_behavior)

        addon_behaviors = registry.get_by_tier(BehaviorTier.ADDON)
        generated_behaviors = registry.get_by_tier(BehaviorTier.GENERATED)

        addon_ids = [b.behavior_id for b in addon_behaviors]
        generated_ids = [b.behavior_id for b in generated_behaviors]

        assert "story_teller" in addon_ids
        assert "character_test" in generated_ids


# =============================================================================
# Action Parameter Tests
# =============================================================================


class TestActionParameters:
    """Tests for action parameter definitions."""

    def test_describe_scene_parameters(self):
        """Test describe_scene action parameters."""
        action = next(
            a for a in STORY_TELLER_TEMPLATE.actions if a.name == "describe_scene"
        )

        param_names = list(action.parameters.keys())
        assert "focus" in param_names
        assert "mood" in param_names

        # Check mood parameter
        mood_param = action.parameters["mood"]
        assert mood_param.required is False

    def test_present_choice_parameters(self):
        """Test present_choice action parameters."""
        action = next(
            a for a in STORY_TELLER_TEMPLATE.actions if a.name == "present_choice"
        )

        param_names = list(action.parameters.keys())
        assert "situation" in param_names
        assert "choices" in param_names
        assert "stakes" in param_names

        # Situation and choices are required
        assert action.parameters["situation"].required is True
        assert action.parameters["choices"].required is True

    def test_speak_parameters(self):
        """Test speak action (NPC) parameters."""
        action = next(
            a for a in STORY_TELLER_CHARACTER_TEMPLATE.actions if a.name == "speak"
        )

        param_names = list(action.parameters.keys())
        assert "message" in param_names
        assert "emotion" in param_names
        assert "subtext" in param_names


# =============================================================================
# Example and Trigger Tests
# =============================================================================


class TestExamplesAndTriggers:
    """Tests for action examples and triggers."""

    def test_describe_scene_has_example(self):
        """Test describe_scene has usage example."""
        action = next(
            a for a in STORY_TELLER_TEMPLATE.actions if a.name == "describe_scene"
        )

        assert len(action.examples) > 0
        example = action.examples[0]
        assert "tavern" in example.user_query.lower()

    def test_describe_scene_triggers(self):
        """Test describe_scene trigger patterns."""
        action = next(
            a for a in STORY_TELLER_TEMPLATE.actions if a.name == "describe_scene"
        )

        assert len(action.triggers) > 0
        triggers = action.triggers
        assert any("look" in t for t in triggers)

    def test_check_inventory_triggers(self):
        """Test check_inventory trigger patterns."""
        action = next(
            a for a in STORY_TELLER_TEMPLATE.actions if a.name == "check_inventory"
        )

        assert len(action.triggers) > 0
        triggers = action.triggers
        assert any("inventory" in t for t in triggers)
