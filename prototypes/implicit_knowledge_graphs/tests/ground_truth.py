"""Ground Truth Datasets for WSD and Entity Classification Testing.

This module provides curated test cases with verified ground truth labels.
These are NOT the same as the mock synset definitions - they test whether
the algorithm can generalize beyond its training data.

Categories:
- WSD: Word Sense Disambiguation test cases
- Entity: Entity Type Classification test cases

Each test case includes:
- Input text and context
- Expected output (synset ID or entity type)
- Difficulty rating
- Source/rationale for the ground truth
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Difficulty(str, Enum):
    """Test case difficulty levels."""
    TRIVIAL = "trivial"      # Unambiguous or obvious
    EASY = "easy"            # Clear context, common sense
    MEDIUM = "medium"        # Requires contextual reasoning
    HARD = "hard"            # Ambiguous or subtle
    ADVERSARIAL = "adversarial"  # Designed to fool the algorithm


@dataclass
class WSDTestCase:
    """A ground-truth test case for word sense disambiguation."""
    id: str
    word: str
    sentence: str
    expected_synset: str
    difficulty: Difficulty
    domain: str | None = None
    rationale: str = ""
    # For cases where multiple senses are acceptable
    acceptable_synsets: list[str] = field(default_factory=list)


@dataclass
class EntityTestCase:
    """A ground-truth test case for entity type classification."""
    id: str
    text: str
    context: str
    expected_type: str  # INSTANCE, CLASS, NAMED_CONCEPT, ROLE, ANAPHORA, GENERIC
    difficulty: Difficulty
    rationale: str = ""
    # Some cases could be multiple types
    acceptable_types: list[str] = field(default_factory=list)


# =============================================================================
# WSD Ground Truth Dataset
# =============================================================================

WSD_GROUND_TRUTH: list[WSDTestCase] = [
    # -------------------------------------------------------------------------
    # TRIVIAL: Unambiguous words (10 cases)
    # -------------------------------------------------------------------------
    WSDTestCase(
        id="wsd-trivial-001",
        word="morning",
        sentence="I wake up early in the morning.",
        expected_synset="morning.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="TIME",
        rationale="'morning' has only one common noun sense",
    ),
    WSDTestCase(
        id="wsd-trivial-002",
        word="river",
        sentence="The river flows through the valley.",
        expected_synset="river.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="GEOGRAPHY",
        rationale="'river' is unambiguous in this context",
    ),
    WSDTestCase(
        id="wsd-trivial-003",
        word="lake",
        sentence="We swam in the lake yesterday.",
        expected_synset="lake.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="GEOGRAPHY",
        rationale="'lake' as body of water is primary sense",
    ),
    WSDTestCase(
        id="wsd-trivial-004",
        word="money",
        sentence="I need to save more money.",
        expected_synset="money.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="FINANCE",
        rationale="'money' as currency is primary sense",
    ),
    WSDTestCase(
        id="wsd-trivial-005",
        word="song",
        sentence="That song has a beautiful melody.",
        expected_synset="song.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="MUSIC",
        rationale="'song' as musical composition is primary sense",
    ),
    WSDTestCase(
        id="wsd-trivial-006",
        word="tea",
        sentence="I prefer tea to coffee.",
        expected_synset="tea.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="FOOD",
        rationale="'tea' as beverage in comparison context",
    ),
    WSDTestCase(
        id="wsd-trivial-007",
        word="morning",
        sentence="Good morning, how are you?",
        expected_synset="morning.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="TIME",
        rationale="Greeting context, still time-of-day sense",
    ),
    WSDTestCase(
        id="wsd-trivial-008",
        word="money",
        sentence="The company lost a lot of money last quarter.",
        expected_synset="money.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="FINANCE",
        rationale="Business context, still currency sense",
    ),
    WSDTestCase(
        id="wsd-trivial-009",
        word="river",
        sentence="They built a bridge over the river.",
        expected_synset="river.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="GEOGRAPHY",
        rationale="Construction context, body of water",
    ),
    WSDTestCase(
        id="wsd-trivial-010",
        word="lake",
        sentence="The cabin sits beside the lake.",
        expected_synset="lake.n.01",
        difficulty=Difficulty.TRIVIAL,
        domain="GEOGRAPHY",
        rationale="Location context, body of water",
    ),

    # -------------------------------------------------------------------------
    # EASY: Clear context makes sense obvious (15 cases)
    # -------------------------------------------------------------------------
    WSDTestCase(
        id="wsd-easy-001",
        word="bank",
        sentence="I deposited my paycheck at the bank.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.EASY,
        domain="FINANCE",
        rationale="'deposited' and 'paycheck' strongly indicate financial institution",
    ),
    WSDTestCase(
        id="wsd-easy-002",
        word="bank",
        sentence="The children played on the bank of the river.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.EASY,
        domain="GEOGRAPHY",
        rationale="'of the river' explicitly indicates riverbank",
    ),
    WSDTestCase(
        id="wsd-easy-003",
        word="bass",
        sentence="He caught a 10-pound bass while fishing.",
        expected_synset="bass.n.01",
        difficulty=Difficulty.EASY,
        domain="FISH",
        rationale="'caught' and 'fishing' indicate the fish",
        acceptable_synsets=["bass.n.01", "bass.n.02"],  # freshwater bass also valid
    ),
    WSDTestCase(
        id="wsd-easy-004",
        word="bass",
        sentence="The bass guitar provides the rhythm section.",
        expected_synset="bass.n.07",
        difficulty=Difficulty.EASY,
        domain="MUSIC",
        rationale="'guitar' and 'rhythm section' indicate musical instrument",
        acceptable_synsets=["bass.n.07", "bass.n.08"],
    ),
    WSDTestCase(
        id="wsd-easy-005",
        word="bank",
        sentence="She works as a teller at the local bank.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.EASY,
        domain="FINANCE",
        rationale="'teller' is a bank-specific job",
    ),
    WSDTestCase(
        id="wsd-easy-006",
        word="bank",
        sentence="The flood eroded the river bank.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.EASY,
        domain="GEOGRAPHY",
        rationale="'flood' and 'eroded' with 'river' context",
    ),
    WSDTestCase(
        id="wsd-easy-007",
        word="bass",
        sentence="The bass notes were too loud in the mix.",
        expected_synset="bass.n.07",
        difficulty=Difficulty.EASY,
        domain="MUSIC",
        rationale="'notes' and 'mix' indicate music context",
        acceptable_synsets=["bass.n.07", "bass.n.08"],
    ),
    WSDTestCase(
        id="wsd-easy-008",
        word="bass",
        sentence="Bass are commonly found in freshwater lakes.",
        expected_synset="bass.n.01",
        difficulty=Difficulty.EASY,
        domain="FISH",
        rationale="'freshwater lakes' indicates the fish species",
        acceptable_synsets=["bass.n.01", "bass.n.02"],
    ),
    WSDTestCase(
        id="wsd-easy-009",
        word="bank",
        sentence="The ATM is inside the bank lobby.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.EASY,
        domain="FINANCE",
        rationale="'ATM' and 'lobby' indicate financial institution",
    ),
    WSDTestCase(
        id="wsd-easy-010",
        word="bank",
        sentence="Wild flowers grow along the bank.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.EASY,
        domain="GEOGRAPHY",
        rationale="'grow along' suggests natural landscape feature",
    ),
    WSDTestCase(
        id="wsd-easy-011",
        word="tea",
        sentence="Would you like tea with milk or lemon?",
        expected_synset="tea.n.01",
        difficulty=Difficulty.EASY,
        domain="FOOD",
        rationale="Preparation options indicate beverage",
    ),
    WSDTestCase(
        id="wsd-easy-012",
        word="tea",
        sentence="We had tea at four o'clock with scones.",
        expected_synset="tea.n.02",
        difficulty=Difficulty.EASY,
        domain="FOOD",
        rationale="'four o'clock' and 'scones' suggest afternoon tea meal",
    ),
    WSDTestCase(
        id="wsd-easy-013",
        word="apple",
        sentence="She bit into the crisp red apple.",
        expected_synset="apple.n.01",
        difficulty=Difficulty.EASY,
        domain="FOOD",
        rationale="'bit into' and 'crisp red' describe the fruit",
    ),
    WSDTestCase(
        id="wsd-easy-014",
        word="deposit",
        sentence="Please deposit the check by Friday.",
        expected_synset="deposit.v.01",
        difficulty=Difficulty.EASY,
        domain="FINANCE",
        rationale="'check' indicates banking context",
    ),
    WSDTestCase(
        id="wsd-easy-015",
        word="catch",
        sentence="Did you catch any fish today?",
        expected_synset="catch.v.10",
        difficulty=Difficulty.EASY,
        domain="FISHING",
        rationale="'fish' indicates capture/hunting sense",
    ),

    # -------------------------------------------------------------------------
    # MEDIUM: Requires more contextual reasoning (15 cases)
    # -------------------------------------------------------------------------
    WSDTestCase(
        id="wsd-medium-001",
        word="bank",
        sentence="The company needs to bank on its new product.",
        expected_synset="bank.v.01",  # Verb: to tip/turn
        difficulty=Difficulty.MEDIUM,
        domain="BUSINESS",
        rationale="Idiomatic usage - 'bank on' means rely on",
        acceptable_synsets=["bank.v.01"],  # May need different synset
    ),
    WSDTestCase(
        id="wsd-medium-002",
        word="bank",
        sentence="The pilot had to bank the aircraft sharply.",
        expected_synset="bank.v.01",
        difficulty=Difficulty.MEDIUM,
        domain="AVIATION",
        rationale="Aviation terminology - banking an aircraft",
    ),
    WSDTestCase(
        id="wsd-medium-003",
        word="bank",
        sentence="We maintain a blood bank for emergencies.",
        expected_synset="bank.n.04",
        difficulty=Difficulty.MEDIUM,
        domain="MEDICAL",
        rationale="'blood bank' is a reserve/supply sense",
    ),
    WSDTestCase(
        id="wsd-medium-004",
        word="bass",
        sentence="He has an incredibly deep bass voice.",
        expected_synset="bass.n.07",
        difficulty=Difficulty.MEDIUM,
        domain="MUSIC",
        rationale="'voice' indicates vocal range, not instrument",
    ),
    WSDTestCase(
        id="wsd-medium-005",
        word="bank",
        sentence="The data bank contains millions of records.",
        expected_synset="bank.n.04",
        difficulty=Difficulty.MEDIUM,
        domain="COMPUTING",
        rationale="'data bank' is metaphorical storage/reserve",
    ),
    WSDTestCase(
        id="wsd-medium-006",
        word="line",
        sentence="Please wait in line for your turn.",
        expected_synset="line.n.05",
        difficulty=Difficulty.MEDIUM,
        domain="GENERAL",
        rationale="Queue/formation sense, not mark sense",
    ),
    WSDTestCase(
        id="wsd-medium-007",
        word="line",
        sentence="Draw a straight line between the two points.",
        expected_synset="line.n.01",
        difficulty=Difficulty.MEDIUM,
        domain="MATH",
        rationale="Geometric mark sense",
    ),
    WSDTestCase(
        id="wsd-medium-008",
        word="bank",
        sentence="The casino has a house bank of one million.",
        expected_synset="bank.n.04",
        difficulty=Difficulty.MEDIUM,
        domain="GAMBLING",
        rationale="Reserve of money for gambling",
    ),
    WSDTestCase(
        id="wsd-medium-009",
        word="catch",
        sentence="I didn't catch what you said.",
        expected_synset="catch.v.02",
        difficulty=Difficulty.MEDIUM,
        domain="COMMUNICATION",
        rationale="Perceive/understand sense",
    ),
    WSDTestCase(
        id="wsd-medium-010",
        word="catch",
        sentence="The police will catch the thief eventually.",
        expected_synset="catch.v.10",
        difficulty=Difficulty.MEDIUM,
        domain="LAW",
        rationale="Capture/apprehend sense",
    ),
    WSDTestCase(
        id="wsd-medium-011",
        word="bank",
        sentence="They sat on the grassy bank overlooking the valley.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.MEDIUM,
        domain="GEOGRAPHY",
        rationale="Slope/embankment sense without water reference",
    ),
    WSDTestCase(
        id="wsd-medium-012",
        word="tea",
        sentence="The Duchess hosted a formal tea in the garden.",
        expected_synset="tea.n.02",
        difficulty=Difficulty.MEDIUM,
        domain="SOCIAL",
        rationale="'formal' and 'hosted' suggest the meal/event",
    ),
    WSDTestCase(
        id="wsd-medium-013",
        word="apple",
        sentence="The apple tree blooms beautifully in spring.",
        expected_synset="apple.n.02",
        difficulty=Difficulty.MEDIUM,
        domain="BOTANY",
        rationale="'tree' and 'blooms' indicate the plant, not fruit",
    ),
    WSDTestCase(
        id="wsd-medium-014",
        word="bass",
        sentence="The orchestra needs a new bass player.",
        expected_synset="bass.n.07",
        difficulty=Difficulty.MEDIUM,
        domain="MUSIC",
        rationale="Orchestra context, but 'player' is ambiguous (instrument vs voice)",
        acceptable_synsets=["bass.n.07", "bass.n.08"],
    ),
    WSDTestCase(
        id="wsd-medium-015",
        word="deposit",
        sentence="Glaciers deposit sediment as they melt.",
        expected_synset="deposit.v.02",
        difficulty=Difficulty.MEDIUM,
        domain="GEOLOGY",
        rationale="Geological sense - put/place, not bank",
    ),

    # -------------------------------------------------------------------------
    # HARD: Ambiguous or requires world knowledge (10 cases)
    # -------------------------------------------------------------------------
    WSDTestCase(
        id="wsd-hard-001",
        word="bank",
        sentence="Meet me at the bank at noon.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be financial institution or riverbank - but financial is more common meeting place",
    ),
    WSDTestCase(
        id="wsd-hard-002",
        word="bass",
        sentence="I love bass.",
        expected_synset="bass.n.01",  # Default to fish? Or music?
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Completely ambiguous without context",
        acceptable_synsets=["bass.n.01", "bass.n.07"],
    ),
    WSDTestCase(
        id="wsd-hard-003",
        word="bank",
        sentence="The bank was crowded today.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Likely financial (crowded = busy with customers) but uncertain",
    ),
    WSDTestCase(
        id="wsd-hard-004",
        word="bass",
        sentence="The bass was really impressive.",
        expected_synset="bass.n.01",  # Could be either
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be fish size or music quality",
        acceptable_synsets=["bass.n.01", "bass.n.07"],
    ),
    WSDTestCase(
        id="wsd-hard-005",
        word="bank",
        sentence="He spent the afternoon at the bank.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Likely financial (business) but could be relaxing by river",
    ),
    WSDTestCase(
        id="wsd-hard-006",
        word="tea",
        sentence="Let's have tea together.",
        expected_synset="tea.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be beverage or afternoon tea meal",
        acceptable_synsets=["tea.n.01", "tea.n.02"],
    ),
    WSDTestCase(
        id="wsd-hard-007",
        word="bank",
        sentence="The children were playing near the bank.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Children playing suggests outdoor/nature context",
    ),
    WSDTestCase(
        id="wsd-hard-008",
        word="line",
        sentence="There's a line here.",
        expected_synset="line.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be mark or queue",
        acceptable_synsets=["line.n.01", "line.n.05"],
    ),
    WSDTestCase(
        id="wsd-hard-009",
        word="catch",
        sentence="Nice catch!",
        expected_synset="catch.v.10",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be catching ball, fish, or understanding something",
        acceptable_synsets=["catch.v.01", "catch.v.02", "catch.v.10"],
    ),
    WSDTestCase(
        id="wsd-hard-010",
        word="apple",
        sentence="I prefer apple.",
        expected_synset="apple.n.01",
        difficulty=Difficulty.HARD,
        domain="AMBIGUOUS",
        rationale="Could be fruit flavor or Apple products",
    ),

    # -------------------------------------------------------------------------
    # ADVERSARIAL: Designed to fool algorithms (5 cases)
    # -------------------------------------------------------------------------
    WSDTestCase(
        id="wsd-adv-001",
        word="bank",
        sentence="The river bank offers low interest rates.",
        expected_synset="bank.n.01",
        difficulty=Difficulty.ADVERSARIAL,
        domain="FINANCE",
        rationale="'river' is misleading; 'interest rates' indicates financial",
    ),
    WSDTestCase(
        id="wsd-adv-002",
        word="bank",
        sentence="The financial institution sits on the bank.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.ADVERSARIAL,
        domain="GEOGRAPHY",
        rationale="'financial institution' is misleading; 'sits on' indicates location on embankment",
    ),
    WSDTestCase(
        id="wsd-adv-003",
        word="bass",
        sentence="The musician caught a large bass.",
        expected_synset="bass.n.01",
        difficulty=Difficulty.ADVERSARIAL,
        domain="FISH",
        rationale="'musician' is misleading; 'caught' indicates fish",
    ),
    WSDTestCase(
        id="wsd-adv-004",
        word="bass",
        sentence="The fisherman played bass in a band.",
        expected_synset="bass.n.07",
        difficulty=Difficulty.ADVERSARIAL,
        domain="MUSIC",
        rationale="'fisherman' is misleading; 'played in a band' indicates instrument",
        acceptable_synsets=["bass.n.07", "bass.n.08"],
    ),
    WSDTestCase(
        id="wsd-adv-005",
        word="bank",
        sentence="Deposit your money on the river bank.",
        expected_synset="bank.n.02",
        difficulty=Difficulty.ADVERSARIAL,
        domain="GEOGRAPHY",
        rationale="'deposit money' suggests financial, but 'on the river bank' is physical location",
    ),
]


# =============================================================================
# Entity Type Ground Truth Dataset
# =============================================================================

ENTITY_GROUND_TRUTH: list[EntityTestCase] = [
    # -------------------------------------------------------------------------
    # INSTANCE: Specific named entities (10 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-inst-001",
        text="Doug",
        context="Doug went to the store yesterday.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Proper noun, specific person",
    ),
    EntityTestCase(
        id="ent-inst-002",
        text="Apple",
        context="Apple announced new products at the conference.",
        expected_type="INSTANCE",
        difficulty=Difficulty.MEDIUM,
        rationale="Company name in business context",
    ),
    EntityTestCase(
        id="ent-inst-003",
        text="Apple Inc.",
        context="Apple Inc. reported record profits.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Full company name with Inc.",
    ),
    EntityTestCase(
        id="ent-inst-004",
        text="New York",
        context="She moved to New York last year.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Specific city name",
    ),
    EntityTestCase(
        id="ent-inst-005",
        text="Microsoft",
        context="Microsoft released a new version of Windows.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Company name",
    ),
    EntityTestCase(
        id="ent-inst-006",
        text="Sarah",
        context="I met Sarah at the coffee shop.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Person's name",
    ),
    EntityTestCase(
        id="ent-inst-007",
        text="The Eiffel Tower",
        context="We visited The Eiffel Tower in Paris.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Specific landmark",
    ),
    EntityTestCase(
        id="ent-inst-008",
        text="Amazon",
        context="I ordered it from Amazon.",
        expected_type="INSTANCE",
        difficulty=Difficulty.MEDIUM,
        rationale="Company name (not the river)",
    ),
    EntityTestCase(
        id="ent-inst-009",
        text="Dr. Smith",
        context="Dr. Smith will see you now.",
        expected_type="INSTANCE",
        difficulty=Difficulty.EASY,
        rationale="Specific person with title",
    ),
    EntityTestCase(
        id="ent-inst-010",
        text="Google",
        context="Just Google it.",
        expected_type="INSTANCE",
        difficulty=Difficulty.MEDIUM,
        rationale="Company name used as verb, still refers to specific entity",
    ),

    # -------------------------------------------------------------------------
    # CLASS: Categories and types (8 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-class-001",
        text="cat",
        context="The cat sat on the mat.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Common noun, category of animal",
    ),
    EntityTestCase(
        id="ent-class-002",
        text="apple",
        context="I ate an apple for lunch.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Lowercase, refers to fruit category",
    ),
    EntityTestCase(
        id="ent-class-003",
        text="company",
        context="The company announced layoffs.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Generic reference to a type of organization",
    ),
    EntityTestCase(
        id="ent-class-004",
        text="person",
        context="A person called for you.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Generic human category",
    ),
    EntityTestCase(
        id="ent-class-005",
        text="doctor",
        context="You should see a doctor about that.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Professional category, not specific person",
    ),
    EntityTestCase(
        id="ent-class-006",
        text="bank",
        context="Go to the bank and deposit this.",
        expected_type="CLASS",
        difficulty=Difficulty.MEDIUM,
        rationale="Generic institution, not a named bank",
    ),
    EntityTestCase(
        id="ent-class-007",
        text="book",
        context="I need to buy a book for class.",
        expected_type="CLASS",
        difficulty=Difficulty.EASY,
        rationale="Generic object category",
    ),
    EntityTestCase(
        id="ent-class-008",
        text="city",
        context="The city has a new mayor.",
        expected_type="CLASS",
        difficulty=Difficulty.MEDIUM,
        rationale="Generic reference to a city, not specific one",
    ),

    # -------------------------------------------------------------------------
    # NAMED_CONCEPT: Proper-named categories (6 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-concept-001",
        text="Christmas",
        context="We celebrate Christmas every year.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.EASY,
        rationale="Named holiday, category of event",
    ),
    EntityTestCase(
        id="ent-concept-002",
        text="Agile",
        context="We use Agile methodology in our team.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.MEDIUM,
        rationale="Named methodology/concept",
    ),
    EntityTestCase(
        id="ent-concept-003",
        text="Buddhism",
        context="She practices Buddhism.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.MEDIUM,
        rationale="Named religion/philosophy",
    ),
    EntityTestCase(
        id="ent-concept-004",
        text="Renaissance",
        context="The Renaissance changed European art.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.MEDIUM,
        rationale="Named historical period",
    ),
    EntityTestCase(
        id="ent-concept-005",
        text="English",
        context="She speaks English fluently.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.MEDIUM,
        rationale="Named language",
    ),
    EntityTestCase(
        id="ent-concept-006",
        text="Thanksgiving",
        context="Thanksgiving is my favorite holiday.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.EASY,
        rationale="Named holiday",
    ),

    # -------------------------------------------------------------------------
    # ROLE: Relational concepts (5 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-role-001",
        text="CEO of Apple",
        context="The CEO of Apple spoke at the event.",
        expected_type="ROLE",
        difficulty=Difficulty.EASY,
        rationale="Role defined relative to company",
    ),
    EntityTestCase(
        id="ent-role-002",
        text="Doug's wife",
        context="Doug's wife is a doctor.",
        expected_type="ROLE",
        difficulty=Difficulty.EASY,
        rationale="Role defined relative to person",
    ),
    EntityTestCase(
        id="ent-role-003",
        text="president of the company",
        context="The president of the company resigned.",
        expected_type="ROLE",
        difficulty=Difficulty.MEDIUM,
        rationale="Role relative to organization",
    ),
    EntityTestCase(
        id="ent-role-004",
        text="author of the book",
        context="The author of the book won an award.",
        expected_type="ROLE",
        difficulty=Difficulty.MEDIUM,
        rationale="Role relative to creative work",
    ),
    EntityTestCase(
        id="ent-role-005",
        text="manager at Google",
        context="She's a manager at Google.",
        expected_type="ROLE",
        difficulty=Difficulty.MEDIUM,
        rationale="Role at specific company",
    ),

    # -------------------------------------------------------------------------
    # ANAPHORA: References needing resolution (6 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-ana-001",
        text="he",
        context="Doug left early. He forgot his keys.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.TRIVIAL,
        rationale="Pronoun referring back to Doug",
    ),
    EntityTestCase(
        id="ent-ana-002",
        text="she",
        context="Sarah called. She wants to meet.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.TRIVIAL,
        rationale="Pronoun referring back to Sarah",
    ),
    EntityTestCase(
        id="ent-ana-003",
        text="it",
        context="The package arrived. It was damaged.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.TRIVIAL,
        rationale="Pronoun referring to package",
    ),
    EntityTestCase(
        id="ent-ana-004",
        text="they",
        context="The team met. They decided to postpone.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.TRIVIAL,
        rationale="Pronoun referring to team",
    ),
    EntityTestCase(
        id="ent-ana-005",
        text="this",
        context="Read the report. This is important.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.EASY,
        rationale="Demonstrative referring to report",
    ),
    EntityTestCase(
        id="ent-ana-006",
        text="that",
        context="He mentioned the deadline. That concerns me.",
        expected_type="ANAPHORA",
        difficulty=Difficulty.EASY,
        rationale="Demonstrative referring to deadline/statement",
    ),

    # -------------------------------------------------------------------------
    # GENERIC: Generic references (5 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-gen-001",
        text="someone",
        context="Someone left their umbrella.",
        expected_type="GENERIC",
        difficulty=Difficulty.TRIVIAL,
        rationale="Indefinite pronoun",
    ),
    EntityTestCase(
        id="ent-gen-002",
        text="everyone",
        context="Everyone enjoyed the party.",
        expected_type="GENERIC",
        difficulty=Difficulty.TRIVIAL,
        rationale="Universal quantifier",
    ),
    EntityTestCase(
        id="ent-gen-003",
        text="people",
        context="People often forget their passwords.",
        expected_type="GENERIC",
        difficulty=Difficulty.EASY,
        rationale="Generic reference to humans",
    ),
    EntityTestCase(
        id="ent-gen-004",
        text="anybody",
        context="Has anybody seen my keys?",
        expected_type="GENERIC",
        difficulty=Difficulty.TRIVIAL,
        rationale="Indefinite pronoun",
    ),
    EntityTestCase(
        id="ent-gen-005",
        text="nobody",
        context="Nobody knew the answer.",
        expected_type="GENERIC",
        difficulty=Difficulty.TRIVIAL,
        rationale="Negative indefinite pronoun",
    ),

    # -------------------------------------------------------------------------
    # HARD/AMBIGUOUS cases (5 cases)
    # -------------------------------------------------------------------------
    EntityTestCase(
        id="ent-hard-001",
        text="Apple",
        context="Apple is delicious.",
        expected_type="CLASS",
        difficulty=Difficulty.HARD,
        rationale="Could be company or fruit - 'delicious' suggests fruit",
        acceptable_types=["CLASS", "INSTANCE"],
    ),
    EntityTestCase(
        id="ent-hard-002",
        text="The President",
        context="The President signed the bill.",
        expected_type="INSTANCE",
        difficulty=Difficulty.HARD,
        rationale="Definite reference to specific person holding role",
        acceptable_types=["INSTANCE", "ROLE"],
    ),
    EntityTestCase(
        id="ent-hard-003",
        text="Jordan",
        context="Jordan is a great basketball player.",
        expected_type="INSTANCE",
        difficulty=Difficulty.HARD,
        rationale="Could be person (Michael Jordan) or country",
    ),
    EntityTestCase(
        id="ent-hard-004",
        text="Python",
        context="I'm learning Python.",
        expected_type="NAMED_CONCEPT",
        difficulty=Difficulty.HARD,
        rationale="Programming language (named concept) not snake (class)",
        acceptable_types=["NAMED_CONCEPT", "CLASS"],
    ),
    EntityTestCase(
        id="ent-hard-005",
        text="Democrats",
        context="Democrats support the bill.",
        expected_type="INSTANCE",
        difficulty=Difficulty.HARD,
        rationale="Refers to specific political party/group",
        acceptable_types=["INSTANCE", "CLASS"],
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_wsd_cases_by_difficulty(difficulty: Difficulty) -> list[WSDTestCase]:
    """Get WSD test cases filtered by difficulty."""
    return [c for c in WSD_GROUND_TRUTH if c.difficulty == difficulty]


def get_entity_cases_by_type(entity_type: str) -> list[EntityTestCase]:
    """Get entity test cases filtered by expected type."""
    return [c for c in ENTITY_GROUND_TRUTH if c.expected_type == entity_type]


def get_wsd_stats() -> dict[str, Any]:
    """Get statistics about the WSD dataset."""
    by_difficulty = {}
    by_domain = {}
    for case in WSD_GROUND_TRUTH:
        by_difficulty[case.difficulty.value] = by_difficulty.get(case.difficulty.value, 0) + 1
        if case.domain:
            by_domain[case.domain] = by_domain.get(case.domain, 0) + 1

    return {
        "total": len(WSD_GROUND_TRUTH),
        "by_difficulty": by_difficulty,
        "by_domain": by_domain,
    }


def get_entity_stats() -> dict[str, Any]:
    """Get statistics about the entity dataset."""
    by_difficulty = {}
    by_type = {}
    for case in ENTITY_GROUND_TRUTH:
        by_difficulty[case.difficulty.value] = by_difficulty.get(case.difficulty.value, 0) + 1
        by_type[case.expected_type] = by_type.get(case.expected_type, 0) + 1

    return {
        "total": len(ENTITY_GROUND_TRUTH),
        "by_difficulty": by_difficulty,
        "by_type": by_type,
    }
