# Test Dataset Specification

**Version:** 1.0.0
**Status:** Specification
**Purpose:** Define test cases for evaluating decomposition quality

---

## Overview

This document specifies the test dataset for validating the implicit knowledge decomposition pipeline. The dataset covers all extraction types with ground truth annotations for evaluation.

---

## Dataset Structure

### Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Simple Sentences | 20 | Baseline extraction |
| Presupposition Triggers | 30 | Each trigger type × 3 |
| Commonsense Events | 20 | Social/physical events |
| Negation Patterns | 15 | Various negation types |
| Temporal/Modal | 15 | Tense, aspect, modality |
| **Total** | **100** | |

### Data Format

```python
@dataclass
class TestCase:
    """A test case for decomposition evaluation."""

    # Identifiers
    id: str                             # "TC-001"
    category: str                       # "presupposition_iterative"

    # Input
    text: str                           # "Doug forgot the meeting again"
    context: str | None                 # Optional surrounding context

    # Expected outputs (ground truth)
    expected_entities: list[EntityAnnotation]
    expected_roles: list[RoleAnnotation]
    expected_presuppositions: list[PresuppositionAnnotation]
    expected_inferences: list[InferenceAnnotation]
    expected_temporal: TemporalAnnotation | None
    expected_modality: ModalityAnnotation | None
    expected_negation: NegationAnnotation | None

    # Metadata
    difficulty: str                     # "easy", "medium", "hard"
    notes: str | None                   # Edge cases, ambiguities


@dataclass
class EntityAnnotation:
    text: str
    entity_type: str                    # "INSTANCE", "CLASS", etc.
    synset: str | None                  # "meeting.n.01"


@dataclass
class RoleAnnotation:
    predicate: str
    predicate_sense: str
    role: str                           # "ARG0", "ARG1"
    filler: str


@dataclass
class PresuppositionAnnotation:
    content: str
    trigger_type: str
    trigger_text: str


@dataclass
class InferenceAnnotation:
    relation: str                       # "xIntent", "xEffect"
    head: str
    tail: str


@dataclass
class TemporalAnnotation:
    tense: str
    aspect: str
    reference: str | None


@dataclass
class ModalityAnnotation:
    modal_type: str
    marker: str | None
    certainty: float | None


@dataclass
class NegationAnnotation:
    is_negated: bool
    cue: str | None
    scope: str | None
```

---

## Test Cases

### Category 1: Simple Sentences (20)

Basic sentences for baseline extraction validation.

```yaml
- id: TC-001
  category: simple
  text: "Doug went to the store."
  difficulty: easy
  expected_entities:
    - text: "Doug"
      entity_type: "INSTANCE"
    - text: "the store"
      entity_type: "INSTANCE"
  expected_roles:
    - predicate: "went"
      predicate_sense: "go.v.01"
      role: "ARG0"
      filler: "Doug"
    - predicate: "went"
      predicate_sense: "go.v.01"
      role: "ARG4"  # Destination
      filler: "the store"
  expected_presuppositions:
    - content: "A specific store exists"
      trigger_type: "definite_description"
      trigger_text: "the store"
  expected_temporal:
    tense: "past"
    aspect: "activity"

- id: TC-002
  category: simple
  text: "Sarah is a doctor."
  difficulty: easy
  expected_entities:
    - text: "Sarah"
      entity_type: "INSTANCE"
    - text: "doctor"
      entity_type: "CLASS"
      synset: "doctor.n.01"
  expected_roles:
    - predicate: "is"
      predicate_sense: "be.v.01"
      role: "ARG1"
      filler: "Sarah"
    - predicate: "is"
      predicate_sense: "be.v.01"
      role: "ARG2"
      filler: "doctor"
  expected_temporal:
    tense: "present"
    aspect: "state"

- id: TC-003
  category: simple
  text: "The cat sat on the mat."
  difficulty: easy
  expected_entities:
    - text: "The cat"
      entity_type: "INSTANCE"
    - text: "the mat"
      entity_type: "INSTANCE"
  expected_roles:
    - predicate: "sat"
      predicate_sense: "sit.v.01"
      role: "ARG0"
      filler: "The cat"
    - predicate: "sat"
      predicate_sense: "sit.v.01"
      role: "ARGM-LOC"
      filler: "on the mat"
  expected_presuppositions:
    - content: "A specific cat exists"
      trigger_type: "definite_description"
      trigger_text: "The cat"
    - content: "A specific mat exists"
      trigger_type: "definite_description"
      trigger_text: "the mat"

- id: TC-004
  category: simple
  text: "John bought a book."
  difficulty: easy
  expected_entities:
    - text: "John"
      entity_type: "INSTANCE"
    - text: "a book"
      entity_type: "CLASS"
      synset: "book.n.01"
  expected_roles:
    - predicate: "bought"
      predicate_sense: "buy.v.01"
      role: "ARG0"
      filler: "John"
    - predicate: "bought"
      predicate_sense: "buy.v.01"
      role: "ARG1"
      filler: "a book"
  expected_inferences:
    - relation: "xEffect"
      head: "John bought a book"
      tail: "John has a book"
    - relation: "xIntent"
      head: "John bought a book"
      tail: "to read it"

- id: TC-005
  category: simple
  text: "The children played in the park."
  difficulty: easy
  expected_entities:
    - text: "The children"
      entity_type: "CLASS"
    - text: "the park"
      entity_type: "INSTANCE"
  expected_roles:
    - predicate: "played"
      predicate_sense: "play.v.01"
      role: "ARG0"
      filler: "The children"
    - predicate: "played"
      predicate_sense: "play.v.01"
      role: "ARGM-LOC"
      filler: "in the park"

# ... (15 more simple sentences)

- id: TC-006
  category: simple
  text: "Mary gave Tom a present."
  difficulty: easy
  notes: "Ditransitive verb - tests ARG2 extraction"
  expected_roles:
    - predicate: "gave"
      role: "ARG0"
      filler: "Mary"
    - predicate: "gave"
      role: "ARG1"
      filler: "a present"
    - predicate: "gave"
      role: "ARG2"
      filler: "Tom"

- id: TC-007
  category: simple
  text: "The dog is sleeping."
  difficulty: easy
  expected_temporal:
    tense: "present"
    aspect: "activity"

- id: TC-008
  category: simple
  text: "She runs every morning."
  difficulty: easy
  expected_temporal:
    tense: "present"
    aspect: "activity"
    reference: "every morning"

- id: TC-009
  category: simple
  text: "They will arrive tomorrow."
  difficulty: easy
  expected_temporal:
    tense: "future"
    aspect: "achievement"
    reference: "tomorrow"

- id: TC-010
  category: simple
  text: "The meeting lasted two hours."
  difficulty: easy
  expected_temporal:
    tense: "past"
    aspect: "accomplishment"
    reference: "two hours"
```

### Category 2: Presupposition Triggers (30)

Three examples per trigger type.

```yaml
# DEFINITE DESCRIPTION (3)
- id: TC-021
  category: presupposition_definite
  text: "The president gave a speech."
  difficulty: easy
  expected_presuppositions:
    - content: "A specific president exists and is contextually identifiable"
      trigger_type: "definite_description"
      trigger_text: "The president"

- id: TC-022
  category: presupposition_definite
  text: "I returned the book to the library."
  difficulty: easy
  expected_presuppositions:
    - content: "A specific book exists that was previously borrowed"
      trigger_type: "definite_description"
      trigger_text: "the book"
    - content: "A specific library exists"
      trigger_type: "definite_description"
      trigger_text: "the library"

- id: TC-023
  category: presupposition_definite
  text: "The winner will receive a prize."
  difficulty: medium
  expected_presuppositions:
    - content: "There will be a winner"
      trigger_type: "definite_description"
      trigger_text: "The winner"

# FACTIVE VERBS (3)
- id: TC-024
  category: presupposition_factive
  text: "Doug realized he was late."
  difficulty: easy
  expected_presuppositions:
    - content: "Doug was late"
      trigger_type: "factive_verb"
      trigger_text: "realized"

- id: TC-025
  category: presupposition_factive
  text: "Sarah regrets selling her car."
  difficulty: easy
  expected_presuppositions:
    - content: "Sarah sold her car"
      trigger_type: "factive_verb"
      trigger_text: "regrets"

- id: TC-026
  category: presupposition_factive
  text: "They knew the answer was wrong."
  difficulty: easy
  expected_presuppositions:
    - content: "The answer was wrong"
      trigger_type: "factive_verb"
      trigger_text: "knew"

# CHANGE OF STATE (3)
- id: TC-027
  category: presupposition_change_of_state
  text: "Doug stopped smoking."
  difficulty: easy
  expected_presuppositions:
    - content: "Doug used to smoke"
      trigger_type: "change_of_state"
      trigger_text: "stopped"

- id: TC-028
  category: presupposition_change_of_state
  text: "She started working from home."
  difficulty: easy
  expected_presuppositions:
    - content: "She was not working from home before"
      trigger_type: "change_of_state"
      trigger_text: "started"

- id: TC-029
  category: presupposition_change_of_state
  text: "The company continues to grow."
  difficulty: easy
  expected_presuppositions:
    - content: "The company was already growing"
      trigger_type: "change_of_state"
      trigger_text: "continues"

# ITERATIVE (3)
- id: TC-030
  category: presupposition_iterative
  text: "Doug forgot the meeting again."
  difficulty: easy
  expected_presuppositions:
    - content: "Doug forgot before"
      trigger_type: "iterative"
      trigger_text: "again"
    - content: "A specific meeting exists"
      trigger_type: "definite_description"
      trigger_text: "the meeting"

- id: TC-031
  category: presupposition_iterative
  text: "She won another award."
  difficulty: easy
  expected_presuppositions:
    - content: "She won an award before"
      trigger_type: "iterative"
      trigger_text: "another"

- id: TC-032
  category: presupposition_iterative
  text: "He is still waiting."
  difficulty: easy
  expected_presuppositions:
    - content: "He was waiting before"
      trigger_type: "iterative"
      trigger_text: "still"

# TEMPORAL CLAUSE (3)
- id: TC-033
  category: presupposition_temporal
  text: "Before Doug left, he locked the door."
  difficulty: medium
  expected_presuppositions:
    - content: "Doug left"
      trigger_type: "temporal_clause"
      trigger_text: "Before"
    - content: "There is a door"
      trigger_type: "definite_description"
      trigger_text: "the door"

- id: TC-034
  category: presupposition_temporal
  text: "After the meeting ended, we went to lunch."
  difficulty: medium
  expected_presuppositions:
    - content: "The meeting ended"
      trigger_type: "temporal_clause"
      trigger_text: "After"

- id: TC-035
  category: presupposition_temporal
  text: "When she arrived, everyone applauded."
  difficulty: medium
  expected_presuppositions:
    - content: "She arrived"
      trigger_type: "temporal_clause"
      trigger_text: "When"

# CLEFT SENTENCES (3)
- id: TC-036
  category: presupposition_cleft
  text: "It was Doug who called."
  difficulty: medium
  expected_presuppositions:
    - content: "Someone called"
      trigger_type: "cleft"
      trigger_text: "It was"

- id: TC-037
  category: presupposition_cleft
  text: "What Mary wanted was a vacation."
  difficulty: medium
  expected_presuppositions:
    - content: "Mary wanted something"
      trigger_type: "cleft"
      trigger_text: "What"

- id: TC-038
  category: presupposition_cleft
  text: "It's the deadline that worries me."
  difficulty: medium
  expected_presuppositions:
    - content: "Something worries me"
      trigger_type: "cleft"
      trigger_text: "It's"

# COMPARATIVE (3)
- id: TC-039
  category: presupposition_comparative
  text: "Doug is taller than Sarah."
  difficulty: easy
  expected_presuppositions:
    - content: "Sarah has a height"
      trigger_type: "comparative"
      trigger_text: "taller than"
    - content: "Doug has a height"
      trigger_type: "comparative"
      trigger_text: "taller than"

- id: TC-040
  category: presupposition_comparative
  text: "This year's sales are better than last year's."
  difficulty: medium
  expected_presuppositions:
    - content: "Last year had sales"
      trigger_type: "comparative"
      trigger_text: "better than"

- id: TC-041
  category: presupposition_comparative
  text: "She speaks French as well as her mother."
  difficulty: medium
  expected_presuppositions:
    - content: "Her mother speaks French"
      trigger_type: "comparative"
      trigger_text: "as well as"

# IMPLICATIVE VERBS (3)
- id: TC-042
  category: presupposition_implicative
  text: "Doug managed to finish the project."
  difficulty: easy
  expected_presuppositions:
    - content: "Finishing the project was difficult"
      trigger_type: "implicative"
      trigger_text: "managed"

- id: TC-043
  category: presupposition_implicative
  text: "She remembered to call her mom."
  difficulty: easy
  expected_presuppositions:
    - content: "She was supposed to call her mom"
      trigger_type: "implicative"
      trigger_text: "remembered"

- id: TC-044
  category: presupposition_implicative
  text: "He forgot to lock the door."
  difficulty: easy
  expected_presuppositions:
    - content: "He was supposed to lock the door"
      trigger_type: "implicative"
      trigger_text: "forgot"

# COUNTERFACTUAL (3)
- id: TC-045
  category: presupposition_counterfactual
  text: "If Doug had known, he would have helped."
  difficulty: hard
  expected_presuppositions:
    - content: "Doug did not know"
      trigger_type: "counterfactual"
      trigger_text: "If Doug had"

- id: TC-046
  category: presupposition_counterfactual
  text: "I wish I had studied harder."
  difficulty: hard
  expected_presuppositions:
    - content: "I did not study hard enough"
      trigger_type: "counterfactual"
      trigger_text: "wish I had"

- id: TC-047
  category: presupposition_counterfactual
  text: "She would have come if she had been invited."
  difficulty: hard
  expected_presuppositions:
    - content: "She was not invited"
      trigger_type: "counterfactual"
      trigger_text: "if she had been"

# POSSESSIVE (3)
- id: TC-048
  category: presupposition_possessive
  text: "Doug's car is red."
  difficulty: easy
  expected_presuppositions:
    - content: "Doug has a car"
      trigger_type: "possessive"
      trigger_text: "Doug's"

- id: TC-049
  category: presupposition_possessive
  text: "Her brother lives in Paris."
  difficulty: easy
  expected_presuppositions:
    - content: "She has a brother"
      trigger_type: "possessive"
      trigger_text: "Her"

- id: TC-050
  category: presupposition_possessive
  text: "The company's CEO resigned."
  difficulty: easy
  expected_presuppositions:
    - content: "The company has a CEO"
      trigger_type: "possessive"
      trigger_text: "company's"
```

### Category 3: Commonsense Events (20)

Social and physical events for inference testing.

```yaml
- id: TC-051
  category: commonsense_social
  text: "Doug helped Sarah move to her new apartment."
  difficulty: medium
  expected_inferences:
    - relation: "xIntent"
      tail: "to be helpful"
    - relation: "xIntent"
      tail: "to help a friend"
    - relation: "xEffect"
      tail: "Sarah moves successfully"
    - relation: "xReact"
      tail: "helpful"
    - relation: "oReact"
      tail: "grateful"

- id: TC-052
  category: commonsense_social
  text: "She apologized for being late."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to make amends"
    - relation: "xReact"
      tail: "sorry"
    - relation: "xAttr"
      tail: "polite"

- id: TC-053
  category: commonsense_social
  text: "He complimented her on her presentation."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to be encouraging"
    - relation: "oReact"
      tail: "pleased"
    - relation: "xAttr"
      tail: "kind"

- id: TC-054
  category: commonsense_social
  text: "They argued about politics."
  difficulty: medium
  expected_inferences:
    - relation: "xReact"
      tail: "frustrated"
    - relation: "xEffect"
      tail: "relationship tension"

- id: TC-055
  category: commonsense_social
  text: "Doug thanked the waiter for the excellent service."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to show appreciation"
    - relation: "xAttr"
      tail: "grateful"
    - relation: "oReact"
      tail: "appreciated"

- id: TC-056
  category: commonsense_physical
  text: "She dropped the glass and it shattered."
  difficulty: easy
  expected_inferences:
    - relation: "xReact"
      tail: "startled"
    - relation: "xEffect"
      tail: "has to clean up"
    - relation: "Causes"
      tail: "broken glass on floor"

- id: TC-057
  category: commonsense_physical
  text: "He overslept and missed his flight."
  difficulty: medium
  expected_inferences:
    - relation: "xReact"
      tail: "panicked"
    - relation: "xEffect"
      tail: "has to rebook flight"
    - relation: "Causes"
      tail: "delayed travel"

- id: TC-058
  category: commonsense_social
  text: "Doug bought flowers for his wife."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to show love"
    - relation: "xIntent"
      tail: "to make her happy"
    - relation: "oReact"
      tail: "touched"
    - relation: "xAttr"
      tail: "romantic"

- id: TC-059
  category: commonsense_social
  text: "She refused to attend the party."
  difficulty: medium
  expected_inferences:
    - relation: "xIntent"
      tail: "to avoid the party"
    - relation: "xReact"
      tail: "relieved"
    - relation: "oReact"
      tail: "disappointed"

- id: TC-060
  category: commonsense_physical
  text: "The ice cream melted in the sun."
  difficulty: easy
  expected_inferences:
    - relation: "Causes"
      tail: "mess"
    - relation: "xEffect"
      tail: "can't eat it anymore"

# ... (10 more commonsense cases)

- id: TC-061
  category: commonsense_social
  text: "Doug lied to his boss about being sick."
  difficulty: hard
  expected_inferences:
    - relation: "xIntent"
      tail: "to avoid work"
    - relation: "xReact"
      tail: "guilty"
    - relation: "xAttr"
      tail: "dishonest"
    - relation: "xEffect"
      tail: "might get caught"

- id: TC-062
  category: commonsense_social
  text: "She surprised him with a birthday party."
  difficulty: medium
  expected_inferences:
    - relation: "xIntent"
      tail: "to make him happy"
    - relation: "oReact"
      tail: "surprised"
    - relation: "oReact"
      tail: "delighted"

- id: TC-063
  category: commonsense_physical
  text: "He spilled coffee on his laptop."
  difficulty: medium
  expected_inferences:
    - relation: "xReact"
      tail: "panicked"
    - relation: "xEffect"
      tail: "laptop damaged"
    - relation: "Causes"
      tail: "potential data loss"

- id: TC-064
  category: commonsense_social
  text: "Doug apologized even though it wasn't his fault."
  difficulty: hard
  expected_inferences:
    - relation: "xIntent"
      tail: "to keep the peace"
    - relation: "xAttr"
      tail: "considerate"

- id: TC-065
  category: commonsense_social
  text: "She stayed late to finish the report."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to meet the deadline"
    - relation: "xReact"
      tail: "tired"
    - relation: "xAttr"
      tail: "hardworking"

- id: TC-066
  category: commonsense_physical
  text: "The tree fell during the storm."
  difficulty: easy
  expected_inferences:
    - relation: "Causes"
      tail: "damage"
    - relation: "xEffect"
      tail: "blocked road"

- id: TC-067
  category: commonsense_social
  text: "He congratulated her on the promotion."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to show support"
    - relation: "oReact"
      tail: "happy"
    - relation: "xAttr"
      tail: "supportive"

- id: TC-068
  category: commonsense_social
  text: "Doug ignored the warning signs."
  difficulty: medium
  expected_inferences:
    - relation: "xReact"
      tail: "unconcerned"
    - relation: "xEffect"
      tail: "faces consequences"
    - relation: "xAttr"
      tail: "reckless"

- id: TC-069
  category: commonsense_physical
  text: "She burned the dinner while on the phone."
  difficulty: medium
  expected_inferences:
    - relation: "xReact"
      tail: "frustrated"
    - relation: "xEffect"
      tail: "has to order food"
    - relation: "Causes"
      tail: "smoke in kitchen"

- id: TC-070
  category: commonsense_social
  text: "They celebrated their anniversary at a fancy restaurant."
  difficulty: easy
  expected_inferences:
    - relation: "xIntent"
      tail: "to commemorate"
    - relation: "xReact"
      tail: "happy"
    - relation: "xAttr"
      tail: "romantic"
```

### Category 4: Negation Patterns (15)

Various negation constructions.

```yaml
- id: TC-071
  category: negation_explicit
  text: "Doug did not attend the meeting."
  difficulty: easy
  expected_negation:
    is_negated: true
    cue: "not"
    scope: "attend the meeting"
  expected_roles:
    - predicate: "attend"
      role: "ARG0"
      filler: "Doug"

- id: TC-072
  category: negation_contraction
  text: "She doesn't like coffee."
  difficulty: easy
  expected_negation:
    is_negated: true
    cue: "n't"
    scope: "like coffee"

- id: TC-073
  category: negation_never
  text: "He never forgets a face."
  difficulty: easy
  expected_negation:
    is_negated: true
    cue: "never"
    scope: "forgets a face"

- id: TC-074
  category: negation_no
  text: "No students passed the exam."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "No"
    scope: "students passed the exam"

- id: TC-075
  category: negation_nothing
  text: "Nothing happened yesterday."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "Nothing"
    scope: "happened yesterday"

- id: TC-076
  category: negation_without
  text: "She left without saying goodbye."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "without"
    scope: "saying goodbye"

- id: TC-077
  category: negation_prefix
  text: "The results were unexpected."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "un-"
    scope: "expected"
  notes: "Morphological negation"

- id: TC-078
  category: negation_double
  text: "I can't not go to the party."
  difficulty: hard
  expected_negation:
    is_negated: false
    cue: "can't not"
  notes: "Double negation = positive"

- id: TC-079
  category: negation_scope_ambiguous
  text: "Doug didn't leave because he was angry."
  difficulty: hard
  expected_negation:
    is_negated: true
    cue: "n't"
  notes: "Ambiguous scope - anger could be reason for staying or for (not) leaving"

- id: TC-080
  category: negation_implicit
  text: "Doug failed to notice the change."
  difficulty: hard
  expected_negation:
    is_negated: true
    cue: "failed"
    scope: "notice the change"
  notes: "Implicit negation via 'fail'"

- id: TC-081
  category: negation_hardly
  text: "She hardly ever complains."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "hardly"
    scope: "ever complains"

- id: TC-082
  category: negation_neither
  text: "Neither candidate won the debate."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "Neither"
    scope: "candidate won"

- id: TC-083
  category: negation_nobody
  text: "Nobody knows the answer."
  difficulty: easy
  expected_negation:
    is_negated: true
    cue: "Nobody"
    scope: "knows the answer"

- id: TC-084
  category: negation_seldom
  text: "He seldom makes mistakes."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "seldom"
    scope: "makes mistakes"

- id: TC-085
  category: negation_refuse
  text: "She refused to answer the question."
  difficulty: medium
  expected_negation:
    is_negated: true
    cue: "refused"
    scope: "answer the question"
  notes: "Implicit negation via 'refuse'"
```

### Category 5: Temporal and Modal (15)

Various tense, aspect, and modality constructions.

```yaml
# TEMPORAL (8)
- id: TC-086
  category: temporal_past_simple
  text: "Doug arrived yesterday."
  difficulty: easy
  expected_temporal:
    tense: "past"
    aspect: "achievement"
    reference: "yesterday"

- id: TC-087
  category: temporal_present_progressive
  text: "She is working on the project."
  difficulty: easy
  expected_temporal:
    tense: "present"
    aspect: "activity"

- id: TC-088
  category: temporal_future
  text: "They will announce the results next week."
  difficulty: easy
  expected_temporal:
    tense: "future"
    aspect: "achievement"
    reference: "next week"

- id: TC-089
  category: temporal_past_perfect
  text: "Doug had already left when she arrived."
  difficulty: medium
  expected_temporal:
    tense: "past"
    aspect: "accomplishment"
  notes: "Past perfect indicates prior past"

- id: TC-090
  category: temporal_habitual
  text: "Doug drinks tea every morning."
  difficulty: easy
  expected_temporal:
    tense: "present"
    aspect: "state"
    reference: "every morning"

- id: TC-091
  category: temporal_duration
  text: "She studied for three hours."
  difficulty: easy
  expected_temporal:
    tense: "past"
    aspect: "accomplishment"
    reference: "three hours"

- id: TC-092
  category: temporal_since
  text: "They have lived here since 2010."
  difficulty: medium
  expected_temporal:
    tense: "present"
    aspect: "state"
    reference: "since 2010"

- id: TC-093
  category: temporal_while
  text: "Doug read while waiting for the bus."
  difficulty: medium
  expected_temporal:
    tense: "past"
    aspect: "activity"

# MODALITY (7)
- id: TC-094
  category: modality_epistemic_certain
  text: "Doug must be at home by now."
  difficulty: medium
  expected_modality:
    modal_type: "epistemic"
    marker: "must"
    certainty: 0.9

- id: TC-095
  category: modality_epistemic_probable
  text: "She will probably arrive late."
  difficulty: easy
  expected_modality:
    modal_type: "epistemic"
    marker: "probably"
    certainty: 0.7

- id: TC-096
  category: modality_epistemic_possible
  text: "He might come to the party."
  difficulty: easy
  expected_modality:
    modal_type: "epistemic"
    marker: "might"
    certainty: 0.5

- id: TC-097
  category: modality_deontic_required
  text: "You must submit the form today."
  difficulty: medium
  expected_modality:
    modal_type: "deontic"
    marker: "must"

- id: TC-098
  category: modality_deontic_recommended
  text: "You should review the document."
  difficulty: easy
  expected_modality:
    modal_type: "deontic"
    marker: "should"

- id: TC-099
  category: modality_evidential_reported
  text: "Apparently, the project was cancelled."
  difficulty: medium
  expected_modality:
    modal_type: "evidential"
    marker: "Apparently"
    certainty: 0.6

- id: TC-100
  category: modality_dynamic
  text: "Doug can speak three languages."
  difficulty: easy
  expected_modality:
    modal_type: "dynamic"
    marker: "can"
  notes: "Ability modal"
```

---

## Evaluation Metrics

### Per-Category Metrics

```python
@dataclass
class CategoryMetrics:
    """Metrics for a test category."""

    category: str
    total_cases: int
    passed_cases: int

    # Extraction metrics
    entity_precision: float
    entity_recall: float
    entity_f1: float

    role_precision: float
    role_recall: float
    role_f1: float

    presup_precision: float
    presup_recall: float
    presup_f1: float

    inference_precision: float
    inference_recall: float
    inference_f1: float

    # Correctness
    temporal_accuracy: float
    modality_accuracy: float
    negation_accuracy: float
```

### Evaluation Functions

```python
def evaluate_entities(
    predicted: list[EntityAnnotation],
    expected: list[EntityAnnotation],
) -> tuple[float, float, float]:
    """Calculate precision, recall, F1 for entity extraction."""

def evaluate_roles(
    predicted: list[RoleAnnotation],
    expected: list[RoleAnnotation],
) -> tuple[float, float, float]:
    """Calculate precision, recall, F1 for role extraction."""

def evaluate_presuppositions(
    predicted: list[PresuppositionAnnotation],
    expected: list[PresuppositionAnnotation],
    fuzzy_match: bool = True,  # Allow semantic similarity
) -> tuple[float, float, float]:
    """Calculate precision, recall, F1 for presupposition extraction."""

def evaluate_inferences(
    predicted: list[InferenceAnnotation],
    expected: list[InferenceAnnotation],
    fuzzy_match: bool = True,
) -> tuple[float, float, float]:
    """Calculate precision, recall, F1 for inference extraction."""
```

---

## Usage

### Loading Test Data

```python
from implicit_knowledge_graphs.tests.data import load_test_dataset

# Load all test cases
dataset = load_test_dataset()

# Load specific categories
presup_cases = load_test_dataset(category="presupposition")
negation_cases = load_test_dataset(category="negation")

# Load by difficulty
hard_cases = load_test_dataset(difficulty="hard")
```

### Running Evaluation

```python
from implicit_knowledge_graphs.evaluation import evaluate_pipeline

# Run evaluation
results = await evaluate_pipeline(
    pipeline=my_pipeline,
    dataset=test_dataset,
    output_dir="./eval_results",
)

print(f"Overall F1: {results.overall_f1}")
print(f"Entity F1: {results.entity_f1}")
print(f"Presupposition F1: {results.presup_f1}")
```

---

## Data Files

### File Format

Test cases stored in YAML for readability:

```
tests/
└── data/
    ├── simple_sentences.yaml
    ├── presupposition_triggers.yaml
    ├── commonsense_events.yaml
    ├── negation_patterns.yaml
    └── temporal_modal.yaml
```

### Schema Validation

```python
from pydantic import BaseModel

class TestCaseSchema(BaseModel):
    """Schema for validating test case YAML."""

    id: str
    category: str
    text: str
    difficulty: Literal["easy", "medium", "hard"]
    # ... all other fields with validation
```

---

**End of Test Dataset Specification**
