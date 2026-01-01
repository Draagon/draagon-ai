"""Tests for Neo4j persistence layer.

Tests saving and loading SemanticGraph to/from Neo4j,
including the 5 reference test cases:
1. RPG Game Master - Complex world state with NPCs, items, events
2. Claude Code MCP - Codebase knowledge graph
3. Pharma Field Rep - Patient/doctor/drug relationships
4. Fitness Coach - User progress and workout tracking
5. Educational Tutor - Student knowledge state

These tests require a running Neo4j instance. Skip with:
    pytest tests/cognition/decomposition/graph/test_neo4j_store.py -v -k "not integration"
"""

import pytest
from datetime import datetime, timezone, timedelta

from draagon_ai.cognition.decomposition.graph import (
    SemanticGraph,
    GraphNode,
    GraphEdge,
    NodeType,
)
from draagon_ai.cognition.decomposition.graph.neo4j_store import (
    Neo4jGraphStoreSync,
    Neo4jConfig,
)


# =============================================================================
# Test Configuration
# =============================================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "draagon-ai-2025"


def neo4j_available() -> bool:
    """Check if Neo4j is available for testing."""
    try:
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        store.close()
        return True
    except Exception:
        return False


# Skip all integration tests if Neo4j is not available
pytestmark = pytest.mark.skipif(
    not neo4j_available(),
    reason="Neo4j not available at localhost:7687"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Create a Neo4j store connection."""
    store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    yield store
    store.close()


@pytest.fixture
def clean_instance(store):
    """Create a clean test instance, cleaning up before and after."""
    instance_id = "test-instance"
    # Clean before
    with store.driver.session(database=store.database) as session:
        session.run(
            "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
            instance_id=instance_id,
        )
    yield instance_id
    # Clean after
    with store.driver.session(database=store.database) as session:
        session.run(
            "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
            instance_id=instance_id,
        )


# =============================================================================
# Test Case Builders
# =============================================================================


def build_rpg_game_master_graph() -> SemanticGraph:
    """Build test graph for RPG Game Master scenario.

    This represents a complex game world with:
    - Players and NPCs with attributes
    - Locations and connections
    - Items with properties
    - Events and history
    - Relationships (owns, knows, located_at, etc.)
    """
    graph = SemanticGraph()

    # Player characters
    player1 = graph.create_node("Aldric the Bold", NodeType.INSTANCE,
        properties={"type": "player", "class": "warrior", "level": 12, "hp": 145})
    player2 = graph.create_node("Seraphina", NodeType.INSTANCE,
        properties={"type": "player", "class": "mage", "level": 11, "hp": 78})

    # NPCs
    npc_blacksmith = graph.create_node("Grimjaw the Blacksmith", NodeType.INSTANCE,
        properties={"type": "npc", "occupation": "blacksmith", "disposition": "friendly"})
    npc_tavern = graph.create_node("Old Martha", NodeType.INSTANCE,
        properties={"type": "npc", "occupation": "tavern_keeper", "disposition": "neutral"})
    npc_villain = graph.create_node("Lord Malachar", NodeType.INSTANCE,
        properties={"type": "npc", "role": "antagonist", "disposition": "hostile"})

    # Locations
    loc_village = graph.create_node("Millbrook Village", NodeType.INSTANCE,
        properties={"type": "location", "region": "Westlands", "population": 450})
    loc_tavern = graph.create_node("The Rusty Tankard", NodeType.INSTANCE,
        properties={"type": "location", "subtype": "tavern", "in_village": True})
    loc_forge = graph.create_node("Grimjaw's Forge", NodeType.INSTANCE,
        properties={"type": "location", "subtype": "shop"})
    loc_dungeon = graph.create_node("Crypts of the Fallen", NodeType.INSTANCE,
        properties={"type": "location", "subtype": "dungeon", "difficulty": "hard"})

    # Items
    sword = graph.create_node("Flamekeeper", NodeType.INSTANCE,
        properties={"type": "item", "subtype": "weapon", "damage": "2d8+3", "enchantment": "fire"})
    potion = graph.create_node("Greater Healing Potion", NodeType.INSTANCE,
        properties={"type": "item", "subtype": "consumable", "effect": "restore_50_hp", "quantity": 3})
    artifact = graph.create_node("Orb of Shadows", NodeType.INSTANCE,
        properties={"type": "item", "subtype": "artifact", "power_level": "legendary"})

    # Classes (semantic types)
    class_warrior = graph.create_node("warrior.n.01", NodeType.CLASS, synset_id="warrior.n.01")
    class_mage = graph.create_node("mage.n.01", NodeType.CLASS, synset_id="mage.n.01")
    class_person = graph.create_node("person.n.01", NodeType.CLASS, synset_id="person.n.01")
    class_location = graph.create_node("location.n.01", NodeType.CLASS, synset_id="location.n.01")
    class_item = graph.create_node("item.n.01", NodeType.CLASS, synset_id="item.n.01")

    # Events (past game sessions)
    event1 = graph.create_node("Battle of Millbrook", NodeType.EVENT,
        properties={"session": 5, "date": "2024-01-15", "outcome": "victory"})
    event2 = graph.create_node("Discovery of the Orb", NodeType.EVENT,
        properties={"session": 8, "date": "2024-02-01", "location": "Crypts"})

    # Relationships - ownership
    graph.create_edge(player1.node_id, sword.node_id, "owns")
    graph.create_edge(player2.node_id, potion.node_id, "owns")
    graph.create_edge(npc_villain.node_id, artifact.node_id, "possesses")

    # Relationships - location
    graph.create_edge(loc_tavern.node_id, loc_village.node_id, "located_in")
    graph.create_edge(loc_forge.node_id, loc_village.node_id, "located_in")
    graph.create_edge(npc_blacksmith.node_id, loc_forge.node_id, "works_at")
    graph.create_edge(npc_tavern.node_id, loc_tavern.node_id, "works_at")

    # Relationships - social
    graph.create_edge(player1.node_id, npc_blacksmith.node_id, "knows")
    graph.create_edge(player2.node_id, npc_tavern.node_id, "knows")
    graph.create_edge(player1.node_id, player2.node_id, "party_member")
    graph.create_edge(player2.node_id, player1.node_id, "party_member")

    # Relationships - class hierarchy
    graph.create_edge(player1.node_id, class_warrior.node_id, "instance_of")
    graph.create_edge(player2.node_id, class_mage.node_id, "instance_of")
    graph.create_edge(class_warrior.node_id, class_person.node_id, "subclass_of")
    graph.create_edge(class_mage.node_id, class_person.node_id, "subclass_of")

    # Relationships - events
    graph.create_edge(player1.node_id, event1.node_id, "participated_in")
    graph.create_edge(player2.node_id, event1.node_id, "participated_in")
    graph.create_edge(event1.node_id, loc_village.node_id, "occurred_at")
    graph.create_edge(player2.node_id, event2.node_id, "discovered_during")
    graph.create_edge(artifact.node_id, event2.node_id, "found_in")

    return graph


def build_code_knowledge_graph() -> SemanticGraph:
    """Build test graph for Claude Code MCP scenario.

    This represents codebase knowledge:
    - Files, classes, functions
    - Dependencies and imports
    - Patterns and conventions
    - Recent changes and context
    """
    graph = SemanticGraph()

    # Files
    file_main = graph.create_node("src/main.py", NodeType.INSTANCE,
        properties={"type": "file", "language": "python", "lines": 245})
    file_utils = graph.create_node("src/utils/helpers.py", NodeType.INSTANCE,
        properties={"type": "file", "language": "python", "lines": 120})
    file_models = graph.create_node("src/models/user.py", NodeType.INSTANCE,
        properties={"type": "file", "language": "python", "lines": 89})
    file_test = graph.create_node("tests/test_user.py", NodeType.INSTANCE,
        properties={"type": "file", "language": "python", "lines": 156})

    # Classes
    class_user = graph.create_node("User", NodeType.INSTANCE,
        properties={"type": "class", "file": "src/models/user.py", "line": 15})
    class_admin = graph.create_node("AdminUser", NodeType.INSTANCE,
        properties={"type": "class", "file": "src/models/user.py", "line": 67})
    class_helper = graph.create_node("DataHelper", NodeType.INSTANCE,
        properties={"type": "class", "file": "src/utils/helpers.py", "line": 8})

    # Functions
    func_validate = graph.create_node("validate_email", NodeType.INSTANCE,
        properties={"type": "function", "file": "src/utils/helpers.py", "line": 45, "params": ["email"]})
    func_create = graph.create_node("create_user", NodeType.INSTANCE,
        properties={"type": "function", "file": "src/main.py", "line": 78, "params": ["data", "role"]})
    func_test = graph.create_node("test_user_creation", NodeType.INSTANCE,
        properties={"type": "function", "file": "tests/test_user.py", "line": 22})

    # Patterns (reusable concepts)
    pattern_singleton = graph.create_node("Singleton Pattern", NodeType.CLASS,
        properties={"type": "pattern", "category": "creational"})
    pattern_repository = graph.create_node("Repository Pattern", NodeType.CLASS,
        properties={"type": "pattern", "category": "structural"})

    # Dependencies (packages)
    pkg_pydantic = graph.create_node("pydantic", NodeType.INSTANCE,
        properties={"type": "package", "version": "2.5.0"})
    pkg_sqlalchemy = graph.create_node("sqlalchemy", NodeType.INSTANCE,
        properties={"type": "package", "version": "2.0.23"})

    # Semantic types
    class_file = graph.create_node("file.n.01", NodeType.CLASS, synset_id="file.n.01")
    class_software = graph.create_node("software.n.01", NodeType.CLASS, synset_id="software.n.01")

    # Relationships - containment
    graph.create_edge(file_models.node_id, class_user.node_id, "defines")
    graph.create_edge(file_models.node_id, class_admin.node_id, "defines")
    graph.create_edge(file_utils.node_id, class_helper.node_id, "defines")
    graph.create_edge(file_utils.node_id, func_validate.node_id, "defines")
    graph.create_edge(file_main.node_id, func_create.node_id, "defines")

    # Relationships - inheritance
    graph.create_edge(class_admin.node_id, class_user.node_id, "extends")

    # Relationships - usage
    graph.create_edge(func_create.node_id, class_user.node_id, "uses")
    graph.create_edge(func_create.node_id, func_validate.node_id, "calls")
    graph.create_edge(func_test.node_id, func_create.node_id, "tests")

    # Relationships - imports
    graph.create_edge(file_models.node_id, pkg_pydantic.node_id, "imports")
    graph.create_edge(file_models.node_id, pkg_sqlalchemy.node_id, "imports")
    graph.create_edge(file_main.node_id, file_models.node_id, "imports")
    graph.create_edge(file_main.node_id, file_utils.node_id, "imports")

    # Relationships - patterns
    graph.create_edge(class_helper.node_id, pattern_singleton.node_id, "implements")
    graph.create_edge(class_user.node_id, pattern_repository.node_id, "follows")

    return graph


def build_pharma_rep_graph() -> SemanticGraph:
    """Build test graph for Pharma Field Rep scenario.

    This represents healthcare relationships:
    - Doctors, patients, facilities
    - Drugs and treatments
    - Visit history and notes
    - Prescribing patterns
    """
    graph = SemanticGraph()

    # Doctors
    dr_smith = graph.create_node("Dr. Sarah Smith", NodeType.INSTANCE,
        properties={"type": "doctor", "specialty": "cardiology", "npi": "1234567890"})
    dr_jones = graph.create_node("Dr. Michael Jones", NodeType.INSTANCE,
        properties={"type": "doctor", "specialty": "internal_medicine", "npi": "0987654321"})
    dr_patel = graph.create_node("Dr. Priya Patel", NodeType.INSTANCE,
        properties={"type": "doctor", "specialty": "endocrinology", "npi": "1122334455"})

    # Facilities
    hosp_general = graph.create_node("Metro General Hospital", NodeType.INSTANCE,
        properties={"type": "facility", "beds": 450, "region": "Northeast"})
    clinic_heart = graph.create_node("Heart Health Clinic", NodeType.INSTANCE,
        properties={"type": "facility", "specialty": "cardiology", "region": "Northeast"})
    clinic_diabetes = graph.create_node("Diabetes Care Center", NodeType.INSTANCE,
        properties={"type": "facility", "specialty": "endocrinology", "region": "Northeast"})

    # Drugs
    drug_cardix = graph.create_node("Cardix-XR", NodeType.INSTANCE,
        properties={"type": "drug", "category": "cardiovascular", "dosage": "50mg", "our_product": True})
    drug_glucomax = graph.create_node("GlucoMax", NodeType.INSTANCE,
        properties={"type": "drug", "category": "diabetes", "dosage": "500mg", "our_product": True})
    drug_competitor = graph.create_node("HeartPro", NodeType.INSTANCE,
        properties={"type": "drug", "category": "cardiovascular", "dosage": "25mg", "competitor": True})

    # Visit events
    visit1 = graph.create_node("Visit to Dr. Smith - 2024-12-15", NodeType.EVENT,
        properties={"date": "2024-12-15", "duration_min": 30, "outcome": "positive"})
    visit2 = graph.create_node("Visit to Dr. Patel - 2024-12-18", NodeType.EVENT,
        properties={"date": "2024-12-18", "duration_min": 20, "outcome": "follow_up_needed"})

    # Notes/Insights
    note1 = graph.create_node("Dr. Smith prefers morning visits", NodeType.ATTRIBUTE,
        properties={"type": "note", "importance": "high"})
    note2 = graph.create_node("Heart Health Clinic expanding cardiology dept", NodeType.ATTRIBUTE,
        properties={"type": "note", "importance": "medium"})

    # Semantic classes
    class_doctor = graph.create_node("doctor.n.01", NodeType.CLASS, synset_id="doctor.n.01")
    class_drug = graph.create_node("drug.n.01", NodeType.CLASS, synset_id="drug.n.01")
    class_hospital = graph.create_node("hospital.n.01", NodeType.CLASS, synset_id="hospital.n.01")

    # Relationships - employment
    graph.create_edge(dr_smith.node_id, hosp_general.node_id, "works_at")
    graph.create_edge(dr_smith.node_id, clinic_heart.node_id, "consults_at")
    graph.create_edge(dr_jones.node_id, hosp_general.node_id, "works_at")
    graph.create_edge(dr_patel.node_id, clinic_diabetes.node_id, "works_at")

    # Relationships - prescribing
    graph.create_edge(dr_smith.node_id, drug_cardix.node_id, "prescribes",
        properties={"frequency": "often", "avg_monthly_scripts": 45})
    graph.create_edge(dr_smith.node_id, drug_competitor.node_id, "prescribes",
        properties={"frequency": "sometimes", "avg_monthly_scripts": 12})
    graph.create_edge(dr_patel.node_id, drug_glucomax.node_id, "prescribes",
        properties={"frequency": "often", "avg_monthly_scripts": 80})

    # Relationships - visits
    graph.create_edge(visit1.node_id, dr_smith.node_id, "with_doctor")
    graph.create_edge(visit1.node_id, drug_cardix.node_id, "discussed")
    graph.create_edge(visit2.node_id, dr_patel.node_id, "with_doctor")
    graph.create_edge(visit2.node_id, drug_glucomax.node_id, "discussed")

    # Relationships - notes
    graph.create_edge(note1.node_id, dr_smith.node_id, "about")
    graph.create_edge(note2.node_id, clinic_heart.node_id, "about")

    # Relationships - type hierarchy
    graph.create_edge(dr_smith.node_id, class_doctor.node_id, "instance_of")
    graph.create_edge(dr_jones.node_id, class_doctor.node_id, "instance_of")
    graph.create_edge(drug_cardix.node_id, class_drug.node_id, "instance_of")
    graph.create_edge(hosp_general.node_id, class_hospital.node_id, "instance_of")

    return graph


def build_fitness_coach_graph() -> SemanticGraph:
    """Build test graph for Fitness Coach scenario.

    This represents fitness tracking:
    - User profile and goals
    - Workout history and progress
    - Exercises and routines
    - Measurements and PRs
    """
    graph = SemanticGraph()

    # User
    user = graph.create_node("Doug", NodeType.INSTANCE,
        properties={
            "type": "user", "age": 35, "height_cm": 180, "weight_kg": 82,
            "goal": "muscle_gain", "experience_level": "intermediate"
        })

    # Body measurements (tracked over time)
    measure_dec = graph.create_node("Body Measurements - Dec 2024", NodeType.ATTRIBUTE,
        properties={"date": "2024-12-01", "weight_kg": 82, "body_fat_pct": 18, "chest_cm": 102})
    measure_nov = graph.create_node("Body Measurements - Nov 2024", NodeType.ATTRIBUTE,
        properties={"date": "2024-11-01", "weight_kg": 80, "body_fat_pct": 19, "chest_cm": 100})

    # Exercise definitions
    ex_bench = graph.create_node("Bench Press", NodeType.CLASS,
        properties={"type": "exercise", "muscle_group": "chest", "equipment": "barbell"})
    ex_squat = graph.create_node("Barbell Squat", NodeType.CLASS,
        properties={"type": "exercise", "muscle_group": "legs", "equipment": "barbell"})
    ex_deadlift = graph.create_node("Deadlift", NodeType.CLASS,
        properties={"type": "exercise", "muscle_group": "back", "equipment": "barbell"})
    ex_pullup = graph.create_node("Pull-up", NodeType.CLASS,
        properties={"type": "exercise", "muscle_group": "back", "equipment": "bodyweight"})

    # Workout templates
    workout_push = graph.create_node("Push Day", NodeType.INSTANCE,
        properties={"type": "workout_template", "focus": "chest_shoulders_triceps"})
    workout_pull = graph.create_node("Pull Day", NodeType.INSTANCE,
        properties={"type": "workout_template", "focus": "back_biceps"})
    workout_legs = graph.create_node("Leg Day", NodeType.INSTANCE,
        properties={"type": "workout_template", "focus": "legs"})

    # Workout sessions (events)
    session1 = graph.create_node("Workout 2024-12-30", NodeType.EVENT,
        properties={"date": "2024-12-30", "duration_min": 75, "type": "push"})
    session2 = graph.create_node("Workout 2024-12-28", NodeType.EVENT,
        properties={"date": "2024-12-28", "duration_min": 60, "type": "pull"})

    # Personal records
    pr_bench = graph.create_node("Bench Press PR", NodeType.ATTRIBUTE,
        properties={"type": "pr", "weight_kg": 100, "reps": 1, "date": "2024-12-15"})
    pr_squat = graph.create_node("Squat PR", NodeType.ATTRIBUTE,
        properties={"type": "pr", "weight_kg": 140, "reps": 1, "date": "2024-12-20"})

    # Goals
    goal_bench = graph.create_node("Bench 110kg by March", NodeType.ATTRIBUTE,
        properties={"type": "goal", "target_weight": 110, "deadline": "2025-03-01", "status": "in_progress"})

    # Semantic classes
    class_person = graph.create_node("person.n.01", NodeType.CLASS, synset_id="person.n.01")
    class_exercise = graph.create_node("exercise.n.01", NodeType.CLASS, synset_id="exercise.n.01")

    # Relationships - user to measurements
    graph.create_edge(user.node_id, measure_dec.node_id, "has_measurement")
    graph.create_edge(user.node_id, measure_nov.node_id, "has_measurement")

    # Relationships - workout template composition
    graph.create_edge(workout_push.node_id, ex_bench.node_id, "includes")
    graph.create_edge(workout_pull.node_id, ex_pullup.node_id, "includes")
    graph.create_edge(workout_pull.node_id, ex_deadlift.node_id, "includes")
    graph.create_edge(workout_legs.node_id, ex_squat.node_id, "includes")

    # Relationships - session to template
    graph.create_edge(session1.node_id, workout_push.node_id, "followed")
    graph.create_edge(session2.node_id, workout_pull.node_id, "followed")

    # Relationships - user to sessions
    graph.create_edge(user.node_id, session1.node_id, "completed")
    graph.create_edge(user.node_id, session2.node_id, "completed")

    # Relationships - PRs
    graph.create_edge(user.node_id, pr_bench.node_id, "achieved")
    graph.create_edge(pr_bench.node_id, ex_bench.node_id, "for_exercise")
    graph.create_edge(user.node_id, pr_squat.node_id, "achieved")
    graph.create_edge(pr_squat.node_id, ex_squat.node_id, "for_exercise")

    # Relationships - goals
    graph.create_edge(user.node_id, goal_bench.node_id, "has_goal")
    graph.create_edge(goal_bench.node_id, ex_bench.node_id, "for_exercise")

    # Relationships - type
    graph.create_edge(user.node_id, class_person.node_id, "instance_of")
    graph.create_edge(ex_bench.node_id, class_exercise.node_id, "subclass_of")

    return graph


def build_educational_tutor_graph() -> SemanticGraph:
    """Build test graph for Educational Tutor scenario.

    This represents student knowledge state:
    - Student profile and learning style
    - Subjects and topics mastery
    - Assessments and progress
    - Prerequisites and learning paths
    """
    graph = SemanticGraph()

    # Student
    student = graph.create_node("Emily Chen", NodeType.INSTANCE,
        properties={
            "type": "student", "grade": 10, "learning_style": "visual",
            "goal": "AP_Calculus", "weekly_hours": 6
        })

    # Subjects
    subj_math = graph.create_node("Mathematics", NodeType.CLASS,
        properties={"type": "subject", "level": "high_school"})
    subj_calc = graph.create_node("Calculus", NodeType.CLASS,
        properties={"type": "subject", "level": "advanced", "ap_course": True})
    subj_algebra = graph.create_node("Algebra II", NodeType.CLASS,
        properties={"type": "subject", "level": "intermediate"})
    subj_trig = graph.create_node("Trigonometry", NodeType.CLASS,
        properties={"type": "subject", "level": "intermediate"})

    # Topics
    topic_derivatives = graph.create_node("Derivatives", NodeType.INSTANCE,
        properties={"type": "topic", "difficulty": "medium", "subject": "calculus"})
    topic_limits = graph.create_node("Limits", NodeType.INSTANCE,
        properties={"type": "topic", "difficulty": "easy", "subject": "calculus"})
    topic_integrals = graph.create_node("Integrals", NodeType.INSTANCE,
        properties={"type": "topic", "difficulty": "hard", "subject": "calculus"})
    topic_trig_func = graph.create_node("Trigonometric Functions", NodeType.INSTANCE,
        properties={"type": "topic", "difficulty": "medium", "subject": "trigonometry"})
    topic_quadratic = graph.create_node("Quadratic Equations", NodeType.INSTANCE,
        properties={"type": "topic", "difficulty": "easy", "subject": "algebra"})

    # Mastery levels (student's knowledge state)
    mastery_limits = graph.create_node("Emily's Limits Mastery", NodeType.ATTRIBUTE,
        properties={"type": "mastery", "level": 0.85, "confidence": 0.9, "last_reviewed": "2024-12-28"})
    mastery_deriv = graph.create_node("Emily's Derivatives Mastery", NodeType.ATTRIBUTE,
        properties={"type": "mastery", "level": 0.65, "confidence": 0.7, "last_reviewed": "2024-12-30"})
    mastery_trig = graph.create_node("Emily's Trig Functions Mastery", NodeType.ATTRIBUTE,
        properties={"type": "mastery", "level": 0.90, "confidence": 0.95, "last_reviewed": "2024-12-15"})

    # Assessments
    quiz1 = graph.create_node("Limits Quiz - Dec 28", NodeType.EVENT,
        properties={"type": "assessment", "score": 92, "max_score": 100, "date": "2024-12-28"})
    quiz2 = graph.create_node("Derivatives Quiz - Dec 30", NodeType.EVENT,
        properties={"type": "assessment", "score": 78, "max_score": 100, "date": "2024-12-30"})

    # Learning resources
    video_deriv = graph.create_node("Derivatives Introduction Video", NodeType.INSTANCE,
        properties={"type": "resource", "format": "video", "duration_min": 15, "difficulty": "medium"})
    practice_deriv = graph.create_node("Derivatives Practice Set", NodeType.INSTANCE,
        properties={"type": "resource", "format": "exercises", "count": 20, "difficulty": "medium"})

    # Semantic classes
    class_student = graph.create_node("student.n.01", NodeType.CLASS, synset_id="student.n.01")
    class_subject = graph.create_node("subject.n.01", NodeType.CLASS, synset_id="subject.n.01")

    # Relationships - subject hierarchy
    graph.create_edge(subj_calc.node_id, subj_math.node_id, "part_of")
    graph.create_edge(subj_algebra.node_id, subj_math.node_id, "part_of")
    graph.create_edge(subj_trig.node_id, subj_math.node_id, "part_of")

    # Relationships - prerequisites
    graph.create_edge(topic_derivatives.node_id, topic_limits.node_id, "requires")
    graph.create_edge(topic_integrals.node_id, topic_derivatives.node_id, "requires")
    graph.create_edge(topic_derivatives.node_id, topic_quadratic.node_id, "requires")

    # Relationships - topic to subject
    graph.create_edge(topic_derivatives.node_id, subj_calc.node_id, "belongs_to")
    graph.create_edge(topic_limits.node_id, subj_calc.node_id, "belongs_to")
    graph.create_edge(topic_integrals.node_id, subj_calc.node_id, "belongs_to")
    graph.create_edge(topic_trig_func.node_id, subj_trig.node_id, "belongs_to")

    # Relationships - student mastery
    graph.create_edge(student.node_id, mastery_limits.node_id, "has_mastery")
    graph.create_edge(mastery_limits.node_id, topic_limits.node_id, "for_topic")
    graph.create_edge(student.node_id, mastery_deriv.node_id, "has_mastery")
    graph.create_edge(mastery_deriv.node_id, topic_derivatives.node_id, "for_topic")
    graph.create_edge(student.node_id, mastery_trig.node_id, "has_mastery")
    graph.create_edge(mastery_trig.node_id, topic_trig_func.node_id, "for_topic")

    # Relationships - assessments
    graph.create_edge(student.node_id, quiz1.node_id, "took")
    graph.create_edge(quiz1.node_id, topic_limits.node_id, "assessed")
    graph.create_edge(student.node_id, quiz2.node_id, "took")
    graph.create_edge(quiz2.node_id, topic_derivatives.node_id, "assessed")

    # Relationships - resources to topics
    graph.create_edge(video_deriv.node_id, topic_derivatives.node_id, "teaches")
    graph.create_edge(practice_deriv.node_id, topic_derivatives.node_id, "practices")

    # Relationships - type
    graph.create_edge(student.node_id, class_student.node_id, "instance_of")
    graph.create_edge(subj_math.node_id, class_subject.node_id, "instance_of")

    return graph


# =============================================================================
# Basic Persistence Tests
# =============================================================================


class TestBasicPersistence:
    """Tests for basic save/load operations."""

    def test_save_empty_graph(self, store, clean_instance):
        """Test saving an empty graph."""
        graph = SemanticGraph()
        result = store.save(graph, clean_instance)

        assert result["nodes"] == 0
        assert result["edges"] == 0

    def test_save_and_load_simple_graph(self, store, clean_instance):
        """Test saving and loading a simple graph."""
        graph = SemanticGraph()
        doug = graph.create_node("Doug", NodeType.INSTANCE)
        whiskers = graph.create_node("Whiskers", NodeType.INSTANCE)
        graph.create_edge(doug.node_id, whiskers.node_id, "owns")

        # Save
        result = store.save(graph, clean_instance)
        assert result["nodes"] == 2
        assert result["edges"] == 1

        # Load
        loaded = store.load(clean_instance)
        assert loaded.node_count == 2
        assert loaded.edge_count == 1

        # Verify nodes
        loaded_doug = loaded.find_node("Doug")
        assert loaded_doug is not None
        assert loaded_doug.node_type == NodeType.INSTANCE

        loaded_whiskers = loaded.find_node("Whiskers")
        assert loaded_whiskers is not None

    def test_save_with_properties(self, store, clean_instance):
        """Test that custom properties are preserved."""
        graph = SemanticGraph()
        node = graph.create_node("Test", NodeType.INSTANCE,
            properties={"color": "blue", "count": 42, "active": True})

        store.save(graph, clean_instance)
        loaded = store.load(clean_instance)

        loaded_node = loaded.find_node("Test")
        assert loaded_node.properties["color"] == "blue"
        assert loaded_node.properties["count"] == 42
        assert loaded_node.properties["active"] is True

    def test_save_with_synset(self, store, clean_instance):
        """Test that synset IDs are preserved."""
        graph = SemanticGraph()
        node = graph.create_node("cat.n.01", NodeType.CLASS, synset_id="cat.n.01")

        store.save(graph, clean_instance)
        loaded = store.load(clean_instance)

        loaded_node = loaded.find_node("cat.n.01")
        assert loaded_node.synset_id == "cat.n.01"
        assert loaded_node.node_type == NodeType.CLASS

    def test_clear_existing(self, store, clean_instance):
        """Test that clear_existing removes old data."""
        # Save first graph
        graph1 = SemanticGraph()
        graph1.create_node("Node1", NodeType.INSTANCE)
        store.save(graph1, clean_instance)

        # Save second graph with clear
        graph2 = SemanticGraph()
        graph2.create_node("Node2", NodeType.INSTANCE)
        store.save(graph2, clean_instance, clear_existing=True)

        # Load and verify only second graph exists
        loaded = store.load(clean_instance)
        assert loaded.node_count == 1
        assert loaded.find_node("Node2") is not None
        assert loaded.find_node("Node1") is None


class TestEdgePersistence:
    """Tests for edge persistence including temporal properties."""

    def test_edge_properties(self, store, clean_instance):
        """Test that edge properties are preserved."""
        graph = SemanticGraph()
        n1 = graph.create_node("Source", NodeType.INSTANCE)
        n2 = graph.create_node("Target", NodeType.INSTANCE)
        graph.create_edge(n1.node_id, n2.node_id, "relates_to",
            properties={"strength": "strong", "count": 5})

        store.save(graph, clean_instance)
        loaded = store.load(clean_instance)

        edges = loaded.get_outgoing_edges(n1.node_id)
        assert len(edges) == 1
        assert edges[0].properties["strength"] == "strong"
        assert edges[0].properties["count"] == 5

    def test_temporal_edges(self, store, clean_instance):
        """Test that bi-temporal edge properties are preserved."""
        graph = SemanticGraph()
        n1 = graph.create_node("Doug", NodeType.INSTANCE)
        n2 = graph.create_node("Cat1", NodeType.INSTANCE)
        n3 = graph.create_node("Cat2", NodeType.INSTANCE)

        # Create an old edge that's no longer valid
        old_edge = graph.create_edge(n1.node_id, n2.node_id, "owned",
            valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc))
        old_edge.invalidate(datetime(2023, 1, 1, tzinfo=timezone.utc))

        # Create a current edge
        graph.create_edge(n1.node_id, n3.node_id, "owns")

        store.save(graph, clean_instance)

        # Load current only
        loaded_current = store.load(clean_instance, current_only=True)
        current_edges = list(loaded_current.iter_edges())
        assert len(current_edges) == 1
        assert loaded_current.find_node("Cat2") is not None

        # Load all including historical
        loaded_all = store.load(clean_instance, current_only=False)
        all_edges = list(loaded_all.iter_edges(current_only=False))
        assert len(all_edges) == 2


class TestStatistics:
    """Tests for graph statistics."""

    def test_get_statistics(self, store, clean_instance):
        """Test getting graph statistics."""
        graph = SemanticGraph()
        graph.create_node("Person1", NodeType.INSTANCE)
        graph.create_node("Person2", NodeType.INSTANCE)
        graph.create_node("class.n.01", NodeType.CLASS)
        p1 = graph.find_node("Person1")
        p2 = graph.find_node("Person2")
        graph.create_edge(p1.node_id, p2.node_id, "knows")

        store.save(graph, clean_instance)
        stats = store.get_statistics(clean_instance)

        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 1
        assert stats["nodes_by_type"]["instance"] == 2
        assert stats["nodes_by_type"]["class"] == 1
        assert stats["edges_by_type"]["knows"] == 1


# =============================================================================
# Reference Case Tests
# =============================================================================


class TestRPGGameMaster:
    """Tests for RPG Game Master reference case."""

    def test_save_and_load(self, store):
        """Test saving and loading RPG graph."""
        instance_id = "test-rpg"
        graph = build_rpg_game_master_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            loaded = store.load(instance_id)

            # Verify key nodes exist
            assert loaded.find_node("Aldric the Bold") is not None
            assert loaded.find_node("Seraphina") is not None
            assert loaded.find_node("Millbrook Village") is not None
            assert loaded.find_node("Flamekeeper") is not None

            # Verify relationships
            aldric = loaded.find_node("Aldric the Bold")
            owns_edges = loaded.get_outgoing_edges(aldric.node_id, "owns")
            assert len(owns_edges) == 1

            # Verify properties preserved
            assert aldric.properties["class"] == "warrior"
            assert aldric.properties["level"] == 12

        finally:
            # Cleanup
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )

    def test_statistics(self, store):
        """Test graph statistics for RPG scenario."""
        instance_id = "test-rpg-stats"
        graph = build_rpg_game_master_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            stats = store.get_statistics(instance_id)

            # Should have multiple node types
            assert stats["total_nodes"] > 15
            assert stats["total_edges"] > 15
            assert "instance" in stats["nodes_by_type"]
            assert "class" in stats["nodes_by_type"]
            assert "event" in stats["nodes_by_type"]

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )


class TestCodeKnowledge:
    """Tests for Claude Code MCP reference case."""

    def test_save_and_load(self, store):
        """Test saving and loading code knowledge graph."""
        instance_id = "test-code"
        graph = build_code_knowledge_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            loaded = store.load(instance_id)

            # Verify files
            assert loaded.find_node("src/main.py") is not None
            assert loaded.find_node("src/models/user.py") is not None

            # Verify classes and functions
            assert loaded.find_node("User") is not None
            assert loaded.find_node("validate_email") is not None

            # Verify inheritance
            admin = loaded.find_node("AdminUser")
            extends_edges = loaded.get_outgoing_edges(admin.node_id, "extends")
            assert len(extends_edges) == 1

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )


class TestPharmaRep:
    """Tests for Pharma Field Rep reference case."""

    def test_save_and_load(self, store):
        """Test saving and loading pharma rep graph."""
        instance_id = "test-pharma"
        graph = build_pharma_rep_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            loaded = store.load(instance_id)

            # Verify doctors
            dr_smith = loaded.find_node("Dr. Sarah Smith")
            assert dr_smith is not None
            assert dr_smith.properties["specialty"] == "cardiology"

            # Verify prescribing relationships
            prescribes_edges = loaded.get_outgoing_edges(dr_smith.node_id, "prescribes")
            assert len(prescribes_edges) >= 1

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )


class TestFitnessCoach:
    """Tests for Fitness Coach reference case."""

    def test_save_and_load(self, store):
        """Test saving and loading fitness coach graph."""
        instance_id = "test-fitness"
        graph = build_fitness_coach_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            loaded = store.load(instance_id)

            # Verify user
            user = loaded.find_node("Doug")
            assert user is not None
            assert user.properties["goal"] == "muscle_gain"

            # Verify workout relationships
            completed_edges = loaded.get_outgoing_edges(user.node_id, "completed")
            assert len(completed_edges) >= 1

            # Verify PRs
            achieved_edges = loaded.get_outgoing_edges(user.node_id, "achieved")
            assert len(achieved_edges) >= 1

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )


class TestEducationalTutor:
    """Tests for Educational Tutor reference case."""

    def test_save_and_load(self, store):
        """Test saving and loading educational tutor graph."""
        instance_id = "test-education"
        graph = build_educational_tutor_graph()

        try:
            store.save(graph, instance_id, clear_existing=True)
            loaded = store.load(instance_id)

            # Verify student
            student = loaded.find_node("Emily Chen")
            assert student is not None
            assert student.properties["learning_style"] == "visual"

            # Verify mastery relationships
            mastery_edges = loaded.get_outgoing_edges(student.node_id, "has_mastery")
            assert len(mastery_edges) >= 1

            # Verify prerequisite chain
            derivatives = loaded.find_node("Derivatives")
            requires_edges = loaded.get_outgoing_edges(derivatives.node_id, "requires")
            assert len(requires_edges) >= 1

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )


class TestInstanceIsolation:
    """Tests for data instance isolation."""

    def test_instances_are_isolated(self, store):
        """Test that different instances don't interfere."""
        instance1 = "test-isolation-1"
        instance2 = "test-isolation-2"

        try:
            # Create different graphs for different instances
            graph1 = SemanticGraph()
            graph1.create_node("Node-Instance1", NodeType.INSTANCE)

            graph2 = SemanticGraph()
            graph2.create_node("Node-Instance2", NodeType.INSTANCE)

            store.save(graph1, instance1)
            store.save(graph2, instance2)

            # Load and verify isolation
            loaded1 = store.load(instance1)
            loaded2 = store.load(instance2)

            assert loaded1.find_node("Node-Instance1") is not None
            assert loaded1.find_node("Node-Instance2") is None

            assert loaded2.find_node("Node-Instance2") is not None
            assert loaded2.find_node("Node-Instance1") is None

        finally:
            with store.driver.session(database=store.database) as session:
                session.run(
                    "MATCH (n:Entity) WHERE n.instance_id IN $ids DETACH DELETE n",
                    ids=[instance1, instance2],
                )
