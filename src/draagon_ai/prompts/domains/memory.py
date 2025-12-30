"""Memory domain prompts.

Episode summaries, graph queries, and meta-knowledge extraction.
"""

EPISODE_SUMMARY_PROMPT = """Summarize this conversation into a concise episode memory.

CONVERSATION HISTORY:
{conversation_history}

**CRITICAL FORMAT RULES - STRUCTURED FACTUAL RECORDS:**
Write the summary as a FACTUAL RECORD of what happened, NOT as self-reflection.

GOOD (factual record style):
- "Checked time: 3:45 PM. Scheduled dentist appointment for Tuesday 2pm."
- "Discussed Sarah's workplace. Sarah works at Acme Corp in Seattle."
- "User set timer for 5 minutes. Timer completed successfully."

BAD (self-referential - NEVER USE):
- "I learned that Sarah works at Acme Corp" ✗
- "I should remember the appointment is Tuesday" ✗
- "The user told me about their sister" ✗
- "I need to note that..." ✗

**NEVER include phrases like:**
- "I learned...", "I should remember...", "I need to..."
- "The user told me...", "The user said..."
- "I discovered...", "I found out..."
- "Remember that...", "Note that..."

**ALWAYS write as:**
- Third-person factual statements
- Direct records of what happened and what information was shared
- Actions taken in past tense without self-reference

Create a brief summary that captures:
1. Main topics discussed (as factual records)
2. Key facts or decisions made (as direct statements)
3. Any actions taken (commands run, events created, etc.)
4. User preferences or context (stated directly, not "I learned X")

Keep it to 2-3 sentences maximum. Focus on information that would be useful to remember later.
Also extract a list of topics (short keywords) for categorization.

Output XML:
<episode>
    <summary>Brief 2-3 sentence summary of the conversation</summary>
    <topics>
        <topic>topic1</topic>
        <topic>topic2</topic>
        <topic>topic3</topic>
    </topics>
</episode>

Examples:
- User asked about weather, then scheduled a dentist appointment
  -> <episode><summary>Checked weather conditions (72F, sunny). Dentist appointment scheduled for tomorrow at 2pm.</summary><topics><topic>weather</topic><topic>calendar</topic><topic>appointment</topic></topics></episode>
- User shared family information
  -> <episode><summary>Sarah is user's sister, works at Acme Corp in Seattle. Maya is user's daughter, age 8.</summary><topics><topic>family</topic><topic>Sarah</topic><topic>Maya</topic></topics></episode>"""


GRAPH_QUERY_GENERATION_PROMPT = """You are a knowledge graph query generator for the assistant's memory system.

Your task: Generate a JSON query to explore the knowledge graph and answer the user's question.

AVAILABLE RELATIONSHIP VOCABULARY:
{relationship_vocabulary}

CURRENT GRAPH CONTEXT (immediate connections to "user" and "assistant"):
{user_graph_context}

QUERY LANGUAGE:
You can generate JSON queries using these operations:

1. **TRAVERSE** - Follow a specific path through relationships
   Example: Find grandparents of Maya
   {{
     "operation": "traverse",
     "start_entity": "maya",
     "path": [
       {{"relationship": "PARENT_OF", "direction": "incoming"}},
       {{"relationship": "PARENT_OF", "direction": "incoming"}}
     ]
   }}

2. **CHECK_RELATIONSHIP** - Yes/no relationship verification
   Example: Is user Maya's parent?
   {{
     "operation": "check_relationship",
     "subject": "user",
     "predicate": "PARENT_OF",
     "object": "maya"
   }}

3. **EXPLORE** - Discover all connections from an entity
   Example: Find all of user's family members within 2 hops
   {{
     "operation": "explore",
     "entity": "user",
     "max_hops": 2,
     "relationship_types": ["PARENT_OF", "CHILD_OF", "SIBLING_OF", "SPOUSE_OF"]
   }}

4. **FIND_PATH** - Find shortest path between two entities
   Example: How is Maya connected to Robert?
   {{
     "operation": "find_path",
     "start": "maya",
     "end": "robert",
     "max_hops": 5
   }}

DIRECTIONS:
- "outgoing": entity → related (subject → object), e.g., user PARENT_OF maya
- "incoming": related → entity (object → subject), e.g., maya's parents (incoming PARENT_OF to maya)
- "both": either direction

MULTI-HOP REASONING:
- Grandparents = 2 hops through PARENT_OF (incoming)
- Great-grandparents = 3 hops through PARENT_OF (incoming)
- Siblings of parents (aunts/uncles) = PARENT_OF (incoming) → SIBLING_OF
- Use TRAVERSE for specific paths, EXPLORE for discovery

USER QUERY: {user_query}

PREVIOUS QUERIES IN THIS REASONING SESSION:
{reasoning_history}

Generate ONE graph query to answer this question or gather more information.
Return ONLY the JSON query, no explanation or markdown formatting."""


GRAPH_REASONING_ASSESSMENT_PROMPT = """You are assessing whether you have enough information to answer the user's question.

ORIGINAL USER QUERY: {user_query}

GRAPH QUERIES EXECUTED SO FAR:
{reasoning_history}

Do you have enough information to answer the user's question?

OPTIONS:
1. **answer** - You have enough information, provide the answer
2. **query_again** - You need more information, generate another graph query
3. **insufficient_data** - The knowledge graph doesn't contain the information needed

Return XML:
<result>
  <action>answer|query_again|insufficient_data</action>
  <reasoning>brief explanation</reasoning>
  <answer>natural language answer (only if action=answer)</answer>
</result>"""


PROPOSE_RELATIONSHIP_TYPE_PROMPT = """You are helping to expand the assistant's relationship vocabulary.

During a conversation, a relationship was detected that isn't in the current vocabulary.

DETECTED RELATIONSHIP: {detected_relationship}
CONTEXT: {context}
EXISTING VOCABULARY CATEGORIES: {categories}

Should this be added as a new relationship type?

If YES, provide XML:
<relationship>
  <should_add>true</should_add>
  <name>RELATIONSHIP_NAME</name>
  <inverse>INVERSE_NAME (or empty if none)</inverse>
  <category>family|professional|social|possession|location|knowledge|meta</category>
  <symmetric>false (true if relationship is bidirectional)</symmetric>
  <description>brief description</description>
</relationship>

If NO (relationship is too vague, already covered, or not useful):
<relationship>
  <should_add>false</should_add>
  <reasoning>why not</reasoning>
</relationship>"""


META_KNOWLEDGE_EXTRACTION_PROMPT = """You are extracting meta-knowledge about a voice assistant.

**Your task:** Analyze the content and extract relationships about the assistant itself (not about the user).

**Content to analyze:**
{content}

**Entities detected:**
{entities}

**Meta-relationship types available:**
- USES_SERVICE: The assistant uses a service/tool (e.g., Qdrant, Groq, Home Assistant)
- HAS_CAPABILITY: The assistant can do something (e.g., "search the web", "control devices")
- HAS_CODE_CHANGE: Code file that was modified (e.g., orchestrator.py, memory.py)
- RUNS_ON: Infrastructure the assistant runs on (e.g., LXC container, IP address)
- CREATED_BY: Who/what created the assistant
- KNOWS_ABOUT: What domains the assistant has knowledge in

**Rules:**
1. Only extract if content is clearly ABOUT THE ASSISTANT (not about the user)
2. Be specific - "search the web" not just "search"
3. For code changes, extract file names
4. For services, use the service name (not URLs)
5. Confidence 0.9 for explicit statements, 0.7 for implied

**Output XML relationships (empty relationships element if none found):**
<relationships>
    <relationship>
        <subject>assistant</subject>
        <predicate>USES_SERVICE</predicate>
        <object>qdrant</object>
        <confidence>0.9</confidence>
    </relationship>
</relationships>

Output ONLY valid XML, no explanation."""


MEMORY_PROMPTS = {
    "EPISODE_SUMMARY_PROMPT": EPISODE_SUMMARY_PROMPT,
    "GRAPH_QUERY_GENERATION_PROMPT": GRAPH_QUERY_GENERATION_PROMPT,
    "GRAPH_REASONING_ASSESSMENT_PROMPT": GRAPH_REASONING_ASSESSMENT_PROMPT,
    "PROPOSE_RELATIONSHIP_TYPE_PROMPT": PROPOSE_RELATIONSHIP_TYPE_PROMPT,
    "META_KNOWLEDGE_EXTRACTION_PROMPT": META_KNOWLEDGE_EXTRACTION_PROMPT,
}
