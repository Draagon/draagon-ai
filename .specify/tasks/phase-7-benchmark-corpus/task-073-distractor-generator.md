# TASK-073: Synthetic Content & Distractor Generator

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Needed for Conversational + Synthetic Distractors)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-070 (Document data models)

---

## Description

Implement LLM-based generator for synthetic content:
- Generate distractor documents across ALL categories
- Generate conversational content (chat, email, support tickets)
- Control similarity level (very different, somewhat similar, very similar)
- Ensure diversity in generated content

**Location:** `src/draagon_ai/testing/benchmarks/downloaders/distractor_generator.py`

---

## Acceptance Criteria

### Distractor Generation
- [ ] `DistractorGenerator` class with LLM provider
- [ ] `generate(categories, count, similarity_distribution)` returns documents
- [ ] Similarity levels: very_different (50%), somewhat_similar (30%), very_similar (20%)
- [ ] Generates content matching each category's style
- [ ] All generated docs marked `is_distractor=True`

### Conversational Content Generation
- [ ] Chat transcript templates (customer service, technical support)
- [ ] Email thread templates (professional, casual)
- [ ] Support ticket templates (bug report, feature request)
- [ ] Realistic conversation flow with multiple turns

### Category-Specific Styles
- [ ] TECHNICAL: API docs, code examples, error messages
- [ ] LEGAL: Contract clauses, terms, formal language
- [ ] NARRATIVE: Story excerpts, character descriptions
- [ ] ACADEMIC: Abstract style, citations, formal vocabulary
- [ ] CONVERSATIONAL: Informal, incomplete sentences, emoji
- [ ] NEWS_BLOG: Headlines, bylines, journalistic style

### Content Quality
- [ ] 500-1000 words per document
- [ ] Unique content (no repetition)
- [ ] Category-appropriate vocabulary
- [ ] "Very similar" distractors use AI/agent terminology even for non-AI topics

---

## Technical Notes

### Prompt Templates by Similarity Level

```python
def _build_prompt(self, category: DocumentCategory, similarity: str) -> str:
    if similarity == "very_different":
        return f"""Write {category.value} content about a random topic.
Length: 500-1000 words. Style: Typical {category.value} writing."""

    elif similarity == "somewhat_similar":
        return f"""Write {category.value} content using these terms:
memory, architecture, system, design, optimization, processing.
Topic: Something unrelated to AI agents.
Length: 500-1000 words."""

    else:  # very_similar (hard negative)
        return f"""Write {category.value} content heavily using AI terminology:
cognitive architecture, memory layers, belief reconciliation,
decision engine, agent orchestration, semantic understanding.
Topic: NOT about AI - use these terms metaphorically for {category.value}.
Length: 500-1000 words."""
```

### Conversational Templates

```python
CHAT_TEMPLATE = """Generate a realistic customer service chat transcript.
Scenario: {scenario}
Turns: 8-12
Include: greetings, issue description, troubleshooting, resolution
Tone: Professional but friendly"""

EMAIL_TEMPLATE = """Generate a professional email thread.
Topic: {topic}
Emails: 3-5
Include: original message, replies, forward
Style: {style}"""  # professional or casual
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_distractor_generation(mock_llm):
    """Generate distractors across categories."""
    generator = DistractorGenerator(llm_provider=mock_llm)
    docs = await generator.generate(
        categories=[DocumentCategory.TECHNICAL, DocumentCategory.LEGAL],
        count=10,
    )

    assert len(docs) == 10
    assert all(doc.is_distractor for doc in docs)
    assert all(doc.category in [DocumentCategory.TECHNICAL, DocumentCategory.LEGAL]
               for doc in docs)

@pytest.mark.asyncio
async def test_similarity_distribution(mock_llm):
    """Similarity levels follow distribution."""
    generator = DistractorGenerator(llm_provider=mock_llm)
    docs = await generator.generate(
        categories=[DocumentCategory.TECHNICAL],
        count=100,
        similarity_distribution={"very_different": 0.5, "somewhat_similar": 0.3, "very_similar": 0.2},
    )

    # Check distribution (with tolerance)
    very_diff = sum(1 for d in docs if d.metadata["similarity_level"] == "very_different")
    assert 40 <= very_diff <= 60  # 50% ± 10%
```

### Integration Test (real LLM)
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_llm_generation():
    """Generate with real LLM provider."""
    from draagon_ai.llm import GroqProvider

    llm = GroqProvider(api_key=os.environ["GROQ_API_KEY"])
    generator = DistractorGenerator(llm_provider=llm)

    docs = await generator.generate(
        categories=[DocumentCategory.CONVERSATIONAL],
        count=2,
    )

    assert len(docs) == 2
    assert all(len(doc.content) > 200 for doc in docs)
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/downloaders/distractor_generator.py`
- Add tests to `tests/benchmarks/test_corpus_builder.py`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Distractors generated across all 8 categories
- [ ] Similarity distribution validated
- [ ] Conversational content is realistic
- [ ] Integration test with real LLM passes
- [ ] Content length 500-1000 words verified
