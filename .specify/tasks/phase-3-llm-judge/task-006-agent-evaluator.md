# TASK-006: Implement Agent Evaluator (LLM-as-Judge)

**Phase**: 3 (LLM-as-Judge)
**Priority**: P0 (Critical for robust testing)
**Effort**: 1.5 days
**Status**: Pending
**Dependencies**: None (can run in parallel with Phase 1/2)

## Description

Implement LLM-based semantic evaluation of agent responses. This replaces brittle string matching with robust semantic validation using XML prompts and retry logic.

**Core Principle:** Test outcomes, not processes.

## Acceptance Criteria

- [ ] `EvaluationResult` dataclass with correct/reasoning/confidence
- [ ] `AgentEvaluator` class with LLM provider protocol
- [ ] `evaluate_correctness()` - semantic outcome evaluation
- [ ] `evaluate_coherence()` - response quality scoring
- [ ] `evaluate_helpfulness()` - UX quality scoring
- [ ] Retry logic with exponential backoff (max 3 attempts)
- [ ] XML-based evaluation prompts (NOT JSON)
- [ ] XML parsing with robust error handling
- [ ] `evaluator` fixture with real LLM
- [ ] Unit test: XML parsing
- [ ] Integration test: real LLM evaluation

## Technical Notes

**AgentEvaluator with Retry:**

```python
class AgentEvaluator:
    def __init__(self, llm: LLMProvider, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    async def _call_llm_with_retry(self, messages: list[dict]) -> str:
        """Call LLM with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                return await self.llm.chat(messages)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    await asyncio.sleep(wait_time)
        raise RuntimeError(f"LLM evaluation failed after {self.max_retries} attempts")

    async def evaluate_correctness(
        self, query, expected_outcome, actual_response
    ) -> EvaluationResult:
        """Evaluate if response achieves expected outcome."""
        prompt = f"""<evaluation>
  <query>{query}</query>
  <expected_outcome>{expected_outcome}</expected_outcome>
  <actual_response>{actual_response}</actual_response>
</evaluation>

Respond with:
<result>
  <correct>true/false</correct>
  <reasoning>Brief explanation</reasoning>
  <confidence>0.0-1.0</confidence>
</result>"""

        response = await self._call_llm_with_retry([{"role": "user", "content": prompt}])
        return self._parse_evaluation(response)
```

**Why Retry Logic?**
- LLM APIs can be flaky (rate limits, network issues)
- Tests should be resilient to transient failures
- Exponential backoff prevents hammering the API

**Why XML, Not JSON?**
- Fewer escaping issues (quotes, backslashes, newlines)
- Better streaming support (incremental parsing)
- More robust to malformed output
- Per CLAUDE.md constitution

## Testing Requirements

### Unit Tests (`tests/framework/test_evaluation.py`)

1. **XML Parsing**
   ```python
   def test_parse_evaluation_correct():
       xml = "<result><correct>true</correct><reasoning>Good</reasoning><confidence>0.9</confidence></result>"
       result = evaluator._parse_evaluation(xml)
       assert result.correct is True
       assert result.confidence == 0.9

   def test_parse_evaluation_malformed():
       xml = "<result><correct>true</result>"  # Malformed
       result = evaluator._parse_evaluation(xml)
       # Should handle gracefully
   ```

2. **Retry Logic (Mock)**
   ```python
   async def test_retry_on_failure():
       mock_llm = Mock()
       mock_llm.chat.side_effect = [
           Exception("Network error"),
           Exception("Rate limit"),
           "<result>...</result>"  # Success on 3rd try
       ]
       evaluator = AgentEvaluator(mock_llm)
       result = await evaluator.evaluate_correctness(...)
       assert mock_llm.chat.call_count == 3

   async def test_retry_exhausted():
       mock_llm = Mock()
       mock_llm.chat.side_effect = Exception("Always fails")
       evaluator = AgentEvaluator(mock_llm)
       with pytest.raises(RuntimeError, match="failed after 3 attempts"):
           await evaluator.evaluate_correctness(...)
   ```

### Integration Test (`tests/integration/test_evaluator.py`)

```python
@pytest.mark.integration
async def test_evaluate_correctness_real_llm(llm_provider):
    """Test with REAL LLM provider."""
    evaluator = AgentEvaluator(llm_provider)

    result = await evaluator.evaluate_correctness(
        query="What are my cats' names?",
        expected_outcome="Agent lists: Whiskers, Mittens, Shadow",
        actual_response="Your cats are Whiskers, Mittens, and Shadow!"
    )

    assert result.correct is True
    assert result.confidence > 0.8
    assert "whiskers" in result.reasoning.lower()
```

## Files to Create

- `src/draagon_ai/testing/evaluation.py` - NEW
  - `EvaluationResult` dataclass
  - `LLMProvider` protocol
  - `AgentEvaluator` class

- `src/draagon_ai/testing/fixtures.py` - EXTEND
  - `evaluator` fixture (uses real LLM)

- `tests/framework/test_evaluation.py` - NEW
  - Test XML parsing
  - Test retry logic (mocked)

- `tests/integration/test_evaluator.py` - NEW
  - Test with real LLM

## Implementation Sequence

1. Define `EvaluationResult` dataclass
2. Define `LLMProvider` protocol
3. Implement `AgentEvaluator.__init__()`
4. Implement `_call_llm_with_retry()` with exponential backoff
5. Implement `evaluate_correctness()` with XML prompt
6. Implement `_parse_evaluation()` with regex extraction
7. Implement `evaluate_coherence()` (similar pattern)
8. Implement `evaluate_helpfulness()` (similar pattern)
9. Add `evaluator` fixture
10. Write unit tests (XML parsing, retry logic)
11. Write integration test with real LLM
12. Test error handling (malformed XML, network failures)

## Cognitive Testing Requirements

**Integration Test Must Verify:**
- [ ] Semantic equivalence detection (not just string matching)
- [ ] Handles paraphrasing (different words, same meaning)
- [ ] Rejects incorrect responses with clear reasoning
- [ ] Confidence scores are reasonable (0.8+ for clear matches)

## Success Criteria

- Evaluator uses XML prompts (NOT JSON)
- Retry logic handles transient failures
- XML parsing is robust to malformed output
- Integration test with real LLM passes
- Evaluator fixture available for all tests
- Ready for use in integration tests
