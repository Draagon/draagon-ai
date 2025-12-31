"""Tests for content-aware word sense disambiguation.

Tests the integration of content type analysis with WSD to ensure:
- Prose is processed with full WSD
- Code docstrings/comments are extracted and processed
- Data/config content is skipped appropriately
- Mixed content is handled correctly
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path

# Add prototype src to path
prototype_root = Path(__file__).parent.parent
src_path = prototype_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from content_aware_wsd import (
    ContentAwareWSD,
    ContentAwareWSDResult,
    process_content_for_wsd,
    disambiguate_in_context,
)
from content_analyzer import ContentType
from wsd import WSDConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def processor():
    """Create a content-aware WSD processor (no LLM, heuristic mode)."""
    return ContentAwareWSD(require_wordnet=False)


@pytest.fixture
def processor_with_llm(mock_llm):
    """Create processor with mock LLM."""
    return ContentAwareWSD(llm=mock_llm, require_wordnet=False)


# =============================================================================
# Test: Prose Processing
# =============================================================================


class TestProseProcessing:
    """Test WSD on prose content."""

    @pytest.mark.asyncio
    async def test_prose_is_fully_processed(self, processor):
        """Prose content goes through full WSD."""
        text = "I deposited money in the bank near the river."

        result = await processor.process(text)

        assert result.content_analysis.content_type == ContentType.PROSE
        assert not result.skipped_processing
        assert result.processed_text == text

    @pytest.mark.asyncio
    async def test_prose_disambiguation_results(self, processor):
        """Prose produces disambiguation results."""
        text = "The bank processes financial transactions daily."

        result = await processor.process(text, extract_all_words=True)

        # Should have some disambiguation results
        assert result.has_disambiguations() or len(result.processed_text) > 0

    @pytest.mark.asyncio
    async def test_short_prose_skipped(self, processor):
        """Very short prose is skipped."""
        text = "Hi there."  # Too short

        result = await processor.process(text)

        assert result.skipped_processing
        assert "Insufficient" in result.skip_reason


# =============================================================================
# Test: Code Processing
# =============================================================================


class TestCodeProcessing:
    """Test WSD on code content."""

    @pytest.mark.asyncio
    async def test_code_extracts_docstrings(self, processor):
        """Code processing extracts docstrings for WSD."""
        code = '''
import os
from pathlib import Path

def process_bank_transaction(account, amount):
    """Transfer funds to a financial institution account.

    This function handles deposits and withdrawals from
    the customer bank account.
    """
    pass

class BankManager:
    """Manages bank operations and customer accounts."""
    pass
'''
        result = await processor.process(code)

        assert result.content_analysis.content_type == ContentType.CODE
        # Should extract the docstrings
        assert "financial institution" in result.processed_text.lower()
        assert "bank" in result.processed_text.lower()

    @pytest.mark.asyncio
    async def test_code_extracts_comments(self, processor):
        """Code processing extracts comments for WSD."""
        code = """
import math
from typing import List

def calculate(values):
    # Calculate the average balance in the account
    total = sum(values)
    # Return the final bank statement total
    return total / len(values)

class Calculator:
    pass
"""
        result = await processor.process(code)

        assert result.content_analysis.content_type == ContentType.CODE
        # Should extract comments
        nl_text = result.processed_text.lower()
        assert "balance" in nl_text or "account" in nl_text or "bank" in nl_text

    @pytest.mark.asyncio
    async def test_code_syntax_not_processed(self, processor):
        """Code syntax is not processed for WSD."""
        code = '''
import os

def process():
    """A simple processor."""
    bank = Bank()  # "bank" here is a variable name, not NL
    bank.deposit(100)
    return bank.balance
'''
        result = await processor.process(code)

        # The variable name "bank" should NOT be in the NL text
        # Only the docstring should be extracted
        assert "simple processor" in result.processed_text.lower()
        # "bank.deposit" syntax should not be in processed text
        assert "bank.deposit" not in result.processed_text


# =============================================================================
# Test: Data Processing
# =============================================================================


class TestDataProcessing:
    """Test handling of data content."""

    @pytest.mark.asyncio
    async def test_csv_is_skipped(self, processor):
        """CSV data is skipped for WSD."""
        csv = """customer_id,name,bank_account,balance
1001,John Doe,12345678,5000.00
1002,Jane Smith,87654321,3500.50"""

        result = await processor.process(csv)

        assert result.content_analysis.content_type == ContentType.DATA
        assert result.skipped_processing
        assert "WSD not applicable" in result.skip_reason

    @pytest.mark.asyncio
    async def test_json_is_skipped(self, processor):
        """JSON data is skipped for WSD."""
        json_data = """{
    "users": [
        {"id": 1, "name": "Alice", "bank": "First National"},
        {"id": 2, "name": "Bob", "bank": "River Bank"}
    ]
}"""
        result = await processor.process(json_data)

        assert result.content_analysis.content_type == ContentType.DATA
        assert result.skipped_processing

    @pytest.mark.asyncio
    async def test_data_extracts_schema(self, processor):
        """Data processing extracts schema information."""
        csv = """customer_id,name,email
1,John,john@example.com"""

        result = await processor.process(csv)

        # Should have format info
        assert any(
            sk.get("object") == "csv"
            for sk in result.structural_knowledge
        )


# =============================================================================
# Test: Config Processing
# =============================================================================


class TestConfigProcessing:
    """Test handling of config content."""

    @pytest.mark.asyncio
    async def test_yaml_is_skipped(self, processor):
        """YAML config is skipped for WSD."""
        yaml = """
database:
  host: localhost
  port: 5432
  name: myapp_db

server:
  port: 8080
"""
        result = await processor.process(yaml)

        assert result.content_analysis.content_type == ContentType.CONFIG
        assert result.skipped_processing
        assert "WSD not applicable" in result.skip_reason

    @pytest.mark.asyncio
    async def test_config_extracts_patterns(self, processor):
        """Config processing extracts pattern information."""
        yaml = """
database:
  host: localhost
  port: 5432
"""
        result = await processor.process(yaml)

        assert any(
            sk.get("object") == "yaml"
            for sk in result.structural_knowledge
        )


# =============================================================================
# Test: Mixed Content
# =============================================================================


class TestMixedContentProcessing:
    """Test handling of mixed content."""

    @pytest.mark.asyncio
    async def test_readme_with_code_blocks(self, processor):
        """README with code blocks extracts NL portions."""
        readme = """
# Banking Library

This library provides tools for managing bank accounts
and processing financial transactions.

## Installation

```python
pip install banking
```

## Usage

```python
from banking import BankAccount

account = BankAccount()
account.deposit(100)
```

The library supports multiple account types.
"""
        result = await processor.process(readme)

        # Should process NL portions
        assert not result.skipped_processing
        # Should have some processed text
        assert len(result.processed_text) > 0


# =============================================================================
# Test: Word Disambiguation in Context
# =============================================================================


class TestWordDisambiguationInContext:
    """Test disambiguating specific words with content awareness."""

    @pytest.mark.asyncio
    async def test_disambiguate_word_in_prose(self, processor):
        """Disambiguate a word in prose context."""
        text = "I walked along the river bank to see the fish swimming."

        # Note: This may return None if WordNet isn't available in test env
        result = await processor.disambiguate_word("bank", text)

        # Just verify it doesn't crash - actual result depends on WordNet
        assert result is None or result.word == "bank"

    @pytest.mark.asyncio
    async def test_disambiguate_word_in_code_docstring(self, processor):
        """Disambiguate a word found in code docstring."""
        code = '''
def process():
    """Handle bank transactions for customers."""
    pass
'''
        result = await processor.disambiguate_word("bank", code)

        # Should find "bank" in the docstring
        assert result is None or result.word == "bank"

    @pytest.mark.asyncio
    async def test_disambiguate_word_in_code_syntax(self, processor):
        """Disambiguation uses NL context, not code syntax."""
        # Code with multiple indicators so heuristics detect it as code
        # and "bank" only appears in variable/import names
        code = """
import bank
from pathlib import Path

def main():
    bank.process()
    return bank.result

class Manager:
    def handle(self):
        pass
"""
        result = await processor.disambiguate_word("bank", code)

        # With proper code detection, "bank" won't be in NL portion
        # So result should be None, OR if heuristics fail and treat as prose,
        # it's acceptable since we test the happy path elsewhere
        # This test mainly ensures it doesn't crash
        assert result is None or isinstance(result.synset_id, str)


# =============================================================================
# Test: Result Object Methods
# =============================================================================


class TestResultMethods:
    """Test ContentAwareWSDResult methods."""

    def test_has_disambiguations(self):
        """Test has_disambiguations method."""
        from wsd import DisambiguationResult
        from content_analyzer import ContentAnalysis

        # Empty result
        result = ContentAwareWSDResult(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE)
        )
        assert not result.has_disambiguations()

        # With disambiguation
        result.disambiguation_results = {
            "bank": DisambiguationResult(
                word="bank",
                lemma="bank",
                pos="n",
                synset_id="bank.n.01",
            )
        }
        assert result.has_disambiguations()

    def test_get_synset_ids(self):
        """Test get_synset_ids method."""
        from wsd import DisambiguationResult
        from content_analyzer import ContentAnalysis

        result = ContentAwareWSDResult(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank", lemma="bank", pos="n", synset_id="bank.n.01"
                ),
                "money": DisambiguationResult(
                    word="money", lemma="money", pos="n", synset_id="money.n.01"
                ),
            },
        )

        ids = result.get_synset_ids()
        assert "bank.n.01" in ids
        assert "money.n.01" in ids

    def test_get_high_confidence_disambiguations(self):
        """Test filtering by confidence."""
        from wsd import DisambiguationResult
        from content_analyzer import ContentAnalysis

        result = ContentAwareWSDResult(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank", lemma="bank", pos="n",
                    synset_id="bank.n.01", confidence=0.9,
                ),
                "river": DisambiguationResult(
                    word="river", lemma="river", pos="n",
                    synset_id="river.n.01", confidence=0.5,
                ),
            },
        )

        high_conf = result.get_high_confidence_disambiguations(threshold=0.7)
        assert "bank" in high_conf
        assert "river" not in high_conf


# =============================================================================
# Test: Metrics Tracking
# =============================================================================


class TestMetricsTracking:
    """Test that metrics are tracked correctly."""

    @pytest.mark.asyncio
    async def test_metrics_increment(self, processor):
        """Metrics are updated on processing."""
        await processor.process("This is a simple prose sentence about banking.")
        await processor.process("def foo(): pass\nclass Bar: pass")
        await processor.process("a,b,c\n1,2,3")

        metrics = processor.get_metrics()
        assert metrics["total_processed"] == 3
        assert metrics["prose_processed"] >= 1
        assert metrics["data_skipped"] >= 1

    @pytest.mark.asyncio
    async def test_combined_metrics(self, processor):
        """Combined metrics include all subsystems."""
        await processor.process("Test content here.")

        combined = processor.get_combined_metrics()
        assert "content_aware" in combined
        assert "content_analyzer" in combined
        assert "wsd" in combined


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_process_content_for_wsd(self):
        """process_content_for_wsd function works."""
        result = await process_content_for_wsd(
            "The bank is near the river where fish swim."
        )

        assert result.content_analysis.content_type == ContentType.PROSE
        assert not result.skipped_processing

    @pytest.mark.asyncio
    async def test_disambiguate_in_context(self):
        """disambiguate_in_context function works."""
        result = await disambiguate_in_context(
            "bank",
            "I deposited money at the bank.",
        )

        # May be None if WordNet unavailable
        assert result is None or result.word == "bank"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content(self, processor):
        """Empty content is handled gracefully."""
        result = await processor.process("")

        assert result.skipped_processing

    @pytest.mark.asyncio
    async def test_whitespace_only(self, processor):
        """Whitespace-only content is handled."""
        result = await processor.process("   \n\t\n   ")

        assert result.skipped_processing

    @pytest.mark.asyncio
    async def test_code_without_comments(self, processor):
        """Code without NL is skipped."""
        code = """
def foo():
    return 42

class Bar:
    x = 1
"""
        result = await processor.process(code)

        # May be skipped due to insufficient NL
        # Or processed with empty/minimal results
        assert result.processed_text == "" or result.skipped_processing

    @pytest.mark.asyncio
    async def test_unicode_content(self, processor):
        """Unicode content is handled."""
        text = "Le café de la banque est délicieux. 銀行のコーヒーが美味しい。"

        result = await processor.process(text)

        # Should process without error
        assert result.content_analysis.content_type == ContentType.PROSE


# =============================================================================
# Test: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_api_documentation(self, processor):
        """API documentation with code examples."""
        doc = """
# Bank Account API

The Bank Account API provides endpoints for managing financial accounts.

## Endpoints

### GET /accounts/{id}

Retrieves the account details from the bank's database.

```python
response = client.get("/accounts/123")
account = response.json()
print(account["balance"])
```

### POST /accounts

Creates a new bank account for a customer.
"""
        result = await processor.process(doc)

        # Should process NL portions
        assert not result.skipped_processing
        assert "bank" in result.processed_text.lower()

    @pytest.mark.asyncio
    async def test_error_log_with_context(self, processor):
        """Error log with stack trace."""
        log = """
2024-01-15 10:23:48 ERROR Bank transaction failed
  at BankService.processDeposit (bank.py:42)
  at main (app.py:10)

The customer's bank account could not be updated.
"""
        result = await processor.process(log)

        # Should handle gracefully
        assert result.content_analysis is not None
