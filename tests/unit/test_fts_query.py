"""Unit tests for the untrusted-input FTS5 query builder."""

from __future__ import annotations

import pytest

from rag_workbench.domain.errors import ValidationError
from rag_workbench.retrieval.bm25 import build_match_query


def test_simple_terms_become_quoted_phrases():
    assert build_match_query("termination convenience") == '"termination" "convenience"'


def test_operators_are_neutralized_as_literals():
    # `NEAR` and `*` are FTS5 syntax; they must be treated as plain text.
    assert build_match_query("foo NEAR bar*") == '"foo" "NEAR" "bar*"'


def test_quotes_are_stripped_not_escaped_into_syntax():
    assert build_match_query('say "hello"') == '"say" "hello"'
    assert build_match_query('foo"bar') == '"foobar"'


def test_boolean_injection_is_neutralized():
    assert build_match_query('x" OR "1=1') == '"x" "OR" "1=1"'


def test_punctuation_only_query_is_rejected():
    with pytest.raises(ValidationError):
        build_match_query("*")
    with pytest.raises(ValidationError):
        build_match_query('" - :')


def test_empty_query_is_rejected():
    with pytest.raises(ValidationError):
        build_match_query("   ")
