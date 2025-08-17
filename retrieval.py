"""Retrieval module for neural-symbolic reasoning.

This module implements a very simple retrieval mechanism for
demonstration purposes.  Given a natural language query, it searches
through the knowledge base for relevant facts and rules.  In a real
application this component would connect to a search engine,
information retrieval system, or knowledge graph embedding model.
"""

from __future__ import annotations

from typing import List, Tuple

from .knowledge_base import KnowledgeBase, Fact, Rule


def retrieve_relevant_knowledge(query: str, kb: KnowledgeBase) -> Tuple[List[Fact], List[Rule]]:
    """Retrieve relevant facts and rules based on keyword matching.

    The retrieval strategy here is intentionally naïve: it looks for
    predicate names that appear as substrings of the query.  This
    should be replaced by a more sophisticated semantic search in
    practical systems.

    Parameters
    ----------
    query: str
        The natural language question or input.
    kb: KnowledgeBase
        The knowledge base from which to retrieve information.

    Returns
    -------
    (facts, rules): tuple of lists
        Facts and rules deemed relevant to the query.
    """
    # Normalise the query: lowercase and replace various hyphens with underscores.
    query_lower = query.lower()
    # Replace common hyphen characters (normal, non-breaking) with underscores to
    # match predicate names (e.g. "warm-blooded" -> "warm_blooded").
    for hy in ["-", "–", "—", "‑"]:
        query_lower = query_lower.replace(hy, "_")
    relevant_facts: List[Fact] = []
    relevant_rules: List[Rule] = []

    # Check for predicate names in query to select relevant facts
    for fact in kb.facts:
        if fact.predicate in query_lower or any(arg in query_lower for arg in fact.arguments):
            relevant_facts.append(fact)

    # Check for head or body predicate names in query for rules
    for rule in kb.rules:
        head_pred, _ = rule.head
        body_preds = [pred for pred, _ in rule.body]
        if head_pred in query_lower or any(pred in query_lower for pred in body_preds):
            relevant_rules.append(rule)

    return relevant_facts, relevant_rules