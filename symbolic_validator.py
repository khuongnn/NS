"""Symbolic constraint enforcement.

This module enforces simple symbolic constraints on a chain-of-thought.
In practical neural‑symbolic systems this component would perform
logical reasoning to ensure that each step of the chain satisfies
well‑formedness conditions and respects the rules of the formal
knowledge base.  Here we implement a lightweight checker that
eliminates steps that contradict the knowledge base.
"""

from __future__ import annotations

from typing import List

from .knowledge_base import KnowledgeBase


def enforce_constraints(steps: List[str], kb: KnowledgeBase) -> List[str]:
    """Filter a chain-of-thought according to knowledge base facts.

    This function removes any step that negates a known fact.  It is a
    very rudimentary example of constraint enforcement; a real system
    would use a logic engine (e.g. Prolog, Z3) to validate each
    inference.

    Parameters
    ----------
    steps: list of str
        The generated reasoning steps.
    kb: KnowledgeBase
        The knowledge base used for validation.

    Returns
    -------
    list of str
        Filtered steps that are consistent with the knowledge base.
    """
    valid_steps: List[str] = []
    for step in steps:
        # Simple heuristic: if the step contains "not" followed by a
        # fact predicate, and that fact exists in the KB, we discard it
        lower = step.lower()
        discard = False
        for fact in kb.facts:
            phrase = f"not {fact.predicate}"
            if phrase in lower:
                discard = True
                break
        if not discard:
            valid_steps.append(step)
    return valid_steps