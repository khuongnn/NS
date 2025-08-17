"""Chain-of-thought generation module.

This module defines a placeholder implementation for generating a
chain-of-thought (CoT) or intermediate reasoning steps from a query
and retrieved knowledge.  In practice this could interface with
large language models (e.g., via the OpenAI API) using prompt
engineering to elicit detailed reasoning.  Here we provide a simple
template that stitches together the query and knowledge snippets.
"""

from __future__ import annotations

from typing import List

from .knowledge_base import Fact, Rule


def generate_chain_of_thought(query: str, facts: List[Fact], rules: List[Rule]) -> List[str]:
    """Generate a naive chain-of-thought from the query and knowledge.

    This function constructs a list of reasoning steps by simply
    restating the query, then listing relevant facts and rules.  A
    real implementation would call into an LLM to produce a more
    sophisticated reasoning chain.

    Parameters
    ----------
    query: str
        Natural language question.
    facts: list of Fact
        Retrieved facts considered relevant.
    rules: list of Rule
        Retrieved rules considered relevant.

    Returns
    -------
    List[str]
        A sequence of textual reasoning steps.
    """
    steps: List[str] = []
    steps.append(f"We are asked: {query}")
    if facts:
        steps.append("Relevant facts:")
        for fact in facts:
            steps.append(f"- {fact}")
    if rules:
        steps.append("Relevant rules:")
        for rule in rules:
            # Build a string for the rule body: predicate(args) for each antecedent
            body_parts: List[str] = []
            for pred, args in rule.body:
                body_parts.append(f"{pred}({', '.join(args)})")
            body_str = " and ".join(body_parts)
            head_pred, head_args = rule.head
            head_str = f"{head_pred}({', '.join(head_args)})"
            steps.append(f"- If {body_str} then {head_str}")
    # Attempt to unify rules with facts to derive concrete conclusions
    for rule in rules:
        body_preds = [pred for pred, _ in rule.body]
        head_pred, head_args = rule.head
        # Only generate an inference if the head predicate appears in the query.
        q_lower = query.lower()
        # Normalise hyphens in the query so that "warm-blooded" matches
        # the underscore-separated predicate name "warm_blooded".
        q_norm = q_lower
        for hy in ["-", "–", "—", "‑"]:
            q_norm = q_norm.replace(hy, "_")
        if head_pred not in q_norm:
            # Skip rules whose conclusions are unrelated to the query
            continue
        # For each body predicate, find matching facts (unary predicates only)
        candidate_constants: List[set[str]] = []
        for pred, args in rule.body:
            if len(args) != 1:
                candidate_constants.append(set())
                continue
            matches = {fact.arguments[0] for fact in facts if fact.predicate == pred}
            candidate_constants.append(matches)
        # If there are no body predicates or no matches, skip
        if not candidate_constants or any(len(c) == 0 for c in candidate_constants):
            continue
        # Compute intersection of candidate constants across all body predicates
        common_consts = set.intersection(*candidate_constants)
        # Filter constants to those mentioned in the query
        selected_consts = [const for const in common_consts if const.lower() in q_lower]
        if not selected_consts:
            continue
        # For each constant, produce an inference step
        for const in selected_consts:
            body_parts = [f"{pred}({const})" for pred, _ in rule.body]
            body_str = " and ".join(body_parts)
            head_str = f"{head_pred}({const})"
            steps.append(f"From {body_str} we can infer {head_str}")
    return steps