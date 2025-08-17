"""Formal Knowledge Base Module.

This module defines a minimal representation for a formal knowledge base
consisting of logical rules and facts.  In a production system these
structures might be stored in a graph database or ontology service.  For
demonstration purposes we provide simple Python classes to store and
query the knowledge.  Each rule is represented by a head and a body
consisting of antecedent conditions.  Facts are stored as simple
predicate tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable


@dataclass
class Fact:
    """Representation of a ground atomic fact.

    A fact is a predicate applied to a sequence of constant arguments.
    For example, Fact("mammal", ("whale",)) represents the fact
    mammal(whale).
    """
    predicate: str
    arguments: Tuple[str, ...]

    def __str__(self) -> str:
        args = ", ".join(self.arguments)
        return f"{self.predicate}({args})"


@dataclass
class Rule:
    """Representation of a Horn clause.

    A rule has a head (consequent) and a body (a list of antecedent
    predicates).  For example, to encode the logical implication
    "∀x (mammal(x) ⇒ warm_blooded(x))" you would use:

    Rule(head=('warm_blooded', ('x',)), body=[('mammal', ('x',))])
    """
    head: Tuple[str, Tuple[str, ...]]
    body: List[Tuple[str, Tuple[str, ...]]]

    def __str__(self) -> str:
        body_str = ", ".join(f"{pred}({', '.join(args)})" for pred, args in self.body)
        head_pred, head_args = self.head
        head_str = f"{head_pred}({', '.join(head_args)})"
        return f"{body_str} -> {head_str}"


@dataclass
class KnowledgeBase:
    """A simple knowledge base containing facts and rules."""
    facts: List[Fact] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)

    def add_fact(self, predicate: str, *args: str) -> None:
        self.facts.append(Fact(predicate, args))

    def add_rule(self, head: Tuple[str, Tuple[str, ...]], body: Iterable[Tuple[str, Tuple[str, ...]]]) -> None:
        self.rules.append(Rule(head=head, body=list(body)))

    def query_facts(self, predicate: str) -> List[Fact]:
        """Return all facts with the given predicate."""
        return [fact for fact in self.facts if fact.predicate == predicate]

    def __str__(self) -> str:
        lines = ["Facts:"]
        for f in self.facts:
            lines.append(f"  {f}")
        lines.append("Rules:")
        for r in self.rules:
            lines.append(f"  {r}")
        return "\n".join(lines)


def build_sample_knowledge_base() -> KnowledgeBase:
    """Create a toy knowledge base for demonstration.

    The rules and facts here are only illustrative.  A real system
    would load domain-specific ontologies, logic rules or knowledge
    graphs.
    """
    kb = KnowledgeBase()
    # ------------------------------------------------------------------
    # Facts
    #
    # Mammals
    kb.add_fact("mammal", "whale")
    kb.add_fact("mammal", "dog")
    kb.add_fact("mammal", "cat")
    # Birds
    kb.add_fact("bird", "penguin")
    kb.add_fact("bird", "ostrich")
    kb.add_fact("bird", "sparrow")
    # Additional properties
    kb.add_fact("has_fur", "dog")
    kb.add_fact("cannot_fly", "penguin")
    kb.add_fact("cannot_fly", "ostrich")
    kb.add_fact("can_fly", "sparrow")
    kb.add_fact("can_move", "cat")
    # ------------------------------------------------------------------
    # Rules
    #
    # All mammals are warm-blooded
    kb.add_rule(head=("warm_blooded", ("x",)), body=[("mammal", ("x",))])
    # All birds are warm-blooded
    kb.add_rule(head=("warm_blooded", ("x",)), body=[("bird", ("x",))])
    # All mammals are animals
    kb.add_rule(head=("animal", ("x",)), body=[("mammal", ("x",))])
    # All birds are animals
    kb.add_rule(head=("animal", ("x",)), body=[("bird", ("x",))])
    # Animals can move
    kb.add_rule(head=("can_move", ("x",)), body=[("animal", ("x",))])
    # Dogs are pets
    kb.add_rule(head=("pet", ("x",)), body=[("dog", ("x",))])
    return kb