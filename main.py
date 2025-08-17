"""Entry point for the neural‑symbolic reasoning framework.

This script ties together the knowledge base, retrieval module,
chain‑of‑thought generator, symbolic validator and answer generator.  It
provides a simple command‑line interface that accepts a question,
retrieves formal knowledge, generates a reasoning chain, enforces
symbolic constraints and returns an answer.

Usage
-----

    python -m neural_symbolic.main --query "Is a whale warm‑blooded?"

This will print the retrieved knowledge, the intermediate reasoning
steps, the constrained chain, and the final answer.
"""

from __future__ import annotations

import argparse
from typing import List

from .knowledge_base import build_sample_knowledge_base
from .retrieval import retrieve_relevant_knowledge
from .cot_generator import generate_chain_of_thought
from .symbolic_validator import enforce_constraints
from .answer_generator import generate_final_answer


def run_pipeline(query: str) -> None:
    # Build or load the knowledge base
    kb = build_sample_knowledge_base()
    print("Knowledge Base:")
    print(kb)
    print("\nQuery:", query)

    # Retrieval
    facts, rules = retrieve_relevant_knowledge(query, kb)
    print("\nRetrieved Facts:", [str(f) for f in facts])
    print("Retrieved Rules:", [str(r) for r in rules])

    # Generate chain-of-thought
    chain = generate_chain_of_thought(query, facts, rules)
    print("\nGenerated Chain of Thought:")
    for step in chain:
        print(step)

    # Enforce symbolic constraints
    constrained_chain = enforce_constraints(chain, kb)
    print("\nChain after enforcing constraints:")
    for step in constrained_chain:
        print(step)

    # Final answer
    answer = generate_final_answer(constrained_chain)
    print("\nFinal Answer:", answer)


def main():
    parser = argparse.ArgumentParser(description="Run neural-symbolic reasoning pipeline")
    parser.add_argument("--query", required=True, help="Input question for the system")
    args = parser.parse_args()
    run_pipeline(args.query)


if __name__ == "__main__":
    main()