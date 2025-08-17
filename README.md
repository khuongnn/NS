# Formal Knowledge Augmented Language Models for Explainable and Robust Reasoning

This repository contains a minimal codebase that demonstrates the
principles of the neural–symbolic integration framework proposed in
the accompanying paper.  The goal is to augment a large language
model (LLM) with formal knowledge (logical rules, ontologies or
knowledge graphs) to produce more reliable and explainable
reasoning.

## Overview

The system is composed of several simple modules:

| Module | Description |
|-------|-------------|
| `knowledge_base.py` | Defines a toy knowledge base consisting of ground facts and Horn rules.  In a real implementation this would load data from an ontology or knowledge graph. |
| `retrieval.py` | Provides a naïve keyword‑based retrieval mechanism to extract relevant facts and rules based on a natural language query.  It should be replaced by a proper information retrieval component. |
| `cot_generator.py` | Generates an intermediate reasoning chain (chain‑of‑thought) from the query and retrieved knowledge.  Currently this simply enumerates the query, facts and rules; in practice this would call an LLM. |
| `symbolic_validator.py` | Enforces symbolic constraints on the chain of thought.  The example implementation removes any step that contradicts known facts.  A full system would integrate with a logic engine to ensure logical soundness. |
| `answer_generator.py` | Synthesises a final answer from the validated chain.  It looks for inference conclusions in the chain and returns the last one. |
| `main.py` | The entry point that ties everything together.  It builds a knowledge base, retrieves relevant knowledge for a user query, generates a reasoning chain, enforces constraints and prints the final answer. |
| `data.py` | Provides helper functions for building toy datasets and splitting them into training, validation and test sets.  In real experiments you would replace this with loaders for GSM8K, CFQ, SCAN, etc. |
| `training.py` | Implements a simple training loop and hyperparameter search for the symbolic loss weight \(\lambda\).  It demonstrates how to evaluate the pipeline on train/validation/test splits and report metrics such as accuracy and loss. |

## Extending the Code

This codebase is intended as a starting point for further
development.  To adapt it for real‑world use you should:

* Replace the toy `KnowledgeBase` with a loader for your own
  knowledge representation (ontologies, logical rules, knowledge graphs).  You
  could integrate with RDF stores, OWL ontologies or custom
  Prolog/logic programming systems.
* Implement a proper retrieval module that performs semantic search
  over your knowledge base.  Techniques like BM25, dense retrieval
  with transformers, or graph embeddings are suitable.
* Integrate with a state‑of‑the‑art LLM in `cot_generator.py` to
  produce rich, human‑readable chains of reasoning using prompt
  engineering.  The chain should cite retrieved knowledge explicitly.
* Enhance the symbolic validator to perform logical inference
  using tools like [PyDatalog](https://github.com/pyDatalog/pyDatalog),
  [Clingo](https://potassco.org/clingo/) or SMT solvers such as
  [Z3](https://github.com/Z3Prover/z3) to verify each reasoning step
  against your knowledge base.
* Develop a more sophisticated answer generator that summarises the
  reasoning chain or presents the result in a user‑friendly format.

By following these guidelines you can build a powerful hybrid reasoning
system that combines the fluency of neural language models with the
rigour of symbolic logic.

## Training and Evaluation on Datasets

In addition to the core pipeline, this repository includes a
lightweight example of how to perform training, validation and testing
on question–answer datasets.  The paper discusses experiments on
benchmark corpora such as GSM8K, CFQ and SCAN, each of which comes
with pre‑defined train–validation–test splits.  To emulate this setup in
our toy environment we provide:

* **`data.py`**, which contains a small set of handcrafted
  question–answer pairs inspired by the knowledge snippets used in the
  paper.  The function `load_toy_dataset()` randomly divides these
  examples into training, validation and test subsets, much like the
  standard splits used for GSM8K and CFQ.
* **`training.py`**, which implements a simple hyperparameter search
  over the symbolic loss weight \(\lambda\) described in Equation 18
  of the paper.  It evaluates candidate \(\lambda\) values on the
  validation set and selects the one that minimises the combined
  neural and symbolic loss.  The selected \(\lambda\) is then used to
  compute accuracy, average loss and symbolic penalties on the test
  set.

To run an end‑to‑end demonstration, use the following Python snippet:

```python
from neural_symbolic.training import train_and_evaluate
from neural_symbolic.data import load_toy_dataset
from neural_symbolic.knowledge_base import build_sample_knowledge_base

# Build the knowledge base and load the toy splits
kb = build_sample_knowledge_base()
train_data, val_data, test_data = load_toy_dataset()

# Train (select lambda) and evaluate
best_lambda, metrics = train_and_evaluate(train_data, val_data, test_data, kb)

print(f"Selected lambda: {best_lambda:.2f}")
print("Test metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.3f}")
```

This code prints the chosen \(\lambda\) and the evaluation metrics on
the test set.  To adapt the framework for real datasets, replace
`load_toy_dataset()` with loaders for GSM8K or CFQ and integrate a
trainable LLM into the `cot_generator` module.  The general structure
of splitting data and tuning hyperparameters based on a validation
set remains the same.
