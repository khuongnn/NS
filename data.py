"""Utility functions for loading toy datasets.

This module defines helper routines to construct small question–answer
datasets for demonstrating the training, validation and testing
pipeline described in the accompanying paper.  In a realistic
implementation, these functions would be replaced with loaders for
benchmark datasets such as GSM8K, CFQ and SCAN.  However, because
network access and large external libraries are unavailable in this
environment, we provide a handful of handcrafted examples to mimic
the structure of those datasets.

Each entry in the dataset is a dictionary with two fields:

``question``
    The natural language query posed to the system.

``answer``
    The expected final answer (a simple string) that the system
    should produce after reasoning over the formal knowledge base.

The ``load_toy_dataset`` function returns three lists of
question–answer pairs corresponding to training, validation and test
splits.  You can adjust the size of these splits or extend the list
of examples as needed.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import random


Example = Dict[str, str]


def _build_examples() -> List[Example]:
    """Construct a list of toy question–answer examples.

    The questions here are inspired by the illustrative knowledge
    snippets used in the paper (e.g. statements about mammals and
    birds).  The answers reflect the conclusions that can be drawn
    from those facts and rules.  If you wish to incorporate your own
    knowledge base or extend the set of examples, feel free to edit
    this list.

    Returns
    -------
    List[Example]
        A list of dictionaries, each containing a question and its
        corresponding answer.
    """
    return [
        {
            "question": "Is a whale a mammal?",
            "answer": "Yes, whales are mammals.",
        },
        {
            "question": "Is a whale warm‑blooded?",
            "answer": "Yes, whales are warm‑blooded animals.",
        },
        {
            "question": "Do penguins fly?",
            "answer": "No, penguins cannot fly.",
        },
        {
            "question": "Are sparrows warm‑blooded?",
            "answer": "Yes, sparrows are warm‑blooded.",
        },
        {
            "question": "Can cats move?",
            "answer": "Yes, cats can move.",
        },
        {
            "question": "Are cats warm‑blooded?",
            "answer": "Yes, cats are warm‑blooded.",
        },
        {
            "question": "Is an ostrich a bird?",
            "answer": "Yes, an ostrich is a bird.",
        },
        {
            "question": "Is an ostrich capable of flight?",
            "answer": "No, ostriches cannot fly.",
        },
    ]


def load_toy_dataset(train_frac: float = 0.6, val_frac: float = 0.2,
                     seed: int = 42) -> Tuple[List[Example], List[Example], List[Example]]:
    """Split the toy examples into training, validation and test sets.

    Parameters
    ----------
    train_frac : float, optional
        The fraction of examples to use for training.  Must be between
        0 and 1.  The default of 0.6 allocates 60 % of the data to
        training.
    val_frac : float, optional
        The fraction of examples to allocate to validation.  The
        remainder goes to the test set.  The default of 0.2 yields
        20 % validation and 20 % test.
    seed : int, optional
        Random seed for reproducibility when shuffling examples.

    Returns
    -------
    Tuple[List[Example], List[Example], List[Example]]
        Three lists containing the training, validation and test
        examples respectively.
    """
    examples = _build_examples()
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be in the range (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0 so that test split is non-empty")
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    train_examples = examples[:train_end]
    val_examples = examples[train_end:val_end]
    test_examples = examples[val_end:]
    return train_examples, val_examples, test_examples
