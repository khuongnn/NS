"""Training, validation and evaluation routines for the neural–symbolic pipeline.

This module implements a simple training loop to illustrate how the
proposed framework in the paper can be adapted for supervised
learning scenarios.  The training procedure minimises a combined
neural and symbolic loss (Equation 18 in the paper【623944606090707†screenshot】) by searching
over a hyperparameter \(\lambda\) that weighs the symbolic
consistency penalty.  Because the core pipeline in this repository
contains no learnable parameters, the optimisation here is limited
to tuning \(\lambda\) via grid search; however, the code is written
to resemble typical machine‑learning workflows with training,
validation and test phases.

The functions provided here take as input a dataset of question–
answer pairs and a knowledge base.  They use the retrieval module,
chain‑of‑thought generator, symbolic validator and answer
generator from the rest of the package to produce predictions.  The
loss function combines a cross‑entropy term (which is 0 for a
correct prediction and 1 for an incorrect one) with a symbolic
penalty proportional to the number of intermediate reasoning steps
removed by the symbolic validator.  The best \(\lambda\) is chosen
based on the lowest average loss on the validation set.

Example usage
-------------

Load a toy dataset, train the model and evaluate on the test set:

```python
from neural_symbolic.training import train_and_evaluate
from neural_symbolic.data import load_toy_dataset
from neural_symbolic.knowledge_base import build_sample_knowledge_base

kb = build_sample_knowledge_base()
train_data, val_data, test_data = load_toy_dataset()
best_lambda, metrics = train_and_evaluate(train_data, val_data, test_data, kb)
print(f"Best lambda: {best_lambda}")
print("Test metrics:", metrics)
```

This will run a simple hyperparameter search over \(\lambda\) values
and report accuracy and loss on the test set.

Note: In a full implementation using real datasets like GSM8K or
CFQ, one would fine‑tune an LLM using gradient descent and compute
the symbolic loss within the training loop.  The code here is a
lightweight approximation intended for educational purposes.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Iterable

from .knowledge_base import KnowledgeBase, build_sample_knowledge_base
from .retrieval import retrieve_relevant_knowledge
from .cot_generator import generate_chain_of_thought
from .symbolic_validator import enforce_constraints
from .answer_generator import generate_final_answer


Example = Dict[str, str]


def _predict_answer(question: str, kb: KnowledgeBase) -> Tuple[str, List[str], List[str]]:
    """Run the pipeline on a single question to obtain the answer and reasoning.

    Parameters
    ----------
    question : str
        The natural language query.
    kb : KnowledgeBase
        The knowledge base used for retrieval and validation.

    Returns
    -------
    Tuple[str, List[str], List[str]]
        A tuple of the predicted final answer, the full chain of
        reasoning before symbolic validation, and the chain after
        validation.
    """
    facts, rules = retrieve_relevant_knowledge(question, kb)
    chain = generate_chain_of_thought(question, facts, rules)
    constrained_chain = enforce_constraints(chain, kb)
    answer = generate_final_answer(constrained_chain)
    return answer, chain, constrained_chain


def _compute_loss(pred: str, gold: str, chain: List[str], constrained: List[str], lam: float) -> float:
    """Compute the combined neural and symbolic loss for one example.

    The neural loss is 0 if the predicted answer exactly matches the
    gold answer and 1 otherwise.  The symbolic loss is the number of
    reasoning steps removed by the validator.  These are combined
    according to Equation 18 of the paper【623944606090707†screenshot】:

    \[ L = L_{\text{LLM}}(y, y^*) + \lambda L_{\text{symbolic}}(C(z, K)) \].

    Parameters
    ----------
    pred : str
        The predicted answer.
    gold : str
        The ground truth answer.
    chain : List[str]
        The chain of thought before enforcing constraints.
    constrained : List[str]
        The chain of thought after enforcing constraints.
    lam : float
        The hyperparameter controlling the trade‑off between the
        two loss terms.

    Returns
    -------
    float
        The total loss for the example.
    """
    llm_loss = 0.0 if pred.strip().lower() == gold.strip().lower() else 1.0
    symbolic_penalty = float(len(chain) - len(constrained))
    return llm_loss + lam * symbolic_penalty


def _evaluate_split(split: Iterable[Example], kb: KnowledgeBase, lam: float) -> Tuple[float, float, float]:
    """Evaluate the pipeline on a dataset split for a given lambda.

    Parameters
    ----------
    split : Iterable[Example]
        An iterable of examples, each with ``question`` and ``answer`` keys.
    kb : KnowledgeBase
        The knowledge base used for inference.
    lam : float
        The symbolic loss weight.

    Returns
    -------
    Tuple[float, float, float]
        The average loss, average symbolic penalty and accuracy over the split.
    """
    total_loss = 0.0
    total_symbolic = 0.0
    correct = 0
    n = 0
    for example in split:
        n += 1
        pred, chain, constrained = _predict_answer(example["question"], kb)
        loss = _compute_loss(pred, example["answer"], chain, constrained, lam)
        total_loss += loss
        total_symbolic += (len(chain) - len(constrained))
        if pred.strip().lower() == example["answer"].strip().lower():
            correct += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_symbolic / n, correct / n


def train_lambda(train_data: Iterable[Example], val_data: Iterable[Example],
                 kb: KnowledgeBase, lambdas: List[float]) -> Tuple[float, float]:
    """Perform a simple hyperparameter search over lambda.

    The function evaluates a set of candidate lambda values on the
    validation set and selects the one with the smallest average
    total loss.  It returns the best lambda and the corresponding
    validation loss.

    Parameters
    ----------
    train_data : Iterable[Example]
        The training examples.  Currently unused, but kept for API
        symmetry if training becomes more complex in the future.
    val_data : Iterable[Example]
        The validation examples used to choose lambda.
    kb : KnowledgeBase
        The knowledge base.
    lambdas : List[float]
        Candidate values of lambda to evaluate.

    Returns
    -------
    Tuple[float, float]
        The best lambda and its corresponding validation loss.
    """
    best_lam = None
    best_loss = float("inf")
    for lam in lambdas:
        val_loss, _, _ = _evaluate_split(val_data, kb, lam)
        if val_loss < best_loss:
            best_loss = val_loss
            best_lam = lam
    return best_lam if best_lam is not None else lambdas[0], best_loss


def train_and_evaluate(train_data: Iterable[Example], val_data: Iterable[Example],
                       test_data: Iterable[Example], kb: KnowledgeBase,
                       lambdas: List[float] | None = None) -> Tuple[float, Dict[str, float]]:
    """Train the neural–symbolic model and evaluate on the test set.

    This function performs hyperparameter tuning to select the best
    symbolic weight \(\lambda\) using the validation set, then
    computes accuracy and loss on the test set with the chosen
    \(\lambda\).  It reports the mean loss, mean symbolic penalty and
    accuracy on the test data.

    Parameters
    ----------
    train_data : Iterable[Example]
        Training examples.  Included for future extensibility but not
        used in the current implementation.
    val_data : Iterable[Example]
        Validation examples used to select \(\lambda\).
    test_data : Iterable[Example]
        Test examples used for the final evaluation.
    kb : KnowledgeBase
        The knowledge base.
    lambdas : List[float], optional
        Candidate values of \(\lambda\).  If None, defaults to
        `[0.0, 0.1, 0.5, 1.0]`.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        The best \(\lambda\) and a dictionary of test metrics:
        ``{"loss": float, "symbolic_penalty": float, "accuracy": float}``.
    """
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.5, 1.0]
    best_lam, _ = train_lambda(train_data, val_data, kb, lambdas)
    test_loss, test_symbolic, test_acc = _evaluate_split(test_data, kb, best_lam)
    return best_lam, {"loss": test_loss, "symbolic_penalty": test_symbolic, "accuracy": test_acc}
