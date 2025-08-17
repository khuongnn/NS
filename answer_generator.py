"""Final answer generation module.

This module synthesises a final answer from a validated chain-of-thought.
In practice this could involve calling a language model conditioned on
the chain to produce a concise answer.  We use a trivial heuristic:
the last inferred proposition in the chain (if any) becomes the
answer, otherwise we return a fallback response.
"""

from __future__ import annotations

from typing import List


def generate_final_answer(valid_steps: List[str]) -> str:
    """Produce a final answer from the reasoning steps.

    The function scans the validated chain for the last inference or
    fact that can be translated into a natural language answer.  It
    handles both affirmative and negative predicates and makes a
    best‑effort attempt at grammatical agreement between the subject
    and predicate.  If no suitable step is found, it returns the
    last step verbatim or a generic fallback.

    Parameters
    ----------
    valid_steps : list of str
        The filtered chain‑of‑thought produced by the symbolic
        validator.

    Returns
    -------
    str
        A short answer suitable for user consumption.
    """
    # Mapping from predicate names to singular and plural phrases.  The
    # singular form should be prefaced with an indefinite article
    # ("a"/"an") when appropriate.  The plural form omits the
    # article and pluralises the noun if necessary.
    pred_phrases = {
        "warm_blooded": ("warm‑blooded", "warm‑blooded"),
        "mammal": ("a mammal", "mammals"),
        "bird": ("a bird", "birds"),
        "animal": ("an animal", "animals"),
        "can_move": ("can move", "can move"),
        "cannot_fly": ("cannot fly", "cannot fly"),
        "can_fly": ("can fly", "can fly"),
        "pet": ("a pet", "pets"),
        "has_fur": ("has fur", "have fur"),
    }

    # Helper to build an answer given a predicate and constant
    def build_answer(predicate: str, constant: str) -> str:
        predicate = predicate.strip()
        constant = constant.strip()
        # Determine subject: pluralise the constant for natural sounding output
        subject_plural = _pluralise(constant)
        # Determine phrase mapping (singular, plural)
        phrase_sing, phrase_plur = pred_phrases.get(predicate, (predicate, predicate))
        # Negative predicates start with "cannot"
        if predicate.startswith("cannot"):
            return f"No, {subject_plural} {phrase_plur}."
        # Positive ability predicates start with "can_"
        if predicate.startswith("can_"):
            return f"Yes, {subject_plural} {phrase_plur}."
        # Affiliation predicates (mammal, bird, animal, pet, etc.)
        # Use "is" for singular subject and "are" for plural subject
        # Identify if the constant itself appears plural (very naïve)
        if subject_plural.lower().endswith('s'):
            # plural subject: use plural phrase without article
            return f"Yes, {subject_plural} are {phrase_plur}."
        else:
            # singular subject: use singular phrase with article
            return f"Yes, {subject_plural} is {phrase_sing}."

    # Scan for the last inference step containing "infer"
    for step in reversed(valid_steps):
        if "infer" not in step.lower():
            continue
        infer_part = step.split("infer", 1)[-1].strip().rstrip('.')
        if '(' in infer_part and infer_part.endswith(')'):
            pred, arg_part = infer_part.split('(', 1)
            const = arg_part[:-1]
            return build_answer(pred, const)

    # Scan for a bullet fact (lines starting with '-')
    # Prioritise facts where both predicate and constant appear in the query.
    # Extract the original query from the first reasoning step if available.
    query_line = valid_steps[0] if valid_steps else ""
    q_lower = query_line.lower()
    # Normalise hyphens for predicate matching
    q_norm = q_lower
    for hy in ["-", "–", "—", "‑"]:
        q_norm = q_norm.replace(hy, "_")
    candidate_fact: tuple[str, str] | None = None
    for step in reversed(valid_steps):
        stripped = step.strip()
        if not stripped.startswith("-"):
            continue
        body = stripped.lstrip("-").strip()
        if '(' not in body or not body.endswith(')'):
            continue
        # Skip rule descriptions (e.g. "If mammal(x) then warm_blooded(x)")
        if body.lower().startswith("if "):
            continue
        pred, arg_part = body.split('(', 1)
        const = arg_part[:-1]
        predicate = pred.strip()
        constant = const.strip()
        # Set initial candidate if none selected
        if candidate_fact is None:
            candidate_fact = (predicate, constant)
        # Check if both predicate and constant are mentioned in the query
        if constant.lower() in q_lower and predicate in q_norm:
            candidate_fact = (predicate, constant)
            break
    if candidate_fact is not None:
        pred, const = candidate_fact
        return build_answer(pred, const)

    # Fallback: return the last line or a generic message
    if valid_steps:
        return valid_steps[-1]
    return "Unable to determine an answer."


def _pluralise(word: str) -> str:
    """Return a simple plural form of a noun for use in answers.

    This helper adds 's' or 'es' to the end of the word according to
    basic English pluralisation rules.  It is not comprehensive but
    suffices for the examples in this repository.

    Parameters
    ----------
    word : str
        Singular noun to pluralise.

    Returns
    -------
    str
        Pluralised noun.
    """
    lower = word.lower()
    if lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return f"{word}es"
    elif lower.endswith('y') and lower[-2] not in 'aeiou':
        return f"{word[:-1]}ies"
    else:
        return f"{word}s"