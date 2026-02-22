"""Grammar-first sentence builder for assistive communication.

Structured 4-group system: Subject -> Adverb -> Adjective -> Action.
Adverb and Adjective are skippable.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)

OTHER_LABEL = "Other"
SKIP_LABEL = "Skip"

GRAMMAR_STEPS = ["subject", "adverb", "adjective", "action"]
SKIPPABLE_STEPS = {"adverb", "adjective"}


class GrammarStep(IntEnum):
    SUBJECT = 0
    ADVERB = 1
    ADJECTIVE = 2
    ACTION = 3


# ── Word Pools ────────────────────────────────────────────────

SUBJECT_POOL: list[dict[str, Any]] = [
    {"word": "I", "person": 1, "number": "singular"},
    {"word": "we", "person": 1, "number": "plural"},
    {"word": "patient", "person": 3, "number": "singular"},
    {"word": "nurse", "person": 3, "number": "singular"},
    {"word": "doctor", "person": 3, "number": "singular"},
    {"word": "he", "person": 3, "number": "singular"},
    {"word": "she", "person": 3, "number": "singular"},
    {"word": "my pain", "person": 3, "number": "singular"},
    {"word": "my chest", "person": 3, "number": "singular"},
    {"word": "my head", "person": 3, "number": "singular"},
    {"word": "my stomach", "person": 3, "number": "singular"},
    {"word": "my back", "person": 3, "number": "singular"},
    {"word": "my arm", "person": 3, "number": "singular"},
    {"word": "my leg", "person": 3, "number": "singular"},
    {"word": "they", "person": 3, "number": "plural"},
]

ADVERB_POOL: list[dict[str, Any]] = [
    {"word": "urgently", "tags": ["emergency", "high_priority"]},
    {"word": "now", "tags": ["immediate", "high_priority"]},
    {"word": "please", "tags": ["polite", "general"]},
    {"word": "slowly", "tags": ["pace", "comfort"]},
    {"word": "again", "tags": ["repetition", "general"]},
    {"word": "today", "tags": ["time", "general"]},
    {"word": "immediately", "tags": ["emergency", "high_priority"]},
    {"word": "still", "tags": ["ongoing", "symptom"]},
    {"word": "sometimes", "tags": ["frequency", "symptom"]},
    {"word": "always", "tags": ["frequency", "symptom"]},
    {"word": "really", "tags": ["intensity", "emphasis"]},
    {"word": "suddenly", "tags": ["onset", "symptom"]},
    {"word": "constantly", "tags": ["frequency", "symptom"]},
]

ADJECTIVE_POOL: list[dict[str, Any]] = [
    {"word": "severe", "tags": ["intensity", "pain", "emergency"]},
    {"word": "mild", "tags": ["intensity", "pain"]},
    {"word": "sharp", "tags": ["quality", "pain"]},
    {"word": "dizzy", "tags": ["symptom", "head"]},
    {"word": "nauseous", "tags": ["symptom", "stomach"]},
    {"word": "better", "tags": ["improvement", "status"]},
    {"word": "worse", "tags": ["decline", "status"]},
    {"word": "weak", "tags": ["symptom", "general"]},
    {"word": "strong", "tags": ["improvement", "status"]},
    {"word": "tired", "tags": ["symptom", "fatigue"]},
    {"word": "cold", "tags": ["temperature", "comfort"]},
    {"word": "hot", "tags": ["temperature", "comfort"]},
    {"word": "uncomfortable", "tags": ["comfort", "general"]},
    {"word": "scared", "tags": ["emotion", "mental"]},
    {"word": "thirsty", "tags": ["need", "hydration"]},
    {"word": "hungry", "tags": ["need", "nutrition"]},
    {"word": "anxious", "tags": ["emotion", "mental"]},
    {"word": "confused", "tags": ["symptom", "mental"]},
]

ACTION_POOL: list[dict[str, Any]] = [
    {"base": "need help", "third": "needs help", "tags": ["request", "emergency"]},
    {"base": "need water", "third": "needs water", "tags": ["request", "hydration"]},
    {"base": "need medicine", "third": "needs medicine", "tags": ["request", "medical"]},
    {"base": "need rest", "third": "needs rest", "tags": ["request", "comfort"]},
    {"base": "need food", "third": "needs food", "tags": ["request", "nutrition"]},
    {"base": "feel pain", "third": "feels pain", "tags": ["symptom", "pain"]},
    {"base": "feel dizzy", "third": "feels dizzy", "tags": ["symptom", "head"]},
    {"base": "feel sick", "third": "feels sick", "tags": ["symptom", "general"]},
    {"base": "feel better", "third": "feels better", "tags": ["status", "improvement"]},
    {"base": "feel worse", "third": "feels worse", "tags": ["status", "decline"]},
    {"base": "want to sit up", "third": "wants to sit up", "tags": ["request", "mobility"]},
    {"base": "want to lie down", "third": "wants to lie down", "tags": ["request", "comfort"]},
    {"base": "want to walk", "third": "wants to walk", "tags": ["request", "mobility"]},
    {"base": "call nurse", "third": "call nurse", "tags": ["request", "emergency"]},
    {"base": "call doctor", "third": "call doctor", "tags": ["request", "emergency"]},
    {"base": "call family", "third": "call family", "tags": ["request", "social"]},
    {"base": "go bathroom", "third": "go bathroom", "tags": ["request", "hygiene"]},
    {"base": "have question", "third": "has question", "tags": ["communication", "general"]},
    {"base": "feel nauseous", "third": "feels nauseous", "tags": ["symptom", "stomach"]},
    {"base": "can't breathe", "third": "can't breathe", "tags": ["symptom", "emergency"]},
    {"base": "can't sleep", "third": "can't sleep", "tags": ["symptom", "comfort"]},
    {"base": "feel cold", "third": "feels cold", "tags": ["symptom", "temperature"]},
    {"base": "feel hot", "third": "feels hot", "tags": ["symptom", "temperature"]},
    {"base": "am okay", "third": "is okay", "tags": ["status", "general"]},
    {"base": "say thank you", "third": "says thank you", "tags": ["communication", "polite"]},
    {"base": "say goodbye", "third": "says goodbye", "tags": ["communication", "social"]},
]

# ── Scoring Rules ─────────────────────────────────────────────

TAG_AFFINITY: dict[str, dict[str, float]] = {
    "urgently": {"emergency": 1.0, "pain": 0.8, "symptom": 0.5},
    "immediately": {"emergency": 1.0, "pain": 0.7, "symptom": 0.5},
    "now": {"emergency": 0.8, "request": 0.7, "symptom": 0.5},
    "slowly": {"comfort": 0.8, "mobility": 0.7},
    "please": {"request": 0.9, "polite": 0.8, "general": 0.5},
    "still": {"ongoing": 0.9, "symptom": 0.8, "pain": 0.7},
    "suddenly": {"onset": 1.0, "emergency": 0.8, "symptom": 0.7},
    "again": {"repetition": 0.7, "general": 0.5},
    "really": {"intensity": 0.9, "pain": 0.7, "symptom": 0.6},
    "constantly": {"frequency": 0.8, "symptom": 0.7, "pain": 0.6},

    "severe": {"pain": 1.0, "emergency": 0.9, "symptom": 0.7},
    "mild": {"pain": 0.6, "symptom": 0.5},
    "sharp": {"pain": 1.0, "symptom": 0.6},
    "dizzy": {"head": 0.9, "symptom": 0.8},
    "nauseous": {"stomach": 0.9, "symptom": 0.8},
    "better": {"improvement": 1.0, "status": 0.8},
    "worse": {"decline": 1.0, "status": 0.8, "emergency": 0.5},
    "weak": {"symptom": 0.8, "fatigue": 0.7},
    "tired": {"fatigue": 0.9, "comfort": 0.5},
    "cold": {"temperature": 0.9, "comfort": 0.6},
    "hot": {"temperature": 0.9, "comfort": 0.6},
    "scared": {"mental": 0.8, "emotion": 0.7},
    "thirsty": {"hydration": 1.0, "need": 0.8},
    "hungry": {"nutrition": 1.0, "need": 0.8},
}

# Priority weights: higher = shown first by default
FREQUENCY_PRIORITY: dict[str, float] = {
    "I": 1.0, "patient": 0.9, "we": 0.7, "nurse": 0.6, "doctor": 0.6,
    "urgently": 0.9, "now": 0.9, "please": 0.8,
    "severe": 0.9, "dizzy": 0.8, "better": 0.7, "worse": 0.7,
    "need help": 1.0, "need water": 0.9, "feel pain": 0.9,
    "call nurse": 0.8, "need medicine": 0.8,
}


def _score_candidate(
    candidate_tags: list[str],
    context_words: list[str],
    candidate_word: str,
) -> float:
    """Score a candidate word based on tag affinity with prior selections."""
    relevance = 0.0
    for ctx_word in context_words:
        affinity_map = TAG_AFFINITY.get(ctx_word.lower(), {})
        for tag in candidate_tags:
            relevance += affinity_map.get(tag, 0.0)

    freq = FREQUENCY_PRIORITY.get(candidate_word, 0.3)
    return relevance * 0.6 + freq * 0.4


class PhraseEngine:
    """Grammar-first sentence builder: Subject -> Adverb -> Adjective -> Action."""

    def __init__(self) -> None:
        self._current_step = GrammarStep.SUBJECT
        self._selected: dict[str, str] = {}
        self._subject_meta: dict[str, Any] | None = None
        self._current_words: list[str] = []
        self._shown_page: dict[str, int] = {}

    @property
    def current_step(self) -> GrammarStep:
        return self._current_step

    @property
    def current_step_name(self) -> str:
        return GRAMMAR_STEPS[self._current_step]

    @property
    def step_index(self) -> int:
        return int(self._current_step)

    @property
    def is_skippable(self) -> bool:
        return self.current_step_name in SKIPPABLE_STEPS

    @property
    def selected_slots(self) -> dict[str, str]:
        return dict(self._selected)

    def has_selections(self) -> bool:
        return bool(self._selected)

    @property
    def sentence(self) -> list[str]:
        """Return selected words as a list (for backward compat)."""
        parts = []
        for step_name in GRAMMAR_STEPS:
            word = self._selected.get(step_name)
            if word:
                parts.append(word)
        return parts

    @property
    def sentence_text(self) -> str:
        return self.assemble_sentence()

    @property
    def history(self) -> list[str]:
        return self.sentence

    def get_words_for_step(self, n: int = 6) -> list[str]:
        """Return n words for the current grammar step, scored by context."""
        step = self._current_step
        context_words = list(self._selected.values())
        page = self._shown_page.get(self.current_step_name, 0)

        if step == GrammarStep.SUBJECT:
            pool = SUBJECT_POOL
            candidates = [(entry["word"], _score_candidate([], context_words, entry["word"])) for entry in pool]
        elif step == GrammarStep.ADVERB:
            pool = ADVERB_POOL
            candidates = [
                (entry["word"], _score_candidate(entry.get("tags", []), context_words, entry["word"]))
                for entry in pool
            ]
        elif step == GrammarStep.ADJECTIVE:
            pool = ADJECTIVE_POOL
            candidates = [
                (entry["word"], _score_candidate(entry.get("tags", []), context_words, entry["word"]))
                for entry in pool
            ]
        elif step == GrammarStep.ACTION:
            pool = ACTION_POOL
            use_third = self._subject_meta is not None and self._subject_meta.get("person") == 3
            candidates = []
            for entry in pool:
                word = entry["third"] if use_third else entry["base"]
                score = _score_candidate(entry.get("tags", []), context_words, entry.get("base", word))
                candidates.append((word, score))
        else:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)

        start = page * n
        if start >= len(candidates):
            start = 0
            self._shown_page[self.current_step_name] = 0

        selected = candidates[start : start + n]
        words = [w for w, _ in selected]

        self._current_words = words
        return words

    def get_other_words(self, n: int = 6) -> list[str]:
        """Get the next page of words for the current step."""
        step_name = self.current_step_name
        self._shown_page[step_name] = self._shown_page.get(step_name, 0) + 1
        return self.get_words_for_step(n)

    def select_word_for_step(self, word: str) -> GrammarStep:
        """Record the word for the current step and advance. Returns the new step."""
        step_name = self.current_step_name
        self._selected[step_name] = word
        logger.info("Grammar slot '%s' = '%s'", step_name, word)

        if self._current_step == GrammarStep.SUBJECT:
            for entry in SUBJECT_POOL:
                if entry["word"].lower() == word.lower():
                    self._subject_meta = entry
                    break

        self._shown_page = {}
        return self._advance_step()

    def skip_step(self) -> GrammarStep:
        """Skip the current step (only valid for Adverb/Adjective). Returns new step."""
        if self.current_step_name not in SKIPPABLE_STEPS:
            logger.warning("Cannot skip step '%s'", self.current_step_name)
            return self._current_step
        logger.info("Skipped grammar step '%s'", self.current_step_name)
        self._shown_page = {}
        return self._advance_step()

    def _advance_step(self) -> GrammarStep:
        next_val = int(self._current_step) + 1
        if next_val > GrammarStep.ACTION:
            return self._current_step
        self._current_step = GrammarStep(next_val)
        return self._current_step

    def is_sentence_complete(self) -> bool:
        """True when Subject + Action are both selected (minimum requirement)."""
        return "subject" in self._selected and "action" in self._selected

    def assemble_sentence(self) -> str:
        """Build the final sentence with grammar agreement."""
        parts: list[str] = []
        subj = self._selected.get("subject", "")
        if subj:
            parts.append(subj.capitalize() if subj[0].islower() else subj)

        adverb = self._selected.get("adverb", "")
        if adverb:
            parts.append(adverb)

        adj = self._selected.get("adjective", "")
        action = self._selected.get("action", "")

        if adj and action:
            if "feel" in action.lower():
                parts.append(action)
                parts.append("and")
                parts.append(adj)
            else:
                parts.append(action)
        elif adj and not action:
            parts.append(f"feel{'s' if self._is_third_person() else ''}")
            parts.append(adj)
        elif action:
            parts.append(action)

        text = " ".join(parts)
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return text

    def _is_third_person(self) -> bool:
        if self._subject_meta is None:
            return False
        return self._subject_meta.get("person") == 3

    def clear_sentence(self) -> None:
        """Clear all selections and reset to Subject step."""
        self._selected.clear()
        self._subject_meta = None
        self._current_step = GrammarStep.SUBJECT
        self._current_words.clear()
        self._shown_page.clear()

    def done_send(self) -> str:
        """Assemble the sentence, clear state, return the text."""
        text = self.assemble_sentence()
        self.clear_sentence()
        return text

    def reset_session(self) -> None:
        """Full session reset."""
        self.clear_sentence()

    # ── Backward-compatible aliases ───────────────────────────

    def select_word(self, word: str) -> None:
        self.select_word_for_step(word)

    def confirm_phrase(self, phrase: str) -> None:
        self.select_word(phrase)

    def undo_last_word(self) -> str | None:
        if not self._selected:
            return None
        for step_name in reversed(GRAMMAR_STEPS):
            if step_name in self._selected:
                removed = self._selected.pop(step_name)
                self._current_step = GrammarStep(GRAMMAR_STEPS.index(step_name))
                if step_name == "subject":
                    self._subject_meta = None
                return removed
        return None

    def delete_last(self) -> str | None:
        return self.undo_last_word()

    def clear_history(self) -> None:
        self.clear_sentence()

    def get_current_words(self) -> list[str]:
        return list(self._current_words)

    async def generate_words(self, n: int = 6) -> list[str]:
        """Async compat wrapper around get_words_for_step."""
        return self.get_words_for_step(n)

    async def generate_other_words(self, n: int = 6) -> list[str]:
        """Async compat wrapper around get_other_words."""
        return self.get_other_words(n)

    async def generate_phrases(self, n: int = 6) -> list[str]:
        """Returns words + Other/Skip labels."""
        words = self.get_words_for_step(n)
        labels = [OTHER_LABEL]
        if self.is_skippable:
            labels.insert(0, SKIP_LABEL)
        return words + labels


phrase_engine = PhraseEngine()
