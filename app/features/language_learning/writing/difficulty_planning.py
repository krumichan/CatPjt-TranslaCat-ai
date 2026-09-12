"""Server-owned production blueprints and bounded, directional recovery.

The state machine/counts/selection are deterministic. The semantic demands in a
blueprint are instructions, NOT proof of linguistic difficulty. Only independent
review can approve the resulting text; no word count, tag or longer source earns
an automatic higher band. This editorial policy still needs live calibration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from app.features.language_learning.writing.difficulty_spec import WRITING_BAND_RUBRIC
if TYPE_CHECKING:
    from app.features.language_learning.writing.generation_contract import WritingSlot
    from app.schemas.language_learning import DailyWritingGenerationRequest

WRITING_DIFFICULTY_CONTROL_VERSION = "writing-difficulty-control-v1"
ADJACENT_RECHECKS_PER_SLOT = 1
_DIRECTION_THRESHOLD = 2
Direction = Literal["INCREASE_PRODUCTION_DEMAND", "DECREASE_PRODUCTION_DEMAND"]

# Use meaningful contrasts rather than an arbitrary minimum source length. All
# operations must be present in the visible task, never only in focusReason/tags.
_BLUEPRINTS: dict[int, tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = {
    1: (("DIRECT_MESSAGE", (
        ("DIRECT_PROPOSITION", "Express one concrete familiar fact, preference or question directly."),
        ("BASIC_PRODUCTION", "Do not introduce linked exceptions, nuanced stance or hidden obligations."),
    )),),
    2: (("SIMPLE_LINK", (
        ("SINGLE_LINK", "Use one simple reason or sequence around familiar information."),
        ("FAMILIAR_REQUEST", "An ordinary polite request is enough; no indirect stance negotiation."),
    )),),
    3: (("PRACTICAL_EXPLANATION", (
        ("CONNECTED_MESSAGE", "Connect a practical reason or condition to its consequence or requested action."),
        ("DIRECT_SCOPE", "Keep who does what and the condition's scope explicit and straightforward."),
        ("LIMIT_NUANCE", "Avoid stacking concessions, evidence limitations and indirect commitments in one task."),
    )), ("PRACTICAL_SEQUENCE", (
        ("CONNECTED_MESSAGE", "Relate timing or a changed condition to a practical explanation or request."),
        ("DIRECT_SCOPE", "Use familiar facts and suitable everyday/work politeness, without layered qualifications."),
    ))),
    4: (("QUALIFIED_MESSAGE", (
        ("QUALIFIED_RELATION", "Express a concession, a hedge or an indirect request with clear logical scope."),
        ("SUITABLE_REGISTER", "The message requires context-appropriate phrasing but not specialist knowledge."),
    )), ("CONCESSION_REQUEST", (
        ("QUALIFIED_RELATION", "Connect a conceded fact to a carefully qualified practical request or explanation."),
        ("SUITABLE_REGISTER", "Keep the relations interpretable and avoid relying solely on technical words."),
    ))),
    5: (("BOUNDED_COMMITMENT", (
        ("SCOPE_INTERACTION", "Make a condition and an explicit exception interact in limiting a commitment; "
         "a flat list of steps must not preserve the same meaning."),
        ("PRECISE_STANCE", "Distinguish what is confirmed from what can only be promised conditionally; "
         "removing the qualification must change the speaker's responsibility or claim."),
        ("COHERENT_REGISTER", "Express the concession and limited commitment as one coherent, "
         "appropriately restrained message. Supply every fact needed; no expert judgment."),
    )), ("EVIDENCE_AND_SCOPE", (
        ("SCOPE_INTERACTION", "Relate an observed fact to a limited conclusion and a dependent action, "
         "while explicitly preserving an exception or limit on that inference."),
        ("PRECISE_STANCE", "State why the observation does not justify an unrestricted conclusion; "
         "the learner must preserve the narrower claim, not solve a reasoning puzzle."),
        ("COHERENT_REGISTER", "Use familiar content with precise qualification and controlled stance, "
         "rather than difficult technical nouns or a longer ordinary request."),
    )), ("QUALIFIED_CONTRAST", (
        ("SCOPE_INTERACTION", "Contrast two stated positions or obligations whose application depends "
         "on a shared condition and a clearly delimited exception."),
        ("PRECISE_STANCE", "Keep whose commitment applies, when it applies and what remains undecided "
         "linguistically explicit; deleting a relation must change that meaning."),
        ("COHERENT_REGISTER", "Require precise reference and stance across the connected message, "
         "not a correct business strategy, specialist facts or more verbose wording."),
    ))),
}
_MODE_RULES = {
    "TRANSLATION": (
        "Put ALL required meanings in originText in originLanguage as the actual source message, "
        "not directions to the learner. One item may contain several connected sentences when useful. "
        "Keep all guidance arrays empty and never supply a learningLanguage model answer. "
        "The learner only translates: do not ask them to invent facts or choose a strategy."
    ),
    "GUIDED": (
        "Supply every necessary fact in providedFacts and make the intended language relationships "
        "explicit in requiredIntents/responseConstraints. Do not increase specialist reasoning burden "
        "or count metadata entries as completed communicative intents."
    ),
    "FREE": (
        "Ask for these language relationships in the learner's own or explicitly imagined content. "
        "Do not supply mandatory factual claims or turn this into guided translation. "
        "State the communicative demands visibly; sophistication the learner might voluntarily "
        "add does not count as task difficulty. Keep all guidance arrays empty."
    ),
}


def production_blueprint(slot: WritingSlot, generation_attempt: int) -> dict:
    """Fresh payload, selected without generator claims or personal/free-form text."""
    if type(slot.target_band) is not int or slot.target_band not in _BLUEPRINTS:
        raise ValueError("Writing target band must be from 1 to 5")
    if type(generation_attempt) is not int or generation_attempt < 1:
        raise ValueError("Generation attempt must be a positive integer")
    variants = _BLUEPRINTS[slot.target_band]
    name, demands = variants[(generation_attempt - 1) % len(variants)]
    return {
        "version": WRITING_DIFFICULTY_CONTROL_VERSION,
        "blueprintId": f"{slot.writing_type.value}.B{slot.target_band}.{name}",
        "semanticRequirements": [{"code": code, "demand": demand} for code, demand in demands],
        "renderingRule": _MODE_RULES[slot.writing_type.value],
        "productionAnchor": WRITING_BAND_RUBRIC[slot.target_band],
        "enforcement": "Independent review of visible content; not regex, word count or generated tags.",
    }


@dataclass
class DifficultyRecoveryState:
    """One slot, one request. Confirmed final mismatches, not confidence or votes."""
    target_band: int
    observed_bands: Counter[int] = field(default_factory=Counter)
    last_observed_attempt: int = 0
    direction: Direction | None = None
    start_attempt: int | None = None

    def __post_init__(self) -> None:
        if type(self.target_band) is not int or self.target_band not in _BLUEPRINTS:
            raise ValueError("Writing target band must be from 1 to 5")

    def record(self, *, estimated_band: int | None, difficulty_status: str | None,
               reason: str, generation_attempt: int) -> bool:
        if (reason != "VERIFIED_BAND_MISMATCH" or difficulty_status != "ASSESSED"
                or type(estimated_band) is not int or estimated_band not in _BLUEPRINTS
                or estimated_band == self.target_band):
            return False
        self.observed_bands[estimated_band] += 1
        self.last_observed_attempt = generation_attempt
        return True

    def select_after_round(self, generation_attempt: int, attempt_limit: int) -> bool:
        if (self.direction is not None or generation_attempt >= attempt_limit
                or self.last_observed_attempt != generation_attempt):
            return False
        lower = sum(n for band, n in self.observed_bands.items() if band < self.target_band)
        higher = sum(n for band, n in self.observed_bands.items() if band > self.target_band)
        if lower >= _DIRECTION_THRESHOLD and higher == 0:
            self.direction = "INCREASE_PRODUCTION_DEMAND"
        elif higher >= _DIRECTION_THRESHOLD and lower == 0:
            self.direction = "DECREASE_PRODUCTION_DEMAND"
        else:
            return False
        self.start_attempt = generation_attempt + 1
        return True

    def payload(self) -> dict | None:
        if self.direction is None:
            return None
        return {
            "policyVersion": WRITING_DIFFICULTY_CONTROL_VERSION,
            "direction": self.direction,
            "targetBand": self.target_band,
            "observedBandCounts": {str(band): count for band, count in sorted(self.observed_bands.items())},
            "additionalGenerationRounds": 1,
            "instruction": (
                "Use the selected productionBlueprint to change the ACTUAL meaning demands, not the band label, "
                "focusReason, sentence padding or specialist vocabulary. This is the final targeted generation round."
            ),
        }


def recovery_context(request: DailyWritingGenerationRequest) -> dict:
    """Generator-only adjustment; never remove diversity or change stored profiles."""
    return request.model_dump(mode="json", by_alias=True, exclude={
        "sentence_count", "difficulty_distribution", "language_complexity", "learning_profile",
        "recent_mistakes", "recently_learned_expressions", "recent_evaluation_summary",
    })
