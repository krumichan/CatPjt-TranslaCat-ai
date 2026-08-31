from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
_HTML_LIKE_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_UNDERLINE_MARKER_RE = re.compile(r"(<u>|</u>)", re.IGNORECASE)
_UNDERLINE_TARGET_RE = re.compile(r"<u>([\s\S]*?)</u>", re.IGNORECASE)
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_OPTION_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_-]{0,15}$")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*\n][\s\S]*?)\*\*")


@dataclass(frozen=True)
class LevelTestGenerationNormalizationStats:
    prompt_text_fallbacks: int = 0
    task_archetype_compactions: int = 0
    semantic_summary_compactions: int = 0
    max_audio_seconds_adjustments: int = 0
    max_answer_length_adjustments: int = 0
    answer_language_adjustments: int = 0
    prompt_markup_sanitizations: int = 0
    instruction_language_repairs: int = 0
    sentence_order_shuffles: int = 0
    option_key_canonicalizations: int = 0
    listening_source_alias_repairs: int = 0
    reference_payload_shape_repairs: int = 0
    reading_structure_repairs: int = 0
    listening_structure_repairs: int = 0
    vocab_emphasis_repairs: int = 0

    @property
    def total(self) -> int:
        return (
            self.prompt_text_fallbacks
            + self.task_archetype_compactions
            + self.semantic_summary_compactions
            + self.max_audio_seconds_adjustments
            + self.max_answer_length_adjustments
            + self.answer_language_adjustments
            + self.prompt_markup_sanitizations
            + self.instruction_language_repairs
            + self.sentence_order_shuffles
            + self.option_key_canonicalizations
            + self.listening_source_alias_repairs
            + self.reference_payload_shape_repairs
            + self.reading_structure_repairs
            + self.listening_structure_repairs
            + self.vocab_emphasis_repairs
        )


class LevelTestGenerationNormalizer:
    """Normalize only provider-output defects that can be repaired without changing the answer semantics."""

    TASK_ARCHETYPE_MAX_LENGTH = 100
    SEMANTIC_SUMMARY_MAX_LENGTH = 500
    MAX_AUDIO_SECONDS = 60
    MAX_ANSWER_LENGTH = 10_000

    @classmethod
    def normalize(
        cls,
        data: dict[str, object],
        *,
        origin_language: str | None = None,
        learning_language: str | None = None,
    ) -> tuple[dict[str, object], LevelTestGenerationNormalizationStats]:
        normalized = copy.deepcopy(data)
        candidates = normalized.get("candidates")
        if not isinstance(candidates, list):
            return normalized, LevelTestGenerationNormalizationStats()

        counts = {
            "prompt_text_fallbacks": 0,
            "task_archetype_compactions": 0,
            "semantic_summary_compactions": 0,
            "max_audio_seconds_adjustments": 0,
            "max_answer_length_adjustments": 0,
            "answer_language_adjustments": 0,
            "prompt_markup_sanitizations": 0,
            "instruction_language_repairs": 0,
            "sentence_order_shuffles": 0,
            "option_key_canonicalizations": 0,
            "listening_source_alias_repairs": 0,
            "reference_payload_shape_repairs": 0,
            "reading_structure_repairs": 0,
            "listening_structure_repairs": 0,
            "vocab_emphasis_repairs": 0,
        }

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            cls._normalize_candidate(
                candidate,
                counts,
                origin_language=origin_language,
                learning_language=learning_language,
            )

        return normalized, LevelTestGenerationNormalizationStats(**counts)

    @classmethod
    def _normalize_candidate(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
        *,
        origin_language: str | None,
        learning_language: str | None,
    ) -> None:
        cls._normalize_vocab_paraphrase_emphasis(candidate, counts)

        prompt_key = cls._existing_key(candidate, "promptText", "prompt_text") or "promptText"
        prompt_text = candidate.get(prompt_key)
        if not cls._has_text(prompt_text):
            instruction = cls._value(candidate, "instruction")
            if cls._has_text(instruction):
                candidate[prompt_key] = cls._compact_whitespace(str(instruction))
                counts["prompt_text_fallbacks"] += 1

        prompt_text = candidate.get(prompt_key)
        if cls._has_text(prompt_text):
            original_prompt = str(prompt_text)
            sanitized_prompt = cls._sanitize_prompt_markup(
                original_prompt,
                item_type=cls._value(candidate, "itemType", "item_type"),
            )
            candidate[prompt_key] = sanitized_prompt
            if sanitized_prompt != original_prompt:
                counts["prompt_markup_sanitizations"] += 1

        cls._normalize_instruction_language(
            candidate,
            origin_language=origin_language,
            counts=counts,
        )
        cls._normalize_listening_source_alias(candidate, counts)
        cls._normalize_reference_payload_shape(candidate, counts)
        cls._normalize_listening_structure(candidate, counts)
        cls._normalize_reading_structure(candidate, counts)
        cls._normalize_reading_emphasis(candidate, counts)
        cls._normalize_option_keys(candidate, counts)
        cls._shuffle_sentence_order_if_needed(candidate, counts)

        diversity = cls._value(candidate, "diversityMetadata", "diversity_metadata")
        if isinstance(diversity, dict):
            cls._normalize_diversity_metadata(diversity, counts)

        answer_mode = cls._value(candidate, "answerMode", "answer_mode")
        cls._normalize_answer_language(
            candidate,
            origin_language=origin_language,
            learning_language=learning_language,
            counts=counts,
        )
        cls._normalize_max_audio_seconds(candidate, answer_mode, counts)
        cls._normalize_max_answer_length(candidate, answer_mode, counts)

    @classmethod
    def _normalize_option_keys(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Repair provider option labels without changing answer semantics.

        Structured providers occasionally emit labels such as ``1``/``2`` or
        lower-case letters even though the public schema requires compact upper-case
        identifiers.  The option text is still usable, so canonicalize the labels and
        update every internal answer reference in the same operation.

        Duplicate/missing labels are intentionally *not* repaired because there is no
        deterministic way to know which duplicate the answer key intended.
        """

        options_key = cls._existing_key(candidate, "options") or "options"
        options = candidate.get(options_key)
        if not isinstance(options, list) or not options:
            return
        if any(not isinstance(option, dict) for option in options):
            return

        original_keys = [cls._value(option, "key") for option in options]
        if any(not isinstance(key, str) or not key.strip() for key in original_keys):
            return
        keys = [str(key).strip() for key in original_keys]
        if len(set(keys)) != len(keys):
            return

        item_type = cls._value(candidate, "itemType", "item_type")
        schema_valid = all(_OPTION_KEY_RE.fullmatch(key) for key in keys)
        standard_choice_valid = True
        if item_type != "GRAMMAR_SENTENCE_ORDER" and len(keys) == 4:
            standard_choice_valid = set(keys) == {"A", "B", "C", "D"}
        if schema_valid and standard_choice_valid:
            return

        canonical_keys = [cls._canonical_option_key(index) for index in range(len(keys))]
        mapping = dict(zip(keys, canonical_keys, strict=True))

        changed = 0
        for option, original, canonical in zip(options, keys, canonical_keys, strict=True):
            key_name = cls._existing_key(option, "key") or "key"
            if original != canonical:
                option[key_name] = canonical
                changed += 1

        answer_key = cls._value(candidate, "internalAnswerKey", "internal_answer_key")
        if isinstance(answer_key, dict):
            correct_key_name = cls._existing_key(
                answer_key,
                "correctOptionKey",
                "correct_option_key",
            )
            if correct_key_name is not None:
                correct_key = answer_key.get(correct_key_name)
                if isinstance(correct_key, str) and correct_key.strip() in mapping:
                    answer_key[correct_key_name] = mapping[correct_key.strip()]

            order_key = cls._existing_key(answer_key, "correctOrder", "correct_order")
            if order_key is not None and isinstance(answer_key.get(order_key), list):
                order = answer_key[order_key]
                if all(isinstance(key, str) and key.strip() in mapping for key in order):
                    answer_key[order_key] = [mapping[key.strip()] for key in order]

        audit = cls._value(candidate, "choiceQualityAudit", "choice_quality_audit")
        if isinstance(audit, dict):
            compatible_key = cls._existing_key(
                audit,
                "directlyCompatibleOptionKeys",
                "directly_compatible_option_keys",
            )
            if compatible_key is not None and isinstance(audit.get(compatible_key), list):
                compatible = audit[compatible_key]
                if all(isinstance(key, str) and key.strip() in mapping for key in compatible):
                    audit[compatible_key] = [mapping[key.strip()] for key in compatible]

        counts["option_key_canonicalizations"] += changed

    @staticmethod
    def _canonical_option_key(index: int) -> str:
        if index < 26:
            return chr(ord("A") + index)
        return f"A{index - 25}"

    @classmethod
    def _shuffle_sentence_order_if_needed(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        if cls._value(candidate, "itemType", "item_type") != "GRAMMAR_SENTENCE_ORDER":
            return
        options_key = cls._existing_key(candidate, "options") or "options"
        options = candidate.get(options_key)
        answer_key = cls._value(candidate, "internalAnswerKey", "internal_answer_key")
        if not isinstance(options, list) or len(options) < 2 or not isinstance(answer_key, dict):
            return
        order = cls._value(answer_key, "correctOrder", "correct_order")
        if not isinstance(order, list) or len(order) != len(options):
            return
        option_keys = [
            cls._value(option, "key") if isinstance(option, dict) else None
            for option in options
        ]
        if option_keys != order:
            return

        # Stable, semantics-preserving repair: rotate once so storage/display order
        # can never be identical to the answer order. correctOrder keys stay unchanged.
        candidate[options_key] = options[1:] + options[:1]
        counts["sentence_order_shuffles"] += 1


    @classmethod
    def _normalize_listening_source_alias(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Canonicalize known provider aliases for the Listening audio script.

        This repair is intentionally conservative: it only moves an explicit provider
        field that already contains the script.  It never fabricates sourceText from
        promptText or instruction because those are learner-facing text with different
        semantics.
        """

        item_type = cls._value(candidate, "itemType", "item_type")
        if not isinstance(item_type, str) or not item_type.startswith("LISTENING_"):
            return

        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return

        canonical = payload.get("sourceText")
        if cls._has_text(canonical):
            return

        aliases = (
            "source_text",
            "audioText",
            "audio_text",
            "transcript",
            "script",
            "referenceText",
            "reference_text",
            "listeningText",
            "listening_text",
        )

        source_value: str | None = None
        nested_alias: str | None = None
        for alias in aliases:
            value = payload.get(alias)
            if cls._has_text(value):
                source_value = str(value).strip()
                nested_alias = alias
                break

        top_level_alias: str | None = None
        if source_value is None:
            for alias in ("sourceText", *aliases):
                value = candidate.get(alias)
                if cls._has_text(value):
                    source_value = str(value).strip()
                    top_level_alias = alias
                    break

        if source_value is None:
            return

        if payload_key is None:
            payload_key = "referencePayload"
            candidate[payload_key] = payload
        payload["sourceText"] = source_value
        if nested_alias is not None and nested_alias != "sourceText":
            payload.pop(nested_alias, None)
        if top_level_alias is not None:
            candidate.pop(top_level_alias, None)
        counts["listening_source_alias_repairs"] += 1

    @classmethod
    def _normalize_reference_payload_shape(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Make the provider reference object deterministic before schema validation.

        This does not invent semantic content. It only canonicalizes known field names
        and fills neutral defaults so the typed provider schema can require a stable
        object shape for every candidate. Domain validators remain responsible for
        requiring sourceText/referenceText/reference meanings when applicable.
        """

        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        changed = False

        if payload is None:
            payload_key = payload_key or "referencePayload"
            payload = {}
            candidate[payload_key] = payload
            changed = True
        if not isinstance(payload, dict):
            return

        aliases = {
            "source_text": "sourceText",
            "reference_meanings": "referenceMeanings",
            "key_meaning_units": "keyMeaningUnits",
            "reference_text": "referenceText",
            "translation_source_text": "translationSourceText",
            "emphasis_text": "emphasisText",
            "reading_passage": "readingPassage",
            "reading_question": "readingQuestion",
            "listening_question": "listeningQuestion",
            "provided_facts": "providedFacts",
            "required_intents": "requiredIntents",
            "response_constraints": "responseConstraints",
        }
        for alias, canonical in aliases.items():
            if canonical not in payload and alias in payload:
                payload[canonical] = payload.pop(alias)
                changed = True

        defaults = {
            "sourceText": None,
            "referenceMeanings": [],
            "keyMeaningUnits": [],
            "referenceText": None,
            "translationSourceText": None,
            "emphasisText": None,
            "readingPassage": None,
            "readingQuestion": None,
            "listeningQuestion": None,
            "providedFacts": [],
            "requiredIntents": [],
            "responseConstraints": [],
        }
        for key, default in defaults.items():
            if key not in payload:
                payload[key] = copy.deepcopy(default)
                changed = True

        if changed:
            counts["reference_payload_shape_repairs"] += 1


    @classmethod
    def _normalize_listening_structure(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Canonicalize the learner-visible Listening question separately from audio text.

        sourceText is audio-only evidence. listeningQuestion is the only text that may
        become promptText. For compatibility with provider outputs that omit the new
        field, an existing promptText can be copied into listeningQuestion without
        inventing content; the semantic validator later rejects any script leakage.
        """

        item_type = cls._value(candidate, "itemType", "item_type")
        if not isinstance(item_type, str) or not item_type.startswith("LISTENING_"):
            return

        prompt_key = cls._existing_key(candidate, "promptText", "prompt_text") or "promptText"
        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        if not isinstance(payload, dict):
            return

        changed = False
        question = payload.get("listeningQuestion")
        prompt = candidate.get(prompt_key)

        if not cls._has_text(question) and cls._has_text(prompt):
            payload["listeningQuestion"] = str(prompt).strip()
            question = payload["listeningQuestion"]
            changed = True

        if cls._has_text(question):
            question_text = cls._sanitize_prompt_markup(
                str(question).strip(),
                item_type=item_type,
            ).strip()
            if payload.get("listeningQuestion") != question_text:
                payload["listeningQuestion"] = question_text
                changed = True
            if candidate.get(prompt_key) != question_text:
                candidate[prompt_key] = question_text
                changed = True

        if changed:
            counts["listening_structure_repairs"] += 1


    @classmethod
    def _normalize_reading_structure(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Canonicalize Reading passage/question into structured reference fields.

        The model is asked to return passage and question separately.  The server then
        assembles learner-visible promptText deterministically.  For legacy/provider
        outputs that still return only promptText, a blank-line split may recover the
        same text without inventing any semantic content.
        """

        item_type = cls._value(candidate, "itemType", "item_type")
        if not isinstance(item_type, str) or not item_type.startswith("READING_"):
            return

        prompt_key = cls._existing_key(candidate, "promptText", "prompt_text") or "promptText"
        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        if not isinstance(payload, dict):
            return

        changed = False
        passage = payload.get("readingPassage")
        question = payload.get("readingQuestion")
        prompt = candidate.get(prompt_key)

        if (not cls._has_text(passage) or not cls._has_text(question)) and cls._has_text(prompt):
            blocks = [block.strip() for block in re.split(r"\n\s*\n", str(prompt)) if block.strip()]
            recovered_passage: str | None = None
            recovered_question: str | None = None
            if len(blocks) >= 2:
                recovered_passage = "\n\n".join(blocks[:-1])
                recovered_question = blocks[-1]
            else:
                recovered_passage, recovered_question = cls._split_legacy_reading_prompt(str(prompt))
            if not cls._has_text(passage) and cls._has_text(recovered_passage):
                payload["readingPassage"] = recovered_passage
                passage = payload["readingPassage"]
                changed = True
            if not cls._has_text(question) and cls._has_text(recovered_question):
                payload["readingQuestion"] = recovered_question
                question = payload["readingQuestion"]
                changed = True

        if cls._has_text(passage) and cls._has_text(question):
            passage_text = str(passage).strip()
            question_text = str(question).strip()
            if item_type == "READING_DISCOURSE_FUNCTION":
                matches = list(_MARKDOWN_BOLD_RE.finditer(passage_text))
                if not cls._has_text(payload.get("emphasisText")) and len(matches) == 1:
                    payload["emphasisText"] = matches[0].group(1).strip()
                    changed = True
                cleaned_passage = _MARKDOWN_BOLD_RE.sub(lambda match: match.group(1), passage_text)
                if cleaned_passage != passage_text:
                    passage_text = cleaned_passage
                    payload["readingPassage"] = passage_text
                    counts["prompt_markup_sanitizations"] += 1
                    changed = True

            passage_text = cls._sanitize_prompt_markup(passage_text, item_type=item_type).strip()
            question_text = cls._sanitize_prompt_markup(question_text, item_type=item_type).strip()
            if payload.get("readingPassage") != passage_text:
                payload["readingPassage"] = passage_text
                changed = True
            if payload.get("readingQuestion") != question_text:
                payload["readingQuestion"] = question_text
                changed = True

            canonical_prompt = f"{passage_text}\n\n{question_text}"
            if candidate.get(prompt_key) != canonical_prompt:
                candidate[prompt_key] = canonical_prompt
                changed = True

        if changed:
            counts["reading_structure_repairs"] += 1


    @staticmethod
    def _split_legacy_reading_prompt(value: str) -> tuple[str | None, str | None]:
        """Recover a final learner-visible question from legacy one-block Reading text.

        This only splits text that already contains an explicit interrogative ending; it
        never invents or paraphrases a question.  New generation uses structured fields
        and should not depend on this compatibility path.
        """

        text = value.strip()
        if not text:
            return None, None
        boundaries = [match.end() for match in re.finditer(r"[。.!！]", text)]
        for boundary in reversed(boundaries[:-1] if boundaries and boundaries[-1] == len(text) else boundaries):
            passage = text[:boundary].strip()
            question = text[boundary:].strip()
            if not passage or len(question) < 6:
                continue
            if question.endswith(("?", "？")) or question.endswith(("か。", "か！")):
                return passage, question
        return None, None


    @classmethod
    def _normalize_reading_emphasis(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Canonicalize a legacy Markdown-bold discourse target into structured metadata.

        Reading emphasis is presentation-critical evidence.  We keep promptText plain and
        expose the exact target separately so the client can underline/highlight it
        deterministically.  This repair only extracts text already present in the prompt.
        """

        item_type = cls._value(candidate, "itemType", "item_type")
        if item_type != "READING_DISCOURSE_FUNCTION":
            return
        prompt_key = cls._existing_key(candidate, "promptText", "prompt_text") or "promptText"
        prompt = candidate.get(prompt_key)
        if not cls._has_text(prompt):
            return

        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        if not isinstance(payload, dict):
            return

        emphasis = payload.get("emphasisText")
        value = str(prompt)
        matches = list(_MARKDOWN_BOLD_RE.finditer(value))
        if not cls._has_text(emphasis) and len(matches) == 1:
            payload["emphasisText"] = matches[0].group(1).strip()
            counts["reference_payload_shape_repairs"] += 1

        if matches:
            cleaned = _MARKDOWN_BOLD_RE.sub(lambda match: match.group(1), value)
            if cleaned != value:
                candidate[prompt_key] = cleaned
                counts["prompt_markup_sanitizations"] += 1


    @classmethod
    def _normalize_instruction_language(
        cls,
        candidate: dict[str, Any],
        *,
        origin_language: str | None,
        counts: dict[str, int],
    ) -> None:
        domain = cls._value(candidate, "domain")
        if domain != "READING" or not origin_language:
            return
        key = cls._existing_key(candidate, "instruction") or "instruction"
        instruction = candidate.get(key)
        if not cls._has_text(instruction):
            return
        value = str(instruction)
        language = origin_language.lower()
        mismatch = (
            language == "ko" and bool(_KANA_RE.search(value))
            or language == "ja" and bool(_HANGUL_RE.search(value))
            or language == "en" and bool(_KANA_RE.search(value) or _HANGUL_RE.search(value))
        )
        if not mismatch:
            return
        fallback = {
            "ko": "다음 글을 읽고 가장 적절한 답을 선택하세요.",
            "ja": "次の文章を読み、最も適切な答えを選んでください。",
            "en": "Read the passage and choose the best answer.",
        }.get(language)
        if fallback is None:
            return
        candidate[key] = fallback
        counts["instruction_language_repairs"] += 1

    @classmethod
    def _normalize_diversity_metadata(
        cls,
        metadata: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        task_key = cls._existing_key(metadata, "taskArchetype", "task_archetype")
        if task_key is not None and cls._has_text(metadata.get(task_key)):
            original = str(metadata[task_key])
            normalized = cls._compact_and_limit(original, cls.TASK_ARCHETYPE_MAX_LENGTH)
            metadata[task_key] = normalized
            if normalized != original:
                counts["task_archetype_compactions"] += 1

        summary_key = cls._existing_key(metadata, "semanticSummary", "semantic_summary")
        if summary_key is not None and cls._has_text(metadata.get(summary_key)):
            original = str(metadata[summary_key])
            normalized = cls._compact_and_limit(original, cls.SEMANTIC_SUMMARY_MAX_LENGTH)
            metadata[summary_key] = normalized
            if normalized != original:
                counts["semantic_summary_compactions"] += 1

        for aliases in (("contentHash", "content_hash"), ("similarityKey", "similarity_key")):
            key = cls._existing_key(metadata, *aliases)
            if key is not None and isinstance(metadata.get(key), str):
                value = metadata[key].strip()
                metadata[key] = value or None

        for aliases in (("grammarFocusCodes", "grammar_focus_codes"), ("lexicalFocusCodes", "lexical_focus_codes")):
            key = cls._existing_key(metadata, *aliases)
            if key is None or not isinstance(metadata.get(key), list):
                continue
            metadata[key] = [
                cls._compact_whitespace(value)
                for value in metadata[key][:20]
                if isinstance(value, str) and value.strip()
            ]


    @classmethod
    def _normalize_answer_language(
        cls,
        candidate: dict[str, Any],
        *,
        origin_language: str | None,
        learning_language: str | None,
        counts: dict[str, int],
    ) -> None:
        item_type = cls._value(candidate, "itemType", "item_type")
        if not isinstance(item_type, str):
            return

        expected: str | None | object = ...
        if item_type == "LISTENING_INTERPRETATION":
            expected = origin_language
        elif item_type == "LISTENING_DICTATION":
            expected = learning_language
        elif item_type.startswith("WRITING_") or item_type.startswith("SPEAKING_"):
            expected = learning_language
        elif item_type in {
            "VOCAB_CONTEXT_CHOICE",
            "VOCAB_PARAPHRASE_CHOICE",
            "GRAMMAR_FORM_CHOICE",
            "GRAMMAR_SENTENCE_ORDER",
            "READING_GIST",
            "READING_DETAIL",
            "READING_DISCOURSE_FUNCTION",
            "READING_TEXT_INFERENCE",
            "LISTENING_GIST_CHOICE",
            "LISTENING_DETAIL_CHOICE",
        }:
            expected = None
        else:
            return

        if expected is ...:
            return
        if expected is None or isinstance(expected, str) and expected.strip():
            key = cls._existing_key(candidate, "answerLanguage", "answer_language") or "answerLanguage"
            if candidate.get(key) != expected:
                candidate[key] = expected
                counts["answer_language_adjustments"] += 1

    @classmethod
    def _normalize_max_audio_seconds(
        cls,
        candidate: dict[str, Any],
        answer_mode: object,
        counts: dict[str, int],
    ) -> None:
        key = cls._existing_key(candidate, "maxAudioSeconds", "max_audio_seconds")
        if key is None:
            return

        original = candidate.get(key)
        if answer_mode != "AUDIO":
            if original is not None:
                candidate[key] = None
                counts["max_audio_seconds_adjustments"] += 1
            return

        bounded = cls._bounded_int(original, minimum=1, maximum=cls.MAX_AUDIO_SECONDS)
        if bounded != original:
            candidate[key] = bounded
            counts["max_audio_seconds_adjustments"] += 1

    @classmethod
    def _normalize_max_answer_length(
        cls,
        candidate: dict[str, Any],
        answer_mode: object,
        counts: dict[str, int],
    ) -> None:
        key = cls._existing_key(candidate, "maxAnswerLength", "max_answer_length")
        if key is None:
            return

        original = candidate.get(key)
        if answer_mode != "TEXT":
            if original is not None:
                candidate[key] = None
                counts["max_answer_length_adjustments"] += 1
            return

        bounded = cls._bounded_int(original, minimum=1, maximum=cls.MAX_ANSWER_LENGTH)
        if bounded != original:
            candidate[key] = bounded
            counts["max_answer_length_adjustments"] += 1

    @classmethod
    def _normalize_vocab_paraphrase_emphasis(
        cls,
        candidate: dict[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Migrate legacy <u> target markup into referencePayload.emphasisText.

        The target text already exists in provider output, so extracting it is a
        semantics-preserving compatibility repair. New v9 generation is expected to
        return plain promptText plus structured emphasisText directly.
        """

        if cls._value(candidate, "itemType", "item_type") != "VOCAB_PARAPHRASE_CHOICE":
            return
        prompt_key = cls._existing_key(candidate, "promptText", "prompt_text") or "promptText"
        prompt = candidate.get(prompt_key)
        if not cls._has_text(prompt):
            return

        payload_key = cls._existing_key(candidate, "referencePayload", "reference_payload")
        payload = candidate.get(payload_key) if payload_key is not None else None
        if payload is None:
            payload_key = payload_key or "referencePayload"
            payload = {}
            candidate[payload_key] = payload
        if not isinstance(payload, dict):
            return

        value = str(prompt)
        matches = list(_UNDERLINE_TARGET_RE.finditer(value))
        changed = False
        if not cls._has_text(payload.get("emphasisText")) and len(matches) == 1:
            target = matches[0].group(1).strip()
            if target and len(target) <= 80:
                payload["emphasisText"] = target
                counts["vocab_emphasis_repairs"] += 1

        if matches:
            cleaned = _UNDERLINE_TARGET_RE.sub(lambda match: match.group(1), value)
            cleaned = _UNDERLINE_MARKER_RE.sub("", cleaned)
            if cleaned != value:
                candidate[prompt_key] = cleaned
                counts["prompt_markup_sanitizations"] += 1
                changed = True

        if changed and payload_key is None:
            candidate["referencePayload"] = payload

    @classmethod
    def _sanitize_prompt_markup(cls, value: str, *, item_type: object = None) -> str:
        """Return plain learner text only; emphasis is structured metadata in v9.

        ``item_type`` is retained for call-site compatibility. No Level Test prompt
        is allowed to rely on raw HTML/XML presentation markup anymore.
        """

        del item_type
        return _HTML_LIKE_TAG_RE.sub("", value)

    @staticmethod
    def _existing_key(mapping: dict[str, Any], *keys: str) -> str | None:
        return next((key for key in keys if key in mapping), None)

    @staticmethod
    def _value(mapping: dict[str, Any], *keys: str) -> object | None:
        key = LevelTestGenerationNormalizer._existing_key(mapping, *keys)
        return mapping.get(key) if key is not None else None

    @staticmethod
    def _has_text(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _compact_whitespace(value: str) -> str:
        return _WHITESPACE_RE.sub(" ", value).strip()

    @classmethod
    def _compact_and_limit(cls, value: str, max_length: int) -> str:
        compact = cls._compact_whitespace(value)
        if len(compact) <= max_length:
            return compact

        prefix = compact[: max_length + 1]
        word_boundary = prefix.rsplit(" ", 1)[0].rstrip()
        if len(word_boundary) >= max_length // 2:
            return word_boundary
        return compact[:max_length].rstrip()

    @staticmethod
    def _bounded_int(value: object, *, minimum: int, maximum: int) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return max(minimum, min(maximum, value))
        if isinstance(value, float) and value.is_integer():
            return max(minimum, min(maximum, int(value)))
        return value
