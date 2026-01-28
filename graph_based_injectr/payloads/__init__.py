"""
Payloads module for AutoInjector.

This module provides injection payloads, encoding functions, and mutation
strategies for prompt injection testing:
    - jailbreaks: Pre-built jailbreak templates
    - encodings: Payload encoding functions
    - mutations: Payload mutation strategies

These components are used by the agent to craft and adapt attack payloads.
"""

from .encodings import (
    encode_base64,
    decode_base64,
    encode_rot13,
    decode_rot13,
    encode_hex,
    decode_hex,
    encode_unicode_escapes,
    encode_reverse,
    inject_whitespace,
    encode_leetspeak,
    encode_mixed_case,
    get_all_encodings,
)
from .jailbreaks import (
    JailbreakCategory,
    JailbreakTemplate,
    JAILBREAK_TEMPLATES,
    get_templates_by_category,
    get_templates_by_effectiveness,
    get_all_templates,
)
from .mutations import (
    add_politeness,
    add_urgency,
    add_authority,
    rephrase_as_question,
    split_into_steps,
    wrap_in_context,
    mutate_payload,
    generate_variations,
    combine_mutations,
)

__all__ = [
    # Jailbreaks
    "JailbreakCategory",
    "JailbreakTemplate",
    "JAILBREAK_TEMPLATES",
    "get_templates_by_category",
    "get_templates_by_effectiveness",
    "get_all_templates",
    # Encodings
    "encode_base64",
    "decode_base64",
    "encode_rot13",
    "decode_rot13",
    "encode_hex",
    "decode_hex",
    "encode_unicode_escapes",
    "encode_reverse",
    "inject_whitespace",
    "encode_leetspeak",
    "encode_mixed_case",
    "get_all_encodings",
    # Mutations
    "add_politeness",
    "add_urgency",
    "add_authority",
    "rephrase_as_question",
    "split_into_steps",
    "wrap_in_context",
    "mutate_payload",
    "generate_variations",
    "combine_mutations",
]
