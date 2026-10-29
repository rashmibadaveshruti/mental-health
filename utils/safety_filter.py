"""
safety_filter.py
Safe Messaging Guidelines filter for MindBridge.

Pre-processes user input to detect boundary-violating requests (e.g. asking
for methods of self-harm) and post-processes model output to ensure it never
contains harmful content.  Based on AFSP / SAMHSA Safe Messaging Guidelines.
"""

from __future__ import annotations
import re


# ── Boundary-triggering patterns ──────────────────────────────────────────────
# These patterns indicate the user is asking for harmful information that the
# agent must NEVER provide.  The agent redirects to support instead.

HARMFUL_REQUEST_PATTERNS: list[str] = [
    r"\bhow (do|can|to) (i |you |one )?((commit|do|attempt) )?suicid",
    r"\bwhat (pills|medication|drugs?|amount) (to (take|overdose|use)|would kill)",
    r"\bhow (many|much) .{0,30}(to die|to overdose|is lethal|to kill)",
    r"\bbest (way|method|place) (to die|to kill|for suicide)",
    r"\b(hanging|jumping|cutting) (method|technique|how to)",
    r"\blethal dose",
    r"\bsuicid(e|al) method",
    r"\bhow (to )?hurt (myself|yourself) (without|effectively)",
]

_HARMFUL_RE = re.compile(
    "|".join(HARMFUL_REQUEST_PATTERNS),
    re.IGNORECASE,
)

# ── Unsafe output patterns (never allowed in model responses) ─────────────────
# These patterns should never appear in the model's output.

UNSAFE_OUTPUT_PATTERNS: list[str] = [
    r"\b(take|swallow|ingest) \d+ (pills|tablets|mg)",
    r"\blethal (dose|amount|quantity)",
    r"\bcut (deep|the vein|the artery|along the)",
    r"\b(best|easiest|quickest|most effective) (way|method) (to die|to end your life)",
    r"\b(jump from|hang (yourself|from)|overdose on)",
]

_UNSAFE_OUTPUT_RE = re.compile(
    "|".join(UNSAFE_OUTPUT_PATTERNS),
    re.IGNORECASE,
)

# ── Replacement for unsafe output ────────────────────────────────────────────
SAFE_REPLACEMENT = (
    "[MindBridge safety filter: This content was replaced to comply with "
    "safe messaging guidelines. Please contact a crisis helpline for support.]"
)


class SafetyFilter:
    """
    Pre- and post-processing filter implementing Safe Messaging Guidelines.
    """

    def is_harmful_request(self, user_message: str) -> bool:
        """
        Returns True if the user message appears to be requesting
        harmful information (methods of self-harm or suicide).
        """
        return bool(_HARMFUL_RE.search(user_message))

    def sanitize_output(self, model_output: str) -> str:
        """
        Scan model output and replace any unsafe content that slipped through.
        Returns the sanitized string.
        """
        if _UNSAFE_OUTPUT_RE.search(model_output):
            # Replace the specific unsafe sentence(s)
            sanitized = _UNSAFE_OUTPUT_RE.sub(SAFE_REPLACEMENT, model_output)
            return sanitized
        return model_output

    def add_disclaimer(self, text: str) -> str:
        """Append a standard disclaimer to any response if not already present."""
        disclaimer = (
            "\n\n_⚕️ **Disclaimer**: MindBridge is an AI companion and is NOT a substitute "
            "for professional medical advice, diagnosis, or treatment. "
            "If you are in crisis, please contact a helpline or emergency services immediately._"
        )
        if "disclaimer" in text.lower() or "not a substitute" in text.lower():
            return text
        return text + disclaimer
