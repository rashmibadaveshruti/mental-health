"""
crisis_connector.py
Crisis helpline lookup and escalation module for MindBridge.

Connects users to appropriate helplines based on their location,
formats crisis messages, and provides immediate intervention language.
Follows AFSP / SAMHSA Safe Messaging Guidelines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Helpline:
    name:        str
    number:      str
    hours:       str
    description: str
    url:         str = ""
    country:     str = ""


# ── Crisis Message Templates (Safe Messaging Guideline-compliant) ────────────

CRISIS_MESSAGES = {
    "critical": (
        "I can hear that you're in a lot of pain right now, and I'm really "
        "glad you're still here talking with me.\n\n"
        "What you're feeling is a mental health emergency, and you deserve "
        "immediate support from a trained professional.\n\n"
        "🆘 **Please reach out to a crisis line right now:**\n\n"
        "{helplines}\n\n"
        "You are not alone. This pain is real, but it is also temporary. "
        "Help is available, and you deserve to receive it.\n\n"
        "Can you tell me — is there someone you trust who can be with you right now?\n\n"
        "_⚕️ I am an AI companion, not a medical professional. "
        "Please contact emergency services (911 / 112) if you are in immediate danger._"
    ),
    "high": (
        "What you're going through sounds incredibly difficult, and I want "
        "you to know that your feelings are valid and you matter.\n\n"
        "When things feel this overwhelming, talking to a trained mental health "
        "professional can make a real difference. You don't have to carry this alone.\n\n"
        "**Support is available:**\n\n"
        "{helplines}\n\n"
        "I'm here with you. Would you like to talk about what's been making things "
        "so hard lately?\n\n"
        "_⚕️ Disclaimer: MindBridge is not a substitute for professional mental health care._"
    ),
    "general": (
        "If you ever feel like things are getting too hard, please remember "
        "that support is always available:\n\n"
        "{helplines}\n\n"
        "Reaching out for help is one of the bravest things you can do. 💙"
    ),
}

# ── Safe messaging boundary response ─────────────────────────────────────────
BOUNDARY_RESPONSE = (
    "I hear you, and I care about your safety. I'm not able to provide information "
    "about methods of self-harm — but I am here to support you.\n\n"
    "Please reach out to a crisis line where a trained counselor can help:\n\n"
    "{helplines}\n\n"
    "Can you tell me what's been happening that's brought you to this point? "
    "I want to understand and I want to help.\n\n"
    "_⚕️ If you are in immediate danger, please call emergency services: 911 (US) / 112 (EU) / 100 (India)_"
)


# ── Crisis Connector ──────────────────────────────────────────────────────────

class CrisisConnector:
    """
    Manages crisis helpline data and generates safe, empathetic escalation messages.
    """

    def __init__(self, helplines_path: Optional[str] = None):
        hp = helplines_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "helplines.json"
        )
        with open(hp, "r", encoding="utf-8") as fh:
            raw: dict[str, list[dict]] = json.load(fh)

        self._helplines: dict[str, list[Helpline]] = {}
        for country, lines in raw.items():
            self._helplines[country] = [
                Helpline(
                    name=l["name"],
                    number=l.get("number", ""),
                    hours=l.get("hours", ""),
                    description=l.get("description", ""),
                    url=l.get("url", ""),
                    country=country,
                )
                for l in lines
            ]

    # ── Public API ───────────────────────────────────────────────────────────

    def get_crisis_message(
        self,
        risk_level: str = "critical",
        country: str = "india",
    ) -> str:
        """
        Return a Safe Messaging Guidelines-compliant crisis message
        with relevant helpline numbers embedded.
        """
        helpline_block = self._format_helplines(country)
        template = CRISIS_MESSAGES.get(risk_level, CRISIS_MESSAGES["general"])
        return template.format(helplines=helpline_block)

    def get_boundary_response(self, country: str = "india") -> str:
        """
        Return the safe boundary response for harmful-method queries.
        Blocks the harmful content and redirects to support.
        """
        helpline_block = self._format_helplines(country)
        return BOUNDARY_RESPONSE.format(helplines=helpline_block)

    def get_helplines(self, country: str = "india") -> list[Helpline]:
        """Return helpline objects for a given country."""
        result = list(self._helplines.get(country.lower(), []))
        # Always append international fallbacks
        intl = self._helplines.get("international", [])
        for h in intl:
            if h not in result:
                result.append(h)
        return result

    def format_helplines_compact(self, country: str = "india") -> str:
        """Format helplines as a compact single-line string for inline use."""
        lines = self.get_helplines(country)[:3]
        return " | ".join(
            f"{h.name}: **{h.number or h.url}**" for h in lines if h.number or h.url
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _format_helplines(self, country: str) -> str:
        """Format helplines for the given country as a markdown list."""
        helplines = self.get_helplines(country)
        lines = []
        for h in helplines[:5]:   # cap at 5 to avoid overwhelming
            contact = h.number or h.url or "See website"
            flag = self._country_flag(h.country)
            lines.append(f"  {flag} **{h.name}**: {contact} _{h.hours}_")
        return "\n".join(lines)

    @staticmethod
    def _country_flag(country: str) -> str:
        flags = {
            "india":         "🇮🇳",
            "usa":           "🇺🇸",
            "uk":            "🇬🇧",
            "australia":     "🇦🇺",
            "international": "🌍",
        }
        return flags.get(country.lower(), "🌐")
