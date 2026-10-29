"""
resources.py
Mental health awareness, education, and coping resource module for MindBridge.
Provides structured psychoeducation, mindfulness exercises, and self-care guidance.
"""

from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class Resource:
    title:       str
    content:     str
    category:    str
    action_type: str   # "read" | "exercise" | "checklist" | "referral"


# ── Resource Library ─────────────────────────────────────────────────────────

RESOURCES: dict[str, list[Resource]] = {

    "breathing": [
        Resource(
            title="4-7-8 Breathing Technique",
            category="mindfulness",
            action_type="exercise",
            content=(
                "**4-7-8 Breathing** — a calming technique based on pranayama yoga:\n\n"
                "1. Sit comfortably and close your eyes\n"
                "2. **Inhale** through your nose for **4 counts**\n"
                "3. **Hold** your breath for **7 counts**\n"
                "4. **Exhale** slowly through your mouth for **8 counts**\n"
                "5. Repeat 3–4 cycles\n\n"
                "_This activates the parasympathetic nervous system, reducing anxiety and stress._"
            ),
        ),
        Resource(
            title="Box Breathing (Navy SEAL Technique)",
            category="mindfulness",
            action_type="exercise",
            content=(
                "**Box Breathing** — used by Navy SEALs to stay calm under pressure:\n\n"
                "1. **Inhale** for **4 counts**\n"
                "2. **Hold** for **4 counts**\n"
                "3. **Exhale** for **4 counts**\n"
                "4. **Hold** for **4 counts**\n"
                "5. Repeat 4–5 times\n\n"
                "_Equalising your breath signals safety to your nervous system._"
            ),
        ),
    ],

    "grounding": [
        Resource(
            title="5-4-3-2-1 Grounding Exercise",
            category="anxiety",
            action_type="exercise",
            content=(
                "**5-4-3-2-1 Grounding** — brings you back to the present moment:\n\n"
                "Notice around you right now:\n"
                "- **5 things you can SEE** 👀\n"
                "- **4 things you can TOUCH** ✋\n"
                "- **3 things you can HEAR** 👂\n"
                "- **2 things you can SMELL** 👃\n"
                "- **1 thing you can TASTE** 👅\n\n"
                "_This technique is highly effective for panic attacks and flashbacks._"
            ),
        ),
        Resource(
            title="Cold Water Grounding",
            category="anxiety",
            action_type="exercise",
            content=(
                "**Cold Water Grounding** — a quick physiological reset:\n\n"
                "1. Run cold water over your wrists and hands for 30 seconds\n"
                "2. Alternatively, hold an ice cube for 10–15 seconds\n"
                "3. Focus entirely on the physical sensation\n\n"
                "_Cold activates the diving reflex, slowing heart rate and calming the nervous system._"
            ),
        ),
    ],

    "self_care": [
        Resource(
            title="Daily Self-Care Checklist",
            category="self_care",
            action_type="checklist",
            content=(
                "**Daily Mental Health Checklist:**\n\n"
                "☐ Did I sleep 7–9 hours?\n"
                "☐ Did I drink enough water today?\n"
                "☐ Did I eat at least one proper meal?\n"
                "☐ Did I move my body for 20+ minutes?\n"
                "☐ Did I connect with someone I care about?\n"
                "☐ Did I do one thing just for me?\n"
                "☐ Did I limit screen time before bed?\n"
                "☐ Did I take my medication (if prescribed)?\n\n"
                "_You don't need to tick all boxes every day. Progress, not perfection._"
            ),
        ),
        Resource(
            title="Journaling for Mental Health",
            category="self_care",
            action_type="exercise",
            content=(
                "**Journaling Prompts** — 5 minutes of writing can shift your perspective:\n\n"
                "1. What am I feeling right now, and where do I feel it in my body?\n"
                "2. What is one thing I am grateful for today (no matter how small)?\n"
                "3. What is one worry I can let go of today?\n"
                "4. What would I tell my best friend if they felt the way I do now?\n"
                "5. What is one small kind act I can do for myself today?\n\n"
                "_Regular journaling reduces cortisol and helps process difficult emotions._"
            ),
        ),
    ],

    "psychoeducation": [
        Resource(
            title="Understanding the Anxiety Cycle",
            category="anxiety",
            action_type="read",
            content=(
                "**The Anxiety Cycle:**\n\n"
                "Anxiety follows a predictable loop: **Trigger → Thought → Physical Sensation → Avoidance**\n\n"
                "Breaking the cycle:\n"
                "- **Trigger**: Identify what sets off anxiety\n"
                "- **Thought**: Challenge catastrophic thinking ('What is the evidence?')\n"
                "- **Sensation**: Use breathing to calm physical symptoms\n"
                "- **Approach** instead of Avoid: Gradual exposure reduces anxiety long-term\n\n"
                "_Understanding anxiety removes its power — it is a false alarm, not a real threat._"
            ),
        ),
        Resource(
            title="The Window of Tolerance",
            category="trauma",
            action_type="read",
            content=(
                "**The Window of Tolerance** — your optimal zone for functioning:\n\n"
                "- **Hyperarousal** (above the window): panic, rage, overwhelm\n"
                "- **Window of Tolerance** (optimal): present, regulated, able to cope\n"
                "- **Hypoarousal** (below the window): numbness, shutdown, depression\n\n"
                "Skills to return to your window:\n"
                "- Hyperarousal → slow breathing, cold water, gentle movement\n"
                "- Hypoarousal → movement, music, social contact, stimulation\n\n"
                "_All healing happens within the window of tolerance._"
            ),
        ),
    ],

    "positive_affirmations": [
        Resource(
            title="Affirmations for Difficult Days",
            category="self_care",
            action_type="read",
            content=(
                "**Reminders for hard days:**\n\n"
                "✦ You are allowed to feel what you are feeling.\n"
                "✦ Asking for help is an act of courage.\n"
                "✦ This feeling is temporary — it will pass.\n"
                "✦ You have survived every hard day so far.\n"
                "✦ Your worth is not measured by your productivity.\n"
                "✦ One small step forward is still progress.\n"
                "✦ You deserve care, rest, and kindness — from yourself and others.\n"
            ),
        ),
    ],
}

# ── Resource recommender ──────────────────────────────────────────────────────

CATEGORY_TO_RESOURCE_TYPES: dict[str, list[str]] = {
    "none":     ["positive_affirmations", "self_care"],
    "low":      ["self_care", "breathing", "psychoeducation"],
    "medium":   ["breathing", "grounding", "self_care", "psychoeducation"],
    "high":     ["grounding", "breathing", "psychoeducation"],
    "critical": ["grounding", "breathing"],   # immediate calming only
}


class ResourceRecommender:
    """Recommends mental health resources based on risk level and context."""

    def get_resources(self, risk_level: str, count: int = 2) -> list[Resource]:
        """Return a list of resources appropriate for the given risk level."""
        resource_types = CATEGORY_TO_RESOURCE_TYPES.get(risk_level, ["self_care"])
        results: list[Resource] = []
        seen_titles: set[str] = set()

        for rtype in resource_types:
            pool = RESOURCES.get(rtype, [])
            if pool:
                chosen = random.choice(pool)
                if chosen.title not in seen_titles:
                    results.append(chosen)
                    seen_titles.add(chosen.title)
            if len(results) >= count:
                break

        return results

    def format_for_prompt(self, risk_level: str, count: int = 1) -> str:
        """Format recommended resources as a string for prompt injection."""
        resources = self.get_resources(risk_level, count)
        if not resources:
            return ""
        parts = ["**Suggested resources to share with the user:**"]
        for r in resources:
            parts.append(f"\n### {r.title}\n{r.content}")
        return "\n".join(parts)

    def get_mindfulness_tip(self) -> str:
        """Return a random quick mindfulness tip."""
        tips = [
            "Take 3 slow, deep breaths right now. Breathe in for 4 counts, out for 6.",
            "Notice 5 things around you right now. Name them silently.",
            "Place one hand on your heart. Feel it beating. You are alive and here.",
            "Tense all your muscles for 5 seconds, then release completely. Notice the relaxation.",
            "Drink a glass of water slowly and mindfully, focusing only on the sensation.",
            "Step outside or open a window. Take 3 breaths of fresh air.",
        ]
        return random.choice(tips)
