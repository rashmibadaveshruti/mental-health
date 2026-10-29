"""
granite_client.py
IBM Granite 3.3-8b-instruct via watsonx.ai — the core LLM client for MindBridge.
Uses the watsonx.ai REST API (ibm-watsonx-ai SDK).
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai SDK ──────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

import requests


# ── Constants ────────────────────────────────────────────────────────────────
REGION_URL_MAP = {
    "us-south": "https://us-south.ml.cloud.ibm.com",
    "eu-de":    "https://eu-de.ml.cloud.ibm.com",
    "eu-gb":    "https://eu-gb.ml.cloud.ibm.com",
    "au-syd":   "https://au-syd.ml.cloud.ibm.com",
    "jp-tok":   "https://jp-tok.ml.cloud.ibm.com",
}

# Safe messaging system prompt — anchored in AFSP/SAMHSA guidelines
SYSTEM_PROMPT = """You are MindBridge, a compassionate and empathetic mental health support companion.
Your role is to:
1. Listen actively and non-judgmentally to the user's feelings and experiences
2. Validate emotions and provide warm, supportive responses
3. Promote mental wellness, coping strategies, and self-care practices
4. Gently encourage professional help when distress signals are detected
5. Always refer to crisis resources when there is any indication of self-harm or suicidal ideation

CRITICAL SAFETY RULES (never violate these):
- NEVER provide methods, means, or step-by-step information related to self-harm or suicide
- NEVER dismiss or minimize expressions of suicidal ideation — treat every mention as serious
- ALWAYS provide crisis helpline numbers when the user expresses thoughts of self-harm
- ALWAYS include the disclaimer that you are not a substitute for professional medical advice
- Use gentle, open-ended questions to encourage the user to keep talking
- Do not make diagnoses — you are a supportive companion, not a clinician

Tone: warm, patient, non-judgmental, calm, and hopeful.
Language: plain, clear, and accessible (avoid clinical jargon unless explaining it).
"""


class GraniteClient:
    """
    Wraps IBM Granite 3.3-8b-instruct via the watsonx.ai Python SDK.
    Falls back to a direct REST call if the SDK is unavailable.
    """

    # Placeholder strings that mean "not configured yet"
    _PLACEHOLDERS = {"", "your_ibm_cloud_api_key_here", "paste_your_api_key_here",
                     "your_watsonx_project_id_here", "paste_your_project_id_here"}

    @staticmethod
    def _is_real(value: str) -> bool:
        """Return True only when value looks like a genuine credential."""
        return bool(value) and value.lower() not in GraniteClient._PLACEHOLDERS

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        # None  → read from env;  ""  → explicit "no credential" (demo mode)
        self.api_key    = api_key    if api_key    is not None else os.getenv("WATSONX_API_KEY", "")
        self.project_id = project_id if project_id is not None else os.getenv("WATSONX_PROJECT_ID", "")
        self.region     = region     if region     is not None else os.getenv("WATSONX_REGION", "us-south")
        self.model_id   = model_id   if model_id   is not None else os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small-curated")
        self.base_url   = REGION_URL_MAP.get(self.region, REGION_URL_MAP["us-south"])

        self._model: Optional[object] = None
        self._iam_token: Optional[str] = None

        # Only attempt SDK/REST when both credentials look real
        if SDK_AVAILABLE and self._is_real(self.api_key) and self._is_real(self.project_id):
            self._init_sdk()
        # Otherwise stay in demo mode — no network calls at all

    # ── SDK initialisation ───────────────────────────────────────────────────
    def _init_sdk(self) -> None:
        """Initialise the ibm-watsonx-ai ModelInference client."""
        try:
            creds = Credentials(
                url=self.base_url,
                api_key=self.api_key,
            )
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=creds,
                project_id=self.project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: 512,
                    GenParams.TEMPERATURE:    0.7,
                    GenParams.TOP_P:          0.9,
                    GenParams.REPETITION_PENALTY: 1.1,
                },
            )
        except Exception as exc:  # pragma: no cover
            print(f"[GraniteClient] SDK init warning: {exc}")
            self._model = None

    # ── IAM token (REST fallback) ────────────────────────────────────────────
    def _get_iam_token(self) -> str:
        """Exchange an IBM Cloud API key for a short-lived IAM bearer token."""
        if self._iam_token:
            return self._iam_token
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        resp.raise_for_status()
        self._iam_token = resp.json()["access_token"]
        return self._iam_token

    def _call_rest(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Direct REST call to the watsonx.ai text generation endpoint."""
        token = self._get_iam_token()
        url   = f"{self.base_url}/ml/v1/text/generation?version=2025-02-11"
        payload = {
            "model_id":   self.model_id,
            "project_id": self.project_id,
            "input":      prompt,
            "parameters": {
                "max_new_tokens":     max_new_tokens,
                "temperature":        0.7,
                "top_p":              0.9,
                "repetition_penalty": 1.1,
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"].strip()

    # ── Public interface ─────────────────────────────────────────────────────
    def chat(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        extra_context: str = "",
        max_new_tokens: int = 512,
    ) -> str:
        """
        Send a user message and get a response from IBM Granite.

        Args:
            user_message:         The latest user input.
            conversation_history: List of {"role": "user"|"assistant", "content": str}
            extra_context:        Additional RAG context injected before the user message.
            max_new_tokens:       Maximum tokens to generate.

        Returns:
            The model's response as a plain string.
        """
        # Build prompt in Granite instruct format
        prompt = self._build_prompt(user_message, conversation_history, extra_context)

        # Prefer SDK path
        if self._model is not None:
            try:
                result = self._model.generate_text(prompt=prompt)
                return result.strip() if isinstance(result, str) else str(result).strip()
            except Exception as exc:
                print(f"[GraniteClient] SDK generate warning: {exc}, falling back to REST")

        # REST fallback — only when both credentials are real (not placeholders)
        if self._is_real(self.api_key) and self._is_real(self.project_id):
            return self._call_rest(prompt, max_new_tokens)

        # Demo fallback — placeholder or missing credentials
        return self._demo_response(user_message)

    def _build_prompt(
        self,
        user_message: str,
        history: list[dict] | None,
        extra_context: str,
    ) -> str:
        """
        Build a Granite chat-instruct prompt.
        Format: <|system|>...<|user|>...<|assistant|>
        """
        parts = [f"<|system|>\n{SYSTEM_PROMPT}\n"]

        if extra_context:
            parts.append(
                f"<|system|>\n[Relevant context to inform your response]:\n{extra_context}\n"
            )

        if history:
            for turn in history[-6:]:  # keep last 6 turns for context window efficiency
                role    = turn.get("role", "user")
                content = turn.get("content", "")
                tag     = "<|user|>" if role == "user" else "<|assistant|>"
                parts.append(f"{tag}\n{content}\n")

        parts.append(f"<|user|>\n{user_message}\n<|assistant|>\n")
        return "".join(parts)

    # ── Demo mode (no API key) ────────────────────────────────────────────────
    def _demo_response(self, user_message: str) -> str:
        """
        Provides a safe, scripted fallback response when no API key is set.
        This lets the UI run in demo mode without IBM Cloud credentials.
        """
        msg_lower = user_message.lower()
        crisis_keywords = ["suicide", "kill myself", "end my life", "don't want to live",
                           "self harm", "hurt myself", "not worth living"]
        distress_keywords = ["hopeless", "worthless", "can't go on", "no point",
                             "anxious", "depressed", "overwhelmed", "crying", "alone"]

        if any(kw in msg_lower for kw in crisis_keywords):
            return (
                "I hear you, and I'm really glad you're reaching out right now. "
                "What you're feeling sounds incredibly painful, and you deserve support. "
                "Please know you are not alone.\n\n"
                "🆘 **Please contact a crisis helpline right now:**\n"
                "- 🇮🇳 iCall: **9152987821**\n"
                "- 🌍 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n"
                "- 🇺🇸 988 Suicide & Crisis Lifeline: **988** (call or text)\n\n"
                "Would you like to tell me more about what's been happening? I'm here to listen.\n\n"
                "_⚕️ Disclaimer: I am an AI companion, not a medical professional. "
                "Please reach out to a qualified mental health professional for clinical support._"
            )

        if any(kw in msg_lower for kw in distress_keywords):
            return (
                "Thank you for trusting me with how you're feeling. "
                "What you're going through sounds really difficult, and those feelings are valid.\n\n"
                "Sometimes when we're overwhelmed, even small steps can help — "
                "like taking a few slow breaths, stepping outside for a moment, or reaching out to someone you trust.\n\n"
                "I'm here with you. Can you tell me more about what's been on your mind lately?\n\n"
                "_⚕️ I'm an AI companion — for professional support, please consult a licensed therapist or counselor._"
            )

        return (
            "Thank you for sharing that with me. I'm here to listen and support you. "
            "How are you feeling right now, in this moment?\n\n"
            "_(Note: MindBridge is running in demo mode. "
            "Configure your IBM Cloud API key in .env to enable full IBM Granite responses.)_"
        )
