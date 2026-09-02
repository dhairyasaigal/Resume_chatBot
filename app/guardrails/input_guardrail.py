"""
Input Guardrail — protects against prompt injection, jailbreaking, toxic queries,
and system secret extraction attempts before any retrieval or LLM execution.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Known prompt injection & jailbreak patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|unrestricted|god\s+mode)",
    r"bypass\s+(your\s+)?(guardrails|safety|guidelines|filters)",
    r"system\s+prompt(\s+reveal|\s+leak|\s+dump|\s+print|\s+show)?",
    r"what\s+is\s+your\s+(original\s+)?system\s+(prompt|message)",
    r"print\s+(the\s+)?(system\s+prompt|initial\s+prompt|hidden\s+prompt)",
    r"reveal\s+(your\s+)?(instructions|prompt|rules)",
    r"(show|print|give|extract)\s+(me\s+)?(the\s+)?(api[_\s-]?key|secret|env|password|token)",
    r"execute\s+command\s*:",
    r"sudo\s+",
    r"<script[\s>]",
]

# Obvious harassment or abusive patterns
TOXIC_PATTERNS = [
    r"\b(kill\s+yourself|die\s+in\s+a\s+fire)\b",
]


def validate_input_guardrail(query: str) -> tuple[bool, str | None]:
    """
    Validate an incoming user query against safety, injection, and security guardrails.

    Returns:
        (is_safe: bool, reason: str | None)
        If is_safe is False, reason describes why it was blocked.
    """
    if not query or not query.strip():
        return False, "Empty query provided."

    cleaned_query = query.strip().lower()

    # 1. Check prompt injection and system prompt extraction
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            logger.warning(f"[Guardrail] Prompt injection detected matching pattern: {pattern}")
            return (
                False,
                "Security Policy Notice: The requested instruction contains patterns that conflict with safety policies (prompt injection / system extraction). Please ask questions about Dhairya's resume, skills, or projects.",
            )

    # 2. Check toxic or abusive keywords
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            logger.warning(f"[Guardrail] Toxic language detected matching pattern: {pattern}")
            return (
                False,
                "Content Policy Notice: Inappropriate language detected. Please keep questions professional and focused on the interview.",
            )

    # 3. Guard against abnormally large malicious payloads (e.g. buffer overflows/DoS)
    if len(query) > 3000:
        logger.warning(f"[Guardrail] Query length exceeded limit: {len(query)}")
        return (
            False,
            "Input Notice: Question is too long (maximum 3000 characters). Please provide a more concise question.",
        )

    return True, None
