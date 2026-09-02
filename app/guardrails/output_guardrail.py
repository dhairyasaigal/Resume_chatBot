"""
Output Guardrail — sanitizes and verifies generated responses to prevent
secret leakage (API keys, connection strings) or unsafe responses.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that must never be emitted in assistant responses
SECRET_PATTERNS = [
    (r"gsk_[a-zA-Z0-9]{30,}", "[REDACTED_API_KEY]"),
    (r"eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}", "[REDACTED_JWT]"),
    (r"postgresql://[^:]+:[^@]+@[^/]+/[^\s]+", "[REDACTED_DATABASE_URL]"),
]


def validate_output_guardrail(text: str) -> str:
    """
    Sanitizes output text, replacing any sensitive data or credential patterns.

    Returns:
        Sanitized output string.
    """
    if not text:
        return text

    sanitized = text
    for pattern, replacement in SECRET_PATTERNS:
        if re.search(pattern, sanitized):
            logger.warning(f"[Guardrail] Sensitive data pattern matched in output and redacted.")
            sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized
