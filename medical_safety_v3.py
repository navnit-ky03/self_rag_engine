"""Final-answer medical claim guard for the portfolio research demo.

This is a narrow publication safety gate, not a medical text classifier.
It only checks statements tied to NeuroScan and does not edit KB evidence.
"""
from __future__ import annotations

import re

NEUROSCAN_PATTERN = re.compile(
    r"\b(?:neuroscan|neuro\s*scan|brain\s*tumou?r\s+segment|MRI\s+segment)\b",
    re.I,
)
# Avoid blocking explicit disclaimers like "not clinically validated".
UNSUPPORTED_CLINICAL_CLAIMS = (
    re.compile(r"\bclinically\s+(?:useful|validated|proven|approved|ready|reliable)\b", re.I),
    re.compile(r"\bclinical[-\s]grade\b", re.I),
    re.compile(r"\b(?:FDA|CE)[-\s]+(?:approved|cleared|certified)\b", re.I),
    re.compile(r"\b(?:ready|approved|cleared|certified)\s+for\s+(?:clinical|diagnostic|patient)\s+use\b", re.I),
    re.compile(r"\b(?:diagnos(?:es|is|tic)|treats?|prescribes?)\s+(?:patients?|brain\s+tumou?rs?)\b", re.I),
)
_NEGATIONS = re.compile(
    r"\b(?:not|never|without|no|isn't|is not|has not been|cannot|can't|unverified|"
    r"not yet|does not|don't|should not|isn't|lacks?)\b",
    re.I,
)
RESEARCH_ONLY_RESPONSE = (
    "NeuroScan AI is a research-oriented 3D MRI segmentation project using "
    "PyTorch and MONAI. Its documentation does not establish clinical "
    "validation, regulatory approval, diagnostic reliability, or readiness "
    "for patient care. The portfolio can describe its technical pipeline "
    "without making those claims."
)


def _claims_clinical_readiness(sentence: str) -> bool:
    """Conservative detection for unqualified *positive* clinical claims."""
    for pattern in UNSUPPORTED_CLINICAL_CLAIMS:
        for match in pattern.finditer(sentence):
            # A negation close to the claim usually means the user is being
            # told the claim has NOT been established.
            preamble = sentence[max(0, match.start() - 48): match.start()]
            if not _NEGATIONS.search(preamble):
                return True
    return False


def clinical_claims_unverified(answer: str) -> bool:
    """Catch affirmative clinical-readiness claims about NeuroScan only."""
    if not NEUROSCAN_PATTERN.search(answer):
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if _claims_clinical_readiness(sentence):
            return True
    return False


def ensure_safe_medical_claims(answer: str) -> tuple[str, bool]:
    """Fail closed rather than silently publish a risky affirmative claim."""
    if clinical_claims_unverified(answer):
        return RESEARCH_ONLY_RESPONSE, True
    return answer, False
