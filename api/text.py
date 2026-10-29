"""All user-facing wording in one place so it stays honest and consistent."""

TARGET_STATEMENT = (
    "The model estimates how likely a person with these answers is to REPORT existing coronary heart disease or "
    "a past heart attack (BRFSS variable _MICHD: ever told by a health professional), based on patterns in the US "
    "CDC BRFSS 2025 survey."
)

DISCLAIMER = (
    "Educational/research prototype. " + TARGET_STATEMENT + " It does NOT predict a future heart attack, does NOT "
    "diagnose any person, and does NOT establish clinical risk. It was built from self-reported US survey data and "
    "may not generalise to people in India. Do not use it to make medical decisions; speak to a qualified health "
    "professional about any health concern."
)

API_DESCRIPTION = (
    "**Educational/research prototype - not a medical device.**\n\n"
    + TARGET_STATEMENT + "\n\n"
    "- It does **not** predict a future heart attack.\n"
    "- It does **not** diagnose a person or establish clinical risk.\n"
    "- It was trained on self-reported **US** survey data and may **not generalise to people in India**.\n"
    "- The decision threshold is fixed (chosen on validation data in Phase 3) and favours catching reported cases over "
    "avoiding false alarms, so many people above the threshold will not have reported heart disease.\n\n"
    "Health answers sent to `/predict` are not stored, and are not written to logs."
)

INTERPRETATION_ABOVE = (
    "The estimated likelihood is at or above this model's fixed decision threshold. This is a statistical flag from a "
    "survey-based research model, not a diagnosis and not a statement about clinical risk. The threshold favours "
    "catching reported cases, so many people flagged will not have reported heart disease."
)
INTERPRETATION_BELOW = (
    "The estimated likelihood is below this model's fixed decision threshold. This is NOT reassurance and NOT a "
    "diagnosis: the model is a survey-based research prototype and can miss people who have reported heart disease."
)
