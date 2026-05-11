"""
model_discovery.py - Dynamic Gemini model discovery

Queries the Gemini API at startup to find the latest available flash and pro models.
Falls back to config defaults if the API call fails.
"""

import re
from typing import List, Optional

GEMINI_FLASH_DEFAULT = "gemini-2.5-flash"
GEMINI_PRO_DEFAULT = "gemini-2.5-pro"


def discover_models(client) -> tuple:
    """
    Query Gemini API for available models, return (flash_model, pro_model).
    Picks the highest version number available for each type.
    Falls back to defaults on any error.
    """
    flash_model = GEMINI_FLASH_DEFAULT
    pro_model = GEMINI_PRO_DEFAULT

    try:
        models = client.models.list()
        flash_candidates: List[tuple] = []
        pro_candidates: List[tuple] = []

        for model in models:
            name = getattr(model, "name", "")
            match = re.match(r"models/gemini-(\d+\.\d+)-(flash|pro)", name)
            if match:
                version = float(match.group(1))
                model_type = match.group(2)
                model_name = name.replace("models/", "")
                if model_type == "flash":
                    flash_candidates.append((version, model_name))
                elif model_type == "pro":
                    pro_candidates.append((version, model_name))

        if flash_candidates:
            flash_candidates.sort(key=lambda x: x[0], reverse=True)
            flash_model = flash_candidates[0][1]

        if pro_candidates:
            pro_candidates.sort(key=lambda x: x[0], reverse=True)
            pro_model = pro_candidates[0][1]

        print("Discovered models: flash={}, pro={}".format(flash_model, pro_model))

    except Exception as e:
        print("Model discovery failed, using defaults: {}".format(e))

    return flash_model, pro_model
