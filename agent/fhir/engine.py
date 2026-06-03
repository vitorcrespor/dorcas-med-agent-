from copy import deepcopy
import json
import os
from pathlib import Path

from fhirpathpy import evaluate
from fhirpathpy.models import models


BLOCKED_PARTS = ("resolve(", "memberOf(", "trace(", "%")
MAX_FILE_BYTES = 10_000_000
MAX_RESULTS = 10


def load_fhir_log() -> dict:
    path = Path(os.environ["FHIR_LOG_PATH"])

    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("FHIR log is too large.")

    text = path.read_text(encoding="utf-8").strip()

    if text.startswith("{"):
        data = json.loads(text)

        if data.get("resourceType") == "Bundle":
            return data

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": data}],
        }

    resources = [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": item} for item in resources],
    }


def bundle_fhirpath(expression: str) -> str:
    expression = expression.strip()

    if len(expression) > 500:
        raise ValueError("FHIRPath expression is too long.")

    if not expression.startswith("Bundle.entry.resource"):
        raise ValueError("Query must select resources from the local Bundle.")

    if any(part in expression for part in BLOCKED_PARTS):
        raise ValueError("Unsupported FHIRPath feature.")

    return load_fhir_log()

def execute_fhirpath(expression: str) -> str:
    bundle = bundle_fhirpath(expression)
    results = evaluate(
        deepcopy(bundle),
        expression,
        {},
        models["r4"],
    )
    if not results:
        return json.dumps({
            "status": "not_found",
            "results": [],
            "message": (
                "No matching FHIR record found. "
                "Do not retry with another FHIRPath expression automatically. "
                "Ask the user to verify the patient name or provide the patient ID."
            ),
        })
    
    payload = {
        "fhirpath": expression,
        "results": results[:MAX_RESULTS],
    }

    return json.dumps(payload, ensure_ascii=False, default=str)