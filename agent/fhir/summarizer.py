import asyncio

from llama_index.core import Document

from fhir.engine import load_fhir_log
from rag.engine import retrieve_from_documents


def code_text(concept: dict | None) -> str:
    if not concept:
        return "unknown"

    if concept.get("text"):
        return concept["text"]

    codings= concept.get("coding", [])

    return ", ".join(
        coding.get("display") or coding.get("code", "unknown")
        for coding in codings
    ) or "unknown"


def quantity_text(quantity: dict | None) -> str:
    if not quantity:
        return "unknown"

    return f"{quantity.get('value', 'unknown')} {quantity.get('unit', '')}".strip()


def subject_reference(resource: dict) -> str:
    return resource.get("subject", {}).get("reference", "unknown")


def format_patient(resource: dict) -> str:
    name = resource.get("name", [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")

    return f"""
Resource: Patient
ID: {resource.get("id", "unknown")}
Name: {given} {family}
Gender: {resource.get("gender", "unknown")}
Birth date: {resource.get("birthDate", "unknown")}
""".strip()


def format_condition(resource: dict) -> str:
    return f"""
Resource: Condition
ID: {resource.get("id", "unknown")}
Patient: {subject_reference(resource)}
Condition: {code_text(resource.get("code"))}
Clinical status: {code_text(resource.get("clinicalStatus"))}
Onset: {resource.get("onsetDateTime", "unknown")}
""".strip()


def format_observation(resource: dict) -> str:
    return f"""
Resource: Observation
ID: {resource.get("id", "unknown")}
Patient: {subject_reference(resource)}
Observation: {code_text(resource.get("code"))}
Value: {quantity_text(resource.get("valueQuantity"))}
Date: {resource.get("effectiveDateTime", "unknown")}
""".strip()


def format_medication(resource: dict) -> str:
    return f"""
Resource: MedicationRequest
ID: {resource.get("id", "unknown")}
Patient: {subject_reference(resource)}
Medication: {code_text(resource.get("medicationCodeableConcept"))}
Status: {resource.get("status", "unknown")}
Authored on: {resource.get("authoredOn", "unknown")}
""".strip()


def format_report(resource: dict) -> str:
    return f"""
Resource: DiagnosticReport
ID: {resource.get("id", "unknown")}
Patient: {subject_reference(resource)}
Report: {code_text(resource.get("code"))}
Conclusion: {resource.get("conclusion", "none")}
Date: {resource.get("effectiveDateTime", "unknown")}
""".strip()


FORMATTERS = {
    "Patient": format_patient,
    "Condition": format_condition,
    "Observation": format_observation,
    "MedicationRequest": format_medication,
    "DiagnosticReport": format_report,
}


def bundle_to_documents() -> list[Document]:
    bundle = load_fhir_log()
    documents = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "unknown")
        formatter = FORMATTERS.get(resource_type)

        if not formatter:
            continue

        documents.append(
            Document(
                text=formatter(resource),
                metadata={
                    "source": "fhir_log",
                    "resource_type": resource_type,
                    "resource_id": resource.get("id", "unknown"),
                    "subject": subject_reference(resource),
                },
            )
        )

    return documents


async def retrieve_fhir_rag_context(
    query: str,
    k: int = 5,
) -> str:
    documents = bundle_to_documents()

    nodes = await asyncio.to_thread(
        retrieve_from_documents,
        query=query,
        documents=documents,
        k=k,
    )

    if not nodes:
        return "No relevant FHIR records found."

    results = []

    for i, node_with_score in enumerate(nodes, start=1):
        node = getattr(node_with_score, "node", node_with_score)
        score = getattr(node_with_score, "score", None)

        results.append(
            f"""[FHIR Record {i} | score={score}]
{node.get_content()}"""
        )

    return "\n\n".join(results)