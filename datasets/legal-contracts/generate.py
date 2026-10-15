"""Generate the synthetic legal-contract retrieval evaluation dataset.

Every contract, clause and query in this dataset is original text authored for
RAG Workbench. It contains no third-party copyrighted material. The script has no
arguments and writes its output next to itself::

    python datasets/legal-contracts/generate.py

The dataset is deliberately clause-anchored: relevance is judged at the
evidence-unit (clause) level via ``evidence.jsonl`` and ``EVIDENCE_MAPPING``, not
at the whole-document level. See ``README.md`` for the file formats and the
relevance-mapping policy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Running the script directly places ``datasets/legal-contracts`` (not ``src``) on
# sys.path, so add the source tree explicitly when the package is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_workbench.evaluation.datasets import (  # noqa: E402
    EVIDENCE_MAPPING,
    CorpusDocument,
    EvalQuery,
    EvaluationDataset,
    EvidenceUnit,
    build_manifest,
    write_dataset,
)

# ---------------------------------------------------------------------- contracts
#
# Each contract is a list of clauses: (section_number, heading_line, body).
# The section number is the clause anchor used in evidence ids; the heading line
# already carries the numbering so that the rendered text reads naturally. The
# numbering scheme varies per contract (integers, decimals, roman numerals,
# letters) so that clause-boundary chunking is exercised for real diversity.

CONTRACTS: list[dict[str, Any]] = [
    {
        "id": "nda-01",
        "title": "Mutual Non-Disclosure Agreement",
        "clauses": [
            (
                "1",
                "1. Definitions",
                "Confidential Information means any non-public technical, financial or "
                "business information disclosed by one party to the other, whether orally, "
                "in writing or by inspection of tangible objects.",
            ),
            (
                "2",
                "2. Confidentiality Obligations",
                "The Receiving Party shall hold the Discloser's Confidential Information in "
                "strict confidence and shall not disclose it to any third party without "
                "prior written consent.",
            ),
            (
                "3",
                "3. Permitted Disclosures",
                "The Receiving Party may disclose Confidential Information to its employees "
                "and professional advisers who need to know it, provided each such person is "
                "bound by confidentiality obligations at least as protective as those in this "
                "Agreement.",
            ),
            (
                "4",
                "4. Exclusions",
                "Confidential Information does not include information that is or becomes "
                "publicly available through no breach of this Agreement, or that was lawfully "
                "known to the Receiving Party before disclosure.",
            ),
            (
                "5",
                "5. Return and Destruction",
                "Upon written request, the Receiving Party shall promptly return or destroy "
                "all materials containing Confidential Information and certify such destruction "
                "in writing.",
            ),
            (
                "6",
                "6. Term and Survival",
                "This Agreement remains in effect for three years from the Effective Date, and "
                "the confidentiality obligations survive termination for a further two years.",
            ),
            (
                "7",
                "7. Remedies",
                "The parties agree that any breach of this Agreement may cause irreparable harm "
                "for which monetary damages would be inadequate, and the disclosing party is "
                "entitled to seek injunctive relief.",
            ),
            (
                "8",
                "8. Governing Law",
                "This Agreement is governed by the laws of the State of Delaware, without regard "
                "to its conflict of laws principles.",
            ),
        ],
    },
    {
        "id": "msa-02",
        "title": "Master Services Agreement",
        "clauses": [
            (
                "1",
                "1. Definitions",
                "Services means the professional services described in a mutually executed "
                "statement of work.",
            ),
            (
                "2.1",
                "2.1 Scope of Services",
                "The Supplier shall perform the Services with reasonable skill and care and in "
                "accordance with the applicable statement of work.",
            ),
            (
                "2.2",
                "2.2 Change Orders",
                "Either party may request a change to the Services by written change order, and "
                "no change is binding until both parties sign it.",
            ),
            (
                "3.1",
                "3.1 Invoicing",
                "The Supplier shall submit invoices monthly in arrears, and the Customer shall "
                "pay each undisputed invoice within thirty days of receipt.",
            ),
            (
                "3.2",
                "3.2 Late Payment",
                "Overdue amounts accrue interest at one percent per month or the maximum rate "
                "permitted by law, whichever is lower.",
            ),
            (
                "4",
                "4. Term and Renewal",
                "This Agreement commences on the Effective Date and renews automatically for "
                "successive twelve-month terms unless either party gives sixty days' written "
                "notice of non-renewal.",
            ),
            (
                "5",
                "5. Termination for Convenience",
                "The Customer may terminate this Agreement for convenience on ninety days' "
                "written notice, subject to payment for Services performed before the effective "
                "date of termination.",
            ),
            (
                "6",
                "6. Limitation of Liability",
                "Except for fraud and wilful misconduct, neither party's aggregate liability "
                "under this Agreement shall exceed the fees paid or payable in the twelve months "
                "preceding the claim.",
            ),
            (
                "7",
                "7. Indemnification",
                "The Supplier shall indemnify the Customer against third-party claims that the "
                "Services infringe a patent, copyright or trade secret, provided the Customer "
                "promptly notifies the Supplier.",
            ),
            (
                "8",
                "8. Assignment",
                "Neither party may assign this Agreement without the other party's prior written "
                "consent, except to an affiliate or in connection with a merger.",
            ),
            (
                "9",
                "9. Governing Law and Dispute Resolution",
                "This Agreement is governed by the laws of England and Wales, and the parties "
                "submit to the exclusive jurisdiction of the courts of London.",
            ),
        ],
    },
    {
        "id": "saas-03",
        "title": "SaaS Subscription Agreement",
        "clauses": [
            (
                "I",
                "I. Subscription Grant",
                "Subject to payment of the applicable fees, the Provider grants the Customer a "
                "non-exclusive, non-transferable right to access and use the hosted service "
                "during the subscription term.",
            ),
            (
                "II",
                "II. Service Levels",
                "The Provider shall make the hosted service available at least 99.9 percent of "
                "the time in each calendar month, excluding scheduled maintenance windows "
                "notified in advance.",
            ),
            (
                "III",
                "III. Fees and Payment",
                "Subscription fees are billed annually in advance and are non-refundable except "
                "as expressly stated in this Agreement.",
            ),
            (
                "IV",
                "IV. Customer Data",
                "As between the parties, the Customer owns all Customer Data, and the Provider "
                "processes Customer Data only to provide the hosted service.",
            ),
            (
                "V",
                "V. Acceptable Use",
                "The Customer shall not use the hosted service to store or transmit unlawful "
                "content or attempt to gain unauthorised access to any part of the service.",
            ),
            (
                "VI",
                "VI. Suspension and Termination",
                "The Provider may suspend the hosted service immediately if continued use poses "
                "a security risk and may terminate this Agreement for material breach not cured "
                "within thirty days of notice.",
            ),
            (
                "VII",
                "VII. Warranties",
                "The Provider warrants that the hosted service will perform materially in "
                "accordance with the documentation, and the Customer's exclusive remedy is "
                "repair or replacement.",
            ),
            (
                "VIII",
                "VIII. Data Protection",
                "The parties shall comply with applicable data protection law, and the Provider "
                "shall implement appropriate technical and organisational measures to protect "
                "personal data.",
            ),
            (
                "IX",
                "IX. Governing Law",
                "This Agreement is governed by the laws of the State of California.",
            ),
        ],
    },
    {
        "id": "dpa-04",
        "title": "Data Processing Addendum",
        "clauses": [
            (
                "1",
                "1. Definitions",
                "Personal Data means any information relating to an identified or identifiable "
                "natural person that the Processor processes on behalf of the Controller.",
            ),
            (
                "2",
                "2. Roles of the Parties",
                "The Controller is the data controller and the Processor processes Personal Data "
                "only as a data processor.",
            ),
            (
                "3",
                "3. Processing Instructions",
                "The Processor shall process Personal Data only on documented instructions from "
                "the Controller, unless required to do so by law.",
            ),
            (
                "4",
                "4. Security Measures",
                "The Processor shall implement appropriate technical and organisational measures, "
                "including encryption of data in transit and at rest, to protect Personal Data.",
            ),
            (
                "5",
                "5. Subprocessors",
                "The Processor shall not engage any subprocessor without the Controller's prior "
                "written authorisation and shall impose equivalent data protection obligations on "
                "each subprocessor.",
            ),
            (
                "6",
                "6. Data Subject Rights",
                "The Processor shall assist the Controller in responding to data subject requests "
                "to access, rectify or erase Personal Data within the timeframes required by law.",
            ),
            (
                "7",
                "7. International Transfers",
                "The Processor shall not transfer Personal Data outside the European Economic Area "
                "unless the transfer is covered by an adequacy decision or appropriate safeguards.",
            ),
            (
                "8",
                "8. Return and Deletion",
                "On termination, the Processor shall delete or return all Personal Data at the "
                "Controller's choice and delete existing copies unless storage is required by law.",
            ),
            (
                "9",
                "9. Liability",
                "Each party's liability under this Addendum is subject to the limitations set out "
                "in the main agreement.",
            ),
        ],
    },
    {
        "id": "reseller-05",
        "title": "Reseller Agreement",
        "clauses": [
            (
                "A",
                "A. Appointment",
                "The Company appoints the Reseller as a non-exclusive reseller of the Products in "
                "the Territory.",
            ),
            (
                "B",
                "B. Reseller Obligations",
                "The Reseller shall use commercially reasonable efforts to promote and sell the "
                "Products and shall not make any representation or warranty on the Company's "
                "behalf.",
            ),
            (
                "C",
                "C. Pricing and Discounts",
                "The Company shall sell the Products to the Reseller at the discount set out in "
                "Schedule 1, and the Reseller may set its own resale prices.",
            ),
            (
                "D",
                "D. Orders and Acceptance",
                "All orders placed by the Reseller are subject to acceptance by the Company, and "
                "no order is binding until the Company issues a written order confirmation.",
            ),
            (
                "E",
                "E. Intellectual Property",
                "The Reseller acknowledges that all intellectual property rights in the Products "
                "remain vested in the Company and shall not modify, reverse engineer or remove any "
                "proprietary notices from the Products.",
            ),
            (
                "F",
                "F. Trademark Licence",
                "The Company grants the Reseller a non-exclusive, revocable licence to use the "
                "Company's trademarks solely to market the Products during the term.",
            ),
            (
                "G",
                "G. Term and Termination",
                "This Agreement continues for one year and may be terminated by either party on "
                "sixty days' written notice, or immediately for material breach that remains "
                "uncured for fifteen days.",
            ),
            (
                "H",
                "H. Governing Law",
                "This Agreement is governed by the laws of the State of New York, and the parties "
                "submit to the exclusive jurisdiction of the state and federal courts located in "
                "New York County.",
            ),
        ],
    },
    {
        "id": "sow-06",
        "title": "Statement of Work",
        "clauses": [
            (
                "1.1",
                "1.1 Engagement",
                "The Supplier shall provide the consulting services described in this Statement "
                "of Work under and subject to the terms of the Master Services Agreement.",
            ),
            (
                "1.2",
                "1.2 Deliverables",
                "The Supplier shall deliver a technical design document, a working prototype and "
                "a final implementation report.",
            ),
            (
                "1.3",
                "1.3 Milestones",
                "The project comprises three milestones, each completed on the dates set out in "
                "the project schedule.",
            ),
            (
                "2.1",
                "2.1 Fees",
                "The Customer shall pay the Supplier a fixed fee of one hundred thousand dollars "
                "for the Deliverables.",
            ),
            (
                "2.2",
                "2.2 Expenses",
                "The Customer shall reimburse reasonable pre-approved travel expenses supported "
                "by receipts.",
            ),
            (
                "3.1",
                "3.1 Acceptance Criteria",
                "The Customer shall review each Deliverable within ten business days and either "
                "accept it or provide written notice of deficiencies.",
            ),
            (
                "3.2",
                "3.2 Project Governance",
                "A steering committee comprising one representative of each party shall meet "
                "fortnightly to review progress.",
            ),
            (
                "4.1",
                "4.1 Assumptions",
                "The Customer shall provide timely access to subject-matter experts and test "
                "environments.",
            ),
            (
                "4.2",
                "4.2 Dependencies",
                "The Supplier's delivery obligations are subject to the Customer's performance of "
                "its dependencies.",
            ),
            (
                "5.1",
                "5.1 Change Control",
                "Any change to the scope, schedule or fees requires a written change order signed "
                "by both parties.",
            ),
        ],
    },
    {
        "id": "license-07",
        "title": "Software Licence Agreement",
        "clauses": [
            (
                "1",
                "1. Definitions",
                "Software means the object code version of the vendor's application made "
                "available to the Licensee under this Agreement.",
            ),
            (
                "2",
                "2. Licence Grant",
                "Subject to payment of the licence fee, the Vendor grants the Licensee a "
                "perpetual, non-exclusive licence to install and use the Software for its "
                "internal business purposes.",
            ),
            (
                "3",
                "3. Restrictions",
                "The Licensee shall not sublicense, rent, lease or distribute the Software, nor "
                "use it to provide services to third parties, without the Vendor's prior written "
                "consent.",
            ),
            (
                "4",
                "4. Licence Fees",
                "The Licensee shall pay the licence fee within thirty days of the invoice date, "
                "and all fees are non-refundable.",
            ),
            (
                "5",
                "5. Support and Maintenance",
                "The Vendor shall provide email support during business hours and shall supply "
                "bug fixes and minor updates at no additional charge.",
            ),
            (
                "6",
                "6. Updates and New Releases",
                "Major new releases are licensed separately, and this Agreement does not entitle "
                "the Licensee to any major new release.",
            ),
            (
                "7",
                "7. Warranty Disclaimer",
                "Except as expressly stated, the Software is provided as is without warranties of "
                "any kind, whether express or implied.",
            ),
            (
                "8",
                "8. Limitation of Liability",
                "The Vendor's total liability shall not exceed the licence fees paid in the twelve "
                "months preceding the claim.",
            ),
            (
                "9",
                "9. Termination",
                "The Vendor may terminate this Agreement immediately if the Licensee materially "
                "breaches it and fails to cure within thirty days of written notice.",
            ),
            (
                "10",
                "10. Governing Law and Notices",
                "Notices must be written and delivered by hand, courier or registered post to the "
                "addresses stated in the order form.",
            ),
        ],
    },
]

# ----------------------------------------------------------------------- queries
#
# Each query is (query_id, text, [(evidence_unit_id, grade), ...]). The first unit
# listed is the primary judgement and is graded 2 when the query has more than one
# relevant clause; all other relevant clauses are graded 1.

QUERIES: list[tuple[str, str, list[tuple[str, int]]]] = [
    # -- exact terminology ----------------------------------------------------
    (
        "lq-001",
        "non-public technical financial or business information disclosed by one party",
        [("nda-01::1", 1)],
    ),
    (
        "lq-002",
        "prior written authorisation before engaging a subprocessor",
        [("dpa-04::5", 1)],
    ),
    (
        "lq-003",
        "99.9 percent availability in each calendar month",
        [("saas-03::II", 1)],
    ),
    (
        "lq-004",
        "overdue amounts accrue interest at one percent per month",
        [("msa-02::3.2", 1)],
    ),
    (
        "lq-005",
        "non-exclusive non-transferable right to access the hosted service",
        [("saas-03::I", 1)],
    ),
    (
        "lq-006",
        "aggregate liability exceeds fees paid in the twelve months preceding the claim",
        [("msa-02::6", 1)],
    ),
    (
        "lq-007",
        "irreparable harm for which monetary damages would be inadequate",
        [("nda-01::7", 1)],
    ),
    (
        "lq-008",
        "encryption of data in transit and at rest",
        [("dpa-04::4", 1)],
    ),
    # -- semantic paraphrase (no guaranteed verbatim overlap) -----------------
    (
        "lq-009",
        "How long does the duty to keep secrets last once the deal has ended?",
        [("nda-01::6", 1)],
    ),
    (
        "lq-010",
        "May a customer walk away simply because it wants to, and what must it still pay?",
        [("msa-02::5", 1)],
    ),
    (
        "lq-011",
        "Who gets to keep the buyer's records when the cloud service stops?",
        [("dpa-04::8", 1)],
    ),
    (
        "lq-012",
        "Can the customer put the program on its office computers for its own work?",
        [("license-07::2", 1)],
    ),
    (
        "lq-013",
        "Who holds the rights to the inventions and branding in the goods?",
        [("reseller-05::E", 1)],
    ),
    (
        "lq-014",
        "What protective steps does the vendor take to keep personal records safe?",
        [("dpa-04::4", 1)],
    ),
    (
        "lq-015",
        "Can the seller bring in another firm to help without permission?",
        [("dpa-04::5", 1)],
    ),
    # -- multi-clause questions (more than one relevant unit) -----------------
    (
        "lq-016",
        "How is confidential information defined and how long do the confidentiality "
        "duties last?",
        [("nda-01::1", 2), ("nda-01::6", 1)],
    ),
    (
        "lq-017",
        "What is the invoicing cycle and what happens if payment is late under the master "
        "agreement?",
        [("msa-02::3.1", 2), ("msa-02::3.2", 1)],
    ),
    (
        "lq-018",
        "Who controls personal data and on what basis may the processor act?",
        [("dpa-04::2", 2), ("dpa-04::3", 1)],
    ),
    (
        "lq-019",
        "What must the reseller do to promote the products and when is an order binding?",
        [("reseller-05::B", 2), ("reseller-05::D", 1)],
    ),
    (
        "lq-020",
        "What must the supplier deliver and what is the fixed fee for it?",
        [("sow-06::1.2", 2), ("sow-06::2.1", 1)],
    ),
    (
        "lq-021",
        "What rights are granted to the licensee and what may it not do with the software?",
        [("license-07::2", 2), ("license-07::3", 1)],
    ),
    (
        "lq-022",
        "Which provisions cap liability under the master agreement and the software licence?",
        [("msa-02::6", 2), ("license-07::8", 1)],
    ),
    (
        "lq-023",
        "What happens to the other side's materials on exit and what duties continue afterwards?",
        [("nda-01::5", 2), ("nda-01::6", 1)],
    ),
    # -- ambiguous terminology ------------------------------------------------
    (
        "lq-024",
        "What does the term Services cover?",
        [("msa-02::1", 1)],
    ),
    (
        "lq-025",
        "What is meant by Software in the licence agreement?",
        [("license-07::1", 1)],
    ),
    (
        "lq-026",
        "What counts as Personal Data?",
        [("dpa-04::1", 1)],
    ),
    (
        "lq-027",
        "What is treated as Confidential Information?",
        [("nda-01::1", 1)],
    ),
    (
        "lq-028",
        "What are the Deliverables under the statement of work?",
        [("sow-06::1.2", 1)],
    ),
    # -- reference to a specific section number -------------------------------
    (
        "lq-029",
        "What does clause 6 of the mutual non-disclosure agreement provide?",
        [("nda-01::6", 1)],
    ),
    (
        "lq-030",
        "Under section 3.1 of the master services agreement, when must invoices be paid?",
        [("msa-02::3.1", 1)],
    ),
    (
        "lq-031",
        "What does Article IV of the SaaS subscription agreement state?",
        [("saas-03::IV", 1)],
    ),
    (
        "lq-032",
        "What does section 7 of the data processing addendum require?",
        [("dpa-04::7", 1)],
    ),
    (
        "lq-033",
        "According to clause H of the reseller agreement, which law governs?",
        [("reseller-05::H", 1)],
    ),
    (
        "lq-034",
        "What does section 6 of the software licence address?",
        [("license-07::6", 1)],
    ),
    (
        "lq-035",
        "What does clause 5.1 of the statement of work cover?",
        [("sow-06::5.1", 1)],
    ),
    # -- exception / subject-to ----------------------------------------------
    (
        "lq-036",
        "When is information not treated as confidential?",
        [("nda-01::4", 1)],
    ),
    (
        "lq-037",
        "When does the limitation of liability not apply under the master agreement?",
        [("msa-02::6", 1)],
    ),
    (
        "lq-038",
        "Under what condition is the software supplied without warranties?",
        [("license-07::7", 1)],
    ),
    (
        "lq-039",
        "When may personal data be transferred outside the European Economic Area?",
        [("dpa-04::7", 1)],
    ),
    (
        "lq-040",
        "What is the customer's only remedy if the service fails to meet its warranty?",
        [("saas-03::VII", 1)],
    ),
    (
        "lq-041",
        "Does the licence include major new releases of the software?",
        [("license-07::6", 1)],
    ),
    (
        "lq-042",
        "When may the processor keep copies instead of deleting personal data?",
        [("dpa-04::8", 1)],
    ),
    # -- queries with more than one relevant evidence unit --------------------
    (
        "lq-043",
        "How is secrecy safeguarded across the non-disclosure agreement and the data "
        "processing addendum?",
        [("nda-01::2", 2), ("dpa-04::4", 1)],
    ),
    (
        "lq-044",
        "How can the master agreement and the reseller agreement be brought to an end?",
        [("msa-02::5", 2), ("reseller-05::G", 1)],
    ),
    (
        "lq-045",
        "Which law governs the master agreement and the reseller agreement?",
        [("msa-02::9", 2), ("reseller-05::H", 1)],
    ),
    (
        "lq-046",
        "What restrictions apply to passing on the products and the software to others?",
        [("license-07::3", 2), ("reseller-05::B", 1)],
    ),
    (
        "lq-047",
        "What payment obligations arise under the master agreement and the statement of work?",
        [("msa-02::3.1", 2), ("sow-06::2.1", 1)],
    ),
    (
        "lq-048",
        "How is personal data protected and returned?",
        [("dpa-04::4", 2), ("dpa-04::8", 1)],
    ),
]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _render(clauses: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(f"{heading}\n{body}" for _section, heading, body in clauses)


def _build_documents() -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for contract in CONTRACTS:
        documents.append(
            CorpusDocument(
                id=contract["id"],
                title=contract["title"],
                text=_render(contract["clauses"]),
            )
        )
    return documents


def _build_evidence(documents: list[CorpusDocument]) -> dict[str, EvidenceUnit]:
    units: dict[str, EvidenceUnit] = {}
    for contract in CONTRACTS:
        for section, _heading, body in contract["clauses"]:
            unit_id = f"{contract['id']}::{section}"
            units[unit_id] = EvidenceUnit(
                id=unit_id, contract=contract["id"], section=section, text=body
            )
    return units


def _validate(
    documents: list[CorpusDocument],
    queries: list[EvalQuery],
    qrels: dict[str, dict[str, int]],
    evidence: dict[str, EvidenceUnit],
) -> None:
    """Fail loudly if the authored content is internally inconsistent."""

    by_id = {document.id: document for document in documents}
    if not 6 <= len(documents) <= 8:
        raise ValueError(f"expected 6-8 contracts, found {len(documents)}")
    if not 40 <= len(queries) <= 48:
        raise ValueError(f"expected 40-48 queries, found {len(queries)}")

    query_ids = {query.id for query in queries}
    if set(qrels) != query_ids:
        raise ValueError("every query must have at least one relevance judgment")

    for query_id, judgments in qrels.items():
        if not judgments:
            raise ValueError(f"query '{query_id}' has no relevant unit")
        for unit_id in judgments:
            if unit_id not in evidence:
                raise ValueError(f"query '{query_id}' references unknown unit '{unit_id}'")

    for unit in evidence.values():
        contract = by_id[unit.contract]
        if _normalise(unit.text) not in _normalise(contract.text):
            raise ValueError(
                f"evidence unit '{unit.id}' text is not a substring of contract '{unit.contract}'"
            )


def build_dataset() -> EvaluationDataset:
    documents = _build_documents()
    evidence = _build_evidence(documents)
    queries = [EvalQuery(id=query_id, text=text) for query_id, text, _rels in QUERIES]
    qrels = {query_id: dict(rels) for query_id, _text, rels in QUERIES}
    _validate(documents, queries, qrels, evidence)

    manifest = build_manifest(
        name="legal-contracts",
        source="Original synthetic contracts authored for RAG Workbench",
        license="CC0-1.0 (synthetic; no third-party text)",
        relevance_mapping=EVIDENCE_MAPPING,
        description=(
            "Seven short commercial contracts (NDA, master services, SaaS subscription, "
            "data processing addendum, reseller, statement of work, software licence) with "
            "48 clause-anchored retrieval queries. Fully synthetic and offline."
        ),
        documents=documents,
        queries=queries,
        qrels=qrels,
        evidence=evidence,
        notes=[
            "All contract text is original and authored for this project; no third-party "
            "material is included.",
            "Relevance is judged at the clause/evidence-unit level, never per whole document.",
            "Clause numbering varies per contract (integers, decimals, roman numerals, "
            "letters) to exercise clause-boundary chunking.",
            "Evidence ids follow '<contract_id>::<section_number>'.",
        ],
    )
    return EvaluationDataset(
        manifest=manifest, documents=documents, queries=queries, qrels=qrels, evidence=evidence
    )


def main() -> None:
    dataset = build_dataset()
    directory = Path(__file__).resolve().parent
    manifest = write_dataset(directory, dataset)
    print(
        f"wrote {manifest.corpus_count} contracts, {manifest.query_count} queries, "
        f"{manifest.evidence_count} evidence units and {manifest.qrel_count} qrels to {directory}"
    )


if __name__ == "__main__":
    main()
