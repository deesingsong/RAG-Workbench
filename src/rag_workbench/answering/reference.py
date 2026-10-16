"""Reference-evidence coverage for *anchored* answer-quality questions (audit §1, §3).

The grounded-answer quality set carries hand-written annotations: a list of
``reference_keywords`` (lexical anchors that must appear for an answer to be
groundable) and/or ``reference_sections`` (``source::section`` ids). Those anchors
are the only place reference requirements are defined — this module never infers a
requirement for an arbitrary question.

For each anchor it records two independent facts:

* **in_retrieved** — the anchor appears in *some* retrieved candidate (whether or
  not that candidate reached the model);
* **in_selected**  — the anchor appears in a passage that actually *was* selected
  into the model context.

That distinction is what separates the three evidence-loss outcomes:

* ``never_retrieved``       — at least one required anchor is absent from the whole
  retrieved pool, so no context policy could have grounded it;
* ``retrieved_but_excluded``— every anchor was retrieved, but at least one was
  dropped from the context (e.g. by ``max_evidence`` or the token budget);
* ``included``              — every required anchor reached the context.

A single ``any()`` check over the anchors cannot tell these apart and *overstates*
coverage for multi-part questions (one matched anchor says nothing about the
others). Coverage here is a **lexical-anchor** check only: it is not evidence that
a passage semantically supports a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_workbench.domain.schemas import EvidenceItem, SearchHit

CoverageClass = str  # one of "never_retrieved" | "retrieved_but_excluded" | "included"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class ReferenceUnit:
    """One required evidence anchor and whether it was retrieved and/or selected."""

    kind: str  # "keyword" | "section"
    anchor: str
    in_selected: bool = False
    in_retrieved: bool = False


@dataclass
class ReferenceCoverage:
    """Coverage of a question's reference anchors by the retrieved/selected sets."""

    units: list[ReferenceUnit] = field(default_factory=list)

    @property
    def required(self) -> int:
        return len(self.units)

    @property
    def selected_count(self) -> int:
        return sum(1 for unit in self.units if unit.in_selected)

    @property
    def retrieved_count(self) -> int:
        return sum(1 for unit in self.units if unit.in_retrieved)

    @property
    def complete_selected(self) -> bool | None:
        """Every anchor reached the context (``None`` when there are no anchors)."""

        return all(unit.in_selected for unit in self.units) if self.units else None

    @property
    def partial_selected(self) -> bool | None:
        """Some but not all anchors reached the context (``None`` when none)."""

        if not self.units:
            return None
        selected = self.selected_count
        return 0 < selected < self.required

    @property
    def complete_retrieved(self) -> bool | None:
        return all(unit.in_retrieved for unit in self.units) if self.units else None

    @property
    def classification(self) -> CoverageClass | None:
        if not self.units:
            return None
        if any(not unit.in_retrieved for unit in self.units):
            return "never_retrieved"
        if any(not unit.in_selected for unit in self.units):
            return "retrieved_but_excluded"
        return "included"

    def missing_retrieved(self) -> list[str]:
        return [unit.anchor for unit in self.units if not unit.in_retrieved]

    def missing_selected(self) -> list[str]:
        return [unit.anchor for unit in self.units if not unit.in_selected]

    def as_dict(self) -> dict:
        return {
            "required": self.required,
            "retrieved": self.retrieved_count,
            "selected": self.selected_count,
            "complete_selected": self.complete_selected,
            "partial_selected": self.partial_selected,
            "complete_retrieved": self.complete_retrieved,
            "classification": self.classification,
            "missing_retrieved": self.missing_retrieved(),
            "missing_selected": self.missing_selected(),
        }


def _phrase_in(anchor: str, text: str) -> bool:
    return _normalize(anchor) in _normalize(text)


def _matches(unit: ReferenceUnit, passage: EvidenceItem | SearchHit) -> bool:
    if unit.kind == "keyword":
        return _phrase_in(unit.anchor, passage.text)
    source, _, section = unit.anchor.partition("::")
    name = passage.source_name or ""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return (name == source or stem == source) and (passage.section_number or "") == section


def build_reference_units(keywords: list[str], sections: list[str]) -> list[ReferenceUnit]:
    """Reference anchors for one question: keyword anchors, else section anchors."""

    units = [ReferenceUnit(kind="keyword", anchor=keyword) for keyword in keywords]
    if not units:
        units = [ReferenceUnit(kind="section", anchor=section) for section in sections]
    return units


def classify_reference_coverage(
    *,
    keywords: list[str],
    sections: list[str],
    selected: list[EvidenceItem],
    retrieved: list[SearchHit],
) -> ReferenceCoverage:
    """Classify how well the retrieved/selected sets cover a question's anchors."""

    units = build_reference_units(keywords, sections)
    for unit in units:
        unit.in_selected = any(_matches(unit, passage) for passage in selected)
        unit.in_retrieved = unit.in_selected or any(
            _matches(unit, passage) for passage in retrieved
        )
    return ReferenceCoverage(units=units)


__all__ = [
    "ReferenceCoverage",
    "ReferenceUnit",
    "build_reference_units",
    "classify_reference_coverage",
]
