"""Deterministic synthetic corpus generation for scalability experiments.

The scaling harness needs documents whose size distribution and vocabulary it
controls, so index growth and retrieval latency can be attributed to a *known*
input rather than to an opaque external corpus. Everything here is deterministic:
a :class:`CorpusConfig` plus a seed yields byte-identical documents, so a
recorded benchmark can be regenerated from the committed configuration alone.

The generated text is entirely synthetic — a small templated vocabulary plus a
per-document distinctive token — so retrieval experiments have real lexical
signal (each document can be told apart) without any personal or third-party
content. It is deliberately *not* one document repeated: every document draws an
independent size class, section count and vocabulary, so the corpus exercises a
realistic mix of short and long documents.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Coarse size classes used to build a realistic short/typical/large mix. Each
#: entry is ``(name, weight, min_sections, max_sections)``. The weights are
#: probabilities and must sum to 1.0.
_SIZE_CLASSES: tuple[tuple[str, float, int, int], ...] = (
    ("short", 0.30, 1, 3),
    ("typical", 0.50, 4, 9),
    ("large", 0.20, 10, 24),
)

#: Boring but varied filler vocabulary. Combined per-document with distinctive
#: tokens so documents are lexically distinguishable during retrieval.
_FILLER_WORDS: tuple[str, ...] = (
    "agreement", "obligation", "obligations", "party", "parties", "clause",
    "provision", "provisions", "schedule", "annex", "notice", "written",
    "consent", "liable", "liability", "payment", "invoice", "terminate",
    "termination", "renewal", "confidential", "disclosure", "governing",
    "jurisdiction", "warranty", "indemnity", "remedy", "breach", "cure",
    "reasonable", "commercially", "material", "materially", "services",
    "supplier", "customer", "provider", "licensee", "vendor", "effective",
    "date", "term", "conditions", "rights", "duties", "perform", "performance",
)

_TOPIC_VERBS: tuple[str, ...] = (
    "shall provide", "shall maintain", "may disclose", "shall not exceed",
    "shall indemnify", "shall promptly deliver", "is entitled to request",
    "shall cooperate with respect to",
)


class CorpusConfig(BaseModel):
    """Explicit, reproducible description of a synthetic corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: int = Field(ge=1, le=1_000_000)
    seed: int = 0
    #: Roughly how many paragraphs each section contains. Together with the size
    #: class this controls the average chunk count per document.
    paragraphs_per_section: int = Field(default=2, ge=1, le=20)
    #: Target paragraph length in words; the generator samples around this value.
    words_per_paragraph: int = Field(default=55, ge=5, le=2000)
    #: Distinctive per-document vocabulary size; larger values give retrieval
    #: more signal to separate near-duplicate documents.
    distinctive_terms: int = Field(default=6, ge=1, le=100)
    #: Stable identifier prefix for generated documents.
    id_prefix: str = "syn"

    @model_validator(mode="after")
    def _check_size_weights(self) -> CorpusConfig:
        total = sum(weight for _name, weight, _lo, _hi in _SIZE_CLASSES)
        if abs(total - 1.0) > 1e-9:  # pragma: no cover - defensive self-check
            raise ValueError(f"size-class weights must sum to 1.0, got {total}")
        return self

    def document_ids(self) -> Iterator[str]:
        for index in range(self.documents):
            yield f"{self.id_prefix}-{index:07d}"

    def as_fingerprint_fields(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "seed": self.seed,
            "paragraphs_per_section": self.paragraphs_per_section,
            "words_per_paragraph": self.words_per_paragraph,
            "distinctive_terms": self.distinctive_terms,
            "id_prefix": self.id_prefix,
        }


@dataclass(frozen=True)
class SyntheticDocument:
    """One generated document, addressed by a stable synthetic id."""

    id: str
    title: str
    text: str
    size_class: str
    section_count: int


def _pick_size_class(rng: random.Random) -> tuple[str, int]:
    draw = rng.random()
    cumulative = 0.0
    for name, weight, low, high in _SIZE_CLASSES:
        cumulative += weight
        if draw <= cumulative:
            return name, rng.randint(low, high)
    # Floating-point guard: fall back to the last class.
    name, _weight, low, high = _SIZE_CLASSES[-1]
    return name, rng.randint(low, high)


def _distinctive_terms(rng: random.Random, count: int, prefix: str) -> list[str]:
    terms = set()
    while len(terms) < count:
        terms.add(f"{prefix}{rng.randint(0, 999_999):06d}")
    return sorted(terms)


def _paragraph(rng: random.Random, distinctive: list[str], words_per_paragraph: int) -> str:
    word_count = max(8, int(rng.gauss(words_per_paragraph, words_per_paragraph * 0.25)))
    words: list[str] = []
    for index in range(word_count):
        if index % 7 == 0:
            words.append(rng.choice(distinctive))
        elif index % 11 == 0:
            words.append(rng.choice(_TOPIC_VERBS))
        else:
            words.append(rng.choice(_FILLER_WORDS))
    sentence = " ".join(words)
    return sentence[:1].upper() + sentence[1:] + "."


def generate_document(config: CorpusConfig, index: int) -> SyntheticDocument:
    """Generate exactly one document, deterministically, from its index."""

    # Seeding a Random with a string is stable across processes (unlike hash());
    # it combines the corpus seed with the document index.
    rng = random.Random(f"rag-workbench:synthetic:{config.seed}:{index}")
    doc_id = f"{config.id_prefix}-{index:07d}"
    token_prefix = f"tok{index:05d}"
    distinctive = _distinctive_terms(rng, config.distinctive_terms, token_prefix)
    size_class, section_count = _pick_size_class(rng)

    parts: list[str] = [f"Document {doc_id}"]
    for section in range(1, section_count + 1):
        heading_style = rng.choice(("numeric", "decimal", "alpha"))
        if heading_style == "numeric":
            heading = f"{section}. Topic {section}"
        elif heading_style == "decimal":
            heading = f"{section}.1 Topic {section}"
        else:
            heading = f"{chr(64 + ((section - 1) % 26) + 1)}. Topic {section}"
        body = [
            _paragraph(rng, distinctive, config.words_per_paragraph)
            for _ in range(config.paragraphs_per_section)
        ]
        parts.append(heading)
        parts.extend(body)

    return SyntheticDocument(
        id=doc_id,
        title=f"Synthetic document {doc_id}",
        text="\n\n".join(parts),
        size_class=size_class,
        section_count=section_count,
    )


def generate_corpus(config: CorpusConfig) -> list[SyntheticDocument]:
    """Generate the full corpus as a list (bounded sizes only)."""

    return [generate_document(config, index) for index in range(config.documents)]


def iter_corpus(config: CorpusConfig) -> Iterator[SyntheticDocument]:
    """Generate the corpus lazily, one document at a time."""

    for index in range(config.documents):
        yield generate_document(config, index)


def corpus_fingerprint(documents: list[SyntheticDocument]) -> str:
    """A stable SHA-256 over the generated ids and text.

    Two runs with the same configuration must produce the same fingerprint; the
    reproducibility test asserts exactly this.
    """

    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.id):
        digest.update(document.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def size_class_counts(documents: list[SyntheticDocument]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name, _w, _lo, _hi in _SIZE_CLASSES}
    for document in documents:
        counts[document.size_class] = counts.get(document.size_class, 0) + 1
    return counts


def write_corpus(documents: list[SyntheticDocument], directory: Path) -> dict[str, object]:
    """Write the corpus to ``directory`` as one ``<id>.txt`` file per document.

    Returns a small manifest (counts and fingerprint) so the output can be
    verified without re-reading every file.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for document in documents:
        (directory / f"{document.id}.txt").write_text(document.text, encoding="utf-8")
    manifest = {
        "documents": len(documents),
        "fingerprint": corpus_fingerprint(documents),
        "size_classes": size_class_counts(documents),
        "config_fields": "see CorpusConfig; regenerate with datasets/scaling generator",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "CorpusConfig",
    "SyntheticDocument",
    "corpus_fingerprint",
    "generate_corpus",
    "generate_document",
    "iter_corpus",
    "size_class_counts",
    "write_corpus",
]
