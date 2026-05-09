"""Graph schema — node + edge types for the UAP knowledge graph.

Why a typed schema (not free-form triples)?
-------------------------------------------
GraphRAG-style extraction with no schema produces an entity for every
proper noun the LLM stumbles across — same person spelled three ways,
"FBI" and "Federal Bureau of Investigation" as separate nodes,
locations duplicated. A schema with bounded types + Pydantic validation
makes each chunk's extraction *consistent*, which matters at merge time
when we resolve entities across thousands of chunks.

Node types
----------
- Document   — a PDF, video, image, or video frame
- Chunk      — a retrievable text span within a document
- Frame      — a video keyframe (subtype of Document with timestamp)
- Agency     — FBI / DOW / NASA / NARA / DOS / DOD / etc.
- Person     — named individual (witness, official, researcher)
- Location   — place name (city, base, country, water body)
- Object     — what was reportedly seen (disc, triangle, light, etc.)
- Incident   — a discrete event (date + place + what happened)
- CaseFile   — government file ID (e.g. "62-HQ-83894")
- Date       — temporal anchor (YYYY-MM-DD or YYYY)
- Community  — GraphRAG community cluster (computed, not extracted)

Edge types
----------
- (:Document)-[:HAS_CHUNK]->(:Chunk)
- (:Document)-[:FROM_AGENCY]->(:Agency)
- (:Document)-[:IN_CASEFILE]->(:CaseFile)
- (:Frame)-[:IN_VIDEO]->(:Document)
- (:Chunk)-[:MENTIONS]->(:Person|:Location|:Object|:Date|:Agency)
- (:Incident)-[:OCCURRED_AT]->(:Location)
- (:Incident)-[:ON_DATE]->(:Date)
- (:Incident)-[:INVOLVES]->(:Object)
- (:Incident)-[:WITNESSED_BY]->(:Person)
- (:Incident)-[:REPORTED_IN]->(:Document)
- (:Incident)-[:HANDLED_BY]->(:Agency)
- (:Community)-[:CONTAINS]->(:Person|:Location|:Incident|:...)

Vector indexes
--------------
- Chunk.text_embedding (384-dim, bge-small) — text retrieval
- Frame.clip_embedding   (512-dim, ViT-B/32) — visual retrieval
- Frame.text_embedding   (384-dim, bge-small) — caption retrieval
- Community.text_embedding (384-dim) — global-question routing
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Bounded vocabularies keep extraction consistent across chunks.
AGENCY_ENUM = Literal[
    "FBI", "DOW", "DOD", "NASA", "NARA", "DOS",
    "USAF", "USN", "USMC", "USA", "CIA", "NSA",
    "AARO", "UAPTF", "OTHER",
]

OBJECT_KIND_ENUM = Literal[
    "disc", "saucer", "triangle", "cigar", "sphere", "egg-shape",
    "tic-tac", "light", "fireball", "boomerang", "rectangle",
    "swarm", "unknown_craft", "other",
]


class ExtractedEntity(BaseModel):
    """One entity mention pulled from a chunk."""

    type: Literal["Person", "Location", "Object", "Agency", "CaseFile", "Date"]
    name: str = Field(description="Canonical surface form (e.g. 'J. Edgar Hoover').")
    aliases: list[str] = Field(default_factory=list)
    # Type-specific extras kept loose so the LLM can fill what it knows.
    attrs: dict[str, str] = Field(default_factory=dict)


class ExtractedIncident(BaseModel):
    """A discrete UAP event extracted from a chunk."""

    summary: str = Field(description="One-sentence factual description of what happened.")
    date: Optional[str] = Field(
        default=None,
        description="ISO date if known (YYYY-MM-DD), or year, or 'unknown'.",
    )
    location: Optional[str] = Field(default=None)
    agency: Optional[str] = Field(default=None)
    object_kind: Optional[str] = Field(
        default=None,
        description="One of disc/saucer/triangle/cigar/sphere/light/etc., or 'unknown'.",
    )
    witnesses: list[str] = Field(default_factory=list)


class ChunkExtraction(BaseModel):
    """The full structured output we ask Claude to produce per chunk.

    Doing entities + incidents in one pass (vs two CLI calls per chunk)
    halves the LLM cost. The two are correlated so the model produces
    consistent linking between them.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    incidents: list[ExtractedIncident] = Field(default_factory=list)


class FrameCaption(BaseModel):
    """Per-keyframe vision output."""

    caption: str = Field(description="One-sentence factual description.")
    object_kind: Optional[str] = Field(
        default=None,
        description="The shape/type of any UAP-like object visible, or null.",
    )
    visible_text: str = Field(default="", description="OCR'd visible text, '' if none.")
    salient_entities: list[str] = Field(default_factory=list)


# -- Cypher schema strings (for `CREATE INDEX` etc.) --------------------------

CYPHER_INDEXES = [
    # Uniqueness constraints — entity dedup happens at write time, but
    # indexes catch duplicates that slip past normalization.
    "CREATE INDEX FOR (a:Agency) ON (a.name)",
    "CREATE INDEX FOR (p:Person) ON (p.name)",
    "CREATE INDEX FOR (l:Location) ON (l.name)",
    "CREATE INDEX FOR (o:Object) ON (o.kind)",
    "CREATE INDEX FOR (d:Document) ON (d.file)",
    "CREATE INDEX FOR (c:Chunk) ON (c.id)",
    "CREATE INDEX FOR (cf:CaseFile) ON (cf.id)",
    "CREATE INDEX FOR (i:Incident) ON (i.id)",
    "CREATE INDEX FOR (com:Community) ON (com.id)",
    # FalkorDB vector indexes — cosine sim for normalized embeddings.
    "CREATE VECTOR INDEX FOR (c:Chunk) ON (c.text_embedding) "
    "OPTIONS { dimension: 384, similarityFunction: 'cosine' }",
    "CREATE VECTOR INDEX FOR (f:Frame) ON (f.clip_embedding) "
    "OPTIONS { dimension: 512, similarityFunction: 'cosine' }",
    "CREATE VECTOR INDEX FOR (f:Frame) ON (f.text_embedding) "
    "OPTIONS { dimension: 384, similarityFunction: 'cosine' }",
    "CREATE VECTOR INDEX FOR (com:Community) ON (com.text_embedding) "
    "OPTIONS { dimension: 384, similarityFunction: 'cosine' }",
]
