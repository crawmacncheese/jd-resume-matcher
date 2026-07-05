from typing import List, Literal, Optional

from pydantic import ConfigDict, Field

from models.base import BaseModel
from models.datapoints import DataPoints, Skill
from models.sentence import Sentence
from models.validation import ValidationIssue


class ExtractorState(BaseModel):
    """Shared LangGraph state for the Keywords Extractor workflow (Phase 1 + Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    # --- Input ---
    document: str
    document_type: Optional[Literal["jd", "resume"]] = None

    # --- Pre-processor ---
    # Ordered, normalized sentences produced from the raw document.
    sentences: List[Sentence] = Field(default_factory=list)

    # --- Phase 1: Build Comprehensive Set ---
    # Keywords Extractor: candidate keywords with sentence references.
    extracted_skills: List[Skill] = Field(default_factory=list)
    # Keywords Merger: deduplicated, merged keyword list.
    merged_skills: List[Skill] = Field(default_factory=list)
    # Supplementary Extractor: gap-filling keywords when coverage is lacking.
    supplementary_skills: List[Skill] = Field(default_factory=list)
    # Comprehensive Checker verdict.
    is_comprehensive: bool = False
    # Loop guard for Comprehensive Checker → Supplementary Extractor → Merger.
    phase1_iteration: int = 0
    max_phase1_iterations: int = 3

    # --- Phase 2: Validate and Polish ---
    # Validator verdict and per-keyword issues.
    is_valid: bool = False
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    # Loop guard for Validator → Modifier → Validator.
    phase2_iteration: int = 0
    max_phase2_iterations: int = 3

    # --- Final output ---
    # Polished, fine-grained keywords returned when Phase 2 completes.
    datapoints: DataPoints = Field(default_factory=lambda: DataPoints(skills=[]))
