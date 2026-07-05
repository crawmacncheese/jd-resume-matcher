from typing import List

from pydantic import ConfigDict

from models.base import BaseModel
from models.datapoints import Skill


class ValidationIssue(BaseModel):
    """A keyword that failed validation along with the reason."""

    model_config = ConfigDict(extra="forbid")
    skill: Skill
    reason: str


class ValidationResult(BaseModel):
    """Output of the Phase 2 Validator node."""

    model_config = ConfigDict(extra="forbid")
    is_valid: bool
    issues: List[ValidationIssue]
