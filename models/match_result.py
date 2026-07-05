from typing import List

from pydantic import ConfigDict

from models.base import BaseModel
from models.datapoints import Skill


class SkillPair(BaseModel):
    """A JD skill matched to a resume skill."""

    model_config = ConfigDict(extra="forbid")
    jd_skill: Skill
    resume_skill: Skill
    similarity: float
    meets_yoe: bool


class MatchResult(BaseModel):
    """Matching metrics between JD and resume keyword lists."""

    model_config = ConfigDict(extra="forbid")
    precision: float
    recall: float
    f1: float
    matched: List[SkillPair]
    missing_from_resume: List[Skill]
    extra_on_resume: List[Skill]
    jd_skill_count: int
    resume_skill_count: int
    matched_count: int
    is_match: bool
