import json
from typing import Any, List

from langchain_core.messages import BaseMessage

from models.datapoints import (
    CategoryEnum,
    ImportanceEnum,
    ProficiencyEnum,
    Skill,
)
from models.sentence import Sentence


def parse_llm_json(response: BaseMessage) -> dict[str, Any]:
    content = response.content
    if isinstance(content, list):
        content = content[0]["text"]
    return json.loads(content)


def format_sentences(sentences: List[Sentence]) -> str:
    return "\n".join(f"[{sentence.id}] {sentence.sentence}" for sentence in sentences)


def parse_skills(data: dict[str, Any]) -> List[Skill]:
    skills: List[Skill] = []
    for item in data.get("skills", []):
        referenced_ids = item.get("referenced_sentence_ids", [])
        skills.append(
            Skill(
                name=item["name"].strip(),
                category=CategoryEnum(item["category"]),
                importance=ImportanceEnum(item.get("importance", ImportanceEnum.UNKOWN.value)),
                proficiency=ProficiencyEnum(item["proficiency"]),
                yoe=item.get("yoe"),
                referenced_sentence_ids=[str(ref_id) for ref_id in referenced_ids],
            )
        )
    return skills


_IMPORTANCE_RANK = {
    ImportanceEnum.UNKOWN: 0,
    ImportanceEnum.PREFERRED: 1,
    ImportanceEnum.REQUIRED: 2,
}

_PROFICIENCY_RANK = {
    ProficiencyEnum.BEGINNER: 0,
    ProficiencyEnum.INTERMEDIATE: 1,
    ProficiencyEnum.PROFICIENT: 2,
    ProficiencyEnum.ADVANCED: 3,
    ProficiencyEnum.EXPERT: 4,
}


def _merge_skill(existing: Skill, incoming: Skill) -> Skill:
    refs = sorted(
        set(existing.referenced_sentence_ids) | set(incoming.referenced_sentence_ids),
        key=lambda ref: int(ref) if ref.isdigit() else ref,
    )
    importance = (
        incoming.importance
        if _IMPORTANCE_RANK[incoming.importance] > _IMPORTANCE_RANK[existing.importance]
        else existing.importance
    )
    proficiency = (
        incoming.proficiency
        if _PROFICIENCY_RANK[incoming.proficiency] > _PROFICIENCY_RANK[existing.proficiency]
        else existing.proficiency
    )
    yoe = existing.yoe
    if incoming.yoe is not None:
        yoe = max(yoe or 0.0, incoming.yoe)

    return existing.model_copy(
        update={
            "referenced_sentence_ids": refs,
            "importance": importance,
            "proficiency": proficiency,
            "yoe": yoe,
        }
    )


def merge_skills(*skill_lists: List[Skill]) -> List[Skill]:
    merged: dict[str, Skill] = {}
    for skills in skill_lists:
        for skill in skills:
            key = skill.name.strip().lower()
            if key not in merged:
                merged[key] = skill.model_copy(deep=True)
            else:
                merged[key] = _merge_skill(merged[key], skill)
    return list(merged.values())


def format_skills(skills: List[Skill]) -> str:
    if not skills:
        return "(none)"
    return json.dumps(
        [skill.model_dump(mode="json") for skill in skills],
        indent=2,
    )
