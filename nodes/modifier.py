from langchain_core.messages import HumanMessage, SystemMessage

from models.datapoints import DataPoints
from models.extractor_state import ExtractorState
from nodes.common import format_sentences, format_skills, parse_llm_json, parse_skills
from utils.llm import LLM_USE_CACHE, invoke_llm

_SKILL_SCHEMA = """
{
  "skills": [
    {
      "name": "Python",
      "category": "hard_skills",
      "importance": "required",
      "proficiency": "advanced",
      "yoe": 3,
      "referenced_sentence_ids": ["0"]
    }
  ]
}
"""


def modifier(state: ExtractorState) -> ExtractorState:
    """Remove or fix invalid keywords identified by the validator."""
    if not state.validation_issues:
        return state

    invalid_names = {
        issue.skill.name.strip().lower() for issue in state.validation_issues
    }
    system_prompt = f"""
    You polish a keyword list by removing or correcting invalid keywords.
    Keep valid keywords unchanged unless a small correction is necessary.
    Return JSON only using this schema:
    {_SKILL_SCHEMA}
    """

    user_prompt = f"""
    Sentences:
    {format_sentences(state.sentences)}

    Current keywords:
    {format_skills(state.datapoints.skills)}

    Invalid keywords to remove or fix:
    {format_skills([issue.skill for issue in state.validation_issues])}

    Reasons:
    {chr(10).join(f"- {issue.skill.name}: {issue.reason}" for issue in state.validation_issues)}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        response_format={"type": "json_object"},
        use_cache=LLM_USE_CACHE,
    )

    revised_skills = parse_skills(parse_llm_json(response))
    if not revised_skills:
        revised_skills = [
            skill
            for skill in state.datapoints.skills
            if skill.name.strip().lower() not in invalid_names
        ]

    state.datapoints = DataPoints(skills=revised_skills)
    state.validation_issues = []
    state.is_valid = False
    return state
