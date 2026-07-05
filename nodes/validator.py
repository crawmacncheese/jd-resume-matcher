from langchain_core.messages import HumanMessage, SystemMessage

from models.datapoints import DataPoints
from models.extractor_state import ExtractorState
from models.validation import ValidationIssue
from nodes.common import format_sentences, format_skills, parse_llm_json, parse_skills
from utils.llm import LLM_USE_CACHE, invoke_llm


def validator(state: ExtractorState) -> ExtractorState:
    """Validate that each keyword can be inferred from its referenced sentences."""
    state.phase2_iteration += 1

    if state.phase2_iteration >= state.max_phase2_iterations:
        state.is_valid = True
        state.validation_issues = []
        return state

    skills = state.datapoints.skills
    if not skills:
        state.is_valid = True
        state.validation_issues = []
        return state

    system_prompt = """
    You validate whether each keyword can be correctly inferred from its referenced sentences.
    A keyword is invalid if the referenced sentences do not support it, if the references are
    wrong, or if the keyword is too vague or duplicated in meaning.
    Return JSON only:
    {
      "is_valid": false,
      "invalid_skills": [
        {
          "name": "Python",
          "category": "hard_skills",
          "importance": "required",
          "proficiency": "advanced",
          "yoe": 3,
          "referenced_sentence_ids": ["0"],
          "reason": "Sentence 0 does not mention Python."
        }
      ]
    }
    If every keyword is valid, return {"is_valid": true, "invalid_skills": []}.
    """

    user_prompt = f"""
    Sentences:
    {format_sentences(state.sentences)}

    Keywords to validate:
    {format_skills(skills)}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        response_format={"type": "json_object"},
        use_cache=LLM_USE_CACHE,
    )

    result = parse_llm_json(response)
    state.is_valid = bool(result.get("is_valid", False))

    issues: list[ValidationIssue] = []
    for item in result.get("invalid_skills", []):
        parsed = parse_skills({"skills": [item]})
        if parsed:
            issues.append(ValidationIssue(skill=parsed[0], reason=item["reason"]))
    state.validation_issues = issues
    return state
