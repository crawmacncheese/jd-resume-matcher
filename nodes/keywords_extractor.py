from langchain_core.messages import HumanMessage, SystemMessage

from models.extractor_state import ExtractorState
from nodes.common import format_sentences, parse_llm_json, parse_skills
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


def keywords_extractor(state: ExtractorState) -> ExtractorState:
    """Extract candidate keywords/skills from preprocessed sentences."""
    if not state.sentences:
        raise ValueError("No sentences available for keyword extraction")

    doc_label = state.document_type or "document"
    system_prompt = f"""
    You extract keywords and skills from a {doc_label}.
    Scan every sentence and return every relevant hard or soft skill you can infer.
    Each skill must reference at least one sentence id from the input.
    Use only these enum values:
    - category: hard_skills | soft_skills
    - importance: unknown | required | preferred
    - proficiency: beginner | intermediate | proficient | advanced | expert
    Return JSON only using this schema:
    {_SKILL_SCHEMA}
    """

    user_prompt = f"""
    Sentences:
    {format_sentences(state.sentences)}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        response_format={"type": "json_object"},
        use_cache=LLM_USE_CACHE,
    )

    state.extracted_skills = parse_skills(parse_llm_json(response))
    return state
