from langchain_core.messages import HumanMessage, SystemMessage

from models.extractor_state import ExtractorState
from nodes.common import format_sentences, format_skills, parse_llm_json, parse_skills
from utils.llm import LLM_USE_CACHE, invoke_llm

_SKILL_SCHEMA = """
{
  "skills": [
    {
      "name": "Terraform",
      "category": "hard_skills",
      "importance": "preferred",
      "proficiency": "intermediate",
      "yoe": null,
      "referenced_sentence_ids": ["12"]
    }
  ]
}
"""


def supplementary_extractor(state: ExtractorState) -> ExtractorState:
    """Extract complementary keywords to fill coverage gaps."""
    doc_label = state.document_type or "document"
    system_prompt = f"""
    You find complementary keywords missing from an existing keyword list for a {doc_label}.
    Review the original sentences and current merged keywords, then return only NEW skills
    that are supported by the sentences but not already covered.
    Do not repeat skills that are already present.
    Return JSON only using this schema:
    {_SKILL_SCHEMA}
    """

    user_prompt = f"""
    Sentences:
    {format_sentences(state.sentences)}

    Existing merged keywords:
    {format_skills(state.merged_skills)}
    """

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        response_format={"type": "json_object"},
        use_cache=LLM_USE_CACHE,
    )

    existing_names = {skill.name.strip().lower() for skill in state.merged_skills}
    state.supplementary_skills = [
        skill
        for skill in parse_skills(parse_llm_json(response))
        if skill.name.strip().lower() not in existing_names
    ]
    state.is_comprehensive = False
    return state
