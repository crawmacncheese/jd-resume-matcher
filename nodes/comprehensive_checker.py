from langchain_core.messages import HumanMessage, SystemMessage

from models.extractor_state import ExtractorState
from nodes.common import format_sentences, format_skills, parse_llm_json
from utils.llm import LLM_USE_CACHE, invoke_llm


def comprehensive_checker(state: ExtractorState) -> ExtractorState:
    """Check whether the merged keyword list fully represents the source content."""
    state.phase1_iteration += 1

    if state.phase1_iteration >= state.max_phase1_iterations:
        state.is_comprehensive = True
        return state

    system_prompt = """
    You evaluate whether a keyword list comprehensively represents a source document.
    Compare the sentences with the merged keywords and decide if important skills,
    tools, qualifications, or responsibilities are still missing.
    Return JSON only:
    {
      "is_comprehensive": true,
      "missing_topics": ["topic 1", "topic 2"]
    }
    """

    user_prompt = f"""
    Sentences:
    {format_sentences(state.sentences)}

    Merged keywords:
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

    result = parse_llm_json(response)
    state.is_comprehensive = bool(result.get("is_comprehensive", False))
    return state
