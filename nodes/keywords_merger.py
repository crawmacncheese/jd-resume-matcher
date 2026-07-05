from models.extractor_state import ExtractorState
from nodes.common import merge_skills


def keywords_merger(state: ExtractorState) -> ExtractorState:
    """Deduplicate and merge extracted or supplementary keywords."""
    if state.supplementary_skills:
        state.merged_skills = merge_skills(
            state.merged_skills,
            state.supplementary_skills,
        )
        state.supplementary_skills = []
    else:
        state.merged_skills = merge_skills(state.merged_skills, state.extracted_skills)
        state.extracted_skills = []

    return state
