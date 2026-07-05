from models.datapoints import DataPoints
from models.extractor_state import ExtractorState
from nodes.common import merge_skills


def phase2_keywords_merger(state: ExtractorState) -> ExtractorState:
    """Deduplicate and merge modified keywords before re-validation."""
    state.datapoints = DataPoints(skills=merge_skills(state.datapoints.skills))
    return state
