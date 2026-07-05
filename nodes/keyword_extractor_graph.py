from typing import Literal

from langgraph.graph import END, StateGraph

from models.datapoints import DataPoints
from models.extractor_state import ExtractorState
from nodes.comprehensive_checker import comprehensive_checker
from nodes.keywords_extractor import keywords_extractor
from nodes.keywords_merger import keywords_merger
from nodes.modifier import modifier
from nodes.preprocessor import preprocessor
from nodes.supplementary_extractor import supplementary_extractor
from nodes.validator import validator


def _route_phase1(
    state: ExtractorState,
) -> Literal["supplementary_extractor", "finalize_phase1"]:
    if state.is_comprehensive:
        return "finalize_phase1"
    return "supplementary_extractor"


def _route_phase2(state: ExtractorState) -> Literal["modifier", "end"]:
    if state.is_valid:
        return "end"
    return "modifier"


def finalize_phase1(state: ExtractorState) -> ExtractorState:
    """Hand merged keywords to Phase 2."""
    state.datapoints = DataPoints(skills=state.merged_skills)
    return state


def build_phase1_graph() -> StateGraph:
    """Construct the LangGraph for Phase 1 with looping until comprehensive or max_iter."""
    workflow = StateGraph(ExtractorState)

    workflow.add_node("preprocessor", preprocessor)
    workflow.add_node("keywords_extractor", keywords_extractor)
    workflow.add_node("keywords_merger", keywords_merger)
    workflow.add_node("comprehensive_checker", comprehensive_checker)
    workflow.add_node("supplementary_extractor", supplementary_extractor)
    workflow.add_node("finalize_phase1", finalize_phase1)

    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", "keywords_extractor")
    workflow.add_edge("keywords_extractor", "keywords_merger")
    workflow.add_edge("keywords_merger", "comprehensive_checker")
    workflow.add_conditional_edges(
        "comprehensive_checker",
        _route_phase1,
        {
            "supplementary_extractor": "supplementary_extractor",
            "finalize_phase1": "finalize_phase1",
        },
    )
    workflow.add_edge("supplementary_extractor", "keywords_merger")
    workflow.add_edge("finalize_phase1", END)

    return workflow


def build_phase2_graph() -> StateGraph:
    """Construct the LangGraph for Phase 2 with looping until valid or max_iter."""
    workflow = StateGraph(ExtractorState)

    workflow.add_node("validator", validator)
    workflow.add_node("modifier", modifier)

    workflow.set_entry_point("validator")
    workflow.add_conditional_edges(
        "validator",
        _route_phase2,
        {
            "modifier": "modifier",
            "end": END,
        },
    )
    workflow.add_edge("modifier", "validator")

    return workflow


def build_extractor_graph() -> StateGraph:
    """Construct the full Keywords Extractor workflow (Phase 1 + Phase 2)."""
    workflow = StateGraph(ExtractorState)

    workflow.add_node("preprocessor", preprocessor)
    workflow.add_node("keywords_extractor", keywords_extractor)
    workflow.add_node("keywords_merger", keywords_merger)
    workflow.add_node("comprehensive_checker", comprehensive_checker)
    workflow.add_node("supplementary_extractor", supplementary_extractor)
    workflow.add_node("finalize_phase1", finalize_phase1)
    workflow.add_node("validator", validator)
    workflow.add_node("modifier", modifier)

    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", "keywords_extractor")
    workflow.add_edge("keywords_extractor", "keywords_merger")
    workflow.add_edge("keywords_merger", "comprehensive_checker")
    workflow.add_conditional_edges(
        "comprehensive_checker",
        _route_phase1,
        {
            "supplementary_extractor": "supplementary_extractor",
            "finalize_phase1": "finalize_phase1",
        },
    )
    workflow.add_edge("supplementary_extractor", "keywords_merger")
    workflow.add_edge("finalize_phase1", "validator")
    workflow.add_conditional_edges(
        "validator",
        _route_phase2,
        {
            "modifier": "modifier",
            "end": END,
        },
    )
    workflow.add_edge("modifier", "validator")

    return workflow


def _initial_state(
    document_text: str,
    *,
    document_type: str | None = None,
    max_phase1_iterations: int = 3,
    max_phase2_iterations: int = 3,
) -> ExtractorState:
    return ExtractorState(
        document=document_text,
        document_type=document_type,
        sentences=[],
        datapoints=DataPoints(skills=[]),
        max_phase1_iterations=max_phase1_iterations,
        max_phase2_iterations=max_phase2_iterations,
    )


def run_phase1(
    document_text: str,
    *,
    document_type: str | None = None,
    max_iter: int = 3,
) -> ExtractorState:
    """Run Phase 1 end-to-end and return the merged, comprehensive ExtractorState."""
    workflow = build_phase1_graph()
    app = workflow.compile()
    result = app.invoke(
        _initial_state(
            document_text,
            document_type=document_type,
            max_phase1_iterations=max_iter,
        )
    )
    return ExtractorState.model_validate(result)


def run_phase2(
    state: ExtractorState,
    *,
    max_iter: int | None = None,
) -> ExtractorState:
    """Run Phase 2 validation and polishing on a Phase 1 state."""
    if max_iter is not None:
        state.max_phase2_iterations = max_iter

    workflow = build_phase2_graph()
    app = workflow.compile()
    result = app.invoke(state)
    return ExtractorState.model_validate(result)


def run_extractor(
    document_text: str,
    *,
    document_type: str | None = None,
    max_phase1_iterations: int = 3,
    max_phase2_iterations: int = 3,
) -> ExtractorState:
    """Run the full Keywords Extractor workflow for a JD or resume."""
    workflow = build_extractor_graph()
    app = workflow.compile()
    result = app.invoke(
        _initial_state(
            document_text,
            document_type=document_type,
            max_phase1_iterations=max_phase1_iterations,
            max_phase2_iterations=max_phase2_iterations,
        )
    )
    return ExtractorState.model_validate(result)
