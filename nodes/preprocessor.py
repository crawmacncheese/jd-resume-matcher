import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.sentence import Sentence
from models.extractor_state import ExtractorState
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
)


def preprocessor(state: ExtractorState) -> ExtractorState:
    """Pre-process the input text into sentences."""
    if not state.document:
        raise ValueError("Document is empty")

    system_prompt = """
    You are a helpful assistant that splits the given text into sentences.
    Ignore meaningless sentences like title, header, footer, etc.
    Return your answer as JSON with this schema:
    {"sentences": ["sentence 1", "sentence 2", ...]}
    """

    user_prompt = state.document

    response = invoke_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        response_format={"type": "json_object"},
        use_cache=LLM_USE_CACHE,
    )

    content = response.content
    if isinstance(content, list):
        content = content[0]["text"]

    state.sentences = [
        Sentence(id=index, sentence=sentence)
        for index, sentence in enumerate(json.loads(content)["sentences"])
    ]
    return state
