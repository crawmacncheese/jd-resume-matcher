import os
from typing import Callable

import numpy as np

from matcher.local_embeddings import build_local_embedding_fn
from models.datapoints import Skill
from models.match_result import MatchResult, SkillPair

DEFAULT_COSINE_THRESHOLD = float(os.getenv("MATCH_COSINE_THRESHOLD", "0.3").strip())
DEFAULT_MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.5").strip())


def keyword_weight(_skill: Skill) -> float:
    """Keyword weight w(x). Evenly weighted until TF-IDF is implemented."""
    return 1.0


def sigma_scale(skill_a: Skill, skill_b: Skill) -> float:
    """
    Proficiency scale σ(a,b).

    σ(a,b) = min(1, yoe_a / yoe_b) when yoe_b > 0, else 1.
    """
    yoe_b = skill_b.get_yoe()
    if yoe_b > 0:
        return min(1.0, skill_a.get_yoe() / yoe_b)
    return 1.0


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def sim_tau(
    skill_a: Skill,
    skill_b: Skill,
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    cosine_threshold: float,
) -> float:
    """
    Directional keyword similarity sim_τ(a,b).

    0 when cos(a,b) < τ, otherwise ((cos + 1) / 2) * σ(a,b).
    """
    cosine = cosine_similarity(embedding_a, embedding_b)
    if cosine < cosine_threshold:
        return 0.0
    return ((cosine + 1.0) / 2.0) * sigma_scale(skill_a, skill_b)


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _get_embedding(
    skill: Skill,
    cache: dict[str, np.ndarray],
    embedding_fn: Callable[[str], np.ndarray],
) -> np.ndarray:
    key = skill.name.strip().lower()
    if key not in cache:
        cache[key] = embedding_fn(skill.name)
    return cache[key]


def _directional_score(
    source_skills: list[Skill],
    target_skills: list[Skill],
    embeddings: dict[str, np.ndarray],
    embedding_fn: Callable[[str], np.ndarray],
    cosine_threshold: float,
) -> float:
    if not source_skills:
        return 0.0

    total_weight = sum(keyword_weight(skill) for skill in source_skills)
    if total_weight == 0.0:
        return 0.0

    weighted_sum = 0.0
    for source_skill in source_skills:
        source_embedding = _get_embedding(source_skill, embeddings, embedding_fn)
        best_similarity = 0.0
        for target_skill in target_skills:
            target_embedding = _get_embedding(target_skill, embeddings, embedding_fn)
            best_similarity = max(
                best_similarity,
                sim_tau(
                    source_skill,
                    target_skill,
                    source_embedding,
                    target_embedding,
                    cosine_threshold,
                ),
            )
        weighted_sum += keyword_weight(source_skill) * best_similarity

    return weighted_sum / total_weight


def _best_target_match(
    source_skill: Skill,
    target_skills: list[Skill],
    embeddings: dict[str, np.ndarray],
    embedding_fn: Callable[[str], np.ndarray],
    cosine_threshold: float,
) -> tuple[Skill | None, float]:
    source_embedding = _get_embedding(source_skill, embeddings, embedding_fn)
    best_skill: Skill | None = None
    best_similarity = 0.0

    for target_skill in target_skills:
        target_embedding = _get_embedding(target_skill, embeddings, embedding_fn)
        similarity = sim_tau(
            source_skill,
            target_skill,
            source_embedding,
            target_embedding,
            cosine_threshold,
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_skill = target_skill

    return best_skill, best_similarity


def _resolve_embedding_fn(
    jd_skills: list[Skill],
    resume_skills: list[Skill],
    embedding_fn: Callable[[str], np.ndarray] | None,
) -> Callable[[str], np.ndarray]:
    if embedding_fn is not None:
        return embedding_fn

    has_remote_embeddings = bool(
        os.getenv("LLM_EMBEDDING_API_KEY") and os.getenv("LLM_EMBEDDING_BASE_URL")
    )
    if has_remote_embeddings:
        from utils.llm import get_embedding

        return get_embedding

    return build_local_embedding_fn(jd_skills + resume_skills)


def run_matcher(
    jd_skills: list[Skill],
    resume_skills: list[Skill],
    *,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    embedding_fn: Callable[[str], np.ndarray] | None = None,
) -> MatchResult:
    """
    BERTScore-style matcher between JD keywords (B) and resume keywords (A).

    P(A,B) averages, over JD keywords b, the best sim(b,a) toward resume keywords.
    R(A,B) averages, over resume keywords a, the best sim(a,b) toward JD keywords.
    F1 is the harmonic mean of precision and recall.

  Uses OpenAI-compatible embeddings when LLM_EMBEDDING_API_KEY is set;
  otherwise falls back to local TF-IDF vectors (works with DeepSeek-only setups).
    """
    embed = _resolve_embedding_fn(jd_skills, resume_skills, embedding_fn)
    embeddings: dict[str, np.ndarray] = {}

    precision = _directional_score(
        jd_skills,
        resume_skills,
        embeddings,
        embed,
        cosine_threshold,
    )
    recall = _directional_score(
        resume_skills,
        jd_skills,
        embeddings,
        embed,
        cosine_threshold,
    )
    f1 = _compute_f1(precision, recall)

    matched_pairs: list[SkillPair] = []
    missing_from_resume: list[Skill] = []

    for jd_skill in jd_skills:
        resume_skill, similarity = _best_target_match(
            jd_skill,
            resume_skills,
            embeddings,
            embed,
            cosine_threshold,
        )
        if resume_skill is None or similarity == 0.0:
            missing_from_resume.append(jd_skill)
            continue

        matched_pairs.append(
            SkillPair(
                jd_skill=jd_skill,
                resume_skill=resume_skill,
                similarity=round(similarity, 4),
                meets_yoe=sigma_scale(resume_skill, jd_skill) >= 1.0,
            )
        )

    extra_on_resume: list[Skill] = []
    for resume_skill in resume_skills:
        _, similarity = _best_target_match(
            resume_skill,
            jd_skills,
            embeddings,
            embed,
            cosine_threshold,
        )
        if similarity == 0.0:
            extra_on_resume.append(resume_skill)

    return MatchResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched=matched_pairs,
        missing_from_resume=missing_from_resume,
        extra_on_resume=extra_on_resume,
        jd_skill_count=len(jd_skills),
        resume_skill_count=len(resume_skills),
        matched_count=len(matched_pairs),
        is_match=f1 >= match_threshold,
    )
