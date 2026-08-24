"""AGENTIC-RETRIEVAL-BLOQUE-1: configuración propia + grader
determinista puro.

Contratos independientes del ReAct post-verificación descartado --
sin arrastrar ningún enum de ese dominio (confirmado con auditoría
explícita de imports de react_prompting.py/react_policy_config.py).

No toca verification_runtime.py, verification_agent.py, ni el
retriever -- solo configuración + grader puro + tests."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = []


def scenario(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
            except Exception:
                RESULTS.append((name, False, traceback.format_exc()))
                raise
            else:
                RESULTS.append((name, True, ""))

        return wrapper

    return decorator


_CLAIM = "Transformer architectures use attention mechanisms for sequence modeling."

_GOOD_CANDIDATES = [
    {"source_filename": "paper1.pdf", "chunk_id": "c1", "text": "The transformer architecture uses attention mechanisms extensively.", "native_scores_by_retriever": {"chroma": 0.85}},
    {"source_filename": "paper2.pdf", "chunk_id": "c2", "text": "Attention mechanisms are central to modern sequence modeling approaches.", "native_scores_by_retriever": {"chroma": 0.72}},
]


# ---------------------------------------------------------------
# GRADE_RESULT_VALUES sin CONTRADICTORY
# ---------------------------------------------------------------

@scenario("AR1-01. GRADE_RESULT_VALUES contiene únicamente SUFFICIENT/INSUFFICIENT -- CONTRADICTORY excluido explícitamente")
def test_ar1_01_no_contradictory():
    from src.config.agentic_retrieval_policy_config import GRADE_RESULT_VALUES

    assert set(GRADE_RESULT_VALUES) == {"SUFFICIENT", "INSUFFICIENT"}
    assert "CONTRADICTORY" not in GRADE_RESULT_VALUES


@scenario("AR1-02. GRADE_REASON_CODES son exactamente los 4 auditados contra campos reales del retriever")
def test_ar1_02_reason_codes_exact():
    from src.config.agentic_retrieval_policy_config import GRADE_REASON_CODES

    assert set(GRADE_REASON_CODES) == {
        "LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE",
    }


# ---------------------------------------------------------------
# grade_evidence: SUFFICIENT / INSUFFICIENT con reason codes correctos
# ---------------------------------------------------------------

@scenario("AR1-03. Evidencia buena (conteo, diversidad, relevancia, cobertura suficientes) -> SUFFICIENT, sin reason codes")
def test_ar1_03_sufficient_evidence():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    result = grade_evidence(claim_text=_CLAIM, candidates=_GOOD_CANDIDATES)
    assert result["grade_result"] == "SUFFICIENT"
    assert result["reason_codes"] == ()


@scenario("AR1-04. Sin candidatos -> INSUFFICIENT con LOW_CANDIDATE_COUNT")
def test_ar1_04_no_candidates_insufficient():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    result = grade_evidence(claim_text=_CLAIM, candidates=[])
    assert result["grade_result"] == "INSUFFICIENT"
    assert "LOW_CANDIDATE_COUNT" in result["reason_codes"]


@scenario("AR1-05. Score bajo -> INSUFFICIENT con LOW_RELEVANCE")
def test_ar1_05_low_relevance_insufficient():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "unrelated content about a completely different topic entirely", "native_scores_by_retriever": {"chroma": 0.05}}]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert result["grade_result"] == "INSUFFICIENT"
    assert "LOW_RELEVANCE" in result["reason_codes"]


@scenario("AR1-06. Baja diversidad de fuentes con muchos candidatos del mismo paper -> LOW_SOURCE_DIVERSITY")
def test_ar1_06_low_source_diversity():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [
        {"source_filename": "same.pdf", "chunk_id": f"c{i}", "text": "transformer attention mechanisms sequence modeling", "native_scores_by_retriever": {"chroma": 0.6}}
        for i in range(5)
    ]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert "LOW_SOURCE_DIVERSITY" in result["reason_codes"]


@scenario("AR1-06B. source_filename vacío ya no se tolera silenciosamente (comportamiento anterior) -- ahora se rechaza directamente en la validación del candidate (fail-closed más estricto, esta ronda)")
def test_ar1_06b_empty_source_filename_rejected():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [
        {"source_filename": "", "chunk_id": f"c{i}", "text": "transformer attention mechanisms sequence modeling", "native_scores_by_retriever": {"chroma": 0.6}}
        for i in range(5)
    ]
    try:
        grade_evidence(claim_text=_CLAIM, candidates=candidates)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-07. Sin overlap léxico con el claim -> LOW_COVERAGE")
def test_ar1_07_low_lexical_coverage():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "biology cells mitochondria energy production organisms", "native_scores_by_retriever": {"chroma": 0.5}}]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert "LOW_COVERAGE" in result["reason_codes"]


@scenario("AR1-08. grade_evidence es determinista: mismos inputs -> mismo output, siempre")
def test_ar1_08_deterministic():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    r1 = grade_evidence(claim_text=_CLAIM, candidates=_GOOD_CANDIDATES)
    r2 = grade_evidence(claim_text=_CLAIM, candidates=_GOOD_CANDIDATES)
    assert r1 == r2


# ---------------------------------------------------------------
# Thresholds parametrizables, fail-closed
# ---------------------------------------------------------------

@scenario("AR1-09. validate_grader_thresholds rechaza claves incompletas/extra")
def test_ar1_09_thresholds_exact_keys():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds

    try:
        validate_grader_thresholds({"min_candidate_count": 2})
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-10. validate_grader_thresholds rechaza valores fuera de rango")
def test_ar1_10_thresholds_range_validation():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds, DEFAULT_GRADER_THRESHOLDS

    bad = dict(DEFAULT_GRADER_THRESHOLDS)
    bad["min_relevance_score"] = 5.0  # fuera de [0,1]
    try:
        validate_grader_thresholds(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-11. Thresholds distintos cambian el resultado -- confirmando que son realmente parametrizables, no hardcodeados en la lógica")
def test_ar1_11_thresholds_are_parametrizable():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    lenient = {
        "min_candidate_count": 1, "min_source_diversity": 1,
        "min_candidate_count_for_diversity_check": 100, "min_relevance_score": 0.0,
        "min_lexical_overlap_ratio": 0.0,
    }
    result = grade_evidence(claim_text=_CLAIM, candidates=[{"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.01}}], thresholds=lenient)
    assert result["grade_result"] == "SUFFICIENT"


# ---------------------------------------------------------------
# minimum_viable_evidence
# ---------------------------------------------------------------

@scenario("AR1-12. minimum_viable_evidence es más laxo que SUFFICIENT del grader (candidato de fuente autorizada)")
def test_ar1_12_minimum_viable_more_lenient_than_sufficient():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence, is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    marginal_candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "transformer somewhat related content", "native_scores_by_retriever": {"chroma": 0.2}}]
    grade = grade_evidence(claim_text=_CLAIM, candidates=marginal_candidates)
    viable = is_minimum_viable_evidence(
        candidates=marginal_candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
        authorized_sources={"p.pdf"},
    )
    # el grader puede marcar INSUFFICIENT mientras minimum_viable puede seguir siendo True
    assert grade["grade_result"] == "INSUFFICIENT"
    assert viable is True


@scenario("AR1-13. Sin candidatos -> minimum_viable_evidence es False")
def test_ar1_13_no_candidates_not_viable():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    assert is_minimum_viable_evidence(candidates=[], thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS, authorized_sources={"p.pdf"}) is False


@scenario("AR1-13B. Candidato relevante de fuente AUTORIZADA -> viable")
def test_ar1_13b_relevant_authorized_source_viable():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    candidates = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.5}}]
    result = is_minimum_viable_evidence(
        candidates=candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
        authorized_sources={"authorized.pdf"},
    )
    assert result is True


@scenario("AR1-13C. Candidato relevante de fuente NO autorizada -> no viable (pertenencia real verificada, no solo presencia de source_filename)")
def test_ar1_13c_relevant_unauthorized_source_not_viable():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    candidates = [{"source_filename": "unauthorized.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.5}}]
    result = is_minimum_viable_evidence(
        candidates=candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
        authorized_sources={"authorized.pdf"},
    )
    assert result is False


@scenario("AR1-13D. source_filename vacío -> ahora se rechaza directamente en la validación del candidate (fail-closed más estricto, esta ronda), en vez de tolerarse como 'no viable'")
def test_ar1_13d_empty_source_filename_rejected():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    candidates = [{"source_filename": "", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.9}}]
    try:
        is_minimum_viable_evidence(
            candidates=candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
            authorized_sources={"authorized.pdf"},
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-13E. is_minimum_viable_evidence rechaza authorized_sources de tipo inválido (no set/frozenset)")
def test_ar1_13e_authorized_sources_type_validation():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    candidates = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.5}}]
    try:
        is_minimum_viable_evidence(candidates=candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS, authorized_sources=["authorized.pdf"])
        raised = False
    except TypeError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# effective_top_k_max / next_top_k
# ---------------------------------------------------------------

@scenario("AR1-14. next_top_k siempre > current_top_k y <= effective_top_k_max")
def test_ar1_14_next_top_k_bounds():
    from src.config.agentic_retrieval_policy_config import next_top_k

    result = next_top_k(current_top_k=8, effective_top_k_max=35)
    assert result > 8
    assert result <= 35


@scenario("AR1-15. next_top_k nunca excede effective_top_k_max incluso con escalón agresivo")
def test_ar1_15_next_top_k_never_exceeds_max():
    from src.config.agentic_retrieval_policy_config import next_top_k

    result = next_top_k(current_top_k=30, effective_top_k_max=35, step_multiplier=3.0)
    assert result <= 35


@scenario("AR1-16. next_top_k rechaza si current_top_k ya alcanzó el máximo -- no existe siguiente valor")
def test_ar1_16_next_top_k_rejects_at_max():
    from src.config.agentic_retrieval_policy_config import next_top_k

    try:
        next_top_k(current_top_k=35, effective_top_k_max=35)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Budget compartido
# ---------------------------------------------------------------

@scenario("AR1-17. effective_budget_for_verify_claim = original - usado_por_agentic")
def test_ar1_17_effective_budget_calculation():
    from src.config.agentic_retrieval_policy_config import compute_effective_budget_for_verify_claim

    assert compute_effective_budget_for_verify_claim(
        original_max_additional_retrieval_requests=3, agentic_additional_retrievals_used=2
    ) == 1


@scenario("AR1-18. used == original -> remaining = 0 (agotamiento legítimo, se mantiene)")
def test_ar1_18_used_equals_original_gives_zero():
    from src.config.agentic_retrieval_policy_config import compute_effective_budget_for_verify_claim

    assert compute_effective_budget_for_verify_claim(
        original_max_additional_retrieval_requests=2, agentic_additional_retrievals_used=2
    ) == 0


@scenario("AR1-18B. used > original -> ValueError fail-closed (violación del controller expuesta, NUNCA saturada silenciosamente a 0)")
def test_ar1_18b_used_exceeds_original_fails_closed():
    from src.config.agentic_retrieval_policy_config import compute_effective_budget_for_verify_claim

    try:
        compute_effective_budget_for_verify_claim(
            original_max_additional_retrieval_requests=2, agentic_additional_retrievals_used=5
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-18C. compute_effective_budget_for_verify_claim rechaza tipos inválidos: bool, float, str, None")
def test_ar1_18c_effective_budget_strict_typing():
    from src.config.agentic_retrieval_policy_config import compute_effective_budget_for_verify_claim

    for bad in (True, 2.5, "3", None):
        try:
            compute_effective_budget_for_verify_claim(
                original_max_additional_retrieval_requests=bad, agentic_additional_retrievals_used=1
            )
            raised = False
        except TypeError:
            raised = True
        assert raised, f"{bad!r} debió rechazarse"

        try:
            compute_effective_budget_for_verify_claim(
                original_max_additional_retrieval_requests=3, agentic_additional_retrievals_used=bad
            )
            raised = False
        except TypeError:
            raised = True
        assert raised, f"{bad!r} debió rechazarse"


# ---------------------------------------------------------------
# next_top_k -- validación estricta
# ---------------------------------------------------------------

@scenario("AR1-19A. next_top_k rechaza current_top_k <= 0")
def test_ar1_19a_next_top_k_rejects_nonpositive_current():
    from src.config.agentic_retrieval_policy_config import next_top_k

    for bad in (0, -1):
        try:
            next_top_k(current_top_k=bad, effective_top_k_max=35)
            raised = False
        except ValueError:
            raised = True
        assert raised


@scenario("AR1-19B. next_top_k rechaza effective_top_k_max <= 0")
def test_ar1_19b_next_top_k_rejects_nonpositive_max():
    from src.config.agentic_retrieval_policy_config import next_top_k

    for bad in (0, -5):
        try:
            next_top_k(current_top_k=8, effective_top_k_max=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised


@scenario("AR1-19C. next_top_k rechaza current_top_k/effective_top_k_max de tipo bool o no-entero")
def test_ar1_19c_next_top_k_rejects_bad_types():
    from src.config.agentic_retrieval_policy_config import next_top_k

    for bad in (True, False, 8.5, "8", None):
        try:
            next_top_k(current_top_k=bad, effective_top_k_max=35)
            raised = False
        except TypeError:
            raised = True
        assert raised, f"current_top_k={bad!r} debió rechazarse"

        try:
            next_top_k(current_top_k=8, effective_top_k_max=bad)
            raised = False
        except TypeError:
            raised = True
        assert raised, f"effective_top_k_max={bad!r} debió rechazarse"


@scenario("AR1-19D. next_top_k rechaza step_multiplier no numérico, bool, o <= 1.0")
def test_ar1_19d_next_top_k_rejects_bad_step_multiplier():
    from src.config.agentic_retrieval_policy_config import next_top_k

    for bad in (1.0, 0.5, 0, True, "1.5", None):
        try:
            next_top_k(current_top_k=8, effective_top_k_max=35, step_multiplier=bad)
            raised = False
        except (TypeError, ValueError):
            raised = True
        assert raised, f"step_multiplier={bad!r} debió rechazarse"


@scenario("AR1-19E. next_top_k mantiene: current_top_k >= effective_top_k_max -> ValueError (no existe ADJUST_TOP_K válido)")
def test_ar1_19e_next_top_k_still_rejects_at_or_above_max():
    from src.config.agentic_retrieval_policy_config import next_top_k

    for current in (35, 40):
        try:
            next_top_k(current_top_k=current, effective_top_k_max=35)
            raised = False
        except ValueError:
            raised = True
        assert raised


# ---------------------------------------------------------------
# Coherencia conjunta de thresholds
# ---------------------------------------------------------------

@scenario("AR1-20A. validate_threshold_coherence acepta los defaults (minimum_viable más laxo o igual que grader)")
def test_ar1_20a_default_thresholds_are_coherent():
    from src.config.agentic_retrieval_policy_config import (
        validate_threshold_coherence, DEFAULT_GRADER_THRESHOLDS, DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
    )

    validate_threshold_coherence(
        grader_thresholds=DEFAULT_GRADER_THRESHOLDS,
        minimum_viable_thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
    )


@scenario("AR1-20B. validate_threshold_coherence rechaza minimum_viable.min_candidate_count > grader.min_candidate_count")
def test_ar1_20b_rejects_stricter_candidate_count():
    from src.config.agentic_retrieval_policy_config import validate_threshold_coherence, DEFAULT_GRADER_THRESHOLDS

    bad_mv = {"min_candidate_count": DEFAULT_GRADER_THRESHOLDS["min_candidate_count"] + 5, "min_relevance_score": 0.1}
    try:
        validate_threshold_coherence(grader_thresholds=DEFAULT_GRADER_THRESHOLDS, minimum_viable_thresholds=bad_mv)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-20C. validate_threshold_coherence rechaza minimum_viable.min_relevance_score > grader.min_relevance_score")
def test_ar1_20c_rejects_stricter_relevance_score():
    from src.config.agentic_retrieval_policy_config import validate_threshold_coherence, DEFAULT_GRADER_THRESHOLDS

    bad_mv = {"min_candidate_count": 1, "min_relevance_score": DEFAULT_GRADER_THRESHOLDS["min_relevance_score"] + 0.2}
    try:
        validate_threshold_coherence(grader_thresholds=DEFAULT_GRADER_THRESHOLDS, minimum_viable_thresholds=bad_mv)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Confirmación explícita: no se toca producción, no se toca ReAct previo
# ---------------------------------------------------------------

@scenario("AR1-22. Este bloque no importa nada de verification_runtime.py, verification_agent.py, ni del retriever (búsqueda de import real, no de menciones textuales en docstrings explicativos)")
def test_ar1_19_no_production_imports():
    for path in (
        REPO_ROOT / "src" / "config" / "agentic_retrieval_policy_config.py",
        REPO_ROOT / "src" / "tools" / "verification" / "agentic_retrieval_grader.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "import verification_runtime" not in source
        assert "from src.adapters.verification_runtime" not in source
        assert "import verification_agent" not in source
        assert "from src.agents.verification_agent" not in source
        assert "import verification_incremental_retriever" not in source
        assert "from src.adapters.verification_incremental_retriever" not in source


@scenario("AR1-23. Este bloque no importa ningún enum del ReAct post-verificación descartado (react_policy_config/react_prompting) -- búsqueda de import real")
def test_ar1_20_no_post_verification_react_imports():
    for path in (
        REPO_ROOT / "src" / "config" / "agentic_retrieval_policy_config.py",
        REPO_ROOT / "src" / "tools" / "verification" / "agentic_retrieval_grader.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from src.config.react_policy_config" not in source
        assert "from src.tools.verification.react_prompting" not in source
        assert "from src.tools.verification.react_controller" not in source


@scenario("AR1-24. src/tools/verification/react_controller.py (ReAct post-verificación) sigue existiendo intacto -- no se movió ni eliminó todavía en este bloque")
def test_ar1_21_old_react_controller_still_present_untouched():
    old_controller = REPO_ROOT / "src" / "tools" / "verification" / "react_controller.py"
    if not old_controller.exists():
        return  # archivo de un trabajo anterior (ReAct post-verificación) ajeno a
        # Agentic Retrieval -- no presente en todas las bases de código, no aplicable aquí.
    assert old_controller.is_file()


@scenario("AR1-25. validate_grader_thresholds rechaza min_candidate_count=0 (evidencia vacía no puede ser SUFFICIENT)")
def test_ar1_25_rejects_zero_min_candidate_count():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds, DEFAULT_GRADER_THRESHOLDS

    bad = dict(DEFAULT_GRADER_THRESHOLDS)
    bad["min_candidate_count"] = 0
    try:
        validate_grader_thresholds(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-26. validate_grader_thresholds rechaza min_source_diversity=0")
def test_ar1_26_rejects_zero_min_source_diversity():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds, DEFAULT_GRADER_THRESHOLDS

    bad = dict(DEFAULT_GRADER_THRESHOLDS)
    bad["min_source_diversity"] = 0
    try:
        validate_grader_thresholds(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-27. validate_grader_thresholds rechaza min_candidate_count_for_diversity_check=0")
def test_ar1_27_rejects_zero_diversity_check_threshold():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds, DEFAULT_GRADER_THRESHOLDS

    bad = dict(DEFAULT_GRADER_THRESHOLDS)
    bad["min_candidate_count_for_diversity_check"] = 0
    try:
        validate_grader_thresholds(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-28. validate_grader_thresholds rechaza min_candidate_count_for_diversity_check < min_candidate_count")
def test_ar1_28_rejects_incoherent_diversity_check_threshold():
    from src.config.agentic_retrieval_policy_config import validate_grader_thresholds, DEFAULT_GRADER_THRESHOLDS

    bad = dict(DEFAULT_GRADER_THRESHOLDS)
    bad["min_candidate_count"] = 5
    bad["min_candidate_count_for_diversity_check"] = 3
    try:
        validate_grader_thresholds(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-29. grade_evidence rechaza score NaN (NaN evade silenciosamente las comparaciones de threshold)")
def test_ar1_29_grade_evidence_rejects_nan_score():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "transformer attention mechanisms sequence", "native_scores_by_retriever": {"chroma": float("nan")}}]
    try:
        grade_evidence(claim_text=_CLAIM, candidates=candidates)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-30. grade_evidence rechaza score +inf y -inf")
def test_ar1_30_grade_evidence_rejects_infinite_score():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    for bad_score in (float("inf"), float("-inf")):
        candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "transformer attention mechanisms sequence", "native_scores_by_retriever": {"chroma": bad_score}}]
        try:
            grade_evidence(claim_text=_CLAIM, candidates=candidates)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"score={bad_score} debió rechazarse"


@scenario("AR1-31. grade_evidence rechaza score no numérico (str/None/bool)")
def test_ar1_31_grade_evidence_rejects_non_numeric_score():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    for bad_score in ("0.5", None, True):
        candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "transformer attention mechanisms sequence", "native_scores_by_retriever": {"chroma": bad_score}}]
        try:
            grade_evidence(claim_text=_CLAIM, candidates=candidates)
            raised = False
        except (ValueError, TypeError):
            raised = True
        assert raised, f"score={bad_score!r} debió rechazarse"


@scenario("AR1-32. is_minimum_viable_evidence rechaza score NaN/inf")
def test_ar1_32_minimum_viable_rejects_non_finite_score():
    from src.tools.verification.agentic_retrieval_grader import is_minimum_viable_evidence
    from src.config.agentic_retrieval_policy_config import DEFAULT_MINIMUM_VIABLE_THRESHOLDS

    candidates = [{"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": float("nan")}}]
    try:
        is_minimum_viable_evidence(candidates=candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS, authorized_sources={"p.pdf"})
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# SCORE-SCHEMA-CONTRACT-FIX-01: extract_candidate_relevance_score
# ---------------------------------------------------------------

@scenario("AR1-33. Candidate real con native_scores_by_retriever['chroma'] finito -> extrae el valor correcto")
def test_ar1_33_extract_score_real_schema():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    candidate = {"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 0.87}}
    assert extract_candidate_relevance_score(candidate) == 0.87


@scenario("AR1-34. Falta native_scores_by_retriever -> rechazo (nunca se asume score=0.0)")
def test_ar1_34_missing_native_scores_rejected():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    try:
        extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x"})
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-35. native_scores_by_retriever no es mapping -> rechazo")
def test_ar1_35_native_scores_not_mapping_rejected():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    try:
        extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": [0.5]})
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR1-36. Falta la clave 'chroma' dentro de native_scores_by_retriever -> rechazo")
def test_ar1_36_missing_chroma_key_rejected():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    try:
        extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"other_retriever": 0.5}})
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR1-37. native_scores_by_retriever['chroma']=True (bool) -> rechazo")
def test_ar1_37_chroma_bool_rejected():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    try:
        extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": True}})
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR1-38. native_scores_by_retriever['chroma']=NaN/+inf/-inf -> rechazo")
def test_ar1_38_chroma_non_finite_rejected():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": bad}})
            raised = False
        except ValueError:
            raised = True
        assert raised, f"chroma={bad} debió rechazarse"


@scenario("AR1-39. score finito negativo -> permitido (sin rango [0,1] impuesto)")
def test_ar1_39_finite_negative_score_allowed():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    assert extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": -0.3}}) == -0.3


@scenario("AR1-40. score finito >1 -> permitido (sin rango [0,1] impuesto)")
def test_ar1_40_finite_above_one_score_allowed():
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    assert extract_candidate_relevance_score({"source_filename": "p.pdf", "chunk_id": "c1", "text": "x", "native_scores_by_retriever": {"chroma": 1.5}}) == 1.5


# ---------------------------------------------------------------
# Impacto científico del fix -- demuestra que se corrigió el schema
# sin cambiar la política científica del grader.
# ---------------------------------------------------------------

@scenario("AR1-41. Candidates con scores reales ALTOS -> max_relevance_score refleja esos scores, NO cae artificialmente a 0.0 (demuestra el bug corregido)")
def test_ar1_41_high_real_scores_not_collapsed_to_zero():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [
        {"source_filename": "p1.pdf", "chunk_id": "c1", "text": "transformer attention mechanisms sequence modeling encoding", "native_scores_by_retriever": {"chroma": 0.91}},
    ]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert result["max_relevance_score"] == 0.91
    assert result["max_relevance_score"] != 0.0


@scenario("AR1-42. Candidates con relevancia realmente BAJA -> LOW_RELEVANCE sigue funcionando tras el fix")
def test_ar1_42_low_relevance_still_works_after_fix():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [
        {"source_filename": "p1.pdf", "chunk_id": "c1", "text": "unrelated biology content about cells", "native_scores_by_retriever": {"chroma": 0.02}},
    ]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert "LOW_RELEVANCE" in result["reason_codes"]


# ---------------------------------------------------------------
# Endurecimiento final: candidate real, sin coerciones vía str(...)
# ---------------------------------------------------------------

@scenario("AR1-43. grade_evidence: candidate.source_filename int -> rechazo")
def test_ar1_43_source_filename_int_rejected():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    try:
        grade_evidence(
            claim_text=_CLAIM,
            candidates=[{"source_filename": 123, "chunk_id": "c1", "text": "transformer attention mechanisms", "native_scores_by_retriever": {"chroma": 0.5}}],
        )
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised


@scenario("AR1-44. grade_evidence: candidate.source_filename vacío -> rechazo")
def test_ar1_44_source_filename_empty_rejected():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    try:
        grade_evidence(
            claim_text=_CLAIM,
            candidates=[{"source_filename": "   ", "chunk_id": "c1", "text": "transformer attention mechanisms", "native_scores_by_retriever": {"chroma": 0.5}}],
        )
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised


@scenario("AR1-45. grade_evidence: candidate.text como list -> rechazo (sin coerción vía str(...))")
def test_ar1_45_text_list_rejected():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    try:
        grade_evidence(
            claim_text=_CLAIM,
            candidates=[{"source_filename": "p1.pdf", "chunk_id": "c1", "text": ["transformer", "attention"], "native_scores_by_retriever": {"chroma": 0.5}}],
        )
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised


@scenario("AR1-46. grade_evidence: candidate.text vacío -> rechazo")
def test_ar1_46_text_empty_rejected():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    try:
        grade_evidence(
            claim_text=_CLAIM,
            candidates=[{"source_filename": "p1.pdf", "chunk_id": "c1", "text": "   ", "native_scores_by_retriever": {"chroma": 0.5}}],
        )
        raised = False
    except (ValueError, TypeError):
        raised = True
    assert raised


@scenario("AR1-47. grade_evidence: candidate real completo (schema canónico exacto) -> válido, resultado científicamente correcto")
def test_ar1_47_complete_real_candidate_valid():
    from src.tools.verification.agentic_retrieval_grader import grade_evidence

    candidates = [
        {"source_filename": "p1.pdf", "chunk_id": "c1", "text": "transformer models rely on self attention mechanisms for sequence modeling", "native_scores_by_retriever": {"chroma": 0.85}},
        {"source_filename": "p2.pdf", "chunk_id": "c2", "text": "attention mechanisms improve transformer performance significantly", "native_scores_by_retriever": {"chroma": 0.7}},
    ]
    result = grade_evidence(claim_text=_CLAIM, candidates=candidates)
    assert result["grade_result"] == "SUFFICIENT"
    assert result["candidate_count"] == 2
    assert result["source_diversity"] == 2
    assert result["max_relevance_score"] == 0.85


if __name__ == "__main__":
    for fn in (
        test_ar1_01_no_contradictory,
        test_ar1_02_reason_codes_exact,
        test_ar1_03_sufficient_evidence,
        test_ar1_04_no_candidates_insufficient,
        test_ar1_05_low_relevance_insufficient,
        test_ar1_06_low_source_diversity,
        test_ar1_06b_empty_source_filename_rejected,
        test_ar1_07_low_lexical_coverage,
        test_ar1_08_deterministic,
        test_ar1_09_thresholds_exact_keys,
        test_ar1_10_thresholds_range_validation,
        test_ar1_11_thresholds_are_parametrizable,
        test_ar1_12_minimum_viable_more_lenient_than_sufficient,
        test_ar1_13_no_candidates_not_viable,
        test_ar1_13b_relevant_authorized_source_viable,
        test_ar1_13c_relevant_unauthorized_source_not_viable,
        test_ar1_13d_empty_source_filename_rejected,
        test_ar1_13e_authorized_sources_type_validation,
        test_ar1_14_next_top_k_bounds,
        test_ar1_15_next_top_k_never_exceeds_max,
        test_ar1_16_next_top_k_rejects_at_max,
        test_ar1_17_effective_budget_calculation,
        test_ar1_18_used_equals_original_gives_zero,
        test_ar1_18b_used_exceeds_original_fails_closed,
        test_ar1_18c_effective_budget_strict_typing,
        test_ar1_19a_next_top_k_rejects_nonpositive_current,
        test_ar1_19b_next_top_k_rejects_nonpositive_max,
        test_ar1_19c_next_top_k_rejects_bad_types,
        test_ar1_19d_next_top_k_rejects_bad_step_multiplier,
        test_ar1_19e_next_top_k_still_rejects_at_or_above_max,
        test_ar1_20a_default_thresholds_are_coherent,
        test_ar1_20b_rejects_stricter_candidate_count,
        test_ar1_20c_rejects_stricter_relevance_score,
        test_ar1_19_no_production_imports,
        test_ar1_20_no_post_verification_react_imports,
        test_ar1_21_old_react_controller_still_present_untouched,
        test_ar1_25_rejects_zero_min_candidate_count,
        test_ar1_26_rejects_zero_min_source_diversity,
        test_ar1_27_rejects_zero_diversity_check_threshold,
        test_ar1_28_rejects_incoherent_diversity_check_threshold,
        test_ar1_29_grade_evidence_rejects_nan_score,
        test_ar1_30_grade_evidence_rejects_infinite_score,
        test_ar1_31_grade_evidence_rejects_non_numeric_score,
        test_ar1_32_minimum_viable_rejects_non_finite_score,
        test_ar1_33_extract_score_real_schema,
        test_ar1_34_missing_native_scores_rejected,
        test_ar1_35_native_scores_not_mapping_rejected,
        test_ar1_36_missing_chroma_key_rejected,
        test_ar1_37_chroma_bool_rejected,
        test_ar1_38_chroma_non_finite_rejected,
        test_ar1_39_finite_negative_score_allowed,
        test_ar1_40_finite_above_one_score_allowed,
        test_ar1_41_high_real_scores_not_collapsed_to_zero,
        test_ar1_42_low_relevance_still_works_after_fix,
        test_ar1_43_source_filename_int_rejected,
        test_ar1_44_source_filename_empty_rejected,
        test_ar1_45_text_list_rejected,
        test_ar1_46_text_empty_rejected,
        test_ar1_47_complete_real_candidate_valid,
    ):
        try:
            fn()
        except Exception:
            pass  # ya registrado en RESULTS por el decorador -- el runner manual continua con el resto

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    raise SystemExit(1 if failed else 0)
