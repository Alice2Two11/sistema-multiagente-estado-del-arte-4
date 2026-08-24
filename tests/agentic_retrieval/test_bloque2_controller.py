"""AGENTIC-RETRIEVAL-BLOQUE-2: controller (Observation, gate de
acciones, ciclo).

Vigilancia especial pedida: el planner SOLO tiene decisiones reales
cuando existen 2+ opciones válidas (REWRITE_QUERY vs ADJUST_TOP_K), y
ACCEPT_EVIDENCE NUNCA aparece antes de lo acordado (nunca en la
primera insuficiencia, nunca sin minimum_viable_evidence).

NO integra todavía con verification_runtime.py, verification_agent.py,
ni el retriever real -- eso es un bloque posterior."""

from __future__ import annotations

import json
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


def _make_observation(**overrides):
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    defaults = dict(
        claim_id="c1", claim_text="test claim", current_query="test claim", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=3,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
        minimum_viable_evidence=True, query_rewrite_count=0,
    )
    defaults.update(overrides)
    return AgenticRetrievalObservation(**defaults)


def _planner_returning(*responses):
    calls = {"i": 0}

    def fn(prompt):
        idx = calls["i"]
        calls["i"] += 1
        payload = responses[idx] if idx < len(responses) else responses[-1]
        return payload if isinstance(payload, str) else json.dumps(payload)

    return fn


# ---------------------------------------------------------------
# SUFFICIENT -> ACCEPT_EVIDENCE automático, sin planner
# ---------------------------------------------------------------

@scenario("AR2-01. grade_result=SUFFICIENT -> ACCEPT_EVIDENCE forzado, planner NUNCA consultado")
def test_ar2_01_sufficient_forces_accept_without_planner():
    from src.tools.verification.agentic_retrieval_controller import (
        determine_forced_outcome, compute_allowed_actions, run_agentic_retrieval_cycle,
    )

    obs = _make_observation(grade_result="SUFFICIENT", reason_codes=())
    assert determine_forced_outcome(obs) == "ACCEPT_EVIDENCE"
    assert compute_allowed_actions(obs) == ()

    def planner_should_not_be_called(prompt):
        raise AssertionError("no debe invocarse el planner si grade_result=SUFFICIENT")

    result = run_agentic_retrieval_cycle(
        initial_observation=obs, invoke_planner_fn=planner_should_not_be_called,
        execute_action_fn=lambda a, d, o: o,
    )
    assert result.outcome == "ACCEPT_EVIDENCE"
    assert result.steps == []


# ---------------------------------------------------------------
# Primera insuficiencia: SOLO REWRITE_QUERY/ADJUST_TOP_K, nunca ACCEPT_EVIDENCE
# ---------------------------------------------------------------

@scenario("AR2-02. Primera insuficiencia (retrieval_round=0): allowed_actions == {REWRITE_QUERY, ADJUST_TOP_K} exactamente, ACCEPT_EVIDENCE ausente incluso con minimum_viable_evidence=True")
def test_ar2_02_first_insufficiency_excludes_accept():
    from src.tools.verification.agentic_retrieval_controller import compute_allowed_actions

    obs = _make_observation(retrieval_round=0, minimum_viable_evidence=True)
    actions = compute_allowed_actions(obs)
    assert set(actions) == {"REWRITE_QUERY", "ADJUST_TOP_K"}
    assert "ACCEPT_EVIDENCE" not in actions


@scenario("AR2-03. Segunda insuficiencia (retrieval_round=1) con minimum_viable_evidence=True: ACCEPT_EVIDENCE se habilita junto a REWRITE_QUERY/ADJUST_TOP_K")
def test_ar2_03_second_insufficiency_enables_accept_when_viable():
    from src.tools.verification.agentic_retrieval_controller import compute_allowed_actions

    obs = _make_observation(retrieval_round=1, minimum_viable_evidence=True)
    actions = compute_allowed_actions(obs)
    assert set(actions) == {"REWRITE_QUERY", "ADJUST_TOP_K", "ACCEPT_EVIDENCE"}


@scenario("AR2-04. Segunda insuficiencia con minimum_viable_evidence=False: ACCEPT_EVIDENCE sigue sin aparecer")
def test_ar2_04_second_insufficiency_no_accept_if_not_viable():
    from src.tools.verification.agentic_retrieval_controller import compute_allowed_actions

    obs = _make_observation(retrieval_round=1, minimum_viable_evidence=False)
    actions = compute_allowed_actions(obs)
    assert "ACCEPT_EVIDENCE" not in actions
    assert set(actions) == {"REWRITE_QUERY", "ADJUST_TOP_K"}


# ---------------------------------------------------------------
# Presupuesto agotado -- forzado, nunca planner
# ---------------------------------------------------------------

@scenario("AR2-05. Presupuesto agotado + minimum_viable_evidence=True -> ACCEPT_EVIDENCE forzado, planner nunca consultado")
def test_ar2_05_budget_exhausted_viable_forces_accept():
    from src.tools.verification.agentic_retrieval_controller import (
        determine_forced_outcome, run_agentic_retrieval_cycle,
    )

    obs = _make_observation(remaining_retrieval_budget=0, minimum_viable_evidence=True)
    assert determine_forced_outcome(obs) == "ACCEPT_EVIDENCE"

    def planner_should_not_be_called(prompt):
        raise AssertionError("no debe invocarse con budget agotado")

    result = run_agentic_retrieval_cycle(
        initial_observation=obs, invoke_planner_fn=planner_should_not_be_called, execute_action_fn=lambda a, d, o: o,
    )
    assert result.outcome == "ACCEPT_EVIDENCE"
    assert result.steps == []


@scenario("AR2-06. Presupuesto agotado + minimum_viable_evidence=False -> FINISH_UNRESOLVED forzado, planner nunca consultado")
def test_ar2_06_budget_exhausted_not_viable_forces_unresolved():
    from src.tools.verification.agentic_retrieval_controller import (
        determine_forced_outcome, run_agentic_retrieval_cycle, FINISH_UNRESOLVED,
    )

    obs = _make_observation(remaining_retrieval_budget=0, minimum_viable_evidence=False, candidate_count=0, evidence_ids=(), max_relevance_score=0.0)
    assert determine_forced_outcome(obs) == FINISH_UNRESOLVED

    def planner_should_not_be_called(prompt):
        raise AssertionError("no debe invocarse con budget agotado")

    result = run_agentic_retrieval_cycle(
        initial_observation=obs, invoke_planner_fn=planner_should_not_be_called, execute_action_fn=lambda a, d, o: o,
    )
    assert result.outcome == FINISH_UNRESOLVED


# ---------------------------------------------------------------
# ADJUST_TOP_K estructuralmente restringido por effective_top_k_max
# ---------------------------------------------------------------

@scenario("AR2-07. current_top_k == effective_top_k_max: ADJUST_TOP_K ausente, REWRITE_QUERY sigue disponible")
def test_ar2_07_adjust_top_k_unavailable_at_max():
    from src.tools.verification.agentic_retrieval_controller import compute_allowed_actions

    obs = _make_observation(current_top_k=35, effective_top_k_max=35)
    actions = compute_allowed_actions(obs)
    assert "ADJUST_TOP_K" not in actions
    assert "REWRITE_QUERY" in actions


# ---------------------------------------------------------------
# Enums cerrados, sin rationale libre
# ---------------------------------------------------------------

@scenario("AR2-08. AGENTIC_RETRIEVAL_ACTIONS son exactamente REWRITE_QUERY, ADJUST_TOP_K, ACCEPT_EVIDENCE -- ni RETRIEVE ni GRADE_EVIDENCE (consecuencias obligatorias, no acciones del planner)")
def test_ar2_08_actions_enum_exact():
    from src.tools.verification.agentic_retrieval_controller import AGENTIC_RETRIEVAL_ACTIONS

    assert set(AGENTIC_RETRIEVAL_ACTIONS) == {"REWRITE_QUERY", "ADJUST_TOP_K", "ACCEPT_EVIDENCE"}
    assert "RETRIEVE" not in AGENTIC_RETRIEVAL_ACTIONS
    assert "GRADE_EVIDENCE" not in AGENTIC_RETRIEVAL_ACTIONS


@scenario("AR2-09. parse_agentic_planner_response rechaza acción fuera de allowed_actions")
def test_ar2_09_action_outside_allowed_rejected():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    raw = json.dumps({"selected_action": "ACCEPT_EVIDENCE", "decision_basis": "EVIDENCE_ACCEPTABLE_DESPITE_GAPS"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY", "ADJUST_TOP_K"))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


@scenario("AR2-10. parse_agentic_planner_response rechaza campo extra (rationale)")
def test_ar2_10_extra_field_rejected():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    raw = json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", "rationale": "porque si"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY",))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


@scenario("AR2-11. decision_basis fuera del enum -> rechazado")
def test_ar2_11_invalid_decision_basis_rejected():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    raw = json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "porque me parece"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY",))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Ciclo completo end-to-end
# ---------------------------------------------------------------

@scenario("AR2-12. Ciclo completo: primera insuficiencia -> planner elige REWRITE_QUERY -> nueva Observation SUFFICIENT -> ACCEPT_EVIDENCE forzado (sin segunda consulta al planner)")
def test_ar2_12_full_cycle_rewrite_then_sufficient():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle,
    )

    planner = _planner_returning(
        {"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"},
    )

    def execute(action, decision_basis, observation):
        assert action == "REWRITE_QUERY"
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="rewritten", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=3, evidence_ids=("e1", "e2", "e3"), max_relevance_score=0.8,
            grade_result="SUFFICIENT", reason_codes=(), minimum_viable_evidence=True,
            query_rewrite_count=observation.query_rewrite_count + 1,
        )

    initial = _make_observation()
    result = run_agentic_retrieval_cycle(initial_observation=initial, invoke_planner_fn=planner, execute_action_fn=execute)
    assert result.outcome == "ACCEPT_EVIDENCE"
    assert len(result.steps) == 1
    assert result.steps[0]["selected_action"] == "REWRITE_QUERY"


@scenario("AR2-13. Ciclo completo: 2 rondas de mejora insuficientes, planner finalmente elige ACCEPT_EVIDENCE explícitamente (segunda insuficiencia, viable)")
def test_ar2_13_full_cycle_planner_accepts_explicitly():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle,
    )

    planner = _planner_returning(
        {"selected_action": "ADJUST_TOP_K", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"},
        {"selected_action": "ACCEPT_EVIDENCE", "decision_basis": "EVIDENCE_ACCEPTABLE_DESPITE_GAPS"},
    )

    def execute(action, decision_basis, observation):
        assert action == "ADJUST_TOP_K"
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query, retrieval_round=observation.retrieval_round + 1,
            current_top_k=12, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=2, evidence_ids=("e1", "e2"), max_relevance_score=0.25,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            query_rewrite_count=observation.query_rewrite_count,
        )

    initial = _make_observation(retrieval_round=0)
    result = run_agentic_retrieval_cycle(initial_observation=initial, invoke_planner_fn=planner, execute_action_fn=execute)
    assert result.outcome == "ACCEPT_EVIDENCE"
    assert len(result.steps) == 2
    assert result.steps[1]["selected_action"] == "ACCEPT_EVIDENCE"


@scenario("AR2-14. Planner con JSON malformado tras agotar reintentos -> AGENTIC_PLANNER_FAILED, sin excepción no controlada")
def test_ar2_14_malformed_planner_fails_closed():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AGENTIC_PLANNER_FAILED

    def bad_planner(prompt):
        return "esto no es json"

    initial = _make_observation()
    result = run_agentic_retrieval_cycle(initial_observation=initial, invoke_planner_fn=bad_planner, execute_action_fn=lambda a, d, o: o)
    assert result.outcome == AGENTIC_PLANNER_FAILED
    assert result.steps == []


@scenario("AR2-15. build_agentic_planner_prompt incluye únicamente las acciones autorizadas -- nunca ACCEPT_EVIDENCE si no está en allowed_actions")
def test_ar2_15_prompt_only_lists_allowed_actions():
    from src.tools.verification.agentic_retrieval_controller import build_agentic_planner_prompt

    obs = _make_observation(retrieval_round=0)
    allowed = ("REWRITE_QUERY", "ADJUST_TOP_K")
    prompt = build_agentic_planner_prompt(observation=obs, allowed_actions=allowed)
    assert json.dumps(list(allowed)) in prompt


@scenario("AR2-16. Observation rechaza current_top_k > effective_top_k_max al construirse")
def test_ar2_16_observation_rejects_top_k_exceeding_max():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=40, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=(), minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 1: el planner NO se invoca con una sola acción válida
# ---------------------------------------------------------------

@scenario("AR2-17. Con allowed_actions de tamaño 1 (top_k ya en máximo, primera insuficiencia), Python ejecuta directamente -- el planner lanza AssertionError si se le invoca, y NUNCA se invoca")
def test_ar2_17_single_action_never_invokes_planner():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, compute_allowed_actions, run_agentic_retrieval_cycle,
    )

    obs = _make_observation(current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=1)
    assert compute_allowed_actions(obs) == ("REWRITE_QUERY",)

    def planner_should_not_be_called(prompt):
        raise AssertionError("el planner no debe invocarse cuando solo existe una acción válida")

    def execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="rewritten different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner_should_not_be_called, execute_action_fn=execute)
    assert result.steps[0]["planner_invoked"] is False
    assert result.steps[0]["selected_action"] == "REWRITE_QUERY"


@scenario("AR2-18. decision_basis derivado determinísticamente cuando el planner no se invoca -- coincide con el primer reason_code de la Observation")
def test_ar2_18_deterministic_decision_basis_matches_reason_code():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation, run_agentic_retrieval_cycle

    obs = _make_observation(
        current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=1,
        reason_codes=("LOW_COVERAGE",),
    )

    def execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="rewritten different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=lambda p: (_ for _ in ()).throw(AssertionError()), execute_action_fn=execute)
    assert result.steps[0]["decision_basis"] == "EVIDENCE_INSUFFICIENT_LOW_COVERAGE"


@scenario("AR2-19. Con 2+ acciones disponibles, el planner SÍ se invoca (planner_invoked=True)")
def test_ar2_19_multiple_actions_invokes_planner():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle

    obs = _make_observation(retrieval_round=0, current_top_k=8, effective_top_k_max=35)
    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=lambda a, d, o: _make_observation(retrieval_round=1, remaining_retrieval_budget=o.remaining_retrieval_budget - 1, current_query="different query now", query_rewrite_count=1))
    assert result.steps[0]["planner_invoked"] is True


# ---------------------------------------------------------------
# Punto 2: validación de la transición producida por execute_action_fn
# ---------------------------------------------------------------

@scenario("AR2-20. REWRITE_QUERY: si la tool produce retrieval_round no incrementado (BUG) junto con query_rewrite_count sí incrementado, la propia Observation lo rechaza al construirse (invariante query_rewrite_count <= retrieval_round, cerrado en esta ronda) -- la excepción se propaga sin convertirse en AGENTIC_TRANSITION_INVALID")
def test_ar2_20_rewrite_query_wrong_round_increment():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = _make_observation(current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=1)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="different query", retrieval_round=observation.retrieval_round,  # BUG: no incrementa
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    try:
        bad_execute("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-21. REWRITE_QUERY: remaining_retrieval_budget no decrementado -> AGENTIC_TRANSITION_INVALID")
def test_ar2_21_rewrite_query_budget_not_decremented():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget,  # BUG: no decrementa
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=lambda p: "", execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-22. REWRITE_QUERY: current_query no cambia -> AGENTIC_TRANSITION_INVALID")
def test_ar2_22_rewrite_query_unchanged_query():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query,  # BUG: query identica
            retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=lambda p: "", execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-23. REWRITE_QUERY: si la tool produce query_rewrite_count no incrementado (BUG) junto con current_query realmente cambiado, la propia Observation lo rechaza al construirse (invariante query_rewrite_count==0 implica current_query==claim_text, cerrado en esta ronda) -- la excepción se propaga sin convertirse en AGENTIC_TRANSITION_INVALID")
def test_ar2_23_rewrite_query_count_not_incremented():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = _make_observation(current_top_k=35, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count,  # BUG: no incrementa
        )

    try:
        bad_execute("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-24. REWRITE_QUERY: current_top_k cambiado indebidamente -> AGENTIC_TRANSITION_INVALID")
def test_ar2_24_rewrite_query_top_k_changed():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k + 4,  # BUG: REWRITE_QUERY no debe tocar top_k
            effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-25. ADJUST_TOP_K: current_top_k no aumenta -> AGENTIC_TRANSITION_INVALID")
def test_ar2_25_adjust_top_k_not_increased():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query, retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k,  # BUG: no aumenta
            effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count,
        )

    planner = _planner_returning({"selected_action": "ADJUST_TOP_K", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-26. ADJUST_TOP_K: si la tool produce current_top_k > effective_top_k_max, la propia Observation lo rechaza al construirse (invariante ya cubierta en __post_init__, AR2-16) -- la excepción se propaga sin convertirse silenciosamente en AGENTIC_TRANSITION_INVALID, porque sería un bug más fundamental en la tool")
def test_ar2_26_adjust_top_k_exceeds_max():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query, retrieval_round=observation.retrieval_round + 1,
            current_top_k=40,  # BUG: excede effective_top_k_max=35 -- Observation.__post_init__ ya lo bloquea
            effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="SUFFICIENT", reason_codes=(),
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count,
        )

    try:
        bad_execute("ADJUST_TOP_K", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-27. ADJUST_TOP_K: query_rewrite_count cambiado indebidamente -> AGENTIC_TRANSITION_INVALID")
def test_ar2_27_adjust_top_k_rewrite_count_changed():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query, retrieval_round=observation.retrieval_round + 1,
            current_top_k=12, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True,
            query_rewrite_count=observation.query_rewrite_count + 1,  # BUG: ADJUST_TOP_K no debe tocar esto
        )

    planner = _planner_returning({"selected_action": "ADJUST_TOP_K", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


# ---------------------------------------------------------------
# Punto 3: Observation endurecida
# ---------------------------------------------------------------

@scenario("AR2-28. Observation rechaza bool donde se espera int (True/False no deben colarse como 1/0)")
def test_ar2_28_rejects_bool_for_int_fields():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    for bad in (True, False):
        try:
            AgenticRetrievalObservation(
                claim_id="c1", claim_text="x", current_query="x", retrieval_round=bad,
                current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
                candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
                grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
            )
            raised = False
        except TypeError:
            raised = True
        assert raised, f"retrieval_round={bad!r} debió rechazarse"


@scenario("AR2-29. Observation rechaza campos negativos (retrieval_round, remaining_retrieval_budget, candidate_count, query_rewrite_count)")
def test_ar2_29_rejects_negative_fields():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    base = dict(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
    )
    for field_name in ("retrieval_round", "remaining_retrieval_budget", "candidate_count", "query_rewrite_count"):
        bad = dict(base)
        bad[field_name] = -1
        try:
            AgenticRetrievalObservation(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"{field_name}=-1 debió rechazarse"


@scenario("AR2-30. Observation rechaza current_top_k <= 0 y effective_top_k_max <= 0")
def test_ar2_30_rejects_non_positive_top_k_fields():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    base = dict(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
    )
    for field_name in ("current_top_k", "effective_top_k_max"):
        bad = dict(base)
        bad[field_name] = 0
        try:
            AgenticRetrievalObservation(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"{field_name}=0 debió rechazarse"


@scenario("AR2-31. Observation rechaza max_relevance_score fuera de [0,1]")
def test_ar2_31_rejects_relevance_score_out_of_range():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    for bad_score in (-0.1, 1.1):
        try:
            AgenticRetrievalObservation(
                claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
                current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
                candidate_count=1, evidence_ids=("e1",), max_relevance_score=bad_score,
                grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
            )
            raised = False
        except ValueError:
            raised = True
        assert raised, f"max_relevance_score={bad_score} debió rechazarse"


@scenario("AR2-32. Observation rechaza claim_id/claim_text/current_query vacíos")
def test_ar2_32_rejects_empty_string_fields():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    base = dict(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
    )
    for field_name in ("claim_id", "claim_text", "current_query"):
        bad = dict(base)
        bad[field_name] = "   "
        try:
            AgenticRetrievalObservation(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"{field_name} vacío debió rechazarse"


@scenario("AR2-33. Observation rechaza reason_codes fuera de GRADE_REASON_CODES (Bloque 1, reutilizado sin duplicar)")
def test_ar2_33_rejects_reason_codes_outside_grade_reason_codes():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("INVENTED_REASON",), minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-34. Coherencia grade_result/reason_codes: SUFFICIENT con reason_codes no vacío -> rechazado")
def test_ar2_34_sufficient_with_reason_codes_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="SUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-35. Coherencia grade_result/reason_codes: INSUFFICIENT con reason_codes vacío -> rechazado")
def test_ar2_35_insufficient_without_reason_codes_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=(), minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 4: coherencia selected_action / decision_basis
# ---------------------------------------------------------------

@scenario("AR2-36. ACCEPT_EVIDENCE con decision_basis distinto de EVIDENCE_ACCEPTABLE_DESPITE_GAPS -> rechazado")
def test_ar2_36_accept_evidence_wrong_decision_basis_rejected():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    raw = json.dumps({"selected_action": "ACCEPT_EVIDENCE", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY", "ACCEPT_EVIDENCE"), observation_reason_codes=("LOW_RELEVANCE",))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


@scenario("AR2-37. REWRITE_QUERY/ADJUST_TOP_K con decision_basis=EVIDENCE_ACCEPTABLE_DESPITE_GAPS -> rechazado (esa justificación es exclusiva de ACCEPT_EVIDENCE)")
def test_ar2_37_improvement_action_with_accept_basis_rejected():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    raw = json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_ACCEPTABLE_DESPITE_GAPS"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY",), observation_reason_codes=("LOW_RELEVANCE",))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


@scenario("AR2-38. decision_basis debe corresponder a un reason_code REALMENTE presente en la Observation, no solo pertenecer al enum general")
def test_ar2_38_decision_basis_must_match_real_reason_codes():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response, AgenticPlannerResponseError

    # La Observation solo tiene LOW_RELEVANCE, pero el planner justifica con LOW_COVERAGE
    raw = json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_COVERAGE"})
    try:
        parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY",), observation_reason_codes=("LOW_RELEVANCE",))
        raised = False
    except AgenticPlannerResponseError:
        raised = True
    assert raised


@scenario("AR2-39. decision_basis coherente con un reason_code real -> aceptado")
def test_ar2_39_coherent_decision_basis_accepted():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response

    raw = json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = parse_agentic_planner_response(raw, allowed_actions=("REWRITE_QUERY",), observation_reason_codes=("LOW_RELEVANCE",))
    assert result["decision_basis"] == "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"


@scenario("AR2-40. ACCEPT_EVIDENCE con decision_basis correcto -> aceptado")
def test_ar2_40_accept_evidence_correct_basis_accepted():
    from src.tools.verification.agentic_retrieval_controller import parse_agentic_planner_response

    raw = json.dumps({"selected_action": "ACCEPT_EVIDENCE", "decision_basis": "EVIDENCE_ACCEPTABLE_DESPITE_GAPS"})
    result = parse_agentic_planner_response(raw, allowed_actions=("ACCEPT_EVIDENCE",), observation_reason_codes=("LOW_RELEVANCE",))
    assert result["selected_action"] == "ACCEPT_EVIDENCE"


# ---------------------------------------------------------------
# Punto 1 (ronda 3): minimum_viable_evidence estrictamente bool
# ---------------------------------------------------------------

@scenario("AR2-41. minimum_viable_evidence='False' (string) -> rechazado (truthy, podría producir ACCEPT_EVIDENCE indebido)")
def test_ar2_41_minimum_viable_evidence_rejects_string_false():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence="False", query_rewrite_count=0,
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-42. minimum_viable_evidence=1 (int) -> rechazado")
def test_ar2_42_minimum_viable_evidence_rejects_int():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=1, query_rewrite_count=0,
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-43. minimum_viable_evidence=None -> rechazado")
def test_ar2_43_minimum_viable_evidence_rejects_none():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=None, query_rewrite_count=0,
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-44. minimum_viable_evidence=True/False (bool real) -> aceptados")
def test_ar2_44_minimum_viable_evidence_accepts_real_bool():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    for value in (True, False):
        obs = AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=value, query_rewrite_count=0,
        )
        assert obs.minimum_viable_evidence is value


# ---------------------------------------------------------------
# Punto 2 (ronda 3): invariantes compartidos completos
# ---------------------------------------------------------------

@scenario("AR2-45. ADJUST_TOP_K: si la tool cambia current_query (BUG) sin haber pasado por REWRITE_QUERY (query_rewrite_count=0), la propia Observation lo rechaza al construirse (invariante query_rewrite_count==0 implica current_query==claim_text, cerrado en esta ronda) -- la excepción se propaga sin convertirse en AGENTIC_TRANSITION_INVALID")
def test_ar2_45_adjust_top_k_changes_query():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query="a totally different query",  # BUG: ADJUST_TOP_K no debe tocar la query
            retrieval_round=observation.retrieval_round + 1,
            current_top_k=12, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count,
        )

    try:
        bad_execute("ADJUST_TOP_K", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-46. REWRITE_QUERY que cambia claim_id -> AGENTIC_TRANSITION_INVALID")
def test_ar2_46_rewrite_query_changes_claim_id():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id="OTHER_CLAIM",  # BUG: nunca debe cambiar
            claim_text=observation.claim_text, current_query="new different query",
            retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-47. REWRITE_QUERY que cambia claim_text -> AGENTIC_TRANSITION_INVALID")
def test_ar2_47_rewrite_query_changes_claim_text():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text="a rewritten claim text",  # BUG
            current_query="new different query", retrieval_round=observation.retrieval_round + 1,
            current_top_k=observation.current_top_k, effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count + 1,
        )

    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


@scenario("AR2-48. ADJUST_TOP_K que cambia effective_top_k_max -> AGENTIC_TRANSITION_INVALID")
def test_ar2_48_adjust_top_k_changes_effective_max():
    from src.tools.verification.agentic_retrieval_controller import (
        AgenticRetrievalObservation, run_agentic_retrieval_cycle, AGENTIC_TRANSITION_INVALID,
    )

    obs = _make_observation(current_top_k=8, effective_top_k_max=35, retrieval_round=0, remaining_retrieval_budget=2)

    def bad_execute(action, decision_basis, observation):
        return AgenticRetrievalObservation(
            claim_id=observation.claim_id, claim_text=observation.claim_text,
            current_query=observation.current_query, retrieval_round=observation.retrieval_round + 1,
            current_top_k=12, effective_top_k_max=40,  # BUG: tope estructural fijo, no debe cambiar
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=observation.reason_codes,
            minimum_viable_evidence=True, query_rewrite_count=observation.query_rewrite_count,
        )

    planner = _planner_returning({"selected_action": "ADJUST_TOP_K", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=bad_execute)
    assert result.outcome == AGENTIC_TRANSITION_INVALID


# ---------------------------------------------------------------
# Punto 1 (ronda 4): SUFFICIENT implica minimum_viable_evidence=True
# ---------------------------------------------------------------

@scenario("AR2-49. SUFFICIENT + minimum_viable_evidence=False -> ValueError (estado imposible: SUFFICIENT es más estricto que minimum_viable_evidence por diseño)")
def test_ar2_49_sufficient_with_minimum_viable_false_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=2, evidence_ids=("e1", "e2"), max_relevance_score=0.8,
            grade_result="SUFFICIENT", reason_codes=(), minimum_viable_evidence=False, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-50. SUFFICIENT + minimum_viable_evidence=True -> válido")
def test_ar2_50_sufficient_with_minimum_viable_true_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=2, evidence_ids=("e1", "e2"), max_relevance_score=0.8,
        grade_result="SUFFICIENT", reason_codes=(), minimum_viable_evidence=True, query_rewrite_count=0,
    )
    assert obs.grade_result == "SUFFICIENT"
    assert obs.minimum_viable_evidence is True


# ---------------------------------------------------------------
# Punto 2 (ronda 4): candidate_count == len(evidence_ids)
# ---------------------------------------------------------------

@scenario("AR2-51. candidate_count=0 + evidence_ids no vacío -> rechazado")
def test_ar2_51_zero_candidate_count_with_nonempty_evidence_ids_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=0, evidence_ids=("e1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=("LOW_CANDIDATE_COUNT",),
            minimum_viable_evidence=False, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-52. candidate_count>0 + evidence_ids vacío -> rechazado")
def test_ar2_52_nonzero_candidate_count_with_empty_evidence_ids_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
            current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=2, evidence_ids=(), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-53. candidate_count == len(evidence_ids) -> válido")
def test_ar2_53_matching_candidate_count_and_evidence_ids_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=3, evidence_ids=("e1", "e2", "e3"), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
        minimum_viable_evidence=True, query_rewrite_count=0,
    )
    assert obs.candidate_count == len(obs.evidence_ids) == 3


# ---------------------------------------------------------------
# Punto 1 (ronda 5): minimum_viable_evidence=True requiere evidencia real
# ---------------------------------------------------------------

def _base_kwargs(**overrides):
    base = dict(
        claim_id="c1", claim_text="x", current_query="x", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        max_relevance_score=0.5, query_rewrite_count=0,
    )
    base.update(overrides)
    return base


@scenario("AR2-54. minimum_viable_evidence=True + candidate_count=0 + evidence_ids=() -> rechazado")
def test_ar2_54_minimum_viable_true_with_zero_candidates_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=0, evidence_ids=(), grade_result="INSUFFICIENT",
                reason_codes=("LOW_CANDIDATE_COUNT",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-55. grade_result=SUFFICIENT + candidate_count=0 -> rechazado")
def test_ar2_55_sufficient_with_zero_candidates_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=0, evidence_ids=(), grade_result="SUFFICIENT",
                reason_codes=(), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-56. minimum_viable_evidence=True + candidate_count>=1 -> válido")
def test_ar2_56_minimum_viable_true_with_candidates_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        )
    )
    assert obs.candidate_count >= 1


# ---------------------------------------------------------------
# Punto 2 (ronda 5): contrato de evidence_ids endurecido
# ---------------------------------------------------------------

@scenario("AR2-57. evidence_ids con un ID vacío ('',) -> rechazado")
def test_ar2_57_evidence_ids_empty_string_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=("",), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-58. evidence_ids con IDs duplicados ('e1','e1') -> rechazado")
def test_ar2_58_evidence_ids_duplicates_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=2, evidence_ids=("e1", "e1"), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-59. evidence_ids con elemento no-str (123,) -> rechazado")
def test_ar2_59_evidence_ids_non_string_element_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=(123,), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-60. evidence_ids como list en vez de tuple -> rechazado")
def test_ar2_60_evidence_ids_list_instead_of_tuple_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=["e1"], grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-61. evidence_ids válido (tuple, str únicos no vacíos) -> aceptado")
def test_ar2_61_evidence_ids_valid_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            candidate_count=2, evidence_ids=("e1", "e2"), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        )
    )
    assert obs.evidence_ids == ("e1", "e2")


# ---------------------------------------------------------------
# Punto 3 (ronda 5): tipado estricto de identidad/query
# ---------------------------------------------------------------

@scenario("AR2-62. claim_id/claim_text/current_query rechazan bool/int/float/None/lista")
def test_ar2_62_identity_fields_reject_non_str_types():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    for field_name in ("claim_id", "claim_text", "current_query"):
        for bad in (True, 123, 1.5, None, ["x"]):
            kwargs = dict(claim_id="c1", claim_text="x", current_query="x")
            kwargs[field_name] = bad
            try:
                AgenticRetrievalObservation(
                    **kwargs,
                    **_base_kwargs(
                        candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                        reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
                    ),
                )
                raised = False
            except TypeError:
                raised = True
            assert raised, f"{field_name}={bad!r} debió rechazarse"


@scenario("AR2-63. claim_id/claim_text/current_query como str reales no vacíos -> aceptados")
def test_ar2_63_identity_fields_accept_real_str():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            claim_id="c1", claim_text="a real claim", current_query="a real claim",
            candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        ),
    )
    assert obs.claim_id == "c1"
    assert obs.current_query == "a real claim"


# ---------------------------------------------------------------
# Punto 1 (ronda 6): query_rewrite_count <= retrieval_round
# ---------------------------------------------------------------

@scenario("AR2-64. retrieval_round=0, query_rewrite_count=1 -> rechazado")
def test_ar2_64_round_zero_rewrites_one_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                retrieval_round=0, query_rewrite_count=1,
                candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-65. retrieval_round=1, query_rewrite_count=2 -> rechazado")
def test_ar2_65_round_one_rewrites_two_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                retrieval_round=1, query_rewrite_count=2,
                candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-66. retrieval_round=2, query_rewrite_count=2 -> válido")
def test_ar2_66_round_two_rewrites_two_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            retrieval_round=2, query_rewrite_count=2,
            candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        )
    )
    assert obs.retrieval_round == 2 and obs.query_rewrite_count == 2


@scenario("AR2-67. retrieval_round=2, query_rewrite_count=1 -> válido")
def test_ar2_67_round_two_rewrites_one_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            retrieval_round=2, query_rewrite_count=1,
            candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        )
    )
    assert obs.retrieval_round == 2 and obs.query_rewrite_count == 1


# ---------------------------------------------------------------
# Punto 2 (ronda 6): candidate_count==0 implica max_relevance_score==0.0
# ---------------------------------------------------------------

@scenario("AR2-68. 0 candidates + score 0.5 -> rechazado")
def test_ar2_68_zero_candidates_positive_score_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=0, evidence_ids=(), max_relevance_score=0.5,
                grade_result="INSUFFICIENT", reason_codes=("LOW_CANDIDATE_COUNT",),
                minimum_viable_evidence=False,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-69. 0 candidates + score 0.0 -> válido")
def test_ar2_69_zero_candidates_zero_score_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            candidate_count=0, evidence_ids=(), max_relevance_score=0.0,
            grade_result="INSUFFICIENT", reason_codes=("LOW_CANDIDATE_COUNT",),
            minimum_viable_evidence=False,
        )
    )
    assert obs.candidate_count == 0 and obs.max_relevance_score == 0.0


@scenario("AR2-70. candidates > 0 + score 0.0 -> permitido (no se impone la inversa)")
def test_ar2_70_nonzero_candidates_zero_score_allowed():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            candidate_count=2, evidence_ids=("e1", "e2"), max_relevance_score=0.0,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=True,
        )
    )
    assert obs.candidate_count == 2 and obs.max_relevance_score == 0.0


# ---------------------------------------------------------------
# Punto 3 (ronda 6): reason_codes como contrato estructural
# ---------------------------------------------------------------

@scenario("AR2-71. reason_codes como list en vez de tuple -> rechazado")
def test_ar2_71_reason_codes_list_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                reason_codes=["LOW_RELEVANCE"], minimum_viable_evidence=True,
            )
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-72. reason_codes con elemento no-str (123,) -> rechazado")
def test_ar2_72_reason_codes_non_string_element_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                reason_codes=(123,), minimum_viable_evidence=True,
            )
        )
        raised = False
    except TypeError:
        raised = True
    assert raised


@scenario("AR2-73. reason_codes con duplicados -> rechazado")
def test_ar2_73_reason_codes_duplicates_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
                reason_codes=("LOW_RELEVANCE", "LOW_RELEVANCE"), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-74. reason_codes válido (tuple de str únicos de GRADE_REASON_CODES) -> aceptado")
def test_ar2_74_reason_codes_valid_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            candidate_count=1, evidence_ids=("e1",), grade_result="INSUFFICIENT",
            reason_codes=("LOW_RELEVANCE", "LOW_COVERAGE"), minimum_viable_evidence=True,
        )
    )
    assert obs.reason_codes == ("LOW_RELEVANCE", "LOW_COVERAGE")


# ---------------------------------------------------------------
# Punto 1 (ronda 7): candidate_count <= current_top_k
# ---------------------------------------------------------------

@scenario("AR2-75. current_top_k=8, candidate_count=9 -> rechazado")
def test_ar2_75_candidate_count_exceeds_top_k_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            **_base_kwargs(
                current_top_k=8, candidate_count=9, evidence_ids=tuple(f"e{i}" for i in range(9)),
                grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            )
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-76. current_top_k=8, candidate_count=8 -> válido")
def test_ar2_76_candidate_count_equals_top_k_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            current_top_k=8, candidate_count=8, evidence_ids=tuple(f"e{i}" for i in range(8)),
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
        )
    )
    assert obs.candidate_count == 8


@scenario("AR2-77. current_top_k=8, candidate_count=0 -> válido")
def test_ar2_77_candidate_count_zero_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        **_base_kwargs(
            current_top_k=8, candidate_count=0, evidence_ids=(), max_relevance_score=0.0,
            grade_result="INSUFFICIENT", reason_codes=("LOW_CANDIDATE_COUNT",), minimum_viable_evidence=False,
        )
    )
    assert obs.candidate_count == 0


# ---------------------------------------------------------------
# Punto 2 (ronda 7): query_rewrite_count==0 implica current_query==claim_text
# ---------------------------------------------------------------

@scenario("AR2-78. query_rewrite_count=0 + current_query==claim_text -> válido")
def test_ar2_78_no_rewrite_matching_query_accepted():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        claim_id="c1", claim_text="original claim text", current_query="original claim text",
        retrieval_round=0, current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
        minimum_viable_evidence=True, query_rewrite_count=0,
    )
    assert obs.current_query == obs.claim_text


@scenario("AR2-79. query_rewrite_count=0 + current_query!=claim_text -> rechazado")
def test_ar2_79_no_rewrite_but_different_query_rejected():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    try:
        AgenticRetrievalObservation(
            claim_id="c1", claim_text="original claim text", current_query="a different query",
            retrieval_round=0, current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
            candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
            minimum_viable_evidence=True, query_rewrite_count=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


@scenario("AR2-80. query_rewrite_count>0 + current_query==claim_text -> permitido (no se impone la inversa)")
def test_ar2_80_rewrite_count_positive_but_same_query_allowed():
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    obs = AgenticRetrievalObservation(
        claim_id="c1", claim_text="original claim text", current_query="original claim text",
        retrieval_round=1, current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=1,
        candidate_count=1, evidence_ids=("e1",), max_relevance_score=0.5,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",),
        minimum_viable_evidence=True, query_rewrite_count=1,
    )
    assert obs.query_rewrite_count == 1 and obs.current_query == obs.claim_text


@scenario("BLOQUE4-CONTRACT-01. decision_basis llega EXACTAMENTE igual al executor -- contrato ampliado execute_action_fn(selected_action, decision_basis, observation)")
def test_bloque4_contract_01_decision_basis_reaches_executor_intact():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle

    received = {}

    def execute(action, decision_basis, observation):
        received["action"] = action
        received["decision_basis"] = decision_basis
        return _make_observation(retrieval_round=1, remaining_retrieval_budget=observation.remaining_retrieval_budget - 1, current_query="different query now", query_rewrite_count=1)

    obs = _make_observation(retrieval_round=0, current_top_k=8, effective_top_k_max=35)
    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=execute)

    assert received["action"] == "REWRITE_QUERY"
    assert received["decision_basis"] == "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"


@scenario("BLOQUE4-CONTRACT-02. Con múltiples reason_codes, el executor recibe EXACTAMENTE el decision_basis que el planner eligió, nunca el primer reason_code por defecto")
def test_bloque4_contract_02_multiple_reason_codes_preserves_exact_planner_choice():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle

    received = {}

    def execute(action, decision_basis, observation):
        received["decision_basis"] = decision_basis
        return _make_observation(
            retrieval_round=1, remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            current_query="different query now", query_rewrite_count=1,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_RELEVANCE"),
        )

    obs = _make_observation(
        retrieval_round=0, current_top_k=8, effective_top_k_max=35,
        reason_codes=("LOW_CANDIDATE_COUNT", "LOW_RELEVANCE"),
    )
    # El planner elige explícitamente LOW_RELEVANCE (segundo elemento),
    # NO el primero (LOW_CANDIDATE_COUNT) -- confirma que el executor
    # nunca infiere/selecciona por su cuenta.
    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    run_agentic_retrieval_cycle(initial_observation=obs, invoke_planner_fn=planner, execute_action_fn=execute)

    assert received["decision_basis"] == "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"
    assert received["decision_basis"] != "EVIDENCE_INSUFFICIENT_LOW_CANDIDATE_COUNT"


if __name__ == "__main__":
    for fn in (
        test_ar2_01_sufficient_forces_accept_without_planner,
        test_ar2_02_first_insufficiency_excludes_accept,
        test_ar2_03_second_insufficiency_enables_accept_when_viable,
        test_ar2_04_second_insufficiency_no_accept_if_not_viable,
        test_ar2_05_budget_exhausted_viable_forces_accept,
        test_ar2_06_budget_exhausted_not_viable_forces_unresolved,
        test_ar2_07_adjust_top_k_unavailable_at_max,
        test_ar2_08_actions_enum_exact,
        test_ar2_09_action_outside_allowed_rejected,
        test_ar2_10_extra_field_rejected,
        test_ar2_11_invalid_decision_basis_rejected,
        test_ar2_12_full_cycle_rewrite_then_sufficient,
        test_ar2_13_full_cycle_planner_accepts_explicitly,
        test_ar2_14_malformed_planner_fails_closed,
        test_ar2_15_prompt_only_lists_allowed_actions,
        test_ar2_16_observation_rejects_top_k_exceeding_max,
        test_ar2_17_single_action_never_invokes_planner,
        test_ar2_18_deterministic_decision_basis_matches_reason_code,
        test_ar2_19_multiple_actions_invokes_planner,
        test_ar2_20_rewrite_query_wrong_round_increment,
        test_ar2_21_rewrite_query_budget_not_decremented,
        test_ar2_22_rewrite_query_unchanged_query,
        test_ar2_23_rewrite_query_count_not_incremented,
        test_ar2_24_rewrite_query_top_k_changed,
        test_ar2_25_adjust_top_k_not_increased,
        test_ar2_26_adjust_top_k_exceeds_max,
        test_ar2_27_adjust_top_k_rewrite_count_changed,
        test_ar2_28_rejects_bool_for_int_fields,
        test_ar2_29_rejects_negative_fields,
        test_ar2_30_rejects_non_positive_top_k_fields,
        test_ar2_31_rejects_relevance_score_out_of_range,
        test_ar2_32_rejects_empty_string_fields,
        test_ar2_33_rejects_reason_codes_outside_grade_reason_codes,
        test_ar2_34_sufficient_with_reason_codes_rejected,
        test_ar2_35_insufficient_without_reason_codes_rejected,
        test_ar2_36_accept_evidence_wrong_decision_basis_rejected,
        test_ar2_37_improvement_action_with_accept_basis_rejected,
        test_ar2_38_decision_basis_must_match_real_reason_codes,
        test_ar2_39_coherent_decision_basis_accepted,
        test_ar2_40_accept_evidence_correct_basis_accepted,
        test_ar2_41_minimum_viable_evidence_rejects_string_false,
        test_ar2_42_minimum_viable_evidence_rejects_int,
        test_ar2_43_minimum_viable_evidence_rejects_none,
        test_ar2_44_minimum_viable_evidence_accepts_real_bool,
        test_ar2_45_adjust_top_k_changes_query,
        test_ar2_46_rewrite_query_changes_claim_id,
        test_ar2_47_rewrite_query_changes_claim_text,
        test_ar2_48_adjust_top_k_changes_effective_max,
        test_ar2_49_sufficient_with_minimum_viable_false_rejected,
        test_ar2_50_sufficient_with_minimum_viable_true_accepted,
        test_ar2_51_zero_candidate_count_with_nonempty_evidence_ids_rejected,
        test_ar2_52_nonzero_candidate_count_with_empty_evidence_ids_rejected,
        test_ar2_53_matching_candidate_count_and_evidence_ids_accepted,
        test_ar2_54_minimum_viable_true_with_zero_candidates_rejected,
        test_ar2_55_sufficient_with_zero_candidates_rejected,
        test_ar2_56_minimum_viable_true_with_candidates_accepted,
        test_ar2_57_evidence_ids_empty_string_rejected,
        test_ar2_58_evidence_ids_duplicates_rejected,
        test_ar2_59_evidence_ids_non_string_element_rejected,
        test_ar2_60_evidence_ids_list_instead_of_tuple_rejected,
        test_ar2_61_evidence_ids_valid_accepted,
        test_ar2_62_identity_fields_reject_non_str_types,
        test_ar2_63_identity_fields_accept_real_str,
        test_ar2_64_round_zero_rewrites_one_rejected,
        test_ar2_65_round_one_rewrites_two_rejected,
        test_ar2_66_round_two_rewrites_two_accepted,
        test_ar2_67_round_two_rewrites_one_accepted,
        test_ar2_68_zero_candidates_positive_score_rejected,
        test_ar2_69_zero_candidates_zero_score_accepted,
        test_ar2_70_nonzero_candidates_zero_score_allowed,
        test_ar2_71_reason_codes_list_rejected,
        test_ar2_72_reason_codes_non_string_element_rejected,
        test_ar2_73_reason_codes_duplicates_rejected,
        test_ar2_74_reason_codes_valid_accepted,
        test_ar2_75_candidate_count_exceeds_top_k_rejected,
        test_ar2_76_candidate_count_equals_top_k_accepted,
        test_ar2_77_candidate_count_zero_accepted,
        test_ar2_78_no_rewrite_matching_query_accepted,
        test_ar2_79_no_rewrite_but_different_query_rejected,
        test_ar2_80_rewrite_count_positive_but_same_query_allowed,
        test_bloque4_contract_01_decision_basis_reaches_executor_intact,
        test_bloque4_contract_02_multiple_reason_codes_preserves_exact_planner_choice,
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
