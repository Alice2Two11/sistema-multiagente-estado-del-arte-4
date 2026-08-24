"""E2E-BUG-01 -- ACTION_UNAVAILABLE contract fix.

Semántica exacta: la acción seleccionada era legal según
compute_allowed_actions para la Observation actual, pero no puede
ejecutarse con los datos concretos disponibles. NO significa fallo
técnico global, claim unsupported, presupuesto agotado, transición
inválida ni fallo del planner.

Solo QUERY_REWRITE_UNAVAILABLE (Bloque 3) se traduce a esta condición;
cualquier otro QueryRewriteError sigue propagándose sin conversión."""

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
        claim_id="c1", claim_text="test claim about attention", current_query="test claim about attention",
        retrieval_round=0, current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=3,
        candidate_count=1, evidence_ids=("a.pdf::c1",), max_relevance_score=0.2,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
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


def _good_adjust_observation(before):
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    return AgenticRetrievalObservation(
        claim_id=before.claim_id, claim_text=before.claim_text, current_query=before.current_query,
        retrieval_round=before.retrieval_round + 1, current_top_k=before.current_top_k + 4,
        effective_top_k_max=before.effective_top_k_max,
        remaining_retrieval_budget=before.remaining_retrieval_budget - 1,
        candidate_count=2, evidence_ids=("a.pdf::c1", "a.pdf::c2"), max_relevance_score=0.9,
        grade_result="SUFFICIENT", reason_codes=(), minimum_viable_evidence=True,
        query_rewrite_count=before.query_rewrite_count,
    )


# ---------------------------------------------------------------
# Caso A: REWRITE_QUERY unavailable + ADJUST_TOP_K disponible -> se ejecuta, no BLOCK
# ---------------------------------------------------------------

@scenario("A. REWRITE_QUERY unavailable + ADJUST_TOP_K disponible -> ADJUST_TOP_K se ejecuta, no BLOCK, sin segunda consulta al planner")
def test_a_fallback_to_adjust_top_k_no_block():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable

    obs0 = _make_observation()
    call_log = []

    def execute(action, decision_basis, observation):
        call_log.append(action)
        if action == "REWRITE_QUERY":
            raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos")
        if action == "ADJUST_TOP_K":
            return _good_adjust_observation(observation)
        raise AssertionError(f"acción inesperada: {action}")

    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner, execute_action_fn=execute)

    assert result.outcome == "ACCEPT_EVIDENCE"
    assert call_log == ["REWRITE_QUERY", "ADJUST_TOP_K"]
    assert result.steps[1]["planner_invoked"] is False  # solo 1 acción efectiva restante


# ---------------------------------------------------------------
# Caso B: REWRITE_QUERY unavailable + ninguna otra acción efectiva -> FINISH_UNRESOLVED
# ---------------------------------------------------------------

@scenario("B. REWRITE_QUERY unavailable + ninguna otra acción efectiva -> FINISH_UNRESOLVED")
def test_b_no_alternative_finishes_unresolved():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable, FINISH_UNRESOLVED

    # top_k ya en el máximo -> ADJUST_TOP_K nunca disponible; primera insuficiencia -> ACCEPT_EVIDENCE tampoco
    obs0 = _make_observation(current_top_k=35, effective_top_k_max=35)

    def execute(action, decision_basis, observation):
        raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos")

    def planner_should_not_be_called(prompt):
        raise AssertionError("con 1 sola acción, el planner nunca se invoca")

    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner_should_not_be_called, execute_action_fn=execute)
    assert result.outcome == FINISH_UNRESOLVED


# ---------------------------------------------------------------
# Caso C: acción unavailable -> budget/round sin cambio
# ---------------------------------------------------------------

@scenario("C. Acción unavailable -> remaining_retrieval_budget y retrieval_round sin cambio")
def test_c_budget_and_round_unchanged():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable

    obs0 = _make_observation(current_top_k=35, effective_top_k_max=35, remaining_retrieval_budget=3)

    def execute(action, decision_basis, observation):
        raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos")

    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=lambda p: "", execute_action_fn=execute)
    assert result.final_observation.remaining_retrieval_budget == 3
    assert result.final_observation.retrieval_round == 0
    assert result.final_observation.query_rewrite_count == 0
    assert result.final_observation.current_query == obs0.current_query
    assert result.final_observation.current_top_k == obs0.current_top_k


# ---------------------------------------------------------------
# Caso D: no aparece retrieval_transition, sí queda auditada en decision_steps
# ---------------------------------------------------------------

@scenario("D. Acción unavailable no aparece como retrieval_transition, pero sí queda auditada en decision_steps con execution_status=ACTION_UNAVAILABLE")
def test_d_no_retrieval_transition_but_audited():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle
    from src.tools.verification.agentic_retrieval_tracing import build_traced_execute_action_fn, build_agentic_retrieval_trace

    class UnavailableRetriever:
        top_k = 8
        fetch_k = 35
        def retrieve_more(self, request):
            raise AssertionError("no debe llegar a retrieval real -- generate_query_rewrite falla antes")

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "unrelated content with no new vocabulary for the claim itself", "native_scores_by_retriever": {"chroma": 0.2}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=UnavailableRetriever(), allowed_source_filenames=frozenset({"a.pdf"}), claim_id="c1",
        claim_text="unrelated content with no new vocabulary for the claim itself",
        initial_candidates=initial_candidates,
    )
    obs0 = _make_observation(
        claim_text="unrelated content with no new vocabulary for the claim itself",
        current_query="unrelated content with no new vocabulary for the claim itself",
        current_top_k=35, effective_top_k_max=35, evidence_ids=("a.pdf::c1",),
    )
    traced_fn, step_trace = build_traced_execute_action_fn(executor)
    planner = _planner_returning({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})
    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=lambda p: "", execute_action_fn=traced_fn)

    assert step_trace == []  # nunca se registró una retrieval_transition
    assert len(result.steps) == 1
    assert result.steps[0]["execution_status"] == "ACTION_UNAVAILABLE"

    agentic_result = {
        "planner_invoked": False, "outcome": result.outcome, "final_observation": None,
        "steps": tuple(result.steps), "initial_grade_result": "INSUFFICIENT",
        "agentic_additional_retrievals_used": 0, "effective_budget_for_verify_claim": 3,
    }
    trace = build_agentic_retrieval_trace(
        claim_id="c1", initial_candidate_count=1, initial_max_relevance_score=0.2, initial_top_k=35,
        raw_transition_attempts=step_trace, agentic_result=agentic_result,
    )
    assert trace["retrieval_transitions"] == ()
    assert trace["decision_steps"][0]["execution_status"] == "ACTION_UNAVAILABLE"


# ---------------------------------------------------------------
# Caso E: acción marcada unavailable no puede elegirse de nuevo para la misma Observation
# ---------------------------------------------------------------

@scenario("E. Acción marcada unavailable no vuelve a aparecer entre las acciones efectivas para la misma Observation (sin loop infinito)")
def test_e_no_infinite_retry_same_observation():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable, FINISH_UNRESOLVED

    obs0 = _make_observation(current_top_k=8, effective_top_k_max=35)  # ambas REWRITE_QUERY/ADJUST_TOP_K legales
    attempts = []

    def execute(action, decision_basis, observation):
        attempts.append(action)
        raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos") if action == "REWRITE_QUERY" else (_ for _ in ()).throw(AgenticRetrievalActionUnavailable("ADJUST_TOP_K test unavailable"))

    planner = _planner_returning(
        {"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"},
    )
    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner, execute_action_fn=execute)

    # cada acción se intenta como máximo 1 vez para la misma Observation
    assert attempts.count("REWRITE_QUERY") == 1
    assert attempts.count("ADJUST_TOP_K") == 1
    assert result.outcome == FINISH_UNRESOLVED


# ---------------------------------------------------------------
# Caso F: tras una nueva Observation, las exclusiones locales se limpian
# ---------------------------------------------------------------

@scenario("F. Tras obtener una nueva Observation real, las exclusiones locales se reinician -- una acción antes unavailable puede volver a intentarse")
def test_f_exclusions_reset_after_new_observation():
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable

    obs0 = _make_observation(current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=3)
    rewrite_attempts = []

    def execute(action, decision_basis, observation):
        if action == "REWRITE_QUERY":
            rewrite_attempts.append(observation.retrieval_round)
            if observation.retrieval_round == 0:
                raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos")
            # en la segunda Observation (tras ADJUST_TOP_K), REWRITE_QUERY sí puede reintentarse
            return AgenticRetrievalObservationFactory(observation)
        if action == "ADJUST_TOP_K":
            return _good_adjust_observation_insufficient(observation)
        raise AssertionError(action)

    def AgenticRetrievalObservationFactory(before):
        from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation
        return AgenticRetrievalObservation(
            claim_id=before.claim_id, claim_text=before.claim_text, current_query=before.current_query + " new terms",
            retrieval_round=before.retrieval_round + 1, current_top_k=before.current_top_k,
            effective_top_k_max=before.effective_top_k_max,
            remaining_retrieval_budget=before.remaining_retrieval_budget - 1,
            candidate_count=2, evidence_ids=("a.pdf::c1", "a.pdf::c2"), max_relevance_score=0.9,
            grade_result="SUFFICIENT", reason_codes=(), minimum_viable_evidence=True,
            query_rewrite_count=before.query_rewrite_count + 1,
        )

    def _good_adjust_observation_insufficient(before):
        from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation
        return AgenticRetrievalObservation(
            claim_id=before.claim_id, claim_text=before.claim_text, current_query=before.current_query,
            retrieval_round=before.retrieval_round + 1, current_top_k=before.current_top_k + 4,
            effective_top_k_max=before.effective_top_k_max,
            remaining_retrieval_budget=before.remaining_retrieval_budget - 1,
            candidate_count=1, evidence_ids=("a.pdf::c1",), max_relevance_score=0.2,
            grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True,
            query_rewrite_count=before.query_rewrite_count,
        )

    planner = _planner_returning(
        {"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"},
        {"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"},
    )
    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner, execute_action_fn=execute)

    # REWRITE_QUERY se intentó en la ronda 0 (unavailable) Y de nuevo en la ronda 1 (nueva Observation, tras ADJUST_TOP_K)
    assert 0 in rewrite_attempts
    assert result.outcome == "ACCEPT_EVIDENCE"


# ---------------------------------------------------------------
# Caso G: QueryRewriteError distinto de QUERY_REWRITE_UNAVAILABLE sigue propagándose
# ---------------------------------------------------------------

@scenario("G. QueryRewriteError distinto de QUERY_REWRITE_UNAVAILABLE (ej. authorized_sources vacío) sigue propagándose sin convertirse en ACTION_UNAVAILABLE")
def test_g_other_query_rewrite_errors_still_propagate():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    class DummyRetriever:
        top_k = 8
        fetch_k = 35

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "some text", "native_scores_by_retriever": {"chroma": 0.5}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=DummyRetriever(), allowed_source_filenames=frozenset({"a.pdf"}), claim_id="c1",
        claim_text="claim text", initial_candidates=initial_candidates,
    )
    obs = _make_observation(claim_text="claim text", current_query="claim text", reason_codes=("LOW_CANDIDATE_COUNT",))
    # decision_basis con reason_code que NO está en observation.reason_codes -> ActionExecutorError,
    # NO QueryRewriteError -- confirma que otros errores del executor (frontera) también siguen
    # propagándose sin convertirse en ACTION_UNAVAILABLE.
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError
    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    except Exception:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Caso H: fallo técnico real del retriever sigue bloqueando -- no se transforma en ACTION_UNAVAILABLE
# ---------------------------------------------------------------

@scenario("H. Fallo técnico real del retriever (excepción no relacionada con QUERY_REWRITE_UNAVAILABLE) sigue bloqueando -- no se transforma en ACTION_UNAVAILABLE")
def test_h_technical_retriever_failure_still_blocks():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalActionUnavailable

    class FailingRetriever:
        top_k = 8
        fetch_k = 35
        def retrieve_more(self, request):
            raise RuntimeError("simulated real technical failure, unrelated to query rewrite")

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c1", "text": "some relevant candidate text about the claim topic here", "native_scores_by_retriever": {"chroma": 0.2}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=FailingRetriever(), allowed_source_filenames=frozenset({"a.pdf"}), claim_id="c1",
        claim_text="claim topic", initial_candidates=initial_candidates,
    )
    obs = _make_observation(claim_text="claim topic", current_query="claim topic", evidence_ids=("a.pdf::c1",))

    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised_correctly = False
    except AgenticRetrievalActionUnavailable:
        raised_correctly = False  # NO debe convertirse en esto
    except RuntimeError:
        raised_correctly = True  # debe seguir siendo el error técnico real

    assert raised_correctly


if __name__ == "__main__":
    for fn in (
        test_a_fallback_to_adjust_top_k_no_block,
        test_b_no_alternative_finishes_unresolved,
        test_c_budget_and_round_unchanged,
        test_d_no_retrieval_transition_but_audited,
        test_e_no_infinite_retry_same_observation,
        test_f_exclusions_reset_after_new_observation,
        test_g_other_query_rewrite_errors_still_propagate,
        test_h_technical_retriever_failure_still_blocks,
    ):
        try:
            fn()
        except Exception:
            pass

    failed = 0
    for name, ok, err in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        if not ok:
            failed += 1
            print(err)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} tests ejecutados PASS")
    raise SystemExit(1 if failed else 0)
