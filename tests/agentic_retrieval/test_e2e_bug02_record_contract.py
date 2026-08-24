"""E2E-BUG-02 -- Agentic Retrieval record contract mismatch.

_independent_retrieve_claim (Bloque 5) es el ÚNICO productor real de
independent_rag_claim_records, y SIEMPRE incluye "agentic_retrieval" --
confirmado por auditoría: solo se invoca cuando dependencies.
retrieval_tool/retriever_binding existen (si no, no se agrega ningún
record); ningún test/fixture del repo construye un record sin ella.
Por tanto "agentic_retrieval" es OBLIGATORIO en el schema validado por
validate_agent07_runtime_result_contract, no opcional."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "agentic_retrieval"))

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


def _real_agentic_retrieval_record(**overrides):
    """Genera un record["agentic_retrieval"] real, atravesando
    _independent_retrieve_claim -- no se construye a mano un dict
    sintético para el caso feliz."""
    from test_bloque5_stage07_wiring import _FakeCollection, _make_retriever, _FakeVerificationLLM, _make_dependencies, _make_context
    from src.adapters.verification_runtime import _independent_retrieve_claim

    insufficient_first = overrides.pop("insufficient_first", False)
    coll = _FakeCollection(insufficient_first=insufficient_first)
    retriever = _make_retriever(coll, top_k=overrides.pop("top_k", 8))
    llm = _FakeVerificationLLM(
        action=overrides.pop("action", "REWRITE_QUERY"),
        basis=overrides.pop("basis", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"),
    )
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()
    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})
    return record


def _wrap_records_in_runtime_result(records):
    """Reutiliza _blocked_runtime_result (código productivo real, el
    mismo que construye el bloqueo operativo en producción) para
    validar la lógica de independent_rag_claim_records sin necesitar
    construir un bundle/resolution científicos completos (fuera del
    alcance de este bug) -- el camino BLOCKED valida
    execution_metrics/independent_rag_claim_records igual que
    COMPLETED, sin exigir bundle/resolution reales."""
    from src.adapters.verification_runtime import _blocked_runtime_result

    return _blocked_runtime_result(
        stage="INDEPENDENT_RAG", claim_id=None, section_id=None,
        error_code="TEST_SIMULATED_BLOCK", classification="TECHNICAL",
        schema_versions={}, metrics={
            "independent_rag_claim_records": tuple(records),
            "independent_rag_claims": len(records),
            "independent_rag_claims_with_results": len(records),
            "independent_rag_claims_without_results": 0,
            "claims_processed": len(records),
        },
    )


# ---------------------------------------------------------------
# A. independent_rag_record con agentic_retrieval válido -> ACCEPT
# ---------------------------------------------------------------

@scenario("A. independent_rag_record con agentic_retrieval válido (SUFFICIENT inicial, real) -> ACCEPT")
def test_a_valid_agentic_retrieval_record_accepted():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    result = _wrap_records_in_runtime_result([record])
    assert result.result_contract_valid is True


@scenario("A2. independent_rag_record con agentic_retrieval válido (ciclo REWRITE_QUERY real) -> ACCEPT")
def test_a2_valid_agentic_retrieval_record_with_cycle_accepted():
    record = _real_agentic_retrieval_record(insufficient_first=True, top_k=3, action="REWRITE_QUERY")
    result = _wrap_records_in_runtime_result([record])
    assert result.result_contract_valid is True


# ---------------------------------------------------------------
# B. falta agentic_retrieval en contrato productivo -> REJECT
# (decisión confirmada por auditoría: es obligatorio, no opcional --
# ningún camino productivo/test legítimo lo omite)
# ---------------------------------------------------------------

@scenario("B. falta agentic_retrieval en el record -> REJECT (es obligatorio, ningún camino productivo real lo omite)")
def test_b_missing_agentic_retrieval_rejected():
    from src.adapters.verification_runtime import validate_agent07_runtime_result_contract

    record = _real_agentic_retrieval_record(insufficient_first=False)
    del record["agentic_retrieval"]
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID" in str(e)
    assert raised


# ---------------------------------------------------------------
# C. campo extra arbitrario -> REJECT
# ---------------------------------------------------------------

@scenario("C. campo extra arbitrario en el record -> REJECT (no se acepta apertura de campos arbitrarios)")
def test_c_extra_field_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["unexpected_extra_field"] = "should not be allowed"
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID" in str(e)
    assert raised


@scenario("C2. campo extra arbitrario DENTRO de agentic_retrieval -> REJECT")
def test_c2_extra_field_inside_agentic_retrieval_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    record["agentic_retrieval"]["unexpected_extra_field"] = "nope"
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID" in str(e)
    assert raised


# ---------------------------------------------------------------
# D. agentic_retrieval con schema inválido -> REJECT
# ---------------------------------------------------------------

@scenario("D. agentic_retrieval con schema inválido (planner_invoked no-bool) -> REJECT")
def test_d_invalid_schema_planner_invoked_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    record["agentic_retrieval"]["planner_invoked"] = "yes"  # no-bool
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID" in str(e)
    assert raised


@scenario("D2. agentic_retrieval.initial_grade_result fuera de {SUFFICIENT, INSUFFICIENT} -> REJECT")
def test_d2_invalid_initial_grade_result_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    record["agentic_retrieval"]["initial_grade_result"] = "MAYBE"
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID" in str(e)
    assert raised


@scenario("D3. agentic_retrieval.steps no es tuple/list -> REJECT")
def test_d3_invalid_steps_type_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    record["agentic_retrieval"]["steps"] = "not-a-sequence"
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID" in str(e)
    assert raised


# ---------------------------------------------------------------
# E. shared budget incoherente -> REJECT
# ---------------------------------------------------------------

@scenario("E. effective_budget_for_verify_claim incoherente con original - used -> REJECT")
def test_e_incoherent_budget_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=True, top_k=3, action="REWRITE_QUERY")
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    # forzar un valor incoherente, distinto de original - used
    record["agentic_retrieval"]["effective_budget_for_verify_claim"] = 999
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_BUDGET_MISMATCH" in str(e)
    assert raised


@scenario("E2. agentic_additional_retrievals_used > original_max_additional_retrieval_requests -> REJECT (violación real del presupuesto)")
def test_e2_used_exceeds_original_rejected():
    record = _real_agentic_retrieval_record(insufficient_first=False)
    record["agentic_retrieval"] = dict(record["agentic_retrieval"])
    record["agentic_retrieval"]["original_max_additional_retrieval_requests"] = 1
    record["agentic_retrieval"]["agentic_additional_retrievals_used"] = 5
    try:
        _wrap_records_in_runtime_result([record])
        raised = False
    except ValueError as e:
        raised = "AGENT07_RUNTIME_AGENTIC_RETRIEVAL_BUDGET_INVALID" in str(e)
    assert raised


# ---------------------------------------------------------------
# F. trace/steps válidos del caso ACTION_UNAVAILABLE -> ACCEPT
# ---------------------------------------------------------------

@scenario("F. trace/steps válidos con execution_status=ACTION_UNAVAILABLE (E2E-BUG-01 real) -> ACCEPT")
def test_f_action_unavailable_case_accepted():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation, run_agentic_retrieval_cycle, AgenticRetrievalActionUnavailable
    from src.tools.verification.agentic_retrieval_tracing import build_traced_execute_action_fn, build_agentic_retrieval_trace

    class RewriteUnavailableThenAdjustRetriever:
        top_k = 8
        fetch_k = 35
        def retrieve_more(self, request):
            docs = [
                "transformer attention mechanisms sequence modeling relevant snippet one here",
                "transformer attention mechanisms sequence modeling relevant snippet two here",
            ]
            metas = [{"source_filename": "a.pdf", "chunk_id": "c1"}, {"source_filename": "a.pdf", "chunk_id": "c2"}]
            return {"selected_candidates": tuple(
                {"source_filename": m["source_filename"], "chunk_id": m["chunk_id"], "text": d, "native_scores_by_retriever": {"chroma": 0.9}}
                for d, m in zip(docs, metas)
            )}

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "transformer models use attention mechanisms for sequence modeling", "native_scores_by_retriever": {"chroma": 0.2}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=RewriteUnavailableThenAdjustRetriever(), allowed_source_filenames=frozenset({"a.pdf"}), claim_id="c1",
        claim_text="transformer models use attention mechanisms for sequence modeling",
        initial_candidates=initial_candidates,
    )

    def raising_rewrite_then_real_adjust(action, decision_basis, observation):
        if action == "REWRITE_QUERY":
            raise AgenticRetrievalActionUnavailable("QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos")
        return executor(action, decision_basis, observation)

    traced_fn, step_trace = build_traced_execute_action_fn(raising_rewrite_then_real_adjust)

    obs0 = AgenticRetrievalObservation(
        claim_id="c1", claim_text="transformer models use attention mechanisms for sequence modeling",
        current_query="transformer models use attention mechanisms for sequence modeling", retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=3,
        candidate_count=1, evidence_ids=("a.pdf::c0",), max_relevance_score=0.2,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
    )

    import json
    def planner(prompt):
        return json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})

    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner, execute_action_fn=traced_fn)

    final_observation_dict = None
    if result.final_observation is not None:
        final_observation_dict = {
            "claim_id": result.final_observation.claim_id, "current_query": result.final_observation.current_query,
            "retrieval_round": result.final_observation.retrieval_round, "current_top_k": result.final_observation.current_top_k,
            "remaining_retrieval_budget": result.final_observation.remaining_retrieval_budget,
            "candidate_count": result.final_observation.candidate_count, "evidence_ids": result.final_observation.evidence_ids,
            "max_relevance_score": result.final_observation.max_relevance_score, "grade_result": result.final_observation.grade_result,
            "reason_codes": result.final_observation.reason_codes, "minimum_viable_evidence": result.final_observation.minimum_viable_evidence,
            "query_rewrite_count": result.final_observation.query_rewrite_count,
        }

    agentic_result = {
        "planner_invoked": True, "outcome": result.outcome, "final_observation": final_observation_dict,
        "steps": tuple(result.steps), "initial_grade_result": "INSUFFICIENT",
        "agentic_additional_retrievals_used": 1, "effective_budget_for_verify_claim": 2,
    }
    trace = build_agentic_retrieval_trace(
        claim_id="c1", initial_candidate_count=1, initial_max_relevance_score=0.2, initial_top_k=8,
        raw_transition_attempts=step_trace, agentic_result=agentic_result,
    )

    record = {
        "claim_id": "c1", "section_id": "sec1", "retrieval_requested": 1, "retrieval_rounds": 1,
        "retrieval_status": "COMPLETED_WITH_RESULTS", "retriever_binding_fingerprint": "a" * 64,
        "retrieved_candidate_ids": ("a.pdf::c1", "a.pdf::c2"),
        "retrieved_candidate_records": (
            {"evidence_id": "a.pdf::c1", "source_filename": "a.pdf", "chunk_id": "c1", "query_ids": ("c1",), "text_fingerprint": "b" * 64},
            {"evidence_id": "a.pdf::c2", "source_filename": "a.pdf", "chunk_id": "c2", "query_ids": ("c1",), "text_fingerprint": "c" * 64},
        ),
        "verification_context_snapshot": {
            "claim_id": "c1", "section_id": "sec1",
            "eligible_evidence": (
                {"evidence_id": "a.pdf::c1", "source_filename": "a.pdf", "chunk_id": "c1", "authorized_for_section": True, "text_fingerprint": "b" * 64},
                {"evidence_id": "a.pdf::c2", "source_filename": "a.pdf", "chunk_id": "c2", "authorized_for_section": True, "text_fingerprint": "c" * 64},
            ),
        },
        "agentic_retrieval": {
            "planner_invoked": agentic_result["planner_invoked"], "outcome": agentic_result["outcome"],
            "final_observation": agentic_result["final_observation"], "steps": agentic_result["steps"],
            "initial_grade_result": agentic_result["initial_grade_result"],
            "original_max_additional_retrieval_requests": 3,
            "agentic_additional_retrievals_used": agentic_result["agentic_additional_retrievals_used"],
            "effective_budget_for_verify_claim": agentic_result["effective_budget_for_verify_claim"],
            "trace": trace,
        },
    }

    assert any(s.get("execution_status") == "ACTION_UNAVAILABLE" for s in record["agentic_retrieval"]["steps"])
    result_wrapped = _wrap_records_in_runtime_result([record])
    assert result_wrapped.result_contract_valid is True


if __name__ == "__main__":
    for fn in (
        test_a_valid_agentic_retrieval_record_accepted,
        test_a2_valid_agentic_retrieval_record_with_cycle_accepted,
        test_b_missing_agentic_retrieval_rejected,
        test_c_extra_field_rejected,
        test_c2_extra_field_inside_agentic_retrieval_rejected,
        test_d_invalid_schema_planner_invoked_rejected,
        test_d2_invalid_initial_grade_result_rejected,
        test_d3_invalid_steps_type_rejected,
        test_e_incoherent_budget_rejected,
        test_e2_used_exceeds_original_rejected,
        test_f_action_unavailable_case_accepted,
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
