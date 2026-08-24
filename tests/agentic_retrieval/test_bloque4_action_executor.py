"""AGENTIC-RETRIEVAL-BLOQUE-4 (F-N): action executor real -- REWRITE_QUERY
y ADJUST_TOP_K con retriever REAL, grader REAL (Bloque 1), query
rewrite REAL (Bloque 3), controller REAL (Bloque 2).

NO integra todavía dentro de verification_runtime.py/
verification_agent.py -- eso es wiring posterior."""

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


_CLAIM = "transformer models use attention mechanisms"


class _FakeCollection:
    """Chroma fake determinista: mejora la relevancia cuando la query
    contiene 'encoding', para simular un rewrite útil."""

    def __init__(self, n_docs=15):
        self.queries_seen = []
        self.n_results_seen = []
        self._n_docs = n_docs

    def query(self, query_texts, n_results):
        self.queries_seen.append(query_texts[0])
        self.n_results_seen.append(n_results)
        if "encoding" in query_texts[0]:
            docs = [
                "transformer models rely on self attention encoding layers for sequence modeling",
                "another authorized snippet about encoding metrics",
            ]
            metas = [{"source_filename": "a.pdf", "chunk_id": "c1"}, {"source_filename": "a.pdf", "chunk_id": "c2"}]
            dists = [0.05, 0.1]
        else:
            docs = [f"doc {i}" for i in range(self._n_docs)]
            metas = [{"source_filename": "a.pdf", "chunk_id": f"c{i}"} for i in range(self._n_docs)]
            dists = [0.5] * self._n_docs
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}


def _make_retriever(collection, top_k=8, fetch_k=35):
    from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever

    return Agent07ChromaRetriever(
        collection=collection, experiment_id="e1", collection_name="col1", embedding_model="m1",
        chroma_manifest_fingerprint="f1", chunks_manifest_fingerprint="f2", top_k=top_k, fetch_k=fetch_k,
    )


def _make_observation(**overrides):
    from src.tools.verification.agentic_retrieval_controller import AgenticRetrievalObservation

    defaults = dict(
        claim_id="claim1", claim_text=_CLAIM, current_query=_CLAIM, retrieval_round=0,
        current_top_k=8, effective_top_k_max=35, remaining_retrieval_budget=3,
        candidate_count=1, evidence_ids=("a.pdf::c0",), max_relevance_score=0.1,
        grade_result="INSUFFICIENT", reason_codes=("LOW_RELEVANCE",), minimum_viable_evidence=True, query_rewrite_count=0,
    )
    defaults.update(overrides)
    return AgenticRetrievalObservation(**defaults)


# ---------------------------------------------------------------
# REWRITE_QUERY real
# ---------------------------------------------------------------

@scenario("BQ4E-01. REWRITE_QUERY real: query cambia, top_k no cambia, rewrite_count+1, round+1, budget-1, retrieval real, grader real")
def test_bq4e_01_rewrite_query_full_real_invariants():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    coll = _FakeCollection()
    retriever = _make_retriever(coll)
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "mostly unrelated snippet with encoding term present", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation()
    new_obs = executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)

    assert new_obs.current_query != obs.current_query
    assert new_obs.current_top_k == obs.current_top_k
    assert new_obs.query_rewrite_count == obs.query_rewrite_count + 1
    assert new_obs.retrieval_round == obs.retrieval_round + 1
    assert new_obs.remaining_retrieval_budget == obs.remaining_retrieval_budget - 1
    assert new_obs.candidate_count == 2
    assert new_obs.evidence_ids == ("a.pdf::c1", "a.pdf::c2")
    assert coll.queries_seen[0] == new_obs.current_query


# ---------------------------------------------------------------
# ADJUST_TOP_K real
# ---------------------------------------------------------------

@scenario("BQ4E-02. ADJUST_TOP_K real: query idéntica, top_k aumenta <= effective_top_k_max, rewrite_count idéntico, round+1, budget-1, retrieval real, grader real")
def test_bq4e_02_adjust_top_k_full_real_invariants():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    coll = _FakeCollection(n_docs=15)
    retriever = _make_retriever(coll)
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "some initial text here", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation(reason_codes=("LOW_CANDIDATE_COUNT",))
    new_obs = executor("ADJUST_TOP_K", "EVIDENCE_INSUFFICIENT_LOW_CANDIDATE_COUNT", obs)

    assert new_obs.current_query == obs.current_query
    assert new_obs.current_top_k > obs.current_top_k
    assert new_obs.current_top_k <= obs.effective_top_k_max
    assert new_obs.query_rewrite_count == obs.query_rewrite_count
    assert new_obs.retrieval_round == obs.retrieval_round + 1
    assert new_obs.remaining_retrieval_budget == obs.remaining_retrieval_budget - 1
    assert coll.n_results_seen[0] == retriever.fetch_k  # n_results a Chroma sigue siendo fetch_k, no top_k


# ---------------------------------------------------------------
# decision_basis -> rewrite_reason: conversión de formato correcta
# ---------------------------------------------------------------

@scenario("BQ4E-03. decision_basis con prefijo EVIDENCE_INSUFFICIENT_ se convierte correctamente al reason_code desnudo para Bloque 3, sin re-derivar desde reason_codes")
def test_bq4e_03_decision_basis_prefix_conversion():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    coll = _FakeCollection()
    retriever = _make_retriever(coll)
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "mostly unrelated snippet with encoding term present", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation(reason_codes=("LOW_CANDIDATE_COUNT", "LOW_RELEVANCE"))
    executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
    assert executor.rewrite_trace[0]["rewrite_reason"] == "LOW_RELEVANCE"


# ---------------------------------------------------------------
# Frontera dura de allowed_source_filenames (K)
# ---------------------------------------------------------------

@scenario("BQ4E-04. REWRITE_QUERY: fuente no autorizada con score alto nunca entra a candidate_count/evidence_ids resultantes")
def test_bq4e_04_rewrite_query_never_crosses_authorization_boundary():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    class UnauthorizedHighScoreCollection:
        def query(self, query_texts, n_results):
            docs = ["unauthorized high relevance content", "authorized encoding content here"]
            metas = [{"source_filename": "unauthorized.pdf", "chunk_id": "u1"}, {"source_filename": "a.pdf", "chunk_id": "c1"}]
            dists = [0.01, 0.3]
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    retriever = _make_retriever(UnauthorizedHighScoreCollection())
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "encoding term present here", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation()
    new_obs = executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
    assert all("unauthorized" not in eid for eid in new_obs.evidence_ids)
    assert all(eid.startswith("a.pdf::") for eid in new_obs.evidence_ids)


# ---------------------------------------------------------------
# candidate schema real funciona con Bloques 1/3
# ---------------------------------------------------------------

@scenario("BQ4E-05. El schema real de candidatos (native_scores_by_retriever, chunk_id, source_filename, text) fluye correctamente desde el retriever hasta Bloque 1/3 sin transformación intermedia")
def test_bq4e_05_real_candidate_schema_flows_through_blocks():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
    from src.tools.verification.agentic_retrieval_grader import extract_candidate_relevance_score

    coll = _FakeCollection()
    retriever = _make_retriever(coll)
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "mostly unrelated snippet with encoding term present", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation()
    executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
    # confirmar que los candidatos actuales del executor ya vienen con el schema real, extraíble por Bloque 1
    for c in executor._current_candidates:
        score = extract_candidate_relevance_score(c)
        assert isinstance(score, float)


# ---------------------------------------------------------------
# Observation resultante pasa los invariantes de Bloque 2
# ---------------------------------------------------------------

@scenario("BQ4E-06. Ciclo aislado real completo: Observation inicial INSUFFICIENT -> run_agentic_retrieval_cycle -> acción real ejecutada -> nueva Observation, usando fake de Chroma determinista pero pasando por el retriever real")
def test_bq4e_06_full_isolated_real_cycle():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
    from src.tools.verification.agentic_retrieval_controller import run_agentic_retrieval_cycle

    coll = _FakeCollection()
    retriever = _make_retriever(coll)
    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "mostly unrelated snippet with encoding term present", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs0 = _make_observation()

    def planner(prompt):
        return json.dumps({"selected_action": "REWRITE_QUERY", "decision_basis": "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"})

    result = run_agentic_retrieval_cycle(initial_observation=obs0, invoke_planner_fn=planner, execute_action_fn=executor)
    assert result.outcome in ("ACCEPT_EVIDENCE", "FINISH_UNRESOLVED")
    assert result.final_observation.candidate_count == len(result.final_observation.evidence_ids)


# ---------------------------------------------------------------
# Corrección 1: coherencia contexto/Observation
# ---------------------------------------------------------------

def _mismatched_context_executor(retriever):
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "encoding term present here", "native_scores_by_retriever": {"chroma": 0.1}}]
    return AgenticRetrievalActionExecutor(
        retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )


@scenario("BQ4E-07. claim_id distinto entre executor y Observation -> fail-closed")
def test_bq4e_07_mismatched_claim_id_fails_closed():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(claim_id="claim_DIFFERENT", evidence_ids=("a.pdf::c0",))
    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


@scenario("BQ4E-08. claim_text distinto entre executor y Observation -> fail-closed")
def test_bq4e_08_mismatched_claim_text_fails_closed():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(claim_text="a completely different claim text", current_query="a completely different claim text", evidence_ids=("a.pdf::c0",))
    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


@scenario("BQ4E-09. evidence_ids de la Observation no corresponde a los candidatos actuales del executor -> fail-closed")
def test_bq4e_09_mismatched_evidence_ids_fails_closed():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(evidence_ids=("b.pdf::other",))  # no coincide con self._current_candidates ("a.pdf::c0")
    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


@scenario("BQ4E-10. Contexto coherente (claim_id/claim_text/evidence_ids alineados) -> ejecución normal")
def test_bq4e_10_coherent_context_executes_normally():
    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(evidence_ids=("a.pdf::c0",))
    new_obs = executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
    assert new_obs.retrieval_round == obs.retrieval_round + 1


# ---------------------------------------------------------------
# Corrección 2: decision_basis validado, no confía ciegamente
# ---------------------------------------------------------------

@scenario("BQ4E-11. REWRITE_QUERY + EVIDENCE_ACCEPTABLE_DESPITE_GAPS (exclusivo de ACCEPT_EVIDENCE) -> rechazo, aunque removeprefix() produciría un string técnicamente utilizable")
def test_bq4e_11_accept_evidence_basis_rejected_for_rewrite_query():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(evidence_ids=("a.pdf::c0",))
    try:
        executor("REWRITE_QUERY", "EVIDENCE_ACCEPTABLE_DESPITE_GAPS", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


@scenario("BQ4E-12. REWRITE_QUERY + decision_basis sin el reason_code presente en observation.reason_codes -> rechazo")
def test_bq4e_12_decision_basis_reason_code_not_in_observation_rejected():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection()))
    obs = _make_observation(evidence_ids=("a.pdf::c0",), reason_codes=("LOW_CANDIDATE_COUNT",))
    try:
        # LOW_RELEVANCE no está en observation.reason_codes (solo LOW_CANDIDATE_COUNT)
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


@scenario("BQ4E-13. ADJUST_TOP_K también valida decision_basis, aunque no lo use para calcular el número")
def test_bq4e_13_adjust_top_k_also_validates_decision_basis():
    from src.tools.verification.agentic_retrieval_action_executor import ActionExecutorError

    executor = _mismatched_context_executor(_make_retriever(_FakeCollection(n_docs=15)))
    obs = _make_observation(evidence_ids=("a.pdf::c0",), reason_codes=("LOW_CANDIDATE_COUNT",))
    try:
        executor("ADJUST_TOP_K", "EVIDENCE_ACCEPTABLE_DESPITE_GAPS", obs)
        raised = False
    except ActionExecutorError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Corrección 3: rewrite_trace solo tras éxito completo
# ---------------------------------------------------------------

@scenario("BQ4E-14. Retriever lanza excepción tras generar el rewrite -> rewrite_trace permanece vacío")
def test_bq4e_14_retriever_failure_leaves_rewrite_trace_empty():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor

    class FailingRetriever:
        def retrieve_more(self, request):
            raise RuntimeError("simulated retriever failure")

    initial_candidates = [{"source_filename": "a.pdf", "chunk_id": "c0", "text": "encoding term present here", "native_scores_by_retriever": {"chroma": 0.1}}]
    executor = AgenticRetrievalActionExecutor(
        retriever=FailingRetriever(), allowed_source_filenames=frozenset({"a.pdf"}), claim_id="claim1",
        claim_text=_CLAIM, initial_candidates=initial_candidates,
    )
    obs = _make_observation(evidence_ids=("a.pdf::c0",))
    try:
        executor("REWRITE_QUERY", "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE", obs)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert executor.rewrite_trace == []


# ---------------------------------------------------------------
# Endurecimiento del constructor
# ---------------------------------------------------------------

@scenario("BQ4E-15. Constructor: claim_id/claim_text no-str o vacíos -> rechazo")
def test_bq4e_15_constructor_rejects_invalid_claim_fields():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor, ActionExecutorError

    retriever = _make_retriever(_FakeCollection())
    for bad_claim_id in (123, "", "   ", None):
        try:
            AgenticRetrievalActionExecutor(
                retriever=retriever, allowed_source_filenames=frozenset({"a.pdf"}), claim_id=bad_claim_id,
                claim_text=_CLAIM, initial_candidates=[],
            )
            raised = False
        except ActionExecutorError:
            raised = True
        assert raised, f"claim_id={bad_claim_id!r} debió rechazarse"


@scenario("BQ4E-16. Constructor: allowed_source_filenames vacío o con elementos inválidos -> rechazo")
def test_bq4e_16_constructor_rejects_invalid_authorized_sources():
    from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor, ActionExecutorError

    retriever = _make_retriever(_FakeCollection())
    for bad_sources in (frozenset(), frozenset({123}), frozenset({""})):
        try:
            AgenticRetrievalActionExecutor(
                retriever=retriever, allowed_source_filenames=bad_sources, claim_id="claim1",
                claim_text=_CLAIM, initial_candidates=[],
            )
            raised = False
        except ActionExecutorError:
            raised = True
        assert raised, f"allowed_source_filenames={bad_sources!r} debió rechazarse"


if __name__ == "__main__":
    for fn in (
        test_bq4e_01_rewrite_query_full_real_invariants,
        test_bq4e_02_adjust_top_k_full_real_invariants,
        test_bq4e_03_decision_basis_prefix_conversion,
        test_bq4e_04_rewrite_query_never_crosses_authorization_boundary,
        test_bq4e_05_real_candidate_schema_flows_through_blocks,
        test_bq4e_06_full_isolated_real_cycle,
        test_bq4e_07_mismatched_claim_id_fails_closed,
        test_bq4e_08_mismatched_claim_text_fails_closed,
        test_bq4e_09_mismatched_evidence_ids_fails_closed,
        test_bq4e_10_coherent_context_executes_normally,
        test_bq4e_11_accept_evidence_basis_rejected_for_rewrite_query,
        test_bq4e_12_decision_basis_reason_code_not_in_observation_rejected,
        test_bq4e_13_adjust_top_k_also_validates_decision_basis,
        test_bq4e_14_retriever_failure_leaves_rewrite_trace_empty,
        test_bq4e_15_constructor_rejects_invalid_claim_fields,
        test_bq4e_16_constructor_rejects_invalid_authorized_sources,
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
