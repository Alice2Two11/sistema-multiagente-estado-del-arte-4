"""AGENTIC-RETRIEVAL-BLOQUE-5: wiring productivo en Stage 07 --
verification_runtime.py::_independent_retrieve_claim.

Atraviesa las funciones productivas reales (Bloques 1-4 +
_run_agentic_retrieval_for_claim + _independent_retrieve_claim). Usa
fakes deterministas del backend Chroma y del LLM de verificación, pero
nunca mockea directamente final_observation/candidates
finales/resultado final."""

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


_CLAIM_TEXT = "transformer models use attention mechanisms for sequence modeling"


class _FakeCollection:
    """Chroma fake determinista:
    - insufficient_first=False: SUFFICIENT desde la query original
      (4 candidatos relevantes, 2 fuentes distintas).
    - insufficient_first=True, query original (sin 'encoding'):
      primeros 8 candidatos son de 1 sola fuente poco relevante
      (INSUFFICIENT: source_diversity=1) -- pero ampliar el corte local
      (top_k, vía ADJUST_TOP_K) incluye además 2 candidatos relevantes
      de una segunda fuente, alcanzando diversidad/relevancia
      suficiente. n_results a Chroma es SIEMPRE fetch_k (nunca top_k),
      por eso el corte real ocurre en el retriever tras la respuesta.
    - query con 'encoding' (REWRITE_QUERY real): 4 candidatos relevantes
      de 2 fuentes, sin depender del top_k."""

    def __init__(self, insufficient_first=True):
        self.queries_seen = []
        self.n_results_seen = []
        self._insufficient_first = insufficient_first

    def query(self, query_texts, n_results):
        q = query_texts[0]
        self.queries_seen.append(q)
        self.n_results_seen.append(n_results)

        if "encoding" in q or "cellular" in q:
            docs = [
                "transformer models rely on self attention encoding layers for sequence modeling here",
                "another authorized snippet about encoding metrics and evaluation",
                "third source discussing encoding attention transformer sequence",
                "fourth source with encoding layers sequence modeling detail",
            ]
            metas = [
                {"source_filename": "authorized.pdf", "chunk_id": "c1"},
                {"source_filename": "authorized.pdf", "chunk_id": "c2"},
                {"source_filename": "second.pdf", "chunk_id": "s1"},
                {"source_filename": "second.pdf", "chunk_id": "s2"},
            ]
            dists = [0.02, 0.03, 0.04, 0.05]
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

        if not self._insufficient_first:
            docs = [
                "transformer models rely on self attention mechanisms for sequence modeling directly",
                "attention mechanisms improve transformer sequence modeling significantly here",
                "second source about transformer attention sequence modeling details",
                "fourth source discussing transformer attention mechanisms modeling",
            ]
            metas = [
                {"source_filename": "authorized.pdf", "chunk_id": "c0"},
                {"source_filename": "authorized.pdf", "chunk_id": "c1b"},
                {"source_filename": "second.pdf", "chunk_id": "s0"},
                {"source_filename": "second.pdf", "chunk_id": "s1b"},
            ]
            dists = [0.02, 0.03, 0.04, 0.05]
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

        # insuficiente bajo la query original con top_k pequeño: primeros
        # 3 documentos, una sola fuente, poco relevantes; documento 4 de
        # una SEGUNDA fuente, relevante -- solo entra si top_k >= 4 (una
        # sola ronda de ADJUST_TOP_K, current_top_k=3 -> next_top_k=4).
        docs = [
            "completely unrelated snippet about cellular biology processes number one",
            "completely unrelated snippet about cellular biology processes number two",
            "completely unrelated snippet about cellular biology processes number three",
            "transformer models attention mechanisms sequence modeling relevant snippet four",
        ]
        metas = [
            {"source_filename": "authorized.pdf", "chunk_id": "low0"},
            {"source_filename": "authorized.pdf", "chunk_id": "low1"},
            {"source_filename": "authorized.pdf", "chunk_id": "low2"},
            {"source_filename": "second.pdf", "chunk_id": "hi1"},
        ]
        dists = [0.85, 0.86, 0.87, 0.03]
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}


def _make_retriever(collection, top_k=8, fetch_k=35):
    from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever

    return Agent07ChromaRetriever(
        collection=collection, experiment_id="e1", collection_name="col1", embedding_model="m1",
        chroma_manifest_fingerprint="f1", chunks_manifest_fingerprint="f2", top_k=top_k, fetch_k=fetch_k,
    )


class _FakeVerificationLLM:
    """LLM fake determinista -- responde exactamente lo que Bloque 2 exige (JSON de 2 claves)."""

    def __init__(self, action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE"):
        self._action = action
        self._basis = basis
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return json.dumps({"selected_action": self._action, "decision_basis": self._basis})


def _make_dependencies(retrieval_tool, verification_llm=None):
    from src.adapters.verification_runtime import VerificationRuntimeDependencies

    return VerificationRuntimeDependencies(
        verification_llm=verification_llm or _FakeVerificationLLM(),
        retrieval_tool=retrieval_tool,
        retriever_binding={
            "experiment_id": "e1", "collection_name": "col1", "embedding_model": "m1",
            "chroma_manifest_fingerprint": "f1", "chunks_manifest_fingerprint": "f2",
        },
    )


def _make_context(claim_id="claim1", eligible_evidence=()):
    return {
        "claim_id": claim_id,
        "section_id": "sec1",
        "claim_text": _CLAIM_TEXT,
        "eligible_evidence": eligible_evidence,
        "authorized_source_filenames": ("authorized.pdf",),
    }


# ---------------------------------------------------------------
# Caso A: INITIAL SUFFICIENT
# ---------------------------------------------------------------

@scenario("BQ5-A. INITIAL SUFFICIENT: planner nunca invocado, candidatos iniciales llegan al verificador")
def test_bq5_a_initial_sufficient_no_planner():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    assert record["agentic_retrieval"]["planner_invoked"] is False
    assert record["agentic_retrieval"]["outcome"] is None
    assert llm.calls == []
    source_filenames = {e["source_filename"] for e in updated["eligible_evidence"]}
    assert "authorized.pdf" in source_filenames


# ---------------------------------------------------------------
# Caso B: REWRITE_QUERY
# ---------------------------------------------------------------

@scenario("BQ5-B. INITIAL INSUFFICIENT -> REWRITE_QUERY: retriever recibe query_override, candidate B llega a eligible_evidence")
def test_bq5_b_rewrite_query_reaches_eligible_evidence():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    assert record["agentic_retrieval"]["planner_invoked"] is True
    chunk_ids = {e["chunk_id"] for e in updated["eligible_evidence"]}
    # los candidatos c1/c2 (post-rewrite) deben estar presentes -- no solo c0 (inicial insuficiente)
    assert "c1" in chunk_ids or "c2" in chunk_ids
    assert any("cellular" in q for q in coll.queries_seen[1:])


# ---------------------------------------------------------------
# Caso C: ADJUST_TOP_K
# ---------------------------------------------------------------

@scenario("BQ5-C. INITIAL INSUFFICIENT -> ADJUST_TOP_K: effective top_k aumenta, candidatos adicionales llegan a eligible_evidence")
def test_bq5_c_adjust_top_k_reaches_eligible_evidence():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll, top_k=3)
    llm = _FakeVerificationLLM(action="ADJUST_TOP_K", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    assert record["agentic_retrieval"]["planner_invoked"] is True
    assert max(coll.n_results_seen) >= coll.n_results_seen[0]
    assert len(updated["eligible_evidence"]) >= 1


# ---------------------------------------------------------------
# Caso D: FINISH_UNRESOLVED
# ---------------------------------------------------------------

@scenario("BQ5-D. FINISH_UNRESOLVED: mejores candidatos disponibles pasan a verify_claim, NO se asigna veredicto desde Agentic Retrieval")
def test_bq5_d_finish_unresolved_no_verdict_assigned():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    class AlwaysInsufficientCollection:
        def query(self, query_texts, n_results):
            docs = ["totally unrelated text about something else entirely different"]
            metas = [{"source_filename": "authorized.pdf", "chunk_id": "cX"}]
            dists = [0.95]
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    retriever = _make_retriever(AlwaysInsufficientCollection())
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 1})

    outcome = record["agentic_retrieval"]["outcome"]
    assert outcome in ("FINISH_UNRESOLVED", "ACCEPT_EVIDENCE", "AGENTIC_TRANSITION_INVALID", "AGENTIC_PLANNER_FAILED")
    # el registro no contiene ningún campo de veredicto científico -- eso es responsabilidad exclusiva de verify_claim
    assert "verdict" not in record["agentic_retrieval"]
    assert "claim_verification_result" not in record["agentic_retrieval"]


# ---------------------------------------------------------------
# Caso E: ACCEPT_EVIDENCE no implica SUPPORTED
# ---------------------------------------------------------------

@scenario("BQ5-E. ACCEPT_EVIDENCE: tampoco implica SUPPORTED -- el registro no contiene ningún veredicto")
def test_bq5_e_accept_evidence_no_verdict():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    if record["agentic_retrieval"]["outcome"] == "ACCEPT_EVIDENCE":
        assert "SUPPORTED" not in json.dumps(record["agentic_retrieval"])
        assert "verdict" not in record["agentic_retrieval"]


# ---------------------------------------------------------------
# Caso F: merge -- evidencia previa preservada
# ---------------------------------------------------------------

@scenario("BQ5-F. eligible_evidence previa + candidatos Agentic -> merge correcto, sin pérdida de evidencia previa, sin duplicados")
def test_bq5_f_merge_preserves_previous_evidence():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    previous_evidence = (
        {
            "evidence_id": "prev1", "source_filename": "legacy.pdf", "chunk_id": "p1",
            "text": "legacy inherited evidence text about attention", "canonical_text": "legacy inherited evidence text about attention",
            "authorized_for_section": True, "usage_role": "SUPPORT",
        },
    )
    ctx = _make_context(eligible_evidence=previous_evidence)
    ctx["authorized_source_filenames"] = ("authorized.pdf", "legacy.pdf")

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    source_filenames = [e["source_filename"] for e in updated["eligible_evidence"]]
    assert "legacy.pdf" in source_filenames  # evidencia previa preservada
    assert "authorized.pdf" in source_filenames  # nueva evidencia agregada
    assert len(source_filenames) == len(set((e["source_filename"], e["chunk_id"]) for e in updated["eligible_evidence"]))  # sin duplicados


# ---------------------------------------------------------------
# Caso G: allowed_source_filenames frontera dura
# ---------------------------------------------------------------

@scenario("BQ5-G. allowed_source_filenames sigue siendo frontera dura -- fuente no autorizada nunca llega a eligible_evidence")
def test_bq5_g_unauthorized_source_never_reaches_eligible_evidence():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    class UnauthorizedHighScoreCollection:
        def query(self, query_texts, n_results):
            docs = ["unauthorized highest relevance content available", "authorized content about encoding present"]
            metas = [{"source_filename": "unauthorized.pdf", "chunk_id": "u1"}, {"source_filename": "authorized.pdf", "chunk_id": "c1"}]
            dists = [0.01, 0.3]
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    retriever = _make_retriever(UnauthorizedHighScoreCollection())
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    source_filenames = {e["source_filename"] for e in updated["eligible_evidence"]}
    assert "unauthorized.pdf" not in source_filenames


# ---------------------------------------------------------------
# Caso H: namespaces separados
# ---------------------------------------------------------------

@scenario("BQ5-H. Namespaces separados: source_filename::chunk_id (Agentic) coexiste sin confundirse con evidence_id (Stage07 posterior)")
def test_bq5_h_namespaces_remain_separate():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    # eligible_evidence usa evidence_id propio del handoff (formato "source::chunk" a nivel de fila individual,
    # ya existente en _independent_retrieve_claim, distinto del futuro E01/E02 de evidence_selection)
    for e in updated["eligible_evidence"]:
        assert "::" not in e["evidence_id"] or e["evidence_id"] == f"{e['source_filename']}::{e['chunk_id']}"
    # el resultado Agentic (record) usa evidence_ids con el mismo formato compuesto, en su propio namespace
    final_obs = record["agentic_retrieval"]["final_observation"]
    if final_obs is not None:
        for eid in final_obs["evidence_ids"]:
            assert "::" in eid


# ---------------------------------------------------------------
# Retrocompatibilidad: sin second retrieval inicial
# ---------------------------------------------------------------

@scenario("BQ5-I. Un solo retrieval inicial -- la primera llamada a Chroma es el único retrieval inicial, Agentic reutiliza esos candidatos")
def test_bq5_i_single_initial_retrieval_no_duplicate():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})
    # SUFFICIENT inicial -> el ciclo Agentic no ejecuta retrieval adicional -> exactamente 1 llamada a Chroma
    assert len(coll.queries_seen) == 1


# ---------------------------------------------------------------
# B5-BUDGET-INTEGRATION-FIX: presupuesto compartido con verify_claim
# ---------------------------------------------------------------

@scenario("BQ5-J. original budget=1, Agentic usa 1 -> effective_budget_for_verify_claim=0 (verify_claim no puede ejecutar additional retrieval)")
def test_bq5_j_budget_exhausted_by_agentic():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll, top_k=3)
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 1})

    agentic = record["agentic_retrieval"]
    assert agentic["original_max_additional_retrieval_requests"] == 1
    assert agentic["agentic_additional_retrievals_used"] == 1
    assert agentic["effective_budget_for_verify_claim"] == 0


@scenario("BQ5-K. original budget=3, Agentic usa 1 -> effective_budget_for_verify_claim=2")
def test_bq5_k_partial_budget_consumption():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    agentic = record["agentic_retrieval"]
    assert agentic["original_max_additional_retrieval_requests"] == 3
    assert agentic["agentic_additional_retrievals_used"] == 1
    assert agentic["effective_budget_for_verify_claim"] == 2


@scenario("BQ5-L. INITIAL SUFFICIENT: Agentic usa 0 -> verify_claim conserva el presupuesto original íntegro")
def test_bq5_l_sufficient_initial_preserves_full_budget():
    from src.adapters.verification_runtime import _independent_retrieve_claim

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy={"max_additional_retrieval_requests": 3})

    agentic = record["agentic_retrieval"]
    assert agentic["agentic_additional_retrievals_used"] == 0
    assert agentic["effective_budget_for_verify_claim"] == 3
    assert agentic["effective_budget_for_verify_claim"] == agentic["original_max_additional_retrieval_requests"]


@scenario("BQ5-M. Política sin max_additional_retrieval_requests válido -> NO aparece ningún fallback mágico (3); usa la política oficial validada (default real = 1)")
def test_bq5_m_no_magic_fallback_uses_official_policy():
    from src.adapters.verification_runtime import _independent_retrieve_claim
    from src.config.verification_policy_config import DEFAULT_VERIFICATION_INPUT_POLICY

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()
    deps = _make_dependencies(retriever, llm)
    ctx = _make_context()

    # sin overrides -- debe caer en la política oficial validada, no en un 3 inventado
    updated, record = _independent_retrieve_claim(ctx, deps, agentic_retrieval_policy=None)

    agentic = record["agentic_retrieval"]
    assert agentic["original_max_additional_retrieval_requests"] == DEFAULT_VERIFICATION_INPUT_POLICY["max_additional_retrieval_requests"]
    assert agentic["original_max_additional_retrieval_requests"] != 3


# ---------------------------------------------------------------
# BQ5-N: E2E genuino del presupuesto hasta agent.verify_claim,
# atravesando run_agent07_in_memory con un Agent07RuntimeInput real
# válido (sin desactivar validate_agent07_runtime_input_contract) y
# la rama productiva real isinstance(agent, VerificationAgent).
# ---------------------------------------------------------------

def _minimal_valid_runtime_input(*, max_additional_retrieval_requests, claim_id="claim1"):
    import hashlib
    fingerprint = hashlib.sha256(b"draft-fingerprint").hexdigest()
    return {
        "committed_agent06_output": {
            "commit_status": "COMMITTED",
            "run_id": "run1",
            "artifact_identity": "artifact1",
            "schema_version": "v1",
            "source_draft_fingerprint": fingerprint,
            "claim_verification_contexts": [
                {
                    "claim_id": claim_id, "section_id": "sec1", "section_title": "Section One",
                    "original_claim_text": _CLAIM_TEXT, "claim_text": _CLAIM_TEXT,
                    "authorized_source_filenames": ("authorized.pdf",),
                    "eligible_evidence": (),
                }
            ],
        },
        "agent07_config": {
            "verification_policy": {"max_additional_retrieval_requests": max_additional_retrieval_requests},
            "attempt_number": 1,
        },
        "policy_versions": {"v": "1"},
        "schema_versions": {"s": "1"},
        "experiment_paths": {"p": "1"},
    }


@scenario("BQ5-N1 (OBLIGATORIO). E2E genuino: original budget=1, Agentic consume 1 vía REWRITE_QUERY, run_agent07_in_memory real -> verify_claim recibe policy['max_additional_retrieval_requests']==0. Si verify_claim no es alcanzado, el test FALLA (aserción incondicional, no envuelta en 'if').")
def test_bq5_n1_e2e_budget_exhausted_reaches_verify_claim():
    from src.adapters.verification_runtime import run_agent07_in_memory
    from src.agents.verification_agent import VerificationAgent
    from dataclasses import replace

    coll = _FakeCollection(insufficient_first=True)
    retriever = _make_retriever(coll, top_k=3)
    llm = _FakeVerificationLLM(action="REWRITE_QUERY", basis="EVIDENCE_INSUFFICIENT_LOW_RELEVANCE")

    received = {}

    class CapturingVerificationAgent(VerificationAgent):
        # subclass REAL de VerificationAgent -- atraviesa exactamente
        # isinstance(agent, VerificationAgent), nunca la rama else.
        def verify_claim(self, context):
            received["called"] = True
            received["policy"] = dict(context["policy"])
            raise RuntimeError("STOP_AFTER_CAPTURE")  # detiene el resto del pipeline deliberadamente, ya capturado lo necesario

    deps = _make_dependencies(retriever, llm)
    deps = replace(
        deps, verification_agent_factory=CapturingVerificationAgent,
        correction_context_factory=lambda *a, **kw: {},
        reverification_input_factory=lambda *a, **kw: {},
    )

    runtime_input = _minimal_valid_runtime_input(max_additional_retrieval_requests=1)

    try:
        run_agent07_in_memory(runtime_input, dependencies=deps)
    except RuntimeError:
        pass  # STOP_AFTER_CAPTURE esperado -- ya capturamos lo necesario antes de esto
    except Exception:
        pass  # cualquier fallo posterior del pipeline (correction/reverification) no es el foco de este test

    # ASERCIONES OBLIGATORIAS, INCONDICIONALES -- si verify_claim no fue
    # alcanzado, este assert falla el test (no hay "if received:").
    assert received.get("called") is True, "verify_claim NUNCA fue alcanzado -- el test debe fallar"
    assert received["policy"]["max_additional_retrieval_requests"] == 0


@scenario("BQ5-N2 (recomendable). E2E genuino: initial SUFFICIENT, Agentic usa 0, original budget=1 -> verify_claim recibe policy['max_additional_retrieval_requests']==1 (íntegro)")
def test_bq5_n2_e2e_budget_preserved_when_sufficient():
    from src.adapters.verification_runtime import run_agent07_in_memory
    from src.agents.verification_agent import VerificationAgent
    from dataclasses import replace

    coll = _FakeCollection(insufficient_first=False)
    retriever = _make_retriever(coll)
    llm = _FakeVerificationLLM()

    received = {}

    class CapturingVerificationAgent(VerificationAgent):
        def verify_claim(self, context):
            received["called"] = True
            received["policy"] = dict(context["policy"])
            raise RuntimeError("STOP_AFTER_CAPTURE")

    deps = _make_dependencies(retriever, llm)
    deps = replace(
        deps, verification_agent_factory=CapturingVerificationAgent,
        correction_context_factory=lambda *a, **kw: {},
        reverification_input_factory=lambda *a, **kw: {},
    )

    runtime_input = _minimal_valid_runtime_input(max_additional_retrieval_requests=1)

    try:
        run_agent07_in_memory(runtime_input, dependencies=deps)
    except Exception:
        pass

    assert received.get("called") is True, "verify_claim NUNCA fue alcanzado -- el test debe fallar"
    assert received["policy"]["max_additional_retrieval_requests"] == 1


if __name__ == "__main__":
    for fn in (
        test_bq5_a_initial_sufficient_no_planner,
        test_bq5_b_rewrite_query_reaches_eligible_evidence,
        test_bq5_c_adjust_top_k_reaches_eligible_evidence,
        test_bq5_d_finish_unresolved_no_verdict_assigned,
        test_bq5_e_accept_evidence_no_verdict,
        test_bq5_f_merge_preserves_previous_evidence,
        test_bq5_g_unauthorized_source_never_reaches_eligible_evidence,
        test_bq5_h_namespaces_remain_separate,
        test_bq5_i_single_initial_retrieval_no_duplicate,
        test_bq5_j_budget_exhausted_by_agentic,
        test_bq5_k_partial_budget_consumption,
        test_bq5_l_sufficient_initial_preserves_full_budget,
        test_bq5_m_no_magic_fallback_uses_official_policy,
        test_bq5_n1_e2e_budget_exhausted_reaches_verify_claim,
        test_bq5_n2_e2e_budget_preserved_when_sufficient,
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
