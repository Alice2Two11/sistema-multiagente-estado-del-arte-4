"""AGENTIC-RETRIEVAL-BLOQUE-4 (B-E): overrides retrocompatibles en
Agent07ChromaRetriever.retrieve_more (query_override/top_k_override).

No se cambió: collection, embeddings, fetch_k, scoring, dedupe,
allowed_source_filenames, source filtering, native_scores_by_retriever,
fused_rrf_score. self.top_k nunca se muta."""

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


class _FakeCollection:
    def __init__(self, documents=None, metadatas=None, distances=None):
        self.last_query = None
        self.last_n_results = None
        self._documents = documents or []
        self._metadatas = metadatas or []
        self._distances = distances or []

    def query(self, query_texts, n_results):
        self.last_query = query_texts[0]
        self.last_n_results = n_results
        return {"documents": [self._documents], "metadatas": [self._metadatas], "distances": [self._distances]}


def _default_docs(n=10, authorized="a.pdf", unauthorized="unauthorized.pdf"):
    docs, metas, dists = [], [], []
    for i in range(n):
        docs.append(f"doc text {i}")
        metas.append({"source_filename": authorized if i % 2 == 0 else unauthorized, "chunk_id": f"c{i}"})
        dists.append(0.1 * i)
    return docs, metas, dists


def _make_retriever(collection, top_k=3, fetch_k=35):
    from src.adapters.verification_incremental_retriever import Agent07ChromaRetriever

    return Agent07ChromaRetriever(
        collection=collection, experiment_id="e1", collection_name="col1", embedding_model="m1",
        chroma_manifest_fingerprint="f1", chunks_manifest_fingerprint="f2", top_k=top_k, fetch_k=fetch_k,
    )


# ---------------------------------------------------------------
# Retrocompatibilidad (punto M)
# ---------------------------------------------------------------

@scenario("BQ4-01. request sin overrides -> misma query=claim_text, mismo self.top_k, comportamiento idéntico al anterior")
def test_bq4_01_backward_compatible_no_overrides():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll, top_k=3)
    result = retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "original claim"},
        "allowed_source_filenames": ("a.pdf",),
    })
    assert coll.last_query == "original claim"
    assert len(result["selected_candidates"]) == 3
    assert retriever.top_k == 3
    assert result["queries"][0]["query_text"] == "original claim"


# ---------------------------------------------------------------
# query_override (punto C)
# ---------------------------------------------------------------

@scenario("BQ4-02. query_override ausente -> claim_text")
def test_bq4_02_query_override_absent_uses_claim_text():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll)
    retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "original claim"},
        "allowed_source_filenames": ("a.pdf",),
    })
    assert coll.last_query == "original claim"


@scenario("BQ4-03. query_override válido -> Chroma recibe rewritten_query, registrado en queries[].query_text")
def test_bq4_03_query_override_valid_reaches_chroma():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll)
    result = retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "original claim"},
        "allowed_source_filenames": ("a.pdf",), "query_override": "rewritten query",
    })
    assert coll.last_query == "rewritten query"
    assert result["queries"][0]["query_text"] == "rewritten query"


@scenario("BQ4-04. query_override vacío/no-str -> rechazo")
def test_bq4_04_query_override_invalid_rejected():
    coll = _FakeCollection()
    retriever = _make_retriever(coll)
    base = {"claim_id": "c1", "claim_context": {"claim_text": "x"}, "allowed_source_filenames": ("a.pdf",)}
    for bad in ("", "   ", 123):
        try:
            retriever.retrieve_more({**base, "query_override": bad})
            raised = False
        except ValueError:
            raised = True
        assert raised, f"query_override={bad!r} debió rechazarse"


# ---------------------------------------------------------------
# top_k_override (punto D)
# ---------------------------------------------------------------

@scenario("BQ4-05. top_k_override ausente -> self.top_k")
def test_bq4_05_top_k_override_absent_uses_self_top_k():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll, top_k=3)
    result = retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "x"},
        "allowed_source_filenames": ("a.pdf",),
    })
    assert len(result["selected_candidates"]) == 3


@scenario("BQ4-06. top_k_override válido -> cambia el corte de esa llamada concreta")
def test_bq4_06_top_k_override_valid_changes_cutoff():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll, top_k=3)
    result = retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "x"},
        "allowed_source_filenames": ("a.pdf",), "top_k_override": 5,
    })
    assert len(result["selected_candidates"]) == 5


@scenario("BQ4-07. top_k_override bool/0/negativo/>fetch_k -> rechazo")
def test_bq4_07_top_k_override_invalid_rejected():
    coll = _FakeCollection()
    retriever = _make_retriever(coll, top_k=8, fetch_k=35)
    base = {"claim_id": "c1", "claim_context": {"claim_text": "x"}, "allowed_source_filenames": ("a.pdf",)}
    for bad in (True, False, 0, -1, 100, "8", 1.5):
        try:
            retriever.retrieve_more({**base, "top_k_override": bad})
            raised = False
        except ValueError:
            raised = True
        assert raised, f"top_k_override={bad!r} debió rechazarse"


@scenario("BQ4-08. self.top_k nunca muta, incluso tras usar top_k_override repetidamente")
def test_bq4_08_self_top_k_never_mutates():
    docs, metas, dists = _default_docs()
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll, top_k=3)
    retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "x"},
        "allowed_source_filenames": ("a.pdf",), "top_k_override": 5,
    })
    assert retriever.top_k == 3
    retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "x"},
        "allowed_source_filenames": ("a.pdf",), "top_k_override": 7,
    })
    assert retriever.top_k == 3


# ---------------------------------------------------------------
# Frontera dura de allowed_source_filenames (punto K)
# ---------------------------------------------------------------

@scenario("BQ4-09. Candidato de score muy alto pero fuente NO autorizada nunca entra a selected_candidates, incluso con query_override/top_k_override activos")
def test_bq4_09_unauthorized_high_score_never_selected_with_overrides():
    docs = ["highest relevance text", "authorized text"]
    metas = [{"source_filename": "unauthorized.pdf", "chunk_id": "c1"}, {"source_filename": "a.pdf", "chunk_id": "c2"}]
    dists = [0.01, 0.5]  # unauthorized.pdf tiene la MEJOR distancia (score mas alto)
    coll = _FakeCollection(docs, metas, dists)
    retriever = _make_retriever(coll, top_k=8)
    result = retriever.retrieve_more({
        "claim_id": "c1", "claim_context": {"claim_text": "x"},
        "allowed_source_filenames": ("a.pdf",),
        "query_override": "rewritten query", "top_k_override": 8,
    })
    sources = [c["source_filename"] for c in result["selected_candidates"]]
    assert "unauthorized.pdf" not in sources
    assert sources == ["a.pdf"]


if __name__ == "__main__":
    for fn in (
        test_bq4_01_backward_compatible_no_overrides,
        test_bq4_02_query_override_absent_uses_claim_text,
        test_bq4_03_query_override_valid_reaches_chroma,
        test_bq4_04_query_override_invalid_rejected,
        test_bq4_05_top_k_override_absent_uses_self_top_k,
        test_bq4_06_top_k_override_valid_changes_cutoff,
        test_bq4_07_top_k_override_invalid_rejected,
        test_bq4_08_self_top_k_never_mutates,
        test_bq4_09_unauthorized_high_score_never_selected_with_overrides,
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
