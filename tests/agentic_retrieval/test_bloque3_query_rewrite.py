"""AGENTIC-RETRIEVAL-BLOQUE-3 (corregido): REWRITE_QUERY real
(determinista, sin LLM) + guardrails de query drift.

Correcciones sobre la versión anterior:
1. rewrite_reason ahora es un parámetro explícito de generate_query_rewrite
   (nunca reason_codes[0]), validado como perteneciente a reason_codes.
2. generate_query_rewrite nunca retorna un NO-OP -- fail-closed
   (QueryRewriteError) si no hay contenido nuevo disponible o
   authorized_sources está vacío. Operación atómica: siempre
   auto-valida antes de retornar.
3. source_terms_used/source_numbers_used deben coincidir EXACTAMENTE
   (completos, sin de más ni de menos) con lo realmente introducido y
   respaldado por candidatos autorizados.
4. validate_query_rewrite impone estrictamente la estrategia aditiva:
   previous_query completa (términos y números) debe permanecer en
   rewritten_query.

NO ejecuta retrieval. NO toca verification_runtime.py/
verification_agent.py/verification_incremental_retriever.py. NO
modifica ningún contador del controller (Bloque 2 permanece intacto)."""

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


_CLAIM = "transformer models use attention mechanisms for sequence modeling"
_CANDIDATES = [
    {"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "transformer models rely on self attention layers for encoding sequences efficiently", "native_scores_by_retriever": {"chroma": 0.8}},
    {"source_filename": "unauthorized.pdf", "chunk_id": "c2", "text": "completely unrelated forbidden secret content about cellular biology processes", "native_scores_by_retriever": {"chroma": 0.99}},
]
_AUTHORIZED = frozenset({"authorized.pdf"})


# ---------------------------------------------------------------
# 1-2. rewrite válido produce query distinta; claim_text permanece idéntico
# ---------------------------------------------------------------

@scenario("BQ3-01. Rewrite válido produce una query distinta de la original")
def test_bq3_01_valid_rewrite_produces_different_query():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    assert result["rewritten_query"] != _CLAIM


@scenario("BQ3-02. claim_text permanece idéntico -- generate_query_rewrite no lo modifica ni lo devuelve alterado")
def test_bq3_02_claim_text_remains_identical():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    claim_before = _CLAIM
    generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    assert _CLAIM == claim_before


# ---------------------------------------------------------------
# 3-6. rechazos básicos de rewritten_query
# ---------------------------------------------------------------

@scenario("BQ3-03. rewritten_query vacía -> rechazo")
def test_bq3_03_empty_rewritten_query_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query="   ", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-04. rewritten_query no-str -> rechazo")
def test_bq3_04_non_string_rewritten_query_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=12345, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-05. rewrite idéntico a previous_query -> rechazo")
def test_bq3_05_identical_rewrite_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-06. Equivalencia solo por case/espacios/puntuación -> rechazo")
def test_bq3_06_trivial_equivalence_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM.upper() + "  ", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# 7-8. términos no autorizados / autorizados
# ---------------------------------------------------------------

@scenario("BQ3-07. Entidad/término nuevo no autorizado -> rechazo")
def test_bq3_07_unauthorized_new_term_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " forbidden", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("forbidden",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-08. Término tomado de candidate autorizado, correctamente trazado -> permitido")
def test_bq3_08_authorized_candidate_term_allowed():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding", claim_text=_CLAIM,
        reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",), source_numbers_used=(),
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar


# ---------------------------------------------------------------
# 9. candidate no autorizado no puede aportar términos
# ---------------------------------------------------------------

@scenario("BQ3-09. Candidate no autorizado no puede aportar términos -- generate_query_rewrite nunca los incorpora")
def test_bq3_09_unauthorized_candidate_never_contributes():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    for forbidden_term in ("forbidden", "secret", "cellular", "biology"):
        assert forbidden_term not in result["source_terms_used"]
        assert forbidden_term not in result["rewritten_query"]


# ---------------------------------------------------------------
# 10. authorized_sources vacío/inválido -> fail-closed
# ---------------------------------------------------------------

@scenario("BQ3-10. authorized_sources no es frozenset/set -> fail-closed")
def test_bq3_10_invalid_authorized_sources_fails_closed():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=["authorized.pdf"],
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-10B. authorized_sources vacío -> fail-closed EXPLÍCITO (corrección de esta ronda: ya no se acepta silenciosamente un NO-OP)")
def test_bq3_10b_empty_authorized_sources_fails_closed():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=frozenset(),
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-10C. Ningún término nuevo disponible (current_query ya contiene todo el vocabulario de los candidatos autorizados) -> fail-closed")
def test_bq3_10c_no_new_terms_available_fails_closed():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    saturated_query = _CLAIM + " transformer models rely self attention layers encoding sequences efficiently"
    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=saturated_query, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-10D. Resultado generado inválido nunca se devuelve al caller -- generate_query_rewrite se auto-valida (operación atómica), confirmado indirectamente: todo resultado exitoso ya pasó validate_query_rewrite sin excepción")
def test_bq3_10d_generated_result_always_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, validate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    # re-validar el mismo resultado no debe lanzar -- prueba que ya pasó por validate internamente
    validate_query_rewrite(
        previous_query=result["previous_query"], rewritten_query=result["rewritten_query"],
        claim_text=_CLAIM, reason_codes=("LOW_COVERAGE",), rewrite_reason=result["rewrite_reason"],
        source_terms_used=result["source_terms_used"], source_numbers_used=result["source_numbers_used"],
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )


# ---------------------------------------------------------------
# 11. source_terms_used debe ser completo y exacto (corrección 3)
# ---------------------------------------------------------------

@scenario("BQ3-11. source_terms_used con un término que NO proviene de ningún candidate autorizado -> rechazo")
def test_bq3_11_source_terms_traceability_violation_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("nonexistent_term",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-11B. source_terms_used incompleto (falta un término realmente introducido) -> rechazo")
def test_bq3_11b_incomplete_source_terms_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding layers", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",),  # falta "layers"
            source_numbers_used=(), candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-11C. source_terms_used declara un término que NO fue realmente incorporado -> rechazo")
def test_bq3_11c_phantom_source_term_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " layers", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",),  # "encoding" no está en rewritten_query
            source_numbers_used=(), candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# 12-13. rewrite_reason / contrato de salida
# ---------------------------------------------------------------

@scenario("BQ3-12. rewrite_reason fuera del vocabulario cerrado -> rechazo")
def test_bq3_12_invalid_rewrite_reason_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="PORQUE_SI_INVENTADO", source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-12B. rewrite_reason == reason_codes[0] por posición, sin ser el real -> ahora rechazado explícitamente (corrección 1)")
def test_bq3_12b_positional_reason_selection_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_RELEVANCE"),
            rewrite_reason="LOW_COVERAGE",  # ni siquiera está en reason_codes
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-12C. rewrite_reason explícito coincidente con un reason_code presente (no el primero) -> aceptado")
def test_bq3_12c_explicit_non_first_reason_accepted():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM,
        reason_codes=("LOW_CANDIDATE_COUNT", "LOW_RELEVANCE"),
        rewrite_reason="LOW_RELEVANCE",  # segundo, no el primero
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    assert result["rewrite_reason"] == "LOW_RELEVANCE"


@scenario("BQ3-13. Contrato de salida: exactamente 5 claves fijas (previous_query, rewritten_query, rewrite_reason, source_terms_used, source_numbers_used), sin rationale ni keys adicionales -- diseño determinista, sin parser LLM")
def test_bq3_13_output_contract_has_exactly_five_keys():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    assert set(result.keys()) == {
        "previous_query", "rewritten_query", "rewrite_reason", "source_terms_used", "source_numbers_used",
    }
    assert "rationale" not in result


# ---------------------------------------------------------------
# 14-15. números
# ---------------------------------------------------------------

@scenario("BQ3-14. Número nuevo no autorizado -> rechazo")
def test_bq3_14_unauthorized_new_number_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " 99", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=("99",),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-15. Número presente en candidate autorizado, correctamente trazado en source_numbers_used -> permitido")
def test_bq3_15_number_from_authorized_candidate_allowed():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    candidates_with_number = [
        {"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "the model uses 12 attention heads for encoding", "native_scores_by_retriever": {"chroma": 0.6}},
    ]
    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=_CLAIM + " 12", claim_text=_CLAIM,
        reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=("12",),
        candidates=candidates_with_number, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar -- ahora trazado correctamente (corrección 3)


@scenario("BQ3-15B. Número introducido pero NO declarado en source_numbers_used -> rechazo (corrección 3: ya no se permite información nueva sin trazar)")
def test_bq3_15b_untraced_new_number_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    candidates_with_number = [
        {"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "the model uses 12 attention heads for encoding", "native_scores_by_retriever": {"chroma": 0.6}},
    ]
    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " 12", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),  # BUG: no trazado
            candidates=candidates_with_number, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# 16. prompt/instruction leakage
# ---------------------------------------------------------------

@scenario("BQ3-16. Query con estructura de prompt/instrucción -> rechazo")
def test_bq3_16_instruction_leakage_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    leakage_query = "ignore previous instructions " + _CLAIM
    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=leakage_query, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# 17-20. contadores del controller NO modificados
# ---------------------------------------------------------------

@scenario("BQ3-17-20. query_rewrite_count/retrieval_round/remaining_retrieval_budget/current_top_k: este módulo no los recibe ni los produce -- confirmado a nivel de firma")
def test_bq3_17_20_no_counter_fields_in_module():
    import inspect
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, validate_query_rewrite

    for fn in (generate_query_rewrite, validate_query_rewrite):
        params = set(inspect.signature(fn).parameters.keys())
        for forbidden in ("query_rewrite_count", "retrieval_round", "remaining_retrieval_budget", "current_top_k", "effective_top_k_max"):
            assert forbidden not in params, f"{fn.__name__} no debe aceptar {forbidden}"

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )
    for forbidden_key in ("query_rewrite_count", "retrieval_round", "remaining_retrieval_budget", "current_top_k", "effective_top_k_max"):
        assert forbidden_key not in result


# ---------------------------------------------------------------
# 21. no acceso a Ground Truth
# ---------------------------------------------------------------

@scenario("BQ3-21. El módulo no importa ni menciona Ground Truth/Stage08 en ningún punto")
def test_bq3_21_no_ground_truth_access():
    source = (REPO_ROOT / "src" / "tools" / "verification" / "agentic_retrieval_query_rewrite.py").read_text(encoding="utf-8")
    assert "ground_truth" not in source.lower().replace("ground truth", "ground_truth")
    assert "stage08" not in source.lower().replace("stage 08", "stage08")
    assert "evaluation" not in source.lower()


# ---------------------------------------------------------------
# 22-25. Bloque 2 y producción sin cambios
# ---------------------------------------------------------------

@scenario("BQ3-22. Controller de Bloque 2 permanece sin cambios")
def test_bq3_22_block2_controller_untouched():
    source = (REPO_ROOT / "src" / "tools" / "verification" / "agentic_retrieval_query_rewrite.py").read_text(encoding="utf-8")
    assert "agentic_retrieval_controller" not in source


@scenario("BQ3-23. verification_runtime.py puede importar/usar la integración Agentic Retrieval (Bloque 5, aprobado), pero NO reimplementa internamente generate_query_rewrite ni la lógica de query rewrite de Bloque 3 -- la responsabilidad sigue siendo runtime -> executor -> agentic_retrieval_query_rewrite, nunca runtime -> copia de lógica de rewrite")
def test_bq3_23_verification_runtime_delegates_query_rewrite_not_reimplements():
    source = (REPO_ROOT / "src" / "adapters" / "verification_runtime.py").read_text(encoding="utf-8")
    # No reimplementación: ninguna definición propia de la función/lógica
    # de query rewrite dentro del runtime.
    assert "def generate_query_rewrite" not in source
    assert "def validate_query_rewrite" not in source
    assert "QUERY_REWRITE_UNAVAILABLE" not in source
    # No acoplamiento directo: el runtime no importa el módulo de Bloque 3
    # directamente -- la responsabilidad pasa por el executor (Bloque 4).
    assert "agentic_retrieval_query_rewrite" not in source
    # Delegación real confirmada: el runtime SÍ usa el executor, que es
    # quien internamente invoca generate_query_rewrite (Bloque 4, ya
    # cerrado) -- runtime -> executor -> query_rewrite, nunca runtime
    # -> query_rewrite directamente.
    assert "AgenticRetrievalActionExecutor" in source
    assert "agentic_retrieval_action_executor" in source


@scenario("BQ3-24. verification_agent.py permanece sin cambios")
def test_bq3_24_verification_agent_untouched():
    source = (REPO_ROOT / "src" / "agents" / "verification_agent.py").read_text(encoding="utf-8")
    assert "agentic_retrieval" not in source


@scenario("BQ3-25. verification_incremental_retriever.py permanece sin cambios")
def test_bq3_25_verification_incremental_retriever_untouched():
    source = (REPO_ROOT / "src" / "adapters" / "verification_incremental_retriever.py").read_text(encoding="utf-8")
    assert "agentic_retrieval" not in source


# ---------------------------------------------------------------
# Adicionales: longitud, estrategia aditiva estricta (corrección 4)
# ---------------------------------------------------------------

@scenario("BQ3-26. rewritten_query excede max_length -> rechazo")
def test_bq3_26_excessive_length_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    huge_query = _CLAIM + " " + "encoding " * 200
    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=huge_query, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED, max_length=100,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-27. Estrategia aditiva: rewritten_query elimina un término de previous_query -> rechazo")
def test_bq3_27_additive_strategy_missing_term_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    previous = "transformer models use attention mechanisms"
    rewritten_missing_models = "transformer use attention mechanisms encoding"  # perdió "models"
    try:
        validate_query_rewrite(
            previous_query=previous, rewritten_query=rewritten_missing_models, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-28. Estrategia aditiva: números existentes en previous_query desaparecen -> rechazo")
def test_bq3_28_additive_strategy_missing_number_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    previous = _CLAIM + " with 8 attention heads"
    rewritten_missing_number = _CLAIM + " with attention heads encoding"  # perdió "8"
    try:
        validate_query_rewrite(
            previous_query=previous, rewritten_query=rewritten_missing_number, claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-29. previous_query completa + expansión autorizada -> válido")
def test_bq3_29_full_previous_plus_authorized_expansion_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding layers", claim_text=_CLAIM,
        reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=("encoding", "layers"), source_numbers_used=(),
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar


@scenario("BQ3-30. NOT_DETERMINISTICALLY_ENFORCEABLE está declarado explícitamente")
def test_bq3_30_not_deterministically_enforceable_declared():
    from src.tools.verification.agentic_retrieval_query_rewrite import NOT_DETERMINISTICALLY_ENFORCEABLE

    assert len(NOT_DETERMINISTICALLY_ENFORCEABLE) >= 1
    assert "polarity_or_negation_preservation" in NOT_DETERMINISTICALLY_ENFORCEABLE


# ---------------------------------------------------------------
# Endurecimiento de inputs (sin coerciones silenciosas)
# ---------------------------------------------------------------

@scenario("BQ3-31. claim_text/current_query no-str o vacíos -> rechazo, sin coacción vía str(...)")
def test_bq3_31_strict_input_typing_no_silent_coercion():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    for bad_claim in (123, None, ["x"], ""):
        try:
            generate_query_rewrite(
                claim_text=bad_claim, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
                rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
            )
            raised = False
        except QueryRewriteError:
            raised = True
        assert raised, f"claim_text={bad_claim!r} debió rechazarse"


@scenario("BQ3-32. max_new_terms/max_length no-int o <=0 -> rechazo")
def test_bq3_32_strict_numeric_params():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    for bad_value in (0, -1, "8", 1.5, True):
        try:
            generate_query_rewrite(
                claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
                rewrite_reason="LOW_COVERAGE", candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
                max_new_terms=bad_value,
            )
            raised = False
        except QueryRewriteError:
            raised = True
        assert raised, f"max_new_terms={bad_value!r} debió rechazarse"


# ---------------------------------------------------------------
# Punto 1 (ronda 2): selección por relevancia (score), no alfabética
# ---------------------------------------------------------------

@scenario("BQ3-33. Candidate con score alto y término 'zeta' se selecciona antes que candidate con score bajo y término 'alpha' -- ranking por relevancia, no alfabético")
def test_bq3_33_selection_by_score_not_alphabetical():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates = [
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "zeta term appears here", "native_scores_by_retriever": {"chroma": 0.9}},
        {"source_filename": "b.pdf", "chunk_id": "c2", "text": "alpha term appears here", "native_scores_by_retriever": {"chroma": 0.1}},
    ]
    authorized = frozenset({"a.pdf", "b.pdf"})
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates, authorized_sources=authorized, max_new_terms=1,
    )
    assert "zeta" in result["source_terms_used"]
    assert "alpha" not in result["source_terms_used"]


@scenario("BQ3-34. Mismo score entre dos candidates -> desempate determinista por source_filename ASC")
def test_bq3_34_deterministic_tie_break_by_source_filename():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates_tie = [
        {"source_filename": "b.pdf", "chunk_id": "c1", "text": "wordb here", "native_scores_by_retriever": {"chroma": 0.5}},
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "worda here", "native_scores_by_retriever": {"chroma": 0.5}},
    ]
    authorized = frozenset({"a.pdf", "b.pdf"})
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates_tie, authorized_sources=authorized, max_new_terms=1,
    )
    assert "worda" in result["source_terms_used"]  # a.pdf gana el tie-break alfabético


@scenario("BQ3-35. Candidate NO autorizado con score mayor nunca aporta términos, sin importar su ranking")
def test_bq3_35_unauthorized_high_score_never_contributes():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates = [
        {"source_filename": "unauthorized.pdf", "chunk_id": "c1", "text": "supreme important term here", "native_scores_by_retriever": {"chroma": 0.99}},
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "legit term here", "native_scores_by_retriever": {"chroma": 0.1}},
    ]
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates, authorized_sources=frozenset({"a.pdf"}),
        max_new_terms=5,
    )
    assert "supreme" not in result["source_terms_used"]
    assert "important" not in result["source_terms_used"]


@scenario("BQ3-36. Orden de aparición dentro de un candidate se preserva (no se reordena alfabéticamente el texto de un mismo candidate)")
def test_bq3_36_within_candidate_appearance_order_preserved():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates = [
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "zeta comes before appears in this text", "native_scores_by_retriever": {"chroma": 0.9}},
    ]
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates, authorized_sources=frozenset({"a.pdf"}),
        max_new_terms=1,
    )
    # "zeta" aparece antes que "appears" en el texto -- debe elegirse "zeta", no el alfabéticamente primero
    assert result["source_terms_used"] == ("zeta",)


# ---------------------------------------------------------------
# Punto 2 (ronda 2): candidates endurecidos, sin coacción vía str(...)
# ---------------------------------------------------------------

@scenario("BQ3-37. candidate.source_filename int -> rechazo")
def test_bq3_37_candidate_source_filename_int_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=[{"source_filename": 123, "text": "some text"}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-38. candidate.source_filename vacío -> rechazo")
def test_bq3_38_candidate_source_filename_empty_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=[{"source_filename": "  ", "text": "some text"}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-39. candidate.text como list -> rechazo")
def test_bq3_39_candidate_text_list_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=[{"source_filename": "authorized.pdf", "text": ["transformer", "attention"]}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-40. candidate.text vacío -> rechazo")
def test_bq3_40_candidate_text_empty_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", candidates=[{"source_filename": "authorized.pdf", "text": "   "}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-41. candidate.score no numérico -> rechazo (participa en el ranking, se valida su tipo)")
def test_bq3_41_candidate_invalid_score_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "text": "some text here", "native_scores_by_retriever": {"chroma": "high"}}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 3 (ronda 2): validate_query_rewrite fail-closed por sí misma
# ---------------------------------------------------------------

@scenario("BQ3-42. validate_query_rewrite: previous_query=123 (int) -> rechazo directo, sin depender de generate_query_rewrite")
def test_bq3_42_validate_directly_rejects_non_str_previous_query():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=123, rewritten_query="x", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=[], authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-43. validate_query_rewrite: previous_query='' (vacío) -> rechazo directo")
def test_bq3_43_validate_directly_rejects_empty_previous_query():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query="", rewritten_query="x", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=[], authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-44. validate_query_rewrite: claim_text=None -> rechazo directo")
def test_bq3_44_validate_directly_rejects_none_claim_text():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query="x", claim_text=None,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=[], authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-45. validate_query_rewrite: max_length=True (bool) -> rechazo directo")
def test_bq3_45_validate_directly_rejects_bool_max_length():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query="x", claim_text=_CLAIM,
            reason_codes=("LOW_CANDIDATE_COUNT", "LOW_SOURCE_DIVERSITY", "LOW_RELEVANCE", "LOW_COVERAGE"),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=[], authorized_sources=_AUTHORIZED, max_length=True,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-46. validate_query_rewrite: max_length=0 -> rechazo directo")
def test_bq3_46_validate_directly_rejects_zero_max_length():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query="x", claim_text=_CLAIM,
            reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE", source_terms_used=(), source_numbers_used=(),
            candidates=[], authorized_sources=_AUTHORIZED, max_length=0,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 1 (ronda 3): score numérico Y finito, mismo contrato que Bloque 1
# ---------------------------------------------------------------

@scenario("BQ3-47. candidate.score=NaN -> rechazo")
def test_bq3_47_candidate_score_nan_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "text": "encoding layers", "native_scores_by_retriever": {"chroma": float("nan")}}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-48. candidate.score=+inf -> rechazo")
def test_bq3_48_candidate_score_positive_infinity_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "text": "encoding layers", "native_scores_by_retriever": {"chroma": float("inf")}}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-49. candidate.score=-inf -> rechazo")
def test_bq3_49_candidate_score_negative_infinity_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "text": "encoding layers", "native_scores_by_retriever": {"chroma": float("-inf")}}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-50. candidate.score finito negativo -> permitido por ahora (no se impone rango [0,1])")
def test_bq3_50_candidate_score_finite_negative_allowed():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE",
        candidates=[{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers", "native_scores_by_retriever": {"chroma": -0.5}}],
        authorized_sources=_AUTHORIZED,
    )
    assert result["source_terms_used"]  # no lanzó, produjo un rewrite real


@scenario("BQ3-51. candidate.score finito > 1 -> permitido por ahora (no se impone rango [0,1])")
def test_bq3_51_candidate_score_finite_above_one_allowed():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE",
        candidates=[{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers", "native_scores_by_retriever": {"chroma": 1.5}}],
        authorized_sources=_AUTHORIZED,
    )
    assert result["source_terms_used"]


# ---------------------------------------------------------------
# Punto 2 (ronda 3): validate_query_rewrite verifica rewrite_reason ∈ reason_codes
# ---------------------------------------------------------------

@scenario("BQ3-52. validate_query_rewrite: rewrite_reason presente en reason_codes -> válido")
def test_bq3_52_rewrite_reason_present_in_reason_codes_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding", claim_text=_CLAIM,
        reason_codes=("LOW_RELEVANCE",), rewrite_reason="LOW_RELEVANCE",
        source_terms_used=("encoding",), source_numbers_used=(),
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar


@scenario("BQ3-53. validate_query_rewrite: rewrite_reason enum válido pero ausente de reason_codes -> rechazo")
def test_bq3_53_rewrite_reason_valid_enum_but_absent_from_reason_codes_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding", claim_text=_CLAIM,
            reason_codes=("LOW_RELEVANCE",), rewrite_reason="LOW_COVERAGE",  # enum válido, pero no ocurrió
            source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 3 (ronda 3): previous_query como bloque inicial exacto
# ---------------------------------------------------------------

@scenario("BQ3-54. rewrite elimina 'AI' de previous_query='AI model performance' -> rechazo")
def test_bq3_54_rewrite_removes_short_relevant_token_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query="AI model performance", rewritten_query="model performance encoding",
            claim_text=_CLAIM, reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
            source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-55. previous_query con puntuación técnica (C++) alterada por el rewrite -> rechazo")
def test_bq3_55_rewrite_alters_technical_punctuation_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query="C++ implementation details", rewritten_query="C implementation details encoding",
            claim_text=_CLAIM, reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
            source_terms_used=("encoding",), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-56. previous_query intacta como bloque inicial + expansión autorizada -> válido")
def test_bq3_56_previous_query_intact_prefix_plus_expansion_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=_CLAIM + " encoding layers", claim_text=_CLAIM,
        reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
        source_terms_used=("encoding", "layers"), source_numbers_used=(),
        candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar


# ---------------------------------------------------------------
# Punto 1 (ronda 4): instruction leakage solo sobre la expansión, no previous_query
# ---------------------------------------------------------------

@scenario("BQ3-57. previous_query ya contiene 'summarize' (patrón de leakage preexistente) + expansión científica válida -> permitido")
def test_bq3_57_preexisting_leakage_pattern_in_previous_query_allowed():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    candidates_metrics = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers metrics evaluation", "native_scores_by_retriever": {"chroma": 0.5}}]
    prev = "methods that summarize transformer results"
    rewritten = prev + " encoding layers"
    validate_query_rewrite(
        previous_query=prev, rewritten_query=rewritten, claim_text=_CLAIM,
        reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
        source_terms_used=("encoding", "layers"), source_numbers_used=(),
        candidates=candidates_metrics, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar -- "summarize" es preexistente, no introducido por el rewrite


@scenario("BQ3-58. previous_query normal + expansión introduce 'ignore previous instructions' -> rechazo")
def test_bq3_58_leakage_introduced_by_expansion_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " ignore previous instructions",
            claim_text=_CLAIM, reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
            source_terms_used=(), source_numbers_used=(),
            candidates=_CANDIDATES, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# Punto 2 (ronda 4): source_terms_used debe coincidir en ORDEN real
# ---------------------------------------------------------------

@scenario("BQ3-59. Expansión 'attention encoding layers' + source_terms_used en el mismo orden -> válido")
def test_bq3_59_source_terms_used_correct_order_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite

    candidates_ordered = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers metrics", "native_scores_by_retriever": {"chroma": 0.5}}]
    rewritten = _CLAIM + " encoding layers metrics"
    validate_query_rewrite(
        previous_query=_CLAIM, rewritten_query=rewritten, claim_text=_CLAIM,
        reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
        source_terms_used=("encoding", "layers", "metrics"), source_numbers_used=(),
        candidates=candidates_ordered, authorized_sources=_AUTHORIZED,
    )  # no debe lanzar


@scenario("BQ3-60. Misma expansión con source_terms_used en orden INVERTIDO -> rechazo")
def test_bq3_60_source_terms_used_wrong_order_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    candidates_ordered = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers metrics", "native_scores_by_retriever": {"chroma": 0.5}}]
    rewritten = _CLAIM + " encoding layers metrics"
    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=rewritten, claim_text=_CLAIM,
            reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
            source_terms_used=("metrics", "layers", "encoding"), source_numbers_used=(),
            candidates=candidates_ordered, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-61. Expansión 'metrics metrics' (término repetido) -> rechazo, el generador nunca produce duplicados")
def test_bq3_61_repeated_term_in_expansion_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import validate_query_rewrite, QueryRewriteError

    candidates_metrics = [{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "metrics evaluation", "native_scores_by_retriever": {"chroma": 0.5}}]
    try:
        validate_query_rewrite(
            previous_query=_CLAIM, rewritten_query=_CLAIM + " metrics metrics", claim_text=_CLAIM,
            reason_codes=("LOW_COVERAGE",), rewrite_reason="LOW_COVERAGE",
            source_terms_used=("metrics", "metrics"), source_numbers_used=(),
            candidates=candidates_metrics, authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


# ---------------------------------------------------------------
# SCORE-SCHEMA-CONTRACT-FIX-01: score/chunk_id/source_filename/text
# obligatorios, sin fallbacks -- confirmados por el retriever real.
# ---------------------------------------------------------------

@scenario("BQ3-62. candidate sin native_scores_by_retriever (sin score) -> rechazo")
def test_bq3_62_candidate_without_score_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers metrics"}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-63. candidate sin chunk_id -> rechazo (ahora obligatorio, confirmado como identificador estable real del retriever)")
def test_bq3_63_candidate_without_chunk_id_rejected():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite, QueryRewriteError

    try:
        generate_query_rewrite(
            claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
            rewrite_reason="LOW_COVERAGE",
            candidates=[{"source_filename": "authorized.pdf", "text": "encoding layers metrics", "native_scores_by_retriever": {"chroma": 0.6}}],
            authorized_sources=_AUTHORIZED,
        )
        raised = False
    except QueryRewriteError:
        raised = True
    assert raised


@scenario("BQ3-64. candidate completo (source_filename + chunk_id + text + native_scores_by_retriever['chroma']) -> válido")
def test_bq3_64_complete_real_schema_candidate_valid():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_COVERAGE",),
        rewrite_reason="LOW_COVERAGE",
        candidates=[{"source_filename": "authorized.pdf", "chunk_id": "c1", "text": "encoding layers metrics", "native_scores_by_retriever": {"chroma": 0.6}}],
        authorized_sources=_AUTHORIZED,
    )
    assert result["source_terms_used"]


@scenario("BQ3-65. Ranking por score DESC sigue intacto con el schema real corregido")
def test_bq3_65_ranking_by_real_score_desc_intact():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates = [
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "zeta term appears here", "native_scores_by_retriever": {"chroma": 0.9}},
        {"source_filename": "b.pdf", "chunk_id": "c2", "text": "alpha term appears here", "native_scores_by_retriever": {"chroma": 0.1}},
    ]
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates,
        authorized_sources=frozenset({"a.pdf", "b.pdf"}), max_new_terms=1,
    )
    assert "zeta" in result["source_terms_used"]
    assert "alpha" not in result["source_terms_used"]


@scenario("BQ3-66. Tie-break determinista sigue intacto con el schema real corregido")
def test_bq3_66_tie_break_intact_with_real_schema():
    from src.tools.verification.agentic_retrieval_query_rewrite import generate_query_rewrite

    candidates_tie = [
        {"source_filename": "b.pdf", "chunk_id": "c1", "text": "wordb here", "native_scores_by_retriever": {"chroma": 0.5}},
        {"source_filename": "a.pdf", "chunk_id": "c1", "text": "worda here", "native_scores_by_retriever": {"chroma": 0.5}},
    ]
    result = generate_query_rewrite(
        claim_text=_CLAIM, current_query=_CLAIM, reason_codes=("LOW_RELEVANCE",),
        rewrite_reason="LOW_RELEVANCE", candidates=candidates_tie,
        authorized_sources=frozenset({"a.pdf", "b.pdf"}), max_new_terms=1,
    )
    assert "worda" in result["source_terms_used"]


if __name__ == "__main__":
    for fn in (
        test_bq3_01_valid_rewrite_produces_different_query,
        test_bq3_02_claim_text_remains_identical,
        test_bq3_03_empty_rewritten_query_rejected,
        test_bq3_04_non_string_rewritten_query_rejected,
        test_bq3_05_identical_rewrite_rejected,
        test_bq3_06_trivial_equivalence_rejected,
        test_bq3_07_unauthorized_new_term_rejected,
        test_bq3_08_authorized_candidate_term_allowed,
        test_bq3_09_unauthorized_candidate_never_contributes,
        test_bq3_10_invalid_authorized_sources_fails_closed,
        test_bq3_10b_empty_authorized_sources_fails_closed,
        test_bq3_10c_no_new_terms_available_fails_closed,
        test_bq3_10d_generated_result_always_valid,
        test_bq3_11_source_terms_traceability_violation_rejected,
        test_bq3_11b_incomplete_source_terms_rejected,
        test_bq3_11c_phantom_source_term_rejected,
        test_bq3_12_invalid_rewrite_reason_rejected,
        test_bq3_12b_positional_reason_selection_rejected,
        test_bq3_12c_explicit_non_first_reason_accepted,
        test_bq3_13_output_contract_has_exactly_five_keys,
        test_bq3_14_unauthorized_new_number_rejected,
        test_bq3_15_number_from_authorized_candidate_allowed,
        test_bq3_15b_untraced_new_number_rejected,
        test_bq3_16_instruction_leakage_rejected,
        test_bq3_17_20_no_counter_fields_in_module,
        test_bq3_21_no_ground_truth_access,
        test_bq3_22_block2_controller_untouched,
        test_bq3_23_verification_runtime_delegates_query_rewrite_not_reimplements,
        test_bq3_24_verification_agent_untouched,
        test_bq3_25_verification_incremental_retriever_untouched,
        test_bq3_26_excessive_length_rejected,
        test_bq3_27_additive_strategy_missing_term_rejected,
        test_bq3_28_additive_strategy_missing_number_rejected,
        test_bq3_29_full_previous_plus_authorized_expansion_valid,
        test_bq3_30_not_deterministically_enforceable_declared,
        test_bq3_31_strict_input_typing_no_silent_coercion,
        test_bq3_32_strict_numeric_params,
        test_bq3_33_selection_by_score_not_alphabetical,
        test_bq3_34_deterministic_tie_break_by_source_filename,
        test_bq3_35_unauthorized_high_score_never_contributes,
        test_bq3_36_within_candidate_appearance_order_preserved,
        test_bq3_37_candidate_source_filename_int_rejected,
        test_bq3_38_candidate_source_filename_empty_rejected,
        test_bq3_39_candidate_text_list_rejected,
        test_bq3_40_candidate_text_empty_rejected,
        test_bq3_41_candidate_invalid_score_rejected,
        test_bq3_42_validate_directly_rejects_non_str_previous_query,
        test_bq3_43_validate_directly_rejects_empty_previous_query,
        test_bq3_44_validate_directly_rejects_none_claim_text,
        test_bq3_45_validate_directly_rejects_bool_max_length,
        test_bq3_46_validate_directly_rejects_zero_max_length,
        test_bq3_47_candidate_score_nan_rejected,
        test_bq3_48_candidate_score_positive_infinity_rejected,
        test_bq3_49_candidate_score_negative_infinity_rejected,
        test_bq3_50_candidate_score_finite_negative_allowed,
        test_bq3_51_candidate_score_finite_above_one_allowed,
        test_bq3_52_rewrite_reason_present_in_reason_codes_valid,
        test_bq3_53_rewrite_reason_valid_enum_but_absent_from_reason_codes_rejected,
        test_bq3_54_rewrite_removes_short_relevant_token_rejected,
        test_bq3_55_rewrite_alters_technical_punctuation_rejected,
        test_bq3_56_previous_query_intact_prefix_plus_expansion_valid,
        test_bq3_57_preexisting_leakage_pattern_in_previous_query_allowed,
        test_bq3_58_leakage_introduced_by_expansion_rejected,
        test_bq3_59_source_terms_used_correct_order_valid,
        test_bq3_60_source_terms_used_wrong_order_rejected,
        test_bq3_61_repeated_term_in_expansion_rejected,
        test_bq3_62_candidate_without_score_rejected,
        test_bq3_63_candidate_without_chunk_id_rejected,
        test_bq3_64_complete_real_schema_candidate_valid,
        test_bq3_65_ranking_by_real_score_desc_intact,
        test_bq3_66_tie_break_intact_with_real_schema,
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
