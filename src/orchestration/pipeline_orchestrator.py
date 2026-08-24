"""Orquestador de las etapas 02-06 usando exclusivamente los componentes de ``src/``.

Diseño
------
Cada etapa migrada ya expone dos piezas reutilizables:

1. Un constructor ``build_real_<etapa>_execution(project_dir, attempt_number)``
   (en ``src/adapters/*_runtime.py``) que arma el agente/capability real y su
   ``AgentInput`` a partir de ``active_experiment.json`` y los artefactos ya
   generados en el proyecto.
2. Un protocolo transaccional (en ``src/runtime/*_protocol.py``) que envuelve
   PREPARE → EXECUTE → persist → COMMIT sobre ``StateStore``, convirtiendo
   cualquier fallo de preparación (dependencia faltante, credencial ausente,
   etc.) en un ``AgentResult`` ``FAILED`` comprometido igualmente al estado,
   en vez de dejar una excepción sin registrar.

Este módulo declara, para cada etapa, su constructor, su protocolo
transaccional y su función de fingerprints, y ofrece un bucle
(``run_pipeline``) que ahora interpreta ``RequestedTransition`` en vez de
limitarse a recorrer ``STAGE_ORDER`` en orden fijo. La semántica de decisión
(ADVANCE/RETRY/RETURN/HALT_STAGE/STOP_PIPELINE, vigencia por fingerprints,
invalidación en cascada) vive en ``decision_engine.py``; este módulo solo la
conecta con la ejecución real de cada etapa:

- reutiliza el ``pipeline_state.json`` canónico del experimento activo;
- si una etapa ya quedó ``COMPLETED`` y sus fingerprints siguen vigentes, la
  salta (``SKIPPED_FRESH``); si los datos de entrada cambiaron, la reejecuta
  aunque no se haya pedido ``force_rerun``;
- si existe una ejecución PENDING de un run anterior, la resuelve (COMMITTED
  o REEXECUTE) antes de continuar, igual que hacen los notebooks;
- sigue la transición que cada etapa solicitó (validada primero: ver
  ``decision_engine.validate_transition``), no un orden fijo.

No orquesta notebooks ni depende de ellos: sólo importa símbolos de ``src/``.

Etapas 01 (ingesta) y 08 (evaluación) no están incluidas: según
``LEEME_PRIMERO.md`` su ejecución operativa permanece en el notebook de Drive
y no exponen un constructor equivalente en ``src/``. La etapa 07
(verificación) tampoco está incluida todavía: su wiring productivo depende de
módulos que generan los propios notebooks dentro del proyecto (``config.py``,
``llm_utils.py``, ``rag_utils.py``, el retriever de Chroma), no de ``src/``;
añadir un ``StageSpec`` para 07 requiere primero decidir cómo exponer esas
piezas fuera del notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.contracts.agent_result import (
    AgentResult,
    AgentWarning,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
    WarningSeverity,
)
from src.orchestration import decision_engine as de
from src.orchestration.decision_engine import (
    CANONICAL_STAGE_ORDER,
    MAX_ATTEMPTS_DEFAULT,
    ValidatedTransition,
    apply_return_with_cycle,
    default_next_stage,
    invalidate_from,
    is_stage_fresh,
    resolve_cycle_if_active,
    validate_transition,
)
from src.state.fingerprints import build_stage_fingerprints
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore

DRAFT_STAGE_NAME = "06_agente_redactor"


# ---------------------------------------------------------------------------
# Resolución de rutas del experimento activo
# ---------------------------------------------------------------------------


def load_active_experiment(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    active_path = root / "active_experiment.json"
    if not active_path.is_file():
        raise FileNotFoundError(
            "No existe active_experiment.json en "
            f"{root}. Ejecuta primero la etapa 00 (bootstrap del proyecto)."
        )
    return json.loads(active_path.read_text(encoding="utf-8"))


def resolve_state_path(project_dir: str | Path) -> tuple[Path, str, str]:
    """Devuelve (state_path, experiment_id, run_id) del experimento activo."""

    root = Path(project_dir).resolve()
    active = load_active_experiment(root)
    experiment_id = active["active_experiment_id"]
    run_id = active.get("run_id", experiment_id)
    experiment_dir = root / experiment_id
    state_path = (
        experiment_dir / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    )
    return state_path, experiment_id, run_id


def ensure_pipeline_state(project_dir: str | Path) -> StateStore:
    """Abre el ``StateStore`` canónico, inicializándolo si es la primera vez."""

    state_path, experiment_id, run_id = resolve_state_path(project_dir)
    store = StateStore(state_path)
    if not state_path.is_file():
        now = datetime.now(timezone.utc).isoformat()
        store.initialize(
            PipelineState(
                identity=PipelineIdentity(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    created_at=now,
                    updated_at=now,
                    schema_version="1.0",
                )
            )
        )
    return store


# ---------------------------------------------------------------------------
# Constructores reales por etapa (envuelven los `build_real_*` existentes)
# ---------------------------------------------------------------------------


def _real_extraction_execution(project_dir: Path, attempt_number: int):
    from src.adapters.extraction_runtime import (
        build_agent_input,
        build_extraction_runtime,
        load_runtime_configuration,
        resolve_openai_api_key,
    )
    from src.agents.extraction_agent import ExtractionAgent

    api_key = resolve_openai_api_key(project_dir=project_dir, required=True)
    configuration = load_runtime_configuration(project_dir)
    runtime = build_extraction_runtime(configuration, api_key=api_key)
    agent_input = build_agent_input(
        configuration,
        attempt_number=attempt_number,
        runtime_resources={
            "df_chunks_clean": runtime.dataframe,
            "collection": runtime.collection,
        },
    )
    return ExtractionAgent(runtime.dependencies), agent_input


def _real_quantitative_execution(project_dir: Path, attempt_number: int):
    from src.adapters.quantitative_extraction_runtime import (
        build_quantitative_agent_input,
        build_quantitative_capability,
        load_quantitative_configuration,
    )

    # El Agente 03B sólo admite attempt_number=1 en el wiring actual de src/.
    configuration = load_quantitative_configuration(project_dir)
    capability = build_quantitative_capability(configuration)
    agent_input = build_quantitative_agent_input(configuration)
    return capability, agent_input


def _real_thematic_execution(project_dir: Path, attempt_number: int):
    from src.adapters.thematic_analysis_runtime import build_real_thematic_execution

    agent, agent_input, _configuration = build_real_thematic_execution(
        project_dir, attempt_number
    )
    return agent, agent_input


def _real_outline_execution(project_dir: Path, attempt_number: int):
    from src.adapters.outline_generation_runtime import build_real_outline_execution

    agent, agent_input, _configuration = build_real_outline_execution(
        project_dir, attempt_number
    )
    return agent, agent_input


def _resolve_draft_execution_mode(project_dir: Path, store) -> dict[str, Any] | None:
    """Detecta si 06 debe ejecutarse en modo REVISION: hay un ciclo
    ``writer_verifier`` ACTIVE con al menos una ronda usada. Si es así,
    lee la ÚLTIMA ronda persistida por 07 (``writer_verifier_cycle/
    round_NN/writer_revision_request.json``) y el borrador comprometido
    más reciente de 06, y devuelve los ``policy_overrides`` para
    construir el ``AgentInput`` de revisión. Devuelve ``None`` si debe
    ejecutarse en modo INITIAL_DRAFT (sin ciclo activo, o ciclo resuelto)."""

    state = store.load()
    cycle = state.cycles.get("writer_verifier")
    if cycle is None or cycle.status != "ACTIVE" or cycle.rounds_used == 0:
        return None

    import json as _json

    from src.tools.verification.cycle_round_persistence import (
        list_persisted_rounds,
        read_round_artifact,
        read_round_status,
        round_is_persisted,
    )

    active_experiment = _json.loads((project_dir / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active_experiment["active_experiment_id"]

    persisted_rounds = list_persisted_rounds(project_dir=project_dir, experiment_id=experiment_id)
    if not persisted_rounds:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_NOT_PERSISTED: el ciclo writer_verifier está "
            "ACTIVE pero no hay ninguna ronda persistida en writer_verifier_cycle/."
        )

    round_number = persisted_rounds[-1]
    if not round_is_persisted(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number):
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")

    status = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number)
    if status is None:
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")
    if status["status"] not in {"AWAITING_REVISION", "REVISION_COMPLETED"}:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_UNEXPECTED_STATUS: round_{round_number:02d} está en "
            f"{status['status']!r}, se esperaba 'AWAITING_REVISION' o 'REVISION_COMPLETED'."
        )
    # 'REVISION_COMPLETED' significa que 06 YA completó esta ronda en una
    # ejecución previa -- no hay nada nuevo que escribir. Esta función NO
    # decide si 06 se reinvoca: solo reconstruye el MISMO AgentInput con el
    # que se comprometió esa revisión, para que su fingerprint coincida con
    # el ya comprometido y run_stage() lo reconozca como SKIPPED_FRESH sin
    # tocar la ronda. writer_revision_request.json y el borrador previo son
    # los MISMOS archivos persistidos en ambos estados (AWAITING_REVISION y
    # REVISION_COMPLETED) -- solo cambia el campo de estado de la ronda, que
    # no forma parte de este AgentInput. Si, pese a la coincidencia de
    # fingerprint, algo más forzara una reinvocación real de 06 sobre esta
    # misma ronda, complete_round_revision() ya rechaza explícitamente un
    # segundo intento de completarla (ver su docstring) -- esa red de
    # seguridad no se toca aquí.

    writer_revision_request = read_round_artifact(
        project_dir=project_dir, experiment_id=experiment_id, round_number=round_number,
        filename="writer_revision_request.json",
    )

    # Consistencia (punto 5): la ronda debe corresponder al MISMO experimento
    # y ronda activos -- si algo no coincide, no se arma un AgentInput de
    # revisión con datos inconsistentes.
    if writer_revision_request.get("experiment_id") != experiment_id:
        raise RuntimeError(
            "DRAFT_REVISION_EXPERIMENT_MISMATCH: writer_revision_request "
            f"pertenece a {writer_revision_request.get('experiment_id')!r}, "
            f"se esperaba {experiment_id!r}."
        )
    if int(writer_revision_request.get("round_number", -1)) != round_number:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_MISMATCH: writer_revision_request declara ronda "
            f"{writer_revision_request.get('round_number')!r}, se esperaba {round_number}."
        )

    experiment_dir = project_dir / experiment_id
    draft_json_path = experiment_dir / "05_outputs" / "05_draft" / "state_of_art_draft.json"
    if not draft_json_path.is_file():
        raise RuntimeError("DRAFT_REVISION_PREVIOUS_DRAFT_NOT_FOUND")
    previous_draft = _json.loads(draft_json_path.read_text(encoding="utf-8"))

    if previous_draft.get("source_draft_fingerprint") not in (
        None,
        writer_revision_request["source_draft_fingerprint"],
    ):
        raise RuntimeError(
            "DRAFT_REVISION_FINGERPRINT_MISMATCH: el borrador en disco no coincide "
            "con source_draft_fingerprint del writer_revision_request."
        )

    return {
        "mode": "REVISION",
        "writer_revision_request": writer_revision_request,
        "previous_draft": previous_draft,
        "round_number": round_number,
        "cycle_project_dir": str(project_dir),
        "experiment_id": experiment_id,
    }


def _real_draft_execution(project_dir: Path, attempt_number: int):
    from src.adapters.draft_writing_runtime import build_real_draft_execution

    project_dir = Path(project_dir)
    store = ensure_pipeline_state(project_dir)
    revision_overrides = _resolve_draft_execution_mode(project_dir, store)

    agent, agent_input, _configuration = build_real_draft_execution(
        project_dir, attempt_number, policy_overrides=revision_overrides
    )
    return agent, agent_input


def _experimental_verification_execution(project_dir: Path, attempt_number: int):
    from src.adapters.verification_orchestrator_runtime import (
        build_experimental_verification_execution,
    )

    return build_experimental_verification_execution(project_dir, attempt_number)


def _experimental_evaluation_execution(project_dir: Path, attempt_number: int):
    from src.adapters.evaluation_stagespec_wiring import build_execution_for_stagespec

    return build_execution_for_stagespec(project_dir, attempt_number)


def _run_evaluation_stage(**kwargs):
    from src.adapters.evaluation_orchestrator_runtime import (
        _run_evaluation_stage as _real_run_evaluation_stage,
    )

    return _real_run_evaluation_stage(**kwargs)


# ---------------------------------------------------------------------------
# Protocolo transaccional para la etapa 06
# ---------------------------------------------------------------------------
#
# A diferencia de 03B/04/05, `src/runtime/draft_writing_protocol.py` sólo
# expone `execute_draft_transaction`, que requiere el agente y el
# `AgentInput` ya construidos, sin capturar fallos de preparación. Para
# mantener el mismo comportamiento que las demás etapas (una etapa que no
# pudo prepararse queda comprometida como FAILED en vez de perderse como una
# excepción no registrada), replicamos aquí el mismo patrón que ya usan
# `execute_thematic_runtime_transaction` / `execute_outline_runtime_transaction`.


def _draft_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    from src.runtime.draft_writing_protocol import (
        DraftWritingTransactionResult,
        build_draft_fingerprints,
    )

    prepared = store.prepare_execution(
        target_stage=DRAFT_STAGE_NAME,
        intended_action="EXECUTE_DRAFT_WRITING",
        attempt_number=attempt_number,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        agent, agent_input = build_execution()
        result = agent.execute(agent_input)
        fingerprints = build_draft_fingerprints(agent_input)
    except Exception as exc:  # noqa: BLE001 - convertido a AgentResult FAILED
        text = str(exc)
        if isinstance(exc, FileNotFoundError):
            code = "DEPENDENCY_NOT_FOUND"
        elif "OPENAI_API_KEY" in text:
            code = "CREDENTIAL_NOT_FOUND"
        else:
            code = "RUNTIME_DEPENDENCY_FAILED"
        now = datetime.now(timezone.utc).isoformat()
        result = AgentResult(
            execution_status=ExecutionStatus.FAILED,
            quality_status=QualityStatus.REJECTED,
            decision=DecisionInfo(
                code="DRAFT_RUNTIME_FAILED",
                rationale="Falló la preparación de la etapa 06.",
            ),
            quality_metrics={"technical": {}, "scientific": {}},
            warnings=(
                AgentWarning(
                    code=code,
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=text,
                ),
            ),
            failure_reason_codes=(code,),
            requested_transition=RequestedTransition(
                action=TransitionAction.HALT_STAGE,
                target_stage=None,
                reason_code=code,
                requires_human_confirmation=False,
            ),
            output_artifacts={},
            tool_usage=ToolUsage(),
            attempt_number=attempt_number,
            started_at=started_at,
            completed_at=now,
            error={"type": type(exc).__name__, "message": text, "stage": DRAFT_STAGE_NAME},
        )
        fingerprints = build_stage_fingerprints(
            input_data={"stage_name": DRAFT_STAGE_NAME, "attempt_number": attempt_number},
            config_data={"runtime_resolution": "FAILED"},
            dependencies_data={},
        )

    persisted_path = store.persist_agent_result(prepared.decision_id, result)
    committed_state = store.commit_execution(
        decision_id=prepared.decision_id,
        result=result,
        stage_name=DRAFT_STAGE_NAME,
        fingerprints=fingerprints,
        observations=dict(observations or {}),
    )
    return DraftWritingTransactionResult(
        prepared, result, str(persisted_path), committed_state
    )


def _quantitative_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    # execute_quantitative_runtime_transaction no admite attempt_number: la
    # etapa 03B siempre corre como attempt_number=1 en el wiring actual.
    from src.runtime.quantitative_extraction_protocol import (
        execute_quantitative_runtime_transaction,
    )

    return execute_quantitative_runtime_transaction(
        store=store, build_execution=build_execution, observations=observations
    )


# ---------------------------------------------------------------------------
# Ejecución dedicada de la etapa 07 (no encaja en build_execution +
# runtime_transaction + resolve_resume genéricos — ver StageSpec.custom_run)
# ---------------------------------------------------------------------------
#
# A diferencia de 02-06, la etapa 07 ya trae su propia semántica de RESUME
# más rica que {NO_PENDING, COMMITTED, REEXECUTE}
# (resume_agent07_execution devuelve COMMITTED, EXECUTED_NOT_COMMITTED,
# REEXECUTE, NO_COMMIT, FINGERPRINT_MISMATCH, ARTIFACT_MISMATCH o
# MANIFEST_INCOMPLETE — ver src/adapters/verification_notebook.py). Esa misma
# función YA decide internamente si el resultado comprometido sigue vigente
# (compara fingerprints), así que aquí no se duplica ese chequeo: se llama
# siempre, incluso sin pending_execution, y se interpreta su resultado.


def _run_verification_stage(
    *,
    store: StateStore,
    project_dir: str | Path,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    from src.adapters.verification_notebook import (
        commit_executed_agent07,
        execute_prepared_agent07,
        prepare_agent07_execution,
        resume_agent07_execution,
    )

    project_dir = Path(project_dir)
    had_pending = store.load().pending_execution is not None

    try:
        dependencies, runtime_input = spec.build_execution(project_dir, attempt_number)
    except Exception as exc:  # noqa: BLE001
        # Ocurre ANTES de cualquier PREPARE (build_execution no toca
        # StateStore) — a diferencia de 02-06, aquí no se fabrica un
        # AgentResult FAILED sintético para no inventar comportamiento que
        # verification_notebook.py no tiene. Se deja como excepción real,
        # capturada por el try/except genérico de run_stage.
        raise

    def _do_fresh_execution() -> AgentResult:
        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(
            store=store, prepared=prepared, dependencies=dependencies
        )
        commit_executed_agent07(store=store, executed=executed)
        return executed.agent_result

    if force_rerun and not had_pending:
        result = _do_fresh_execution()
        status = (
            "COMMITTED"
            if result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
    else:
        resume = resume_agent07_execution(store=store, runtime_input=runtime_input)
        if resume.action == "COMMITTED":
            result = resume.committed_result
            status = "COMMITTED" if had_pending else "SKIPPED_FRESH"
        elif resume.action == "EXECUTED_NOT_COMMITTED":
            commit_executed_agent07(store=store, executed=resume.executed)
            result = resume.executed.agent_result
            status = "COMMITTED"
        elif resume.action in {
            "NO_COMMIT",
            "REEXECUTE",
            "FINGERPRINT_MISMATCH",
            "ARTIFACT_MISMATCH",
            "MANIFEST_INCOMPLETE",
        }:
            result = _do_fresh_execution()
            status = (
                "COMMITTED"
                if result.execution_status == ExecutionStatus.COMPLETED
                else "FAILED"
            )
        else:  # pragma: no cover - RESUME_ACTIONS es cerrado en el repo real
            raise RuntimeError(
                f"resume_agent07_execution devolvió una acción inesperada: {resume.action}"
            )

    state = store.load()
    attempts_used = state.stages[spec.key].attempts_used
    return _outcome_from_result(spec, result, status, attempts_used=attempts_used)


# ---------------------------------------------------------------------------
# Registro de etapas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    build_execution: Callable[[Path, int], tuple[Any, Any]]
    runtime_transaction: Callable[..., Any]
    resolve_resume: Callable[..., Any]
    # None cuando la etapa no expone (todavía) una función pública de
    # fingerprints con la misma firma que build_thematic_fingerprints/etc.
    # (caso de 07: solo existe una versión privada, _stage_fingerprints, no
    # se importa aquí — ver verification_orchestrator_runtime.py). Cuando es
    # None, run_stage usa el chequeo antiguo (solo COMPLETED, sin comparar
    # vigencia) en vez del chequeo de vigencia por fingerprints.
    build_fingerprints: Callable[[Any], Any] | None = None
    max_attempt_number: int | None = None
    # Punto 7 del pedido: APPROVED_PENDING_MANUAL_REVIEW no se trata como
    # ADVANCE automático salvo que la etapa lo permita explícitamente. Hoy
    # ninguna etapa lo permite (False en las 6); queda aquí como el punto de
    # extensión previsto, no como una regla de negocio ya decidida.
    bypass_manual_review: bool = False
    # Escape hatch: cuando una etapa no encaja en el patrón genérico
    # build_execution+runtime_transaction+resolve_resume (caso de 07, que
    # tiene PREPARE/EXECUTE/COMMIT como 3 llamadas separadas con firmas
    # propias, y semántica de RESUME más rica que {NO_PENDING,COMMITTED,
    # REEXECUTE}), custom_run reemplaza por completo la lógica de run_stage
    # para esa etapa. Firma: (*, store, project_dir, spec, attempt_number,
    # observations, force_rerun) -> StageOutcome.
    custom_run: Callable[..., "StageOutcome"] | None = None


def _stage_registry() -> list[StageSpec]:
    # Los imports quedan diferidos a la primera llamada para no forzar
    # dependencias pesadas (langchain, chromadb) sólo por importar este
    # módulo o inspeccionar el registro.
    from src.runtime.extraction_protocol import (
        build_agent_input_fingerprints,
        execute_extraction_runtime_transaction,
        resolve_extraction_resume,
    )
    from src.runtime.outline_generation_protocol import (
        build_outline_fingerprints,
        execute_outline_runtime_transaction,
        resolve_outline_resume,
    )
    from src.runtime.quantitative_extraction_protocol import (
        build_quantitative_fingerprints,
        resolve_quantitative_resume,
    )
    from src.runtime.thematic_analysis_protocol import (
        build_thematic_fingerprints,
        execute_thematic_runtime_transaction,
        resolve_thematic_resume,
    )
    from src.runtime.draft_writing_protocol import (
        build_draft_fingerprints,
        resolve_draft_resume,
    )

    return [
        StageSpec(
            key="03_agente_extraccion_kb",
            label="02 · Extracción de información científica",
            build_execution=_real_extraction_execution,
            runtime_transaction=execute_extraction_runtime_transaction,
            resolve_resume=resolve_extraction_resume,
            build_fingerprints=build_agent_input_fingerprints,
            max_attempt_number=2,
        ),
        StageSpec(
            key="03B_extraccion_cuantitativa_kb",
            label="03 · Extracción y normalización cuantitativa",
            build_execution=_real_quantitative_execution,
            runtime_transaction=_quantitative_runtime_transaction,
            resolve_resume=resolve_quantitative_resume,
            build_fingerprints=build_quantitative_fingerprints,
            max_attempt_number=1,
        ),
        StageSpec(
            key="04_agente_analisis_tematico",
            label="04 · Análisis temático",
            build_execution=_real_thematic_execution,
            runtime_transaction=execute_thematic_runtime_transaction,
            resolve_resume=resolve_thematic_resume,
            build_fingerprints=build_thematic_fingerprints,
        ),
        StageSpec(
            key="05_generador_esquema",
            label="05 · Generación del esquema",
            build_execution=_real_outline_execution,
            runtime_transaction=execute_outline_runtime_transaction,
            resolve_resume=resolve_outline_resume,
            build_fingerprints=build_outline_fingerprints,
        ),
        StageSpec(
            key=DRAFT_STAGE_NAME,
            label="06 · Redacción del borrador",
            build_execution=_real_draft_execution,
            runtime_transaction=_draft_runtime_transaction,
            resolve_resume=resolve_draft_resume,
            build_fingerprints=build_draft_fingerprints,
        ),
        StageSpec(
            key="07_agente_verificador",
            label="07 · Verificación y trazabilidad",
            build_execution=_experimental_verification_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_verification_stage,
        ),
        StageSpec(
            key="08_evaluacion_experimental",
            label="08 · Evaluación experimental",
            build_execution=_experimental_evaluation_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_evaluation_stage,
        ),
    ]


# Etapas con StageSpec ejecutable hoy. 08_evaluacion_experimental ya tiene
# StageSpec real (a diferencia de rondas anteriores) — nombre
# "experimental" conservado: la equivalencia de configuración de 08 no
# tuvo la misma ronda de verificación campo-por-campo que sí tuvo 07.
STAGE_ORDER: tuple[str, ...] = (
    "03_agente_extraccion_kb",
    "03B_extraccion_cuantitativa_kb",
    "04_agente_analisis_tematico",
    "05_generador_esquema",
    DRAFT_STAGE_NAME,
    "07_agente_verificador",
    "08_evaluacion_experimental",
)


# ---------------------------------------------------------------------------
# Ejecución de una etapa y del pipeline completo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageOutcome:
    key: str
    label: str
    # SKIPPED_FRESH | COMMITTED (via resume) | COMMITTED | FAILED
    status: str
    execution_status: str | None
    quality_status: str | None
    warnings: tuple[str, ...]
    error: Mapping[str, Any] | None
    attempt_number: int
    # Transición ya validada por decision_engine.validate_transition:
    # ADVANCE | RETRY | RETURN | HALT_STAGE | STOP_PIPELINE
    next_action: str
    target_stage: str | None
    reason_code: str


def _validated_transition_for(
    spec: StageSpec,
    *,
    requested_transition: RequestedTransition,
    quality_status: QualityStatus,
    attempts_used: int,
) -> ValidatedTransition:
    return validate_transition(
        current_stage=spec.key,
        requested_transition=requested_transition,
        quality_status=quality_status,
        attempts_used=attempts_used,
        max_attempts=spec.max_attempt_number or MAX_ATTEMPTS_DEFAULT,
        known_stages=frozenset(CANONICAL_STAGE_ORDER),
        bypass_manual_review=spec.bypass_manual_review,
    )


def _outcome_from_result(
    spec: StageSpec, result: AgentResult, status: str, *, attempts_used: int
) -> StageOutcome:
    validated = _validated_transition_for(
        spec,
        requested_transition=result.requested_transition,
        quality_status=result.quality_status,
        attempts_used=attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=result.execution_status.value,
        quality_status=result.quality_status.value,
        warnings=tuple(w.message for w in result.warnings),
        error=result.error,
        attempt_number=result.attempt_number,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


def _outcome_from_committed_stage(spec: StageSpec, committed, *, status: str) -> StageOutcome:
    """Construye un StageOutcome a partir de un StageState ya comprometido
    (usado para SKIPPED_FRESH), reutilizando su ``requested_transition``
    histórica en vez de asumir ADVANCE por defecto."""

    requested = committed.requested_transition or RequestedTransition(
        action=TransitionAction.ADVANCE, reason_code="ASSUMED_ADVANCE_NO_HISTORY"
    )
    validated = _validated_transition_for(
        spec,
        requested_transition=requested,
        quality_status=committed.quality_status,
        attempts_used=committed.attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=committed.execution_status.value,
        quality_status=(
            committed.quality_status.value if committed.quality_status else None
        ),
        warnings=tuple(w.get("message", "") for w in committed.warnings),
        error=committed.last_error,
        attempt_number=committed.attempts_used,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


def run_stage(
    *,
    store: StateStore,
    project_dir: str | Path,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    """Ejecuta (o resuelve/salta) una única etapa y devuelve su resultado.

    Si ``spec.custom_run`` está definido (caso de 07), delega por completo en
    él y el resto de esta función no se ejecuta — ver el comentario en el
    campo ``custom_run`` de ``StageSpec`` para el porqué.

    Para las demás etapas, orden de decisiones:
    1. Si hay una ``pending_execution`` de esta etapa (interrupción previa),
       se resuelve primero (COMMIT del resultado ya persistido, o liberación
       para reejecutar) — igual que antes.
    2. Si la etapa ya quedó COMPLETED, y ``spec.build_fingerprints`` existe,
       se reconstruye su AgentInput actual y se comparan sus fingerprints
       contra los comprometidos: si coinciden, se salta (SKIPPED_FRESH); si
       no, se considera obsoleta y se reejecuta aunque no se haya pedido
       force_rerun explícitamente. Si ``spec.build_fingerprints`` es None
       (ninguna etapa hoy), se usa el chequeo antiguo (solo COMPLETED).
    3. En cualquier otro caso, se ejecuta la transacción real de la etapa.

    Cualquier excepción no capturada por la propia etapa (solo puede ocurrir
    hoy en 07, que no envuelve fallos de preparación en un AgentResult — ver
    ``_run_verification_stage``) se convierte aquí en un StageOutcome
    ``status="FAILED"``/``next_action="HALT_STAGE"`` para que run_pipeline
    nunca termine con una excepción sin registrar, en vez de dejarla
    propagar sin control.
    """

    project_dir = Path(project_dir)
    if spec.max_attempt_number is not None and attempt_number > spec.max_attempt_number:
        raise ValueError(
            f"{spec.key} admite como máximo attempt_number={spec.max_attempt_number}."
        )

    try:
        if spec.custom_run is not None:
            return spec.custom_run(
                store=store,
                project_dir=project_dir,
                spec=spec,
                attempt_number=attempt_number,
                observations=observations,
                force_rerun=force_rerun,
            )

        def build_execution() -> tuple[Any, Any]:
            return spec.build_execution(project_dir, attempt_number)

        state = store.load()

        if (
            state.pending_execution is not None
            and state.pending_execution.target_stage == spec.key
        ):
            _agent, agent_input = build_execution()
            resume = spec.resolve_resume(
                store=store, agent_input=agent_input, observations=observations
            )
            if resume.action == "COMMITTED":
                state = store.load()
                attempts_used = state.stages[spec.key].attempts_used
                return _outcome_from_result(
                    spec, resume.committed_result, "COMMITTED", attempts_used=attempts_used
                )
            # REEXECUTE o NO_PENDING: el pending quedó liberado; se sigue abajo.
            state = store.load()

        committed = state.stages.get(spec.key)
        if (
            committed is not None
            and committed.execution_status == ExecutionStatus.COMPLETED
            and not force_rerun
        ):
            if spec.build_fingerprints is not None:
                _agent, agent_input = build_execution()
                current_fingerprints = spec.build_fingerprints(agent_input)
                if is_stage_fresh(committed, current_fingerprints):
                    return _outcome_from_committed_stage(
                        spec, committed, status="SKIPPED_FRESH"
                    )
                # Fingerprints obsoletos: no se salta aunque no se haya pedido
                # force_rerun explícito.
                observations = dict(observations or {})
                observations["orchestrator_note"] = (
                    "stage_stale_fingerprint_mismatch_reexecuting"
                )
            else:
                # Sin build_fingerprints disponible: se conserva el chequeo
                # antiguo (solo COMPLETED, sin comparar vigencia).
                return _outcome_from_committed_stage(
                    spec, committed, status="SKIPPED_FRESH"
                )

        transaction = spec.runtime_transaction(
            store=store,
            build_execution=build_execution,
            attempt_number=attempt_number,
            observations=observations,
        )
        status = (
            "COMMITTED"
            if transaction.agent_result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
        state = store.load()
        attempts_used = state.stages[spec.key].attempts_used
        return _outcome_from_result(
            spec, transaction.agent_result, status, attempts_used=attempts_used
        )
    except Exception as exc:  # noqa: BLE001 - red de seguridad genérica, ver docstring
        return StageOutcome(
            key=spec.key,
            label=spec.label,
            status="FAILED",
            execution_status=None,
            quality_status=None,
            warnings=(),
            error={"type": type(exc).__name__, "message": str(exc)},
            attempt_number=attempt_number,
            next_action="HALT_STAGE",
            target_stage=None,
            reason_code=f"UNCAUGHT_EXCEPTION:{type(exc).__name__}",
        )


def _reconcile_pending_execution_for_other_stage(
    *,
    store,
    project_dir: Path,
    registry: Mapping[str, "StageSpec"],
    current_stage: str,
    attempt_numbers: Mapping[str, int],
    observations: Mapping[str, Any] | None,
) -> tuple[list["StageOutcome"], bool]:
    """Si ``state.pending_execution`` existe y apunta a una etapa DISTINTA
    de ``current_stage``, la reconcilia vía el protocolo OFICIAL de esa
    otra etapa (``run_stage`` sobre su propio ``StageSpec``) antes de que
    se intente preparar ``current_stage``.

    Motivo: ``StateStore.prepare_execution`` mantiene un único slot GLOBAL
    de ``pending_execution`` (no uno por etapa) -- una ejecución
    interrumpida de OTRA etapa (ej. 07 crasheando antes de comprometer)
    deja ese slot ocupado y bloquea la preparación de CUALQUIER otra
    etapa, incluida la que el pipeline está intentando ahora, con
    ``RuntimeError("a pending execution already exists")``. Esta función
    nunca lee ni escribe ``pending_execution`` directamente -- delega
    por completo en ``run_stage()`` para la etapa a la que realmente
    pertenece, que ya sabe resolverla oficialmente (COMMIT del resultado
    persistido, liberar para reejecutar, o lo que corresponda según su
    propio protocolo -- para 07 esto enruta a
    ``_run_verification_stage``/``resume_agent07_execution``).

    Devuelve ``(outcomes_a_agregar, debe_detenerse)``: si
    ``debe_detenerse`` es ``True``, el llamador no debe intentar preparar
    ``current_stage`` en esta vuelta (o bien la pending sigue sin
    resolverse tras el intento oficial, o bien apunta a una etapa sin
    ``StageSpec`` registrado -- inconsistencia real que se reporta
    explícitamente, nunca se oculta ni se fuerza)."""

    state = store.load()
    pending = state.pending_execution
    if pending is None or pending.target_stage == current_stage:
        return [], False

    pending_stage_key = pending.target_stage
    if pending_stage_key not in registry:
        return (
            [
                StageOutcome(
                    key=current_stage,
                    label=registry[current_stage].label if current_stage in registry else current_stage,
                    status="FAILED",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error={
                        "type": "PendingExecutionUnknownTargetStage",
                        "message": (
                            f"pending_execution.target_stage={pending_stage_key!r} "
                            "no está en el registro de etapas -- no se puede "
                            "reconciliar automáticamente."
                        ),
                    },
                    attempt_number=0,
                    next_action="HALT_STAGE",
                    target_stage=None,
                    reason_code="PENDING_EXECUTION_UNKNOWN_TARGET_STAGE",
                )
            ],
            True,
        )

    pending_spec = registry[pending_stage_key]
    reconcile_outcome = run_stage(
        store=store,
        project_dir=project_dir,
        spec=pending_spec,
        attempt_number=attempt_numbers.get(pending_stage_key, 1),
        observations=observations,
        force_rerun=False,
    )

    state = store.load()
    if state.pending_execution is not None:
        # El protocolo oficial de esa etapa no logró liberar la pending
        # (ej. sigue EXECUTED_NOT_COMMITTED esperando otra vuelta) -- no
        # se fuerza nada más. El llamador se detiene aquí en vez de
        # intentar preparar current_stage, que el store rechazaría de
        # nuevo con el mismo error.
        return [reconcile_outcome], True

    return [reconcile_outcome], False


def _apply_stage_transition(
    outcome: "StageOutcome",
    *,
    store,
    stage_key: str,
    attempt_number: int,
    attempt_numbers: dict[str, int],
    until: str | None,
    outcomes: list["StageOutcome"],
) -> tuple[str | None, bool]:
    """Interpreta ``outcome.next_action`` con la MISMA semántica que
    gobierna el bucle principal de ``run_pipeline`` (ADVANCE con
    resolución de ciclo, RETRY, RETURN con ``apply_return_with_cycle`` y
    posible agotamiento del ciclo, o HALT_STAGE/STOP_PIPELINE) --
    factorizada para poder aplicarse tanto a la etapa que el bucle está
    procesando en su vuelta normal como a una etapa reconciliada fuera de
    orden (ver ``_reconcile_pending_execution_for_other_stage``): antes
    de esta función, la transición de una etapa reconciliada (ej. 07,
    resuelta porque tenía una ``pending_execution`` vieja) se ignoraba
    por completo -- el bucle seguía su recorrido normal desde
    ``current_stage`` sin importar si la reconciliación había producido
    HALT_STAGE, RETURN o ADVANCE, lo que permitía llegar a intentar una
    etapa posterior (06) que ya no correspondía tocar.

    Devuelve ``(nuevo_current_stage_o_None, debe_detenerse)`` -- si
    ``debe_detenerse`` es ``True``, el llamador debe terminar el bucle
    (pipeline completo, HALT_STAGE/STOP_PIPELINE, o ciclo agotado -- en
    este último caso ya se agregó el ``StageOutcome`` de
    ``CYCLE_EXHAUSTED`` a ``outcomes`` antes de devolver)."""

    if until is not None and stage_key == until:
        return None, True

    if outcome.next_action == "ADVANCE":
        if outcome.target_stage is None:
            return None, True  # pipeline completo
        if stage_key == de.WRITER_VERIFIER_TRIGGER_STAGE:
            resolve_cycle_if_active(store)
        return outcome.target_stage, False

    if outcome.next_action == "RETRY":
        attempt_numbers[stage_key] = attempt_number + 1
        return stage_key, False

    if outcome.next_action == "RETURN":
        cycle_result = apply_return_with_cycle(
            store,
            from_stage=stage_key,
            target_stage=outcome.target_stage,
            reason=f"INVALIDATED_BY_RETURN_FROM_{stage_key}",
        )
        if cycle_result.cycle_exhausted:
            outcomes.append(
                StageOutcome(
                    key=stage_key,
                    label=f"(ciclo {de.WRITER_VERIFIER_CYCLE_NAME} agotado)",
                    status="CYCLE_EXHAUSTED",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error=None,
                    attempt_number=attempt_number,
                    next_action="HALT_STAGE",
                    target_stage=None,
                    reason_code="WRITER_VERIFIER_CYCLE_EXHAUSTED",
                )
            )
            return None, True
        for stage_key_to_clear in CANONICAL_STAGE_ORDER[
            CANONICAL_STAGE_ORDER.index(outcome.target_stage) :
        ]:
            attempt_numbers.pop(stage_key_to_clear, None)
        return outcome.target_stage, False

    # HALT_STAGE o STOP_PIPELINE: se detiene el bucle.
    return None, True


from src.orchestration.decision_log_frontier import (
    _causally_connects,
    _segment_decision_log,
    _reconstruct_authoritative_frontier,
)


def _check_already_terminal_state(
    *, store, registry: Mapping[str, "StageSpec"], start_stage: str | None, force_rerun: bool
) -> "StageOutcome | None":
    """Si la decisión AUTORITATIVA del ``decision_log`` (ver
    ``_reconstruct_authoritative_frontier`` -- ni la última entrada
    cronológica ni asumir un único tramo desde el principio) pidió
    explícitamente ``HALT_STAGE`` o ``STOP_PIPELINE``, el pipeline ya
    está en un estado TERMINAL -- un restart sin ``start_stage``
    explícito ni ``--force-rerun`` no debe recorrer las etapas de nuevo
    asumiendo que hay trabajo pendiente. Devuelve el ``StageOutcome``
    terminal a reportar tal cual (sin tocar ningún estado), o ``None``
    si no aplica.

    El ``StageOutcome`` se construye SIEMPRE a partir del propio
    ``frontier_entry.result`` (el ``AgentResult`` persistido en esa
    entrada exacta del log, vía ``AgentResult.from_dict`` +
    ``_outcome_from_result`` -- la misma función que ya usa el resto del
    módulo para construir un ``StageOutcome`` desde un ``AgentResult``
    real) -- nunca desde ``state.stages[stage]`` (el estado COMPROMETIDO
    VIGENTE de esa etapa), que puede corresponder a una ejecución
    POSTERIOR y distinta de la entrada histórica que este chequeo
    determinó como terminal. Mezclar ambas fuentes es exactamente lo
    que producía ``ALREADY_TERMINAL`` con un ``next_action=ADVANCE`` --
    una contradicción de contrato que nunca debe poder ocurrir: se
    afirma explícitamente como invariante antes de devolver."""

    if start_stage is not None or force_rerun:
        return None

    state = store.load()
    frontier_entry = _reconstruct_authoritative_frontier(state.decision_log)
    if frontier_entry is None:
        return None

    frontier_transition = frontier_entry.requested_transition
    if (
        frontier_transition is None
        or frontier_transition.action not in (TransitionAction.HALT_STAGE, TransitionAction.STOP_PIPELINE)
        or frontier_entry.stage not in registry
    ):
        return None

    frontier_result = AgentResult.from_dict(frontier_entry.result)
    outcome = _outcome_from_result(
        registry[frontier_entry.stage], frontier_result, "ALREADY_TERMINAL", attempts_used=frontier_entry.attempt
    )

    # Invariante obligatoria: ALREADY_TERMINAL nunca puede coexistir con
    # una transición no terminal. Si por cualquier motivo no se cumple
    # (no debería, dado el chequeo de frontier_transition.action arriba,
    # pero se verifica explícitamente en vez de confiar en eso
    # implícitamente), no se afirma un estado terminal que el propio
    # outcome contradice -- se deja que el flujo normal decida.
    if outcome.next_action not in ("HALT_STAGE", "STOP_PIPELINE"):
        return None

    return outcome


def run_pipeline(
    project_dir: str | Path,
    *,
    start_stage: str | None = None,
    until: str | None = None,
    attempt_numbers: Mapping[str, int] | None = None,
    force_rerun: bool = False,
    max_iterations: int = 50,
    observations: Mapping[str, Any] | None = None,
) -> list[StageOutcome]:
    """Corre el pipeline interpretando las transiciones solicitadas por cada etapa.

    A diferencia de la versión anterior (un ``for`` fijo sobre ``STAGE_ORDER``
    que se detenía en el primer FAILED), esto ahora es un bucle guiado por
    ``RequestedTransition`` validado con ``decision_engine.validate_transition``:

    - ``ADVANCE`` → sigue a la etapa objetivo (por defecto la siguiente).
    - ``RETRY`` → reintenta la misma etapa (respetando el límite de intentos).
    - ``RETURN`` → invalida la etapa objetivo y todas las posteriores
      (``decision_engine.invalidate_from``) y continúa desde ahí.
    - ``HALT_STAGE`` / ``STOP_PIPELINE`` → detiene el bucle.

    ``until``: si se da, se detiene apenas la etapa con esa clave produce un
    resultado (antes de avanzar a la siguiente), incluso si el resultado
    pedía ADVANCE.

    ``force_rerun`` sólo se aplica a ``start_stage`` (o a la primera etapa si
    no se indica); las etapas alcanzadas después por ADVANCE/RETRY/RETURN se
    evalúan normalmente (con su propio chequeo de fingerprints).
    """

    attempt_numbers = dict(attempt_numbers or {})
    store = ensure_pipeline_state(project_dir)
    registry = {spec.key: spec for spec in _stage_registry()}

    if until is not None and until not in STAGE_ORDER:
        raise ValueError(f"Etapa desconocida en 'until': {until}")

    current_stage = start_stage or STAGE_ORDER[0]
    outcomes: list[StageOutcome] = []
    force_rerun_current = force_rerun

    terminal_outcome = _check_already_terminal_state(
        store=store, registry=registry, start_stage=start_stage, force_rerun=force_rerun
    )
    if terminal_outcome is not None:
        _print_outcome(terminal_outcome)
        return [terminal_outcome]

    for _ in range(max_iterations):
        if current_stage not in registry:
            outcomes.append(
                StageOutcome(
                    key=current_stage,
                    label=f"(sin StageSpec ejecutable todavía: {current_stage})",
                    status="REACHED_UNREGISTERED_STAGE",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error=None,
                    attempt_number=0,
                    next_action="STOP_PIPELINE",
                    target_stage=None,
                    reason_code="STAGE_NOT_REGISTERED",
                )
            )
            break

        spec = registry[current_stage]

        reconcile_outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=project_dir, registry=registry, current_stage=current_stage,
            attempt_numbers=attempt_numbers, observations=observations,
        )
        if reconcile_outcomes:
            reconcile_outcome = reconcile_outcomes[0]
            outcomes.append(reconcile_outcome)
            _print_outcome(reconcile_outcome)
            if must_stop:
                # Inconsistencia real (etapa de la pending sin StageSpec
                # registrado) o la pending sigue sin resolverse tras el
                # intento oficial -- no hay transición válida que
                # despachar; se detiene aquí, igual que antes.
                break
            # La transición REAL de la etapa reconciliada (HALT_STAGE,
            # RETURN o ADVANCE) gobierna el flujo principal a partir de
            # aquí -- nunca se ignora para seguir el recorrido normal
            # desde current_stage.
            reconciled_stage_key = reconcile_outcome.key
            reconciled_attempt_number = attempt_numbers.get(reconciled_stage_key, 1)
            new_stage, should_stop = _apply_stage_transition(
                reconcile_outcome, store=store, stage_key=reconciled_stage_key,
                attempt_number=reconciled_attempt_number, attempt_numbers=attempt_numbers,
                until=until, outcomes=outcomes,
            )
            if should_stop:
                break
            current_stage = new_stage
            continue

        attempt_number = attempt_numbers.get(current_stage, 1)
        outcome = run_stage(
            store=store,
            project_dir=project_dir,
            spec=spec,
            attempt_number=attempt_number,
            observations=observations,
            force_rerun=force_rerun_current,
        )
        force_rerun_current = False
        outcomes.append(outcome)
        _print_outcome(outcome)

        new_stage, should_stop = _apply_stage_transition(
            outcome, store=store, stage_key=current_stage, attempt_number=attempt_number,
            attempt_numbers=attempt_numbers, until=until, outcomes=outcomes,
        )
        if should_stop:
            break
        current_stage = new_stage
    else:
        raise RuntimeError(
            "run_pipeline alcanzó max_iterations sin converger a un estado "
            "terminal; posible ciclo ADVANCE/RETURN entre etapas."
        )

    return outcomes


def _print_outcome(outcome: StageOutcome) -> None:
    print(
        f"[{outcome.status:24s}] {outcome.label:45s} "
        f"execution={outcome.execution_status} quality={outcome.quality_status} "
        f"next={outcome.next_action}->{outcome.target_stage}"
    )
    for warning in outcome.warnings:
        print(f"    warning: {warning}")
    if outcome.error:
        print(f"    error: {outcome.error}")


# ---------------------------------------------------------------------------
# CLI para uso directo en Colab: `python -m src.orchestration.pipeline_orchestrator`
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True, help="Ruta a PROJECT_DIR (contiene active_experiment.json)."
    )
    parser.add_argument(
        "--until",
        default=None,
        choices=STAGE_ORDER,
        help="Detenerse tras completar esta etapa (por defecto corre hasta 06).",
    )
    parser.add_argument(
        "--start-stage",
        default=None,
        choices=STAGE_ORDER,
        help=(
            "Empezar el recorrido directamente en esta etapa, en vez de "
            "STAGE_ORDER[0] -- ejecuta ÚNICAMENTE esta etapa y las que "
            "resulten de sus transiciones reales (nunca las anteriores). "
            "Con start-stage explícito, el chequeo de estado ya-terminal "
            "se omite deliberadamente (se respeta la petición explícita "
            "del llamador, igual que --force-rerun) -- si la etapa ya "
            "está COMPLETED y vigente (fingerprints sin cambios), sigue "
            "reconociéndose SKIPPED_FRESH con normalidad; si su último "
            "commit fue FAILED (ej. HALT_STAGE), esto la reintenta con "
            "un decision_id nuevo, SIN --force-rerun y sin tocar ninguna "
            "etapa previa."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Reejecuta la etapa inicial aunque ya esté COMPLETED y vigente en pipeline_state.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outcomes = run_pipeline(
        args.project_dir,
        start_stage=args.start_stage,
        until=args.until,
        force_rerun=args.force_rerun,
    )
    return 0 if all(o.status not in {"FAILED", "REACHED_UNREGISTERED_STAGE"} for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
