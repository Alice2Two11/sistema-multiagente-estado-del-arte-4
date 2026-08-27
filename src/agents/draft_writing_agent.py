from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.contracts.agent_input import ArtifactReference
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
from src.state.fingerprints import sha256_file
from src.tools.draft_writing.artifacts import (
    NAMES,
    write_draft_artifacts,
    write_partial_validation,
)
from src.tools.draft_writing.input_validation import validate_draft_dependencies
from src.tools.draft_writing.prompting import (
    assign_section_budgets,
    build_source_free_organizational_section,
)
from src.tools.draft_writing.retrieval import (
    build_section_query,
    retrieve_section_evidence,
)
from src.tools.draft_writing.validation import (
    build_draft_reports,
    validate_draft_global,
    section_allows_no_sources,
)
from src.tools.draft_writing.length_repair import attempt_directed_length_repair


LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted"

# canonical_sentences_v2 es la única representación productiva,
# confirmada empíricamente en las 10 corridas experimentales reales
# (draft_generation_manifest.json de cada una). draft_representation_
# contract ausente o con cualquier otro valor (incluido "legacy"
# explícito) falla fail-closed -- ver validación más abajo. El bloque
# de generación bajo contrato legacy y HYBRID_RETRIEVAL_STRATEGY (única
# ruta alterna que existió) fueron eliminados por completo -- auditoría
# STAGE06-DEAD-CONTRACT-LEGACY, cero consumidores confirmados.
CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT = "canonical_sentences_v2"

# Reason codes de longitud (Stage 06, corrección del gate configured_
# min_total_words/configured_max_total_words) -- nunca genéricos bajo
# INVALID_DRAFT, para que el motivo real (déficit, exceso, o evidencia
# insuficiente tras intentar reparar) sea auditable explícitamente.
TOTAL_WORD_COUNT_BELOW_MINIMUM = "TOTAL_WORD_COUNT_BELOW_MINIMUM"
TOTAL_WORD_COUNT_ABOVE_MAXIMUM = "TOTAL_WORD_COUNT_ABOVE_MAXIMUM"
INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH = "INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH"

LEGACY_VERSIONS = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    # v2 (sufijo "_hard_word_range_configured_min_gate_soft_failure_
    # length_repair"): el gate de longitud total del borrador cambió de
    # contrato -- antes, global_length_valid podía aprobarse usando
    # effective_min_total_words (rebajado silenciosamente por número de
    # secciones source-free), permitiendo que un borrador muy por
    # debajo de min_total_words configurado (ej. 1081 con
    # configured_min=1300) se aprobara. Desde esta versión:
    # configured_min_total_words/configured_max_total_words (los
    # valores reales del generation_profile) son el único gate;
    # incumplir el rango produce reason codes explícitos
    # (TOTAL_WORD_COUNT_BELOW_MINIMUM/ABOVE_MAXIMUM/INSUFFICIENT_
    # SUPPORTED_CONTENT_FOR_MIN_LENGTH) en vez de INVALID_DRAFT
    # genérico, nunca una excepción técnica, y se intenta una
    # reparación dirigida (src/tools/draft_writing/length_repair.py,
    # exclusivamente dentro del contrato Evidence Handles V2) antes de
    # agotar los intentos. Este es un cambio de CONTRATO real -- por
    # eso participa en el fingerprint (ver _draft_signature) e invalida
    # cualquier draft/manifest de 06 producido bajo una versión
    # anterior, sin necesitar --force-rerun.
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    # LEGACY_VERSIONS es la fuente canónica única -- src/adapters/
    # draft_writing_runtime.py ya no define su propia copia; importa
    # este mismo dict con alias (LEGACY_RUNTIME_VERSIONS). Antes eran
    # dos copias idénticas mantenidas a mano en paralelo.
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}

class DraftWritingAgent:
    """Contractual Agent 06. Única ruta productiva: retrieval_strategy=
    legacy_chroma_then_csv_restricted + draft_representation_contract=
    canonical_sentences_v2 (confirmado empíricamente en las 10 corridas
    experimentales reales). La rama V17 híbrida fue removida por no
    tener consumidores en runtime real (ver auditoría previa)."""

    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def _section_sources(section: Mapping[str, Any]) -> list[str]:
        sources: list[str] = []
        for paper in section.get("papers_to_use") or []:
            if not isinstance(paper, Mapping):
                continue
            source = str(paper.get("source_filename", "")).strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _valid_source_chunk_pairs(chunks: pd.DataFrame) -> set[tuple[str, str]]:
        if chunks.empty or not {"source_filename", "chunk_id"}.issubset(chunks.columns):
            return set()
        return {
            (str(row["source_filename"]).strip(), str(row["chunk_id"]).strip())
            for _, row in chunks.iterrows()
            if str(row["source_filename"]).strip() and str(row["chunk_id"]).strip()
        }

    def _quant_context(
        self,
        section: Mapping[str, Any],
        bundle: Mapping[str, Any],
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        sources = set(self._section_sources(section))
        quantitative = bundle["quantitative"]
        dataset_summary = bundle["dataset_summary"]
        quantitative_rows = (
            quantitative[
                quantitative["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not quantitative.empty and "source_filename" in quantitative.columns
            else []
        )
        dataset_rows = (
            dataset_summary[
                dataset_summary["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not dataset_summary.empty and "source_filename" in dataset_summary.columns
            else []
        )
        return {
            "quantitative_results": quantitative_rows,
            "dataset_technique_summary": dataset_rows,
        }

    @staticmethod
    def _strategy(policy: Mapping[str, Any]) -> str:
        # Fail-closed: única ruta productiva confirmada empíricamente en
        # las 10 corridas experimentales reales (draft_generation_
        # manifest.json / state_of_art_draft.json de cada una). Ausencia
        # de la clave o cualquier otro valor (incluida la estrategia
        # híbrida V17, nunca ejercitada en runtime real) es un error
        # explícito, nunca un fallback silencioso.
        strategy = policy.get("retrieval_strategy")
        if strategy != LEGACY_RETRIEVAL_STRATEGY:
            raise ValueError(
                f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy!r}"
            )
        return strategy

    @staticmethod
    def _effective_versions(
        policy: Mapping[str, Any], strategy: str
    ) -> dict[str, str]:
        del policy, strategy
        return dict(LEGACY_VERSIONS)

    @staticmethod
    def _section_budgets(
        sections: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        strategy: str,
    ) -> dict[str, dict[str, Any]]:
        del strategy
        target_total_words = int(policy["target_total_words"])
        return assign_section_budgets(sections, target_total_words)

    def _retrieve_section_evidence(
        self,
        section: Mapping[str, Any],
        bundle: Mapping[str, Any],
        policy: Mapping[str, Any],
        strategy: str,
        quantitative_context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        chunks = bundle["chunks"]
        # CONFIG-E (Stage 06): top_k_evidence_per_section/max_evidence_
        # chars/max_quantitative_rows_per_section son responsabilidad de
        # 00 y ya son obligatorios en get_draft_writing_policy -- sin
        # fallback downstream residual.
        del strategy, quantitative_context
        top_k = int(policy["top_k_evidence_per_section"])
        max_chars = int(policy["max_evidence_chars"])
        return retrieve_section_evidence(
            section,
            self.runtime.collection,
            chunks,
            top_k,
            max_chars,
        )

    @staticmethod
    def _build_v2_section_validation_failed_result(
        *,
        agent_input,
        sid: str,
        out: Path,
        raw_dir: Path,
        attempt_logs: dict[str, list[dict[str, Any]]],
        retrieval_rounds: int,
        llm_calls: int,
        validation_calls: int,
        last_errors: list[str],
        validation_version: Any,
        start: str,
    ) -> AgentResult:
        """Función auxiliar NUEVA (extraída, no movida) que replica
        EXACTAMENTE el mismo contrato externo que el bloque legacy
        ``if accepted is None:`` (más abajo en este archivo, NUNCA
        tocado por esta función) ya produce cuando una sección agota
        sus intentos: ``execution_status=COMPLETED``,
        ``quality_status=NEEDS_REVISION``, ``RETRY`` en el primer
        intento externo, ``HALT_STAGE`` en los siguientes -- con
        artefactos parciales y los códigos de error REALES del último
        intento preservados.

        Campos comunes con el reporte legacy (``validation_version``,
        ``section_attempts``, en ambos lugares donde legacy los
        escribe -- ``draft_validation_report.json`` y ``quality_
        metrics["technical"]["section_attempts"]``) -- ``validation_
        version`` se pasa como VALOR explícito, no la policy completa,
        para no acoplar este helper a la forma de policy que usa el
        camino legacy.

        Se llama únicamente desde el branch V2 (canonical_sentences_v2
        agotó sus reintentos). Nunca desde el camino legacy -- ese
        bloque sigue construyendo su propio AgentResult inline, sin
        cambios, para no arriesgar el aislamiento ya garantizado
        (LEGACY 11/11)."""

        section_attempts = len(attempt_logs.get(sid) or [])
        partial_validation = {
            "stage": "06_agente_redactor",
            "experiment_id": agent_input.experiment_id,
            "validation_version": validation_version,
            "validation_ok": False,
            "failed_section": sid,
            "section_attempts": section_attempts,
            "contract": "canonical_sentences_v2",
            "last_attempt_errors": list(last_errors),
            "generation_attempts": attempt_logs,
            "current_raw_attempt_directory": str(raw_dir),
            "published_draft": False,
        }
        report_path = write_partial_validation(out, partial_validation)
        artifacts = {
            "draft_validation_report.json": ArtifactReference(
                str(report_path), sha256_file(report_path)
            ),
            "raw_section_outputs": ArtifactReference(
                str(out / "raw_section_outputs"), "DIRECTORY"
            ),
        }
        action = (
            TransitionAction.RETRY
            if agent_input.is_first_attempt()
            else TransitionAction.HALT_STAGE
        )
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED,
            quality_status=QualityStatus.NEEDS_REVISION,
            decision=DecisionInfo(
                code="SECTION_VALIDATION_FAILED",
                rationale=(
                    f"La sección {sid} (canonical_sentences_v2) agotó sus "
                    "reintentos; se preservaron salidas y validaciones por "
                    "intento."
                ),
            ),
            quality_metrics={
                "scientific": {},
                "technical": {
                    "validation_ok": False,
                    "reused": False,
                    "failed_section": sid,
                    "section_attempts": section_attempts,
                    "contract": "canonical_sentences_v2",
                },
            },
            warnings=(
                AgentWarning(
                    code="SECTION_VALIDATION_FAILED",
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=(
                        f"La sección {sid} (canonical_sentences_v2) no "
                        "superó la validación tras agotar sus intentos."
                    ),
                ),
            ),
            failure_reason_codes=("SECTION_VALIDATION_FAILED",),
            requested_transition=RequestedTransition(
                action=action,
                target_stage=None,
                reason_code="NEEDS_REVISION",
                requires_human_confirmation=False,
            ),
            output_artifacts=artifacts,
            tool_usage=ToolUsage(
                retrieval_rounds=retrieval_rounds,
                llm_calls=llm_calls,
                validation_calls=validation_calls,
            ),
            attempt_number=agent_input.attempt_number,
            started_at=start,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0
        retrieval_rounds = 0
        validation_calls = 0
        out = Path(agent_input.agent_context.output_directory)
        # Versionado por intento EXTERNO del orquestador (agent_input.
        # attempt_number) -- distinto del contador interno de reintentos
        # por sección (generation_attempt, usado en el nombre de cada
        # archivo dentro de este subdirectorio). Sin esto, una segunda
        # ejecución externa de 06 (tras RETRY/HALT_STAGE) sobrescribía en
        # silencio los .txt/_validation.json/_rag_trace.json del intento
        # externo anterior, en el MISMO directorio -- perdiendo toda
        # trazabilidad histórica de por qué falló un intento previo.
        #
        # CONTRATO EXPLÍCITO del artefacto (no cambia silenciosamente):
        #   - ArtifactReference["raw_section_outputs"] SIEMPRE apunta al
        #     directorio PADRE (out / "raw_section_outputs") -- el
        #     histórico COMPLETO de todos los intentos externos
        #     preservados (agent_attempt_01/, agent_attempt_02/, ...).
        #     Nunca a un subdirectorio de un intento en particular.
        #   - raw_dir (esta variable) es el subdirectorio de ESTE intento
        #     externo únicamente -- se usa para ESCRIBIR los archivos de
        #     esta ejecución, nunca se registra como el ArtifactReference
        #     en sí. Cuando el reporte/manifest necesita señalar dónde
        #     quedaron los archivos de ESTE intento específico, usa la
        #     clave "current_raw_attempt_directory" (metadata, no un
        #     ArtifactReference nuevo).
        # Nunca leída directamente por 07/08 ni por ningún otro consumidor
        # del pipeline (confirmado: ningún módulo fuera de este archivo y
        # artifacts.py referencia raw_section_outputs).
        raw_dir = out / "raw_section_outputs" / f"agent_attempt_{agent_input.attempt_number:02d}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            bundle = validate_draft_dependencies(agent_input)
            policy = dict(agent_input.policy)
            strategy = self._strategy(policy)
            versions = self._effective_versions(policy, strategy)
            policy.update(versions)
            manifest_path = out / "draft_generation_manifest.json"
            reuse = False
            required_reuse = (
                "state_of_art_draft.json",
                "state_of_art_draft.md",
                "draft_sections.csv",
                "draft_rag_evidence.csv",
                "draft_quality_check.csv",
                "draft_length_check.csv",
                "draft_claim_evidence.csv",
                "numeric_hallucination_check.csv",
                "draft_validation_report.json",
                "draft_generation_manifest.json",
            )
            if manifest_path.exists() and not policy["force_rebuild"]:
                try:
                    old = json.loads(manifest_path.read_text())
                    report = json.loads((out / "draft_validation_report.json").read_text())
                    reuse = (
                        old.get("fingerprint") == policy.get("current_fingerprint")
                        and report.get("validation_ok") is True
                        and all((out / name).exists() for name in required_reuse)
                    )
                except Exception:
                    reuse = False

            if reuse:
                artifacts = {
                    name: ArtifactReference(str(out / name), sha256_file(out / name))
                    for name in NAMES
                    if (out / name).exists()
                }
                # Contrato explícito: raw_section_outputs SIEMPRE apunta
                # al directorio PADRE (out / "raw_section_outputs"), que
                # representa el histórico COMPLETO de todos los intentos
                # externos preservados (agent_attempt_01/, agent_attempt_
                # 02/, ...) -- nunca a un subdirectorio de un intento en
                # particular. Ver docstring de execute() para el
                # contrato completo.
                artifacts["raw_section_outputs"] = ArtifactReference(
                    str(out / "raw_section_outputs"), "DIRECTORY"
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.APPROVED,
                    decision=DecisionInfo(
                        code="DRAFT_REUSED",
                        rationale="Borrador válido reutilizado con fingerprint vigente.",
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": True, "reused": True},
                    },
                    warnings=(),
                    failure_reason_codes=(),
                    requested_transition=RequestedTransition(
                    action=TransitionAction.ADVANCE,
                    target_stage="07_agente_verificador",
                    reason_code="APPROVED",
                    requires_human_confirmation=False,
                ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=0, llm_calls=0, validation_calls=0
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            sections = bundle["outline"].get("sections") or []
            if not isinstance(sections, list) or not sections:
                raise ValueError("INVALID_OUTLINE_SCHEMA")
            policy["outline_sections"] = sections
            policy["section_budgets"] = self._section_budgets(
                sections, policy, strategy
            )

            # Validación fail-closed del contrato de representación de
            # secciones: una sola vez, ANTES de procesar cualquier
            # sección -- incluidas las organizativas/source-free, que
            # antes podían generarse (generated.append(...); continue)
            # sin haber pasado por esta validación. Única ruta productiva
            # confirmada empíricamente en las 10 corridas reales: ausencia
            # de la clave o cualquier valor distinto de
            # canonical_sentences_v2 (incluido "legacy" explícito) falla
            # antes de tocar la primera sección, sin fallback silencioso.
            contract = policy.get("draft_representation_contract")
            if contract != CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                raise ValueError(
                    f"UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:{contract!r}"
                )

            generated: list[dict[str, Any]] = []
            all_evidence: list[dict[str, Any]] = []
            attempt_logs: dict[str, list[dict[str, Any]]] = {}

            for section in sections:
                sid = str(section.get("section_id", "")).strip()
                section_query = build_section_query(section)
                quant_context = self._quant_context(
                    section,
                    bundle,
                    int(policy["max_quantitative_rows_per_section"]),
                )
                evidence = self._retrieve_section_evidence(
                    section,
                    bundle,
                    policy,
                    strategy,
                    quant_context,
                )
                if section.get("papers_to_use"):
                    retrieval_rounds += 1
                all_evidence.extend({"section_id": sid, **row} for row in evidence)

                if not evidence:
                    if not section_allows_no_sources(section):
                        raise ValueError(f"MISSING_SECTION_EVIDENCE:{sid}")
                    generated_section = build_source_free_organizational_section(
                        section, policy.get("output_language", "español")
                    )
                    attempt_logs[sid] = [
                        {
                            "attempt": 0,
                            "mode": "deterministic_source_free_organizational_section",
                            "validation": generated_section["section_validation"],
                        }
                    ]
                    generated.append(generated_section)
                    continue

                # Bifurcación mínima, un solo punto: el contrato ya fue
                # validado UNA VEZ antes del bucle (ver arriba) --
                # aquí solo se lee la variable ya calculada, nunca se
                # revalida por sección. El código legacy de abajo sigue
                # EXACTAMENTE igual, sin ninguna línea movida. Solo
                # cuando la policy selecciona V2 explícitamente se
                # invoca el módulo nuevo (import local: sin efectos
                # secundarios al cargar draft_writing_agent.py, y sin
                # ejecutar ninguna línea de canonical_sentences.py
                # durante una corrida legacy).
                if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                    from src.tools.draft_writing.canonical_sentences import (
                        generate_section_canonical_v2,
                    )

                    generated_section = generate_section_canonical_v2(
                        section=section,
                        evidence=evidence,
                        quant_context=quant_context,
                        previous_errors=[],
                        policy=policy,
                        runtime=self.runtime,
                        raw_dir=raw_dir,
                        sid=sid,
                        runtime_invoke_sequence_base=llm_calls,
                    )
                    # Metadata de ejecución de V2 (nunca inferida de
                    # archivos, nunca duplicando contadores por
                    # separado): actualiza los contadores GLOBALES de
                    # Agent06 con lo que V2 realmente ejecutó, y se
                    # elimina antes de publicar la sección -- no forma
                    # parte del contrato externo que consumen 07/08.
                    v2_execution = generated_section.pop("_v2_execution")
                    llm_calls += v2_execution["llm_calls"]
                    validation_calls += v2_execution["validation_calls"]
                    attempt_logs[sid] = v2_execution["attempt_logs"]

                    if v2_execution["failed"]:
                        # Agotamiento de intentos V2: NUNCA fallback a
                        # legacy, NUNCA execution_status=FAILED vía
                        # excepción -- mismo contrato externo que ya
                        # usa legacy (COMPLETED + NEEDS_REVISION +
                        # RETRY/HALT_STAGE según intento externo), vía
                        # la función auxiliar nueva (nunca toca el
                        # bloque legacy existente).
                        return self._build_v2_section_validation_failed_result(
                            agent_input=agent_input,
                            sid=sid,
                            out=out,
                            raw_dir=raw_dir,
                            attempt_logs=attempt_logs,
                            retrieval_rounds=retrieval_rounds,
                            llm_calls=llm_calls,
                            validation_calls=validation_calls,
                            last_errors=v2_execution["last_errors"],
                            validation_version=policy.get("validation_version"),
                            start=start,
                        )

                    generated.append(generated_section)
                    continue

            evidence_map: dict[str, list[dict[str, Any]]] = {}
            for row in all_evidence:
                evidence_map.setdefault(row["section_id"], []).append(
                    {key: value for key, value in row.items() if key != "section_id"}
                )
            _, quality_rows, section_rows, claim_rows, numeric_rows = (
                build_draft_reports(generated, sections, evidence_map, policy)
            )
            validation = validate_draft_global(
                generated, sections, evidence_map, policy
            )
            validation.update(
                {
                    "stage": "06_agente_redactor",
                    "experiment_id": agent_input.experiment_id,
                    "validation_version": policy.get("validation_version"),
                    "generation_attempts": attempt_logs,
                    "current_raw_attempt_directory": str(raw_dir),
                }
            )
            validation_calls += 1

            # Reparación dirigida de longitud (requisito: configured_
            # min_total_words/configured_max_total_words son el único
            # gate contractual real -- ver validation.py). Se intenta
            # ÚNICAMENTE cuando el ÚNICO problema es longitud (todo lo
            # demás -- citas, claims, numérico, secciones -- ya está
            # correcto) y el contrato es canonical_sentences_v2 (Evidence
            # Handles es una construcción exclusivamente V2; legacy
            # conserva el comportamiento histórico sin reparación).
            length_repair_attempted = False
            length_repair_successful = False
            length_only_failure = (
                not validation["validation_ok"]
                and validation.get("word_count_compliant", True) is False
                and validation.get("all_section_validations_ok", False)
                and validation.get("invalid_citation_count", 1) == 0
                and not validation.get("sections_without_valid_citations", True)
                and not validation.get("sections_with_low_citation_density", True)
                and not validation.get("sections_with_claim_support_errors", True)
                and not validation.get("sections_with_quantitative_support_errors", True)
                and validation.get("numeric_failure_count", 1) == 0
            )
            if length_only_failure and contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                length_repair_attempted = True
                repaired_generated, repair_meta = attempt_directed_length_repair(
                    generated, sections, evidence_map, policy, self.runtime,
                )
                if repair_meta["attempted"]:
                    _, quality_rows, section_rows, claim_rows, numeric_rows = (
                        build_draft_reports(repaired_generated, sections, evidence_map, policy)
                    )
                    repaired_validation = validate_draft_global(
                        repaired_generated, sections, evidence_map, policy
                    )
                    if repaired_validation["validation_ok"]:
                        generated = repaired_generated
                        validation = repaired_validation
                        length_repair_successful = True
                    else:
                        # La reparación se intentó pero no cerró el
                        # déficit/exceso con evidencia real disponible --
                        # se conserva la validación ORIGINAL (nunca una
                        # mezcla parcial) para que el reason_code y las
                        # métricas de auditoría reflejen el estado real.
                        validation = repaired_validation
            validation["length_repair_attempted"] = length_repair_attempted
            validation["length_repair_successful"] = length_repair_successful

            if not validation["validation_ok"]:
                path = write_partial_validation(out, validation)
                artifacts = {
                    "draft_validation_report.json": ArtifactReference(
                        str(path), sha256_file(path)
                    ),
                    "raw_section_outputs": ArtifactReference(
                        str(out / "raw_section_outputs"), "DIRECTORY"
                    ),
                }
                is_final_attempt = agent_input.attempt_number != 1

                # Reason code específico de longitud -- nunca el genérico
                # INVALID_DRAFT para estos casos. Si ya se intentó reparar
                # sin éxito (evidencia insuficiente para el mínimo), se
                # distingue explícitamente de un simple déficit no
                # reparado todavía.
                if validation.get("word_count_compliant", True) is False and validation.get("all_section_validations_ok", False) and validation.get("invalid_citation_count", 1) == 0 and not validation.get("sections_without_valid_citations", True) and not validation.get("sections_with_low_citation_density", True) and not validation.get("sections_with_claim_support_errors", True) and not validation.get("sections_with_quantitative_support_errors", True) and validation.get("numeric_failure_count", 1) == 0:
                    if validation.get("word_deficit", 0) > 0:
                        length_reason_code = (
                            INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH
                            if length_repair_attempted
                            else TOTAL_WORD_COUNT_BELOW_MINIMUM
                        )
                    else:
                        length_reason_code = TOTAL_WORD_COUNT_ABOVE_MAXIMUM

                    # Intento final agotado con problema EXCLUSIVAMENTE de
                    # longitud: fail-closed CIENTÍFICO, nunca técnico --
                    # APPROVED_PENDING_MANUAL_REVIEW, nunca usable_for_
                    # evaluation (ese campo no existe en el contrato de
                    # Agent06; la evaluación parcial solo es válida
                    # después de Agent07) y nunca ADVANCE hacia 08 -- el
                    # target_stage permanece None, igual que el camino
                    # histórico.
                    if is_final_attempt:
                        return AgentResult(
                            execution_status=ExecutionStatus.COMPLETED,
                            quality_status=QualityStatus.APPROVED_PENDING_MANUAL_REVIEW,
                            decision=DecisionInfo(
                                code="DRAFT_LENGTH_OUT_OF_RANGE_MANUAL_REVIEW",
                                rationale=(
                                    "El borrador quedó fuera del rango de longitud contractual "
                                    "tras agotar los intentos; requiere revisión manual antes de "
                                    "continuar. No se publican salidas finales desde Agent06."
                                ),
                            ),
                            quality_metrics={
                                "scientific": {
                                    "configured_min_total_words": validation["configured_min_total_words"],
                                    "configured_max_total_words": validation["configured_max_total_words"],
                                    "target_total_words": validation["target_total_words"],
                                    "actual_total_words": validation["actual_total_words"],
                                    "effective_min_total_words": validation["effective_min_total_words"],
                                    "word_deficit": validation.get("word_deficit"),
                                    "word_excess": validation["word_excess"],
                                    "word_count_compliant": validation.get("word_count_compliant"),
                                },
                                "technical": {
                                    "validation_ok": False, "reused": False,
                                    "length_repair_attempted": length_repair_attempted,
                                    "length_repair_successful": length_repair_successful,
                                },
                            },
                            warnings=(
                                AgentWarning(
                                    code=length_reason_code,
                                    severity=WarningSeverity.WARNING,
                                    blocking=True,
                                    message=(
                                        f"Longitud fuera de rango tras agotar intentos: "
                                        f"actual_total_words={validation['actual_total_words']}, "
                                        f"configured_min_total_words={validation['configured_min_total_words']}, "
                                        f"configured_max_total_words={validation['configured_max_total_words']}."
                                    ),
                                ),
                            ),
                            failure_reason_codes=(length_reason_code,),
                            requested_transition=RequestedTransition(
                                action=TransitionAction.HALT_STAGE,
                                target_stage=None,
                                reason_code=length_reason_code,
                                requires_human_confirmation=True,
                            ),
                            output_artifacts=artifacts,
                            tool_usage=ToolUsage(
                                retrieval_rounds=retrieval_rounds,
                                llm_calls=llm_calls,
                                validation_calls=validation_calls,
                            ),
                            attempt_number=agent_input.attempt_number,
                            started_at=start,
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )

                    return AgentResult(
                        execution_status=ExecutionStatus.COMPLETED,
                        quality_status=QualityStatus.NEEDS_REVISION,
                        decision=DecisionInfo(
                            code="DRAFT_VALIDATION_FAILED",
                            rationale=(
                                "El borrador no superó la validación global; "
                                "no se publicaron salidas finales."
                            ),
                        ),
                        quality_metrics={
                            "scientific": {
                                "configured_min_total_words": validation["configured_min_total_words"],
                                "configured_max_total_words": validation["configured_max_total_words"],
                                "target_total_words": validation["target_total_words"],
                                "actual_total_words": validation["actual_total_words"],
                                "effective_min_total_words": validation["effective_min_total_words"],
                                "word_deficit": validation.get("word_deficit"),
                                "word_excess": validation["word_excess"],
                                "word_count_compliant": validation.get("word_count_compliant"),
                            },
                            "technical": {
                                "validation_ok": False, "reused": False,
                                "length_repair_attempted": length_repair_attempted,
                                "length_repair_successful": length_repair_successful,
                            },
                        },
                        warnings=(
                            AgentWarning(
                                code=length_reason_code,
                                severity=WarningSeverity.ERROR,
                                blocking=True,
                                message="La validación global de longitud fue negativa.",
                            ),
                        ),
                        failure_reason_codes=(length_reason_code,),
                        requested_transition=RequestedTransition(
                            action=TransitionAction.RETRY,
                            target_stage=None,
                            reason_code=length_reason_code,
                            requires_human_confirmation=False,
                        ),
                        output_artifacts=artifacts,
                        tool_usage=ToolUsage(
                            retrieval_rounds=retrieval_rounds,
                            llm_calls=llm_calls,
                            validation_calls=validation_calls,
                        ),
                        attempt_number=agent_input.attempt_number,
                        started_at=start,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )

                action = (
                    TransitionAction.RETRY
                    if agent_input.is_first_attempt()
                    else TransitionAction.HALT_STAGE
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.NEEDS_REVISION,
                    decision=DecisionInfo(
                        code="DRAFT_VALIDATION_FAILED",
                        rationale=(
                            "El borrador no superó la validación global; "
                            "no se publicaron salidas finales."
                        ),
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": False, "reused": False},
                    },
                    warnings=(
                        AgentWarning(
                            code="INVALID_DRAFT",
                            severity=WarningSeverity.ERROR,
                            blocking=True,
                            message="La validación global fue negativa.",
                        ),
                    ),
                    failure_reason_codes=("INVALID_DRAFT",),
                    requested_transition=RequestedTransition(
                        action=action,
                        target_stage=None,
                        reason_code="NEEDS_REVISION",
                        requires_human_confirmation=False,
                    ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=retrieval_rounds,
                        llm_calls=llm_calls,
                        validation_calls=validation_calls,
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            draft = {
                "title": bundle["outline"].get(
                    "title", "Borrador del estado del arte"
                ),
                "topic": bundle["outline"].get("topic", ""),
                "status": "draft_validated_for_verification",
                "sections": generated,
                "generation_summary": {
                    "experiment_id": agent_input.experiment_id,
                    "section_count": len(generated),
                    "ground_truth_used": False,
                    "open_search_used": False,
                    "citation_format": "[source_filename | chunk_id]",
                    "retrieval_strategy": strategy,
                    **versions,
                },
            }
            manifest_versions = {
                "stage": versions["stage_version"],
                "prompt": policy.get("prompt_version"),
                "rag": versions["rag_version"],
                "validation": versions["validation_version"],
                "normalization": versions["normalization_version"],
            }
            manifest = {
                "stage": agent_input.stage_name,
                "experiment_id": agent_input.experiment_id,
                "run_id": agent_input.run_id,
                "attempt_number": agent_input.attempt_number,
                "fingerprint": policy.get("current_fingerprint"),
                "retrieval_strategy": strategy,
                "validation_ok": True,
                "safety_policy": {
                    "uses_ground_truth": False,
                    "uses_external_knowledge": False,
                    "open_search_used": False,
                },
                "counts": {
                    "sections": len(generated),
                    "llm_calls": llm_calls,
                    "retrieval_rounds": retrieval_rounds,
                },
                "versions": manifest_versions,
                # Mismo contrato que en los caminos de fallo (ver
                # docstring de raw_dir más arriba): el subdirectorio de
                # ESTE intento externo, dentro del histórico completo que
                # representa el ArtifactReference "raw_section_outputs".
                "current_raw_attempt_directory": str(raw_dir),
            }
            if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                # ÚNICAMENTE V2 declara esta clave -- legacy (ausente o
                # "legacy" explícito) mantiene el manifest histórico
                # sin campo nuevo, byte a byte, tal como garantiza
                # LEGACY05/LEGACY07b (fase 1). Mismo criterio que
                # _draft_signature (draft_writing_runtime.py): la clave
                # solo existe cuando el contrato realmente es V2.
                manifest["draft_representation_contract"] = contract
            artifacts = write_draft_artifacts(
                out,
                draft,
                all_evidence,
                validation,
                bundle["quantitative"],
                bundle["dataset_summary"],
                manifest,
                quality_rows,
                section_rows,
                claim_rows,
                numeric_rows,
            )
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=QualityStatus.APPROVED,
                decision=DecisionInfo(
                    code="DRAFT_APPROVED",
                    rationale=(
                        "Borrador generado por secciones y validado con "
                        "evidencia restringida."
                    ),
                ),
                quality_metrics={
                    "scientific": {
                        "configured_min_total_words": validation.get("configured_min_total_words"),
                        "configured_max_total_words": validation.get("configured_max_total_words"),
                        "target_total_words": validation.get("target_total_words"),
                        "actual_total_words": validation.get("actual_total_words"),
                        "effective_min_total_words": validation.get("effective_min_total_words"),
                        "word_deficit": validation.get("word_deficit"),
                        "word_excess": validation.get("word_excess"),
                        "word_count_compliant": validation.get("word_count_compliant"),
                    },
                    "technical": {
                        "validation_ok": True, "reused": False,
                        "length_repair_attempted": validation.get("length_repair_attempted", False),
                        "length_repair_successful": validation.get("length_repair_successful", False),
                    },
                },
                warnings=(),
                failure_reason_codes=(),
                requested_transition=RequestedTransition(
                    action=TransitionAction.ADVANCE,
                    target_stage="07_agente_verificador",
                    reason_code="APPROVED",
                    requires_human_confirmation=False,
                ),
                output_artifacts=artifacts,
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            message = str(exc)
            known = (
                "DRAFT_INPUT_NOT_FOUND",
                "OUTLINE_NOT_APPROVED",
                "OUTLINE_MANIFEST_MISMATCH",
                "GROUND_TRUTH_POLICY_VIOLATION",
                "INVALID_DRAFT_KB_SCHEMA",
                "INVALID_CHUNKS_SCHEMA",
                "INVALID_QUANTITATIVE_CONTEXT",
                "THEMATIC_NOT_APPROVED",
                "OUTLINE_MANIFEST_NOT_APPROVED",
                "THEMATIC_MANIFEST_NOT_APPROVED",
                "OUTLINE_SOURCES_NOT_VALIDATED",
                "OUTLINE_TITLES_NOT_VALIDATED",
                "CHROMA_COLLECTION_MISMATCH",
                "CHROMA_EMBEDDING_MODEL_MISMATCH",
                "UNSAFE_CHROMA_INDEX",
                "DUPLICATE_KB_SOURCE",
                "DUPLICATE_CHUNK_ID",
                "UNSAFE_CHUNKS",
                "CHROMA_CHUNK_COUNT_MISMATCH",
                "INVALID_OUTLINE_SECTION_IDS",
                "INVALID_OUTLINE_MAPPING_SCHEMA",
                "OUTLINE_MAPPING_INCONSISTENT",
                "QUANTITATIVE_MANIFEST_MISMATCH",
                "INVALID_OUTLINE_SCHEMA",
                "MISSING_SECTION_EVIDENCE",
                "SECTION_VALIDATION_FAILED",
                "INVALID_LLM_OUTPUT",
                "CREDENTIAL_NOT_FOUND",
                "ATOMIC_WRITE_FAILED",
                "UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY",
            )
            code = next((item for item in known if item in message), "RUNTIME_DEPENDENCY_FAILED")
            return AgentResult(
                execution_status=ExecutionStatus.FAILED,
                quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(
                    code="DRAFT_WRITING_FAILED",
                    rationale="Falló la ejecución del Agente Redactor.",
                ),
                quality_metrics={"scientific": {}, "technical": {}},
                warnings=(
                    AgentWarning(
                        code=code,
                        severity=WarningSeverity.ERROR,
                        blocking=True,
                        message=message,
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
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={
                    "type": type(exc).__name__,
                    "message": message,
                    "stage": agent_input.stage_name,
                },
            )
