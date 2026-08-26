# ============================================================
# 06 - AGENTE REDACTOR DEL ESTADO DEL ARTE
# Redacta el borrador del estado del arte a partir del esquema
# y de la evidencia científica disponible.
# ============================================================

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import pandas as pd

# Importa los contratos y herramientas que necesita el agente 06
# para redactar, recuperar evidencia, validar, reparar y guardar el borrador.
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
    write_raw_section_output,
    write_raw_section_rag_trace,
    write_raw_section_validation,
)
from src.tools.draft_writing.hybrid_retrieval import retrieve_section_evidence_hybrid
from src.tools.draft_writing.input_validation import validate_draft_dependencies
from src.tools.draft_writing.normalization import (
    detect_claims_missing_leading_discourse_connector,
    normalize_generated_section,
    split_sentences_preserving_citations,
)
from src.tools.draft_writing.prompting import (
    assign_section_budgets,
    build_section_prompt,
    build_source_free_organizational_section,
)
from src.tools.draft_writing.quantitative_augmentation import (
    augment_evidence_with_quantitative_chunks_greedy,
)
from src.tools.draft_writing.retrieval import (
    build_section_query,
    retrieve_section_evidence,
)
from src.tools.draft_writing.source_aware_budgets import (
    assign_source_aware_section_budgets,
)
from src.tools.draft_writing.validation import (
    CITATION_RE,
    build_draft_reports,
    count_words,
    validate_draft_global,
    validate_generated_section,
    section_allows_no_sources,
)
from src.tools.draft_writing.length_repair import attempt_directed_length_repair

# Define las dos estrategias de recuperación de evidencia disponibles para la redacción de las secciones del estado del arte.
LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted" #primero busca en Chroma y luego usa el CSV restringido
HYBRID_RETRIEVAL_STRATEGY = "hybrid_chroma_csv_rrf_balanced" #combina resultados de Chroma + CSV y los fusiona de forma balanceada mediante RRF (combinar varios rankings de resultados en uno solo.)
# Define el contrato histórico y el contrato canónico de representación del borrador; el legacy se mantiene por compatibilidad del pipeline.
LEGACY_DRAFT_REPRESENTATION_CONTRACT = "legacy"
CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT = "canonical_sentences_v2"
# Define códigos de validación relacionados con la longitud total del borrador y con la falta de contenido respaldado suficiente para alcanzar el mínimo.
TOTAL_WORD_COUNT_BELOW_MINIMUM = "TOTAL_WORD_COUNT_BELOW_MINIMUM"
TOTAL_WORD_COUNT_ABOVE_MAXIMUM = "TOTAL_WORD_COUNT_ABOVE_MAXIMUM"
INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH = "INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH"
# Registra las versiones históricas de los componentes usados por el agente 06
# para mantener trazabilidad y compatibilidad con ejecuciones anteriores
LEGACY_VERSIONS = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}
# Registra las versiones de los componentes usados por la estrategia híbrida
# del agente 06 para mantener trazabilidad y reproducibilidad.
HYBRID_VERSIONS = {
    "stage_version": "06_AGENTIC_V17_HYBRID_QUANTITATIVE_SOURCE_AWARE",
    "rag_version": "hybrid_chroma_csv_rrf_balanced_v1",
    "quantitative_selection_version": "confirmed_literal_greedy_coverage_v1",
    "budget_version": "source_aware_exact_total_v1",
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}


class DraftWritingAgent:
    """Contractual Agent 06 with explicit legacy and V17 hybrid branches."""

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
        strategy = str(
            policy.get("retrieval_strategy", LEGACY_RETRIEVAL_STRATEGY)
        ).strip()
        if strategy not in {LEGACY_RETRIEVAL_STRATEGY, HYBRID_RETRIEVAL_STRATEGY}:
            raise ValueError(f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy}")
        return strategy

    @staticmethod
    def _effective_versions(
        policy: Mapping[str, Any], strategy: str
    ) -> dict[str, str]:
        del policy
        if strategy == HYBRID_RETRIEVAL_STRATEGY:
            return dict(HYBRID_VERSIONS)
        return dict(LEGACY_VERSIONS)

    @staticmethod
    def _section_budgets(
        sections: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        strategy: str,
    ) -> dict[str, dict[str, Any]]:
        target_total_words = int(policy["target_total_words"])
        if strategy == HYBRID_RETRIEVAL_STRATEGY:
            return assign_source_aware_section_budgets(
                sections,
                target_total_words,
                policy=policy,
            )
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
        top_k = int(policy["top_k_evidence_per_section"])
        max_chars = int(policy["max_evidence_chars"])
        if strategy == LEGACY_RETRIEVAL_STRATEGY:
            return retrieve_section_evidence(
                section,
                self.runtime.collection,
                chunks,
                top_k,
                max_chars,
            )

        hybrid_evidence = retrieve_section_evidence_hybrid(
            section,
            self.runtime.collection,
            chunks,
            candidate_multiplier=int(policy["candidate_multiplier"]),
            chroma_quota=int(policy["chroma_quota"]),
            csv_quota=int(policy["csv_quota"]),
            rrf_quota=int(policy["rrf_quota"]),
            rrf_k=int(policy["rrf_k"]),
            top_k_evidence_per_section=top_k,
            max_evidence_chars=max_chars,
            max_candidates_per_source=int(policy["max_candidates_per_source"]),
        )
        return augment_evidence_with_quantitative_chunks_greedy(
            hybrid_evidence,
            chunks,
            quantitative_context,
            allowed_papers=self._section_sources(section),
            top_k_evidence_per_section=top_k,
            quantitative_evidence_quota=int(
                policy.get("quantitative_evidence_quota", 0)
            ),
            max_evidence_chars=max_chars,
            max_candidates_per_source=int(policy["max_candidates_per_source"]),
            valid_source_chunk_pairs=self._valid_source_chunk_pairs(chunks),
            max_quantitative_rows_per_section=int(
                policy["max_quantitative_rows_per_section"]
            ),
        )

    @staticmethod
    def _trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
        trace_fields = (
            "source_filename",
            "chunk_id",
            "text",
            "score",
            "retrieval_method",
            "retrieval_source",
            "retrieval_sources",
            "chroma_rank",
            "csv_rank",
            "rrf_score",
            "selection_bucket",
            "selection_order",
            "quantitative_values",
            "quantitative_coverage_keys",
            "quantitative_marginal_gain",
            "quantitative_row_ids",
            "verification_statuses",
        )
        return {field: row.get(field) for field in trace_fields if field in row}

    @staticmethod
    def _unique_validation_items(items: Sequence[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @classmethod
    def _combine_section_validations(
        cls,
        original_validation: Mapping[str, Any],
        normalized_validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        combined: dict[str, Any] = dict(normalized_validation)
        for field in ("errors", "citation_errors", "claim_errors", "numeric_errors"):
            combined[field] = cls._unique_validation_items(
                list(original_validation.get(field) or [])
                + list(normalized_validation.get(field) or [])
            )
        combined["validation_ok"] = bool(
            original_validation.get("validation_ok")
            and normalized_validation.get("validation_ok")
        )
        combined["original_validation_ok"] = bool(
            original_validation.get("validation_ok")
        )
        combined["normalized_validation_ok"] = bool(
            normalized_validation.get("validation_ok")
        )
        return combined

    # ------------------------------------------------------------------
    # Conservative deterministic numeric salvage
    # ------------------------------------------------------------------

    @staticmethod
    def _unsupported_numeric_values(
        validation: Mapping[str, Any],
    ) -> list[str]:
        """
        Return unsupported numeric literals only when the section has no
        non-numeric validation failures.

        This is intentionally fail-closed: citation, claim, structural, or
        mixed numeric errors prevent salvage.
        """
        if list(validation.get("errors") or []):
            return []
        if list(validation.get("citation_errors") or []):
            return []
        if list(validation.get("claim_errors") or []):
            return []

        numeric_errors = list(validation.get("numeric_errors") or [])
        if not numeric_errors:
            return []

        prefix = "UNSUPPORTED_NUMERIC_VALUE:"
        values: list[str] = []

        for item in numeric_errors:
            text = str(item).strip()
            if not text.startswith(prefix):
                return []

            value = text[len(prefix):].strip()
            if not value:
                return []

            values.append(value)

        return values

    @staticmethod
    def _numeric_literal_pattern(raw_value: str) -> re.Pattern[str] | None:
        """
        Build an exact numeric literal matcher.

        Examples:
        40%   -> matches 40% or 40 %
        4.5   -> matches 4.5 (and text normalized from 4,5)
        20    -> does not match 120 or 20.5
        """
        value = str(raw_value or "").strip().replace(",", ".")
        if not value:
            return None

        has_percent = value.endswith("%")
        numeric = value[:-1].strip() if has_percent else value

        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
            return None

        suffix = r"\s*%" if has_percent else ""
        return re.compile(
            rf"(?<![\d.]){re.escape(numeric)}{suffix}(?![\d.])"
        )

    @classmethod
    def _contains_unsupported_numeric(
        cls,
        text: str,
        values: Sequence[str],
    ) -> bool:
        normalized_text = str(text or "").replace(",", ".")
        for value in values:
            pattern = cls._numeric_literal_pattern(value)
            if pattern is not None and pattern.search(normalized_text):
                return True
        return False

    @classmethod
    def _salvage_numeric_only_section(
        cls,
        generated: Mapping[str, Any],
        validation: Mapping[str, Any],
        allowed_pairs: set[tuple[str, str]],
    ) -> tuple[dict[str, Any], list[str], list[str]] | None:
        """
        Conservatively delete entire sentences/claims containing unsupported
        numeric literals after LLM revision attempts are exhausted.

        No value is replaced, inferred, rounded, or invented.
        """
        values = cls._unsupported_numeric_values(validation)
        if not values:
            return None

        draft_text = str(generated.get("draft_text", "") or "")
        sentences = split_sentences_preserving_citations(draft_text)
        if not sentences:
            return None

        kept_sentences: list[str] = []
        removed_sentences: list[str] = []

        for sentence in sentences:
            if cls._contains_unsupported_numeric(sentence, values):
                removed_sentences.append(sentence)
            else:
                kept_sentences.append(sentence)

        # Fail closed if no offending sentence was localized, or salvage would
        # erase the whole section.
        if not removed_sentences or not kept_sentences:
            return None

        kept_claims: list[dict[str, Any]] = []
        for claim in generated.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim", "") or "")
            if cls._contains_unsupported_numeric(claim_text, values):
                continue
            kept_claims.append(dict(claim))

        candidate = dict(generated)
        candidate["draft_text"] = " ".join(kept_sentences)
        candidate["claims"] = kept_claims

        # Re-run canonical normalization so citations and claims preserve the
        # same closed allowed-pair contract used for ordinary generation.
        candidate = normalize_generated_section(candidate, allowed_pairs)
        candidate["generation_attempt"] = generated.get("generation_attempt")

        return candidate, removed_sentences, values

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

    def _try_immediate_numeric_salvage(
        self,
        *,
        sid: str,
        generation_attempt: int,
        normalized: Mapping[str, Any],
        validation: Mapping[str, Any],
        allowed: set[tuple[str, str]],
        section: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        raw_dir: Path,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
        """
        Deterministic numeric salvage, intentado INMEDIATAMENTE sobre el
        intento que acaba de generarse -- nunca esperando a que se agoten
        todos los intentos internos. Se activa únicamente cuando la
        validación de ESTE intento falla exclusivamente por
        UNSUPPORTED_NUMERIC_VALUE:* (fail-closed heredado sin cambios de
        _unsupported_numeric_values: cualquier error de cita, claim o
        estructura desactiva el salvage por completo).

        Devuelve (sección_aceptada_o_None, entrada_de_log_o_None,
        validation_calls_delta). Si el salvage no aplica o no logra pasar
        validate_generated_section, devuelve (None, None, 0) -- el
        llamador simplemente continúa con el siguiente intento normal,
        sin consumir ninguna llamada LLM adicional por este intento de
        salvage en sí (es determinista, nunca invoca al LLM).
        """
        salvage = self._salvage_numeric_only_section(normalized, validation, allowed)
        if salvage is None:
            return None, None, 0

        salvaged_section, removed_sentences, removed_values = salvage
        salvage_validation = validate_generated_section(salvaged_section, section, evidence)
        salvaged_section["section_validation"] = salvage_validation

        salvage_tag = f"numeric_salvage_from_{generation_attempt}"
        salvage_raw_path = write_raw_section_output(
            raw_dir, sid, salvage_tag,
            json.dumps(salvaged_section, ensure_ascii=False, indent=2),
        )
        salvage_validation_payload = {
            "section_id": sid,
            "generation_attempt": salvage_tag,
            "mode": "deterministic_numeric_sentence_salvage",
            "salvaged_from_attempt": generation_attempt,
            "validation_ok": bool(salvage_validation.get("validation_ok")),
            "validation_errors": self._unique_validation_items(
                list(salvage_validation.get("errors") or [])
                + list(salvage_validation.get("citation_errors") or [])
                + list(salvage_validation.get("claim_errors") or [])
                + list(salvage_validation.get("numeric_errors") or [])
            ),
            "numeric_support_errors": list(salvage_validation.get("numeric_errors") or []),
            "removed_unsupported_numeric_values": removed_values,
            "removed_sentences": removed_sentences,
            "word_count": count_words(salvaged_section.get("draft_text", "")),
            "citation_count": len(CITATION_RE.findall(str(salvaged_section.get("draft_text", "")))),
            "raw_output_path": str(salvage_raw_path),
        }
        salvage_validation_path = write_raw_section_validation(
            raw_dir, sid, salvage_tag, salvage_validation_payload,
        )
        log_entry = {
            "attempt": salvage_tag,
            "mode": "deterministic_numeric_sentence_salvage",
            "salvaged_from_attempt": generation_attempt,
            "validation": salvage_validation,
            "attempt_validation_path": str(salvage_validation_path),
            "removed_unsupported_numeric_values": removed_values,
            "removed_sentences": removed_sentences,
        }

        if bool(salvage_validation.get("validation_ok")):
            return salvaged_section, log_entry, 1
        return None, log_entry, 1

    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0
        retrieval_rounds = 0
        validation_calls = 0
        out = Path(agent_input.agent_context.output_directory)
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
            contract = policy.get(
                "draft_representation_contract",
                LEGACY_DRAFT_REPRESENTATION_CONTRACT,
            )
            if contract not in {
                LEGACY_DRAFT_REPRESENTATION_CONTRACT,
                CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
            }:
                raise ValueError(
                    f"UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:{contract}"
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
                if (
                    strategy == HYBRID_RETRIEVAL_STRATEGY
                    and not self._section_sources(section)
                    and section_allows_no_sources(section)
                ):
                    evidence = []
                else:
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
                    v2_execution = generated_section.pop("_v2_execution")
                    llm_calls += v2_execution["llm_calls"]
                    validation_calls += v2_execution["validation_calls"]
                    attempt_logs[sid] = v2_execution["attempt_logs"]

                    if v2_execution["failed"]:
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

                previous_errors: list[Any] = []
                logs: list[dict[str, Any]] = []
                accepted = None
                normalized: dict[str, Any] | None = None
                allowed: set[tuple[str, str]] = set()

                for generation_attempt in range(
                    1, int(policy["max_section_revision_attempts"]) + 2
                ):
                    previous_errors_codes_for_this_attempt = list(previous_errors)
                    prompt = build_section_prompt(
                        section,
                        evidence,
                        quant_context,
                        previous_errors,
                        policy,
                    )
                    prompt_sha256 = hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest()
                    runtime_invoke_sequence_before = llm_calls
                    raw = self.runtime.invoke(prompt)
                    llm_calls += 1
                    raw_response_sha256 = hashlib.sha256(
                        str(raw).encode("utf-8")
                    ).hexdigest()
                    runtime_response_metadata: dict[str, Any] = {}
                    response_id = getattr(raw, "id", None)
                    if isinstance(response_id, str) and response_id:
                        runtime_response_metadata["response_id"] = response_id
                    provider_metadata = getattr(raw, "response_metadata", None)
                    if isinstance(provider_metadata, Mapping):
                        for key in ("cache_hit", "cached", "cache_read"):
                            if key in provider_metadata:
                                runtime_response_metadata["provider_" + key] = provider_metadata[key]
                    raw_path = write_raw_section_output(
                        raw_dir, sid, generation_attempt, raw
                    )
                    parsed = self.runtime.parse(raw)
                    allowed = {
                        (row["source_filename"], row["chunk_id"]) for row in evidence
                    }

                    original_validation = None
                    if strategy == HYBRID_RETRIEVAL_STRATEGY:
                        original_validation = validate_generated_section(
                            parsed, section, evidence
                        )
                        validation_calls += 1

                    normalized = normalize_generated_section(parsed, allowed)
                    normalized["generation_attempt"] = generation_attempt
                    normalized_validation = validate_generated_section(
                        normalized, section, evidence
                    )
                    validation_calls += 1

                    if original_validation is None:
                        validation = normalized_validation
                    else:
                        validation = self._combine_section_validations(
                            original_validation,
                            normalized_validation,
                        )

                    normalized["section_validation"] = validation
                    citation_errors = list(validation.get("citation_errors") or [])
                    claim_errors = list(validation.get("claim_errors") or [])
                    numeric_errors = list(validation.get("numeric_errors") or [])

                    
                    for finding in detect_claims_missing_leading_discourse_connector(parsed):
                        claim_errors.append(
                            "CLAIM_MISSING_INITIAL_DISCOURSE_CONNECTOR: "
                            f"el claim declarado omite el conector discursivo inicial "
                            f"'{finding['connector']},' presente en la oración real del "
                            f"draft_text ('{finding['sentence'][:100]}'). "
                            "claims[].claim debe ser copia literal COMPLETA de la "
                            "oración correspondiente (sin las citas inline), "
                            "conservando cualquier conector discursivo inicial "
                            "(ej. 'Por ejemplo,', 'Además,', 'Finalmente,')."
                        )

                    def reason(item: Any) -> str:
                        return (
                            str(item.get("reason", ""))
                            if isinstance(item, dict)
                            else str(item)
                        )

                    validation_errors = self._unique_validation_items(
                        list(validation.get("errors") or [])
                        + citation_errors
                        + claim_errors
                        + numeric_errors
                    )
                    attempt_validation = {
                        "section_id": sid,
                        "generation_attempt": generation_attempt,
                        "validation_ok": bool(validation.get("validation_ok")),
                        "validation_errors": validation_errors,
                       
                        "retry_audit": {
                            "previous_errors_codes_used_in_prompt": previous_errors_codes_for_this_attempt,
                            "prompt_sha256": prompt_sha256,
                            "raw_response_sha256": raw_response_sha256,
                            "runtime_invoke_sequence_number": runtime_invoke_sequence_before + 1,
                            "runtime_invoke_executed": True,
                            "runtime_response_metadata": runtime_response_metadata,
                        },
                        "invalid_citations": [
                            item
                            for item in citation_errors
                            if reason(item)
                            in {
                                "invalid_citation",
                                "citation_not_in_section_evidence",
                                "citation_in_source_free_section",
                            }
                        ],
                        "unsupported_claims": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "missing_claim_for_sentence",
                                "claim_without_supporting_citations",
                                "claim_citation_not_in_section_evidence",
                                "claim_not_exact_sentence",
                                "substantive_sentence_missing_from_claims",
                            }
                        ],
                        "substantive_sentences_without_claim": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "missing_claim_for_sentence",
                                "substantive_sentence_missing_from_claims",
                            }
                        ],
                        "substantive_sentences_without_citation": [
                            item
                            for item in citation_errors
                            if reason(item)
                            in {
                                "uncited_substantive_sentence",
                                "substantive_sentence_without_citation",
                                "section_without_citations",
                            }
                        ],
                        "claim_sentence_mismatches": [
                            item
                            for item in claim_errors
                            if reason(item)
                            in {
                                "claim_citation_mismatch",
                                "claim_sentence_citation_mismatch",
                                "claim_not_exact_sentence",
                            }
                        ],
                        "numeric_support_errors": numeric_errors,
                        "word_count": count_words(normalized.get("draft_text", "")),
                        "citation_count": len(
                            CITATION_RE.findall(str(normalized.get("draft_text", "")))
                        ),
                        "raw_output_path": str(raw_path),
                    }
                    if strategy == HYBRID_RETRIEVAL_STRATEGY:
                        attempt_validation["original_validation"] = original_validation
                        attempt_validation["normalized_validation"] = normalized_validation
                    validation_path = write_raw_section_validation(
                        raw_dir, sid, generation_attempt, attempt_validation
                    )
                    raw_draft_text = (
                        str(parsed.get("draft_text", ""))
                        if isinstance(parsed, dict)
                        else ""
                    )
                    normalized_draft_text = str(normalized.get("draft_text", ""))
                    rag_trace = {
                        "section_id": sid,
                        "generation_attempt": generation_attempt,
                        "retrieval_strategy": strategy,
                        "query": section_query,
                        "retrieved_chunks": [self._trace_row(row) for row in evidence],
                        "allowed_citations": [
                            f"[{row.get('source_filename', '')} | {row.get('chunk_id', '')}]"
                            for row in evidence
                        ],
                        "llm_citations": CITATION_RE.findall(raw_draft_text),
                        "normalized_citations": CITATION_RE.findall(
                            normalized_draft_text
                        ),
                    }
                    rag_trace_path = write_raw_section_rag_trace(
                        raw_dir, sid, generation_attempt, rag_trace
                    )
                    logs.append(
                        {
                            "attempt": generation_attempt,
                            "validation": validation,
                            "attempt_validation_path": str(validation_path),
                            "rag_trace_path": str(rag_trace_path),
                        }
                    )
                    if validation["validation_ok"]:
                        accepted = normalized
                        break

                    (
                        salvaged_accepted,
                        salvage_log_entry,
                        salvage_validation_calls,
                    ) = self._try_immediate_numeric_salvage(
                        sid=sid,
                        generation_attempt=generation_attempt,
                        normalized=normalized,
                        validation=validation,
                        allowed=allowed,
                        section=section,
                        evidence=evidence,
                        raw_dir=raw_dir,
                    )
                    validation_calls += salvage_validation_calls
                    if salvage_log_entry is not None:
                        logs.append(salvage_log_entry)
                    if salvaged_accepted is not None:
                        accepted = salvaged_accepted
                        break

                    previous_errors = (
                        list(validation.get("errors") or [])
                        + citation_errors
                        + claim_errors
                        + numeric_errors
                    )

                attempt_logs[sid] = logs

                if accepted is None:
                    last_validation = (
                        (logs[-1].get("validation") or {}) if logs else {}
                    )
                    partial_validation = {
                        "stage": "06_agente_redactor",
                        "experiment_id": agent_input.experiment_id,
                        "validation_version": policy.get("validation_version"),
                        "validation_ok": False,
                        "failed_section": sid,
                        "section_attempts": len(logs),
                        "last_attempt_errors": list(
                            last_validation.get("errors") or []
                        )
                        + list(last_validation.get("citation_errors") or [])
                        + list(last_validation.get("claim_errors") or [])
                        + list(last_validation.get("numeric_errors") or []),
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
                                f"La sección {sid} agotó sus reintentos internos; "
                                "se preservaron salidas y validaciones por intento."
                            ),
                        ),
                        quality_metrics={
                            "scientific": {},
                            "technical": {
                                "validation_ok": False,
                                "reused": False,
                                "failed_section": sid,
                                "section_attempts": len(logs),
                            },
                        },
                        warnings=(
                            AgentWarning(
                                code="SECTION_VALIDATION_FAILED",
                                severity=WarningSeverity.ERROR,
                                blocking=True,
                                message=(
                                    f"La sección {sid} no superó la validación "
                                    f"tras {len(logs)} intentos."
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

                generated.append(accepted)

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

                if validation.get("word_count_compliant", True) is False and validation.get("all_section_validations_ok", False) and validation.get("invalid_citation_count", 1) == 0 and not validation.get("sections_without_valid_citations", True) and not validation.get("sections_with_low_citation_density", True) and not validation.get("sections_with_claim_support_errors", True) and not validation.get("sections_with_quantitative_support_errors", True) and validation.get("numeric_failure_count", 1) == 0:
                    if validation.get("word_deficit", 0) > 0:
                        length_reason_code = (
                            INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH
                            if length_repair_attempted
                            else TOTAL_WORD_COUNT_BELOW_MINIMUM
                        )
                    else:
                        length_reason_code = TOTAL_WORD_COUNT_ABOVE_MAXIMUM

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
            if strategy == HYBRID_RETRIEVAL_STRATEGY:
                manifest_versions.update(
                    {
                        "quantitative_selection": versions[
                            "quantitative_selection_version"
                        ],
                        "budget": versions["budget_version"],
                    }
                )
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
               
                "current_raw_attempt_directory": str(raw_dir),
            }
            if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
    
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
