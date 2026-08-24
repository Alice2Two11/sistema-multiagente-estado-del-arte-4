from __future__ import annotations
import re
from .retrieval import safe_str
from .normalization import CITATION_RE, split_sentences_preserving_citations, is_substantive_sentence, normalize_claim_text
from .prompting import assign_section_budgets
from src.tools.shared.section_source_requirement import section_is_source_free_organizational


def compute_unsupported_numeric_values(
    text, citation_pairs, evidence_lookup, section_evidence_tokens, *,
    allow_section_evidence_fallback=True,
):
    """Función PURA y compartida: para un texto (oración o claim) y los
    pares de cita asociados a él, calcula qué valores numéricos NO
    están soportados por el texto de sus propias citas
    (``number_exists_in_text``).

    ``allow_section_evidence_fallback`` (default ``True``, preserva
    exactamente la semántica histórica de legacy): cuando es ``True``,
    un número también se considera soportado si aparece en el conjunto
    de tokens numéricos canónicos de TODA la evidencia de la sección
    (``_canonical_numeric_tokens``), no solo en las citas propias del
    texto -- perdón por soporte indirecto. ``canonical_sentences_v2``
    llama a esta función con ``allow_section_evidence_fallback=False``
    explícitamente: V2 exige la MISMA regla que después lo juzga
    ``build_draft_reports`` (soportado únicamente si aparece en alguno
    de los ``supporting_citations`` propios del claim) -- nunca el
    perdón por evidencia recuperada pero no citada, que produciría
    exactamente la inconsistencia observada en Exp07 (``FOUND_IN_
    OTHER_RETRIEVED_EVIDENCE`` aceptado localmente por V2 pero luego
    rechazado por el reporte global, que nunca aplicó ese perdón).

    Esta función es una extracción, no una reimplementación: legacy la
    consume internamente sin cambio de comportamiento (ver más abajo),
    y es la única pieza que ``canonical_sentences_v2`` importa de este
    módulo -- nunca el resto de la lógica legacy (matching exacto de
    claim==oración, citation_errors, etc.), que no aplica al contrato
    V2.

    Devuelve una lista de ``"UNSUPPORTED_NUMERIC_VALUE:<valor>"`` (sin
    deduplicar ni ordenar -- eso es responsabilidad del llamador, igual
    que antes de esta extracción)."""

    errors = []
    for number in re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?", text):
        if any(number_exists_in_text(number, evidence_lookup.get(pair, "")) for pair in citation_pairs):
            continue
        if allow_section_evidence_fallback and (_canonical_numeric_tokens(number) & section_evidence_tokens):
            continue
        errors.append(f"UNSUPPORTED_NUMERIC_VALUE:{number}")
    return errors


def build_section_evidence_numeric_tokens(evidence):
    """Función PURA y compartida: conjunto de tokens numéricos
    canónicos presentes en CUALQUIER fila de evidencia de la sección
    (no solo la citada por una oración específica) -- MISMA semántica
    histórica exacta que ya construía ``validate_generated_section``
    inline antes de esta extracción."""

    tokens = set()
    for row in evidence or []:
        if not isinstance(row, dict):
            continue
        tokens.update(_canonical_numeric_tokens(row.get("text") or row.get("chunk_text") or ""))
    return tokens


def count_words(text):
    return len(re.findall(r"\b[\wáéíóúüñ]+\b", safe_str(text), flags=re.IGNORECASE))


def count_content_words(text):
    """Cuenta palabras del CONTENIDO LINGÜÍSTICO real, excluyendo citas
    estructuradas reconocidas por CITATION_RE (``[source_filename |
    chunk_id]``) -- las citas permanecen intactas en ``draft_text`` en
    todo momento; esta función NUNCA lo modifica, solo calcula un
    conteo aparte que excluye los tokens internos de la cita (nombre de
    archivo, extensión, chunk_id) de la extensión narrativa del estado
    del arte.

    No elimina nada más: cualquier texto entre corchetes que NO haga
    match exacto con CITATION_RE (ej. una referencia bibliográfica
    real como "[1]" o una aclaración entre corchetes) permanece intacto
    y se sigue contando -- solo el patrón exacto de cita estructurada
    se excluye. Números científicos normales (porcentajes, decimales,
    años) nunca están dentro de una cita real, así que nunca se ven
    afectados.

    Reutiliza count_words tal cual (sin duplicar su regex) sobre el
    texto ya despojado de citas -- misma semántica de conteo de
    palabras en ambos casos, la única diferencia es qué texto de
    entrada reciben."""

    return count_words(CITATION_RE.sub("", safe_str(text)))


def number_exists_in_text(value, text):
    token = safe_str(value).replace(",", ".")
    return token in safe_str(text).replace(",", ".")


def validate_generated_section(generated, section, evidence):
    errors = []
    citation_errors = []
    claim_errors = []
    numeric_errors = []
    allowed = {(r["source_filename"], r["chunk_id"]): r.get("text", "") for r in evidence}
    if not isinstance(generated, dict):
        return {"validation_ok": False, "errors": ["section_output_not_object"], "citation_errors": [], "claim_errors": [], "numeric_errors": [], "valid_citation_count": 0, "substantive_sentence_count": 0}
    if safe_str(generated.get("section_id")) != safe_str(section.get("section_id")):
        errors.append("SECTION_ID_MISMATCH")
    if not safe_str(generated.get("section_title")):
        errors.append("MISSING_SECTION_TITLE")
    text = safe_str(generated.get("draft_text"))
    claims = generated.get("claims")
    if not text:
        errors.append("EMPTY_DRAFT_TEXT")
    if not isinstance(claims, list):
        errors.append("INVALID_CLAIMS")
        claims = []
    sentences = split_sentences_preserving_citations(text)
    substantive = [s for s in sentences if is_substantive_sentence(s)]
    claim_map = {}
    for claim in claims:
        if not isinstance(claim, dict):
            claim_errors.append("claim_not_object")
            continue
        key = normalize_claim_text(claim.get("claim"))
        if not key:
            claim_errors.append("empty_claim")
            continue
        if key in claim_map:
            claim_errors.append("duplicate_claim_text")
        claim_map[key] = claim
    for sentence in substantive:
        pairs = [(a.strip(), b.strip()) for a, b in CITATION_RE.findall(sentence)]
        if not pairs:
            citation_errors.append("uncited_substantive_sentence")
        for pair in pairs:
            if pair not in allowed:
                citation_errors.append("invalid_citation")
        claim = claim_map.get(normalize_claim_text(sentence))
        if not claim:
            claim_errors.append("missing_claim_for_sentence")
            continue
        claim_pairs = []
        for value in claim.get("supporting_citations") or []:
            match = CITATION_RE.fullmatch(safe_str(value))
            if match:
                claim_pairs.append((match.group(1).strip(), match.group(2).strip()))
        if set(claim_pairs) != set(pairs):
            claim_errors.append("claim_citation_mismatch")
        numeric_errors.extend(
            compute_unsupported_numeric_values(normalize_claim_text(sentence), pairs, allowed, set())
        )
    all_errors = errors + citation_errors + claim_errors
    return {
        "validation_ok": not all_errors and not numeric_errors,
        "errors": sorted(set(errors)),
        "citation_errors": sorted(set(citation_errors)),
        "claim_errors": sorted(set(claim_errors)),
        "numeric_errors": sorted(set(numeric_errors)),
        "valid_citation_count": sum(1 for s in sentences for pair in CITATION_RE.findall(s) if (pair[0].strip(), pair[1].strip()) in allowed),
        "substantive_sentence_count": len(substantive),
    }


def section_allows_no_sources(section):
    """Reutiliza classify_section_source_requirement (``src/tools/
    shared/section_source_requirement.py``), la MISMA fuente que
    consume Stage 05 (``section_allows_empty_papers``), para que una
    sección aprobada por 05 como source-free nunca pueda ser
    rechazada aquí por una definición diferente."""
    return section_is_source_free_organizational(section)


def build_draft_reports(sections, outline_sections, evidence_map, policy):
    budgets = policy.get("section_budgets") or assign_section_budgets(outline_sections, policy["target_total_words"])
    quality_rows = []
    section_rows = []
    claim_evidence_rows = []
    numeric_rows = []
    sections_without_valid_citations = []
    sections_with_low_citation_density = []
    sections_with_claim_support_errors = []
    sections_with_quantitative_support_errors = []
    invalid_citation_count = 0
    for section in sections:
        sid = safe_str(section.get("section_id"))
        title = safe_str(section.get("section_title"))
        text = safe_str(section.get("draft_text"))
        evidence = evidence_map.get(sid, [])
        outline = next((item for item in outline_sections if safe_str(item.get("section_id")) == sid), {"section_id": sid})
        validation = section.get("section_validation") or validate_generated_section(section, outline, evidence)
        claims = section.get("claims") if isinstance(section.get("claims"), list) else []
        citation_pairs = [(a.strip(), b.strip()) for a, b in CITATION_RE.findall(text)]
        allowed_pairs = {(r["source_filename"], r["chunk_id"]) for r in evidence}
        valid_pairs = [pair for pair in citation_pairs if pair in allowed_pairs]
        invalid_pairs = [pair for pair in citation_pairs if pair not in allowed_pairs]
        invalid_citation_count += len(invalid_pairs)
        if evidence and not valid_pairs:
            sections_without_valid_citations.append(sid)
        substantive = [s for s in split_sentences_preserving_citations(text) if is_substantive_sentence(s)]
        uncited = [s for s in substantive if not CITATION_RE.search(s)]
        if evidence and uncited:
            sections_with_low_citation_density.append({"section_id": sid, "uncited_sentences": uncited})
        if validation.get("claim_errors"):
            sections_with_claim_support_errors.append({"section_id": sid, "errors": validation["claim_errors"]})
        if validation.get("numeric_errors"):
            sections_with_quantitative_support_errors.append({"section_id": sid, "errors": validation["numeric_errors"]})
        lookup = {(r["source_filename"], r["chunk_id"]): safe_str(r.get("text")) for r in evidence}
        for idx, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                continue
            claim_id = f"{sid}_C{idx}"
            claim_text = safe_str(claim.get("claim"))
            parsed = []
            for citation in claim.get("supporting_citations") or []:
                match = CITATION_RE.fullmatch(safe_str(citation))
                if match:
                    parsed.append((match.group(1).strip(), match.group(2).strip()))
            for rank, pair in enumerate(parsed, start=1):
                claim_evidence_rows.append({"section_id": sid, "claim_id": claim_id, "claim_text": claim_text, "source_filename": pair[0], "chunk_id": pair[1], "rank": rank, "retrieval_method": "supporting_citation_from_draft", "evidence_text": lookup.get(pair, "")[:int(policy["max_evidence_chars"])], "allowed_for_section": pair in allowed_pairs})
            for numeric_value in re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?", claim_text):
                found = [pair for pair in parsed if number_exists_in_text(numeric_value, lookup.get(pair, ""))]
                numeric_rows.append({"section_id": sid, "claim_id": claim_id, "claim_text": claim_text, "numeric_value": numeric_value, "found_in_cited_chunks": bool(found), "matching_citations": "; ".join(f"[{a} | {b}]" for a, b in found), "risk": "none" if found else "high"})
        word_count = count_content_words(text)
        budget = budgets[sid]
        source_free = bool((section.get("section_validation") or {}).get("source_free_organizational_section", False))
        quality_rows.append({"section_id": sid, "section_title": title, "word_count": word_count, "source_free_organizational_section": source_free, "citation_count": len(citation_pairs), "valid_citation_count": len(valid_pairs), "invalid_citation_count": len(invalid_pairs), "claim_count": len(claims), "substantive_sentence_count": len(substantive), "uncited_substantive_sentence_count": len(uncited), "section_validation_ok": bool(validation.get("validation_ok"))})
        section_rows.append({"section_id": sid, "section_title": title, "draft_text": text, "word_count": word_count, "target_words": budget["target_words"], "minimum_words": budget["minimum_words"], "maximum_words": budget["maximum_words"], "source_free_organizational_section": source_free, "within_section_range": True if source_free else budget["minimum_words"] <= word_count <= budget["maximum_words"], "citation_count": len(citation_pairs), "claim_count": len(claims)})
    total_words = sum(row["word_count"] for row in section_rows)
    source_free_count = sum(1 for row in section_rows if row["source_free_organizational_section"])
    target_total = int(policy["target_total_words"])
    configured_min = int(policy["min_total_words"])
    max_total = int(policy["max_total_words"])
    # effective_min_total_words: métrica DIAGNÓSTICA únicamente -- nunca
    # participa en el gate de aprobación. Antes de esta corrección,
    # global_length_valid usaba directamente effective_min, permitiendo
    # que un borrador muy por debajo de configured_min_total_words
    # (ej. 1081 con configured_min=1300) se aprobara con validation_ok=
    # True solo porque tenía suficientes secciones source-free
    # organizacionales. configured_min_total_words es AHORA el único
    # mínimo contractual real -- el mismo valor que el generation_
    # profile declaró, sin rebajarse silenciosamente.
    effective_min = max(1, configured_min - source_free_count * max(0, int(target_total / max(len(sections), 1)) - 40))
    word_deficit = max(0, configured_min - total_words)
    word_excess = max(0, total_words - max_total)
    word_count_compliant = configured_min <= total_words <= max_total
    global_length_valid = word_count_compliant
    all_section_validations_ok = all(bool(row["section_validation_ok"]) for row in quality_rows)
    numeric_failures = sum(1 for row in numeric_rows if not row["found_in_cited_chunks"])
    sections_outside_word_range = [row['section_id'] for row in section_rows if not row['within_section_range']]
    validation_ok = all_section_validations_ok and invalid_citation_count == 0 and not sections_without_valid_citations and not sections_with_low_citation_density and not sections_with_claim_support_errors and not sections_with_quantitative_support_errors and numeric_failures == 0 and global_length_valid
    report = {"validation_ok": validation_ok, "invalid_citation_count": invalid_citation_count, "sections_without_valid_citations": sections_without_valid_citations, "sections_with_low_citation_density": sections_with_low_citation_density, "sections_with_claim_support_errors": sections_with_claim_support_errors, "sections_with_quantitative_support_errors": sections_with_quantitative_support_errors, "numeric_failure_count": numeric_failures, "total_words": total_words, "actual_total_words": total_words, "target_total_words": target_total, "configured_min_total_words": configured_min, "effective_min_total_words": effective_min, "configured_max_total_words": max_total, "max_total_words": max_total, "word_deficit": word_deficit, "word_excess": word_excess, "word_count_compliant": word_count_compliant, "source_free_organizational_section_count": source_free_count, "global_length_valid": global_length_valid, "section_count": len(sections), "all_section_validations_ok": all_section_validations_ok, "open_search_used": False, "ground_truth_used": False, "sections_outside_word_range": sections_outside_word_range}
    return report, quality_rows, section_rows, claim_evidence_rows, numeric_rows


def validate_draft_global(sections, outline_sections=None, evidence_map=None, policy=None):
    if outline_sections is None or evidence_map is None or policy is None:
        bad = [s.get("section_id") for s in sections if not (s.get("section_validation") or {}).get("validation_ok")]
        return {"validation_ok": not bad, "invalid_sections": bad, "section_count": len(sections)}
    report, _, _, _, _ = build_draft_reports(sections, outline_sections, evidence_map, policy)
    return report


# --- NUMERIC_NORMALIZATION_FIX_V1 ---
# Wrapper conservador sobre el validador original.
# Solo elimina UNSUPPORTED_NUMERIC_VALUE cuando el mismo valor
# aparece realmente en la evidencia entregada a 06.

_validate_generated_section_original = validate_generated_section


def _canonical_numeric_tokens(text):
    import re

    value = str(text or "")

    # Normaliza decimal comma y espacios alrededor del %
    value = value.replace(",", ".")

    tokens = set()

    # 42.9%, 42.9 %, 500, 0.001, etc.
    for match in re.finditer(
        r"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*%?",
        value,
    ):
        token = match.group(0).strip()
        token = re.sub(r"\s+", "", token)

        if token:
            tokens.add(token)

    # Formato invertido poco habitual: %42.9
    for match in re.finditer(
        r"%\s*[-+]?\d+(?:\.\d+)?",
        value,
    ):
        token = match.group(0)
        token = token.replace("%", "").strip()
        token = re.sub(r"\s+", "", token)

        if token:
            tokens.add(token + "%")

    return tokens


def validate_generated_section(
    generated,
    section,
    evidence,
):
    result = _validate_generated_section_original(
        generated,
        section,
        evidence,
    )

    if not isinstance(result, dict):
        return result

    numeric_errors = list(
        result.get("numeric_errors") or []
    )

    if not numeric_errors:
        return result

    evidence_tokens = build_section_evidence_numeric_tokens(evidence)

    retained_numeric_errors = []

    for error in numeric_errors:

        prefix = "UNSUPPORTED_NUMERIC_VALUE:"

        if not str(error).startswith(prefix):
            retained_numeric_errors.append(error)
            continue

        raw_number = str(error)[len(prefix):].strip()

        target_tokens = _canonical_numeric_tokens(
            raw_number
        )

        # Solo se perdona el error si el valor está realmente
        # presente en la evidencia usada por esta sección.
        if target_tokens & evidence_tokens:
            continue

        retained_numeric_errors.append(error)

    old_numeric = set(numeric_errors)
    removed_numeric = old_numeric - set(
        retained_numeric_errors
    )

    result["numeric_errors"] = retained_numeric_errors

    # validation_errors/errors pueden contener el mismo código.
    for key in ("errors", "validation_errors"):
        if isinstance(result.get(key), list):
            result[key] = [
                item
                for item in result[key]
                if item not in removed_numeric
            ]

    result["validation_ok"] = not any(
        [
            result.get("errors"),
            result.get("validation_errors"),
            result.get("citation_errors"),
            result.get("claim_errors"),
            result.get("numeric_errors"),
        ]
    )

    return result
