from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
from src.contracts.agent_result import *
from src.tools.outline_generation import *
class OutlineGenerationAgent:
 def __init__(self,runtime): self.runtime=runtime
 def execute(self,agent_input):
  start=datetime.now(timezone.utc).isoformat();llm_calls=0
  try:
   bundle=validate_outline_dependencies(agent_input);ctx=build_outline_context(bundle,agent_input);valid=set(ctx['valid_source_filenames']);tm=dict(zip(bundle['kb'].source_filename.astype(str),bundle['kb'].title.astype(str)));title_to_source={v:k for k,v in tm.items()};out=Path(agent_input.agent_context.output_directory);manifest_path=out/'outline_generation_manifest.json';reuse=False;raw=''
   if manifest_path.exists() and not agent_input.policy['force_rebuild']:
    try:
     pm=json.loads(manifest_path.read_text())
     # Nunca reutilizar una salida previa cuyo propio validation_report
     # no fue APPROVED (validation_ok=True) -- de lo contrario un RETRY
     # tras un HALT_STAGE científico simplemente relee del disco el
     # MISMO outline inválido, lo revalida, obtiene el MISMO fallo, y
     # nunca vuelve a intentar generar/reparar nada. El fingerprint
     # coincidiendo solo demuestra que las entradas upstream (03/03B/04)
     # no cambiaron -- no demuestra que la salida anterior sea válida.
     reuse=(
      pm.get('fingerprint')==agent_input.policy.get('current_fingerprint')
      and all((out/n).exists() for n in NAMES)
      and bool(pm.get('validation_report',{}).get('validation_ok'))
     )
    except Exception:reuse=False
   if reuse:
    outline=json.loads((out/'state_of_art_outline.json').read_text());raw=(out/'state_of_art_outline_raw.txt').read_text()
    if not isinstance(outline,dict):reuse=False
   if not reuse:
    raw=self.runtime.invoke(build_outline_generation_prompt(ctx));llm_calls=1;outline=self.runtime.parse(raw)
   if not isinstance(outline,dict):raise ValueError('INVALID_LLM_OUTPUT')
   sr,us=repair_outline_sources(outline,valid,tm,title_to_source,float(agent_input.policy.get('title_match_cutoff',0.55)));cr,uc=repair_coverage_summary(outline,valid,tm,title_to_source,float(agent_input.policy.get('title_match_cutoff',0.55)))
   # Reparación determinista de secciones no exentas que quedaron con
   # papers_to_use=[] -- usa EXCLUSIVAMENTE los temas/papers
   # representativos ya producidos por 04_agente_analisis_tematico
   # (bundle['thematic']['themes']), nunca texto libre ni el KB
   # completo sin pasar por ese mapping. Si no hay evidencia upstream
   # real para una sección, queda sin reparar -- la validación
   # posterior la marcará como problemática, correctamente (fail-closed).
   themes=bundle['thematic'].get('themes',[]) if isinstance(bundle.get('thematic'),dict) else []
   section_papers_repairs,sections_without_upstream_evidence=repair_empty_section_papers(outline,themes,valid)
   validation=validate_outline(outline,valid,int(agent_input.policy.get('min_sections',4)),int(agent_input.policy.get('max_sections',5)),sr,us,cr,uc);validation.update({'experiment_id':agent_input.experiment_id,'validation_version':agent_input.policy.get('validation_version'),'section_papers_repairs':section_papers_repairs,'sections_without_upstream_evidence':sections_without_upstream_evidence});codes=reason_codes(validation);quality=QualityStatus.APPROVED if validation['validation_ok'] else QualityStatus.NEEDS_REVISION
   if quality is QualityStatus.APPROVED:action=TransitionAction.ADVANCE
   elif agent_input.attempt_number==1:action=TransitionAction.RETRY
   else:action=TransitionAction.HALT_STAGE
   manifest={'stage':agent_input.stage_name,'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'attempt_number':agent_input.attempt_number,'fingerprint':agent_input.policy.get('current_fingerprint'),'automatic_decision':{'status':'outputs_are_current' if reuse else 'rebuild_executed','rebuild_executed':not reuse},'generation_constraints':{k:agent_input.policy.get(k) for k in ['length_profile','min_sections','max_sections','output_language','writing_mode','focus_mode','citation_style']},'safety_policy':{'uses_kb_final_for_thematic_analysis':True,'uses_full_kb_for_outline_generation':False,'uses_ground_truth':False,'uses_raw_pdfs':False,'external_knowledge_allowed':False,'source_filenames_validated_and_repaired':True},'validation_report':validation,'counts':{'sections':len(outline.get('sections',[])),'papers_available':len(valid),'papers_used':validation.get('papers_used_count',0)}}
   artifacts=write_outline_artifacts(out,outline,raw,validation,manifest);warn=tuple(AgentWarning(code=c,severity=WarningSeverity.WARNING,blocking=True,message=c) for c in codes)
   return AgentResult(execution_status=ExecutionStatus.COMPLETED,quality_status=quality,decision=DecisionInfo(code='OUTLINE_GENERATION_EVALUATED',rationale='Esquema generado y validado preservando comportamiento del notebook 05.'),quality_metrics={'scientific':{},'technical':{'validation_ok':validation['validation_ok'],'reused':reuse}},warnings=warn,failure_reason_codes=codes,requested_transition=RequestedTransition(action=action,target_stage=None,reason_code=quality.value,requires_human_confirmation=False),output_artifacts=artifacts,tool_usage=ToolUsage(llm_calls=llm_calls,validation_calls=1),attempt_number=agent_input.attempt_number,started_at=start,completed_at=datetime.now(timezone.utc).isoformat())
  except Exception as exc:
   msg=str(exc);known=['OUTLINE_INPUT_NOT_FOUND','INVALID_THEMATIC_ANALYSIS_INPUT','THEMATIC_MANIFEST_MISMATCH','DEPENDENCY_HASH_MISMATCH','EMPTY_OUTLINE_KB','INVALID_OUTLINE_KB_SCHEMA','INVALID_SOURCE_TITLE','GROUND_TRUTH_POLICY_VIOLATION','INVALID_CONFIGURATION','INVALID_LLM_OUTPUT'];code=next((x for x in known if x in msg),'RUNTIME_DEPENDENCY_FAILED')
   return AgentResult(execution_status=ExecutionStatus.FAILED,quality_status=QualityStatus.REJECTED,decision=DecisionInfo(code='OUTLINE_GENERATION_FAILED',rationale='Falló la ejecución del generador de esquema.'),quality_metrics={'scientific':{},'technical':{}},warnings=(AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=msg),),failure_reason_codes=(code,),requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE,target_stage=None,reason_code=code,requires_human_confirmation=False),output_artifacts={},tool_usage=ToolUsage(llm_calls=llm_calls),attempt_number=agent_input.attempt_number,started_at=start,completed_at=datetime.now(timezone.utc).isoformat(),error={'type':type(exc).__name__,'message':msg,'stage':agent_input.stage_name})
