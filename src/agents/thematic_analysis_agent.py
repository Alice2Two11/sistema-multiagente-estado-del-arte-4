from __future__ import annotations
from datetime import datetime,timezone
import json
from src.contracts.agent_result import *
from src.tools.thematic_analysis import *
class ThematicAnalysisAgent:
    def __init__(self,dependencies): self.dependencies=dependencies
    def execute(self,agent_input):
        start=datetime.now(timezone.utc).isoformat(); llm_calls=0
        try:
            if agent_input.stage_name!='04_agente_analisis_tematico': raise ValueError('INVALID_CONFIGURATION')
            if agent_input.attempt_number not in (1,2): raise ValueError('INVALID_CONFIGURATION')
            bundle,df,m03=validate_dependencies(agent_input); final,excluded=filter_corpus(df); final,qmeta,qwarn=integrate_quantitative_context(final,bundle)
            context=compact_for_thematic_analysis(final,int(agent_input.policy.get('max_field_chars',3500))); valid=set(final.source_filename.astype(str)); title_map=dict(zip(final.source_filename.astype(str),final.title.astype(str)))
            repair_plan=build_repair_plan(agent_input.previous_attempt.failure_reason_codes if agent_input.previous_attempt else ()) if agent_input.attempt_number==2 else None
            min_s=agent_input.policy.get('min_sections'); max_s=agent_input.policy.get('max_sections')
            enforce_section_count=bool(agent_input.policy.get('structure_policy',{}).get('enforce_section_count',False))
            prompt=self.dependencies.build_prompt(context,list(valid),title_map,repair_plan,min_sections=min_s,max_sections=max_s,enforce_section_count=enforce_section_count); raw=self.dependencies.invoke(prompt); llm_calls=1
            payload=self.dependencies.parse(raw); raw_counts=inspect_thematic_payload(payload); data,schema_issues,alias_repairs=normalize_thematic_output(payload,return_repairs=True); data,deterministic_repairs=apply_deterministic_repairs(data,title_map,valid); repairs=alias_repairs+deterministic_repairs
            ref_codes,counts,_=validate_references(data,final); table_counts=thematic_table_counts(data); flattening_codes,consistency=validate_json_to_tables(raw_counts,table_counts); codes=[x['code'] for x in schema_issues]+ref_codes+flattening_codes
            if any(r.get('type')=='INVALID_REFERENCE_REMOVED' for r in repairs): codes.append('INVALID_REPRESENTATIVE_SOURCE')
            if not data['themes']: codes.append('EMPTY_THEMATIC_OUTPUT')
            if not data['research_gaps']: codes.append('EMPTY_THEMATIC_OUTPUT')
            if not data['comparative_dimensions']: codes.append('EMPTY_THEMATIC_OUTPUT')
            sc=len(data['suggested_state_of_art_structure'])
            structure_too_short=min_s is not None and sc<int(min_s); structure_too_long=max_s is not None and sc>int(max_s)
            # El conteo de secciones es orientativo por defecto (contrato
            # estructural: la longitud en palabras, validada en etapas
            # posteriores, es la restricción dura -- no el número de
            # secciones). structure_too_short/long SIEMPRE se calculan como
            # diagnóstico (ver calculate_diagnostic_metrics más abajo), pero
            # solo entran a `codes` -- y por tanto a failure_reason_codes,
            # classify_quality, build_repair_plan y validation_ok -- cuando
            # structure_policy.enforce_section_count=True conserva
            # exactamente el comportamiento histórico bloqueante.
            if enforce_section_count:
                if structure_too_short: codes.append('STRUCTURE_TOO_SHORT')
                if structure_too_long: codes.append('STRUCTURE_TOO_LONG')
            metrics=calculate_diagnostic_metrics(data,final,counts,min_sections=min_s,max_sections=max_s,enforce_section_count=enforce_section_count)
            if repair_plan and not repairs: codes.append('REPAIR_PLAN_NOT_APPLIED')
            codes=tuple(dict.fromkeys(codes)); quality,action=classify_quality(codes,agent_input.attempt_number,bool(agent_input.policy.get('manual_review_policy',{}).get('allowed',True)))
            validation={'validation_ok':not codes,'failure_reason_codes':list(codes),'metrics':metrics,'repairs':repairs,'repair_plan':repair_plan or [],'json_to_tables_consistency':consistency,**qmeta}
            manifest={'stage':agent_input.stage_name,'experiment_id':agent_input.experiment_id,'run_id':agent_input.run_id,'attempt_number':agent_input.attempt_number,'quality_status':quality.value,'safety_policy':{'uses_ground_truth':False,'uses_external_knowledge':False,'uses_review_sections':False,'uses_bibliography':False},'diagnostic_metrics':metrics,'quantitative_context':qmeta}
            artifacts=write_thematic_artifacts(agent_input.agent_context.output_directory,final,excluded,raw if isinstance(raw,str) else json.dumps(raw,ensure_ascii=False),data,validation,manifest)
            warnings=tuple(AgentWarning(code=c,severity=WarningSeverity.WARNING,blocking=codes!=(),message=c) for c in tuple(qwarn)+codes)
            transition=RequestedTransition(action=TransitionAction(action),target_stage=None,reason_code=quality.value,requires_human_confirmation=quality==QualityStatus.APPROVED_PENDING_MANUAL_REVIEW)
            return AgentResult(execution_status=ExecutionStatus.COMPLETED,quality_status=quality,decision=DecisionInfo(code='THEMATIC_ANALYSIS_EVALUATED',rationale='Análisis temático validado contra corpus cerrado.'),quality_metrics={'scientific':metrics,'technical':qmeta},warnings=warnings,failure_reason_codes=codes,requested_transition=transition,output_artifacts=artifacts,tool_usage=ToolUsage(llm_calls=llm_calls,validation_calls=1),attempt_number=agent_input.attempt_number,started_at=start,completed_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            msg=str(exc); known=[c for c in ['KB_NOT_FOUND','KB_JSONL_NOT_FOUND','SCIENTIFIC_EXTRACTION_MANIFEST_NOT_FOUND','SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH','DEPENDENCY_HASH_MISMATCH','INVALID_KB_SCHEMA','EMPTY_THEMATIC_CORPUS','QUANTITATIVE_MANIFEST_NOT_FOUND','QUANTITATIVE_MANIFEST_MISMATCH','QUANTITATIVE_ARTIFACT_NOT_FOUND','INVALID_QUANTITATIVE_CONTEXT','GROUND_TRUTH_POLICY_VIOLATION','CREDENTIAL_NOT_FOUND','INVALID_CONFIGURATION'] if c in msg]
            code=known[0] if known else ('DEPENDENCY_NOT_FOUND' if isinstance(exc,FileNotFoundError) else 'RUNTIME_DEPENDENCY_FAILED')
            return AgentResult(execution_status=ExecutionStatus.FAILED,quality_status=QualityStatus.REJECTED,decision=DecisionInfo(code='THEMATIC_ANALYSIS_FAILED',rationale='Falló la ejecución de 04.'),quality_metrics={'scientific':{},'technical':{}},warnings=(AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=msg),),failure_reason_codes=(code,),requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE,target_stage=None,reason_code=code,requires_human_confirmation=False),output_artifacts={},tool_usage=ToolUsage(llm_calls=llm_calls),attempt_number=agent_input.attempt_number,started_at=start,completed_at=datetime.now(timezone.utc).isoformat(),error={'type':type(exc).__name__,'message':msg,'stage':agent_input.stage_name})
