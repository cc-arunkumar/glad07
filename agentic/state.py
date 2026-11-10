from typing import TypedDict,Any
class ClearanceAssessmentState(TypedDict, total=False):
 run_id:str
 campaign_reference:str
 started_by:str
 thread_id:str
 profile:dict
 spots:list
 schedules:list
 notes:list
 amendment_facts:dict
 exposure_facts:list
 issue_facts:dict
 spot_checks:list
 summary:dict
 recommendation:str
 assessment_plan_text:str
 assessment_text:str
 restriction_notice:str
 clearance_certificate:str
 messages:list
 assessment_id:str
 status:str
 decision:str
 decision_reason:str
 controller_decision:dict
