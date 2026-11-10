from langgraph.graph import StateGraph,END
from agentic.state import ClearanceAssessmentState
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

_checkpoint_savers={}

def _passthrough(state):
 return state

def _decision_route(state):
 decision=(state.get('controller_decision') or {}).get('decision')
 from config import config
 routes={
  config.DECISION_CLEAR:'draft_clearance_certificate',
  config.DECISION_CLEAR_WITH_RESTRICTIONS:'draft_restriction_notice',
  config.DECISION_REJECT:'record_clearance_outcome',
 }
 if decision not in routes:
  raise ValueError('Unknown controller decision')
 return routes[decision]

def build_workflow(checkpoint_db_path=None):
 g=StateGraph(ClearanceAssessmentState)
 nodes=[
  'load_campaign_context','plan_assessment','assess_copy_risk','identify_issues',
  'check_spot_schedule','compose_clearance_assessment','request_controller_decision',
  'draft_restriction_notice','draft_clearance_certificate','record_clearance_outcome',
 ]
 for name in nodes:
  g.add_node(name,_passthrough)
 g.set_entry_point('load_campaign_context')
 for source,target in zip(nodes[:6],nodes[1:6]):
  g.add_edge(source,target)
 g.add_edge('compose_clearance_assessment','request_controller_decision')
 g.add_conditional_edges(
  'request_controller_decision',_decision_route,
  {
   'draft_clearance_certificate':'draft_clearance_certificate',
   'draft_restriction_notice':'draft_restriction_notice',
   'record_clearance_outcome':'record_clearance_outcome',
  },
 )
 g.add_edge('draft_restriction_notice','draft_clearance_certificate')
 g.add_edge('draft_clearance_certificate','record_clearance_outcome')
 g.add_edge('record_clearance_outcome',END)
 checkpointer=None
 if checkpoint_db_path:
  path=os.path.abspath(checkpoint_db_path)
  os.makedirs(os.path.dirname(path),exist_ok=True)
  if path not in _checkpoint_savers:
   connection=sqlite3.connect(path,check_same_thread=False)
   saver=SqliteSaver(connection)
   saver.setup()
   _checkpoint_savers[path]=saver
  checkpointer=_checkpoint_savers[path]
 compiled=g.compile(checkpointer=checkpointer,interrupt_before=['request_controller_decision'])
 return compiled
