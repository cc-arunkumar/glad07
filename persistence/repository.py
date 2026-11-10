import json
from persistence.database import session_factory
from persistence.models import UserAccount,ClearanceAssessment,ChatTurn

def _account_dict(user):
 if user is None:
  return None
 return {column.name:getattr(user,column.name) for column in UserAccount.__table__.columns}

def fetch_user_account(session_or_username,criterion=None):
 if hasattr(session_or_username,'query'):
  return _account_dict(session_or_username.query(UserAccount).filter(criterion).first())
 with session_factory() as session:
  return _account_dict(session.get(UserAccount,session_or_username))

def insert_chat_turn(session_or_username,turn=None,answer=None):
 if hasattr(session_or_username,'add'):
  session=session_or_username
  data=dict(turn or {})
 else:
  session=session_factory()
  data={'username':session_or_username,'question':turn,'answer':answer,
        'question_text':turn or '','answer_text':answer or ''}
 close=not hasattr(session_or_username,'add')
 try:
  if 'question_text' in data:
   data.setdefault('question',data['question_text'])
  if 'answer_text' in data:
   data.setdefault('answer',data['answer_text'])
  row=ChatTurn(**data)
  session.add(row);session.commit()
  return row
 finally:
  if close:session.close()

def fetch_assessment_records(session_or_campaign=None,campaign_reference=None):
 if hasattr(session_or_campaign,'query'):
  session=session_or_campaign
  criterion=campaign_reference
  close=False
 else:
  session=session_factory()
  criterion=None
  reference=session_or_campaign if campaign_reference is None else campaign_reference
  close=True
 try:
  query=session.query(ClearanceAssessment)
  if criterion is not None:query=query.filter(criterion)
  elif reference:query=query.filter_by(campaign_reference=reference)
  return query.order_by(ClearanceAssessment.changed_at.desc()).all()
 finally:
  if close:session.close()
