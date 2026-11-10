import numpy as np,pandas as pd
from config import config
from data_pipeline.feature_builder import build_feature_frame

def build_amendment_facts(model,schema_or_rows,rows=None):
 if rows is None: schema=None; data=schema_or_rows
 else:schema=schema_or_rows;data=rows
 df=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data if isinstance(data,list) else [data]);features=build_feature_frame(df,config.AMENDMENT_NUMERIC_FEATURES,config.AMENDMENT_CATEGORICAL_FEATURES)
 if schema:features=features[list(schema)]
 probs=model.predict_proba(features)[:,1];facts=[]
 for record,prob in zip(df.to_dict('records'),probs):
  score=float(prob);band=config.AMENDMENT_BAND_HIGH if score>=config.AMENDMENT_BAND_HIGH_THRESHOLD else config.AMENDMENT_BAND_MEDIUM if score>=config.AMENDMENT_BAND_MEDIUM_THRESHOLD else config.AMENDMENT_BAND_LOW
  facts.append({'campaign_reference':record.get(config.CAMPAIGN_REFERENCE_COLUMN),'submission_id':record.get('submission_id'),'amendment_score':round(score,config.AMENDMENT_SCORE_DECIMALS),'amendment_band':band})
 return facts

def build_exposure_facts(model,airings):
 df=pd.DataFrame(airings if isinstance(airings,list) else [airings]);X=build_feature_frame(df,config.EXPOSURE_NUMERIC_FEATURES,config.EXPOSURE_CATEGORICAL_FEATURES);values=model.predict(X);return [{'airing_id':r.get('airing_id'),'forecast_under18_viewers':round(float(v),config.EXPOSURE_DECIMALS)} for r,v in zip(df.to_dict('records'),values)]
def build_issue_facts(model,notes):
 from collections import defaultdict
 if not notes:raise ValueError('No campaign notes')
 votes=defaultdict(float);confidence=defaultdict(float);read=0;short=0;readings=[]
 for n in notes:
  text=str(n.get('note_text') or '').strip();source=n.get('note_source');weight=config.NOTE_SOURCE_WEIGHTS.get(source)
  if weight is None:raise ValueError('Unknown note source weight')
  if len(text.split())<config.MINIMUM_NOTE_WORDS:short+=1;continue
  supplied=n.get('_predicted_probabilities')
  if supplied is not None:
   probabilities=np.asarray([supplied.get(str(label),0.0) for label in model.classes_],
                            dtype=float)
  elif hasattr(model,'predict_proba'):
   probabilities=model.predict_proba([text])[0]
  else:
   label=str(model.predict([text])[0])
   probabilities=None
  if probabilities is not None:
   confidence_by_label={str(label):float(probability)
                        for label,probability in zip(model.classes_,probabilities)}
   label=max(confidence_by_label,key=confidence_by_label.get)
   for candidate,score in confidence_by_label.items():
    confidence[candidate]+=score*float(weight)
  else:
   confidence[label]+=float(weight)
  label=str(label)
  votes[label]+=float(weight);read+=1
  readings.append({'note_sequence':n.get(config.NOTE_SEQUENCE_COLUMN),
                   'issue_category':label,'weight':float(weight)})
 if not votes:raise ValueError('No readable notes')
 ordered=sorted(votes.items(),key=lambda p:(-p[1],-confidence[p[0]],p[0]))
 agreement=round(ordered[0][1]/sum(votes.values()),config.NOTE_AGREEMENT_DECIMALS)
 return {'issue_category':ordered[0][0],'weighted_votes':dict(votes),
         'weighted_agreement':agreement,'issue_agreement':agreement,'notes_read':read,
         'notes_too_short_to_read':short,'note_readings':readings}
