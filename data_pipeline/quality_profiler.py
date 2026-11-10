import pandas as pd,json,os
from config import config

def profile_dataset(raw,clean,target_column):
 idcol=next((c for c in ['submission_id','airing_id','note_id'] if c in raw.columns),None)
 dup=int(raw.duplicated(subset=[idcol] if idcol else None).sum())
 first=raw.drop_duplicates(subset=[idcol] if idcol else None,keep='first')
 missing_mask=first[target_column].map(lambda v: pd.isna(v) or str(v).strip().lower() in [t.lower() for t in config.NULL_TOKENS]) if target_column in first else pd.Series([],dtype=bool)
 if target_column==config.AUDIENCE_TARGET_COLUMN: missing_mask=missing_mask & ~(pd.to_numeric(first[target_column],errors='coerce')<0)
 missing_target=int(missing_mask.sum())
 entry={'rows_received':len(raw),'rows_kept':len(clean),'rows_removed':len(raw)-len(clean),'duplicate_rows':dup,'rows_missing_target':missing_target,'columns':len(raw.columns),'missing_cells_as_received':{k:int(v) for k,v in raw.isna().sum().items() if v},'missing_cells_after_cleaning':{k:int(v) for k,v in clean.isna().sum().items() if v}}
 if target_column==config.SUBMISSION_TARGET_COLUMN:
  l=pd.to_numeric(raw.get(config.LEGAL_TEXT_SECONDS_COLUMN),errors='coerce');c=pd.to_numeric(raw.get(config.COPY_LENGTH_COLUMN).map(lambda x:__import__('data_pipeline.common_cleaning',fromlist=['parse_spot_length']).parse_spot_length(x)),errors='coerce');a=pd.to_numeric(raw.get(config.ADVERTISER_AMENDMENTS_COLUMN),errors='coerce');b=pd.to_numeric(raw.get(config.ADVERTISER_CAMPAIGNS_COLUMN),errors='coerce')
  entry['legal_text_longer_than_copy_as_received']=int((l>c).sum());entry['legal_text_longer_than_copy_after_cleaning']=int((pd.to_numeric(clean[config.LEGAL_TEXT_SECONDS_COLUMN],errors='coerce')>pd.to_numeric(clean[config.COPY_LENGTH_COLUMN],errors='coerce')).sum());entry['amendments_exceeding_campaigns_as_received']=int((a>b).sum());entry['amendments_exceeding_campaigns_after_cleaning']=int((pd.to_numeric(clean[config.ADVERTISER_AMENDMENTS_COLUMN],errors='coerce')>pd.to_numeric(clean[config.ADVERTISER_CAMPAIGNS_COLUMN],errors='coerce')).sum())
 if target_column==config.AUDIENCE_TARGET_COLUMN:
  entry['rows_with_impossible_target']=int((pd.to_numeric(first[target_column],errors='coerce')<0).sum())
 if target_column==config.NOTE_TARGET_COLUMN:
  entry['rows_with_too_few_words']=int(((first[config.NOTE_TEXT_COLUMN].fillna('').astype(str).str.split().str.len()<config.MINIMUM_NOTE_WORDS) & ~first[target_column].map(lambda v: pd.isna(v) or str(v).strip().lower() in [t.lower() for t in config.NULL_TOKENS])).sum());from data_pipeline.common_cleaning import clean_text
  visible=raw[config.NOTE_TEXT_COLUMN].map(lambda value:clean_text(value) or '')
  entry['note_lengths_disagreeing_with_plain_text_as_received']=int((pd.to_numeric(raw[config.NOTE_LENGTH_COLUMN],errors='coerce')!=visible.str.len()).sum());entry['note_lengths_disagreeing_with_plain_text_after_cleaning']=int((pd.to_numeric(clean[config.NOTE_LENGTH_COLUMN],errors='coerce')!=clean[config.NOTE_TEXT_COLUMN].fillna('').astype(str).str.len()).sum())
 return entry

def write_report(profiles,path=config.DATA_QUALITY_REPORT_FILE):
 os.makedirs(os.path.dirname(path) or '.',exist_ok=True);json.dump(profiles,open(path,'w'),indent=2)
