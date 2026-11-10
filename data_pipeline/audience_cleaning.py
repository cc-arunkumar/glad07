import re
import pandas as pd
from config import config
from data_pipeline.common_cleaning import normalize_table,parse_number
from config.config import WEEKDAY_MAPPING,PREVIOUS_WEEKDAY

def clean_audience_dataset(input_path, output_path):
 raw=pd.read_csv(input_path,dtype=str,keep_default_na=False,na_values=[''])
 for field in config.AUDIENCE_COLUMNS_IN_THOUSANDS:
  if field in raw:
   for val in raw[field].dropna():
    text=str(val).strip()
    if text.lower() in [t.lower() for t in config.NULL_TOKENS]:continue
    if not re.fullmatch(r'-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:thousand|k|m)?',text,re.I):raise ValueError('Invalid audience count')
 out=normalize_table(raw,config.AUDIENCE_VALUE_MAPPINGS,config.AUDIENCE_WHOLE_NUMBER_PATTERNS,{},config.AUDIENCE_RANGE_LIMITS)
 for col in config.AUDIENCE_COLUMNS_IN_THOUSANDS:
  if col in raw:
   out[col]=raw[col].map(lambda value:parse_number(value,'audience'))
   out.loc[pd.to_numeric(out[col],errors='coerce')<0,col]=None
 if config.AUDIENCE_TARGET_COLUMN in raw:
  out[config.AUDIENCE_TARGET_COLUMN]=raw[config.AUDIENCE_TARGET_COLUMN].map(lambda value:parse_number(value,'audience'))
  out.loc[pd.to_numeric(out[config.AUDIENCE_TARGET_COLUMN],errors='coerce')<0,config.AUDIENCE_TARGET_COLUMN]=None
 days=[];starts=[];ends=[]
 for day,start,end,system in zip(raw[config.TRANSMISSION_DAY_COLUMN],raw[config.START_TIME_COLUMN],raw[config.END_TIME_COLUMN],raw[config.SCHEDULING_SYSTEM_COLUMN]):
  sysname=str(system).strip().lower();known={v.lower() for v in config.SCHEDULING_SYSTEM_MAPPING.values()}
  if sysname not in known:raise ValueError('Unknown or missing scheduling system')
  sm=re.fullmatch(r'(\d{1,2}):(\d{2})',str(start).strip());em=re.fullmatch(r'(\d{1,2}):(\d{2})',str(end).strip())
  if not sm or not em:raise ValueError('Invalid scheduler clock time')
  sh,mi=map(int,sm.groups());eh,ei=map(int,em.groups())
  if sysname in {v.lower() for v in config.BROADCAST_CLOCK_SYSTEMS} and eh*60+ei <= sh*60+mi: raise ValueError('Broadcast programme ends before it starts')
  if sysname in {v.lower() for v in config.BROADCAST_CLOCK_SYSTEMS} and (sh<6 or eh<6):raise ValueError('Invalid broadcast clock time')
  if sysname not in {v.lower() for v in config.BROADCAST_CLOCK_SYSTEMS} and (sh>=24 or eh>=24):raise ValueError('Invalid calendar clock time')
  weekday=WEEKDAY_MAPPING.get(str(day).strip().lower())
  if not weekday:raise ValueError('Unknown transmission weekday')
  broadcast=weekday
  if sysname not in {v.lower() for v in config.BROADCAST_CLOCK_SYSTEMS}:
   if sh<6:
    broadcast=PREVIOUS_WEEKDAY[weekday];sh+=24
    if eh<6 or (eh==6 and ei==0):eh+=24
   elif (eh*60+ei)<(sh*60+mi) and eh<6:
    eh+=24
  days.append(broadcast);starts.append(f'{sh:02d}:{mi:02d}');ends.append(f'{eh:02d}:{ei:02d}')
 out[config.TRANSMISSION_DAY_COLUMN]=days;out[config.START_TIME_COLUMN]=starts;out[config.END_TIME_COLUMN]=ends
 if 'airing_id' in out:out=out.drop_duplicates(subset=['airing_id'],keep='first')
 out=out[out[config.AUDIENCE_TARGET_COLUMN].notna()].copy()
 out.to_csv(output_path,index=False);return out
