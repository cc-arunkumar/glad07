import re
import pandas as pd
from config import config
from data_pipeline.common_cleaning import normalize_table,parse_spot_length

def clean_submission_dataset(input_path, output_path):
 raw=pd.read_csv(input_path,dtype=str,keep_default_na=False,na_values=[''])
 for field,pattern in {**config.SUBMISSION_WHOLE_NUMBER_PATTERNS,**config.SUBMISSION_DECIMAL_NUMBER_PATTERNS}.items():
  if field in raw:
   for value in raw[field].dropna():
    text=str(value).strip()
    if text.lower() in [v.lower() for v in config.NULL_TOKENS]: continue
    if field in ('spots_booked','channels_booked'):
     if not re.fullmatch(r'-?\d+(?:\s*(?:spots?|channels?))?',text,re.I): raise ValueError('Invalid numeric field '+field)
    elif not re.fullmatch(pattern,text): raise ValueError('Invalid numeric field '+field)
 out=normalize_table(raw,config.SUBMISSION_VALUE_MAPPINGS,config.SUBMISSION_WHOLE_NUMBER_PATTERNS,config.SUBMISSION_DECIMAL_NUMBER_PATTERNS,config.SUBMISSION_RANGE_LIMITS)
 if config.COPY_LENGTH_COLUMN in out:
  found=[]
  for clock,written in zip(raw[config.CLOCK_NUMBER_COLUMN],raw[config.COPY_LENGTH_COLUMN]):
   match=re.fullmatch(config.CLOCK_NUMBER_PATTERN,str(clock).strip())
   if not match:raise ValueError('Invalid or missing copy clock number')
   seconds=int(match.group(1));length=parse_spot_length(written)
   if seconds not in config.STANDARD_SPOT_LENGTHS:raise ValueError('Unsupported standard spot length')
   if length is None:raise ValueError('Unrecognized spot length')
   found.append(seconds if seconds!=length else length)
  out[config.COPY_LENGTH_COLUMN]=found
 # Repair impossible or self-contradictory operational readings by marking only that feature missing.
 if config.LEGAL_TEXT_SECONDS_COLUMN in out and config.COPY_LENGTH_COLUMN in out:
  bad=pd.to_numeric(out[config.LEGAL_TEXT_SECONDS_COLUMN],errors='coerce')>pd.to_numeric(out[config.COPY_LENGTH_COLUMN],errors='coerce')
  out.loc[bad,config.LEGAL_TEXT_SECONDS_COLUMN]=None
 if config.ADVERTISER_CAMPAIGNS_COLUMN in out and config.ADVERTISER_AMENDMENTS_COLUMN in out:
  bad=pd.to_numeric(out[config.ADVERTISER_AMENDMENTS_COLUMN],errors='coerce')>pd.to_numeric(out[config.ADVERTISER_CAMPAIGNS_COLUMN],errors='coerce')
  out.loc[bad,config.ADVERTISER_AMENDMENTS_COLUMN]=None
 if config.SUBMISSION_TARGET_COLUMN in out:out[config.SUBMISSION_TARGET_COLUMN]=out[config.SUBMISSION_TARGET_COLUMN].map(lambda x:config.BOOLEAN_MAPPING.get(str(x).lower()))
 if 'submission_id' in out:out=out.drop_duplicates(subset=['submission_id'],keep='first')
 out=out[out[config.SUBMISSION_TARGET_COLUMN].notna()].copy()
 out.to_csv(output_path,index=False);return out
