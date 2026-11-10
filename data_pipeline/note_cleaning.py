import pandas as pd,re
from config import config
from data_pipeline.common_cleaning import clean_text,clean_mapped,parse_number
from config.config import WEEKDAY_MAPPING

def _clean_timestamp(value):
 text=clean_text(value)
 if not text:
  return None
 match=re.fullmatch(config.NOTE_TIMESTAMP_PATTERN,text,re.I)
 if not match:
  raise ValueError('Invalid note timestamp')
 weekday=WEEKDAY_MAPPING.get(match.group('note_weekday').lower())
 if not weekday:
  raise ValueError('Unknown note timestamp weekday')
 return f"{weekday} {match.group('note_time')}"

def clean_note_dataset(input_path, output_path):
 raw=pd.read_csv(input_path,dtype=str,keep_default_na=False,na_values=[""]); out=raw.copy()
 for c in out.columns:
  if c==config.NOTE_TIMESTAMP_COLUMN: out[c]=out[c].map(_clean_timestamp)
  elif c==config.NOTE_TEXT_COLUMN:
   out[c]=out[c].map(lambda x: (clean_text(x) or '').lower()
                     if (clean_text(x) or '').isupper() else (clean_text(x) or ''))
  elif c in config.NOTE_VALUE_MAPPINGS: out[c]=out[c].map(lambda x:clean_mapped(x,config.NOTE_VALUE_MAPPINGS[c]))
  elif c in config.NOTE_WHOLE_NUMBER_PATTERNS: out[c]=out[c].map(parse_number)
  else: out[c]=out[c].map(clean_text)
 if 'note_length_chars' in out: out['note_length_chars']=out['note_text'].str.len()
 if 'note_id' in out: out=out.drop_duplicates(subset=['note_id'])
 out=out[out[config.NOTE_TEXT_COLUMN].fillna('').astype(str).str.split().str.len()>=config.MINIMUM_NOTE_WORDS]
 out=out[out[config.NOTE_TARGET_COLUMN].notna()]
 out.to_csv(output_path,index=False);return out
