from config import config
def _one(df,col,value):
 rows=df[df[col].astype(str)==str(value)]
 if len(rows)!=1:raise ValueError(f'Expected one row for {value}, found {len(rows)}')
 return rows.iloc[0].to_dict()
def fetch_campaign_profile(campaigns,campaign_reference):return _one(campaigns,config.CAMPAIGN_REFERENCE_COLUMN,campaign_reference)
def fetch_campaign_spots(spots,campaign_reference):
 rows=spots[spots[config.CAMPAIGN_REFERENCE_COLUMN].astype(str)==str(campaign_reference)]
 if rows.empty:raise ValueError('No booked spots for campaign')
 return rows.to_dict('records')
def fetch_campaign_notes(notes,campaign_reference):
 rows=notes[notes[config.CAMPAIGN_REFERENCE_COLUMN].astype(str)==str(campaign_reference)].copy()
 if rows.empty:raise ValueError('No notes for campaign')
 rows['_seq']=__import__('pandas').to_numeric(rows[config.NOTE_SEQUENCE_COLUMN],errors='coerce')
 if rows['_seq'].isna().any() or rows['_seq'].duplicated().any() or rows[config.NOTE_TEXT_COLUMN].isna().any() or (rows[config.NOTE_TEXT_COLUMN].astype(str).str.strip()=='').any():raise ValueError('Invalid note record sequence/text')
 rows=rows.sort_values('_seq');return rows.to_dict('records')
def fetch_campaign_schedules(schedule,spots):
 channels={config.CHANNEL_MAPPING.get(str(s.get(config.CHANNEL_COLUMN,'')).strip().lower(),
                                      str(s.get(config.CHANNEL_COLUMN,'')).strip())
           for s in spots}
 schedule_channels=schedule[config.CHANNEL_COLUMN].astype(str).map(
  lambda value:config.CHANNEL_MAPPING.get(value.strip().lower(),value.strip()))
 return schedule[schedule_channels.isin(channels)].to_dict('records')
