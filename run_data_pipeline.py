import os,json,pandas as pd
from config import config
from data_pipeline.submission_cleaning import clean_submission_dataset
from data_pipeline.audience_cleaning import clean_audience_dataset
from data_pipeline.note_cleaning import clean_note_dataset
from data_pipeline.quality_profiler import profile_dataset,write_report
from data_pipeline.dataset_loader import load_raw_dataset
def main():
 os.makedirs('processed',exist_ok=True); os.makedirs('output',exist_ok=True)
 tasks=[(config.RAW_SUBMISSIONS_TRAIN_FILE,config.CLEAN_SUBMISSIONS_TRAIN_FILE,clean_submission_dataset,config.SUBMISSION_TARGET_COLUMN),(config.RAW_SUBMISSIONS_EVAL_FILE,config.CLEAN_SUBMISSIONS_EVAL_FILE,clean_submission_dataset,config.SUBMISSION_TARGET_COLUMN),(config.RAW_AUDIENCES_TRAIN_FILE,config.CLEAN_AUDIENCES_TRAIN_FILE,clean_audience_dataset,config.AUDIENCE_TARGET_COLUMN),(config.RAW_AUDIENCES_EVAL_FILE,config.CLEAN_AUDIENCES_EVAL_FILE,clean_audience_dataset,config.AUDIENCE_TARGET_COLUMN),(config.RAW_CLEARANCE_NOTES_FILE,config.CLEAN_CLEARANCE_NOTES_FILE,clean_note_dataset,config.NOTE_TARGET_COLUMN)]
 profiles={}
 for src,dst,fn,target in tasks:
  cleaned=fn(src,dst); raw=load_raw_dataset(src); profiles[__import__('pathlib').Path(src).stem]=profile_dataset(raw,cleaned,target)
 write_report(profiles)
if __name__=='__main__':main()
