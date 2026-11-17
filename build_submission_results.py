#!/usr/bin/env python3
"""Build the assignment's exact 12-column results.csv from local request-level CSVs.
Run from the project root after all tests. No API calls are made.
"""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results'/'raw'
OUT=ROOT/'results.csv'
COLS=['request_id','scenario','source','start_time','first_token_ms','total_ms','status','retries','tokens_in','tokens_out','retrieval_setting','answer_grade']
files=[]
base=RAW/'load_experiments_results_batch3_corrected_20261113T075123Z.csv'
if base.exists(): files.append(base)
files += sorted(RAW.glob('official_profiles_*.csv'))
files += sorted(RAW.glob('answer_quality_*.csv'))
if not files:
    raise SystemExit(f'No request-level CSV files found under {RAW}')
frames=[]
for path in files:
    try:
        df=pd.read_csv(path)
    except Exception as exc:
        print(f'SKIP {path.name}: {exc}')
        continue
    if 'request_id' not in df.columns:
        print(f'SKIP {path.name}: missing request_id')
        continue
    frames.append(df)
    print(f'Loaded {len(df)} rows: {path.name}')
pf_path=RAW/'groq_live_preflight.json'
if pf_path.exists():
    pf=json.loads(pf_path.read_text())
    ts=pd.to_datetime(pf.get('timestamp'),utc=True).timestamp() if pf.get('timestamp') else ''
    frames.append(pd.DataFrame([{'request_id':'groq_live_preflight_20261113T090906Z','scenario':'real_service_preflight','source':'real','start_time':ts,'first_token_ms':pf.get('first_token_ms',''),'total_ms':pf.get('latency_ms',''),'status':'ok' if pf.get('success') else 'error','retries':0,'tokens_in':pf.get('tokens_in',''),'tokens_out':pf.get('tokens_out',''),'retrieval_setting':'not_applicable','answer_grade':''}]))
    print('Loaded 1 row: groq_live_preflight.json')
if not frames: raise SystemExit('No usable request-level CSVs')
df=pd.concat(frames,ignore_index=True,sort=False)
# Mark the first overload attempt as underloaded; the corrected run remains sustained_overload.
if 'run_batch_id' in df.columns:
    m=(df.scenario.astype(str)=='sustained_overload') & df.run_batch_id.astype(str).str.contains('090950Z')
    df.loc[m,'scenario']='sustained_overload_initial_underloaded'
# De-dupe identical request records only; repeated questions in separate runs have different IDs.
df=df.drop_duplicates(subset=['request_id'],keep='first')
for c in COLS:
    if c not in df.columns: df[c]=''
result=df[COLS].copy().fillna('')
allowed_status={'ok','rate_limited','timeout','error'}
allowed_grade={'','correct','wrong','refused_correctly','refused_wrongly'}
unknown_status=set(result.status.astype(str))-allowed_status
unknown_grade=set(result.answer_grade.astype(str))-allowed_grade
if unknown_status: raise SystemExit(f'Invalid statuses: {unknown_status}')
if unknown_grade: raise SystemExit(f'Invalid answer grades: {unknown_grade}')
if result.request_id.duplicated().any(): raise SystemExit('Duplicate request_id remains')
result.to_csv(OUT,index=False)
print(f'Wrote {len(result)} unique request rows to {OUT}')
print(result.groupby(['scenario','source','status']).size().to_string())
