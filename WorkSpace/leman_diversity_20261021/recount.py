"""Portable independent BIO recount; standard library only, no GPU or fitting."""
import argparse,json
from pathlib import Path
from evidence_core import read_bio,read_predictions,score,sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('/root/data/WorkSpace/data/test2.txt'))
    p.add_argument('--result-dir',type=Path,default=Path('/root/data/WorkSpace/output/leman_diversity_20261021/test'))
    a=p.parse_args();result=json.loads((a.result_dir/'RESULT.json').read_text())
    freeze=json.loads((a.result_dir.parent/'SELECTED_FREEZE.json').read_text())
    assert sha256(a.data)==result['test_sha256']
    assert sha256(a.result_dir.parent/'SELECTED_FREEZE.json')==result['selected_freeze_sha256']
    for n,h in freeze['files'].items():assert sha256(a.result_dir.parent/n)==h,n
    path=a.result_dir/'test_predictions.jsonl';assert sha256(path)==result['prediction_sha256']
    gold=read_bio(a.data);pred=read_predictions(path,gold);report=score(gold,pred,freeze['types'])
    assert report['micro']==result['micro'] and report['macro13']==result['macro13']
    print(json.dumps(dict(verified=True,sentences=len(gold),**report),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
