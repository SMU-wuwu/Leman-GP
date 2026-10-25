"""Archive dev analysis without weights or inference shards."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
out = root/'output/leman_fgm_v14_analysis_seed44'
files = [p for p in out.rglob('*') if p.is_file() and
         not any(part in ('shards','apply_shards') for part in p.relative_to(out).parts)
         and p.suffix != '.pt']
manifest = {'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'test_evaluated':False}
target = root/'output/return_leman_fgm_v14_analysis.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p,str(p.relative_to(out)))
    z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(target,flush=True)
