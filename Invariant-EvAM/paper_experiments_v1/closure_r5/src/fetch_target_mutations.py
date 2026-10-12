"""Reproduce API fallback; default verifies cached responses, --refresh re-downloads."""
from pathlib import Path
import json,hashlib,sys,requests
D=Path(__file__).resolve().parents[2]/'g4/revision_r5/data_external'
receipts=json.loads((D/'API_RECEIPTS.json').read_text())
for r in receipts:
 study=r['body']['sampleListId'].removesuffix('_sequenced');p=D/study/'target_mutations_api.json'
 if p.exists() and '--refresh' not in sys.argv:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];print(study,'cached response hash verified');continue
 response=requests.post(r['url'],json=r['body'],timeout=180);response.raise_for_status()
 data=response.content
 if hashlib.sha256(data).hexdigest()!=r['sha256']:
  raise RuntimeError('Upstream bytes changed; frozen input not overwritten. Audit new release separately.')
 p.write_bytes(data);print(study,'downloaded exact frozen response')
