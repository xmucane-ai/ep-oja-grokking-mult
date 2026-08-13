import json, hashlib, os, datetime

# Load existing provenance to preserve structure
with open('provenance.json') as f:
    prov = json.load(f)

# Collect all artifacts in outputs/ (files, not dirs)
artifacts = {}
for root, dirs, files in os.walk('outputs'):
    for fn in sorted(files):
        fp = os.path.join(root, fn)
        rel = os.path.normpath(fp).replace(os.sep, '/')
        h = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
        artifacts[rel] = h

prov['artifacts'] = artifacts
prov['generated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
prov['note'] = ('Artifact provenance manifest — public repository xmucane-ai/ep-oja-grokking-mult. '
                'Regenerated 2026-08-13 to include follow-up artifacts (living EC, DG-on-CFG, Paper 2 docs).')

with open('provenance.json', 'w') as f:
    json.dump(prov, f, indent=2, sort_keys=True)

print(f'artifacts in manifest: {len(artifacts)}')
print('has living_ec:', 'outputs/living_ec_exp0_real_results.json' in artifacts)
print('has dg_cfg:', 'outputs/dg_cfg_results.json' in artifacts)
print('has exp5:', 'outputs/living_ec_exp5_results.json' in artifacts)
