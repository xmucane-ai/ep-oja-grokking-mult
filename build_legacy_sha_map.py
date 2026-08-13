#!/usr/bin/env python3
"""build_legacy_sha_map.py v6 — resolve paper private SHAs → public paths.

25 SHAs map to outputs/ artifacts (from the claims table). 5 more map to
non-artifact paths (scripts/, docs/) or are prose refs with no file. v6 merges
both, plus records explicit NO-FILE for prose-only SHAs so no reviewer hits a
dead end.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "paper", "main.tex")
tex = open(TEX, encoding="utf-8", errors="replace").read()

sha_re = re.compile(r"\b[0-9a-f]{7}\b")

# ── pass 1: artifact mappings (from claims table) ──
path_sha_re = re.compile(
    r"\\path\{(?:outputs/)?([a-zA-Z0-9_\-\./\\]+\.(?:json|txt))"
    r"\}[^()]{0,80}?\(([0-9a-f]{7})\)"
)
sha_map = {}
for m in path_sha_re.finditer(tex):
    artifact = m.group(1).replace("\\_", "_")
    sha_map.setdefault(m.group(2), set()).add("outputs/" + artifact)

# ── pass 2: script/doc paths ──
for m in re.finditer(r"\\path\{((?:scripts|docs)/[a-zA-Z0-9_\-\./\\]+)\}\s*,?\s*\(([0-9a-f]{7})\)", tex):
    p = m.group(1).replace("\\_", "_")
    sha_map.setdefault(m.group(2), set()).add(p)

# ── pass 2b: fix known mis-prefixed refs ──
for sha in list(sha_map):
    fixed = set()
    for p in sha_map[sha]:
        if p.startswith("outputs/scripts/") or p.startswith("outputs/docs/"):
            p = p[len("outputs/"):]
        fixed.add(p)
    sha_map[sha] = fixed

# ── pass 3: prose-only SHAs (context lookup, no file) ──
prose_shas = {"beb51ae": "thresholds commit (prose; no artifact)", }
for m in sha_re.finditer(tex):
    sha = m.group(0)
    if sha in sha_map:
        continue
    # look for a path anywhere in ±200 chars
    window = tex[max(0, m.start()-200):m.end()+200]
    paths = re.findall(r"((?:outputs|scripts|docs)/[a-zA-Z0-9_\-\./\\]+\.(?:json|txt|py|md))", window)
    if paths:
        sha_map[sha] = {p.replace("\\_", "_") for p in paths[-1:]}
    elif sha in prose_shas:
        sha_map[sha] = set()  # prose-only, no file

out = {}
for sha, paths in sha_map.items():
    # expand the paper's T1/T5/T10 shorthand into the three real files
    expanded = set()
    for p in paths:
        if "T1/T5/T10" in p:
            expanded.add(p.replace("T1/T5/T10", "T1"))
            expanded.add(p.replace("T1/T5/T10", "T5"))
            expanded.add(p.replace("T1/T5/T10", "T10"))
        else:
            expanded.add(p)
    out[sha] = {"files": sorted(expanded)} if expanded else {"note": "prose ref, no file"}

with open(os.path.join(HERE, "legacy_sha_map.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)

cited = set(sha_re.findall(tex))
unmapped = cited - set(out.keys())
print(f"mapped {len(out)}/{len(cited)} SHAs; unmapped: {len(unmapped)}")
for sha in sorted(unmapped):
    print(f"  UNMAPPED {sha}")

# verify: every mapped FILE exists
missing_files = []
for sha, entry in out.items():
    for p in entry.get("files", []):
        if not os.path.isfile(os.path.join(HERE, p)):
            missing_files.append(f"{sha} -> {p}")
print(f"\nfile-existence failures: {len(missing_files)}")
for mf in missing_files:
    print(f"  [MISSING] {mf}")
print(f"\nALL {len(out)} legacy SHAs resolved" if not unmapped and not missing_files else "\nPARTIAL")
