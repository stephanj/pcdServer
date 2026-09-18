#!/usr/bin/env python3
"""Converts parallelConstraintDecoding preset files into ui/presets.json.

Usage: scripts/import-presets.py <presets-dir> [max-samples-per-preset]
The reference format keeps fields in a `schema` object with `type`; this
server takes an ordered `fields` array where booleans are `[false, true]`.
"""
import json
import pathlib
import sys

ORDER = ["spam_email", "devoxx_cfp", "support_triage", "fintech_fraud", "code_security", "high_cardinality_255"]

src = pathlib.Path(sys.argv[1])
max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 6
out = []
for pid in ORDER:
    raw = json.loads((src / f"{pid}.json").read_text())
    fields = []
    for name, spec in raw["schema"].items():
        choices = [False, True] if spec["type"] == "boolean" else spec["choices"]
        fields.append({"name": name, "description": spec.get("description", ""), "choices": choices})
    preset = {"id": raw["id"], "title": raw["title"], "description": raw.get("description", ""), "context": raw["context"], "fields": fields}
    if raw.get("samples"):
        preset["samples"] = [{"label": s["label"], "context": s["context"], **({"expected": s["expected"]} if s.get("expected") else {})} for s in raw["samples"][:max_samples]]
    out.append(preset)
pathlib.Path("ui/presets.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} presets, {sum(len(p['fields']) for p in out)} fields")
