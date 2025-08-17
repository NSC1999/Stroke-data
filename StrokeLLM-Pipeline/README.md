# StrokeLLM-Pipeline (Full Rules Package)

This package includes the full 64-feature rules inferred from `stroke.csv`, a JSON Schema for validation, a QC script, and an evaluation template.

## Contents
- `rules/rules.json` — full rules (types/enums/ranges per feature)
- `schema/feature_schema.json` — schema generated from rules
- `qc_scripts/validate_schema.py` — CSV validator against the schema
- `eval_results/metrics.json` — fill with P/R/F1 + error breakdown

## Usage
```bash
python qc_scripts/validate_schema.py schema/feature_schema.json YOUR_EXTRACTED.csv
```
