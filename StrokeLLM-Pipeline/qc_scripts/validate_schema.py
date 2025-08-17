import json, sys, csv

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def is_int_str(x):
    try:
        return str(int(float(x))).strip() == str(x).strip()
    except:
        return False

def validate(schema_file, csv_file):
    with open(schema_file, encoding="utf-8") as f:
        schema = json.load(f)
    props = schema.get("properties", {})
    required = schema.get("required", [])
    errors = 0
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            for field in required:
                if field not in row:
                    print(f"[Row {i}] Missing field: {field}")
                    errors += 1
            for col, spec in props.items():
                if col not in row:
                    continue
                val = row[col]
                if val == "" or val is None:
                    continue
                typ = spec.get("type")
                ok = True
                if typ == "number" and not is_number(val):
                    ok = False
                elif typ == "integer" and not is_int_str(val):
                    ok = False
                elif isinstance(typ, list):
                    ok = any([
                        (t == "string"),
                        (t == "integer" and is_int_str(val)),
                        (t == "number" and is_number(val)),
                        (t == "null" and val in ("", "null", None))
                    ] for t in typ)
                if not ok:
                    print(f"[Row {i}] Type mismatch for '{col}': got '{val}', expected {typ}")
                    errors += 1
                    continue
                if "enum" in spec:
                    allowed = spec["enum"]
                    v = int(float(val)) if all(isinstance(x, int) for x in allowed) else val
                    if v not in allowed:
                        print(f"[Row {i}] Value {val} not in enum for '{col}': {allowed}")
                        errors += 1
                if "minimum" in spec and is_number(val) and float(val) < float(spec["minimum"]):
                    print(f"[Row {i}] {col}={val} < minimum {spec['minimum']}")
                    errors += 1
                if "maximum" in spec and is_number(val) and float(val) > float(spec["maximum"]):
                    print(f"[Row {i}] {col}={val} > maximum {spec['maximum']}")
                    errors += 1
    if errors == 0:
        print("Validation OK: no issues found.")
    else:
        print(f"\nTotal issues: {errors}")
        raise SystemExit(2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python qc_scripts/validate_schema.py schema/feature_schema.json YOUR_EXTRACTED.csv")
    else:
        validate(sys.argv[1], sys.argv[2])
