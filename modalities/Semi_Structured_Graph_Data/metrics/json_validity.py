"""
JSON Validity — Valid JSON Rate & Schema Correctness Rate  (Survey §5.2.2)

    ValidJSONRate    = |{J : IsParsable(J)}| / |J|
    CorrectnessRate  = (1/|J|) Σ V(J, S),  V(J, S) = I[parsable(J) ∧ schema(J, S)]
"""

import json
from typing import List, Dict, Any

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def compute_json_validity(
    data: List[Dict[str, Any]],
    cfg: Any = None,
) -> Dict[str, Any]:
    """
    Compute Valid JSON Rate and optional Schema Correctness Rate.

    Args:
        data: list of dicts, each containing a field with generated JSON text
        cfg: optional config with json_eval.{json_field, schema_path}
    """
    json_field = "output"
    schema_path = ""

    if cfg is not None:
        jcfg = getattr(cfg, 'json_eval', None)
        if jcfg:
            json_field = getattr(jcfg, 'json_field', json_field)
            schema_path = getattr(jcfg, 'schema_path', schema_path)

    schema = None
    if schema_path:
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        except Exception:
            pass

    total = 0
    valid_json = 0
    schema_valid = 0
    details = []

    for item in data:
        text = item.get(json_field, '')
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        if not text:
            continue
        total += 1

        parsed = None
        parse_ok = False
        schema_ok = False

        try:
            parsed = json.loads(text)
            parse_ok = True
            valid_json += 1
        except (json.JSONDecodeError, TypeError):
            pass

        if parse_ok and schema is not None and HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=parsed, schema=schema)
                schema_ok = True
                schema_valid += 1
            except (jsonschema.ValidationError, jsonschema.SchemaError):
                pass

        details.append({
            'id': item.get('id', total - 1),
            'valid_json': parse_ok,
            'schema_valid': schema_ok if schema else None,
        })

    vjr = valid_json / total if total > 0 else 0.0
    scr = schema_valid / total if total > 0 and schema else None

    result = {
        'valid_json_rate': vjr,
        'total': total,
        'valid_json_count': valid_json,
        'details': details[:50],
    }
    if scr is not None:
        result['schema_correctness_rate'] = scr
        result['schema_valid_count'] = schema_valid

    return result
