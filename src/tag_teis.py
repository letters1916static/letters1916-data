from glob import glob
from pathlib import Path
from copy import deepcopy
from time import perf_counter
from lxml import etree
from openai import OpenAI
import os
import json
import csv
import re

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

LLM_STUDIO = True
INPUT_DIR = "./data/editions"
MODEL = "google/gemma-3-4b"
OUTPUT_DIR = f"./llm/{MODEL}"
LOG_FILE = Path("./llm/log.csv")
SCHEMA_PATH = Path("schema/tei_all.rng")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

with SCHEMA_PATH.open("rb") as schema_file:
    validator = etree.RelaxNG(etree.parse(schema_file))

LOG_FIELDNAMES = ["model", "file", "duration", "status", "valid"]

if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
    with LOG_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(LOG_FIELDNAMES)
else:
    with LOG_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        existing_fieldnames = reader.fieldnames or []

        if "valid" not in existing_fieldnames:
            rows = list(reader)
            with LOG_FILE.open("w", encoding="utf-8", newline="") as rewrite_handle:
                writer = csv.DictWriter(
                    rewrite_handle,
                    fieldnames=LOG_FIELDNAMES,
                    delimiter="|",
                )
                writer.writeheader()
                for row in rows:
                    row["valid"] = "false"
                    writer.writerow(row)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# -----------------------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# -----------------------------------------------------------------------------

BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "body_xml": {
            "type": "string",
            "description": "Full TEI <body> element with opener/closer markup added."
        }
    },
    "required": ["body_xml"],
    "additionalProperties": False,
}

# -----------------------------------------------------------------------------
# PROMPT
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a TEI P5 XML expert.

Task:
- Annotate letter structure inside a TEI <body> element.

Rules:
- Return ONLY valid XML inside the <body> element.
- Do NOT modify text content.
- Preserve <pb/>, <lb/>, and all existing markup.
- Add:
  <opener> with <dateline> and <salute>
  <closer> with <salute> and <signed>
- Replace constructs like:
  <seg type="closer">...</seg>
- If there are text nodes after (!) the the closing <closer> element, wrap those into <postscript><p> element-structure
- Replace <ab></ab> Element with <div></div>
- Wrap direct text node children of the .//body/div into <p> elements but <opener> and <closer> must not be children of <p> and must not contain <p>
- The final structure of the body element should follow this pattern
```xml
<div>
    <opener>
        <dateline></dateline><salute></salute>
    </opener>
    <p></p>
    <closer>
        <salute></salute>
    </closer>
    <postscript><p></p></postscript>
</div>
```

- Output must remain valid XML.
"""

USER_PROMPT = """
Annotate this TEI body:

{body}
"""

JSON_MODE_SYSTEM_PROMPT = """
You are a TEI P5 XML expert.

Task:
- Annotate letter structure inside a TEI <body> element.

Rules:
- Return JSON only.
- The JSON object must have exactly one key: body_xml.
- The value of body_xml must be a string containing the full TEI <body> element.
- Do NOT modify text content.
- Preserve <pb/>, <lb/>, and all existing markup.
- Add:
    <opener> with <dateline> and <salute>
    <closer> with <salute> and <signed>
- Replace constructs like:
    <seg type="closer">...</seg>
- If there are text nodes after (!) the the closing <closer> element, wrap those into <postscript><p> element-structure
- Replace <ab></ab> Element with <div></div>
- Wrap direct text node children of the .//body/div into <p> elements but <opener> and <closer> must not be children of <p> and must not contain <p>
- The final structure of the body element should follow this pattern
```xml
<div>
        <opener>
                <dateline></dateline><salute></salute>
        </opener>
        <p></p>
        <closer>
                <salute></salute>
        </closer>
        <postscript><p></p></postscript>
</div>
```

- Output must remain valid XML.
"""

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def serialize(elem):
    return etree.tostring(elem, encoding="unicode", pretty_print=True)


def validate_xml_file(xml_path):
    try:
        xml_doc = etree.parse(str(xml_path))
    except (OSError, etree.XMLSyntaxError):
        return False

    return validator.validate(xml_doc)


def append_log_row(file_path, duration_seconds, status, valid):
    with LOG_FILE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow([MODEL, file_path, f"{duration_seconds:.2f}", status, valid])


def extract_body(xml_tree):
    return xml_tree.find(".//tei:body", NS)


def parse_body(xml_string):
    parser = etree.XMLParser(remove_blank_text=False)
    try:
        elem = etree.fromstring(xml_string.encode("utf-8"), parser)
    except Exception as e:
        raise ValueError(f"Invalid XML returned by model:\n{e}")

    if etree.QName(elem).localname != "body":
        raise ValueError("Model did not return <body> element")

    return elem


def create_structured_response(original_body):
    return client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(body=original_body),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "tei_body",
                "schema": BODY_SCHEMA,
                "strict": True,
            }
        },
    )


def create_json_mode_response(original_body):
    return client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": JSON_MODE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(body=original_body),
            },
        ],
        text={
            "format": {
                "type": "json_object",
            }
        },
    )


def _strip_markdown_code_fence(text):
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    return stripped


def _parse_json_payload(text):
    cleaned = _strip_markdown_code_fence(text)

    # First try parsing as-is.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: if additional text surrounds JSON, extract the first JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("No JSON object found in model response")


def _extract_response_text(response):
    if getattr(response, "output_text", None):
        return response.output_text

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(getattr(content, "text", ""))

    return "\n".join(part for part in parts if part)


def get_body_xml_response(original_body):
    if RESPONSE_FORMAT == "json_schema":
        response = create_structured_response(original_body)
    else:
        response = create_json_mode_response(original_body)

    response_text = _extract_response_text(response)

    if not response_text:
        raise ValueError("Empty model response")

    try:
        payload = _parse_json_payload(response_text)
    except json.JSONDecodeError as e:
        preview = response_text[:300].replace("\n", "\\n")
        raise ValueError(
            f"Invalid JSON returned by model (possibly truncated): {e}. "
            f"Response preview: {preview}"
        ) from e

    if not isinstance(payload, dict) or "body_xml" not in payload:
        raise ValueError("Model JSON response does not contain required 'body_xml' key")

    return payload


RESPONSE_FORMAT = "json_object" if LLM_STUDIO else "json_schema"
print(f"Using response format: {RESPONSE_FORMAT}")


# -----------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------

files = glob(f"{INPUT_DIR}/*.xml")

for file_path in files[0:10]:
    started_at = perf_counter()
    print(f"Processing {file_path}")

    parser = etree.XMLParser(remove_blank_text=False)

    try:
        tree = etree.parse(file_path, parser)
    except Exception as e:
        elapsed_seconds = perf_counter() - started_at
        print(f"Parse error: {file_path}: {e}")
        append_log_row(file_path, elapsed_seconds, "parse_error", "false")
        continue

    root = tree.getroot()
    body = extract_body(root)

    if body is None:
        elapsed_seconds = perf_counter() - started_at
        print(f"No <body> found in {file_path}")
        append_log_row(file_path, elapsed_seconds, "no_body", "false")
        continue

    original_body = serialize(body)

    try:
        result = get_body_xml_response(original_body)
        new_body_xml = result["body_xml"]

        new_body = parse_body(new_body_xml)

        parent = body.getparent()
        parent.replace(body, deepcopy(new_body))

        output_path = Path(OUTPUT_DIR) / Path(file_path).name

        tree.write(
            str(output_path),
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True,
        )

        is_valid = validate_xml_file(output_path)

        elapsed_seconds = perf_counter() - started_at
        print(f"✓ Written {output_path} ({elapsed_seconds:.2f}s)")
        print(f"Response format: {RESPONSE_FORMAT}")
        print(f"Valid: {is_valid}")
        append_log_row(file_path, elapsed_seconds, "success", "true" if is_valid else "false")

    except Exception as e:
        elapsed_seconds = perf_counter() - started_at
        print(f"✗ Failed {file_path}")
        print(f"Elapsed: {elapsed_seconds:.2f}s")
        print(e)
        append_log_row(file_path, elapsed_seconds, "failed", "false")

    print("Done.")