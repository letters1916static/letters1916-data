from glob import glob
from pathlib import Path
from copy import deepcopy
from time import perf_counter
from lxml import etree
from openai import OpenAI
import os
import json
import csv

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

INPUT_DIR = "./data/editions"
MODEL = "gpt-5.4-nano"
OUTPUT_DIR = f"./llm/{MODEL}"
LOG_FILE = Path("./llm/log.csv")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
    with LOG_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["model", "file", "duration", "status"])

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

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def serialize(elem):
    return etree.tostring(elem, encoding="unicode", pretty_print=True)


def append_log_row(file_path, duration_seconds, status):
    with LOG_FILE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow([MODEL, file_path, f"{duration_seconds:.2f}", status])


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
        append_log_row(file_path, elapsed_seconds, "parse_error")
        continue

    root = tree.getroot()
    body = extract_body(root)

    if body is None:
        elapsed_seconds = perf_counter() - started_at
        print(f"No <body> found in {file_path}")
        append_log_row(file_path, elapsed_seconds, "no_body")
        continue

    original_body = serialize(body)

    try:
        response = client.responses.create(
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

        result = json.loads(response.output_text)
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

        elapsed_seconds = perf_counter() - started_at
        print(f"✓ Written {output_path} ({elapsed_seconds:.2f}s)")
        append_log_row(file_path, elapsed_seconds, "success")

    except Exception as e:
        elapsed_seconds = perf_counter() - started_at
        print(f"✗ Failed {file_path}")
        print(f"Elapsed: {elapsed_seconds:.2f}s")
        print(e)
        append_log_row(file_path, elapsed_seconds, "failed")

    print("Done.")