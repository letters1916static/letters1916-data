from glob import glob
from pathlib import Path
from copy import deepcopy
from lxml import etree
from openai import OpenAI
import os
import json

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

INPUT_DIR = "./data/editions"
OUTPUT_DIR = "./data/editions"
MODEL = "gpt-5.5"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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

for file_path in files:
    print(f"Processing {file_path}")

    parser = etree.XMLParser(remove_blank_text=False)

    try:
        tree = etree.parse(file_path, parser)
    except Exception as e:
        print(f"Parse error: {file_path}: {e}")
        continue

    root = tree.getroot()
    body = extract_body(root)

    if body is None:
        print(f"No <body> found in {file_path}")
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

        print(f"✓ Written {output_path}")

    except Exception as e:
        print(f"✗ Failed {file_path}")
        print(e)

print("Done.")