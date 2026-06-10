import csv
from pathlib import Path
from lxml import etree as ET


SCHEMA_PATH = Path("schema/tei_all.rng")
with SCHEMA_PATH.open("rb") as schema_file:
    validator = ET.RelaxNG(ET.parse(schema_file))

CSV_FILE = Path("llm/log.csv")
with CSV_FILE.open(newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, delimiter="|")
    rows = []

    for row in reader:
        xml_path = Path(row["file"])
        try:
            xml_doc = ET.parse(xml_path)
            is_valid = validator.validate(xml_doc)
        except (OSError, ET.XMLSyntaxError):
            is_valid = False

        row["valid"] = "1" if is_valid else "0"
        print(f"{xml_path} is valid: {is_valid}")
        rows.append(row)

fieldnames = list(reader.fieldnames or [])
if "valid" not in fieldnames:
    fieldnames.append("valid")

with CSV_FILE.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")
    writer.writeheader()
    writer.writerows(rows)