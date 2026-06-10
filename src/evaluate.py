from pathlib import Path
import csv
from collections import defaultdict

LOG_FILE = Path("./llm/log.csv")
STATS_FILE = Path("./llm/stats.csv")


def compute_stats():
    rows = []
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)

    with open(STATS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_runs", "avg_duration_s", "valid_pct"])
        for model, entries in sorted(groups.items()):
            num_runs = len(entries)
            avg_duration = sum(float(e["duration"]) for e in entries) / num_runs
            valid_pct = sum(1 for e in entries if e["valid"].strip().lower() == "true") / num_runs * 100
            writer.writerow([model, num_runs, f"{avg_duration:.2f}", f"{valid_pct:.1f}"])

    print(f"Stats written to {STATS_FILE}")


if __name__ == "__main__":
    compute_stats()
