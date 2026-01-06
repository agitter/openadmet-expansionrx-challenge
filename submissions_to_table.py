#!/usr/bin/env python3
# Written by ChatGPT

import sys
import re
from pathlib import Path

def parse_markdown(md_text):
    """
    Parse submission blocks from the markdown text.
    Returns (headers, rows)
    """
    submission_re = re.compile(
        r"^### Submission (\d+):\s+(\S+)\s+(\S+)\s*$",
        re.MULTILINE
    )

    lines = md_text.splitlines()
    rows = []
    all_metric_headers = None

    i = 0
    while i < len(lines):
        match = submission_re.match(lines[i])
        if not match:
            i += 1
            continue

        submission_num, model, prompt = match.groups()
        i += 1

        # Collect metric headers
        metric_headers = []
        while i < len(lines) and lines[i].strip():
            metric_headers.append(lines[i].strip())
            i += 1

        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1

        # Collect metric values
        metric_values = []
        for _ in range(len(metric_headers)):
            if i >= len(lines):
                break
            metric_values.append(lines[i].strip())
            i += 1

        if all_metric_headers is None:
            all_metric_headers = metric_headers

        row = {
            "Submission": submission_num,
            "Model": model,
            "Prompt": prompt,
        }

        for h, v in zip(metric_headers, metric_values):
            row[h] = v

        rows.append(row)

    headers = ["Submission", "Model", "Prompt"] + (all_metric_headers or [])
    return headers, rows


def render_markdown_table(headers, rows):
    def esc(val):
        return val.replace("|", "\\|")

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    body_lines = []
    for row in rows:
        body_lines.append(
            "| " + " | ".join(esc(row.get(h, "")) for h in headers) + " |"
        )

    return "\n".join([header_line, sep_line] + body_lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python submissions_to_table.py input.md output.md")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    md_text = input_path.read_text(encoding="utf-8")
    headers, rows = parse_markdown(md_text)
    table_md = render_markdown_table(headers, rows)

    output_path.write_text(table_md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

