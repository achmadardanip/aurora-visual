#!/usr/bin/env python3
"""Remove volatile timing and host metadata from the committed pytest report."""

import re
from pathlib import Path

path = Path("artifacts/reports/pytest.xml")
report = path.read_text()
report = re.sub(
    r' time="[0-9.]+" timestamp="[^"]+" hostname="[^"]+"',
    "",
    report,
    count=1,
)
report = re.sub(r' time="[0-9.]+"', "", report)
path.write_text(report.rstrip("\n") + "\n")
