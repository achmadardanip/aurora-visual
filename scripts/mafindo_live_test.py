"""Opt-in live MAFINDO compatibility check; no credential is read from arguments or output."""

import json
import os

from aurora_visual.mafindo import MafindoClient

if not os.environ.get("AURORA_MAFINDO_API_KEY"):
    raise SystemExit("AURORA_MAFINDO_API_KEY must be configured in the local server environment")

result = MafindoClient(
    os.environ["AURORA_MAFINDO_API_KEY"],
    float(os.getenv("AURORA_MAFINDO_TIMEOUT_SECONDS", "12")),
).latest(1)
if result["status"] != "ok":
    raise SystemExit("MAFINDO diagnostic failed without exposing credential details")
print(
    json.dumps(
        {k: v for k, v in result.items() if k != "results"} | {"result_count": len(result["results"])},
        indent=2,
    )
)
