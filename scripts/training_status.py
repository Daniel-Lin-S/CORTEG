"""Inspect and reconcile repository-local training progress.

Input: logs/training/<campaign>/registry.json and completion markers.
Output: refreshed logs/training/<campaign>/status.json files and a concise
terminal summary. No fit artifacts are modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from infrastructure.training_lifecycle import LOG, refresh_status


def main() -> None:
    """Print current groups after reconciling stopped coordinators."""
    status = refresh_status()
    print(LOG.parent.resolve())
    if not status["campaigns"]:
        print("No training campaigns are registered.")
        return
    for campaign in status["campaigns"].values():
        path = LOG.parent / campaign["name"] / "status.json"
        print(f"{path.resolve()}: {campaign['state']}")
        for name, group in sorted(campaign["groups"].items()):
            print(
                f"  {name}: {group['completed']}/{group['total']} "
                f"completed, {group['pending']} pending, "
                f"{group['running']} running"
            )
            for task in group["active"]:
                print(f"    active {json.dumps(task, sort_keys=True)}")


if __name__ == "__main__":
    main()
