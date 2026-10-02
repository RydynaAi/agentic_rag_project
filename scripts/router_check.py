import json
import time
from pathlib import Path

from app import agents

items = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
wrong = []

for item in items:
    result = agents.route_question({"question": item["question"]})
    route = "web_search" if result == "web_search" else "vectorstore"
    if route != item["expected_route"]:
        wrong.append((item["id"], item["expected_route"], route))
    time.sleep(1)

accuracy = 100 * (len(items) - len(wrong)) / len(items)
print(f"Router accuracy: {accuracy:.1f}% ({len(items) - len(wrong)}/{len(items)})")
for item_id, expected, got in wrong:
    print(f"  {item_id}: expected {expected}, got {got}")
