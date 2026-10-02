import argparse
import json
import re
import statistics
import time
from pathlib import Path

from app import agents
from app.graph import app as graph

QUESTIONS_PATH = Path("eval/questions.json")
RAW_PATH = Path("results/raw.jsonl")
REPORT_PATH = Path("results/eval.md")

CHAR_MAP = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u202f": " ",
})


def normalize(text):
    return text.translate(CHAR_MAP).lower()


def matches(keyword, text):
    pattern = r"(?<![\w.])" + re.escape(normalize(keyword)) + r"(?!\w)"
    return re.search(pattern, text) is not None


def is_correct(keywords, text):
    if not keywords:
        return None
    body = normalize(text)
    for item in keywords:
        options = item if isinstance(item, list) else [item]
        if not any(matches(option, body) for option in options):
            return False
    return True


def with_retries(fn, attempts=4, base_delay=5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** attempt)


def run_agentic(question):
    start = time.perf_counter()
    nodes = []
    generation = ""
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node, payload in update.items():
            nodes.append(node)
            if node == "generate" and payload:
                generation = payload["generation"]
    latency = time.perf_counter() - start
    route = "web_search" if nodes and nodes[0] == "web_search" else "vectorstore"
    return {
        "route": route,
        "used_web": "web_search" in nodes,
        "generations": nodes.count("generate"),
        "generation": generation,
        "latency": latency,
    }


def run_plain(question):
    start = time.perf_counter()
    docs = agents.retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    generation = agents.gen_chain.invoke({"context": context, "question": question}).content
    return {"generation": generation, "latency": time.perf_counter() - start}


def evaluate_item(item):
    question = item["question"]
    keywords = item["keywords"]
    record = {"id": item["id"], "question": question, "expected_route": item["expected_route"]}
    if item["expected_route"] == "vectorstore":
        docs = agents.retriever.invoke(question)
        record["retrieval_hit"] = is_correct(keywords, " ".join(d.page_content for d in docs))
    else:
        record["retrieval_hit"] = None
    agentic = with_retries(lambda: run_agentic(question))
    plain = with_retries(lambda: run_plain(question))
    record.update({
        "route": agentic["route"],
        "used_web": agentic["used_web"],
        "generations": agentic["generations"],
        "agentic_answer": agentic["generation"],
        "agentic_correct": is_correct(keywords, agentic["generation"]),
        "agentic_latency": agentic["latency"],
        "plain_answer": plain["generation"],
        "plain_correct": is_correct(keywords, plain["generation"]),
        "plain_latency": plain["latency"],
    })
    return record


def load_done():
    done = {}
    if not RAW_PATH.exists():
        return done
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "error" not in record:
            done[record["id"]] = record
    return done


def rate(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def pct(result):
    value, n = result
    return "n/a" if value is None else f"{value * 100:.1f}% (n={n})"


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def router_metrics(records):
    tp = sum(1 for r in records if r["route"] == "web_search" and r["expected_route"] == "web_search")
    fp = sum(1 for r in records if r["route"] == "web_search" and r["expected_route"] != "web_search")
    fn = sum(1 for r in records if r["route"] != "web_search" and r["expected_route"] == "web_search")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(1 for r in records if r["route"] == r["expected_route"]) / len(records)
    return accuracy, precision, recall, f1


def build_report(records):
    paper = [r for r in records if r["expected_route"] == "vectorstore"]
    web = [r for r in records if r["expected_route"] == "web_search"]
    accuracy, precision, recall, f1 = router_metrics(records)
    a_lat = [r["agentic_latency"] for r in records]
    p_lat = [r["plain_latency"] for r in records]
    retrieval = pct(rate(r["retrieval_hit"] for r in paper))
    rows = [
        ("Answer correctness, all scored questions", pct(rate(r["agentic_correct"] for r in records)), pct(rate(r["plain_correct"] for r in records))),
        ("Answer correctness, paper questions", pct(rate(r["agentic_correct"] for r in paper)), pct(rate(r["plain_correct"] for r in paper))),
        ("Answer correctness, web questions", pct(rate(r["agentic_correct"] for r in web)), pct(rate(r["plain_correct"] for r in web))),
        ("Retrieval hit rate, paper questions (shared retriever)", retrieval, retrieval),
        ("Router accuracy", f"{accuracy * 100:.1f}% (n={len(records)})", "n/a"),
        ("Router precision / recall / F1 (web_search class)", f"{precision:.2f} / {recall:.2f} / {f1:.2f}", "n/a"),
        ("Web-search fallback rate, all questions", pct(rate(r["used_web"] for r in records)), "0.0%"),
        ("Web-search fallback rate, paper questions", pct(rate(r["used_web"] for r in paper)), "0.0%"),
        ("Latency mean (s)", f"{statistics.mean(a_lat):.2f}", f"{statistics.mean(p_lat):.2f}"),
        ("Latency p95 (s)", f"{percentile(a_lat, 0.95):.2f}", f"{percentile(p_lat, 0.95):.2f}"),
    ]
    lines = [
        "# Evaluation results",
        "",
        f"Questions evaluated: {len(records)} ({len(paper)} paper, {len(web)} web)",
        "",
        "| Metric | Agentic RAG | Plain RAG |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {agentic} | {plain} |" for name, agentic, plain in rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--only", choices=["vectorstore", "web_search"], default=None)
    args = parser.parse_args()

    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["expected_route"] == args.only]
    if args.limit:
        items = items[: args.limit]

    RAW_PATH.parent.mkdir(exist_ok=True)
    done = load_done()
    pending = [i for i in items if i["id"] not in done]
    print(f"{len(items)} selected, {len(pending)} pending")

    for index, item in enumerate(pending, 1):
        try:
            record = evaluate_item(item)
        except Exception as exc:
            record = {"id": item["id"], "error": str(exc)}
            print(f"[{index}/{len(pending)}] {item['id']} FAILED: {exc}")
        else:
            print(
                f"[{index}/{len(pending)}] {item['id']} route={record['route']} "
                f"correct={record['agentic_correct']} plain={record['plain_correct']} "
                f"{record['agentic_latency']:.1f}s"
            )
        with RAW_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        time.sleep(args.delay)

    records = list(load_done().values())
    if records:
        REPORT_PATH.write_text(build_report(records), encoding="utf-8")
        print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
