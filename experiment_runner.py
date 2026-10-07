"""Run a fixed benchmark with fresh conversation state; save raw auditable records.

This collects experiments, not automated semantic-grounding scores. Labels are
copied unchanged for human review. It never calls a model judge or invents metrics.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

from src.config import RetrievalConfig, GenerationConfig
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore, load_embedder
from src.chatbot import HardwarePlanner


def read_benchmark(path):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if not dataset.get("dataset_id") or not dataset.get("version"):
        raise ValueError("Benchmark needs dataset_id and version")
    cases = dataset.get("cases", [])
    if not cases:
        raise ValueError("Benchmark contains no cases")
    ids = set()
    for case in cases:
        if not case.get("id") or case["id"] in ids:
            raise ValueError("Case IDs must be nonempty and unique")
        ids.add(case["id"])
        if not case.get("objective") or not case.get("pdf"):
            raise ValueError("Each case needs objective and pdf")
        if case.get("split") not in {"dev", "held_out"}:
            raise ValueError("Each case needs split: dev or held_out")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError("Each case needs a human-reviewed answerable boolean")
        if not isinstance(case.get("reference_evidence"), list):
            raise ValueError("Each case needs a reference_evidence list")
        if not isinstance(case.get("expected_behaviors"), list):
            raise ValueError("Each case needs an expected_behaviors list")
    return dataset


def evaluate_case(planner, store, case, repeat):
    """Injectable for offline regression tests; reset even after a prior failure."""
    planner.init_chat()
    before = len(planner.calls)
    started = perf_counter()
    record = {"case_id": case["id"], "repeat": repeat, "case": case,
              "status": "started", "output": None, "call": None,
              "human_scores": None, "cost": None, "peak_memory_bytes": None}
    try:
        answer, _ = planner.chat(case["objective"], store=store, fresh=True,
                                 retrieval_query=case.get("retrieval_query"))
        record.update(status="completed", output=answer)
    except Exception as exc:
        # Do not persist arbitrary provider exception strings that might contain
        # request/credential details. Raw partial output remains in call records.
        record.update(status="failed", error_type=type(exc).__name__)
    record["case_seconds"] = perf_counter() - started
    if len(planner.calls) > before:
        record["call"] = planner.calls[-1]
    return record


def environment():
    packages = {}
    for name in ["docling", "docling-core", "sentence-transformers", "transformers",
                 "torch", "faiss-cpu", "groq", "numpy", "tiktoken", "streamlit"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "packages": packages, "device": "cpu"}


def run(args):
    benchmark_path = Path(args.benchmark).resolve()
    dataset = read_benchmark(benchmark_path)
    if dataset.get("annotation_status") != "human_reviewed":
        raise ValueError("Replace template labels and set annotation_status to human_reviewed")
    cases = [c for c in dataset["cases"] if args.split == "all" or c["split"] == args.split]
    if not cases or args.repeats < 1:
        raise ValueError("Need selected cases and repeats >= 1")
    configurations = [
        RetrievalConfig(top_k=args.top_k, candidate_k=args.candidate_k,
                        chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens,
                        rerank=rerank) for rerank in
        ([False, True] if args.variant == "both" else [args.variant == "dense_rerank"])]
    generation = GenerationConfig(model=args.model, max_output_tokens=args.max_output_tokens,
                                  context_tokens=args.context_tokens)
    # Fail before any downloads if no credentials are configured.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("GROQ_API_KEY", "")
    if not key.strip():
        raise ValueError("Set GROQ_API_KEY before running live experiments")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never silently overwrite results.
    source_root = Path(__file__).resolve().parent
    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "benchmark_sha256": sha256(benchmark_path.read_bytes()).hexdigest(),
                "dataset": dataset, "split": args.split, "repeats": args.repeats,
                "environment": environment(), "retrieval_configs": [asdict(c) for c in configurations],
                "generation_config": asdict(generation), "documents": {},
                "source_sha256": {str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                                  for p in [source_root / "experiment_runner.py", *sorted((source_root / "src").glob("*.py"))]},
                "timing_policy": "per-call elapsed; lazy reranker load included in first call",
                "limitations": ["Raw outputs require human grounding annotation",
                                "No cost or peak memory measurement implemented",
                                "Context token counter is a proxy; provider usage is authoritative",
                                "Fixed configured model names may resolve to mutable provider versions"]}
    manifest_path = output / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    save_manifest()
    planner = None
    try:
        planner = HardwarePlanner(api_key=key, config=generation)
        manifest["generation_settings"] = planner.settings()
        started = perf_counter()
        embedder = load_embedder()
        manifest["embedder_load_seconds"] = perf_counter() - started
        save_manifest()
        with (output / "results.jsonl").open("w", encoding="utf-8") as results:
            # Parse each document ONCE and reuse exactly the same chunks/index.
            for pdf_name in dict.fromkeys(c["pdf"] for c in cases):
                pdf_path = (benchmark_path.parent / pdf_name).resolve()
                data = pdf_path.read_bytes()
                started = perf_counter()
                with PDFProcessor(data, source_name=pdf_path.name, max_pages=args.max_pages) as processor:
                    doc, chunks, pages = processor.extract_document()
                parse_seconds = perf_counter() - started
                store = VectorStore(embedder, configurations[0])
                started = perf_counter()
                store.build_index(chunks, processor.document_id, pdf_path.name)
                index_seconds = perf_counter() - started
                # Snapshot all records so references and chunk IDs can be reviewed.
                evidence_file = f"evidence-{processor.document_id}.json"
                (output / evidence_file).write_text(json.dumps(store.chunks, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
                manifest["documents"][pdf_name] = {"sha256": processor.document_id,
                    "pages": pages, "parsed_pages": len(doc.pages), "chunks": len(store.chunks),
                    "parse_seconds": parse_seconds, "index_seconds": index_seconds,
                    "evidence_file": evidence_file, "retrieval_settings": store.settings()}
                save_manifest()
                for config in configurations:
                    store.config = config
                    variant = "dense_rerank" if config.rerank else "dense"
                    for case in [c for c in cases if c["pdf"] == pdf_name]:
                        for repeat in range(1, args.repeats + 1):
                            record = evaluate_case(planner, store, case, repeat)
                            record.update(variant=variant, document_id=processor.document_id,
                                          retrieval_settings=store.settings())
                            results.write(json.dumps(record, ensure_ascii=False) + "\n")
                            results.flush()
                            print(f"{variant}: {case['id']} repeat={repeat} {record['status']}")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        if planner is not None:
            planner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing directories are rejected")
    parser.add_argument("--variant", choices=["dense", "dense_rerank", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "held_out", "all"], default="dev")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--chunk-tokens", type=int, default=240)
    parser.add_argument("--overlap-tokens", type=int, default=24)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--model", default=GenerationConfig().model)
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--context-tokens", type=int, default=3000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
