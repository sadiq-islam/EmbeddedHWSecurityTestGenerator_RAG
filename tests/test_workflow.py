"""Offline behavioral checks with real FAISS and fake inference; no model downloads."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from src.chunking import bounded_parts, prepare_chunks, token_count
from src.config import RetrievalConfig, GenerationConfig
from src.vector_store import VectorStore
from src.chatbot import HardwarePlanner, GenerationError
from experiment_runner import evaluate_case, read_benchmark
from experiment_runner import run


class Tokenizer:
    # Character tokens deliberately force long technical strings to split.
    def encode(self, text, add_special_tokens=True):
        return list(text) + ([0, 0] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Embedder:
    tokenizer = Tokenizer()
    max_seq_length = 80

    def encode(self, texts, **kwargs):
        vectors = np.array([[1 + t.lower().count("debug"),
                             1 + t.lower().count("voltage")] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Position:
    def __init__(self, page):
        self.page = page

    def model_dump(self, **kwargs):
        return {"page_no": self.page, "bbox": {"l": 10, "t": 20, "r": 30, "b": 40},
                "charspan": [0, 20]}


def chunk(text="Debug access disabled in production.", pages=(2, 3), headings=("Security",)):
    items = [NS(self_ref=f"#/texts/{i}", label="text", prov=[Position(p)])
             for i, p in enumerate(pages)]
    return NS(text=text, meta=NS(headings=list(headings), doc_items=items))


class Reranker:
    def predict(self, pairs):
        # Prefer the voltage passage to demonstrate actual ordering changes.
        return [10.0 if "voltage" in passage.lower() else -10.0 for _, passage in pairs]


class Counter:
    def encode(self, text):
        return list(text)


class FakeClient:
    api_key = "test-key"
    def __init__(self, finish="stop", fail=False):
        self.finish, self.fail = finish, fail
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("simulated failure")
        return NS(model="fake-provider", id="fake-id",
                  usage=NS(prompt_tokens=101, completion_tokens=21, total_tokens=122),
                  choices=[NS(finish_reason=self.finish, message=NS(content="Proposed answer"))])

    def close(self):
        pass


class FakeStore:
    document_id = "manual-A"
    def search(self, query, final_top_k=None):
        return [{"chunk_id": "a", "document_id": self.document_id,
                 "text": "Debug disabled.", "pages": [2], "provenance": []}]


class ChunkTests(unittest.TestCase):
    def test_actual_docling_objects_and_fast_tokenizer(self):
        from docling_core.types.doc import DoclingDocument, ProvenanceItem, BoundingBox, DocItemLabel
        from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
        from tokenizers import Tokenizer as Backend, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast
        backend = Backend(models.WordPiece(vocab={"[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                                                   "debug": 3, "voltage": 4, "production": 5}))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        text = "debug voltage production " * 40
        document = DoclingDocument(name="test")
        item = document.add_text(label=DocItemLabel.TEXT, text=text,
            prov=ProvenanceItem(page_no=2, bbox=BoundingBox(l=10, t=20, r=30, b=40), charspan=(0, len(text))))
        item.prov.append(ProvenanceItem(page_no=3, bbox=BoundingBox(l=10, t=20, r=30, b=40),
                                        charspan=(0, len(text))))
        actual_chunk = DocChunk(text=text, meta=DocMeta(doc_items=[item], headings=["production"]))
        records = prepare_chunks([actual_chunk], tokenizer, 32, 4, "hash", "manual.pdf")
        self.assertGreater(len(records), 1)
        self.assertTrue(all(r["embedding_tokens"] <= 32 for r in records))
        self.assertTrue(all(r["pages"] == [2, 3] for r in records))
        self.assertEqual(records[0]["provenance"][0]["item_ref"], item.self_ref)

    def test_budget_includes_headings_and_special_tokens(self):
        text = "VDD ≤ 3.3 V ±5%; Register [7:0]=0xFF.\n" * 20
        records = prepare_chunks([chunk(text)], Tokenizer(), 80, 10, "hash", "manual.pdf")
        self.assertGreater(len(records), 1)
        covered = set()
        for r in records:
            self.assertLessEqual(token_count(Tokenizer(), r["content"]), 80)
            self.assertEqual(r["text"], text[r["char_start"]:r["char_end"]])
            covered.update(range(r["char_start"], r["char_end"]))
        self.assertEqual(covered, set(range(len(text))))

    def test_complete_parent_provenance_survives_splitting(self):
        records = prepare_chunks([chunk("x" * 300)], Tokenizer(), 80, 10, "hash", "manual.pdf")
        self.assertTrue(all(r["pages"] == [2, 3] and r["page"] is None for r in records))
        self.assertTrue(all(len(r["provenance"]) == 2 for r in records))
        self.assertEqual(records[0]["provenance"][1]["positions"][0]["bbox"]["l"], 10)
        self.assertEqual(len({r["chunk_id"] for r in records}), len(records))
        self.assertEqual(records, prepare_chunks([chunk("x" * 300)], Tokenizer(), 80, 10,
                                                 "hash", "manual.pdf"))

    def test_missing_provenance_is_unknown(self):
        record = prepare_chunks([chunk(pages=())], Tokenizer(), 80, 0, "hash", "manual.pdf")[0]
        self.assertEqual(record["pages"], [])
        self.assertIsNone(record["page"])

    def test_oversized_headings_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Headings"):
            list(bounded_parts("hello", "h" * 100, Tokenizer(), 80, 10))

    def test_overlap_larger_than_payload_still_progresses(self):
        parts = list(bounded_parts("x" * 100, "h" * 65, Tokenizer(), 80, 70))
        self.assertEqual(parts[-1][1], 100)
        self.assertTrue(all(b[0] > a[0] for a, b in zip(parts, parts[1:])))


class RetrievalTests(unittest.TestCase):
    def test_dense_never_loads_reranker_and_retains_cosine(self):
        with patch("src.vector_store.load_reranker", side_effect=AssertionError("must not load")):
            store = VectorStore(Embedder(), RetrievalConfig(rerank=False, top_k=1,
                                candidate_k=2, chunk_tokens=240, overlap_tokens=0))
            store.build_index([chunk("Debug debug", (7,), ()), chunk("Voltage voltage", (8,), ())], "hash")
            hit = store.search("Debug debug")[0]
        self.assertEqual(hit["pages"], [7])
        self.assertAlmostEqual(hit["score"], 1.0, places=5)
        self.assertEqual(hit["score_kind"], "cosine")
        self.assertEqual(store.effective_chunk_tokens, 80)

    def test_reranking_changes_order_and_preserves_dense_score(self):
        store = VectorStore(Embedder(), RetrievalConfig(top_k=1, candidate_k=2,
                            chunk_tokens=80, overlap_tokens=0), reranker=Reranker())
        store.build_index([chunk("Debug debug", (7,), ()), chunk("Voltage voltage", (8,), ())], "hash")
        hit = store.search("Debug debug")[0]
        self.assertEqual(hit["pages"], [8])
        self.assertEqual(hit["score"], 10.0)
        self.assertLess(hit["initial_score"], 1.0)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(top_k=10, candidate_k=2)

    def test_long_query_is_rejected_instead_of_truncated(self):
        store = VectorStore(Embedder(), RetrievalConfig(rerank=False))
        store.build_index([chunk()], "hash")
        with self.assertRaisesRegex(ValueError, "query exceeds"):
            store.search("q" * 100)


class ChatTests(unittest.TestCase):
    def planner(self, client=None, budget=10000):
        return HardwarePlanner(client=client or FakeClient(), counter=Counter(),
                               config=GenerationConfig(context_tokens=budget))

    def test_benchmark_cases_have_fresh_messages_and_record_usage(self):
        planner = self.planner()
        store = FakeStore()
        for i in range(2):
            result = evaluate_case(planner, store, {"id": str(i), "objective": f"Objective {i}"}, 1)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["call"]["messages"]), 2)
            self.assertEqual(result["call"]["usage"]["total_tokens"], 122)
            self.assertIsNone(result["cost"])
            self.assertGreaterEqual(result["call"]["generation_seconds"], 0)
        self.assertNotIn("Objective 0", planner.client.requests[1]["messages"][-1]["content"])

    def test_document_change_resets_history(self):
        planner = self.planner()
        store = FakeStore()
        planner.chat("First objective", store)
        store.document_id = "manual-B"
        planner.chat("Second objective", store)
        self.assertEqual(len(planner.client.requests[-1]["messages"]), 2)

    def test_old_excerpts_are_not_saved_in_chat_history(self):
        planner = self.planner()
        planner.chat("First objective", FakeStore())
        self.assertEqual(planner.chat_history[0]["content"], "First objective")
        self.assertNotIn("Debug disabled", json.dumps(planner.chat_history))

    def test_failure_does_not_commit_history(self):
        planner = self.planner(FakeClient(fail=True))
        with self.assertRaises(RuntimeError):
            planner.chat("Objective", FakeStore())
        self.assertEqual(planner.chat_history, [])
        self.assertEqual(planner.calls[-1]["status"], "failed")

    def test_truncation_is_flagged_and_partial_output_retained(self):
        planner = self.planner(FakeClient(finish="length"))
        with self.assertRaises(GenerationError):
            planner.chat("Objective", FakeStore())
        self.assertEqual(planner.chat_history, [])
        self.assertEqual(planner.calls[-1]["output"], "Proposed answer")
        self.assertEqual(planner.calls[-1]["finish_reason"], "length")

    def test_whole_evidence_records_obey_context_budget(self):
        planner = self.planner(budget=5)
        _, evidence = planner.chat("Objective", FakeStore())
        self.assertEqual(evidence, [])
        self.assertEqual(planner.calls[-1]["context_token_count_estimate"], 2)

    def test_case_failure_is_recorded_without_stopping_next_case(self):
        planner = self.planner(FakeClient(fail=True))
        failed = evaluate_case(planner, FakeStore(), {"id": "bad", "objective": "Bad"}, 1)
        planner.client.fail = False
        passed = evaluate_case(planner, FakeStore(), {"id": "good", "objective": "Good"}, 1)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(passed["status"], "completed")
        self.assertEqual(len(passed["call"]["messages"]), 2)

    def test_benchmark_requires_explicit_labels_and_unique_ids(self):
        case = {"id": "a", "objective": "test", "pdf": "manual.pdf", "split": "dev",
                "answerable": False, "reference_evidence": [], "expected_behaviors": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            data = {"dataset_id": "d", "version": "1", "cases": [case]}
            path.write_text(json.dumps(data))
            self.assertEqual(read_benchmark(path)["cases"][0]["id"], "a")
            data["cases"].append(case)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_benchmark(path)

    def test_runner_writes_both_variants_and_shared_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manual.pdf").write_bytes(b"synthetic-pdf")
            dataset = {"dataset_id": "test", "version": "1", "annotation_status": "human_reviewed",
                       "cases": [{"id": "a", "objective": "debug", "pdf": "manual.pdf", "split": "dev",
                                  "answerable": True, "reference_evidence": [], "expected_behaviors": []}]}
            path = root / "cases.json"
            path.write_text(json.dumps(dataset))
            args = NS(benchmark=str(path), output=str(root / "results"), split="dev", repeats=2,
                      variant="both", top_k=1, candidate_k=2, chunk_tokens=80, overlap_tokens=0,
                      model="fake", max_output_tokens=100, context_tokens=10000, max_pages=100)
            planner = self.planner()
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}), \
                 patch("experiment_runner.HardwarePlanner", return_value=planner), \
                 patch("experiment_runner.load_embedder", return_value=Embedder()), \
                 patch("src.vector_store.load_reranker", return_value=Reranker()), \
                 patch("src.pdf_processor.PDFProcessor.extract_document",
                       return_value=(NS(pages={1: object()}), [chunk(headings=())], 1)):
                run(args)
            records = [json.loads(line) for line in (root / "results/results.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual({r["variant"] for r in records}, {"dense", "dense_rerank"})
            self.assertTrue(all(len(r["call"]["messages"]) == 2 for r in records))
            self.assertEqual(len(list((root / "results").glob("evidence-*.json"))), 1)
            manifest = json.loads((root / "results/manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertNotIn("test-key", json.dumps(manifest))

    def test_runner_rejects_unreviewed_template(self):
        with self.assertRaisesRegex(ValueError, "template labels"):
            run(NS(benchmark=Path(__file__).resolve().parents[1] / "benchmarks/example.json"))


if __name__ == "__main__":
    unittest.main()
