# Validation performed

Date: 2026-10-05. Runtime: Linux, Python 3.12.

Command:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Result: **21 tests passed**.

Coverage:

- Exact embedding budgets, headings and special-token accounting.
- Long technical strings, overlap progress, complete character coverage.
- All parent page/item/coordinate provenance and unknown-page handling.
- Actual Docling `DocChunk`/`DocMeta`/`ProvenanceItem` objects with a locally
  constructed Hugging Face fast WordPiece tokenizer; no weights downloaded.
- Real FAISS dense ranking and fake reranker ordering; dense score retention.
- Reranking off avoids reranker loading; oversized queries fail explicitly.
- Fresh benchmark cases, per-call timing/provider-token logging, no stale excerpts.
- Document-change history reset, failed-call rollback, truncated-output retention.
- Whole-record generation-context packing and benchmark-label validation.
- Runner writes both variants and shared evidence snapshots using fake inference.
- Streamlit processing and upload-change/removal reset using mocked conversion.

CLI help also executed successfully. Python source syntax was checked.

Selected tested dependency versions: FAISS CPU 1.15.1, Streamlit 1.65.0,
Docling Core 2.99.0, Transformers 5.18.0, Groq SDK 1.7.0. These are observations,
not a full validated production lockfile. A Docling captions-field deprecation
warning and Streamlit bare-context warnings were observed; tests passed.

Not performed: full Docling PDF conversion/OCR, real embedding/reranker inference,
Groq requests, actual Qwen output assessment, Windows launcher execution, memory
or cost measurement, semantic grounding annotation, or hardware execution.

No benchmark quality scores are claimed. Fake provider usage in regression tests
is a test fixture and is not presented as a real experiment measurement.
