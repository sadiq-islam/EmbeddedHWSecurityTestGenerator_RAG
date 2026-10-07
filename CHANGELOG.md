# Updated baseline: 2026-10-05

- Added exact encoder-budget checks after heading injection and special tokens.
- Added token-offset splitting, overlap, deterministic IDs and all parent provenance.
- Removed missing-provenance fallback to page 1.
- Made reranking lazy/optional and retrieval settings explicit.
- Added query and reranker-pair overflow checks.
- Reset PDF-dependent UI/history/draft state on changed or removed uploads.
- Bound conversational history; retrieved context is supplied only for the current turn.
- Added fresh independent benchmark cases and matched dense/reranked runs.
- Persisted dataset/configuration/source hashes, evidence, request messages, output,
  elapsed timing, returned usage, finish reason, and available local model revisions.
- Replaced obsolete workflow/UI tests with tests for the current interfaces.
- Fixed Markdown fence guidance and made incomplete outputs explicit failures.
- Retained free-form output: schema/claim grounding still needs separate enforcement
  and human review; no quality scores or hardware outcomes are claimed.
