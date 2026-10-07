"""Bound the exact embedding input while retaining parent evidence provenance.

Subchunks retain ALL parent item references and page coordinates conservatively.
They do not pretend a particular quote was localized to one page of a multi-page
table. Character offsets are into the parent's serialized text, not PDF bytes.
"""
from hashlib import sha256


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True))


def bounded_parts(text, prefix, tokenizer, budget, overlap):
    """Split at tokenizer character offsets; verify every enriched input exactly."""
    if token_count(tokenizer, prefix) >= budget:
        raise ValueError("Headings exhaust the chunk budget; raise it or shorten headings")
    offsets = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)["offset_mapping"]
    offsets = [(a, b) for a, b in offsets if b > a]
    if not offsets:
        if text.strip():
            raise ValueError("Tokenizer returned no offsets for nonempty text")
        return
    start = 0
    while start < len(text):
        ends = sorted({b for _, b in offsets if b > start} | {len(text)})
        # Token counts can change at substring boundaries; exact validation
        # protects against this instead of relying on a character/token ratio.
        lo, hi, best = 0, len(ends) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            end = ends[mid]
            if token_count(tokenizer, prefix + text[start:end]) <= budget:
                best = end
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError("One token plus headings exceeds the embedding budget")
        content = text[start:best]
        assert token_count(tokenizer, prefix + content) <= budget
        yield start, best, content
        if best == len(text):
            break
        covered = [(a, b) for a, b in offsets if a >= start and b <= best]
        # At least one token of forward progress even when requested overlap
        # exceeds the available payload after heading injection.
        keep = min(overlap, max(0, len(covered) - 1))
        next_start = covered[-keep][0] if keep else best
        start = next_start if next_start > start else best


def prepare_chunks(doc_chunks, tokenizer, budget, overlap, document_id, source_name):
    records = []
    for ordinal, chunk in enumerate(doc_chunks):
        text = chunk.text
        if not text.strip():
            continue
        headings = list(getattr(chunk.meta, "headings", None) or [])
        captions = list(getattr(chunk.meta, "captions", None) or [])
        origin = getattr(chunk.meta, "origin", None)
        origin = origin.model_dump(mode="json") if origin is not None else None
        prefix = f"Section: {' > '.join(headings)}\n\n" if headings else ""
        provenance = []
        for item in getattr(chunk.meta, "doc_items", None) or []:
            positions = [p.model_dump(mode="json") for p in (getattr(item, "prov", None) or [])]
            provenance.append({"item_ref": getattr(item, "self_ref", None),
                               "label": str(getattr(item, "label", "unknown")),
                               "positions": positions})
        pages = sorted({p["page_no"] for item in provenance for p in item["positions"]
                        if p.get("page_no") is not None})
        parent_id = sha256(f"{document_id}:{ordinal}:{text}".encode()).hexdigest()
        for start, end, part in bounded_parts(text, prefix, tokenizer, budget, overlap):
            chunk_id = sha256(f"{parent_id}:{start}:{end}".encode()).hexdigest()
            records.append({"chunk_id": chunk_id, "parent_id": parent_id,
                            "document_id": document_id, "source_name": source_name,
                            "text": part, "content": prefix + part, "headings": headings,
                            "captions": captions, "origin": origin,
                            "pages": pages, "page": pages[0] if len(pages) == 1 else None,
                            "kind": "parsed_text", "provenance": provenance,
                            "provenance_scope": "parent_chunk",
                            "char_start": start, "char_end": end,
                            "embedding_tokens": token_count(tokenizer, prefix + part)})
    if not records:
        raise ValueError("No usable text found to index")
    return records
