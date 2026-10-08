"""Configurable dense cosine retrieval with lazy, optional reranking."""
from dataclasses import asdict, replace
import numpy as np
from src.chunking import prepare_chunks, token_count
from src.config import RetrievalConfig

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_NAME, device="cpu")


class VectorStore:
    def __init__(self, model, config=None, reranker=None):
        self.model = model
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        self.chunks = []
        self.index = None
        self.document_id = None
        self.effective_chunk_tokens = None

    def build_index(self, doc_chunks, document_id="unknown", source_name="uploaded_spec.pdf"):
        from faiss import IndexFlatIP

        # Enforce the actual loaded encoder's limit, including special tokens.
        # Overrides config if the deployed model is physically smaller.
        budget = min(self.config.chunk_tokens, self.model.max_seq_length)

        chunks = prepare_chunks(doc_chunks, self.model.tokenizer, budget,
                                self.config.overlap_tokens, document_id, source_name)

        # Standardize arrays to float32 for FAISS compatibility and normalize
        # for IndexFlatIP so that inner product calculates Cosine Similarity.
        vectors = self.model.encode([c["content"] for c in chunks],
                                    batch_size=self.config.batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)

        index = IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        # Commit atomically only after preparation and encoding succeed.
        # This prevents the object from getting stuck in a dirty state.
        self.chunks, self.index = chunks, index
        self.document_id = document_id
        self.effective_chunk_tokens = budget

    def search(self, query, final_top_k=None, initial_fetch=None):
        if self.index is None:
            raise ValueError("Build the index before searching")

        cfg = replace(self.config,
                      top_k=self.config.top_k if final_top_k is None else final_top_k,
                      candidate_k=self.config.candidate_k if initial_fetch is None else initial_fetch)

        # Reject over-length queries that the transformer will clip
        if token_count(self.model.tokenizer, query) > self.model.max_seq_length:
            raise ValueError("Retrieval query exceeds encoder token limit; use a shorter retrieval_query")

        vector = self.model.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        # Execute approximate nearest neighbors
        scores, indices = self.index.search(vector, min(cfg.candidate_k, len(self.chunks)))

        hits = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                # Map back to source documentation payloads safely
                hit = dict(self.chunks[index])
                hit.update(initial_score=float(score), score=float(score), score_kind="cosine")
                hits.append(hit)

        if cfg.rerank and hits:
            # Lazy load cross-encoder to conserve memory if unused
            if self.reranker is None:
                self.reranker = load_reranker()

            tokenizer = getattr(self.reranker, "tokenizer", None)
            pair_limit = getattr(self.reranker, "max_length", None)

            # Additional safety: Verify the query + chunk don't exceed the reranker window
            if tokenizer is not None and pair_limit is not None:
                for hit in hits:
                    count = len(tokenizer.encode(query, hit["content"], add_special_tokens=True))
                    if count > pair_limit:
                        raise ValueError("Reranker pair exceeds token limit; shorten query or chunk budget")

            values = self.reranker.predict([[query, h["content"]] for h in hits])
            for hit, score in zip(hits, values):
                hit.update(score=float(score), score_kind="cross_encoder")

            # Override hits with newly ranked precision sorts
            hits.sort(key=lambda h: h["score"], reverse=True)

        return hits[:cfg.top_k]

    def settings(self):
        # Extract actual model revision hashes for deployment traceability
        try:
            encoder_revision = self.model._first_module().auto_model.config._commit_hash
        except AttributeError:
            encoder_revision = None

        reranker_model = getattr(self.reranker, "model", None)
        reranker_revision = getattr(getattr(reranker_model, "config", None), "_commit_hash", None)

        return {**asdict(self.config), "embedding_model": MODEL_NAME,
                "reranker_model": RERANKER_NAME if self.config.rerank else None,
                "embedding_revision": encoder_revision,
                "reranker_revision": reranker_revision if self.config.rerank else None,
                "effective_chunk_tokens": self.effective_chunk_tokens,
                "embedding_max_seq_length": self.model.max_seq_length,
                "index": "FAISS IndexFlatIP; normalized embeddings"}