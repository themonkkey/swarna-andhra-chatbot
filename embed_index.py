"""Build the semantic (embedding) index for the Swarna Andhra RAG chatbot.

Reuses the exact chunking from ingest.py, then embeds every chunk with the
configured provider (Gemini free tier by default) and saves:

  embed_index.npz   float16 embedding matrix (n, dim)  -- compact, deploy-friendly
  embed_chunks.pkl  the chunk list + model id + dim

Query-time cosine similarity is a plain dot product because vectors are
L2-normalized at embed time. Matrix is stored float16 to keep it small enough
for a free-tier host's RAM; it is cast back to float32 for the dot product.

Run:  EMBED_PROVIDER=gemini GEMINI_API_KEY=... python embed_index.py
Resumable: partial progress is checkpointed, so a rate-limit stop can be re-run.
"""
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import embeddings
from ingest import build_all_chunks

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE, "embed_index.npz")
CHUNKS_PATH = os.path.join(BASE, "embed_chunks.pkl")
CKPT_PATH = os.path.join(BASE, ".embed_ckpt.npy")
RAW_CHUNKS_CACHE = os.path.join(BASE, ".raw_chunks_cache.pkl")

BATCH = int(os.environ.get("EMBED_BATCH", "96"))
MAX_RETRIES = 8
# Each Cohere trial key allows ~100k tokens/min. With avg ~330-token chunks, a batch
# of 96 is ~32k tokens, so ~2.6 batches/min/key stays safely under. Multiple keys
# (COHERE_API_KEYS, comma-separated) run as parallel lanes -> throughput scales ~linearly.
PER_KEY_BATCHES_PER_MIN = float(os.environ.get("EMBED_PER_KEY_BPM", "2.6"))


def get_keys():
    multi = os.environ.get("COHERE_API_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("COHERE_API_KEY", "").strip()
    return [single] if single else [None]


def embed_batch_with_retry(texts, api_key):
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed(texts, is_query=False, api_key=api_key)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def main():
    # PDF text extraction is the expensive part (~45min for this corpus) — cache it
    # so an embedding-provider failure never forces redoing it.
    if os.path.exists(RAW_CHUNKS_CACHE):
        print(f"Loading cached extracted chunks from {RAW_CHUNKS_CACHE}")
        with open(RAW_CHUNKS_CACHE, "rb") as f:
            chunks = pickle.load(f)
    else:
        chunks = build_all_chunks()
        if chunks:
            with open(RAW_CHUNKS_CACHE, "wb") as f:
                pickle.dump(chunks, f)
            print(f"Cached {len(chunks)} extracted chunks -> {RAW_CHUNKS_CACHE}")

    if not chunks:
        print("No chunks extracted.")
        return
    texts = [c["text"] for c in chunks]
    n = len(texts)
    mid = embeddings.model_id()
    keys = get_keys()
    print(f"\nEmbedding {n} chunks via {mid}, {len(keys)} key lane(s), batch {BATCH}...")

    # dim: probe once so we can allocate the matrix and detect a resumable checkpoint
    probe = embed_batch_with_retry(texts[:1], keys[0])
    dim = probe.shape[1]
    vecs = np.zeros((n, dim), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)
    if os.path.exists(CKPT_PATH):
        saved = np.load(CKPT_PATH)
        m = min(saved.shape[0], n)
        vecs[:m] = saved[:m]
        # a saved row is "done" if it's non-zero (embeddings are never all-zero)
        done_mask[:m] = np.any(saved[:m] != 0, axis=1)
        print(f"Resuming: {int(done_mask.sum())}/{n} already embedded")

    # batch start indices still needing work
    todo = [i for i in range(0, n, BATCH) if not done_mask[i]]
    total_batches = (n + BATCH - 1) // BATCH
    min_secs_per_batch = 60.0 / PER_KEY_BATCHES_PER_MIN
    lock = threading.Lock()
    counter = {"done": int(done_mask.sum()), "batches": 0}
    t0 = time.time()

    def worker(key, my_batches):
        for i in my_batches:
            req_t0 = time.time()
            emb = embed_batch_with_retry(texts[i : i + BATCH], key)
            with lock:
                vecs[i : i + emb.shape[0]] = emb
                counter["done"] += emb.shape[0]
                counter["batches"] += 1
                d = counter["done"]
                b = counter["batches"]
                if b % 10 == 0 or d >= n:
                    rate = d / max(time.time() - t0, 1e-9)
                    eta = (n - d) / max(rate, 1e-9)
                    print(f"  {d}/{n}  (eta {eta/60:.1f}m)", flush=True)
                    np.save(CKPT_PATH, vecs)
            # pace THIS key's lane under its own token/min budget
            elapsed = time.time() - req_t0
            if elapsed < min_secs_per_batch:
                time.sleep(min_secs_per_batch - elapsed)

    # round-robin the outstanding batches across keys
    lanes = {ki: [] for ki in range(len(keys))}
    for idx, i in enumerate(todo):
        lanes[idx % len(keys)].append(i)

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        futures = [pool.submit(worker, keys[ki], lanes[ki]) for ki in range(len(keys))]
        for f in futures:
            f.result()

    np.save(CKPT_PATH, vecs)
    np.savez_compressed(NPZ_PATH, matrix=vecs.astype(np.float16))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "model_id": mid, "dim": vecs.shape[1]}, f)
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    size_mb = os.path.getsize(NPZ_PATH) / 1e6
    print(f"\nDone. {n} chunks, dim {vecs.shape[1]}, matrix {size_mb:.1f} MB")
    print(f"  {NPZ_PATH}\n  {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
