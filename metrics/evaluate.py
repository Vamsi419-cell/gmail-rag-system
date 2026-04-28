"""
Gmail RAG System - Evaluation Metrics
Reads REAL data from the trained FAISS index + data files.
Generates scores + graphs for IEEE paper.

Usage:  cd gmail-rag-system/metrics && python evaluate.py
"""

import sys, os, json, time, math, re
from collections import Counter
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Paths ──
PROJECT = Path(__file__).resolve().parent.parent
INDEX_PATH   = PROJECT / "models" / "email_index.faiss"
CHUNKS_PATH  = PROJECT / "data"   / "chunks.json"
META_PATH    = PROJECT / "data"   / "metadata.json"
OUT_DIR      = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ──
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


# ══════════════════════════════════════════════
#  1. LOAD REAL DATA (just FAISS + files, no model)
# ══════════════════════════════════════════════
def load_real_data():
    """Load the actual trained FAISS index, chunks, and metadata."""
    import faiss

    print("Loading real data...")

    assert INDEX_PATH.exists(), f"FAISS index not found at {INDEX_PATH}"
    index = faiss.read_index(str(INDEX_PATH))
    n_vectors = index.ntotal
    dim = index.d
    print(f"  FAISS index: {n_vectors} vectors, {dim}-dim, size={INDEX_PATH.stat().st_size/1024:.1f}KB")

    # Extract actual vectors stored in the index
    vectors = faiss.rev_swig_ptr(index.get_xb(), n_vectors * dim).reshape(n_vectors, dim).copy()

    chunks, metadata = [], []
    if CHUNKS_PATH.exists() and META_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Chunks: {len(chunks)}  |  Metadata entries: {len(metadata)}")
    else:
        print("  [!] chunks.json / metadata.json not found")
        print("      (Run 'python src/process.py' first to generate them)")
        print("      Proceeding with embedding-only metrics from FAISS index...")

    return index, vectors, chunks, metadata, n_vectors, dim


# ══════════════════════════════════════════════
#  2. RETRIEVAL METRICS  (Precision@K, Recall@K, MRR, NDCG@K)
# ══════════════════════════════════════════════
def eval_retrieval(index, vectors, chunks, metadata):
    """
    Evaluate retrieval using REAL vectors from the index.
    Uses actual email chunks as queries (leave-one-out style).
    """
    print("\n── Retrieval Metrics ──")

    if not chunks or not metadata:
        print("  Skipped (no chunks/metadata)")
        return None

    # Build ground truth: chunks from the same email are relevant to each other
    email_groups = {}
    for i, m in enumerate(metadata):
        eid = m.get("id", str(i))
        email_groups.setdefault(eid, []).append(i)

    # Test: use each email's first chunk as query, other chunks as relevant
    K_VALUES = [1, 3, 5, 10]
    pk = {k: [] for k in K_VALUES}
    rk = {k: [] for k in K_VALUES}
    ndcg = {k: [] for k in K_VALUES}
    mrr_scores = []

    n_queries = 0
    for eid, idxs in email_groups.items():
        if len(idxs) < 1:
            continue
        q_idx = idxs[0]
        # All chunks from same email are relevant
        relevant = set(idxs)

        q_vec = vectors[q_idx:q_idx+1].copy()
        distances, indices = index.search(q_vec, max(K_VALUES) + 1)
        # Remove self from results
        retrieved = [int(i) for i in indices[0] if i != q_idx][:max(K_VALUES)]

        # MRR
        rr = 0.0
        for rank, idx in enumerate(retrieved):
            if idx in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        for k in K_VALUES:
            top_k = retrieved[:k]
            hits = sum(1 for i in top_k if i in relevant)
            pk[k].append(hits / k)
            rk[k].append(hits / max(len(relevant) - 1, 1))  # -1 for query itself
            # NDCG
            dcg = sum((1.0 if top_k[i] in relevant else 0.0) / math.log2(i + 2)
                      for i in range(len(top_k)))
            ideal = sum(1.0 / math.log2(i + 2)
                        for i in range(min(k, max(len(relevant) - 1, 1))))
            ndcg[k].append(dcg / ideal if ideal > 0 else 0)

        n_queries += 1

    avg = {}
    avg["MRR"] = round(float(np.mean(mrr_scores)), 4)
    for k in K_VALUES:
        avg[f"P@{k}"]    = round(float(np.mean(pk[k])), 4)
        avg[f"R@{k}"]    = round(float(np.mean(rk[k])), 4)
        avg[f"NDCG@{k}"] = round(float(np.mean(ndcg[k])), 4)

    print(f"  Queries evaluated: {n_queries}")
    for key in ["MRR"] + [f"P@{k}" for k in K_VALUES]:
        print(f"  {key:>8s}: {avg[key]:.4f}")

    return avg, K_VALUES


# ══════════════════════════════════════════════
#  3. EMBEDDING METRICS  (from real FAISS vectors)
# ══════════════════════════════════════════════
def eval_embeddings(vectors, n_vectors, dim):
    """Evaluate embedding quality from real stored vectors."""
    print("\n── Embedding Metrics ──")

    # Pairwise cosine similarity (sample pairs)
    n_pairs = min(500, n_vectors * (n_vectors - 1) // 2)
    sims = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_vectors, 2, replace=False)
        sim = float(np.dot(vectors[i], vectors[j]))  # normalized = cosine sim
        sims.append(sim)

    sim_stats = {
        "mean": round(float(np.mean(sims)), 4),
        "std":  round(float(np.std(sims)), 4),
        "min":  round(float(np.min(sims)), 4),
        "max":  round(float(np.max(sims)), 4),
    }

    # SVD for effective dimensionality
    centered = vectors - vectors.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    cumulative = np.cumsum(explained)
    dims_95 = int(np.searchsorted(cumulative, 0.95) + 1)

    norms = np.linalg.norm(vectors, axis=1)

    metrics = {
        "n_vectors": n_vectors,
        "dimension": dim,
        "similarity": sim_stats,
        "dims_for_95pct_variance": dims_95,
        "norm_mean": round(float(np.mean(norms)), 4),
        "top5_variance": [round(float(v), 4) for v in explained[:5]],
        "_sims": sims,
    }

    print(f"  Vectors: {n_vectors} x {dim}")
    print(f"  Cosine Sim: mean={sim_stats['mean']:.4f}  std={sim_stats['std']:.4f}")
    print(f"  Dims for 95% variance: {dims_95}")

    return metrics


# ══════════════════════════════════════════════
#  4. LATENCY METRICS (real FAISS search timing)
# ══════════════════════════════════════════════
def eval_latency(index, vectors, n_vectors):
    """Measure real search latency on the trained index."""
    print("\n── Latency Metrics ──")

    # Use random real vectors as queries
    q_indices = np.random.choice(n_vectors, size=min(20, n_vectors), replace=False)
    search_times = []

    for qi in q_indices:
        q_vec = vectors[qi:qi+1].copy()
        t0 = time.perf_counter()
        index.search(q_vec, 5)
        search_times.append(time.perf_counter() - t0)

    lat = {
        "mean_ms": round(float(np.mean(search_times)) * 1000, 3),
        "p95_ms":  round(float(np.percentile(search_times, 95)) * 1000, 3),
        "min_ms":  round(float(np.min(search_times)) * 1000, 3),
        "max_ms":  round(float(np.max(search_times)) * 1000, 3),
        "n_queries": len(search_times),
    }

    print(f"  Mean:  {lat['mean_ms']:.3f} ms")
    print(f"  P95:   {lat['p95_ms']:.3f} ms")
    print(f"  Range: [{lat['min_ms']:.3f}, {lat['max_ms']:.3f}] ms")

    return lat, search_times


# ══════════════════════════════════════════════
#  5. GENERATION METRICS  (BLEU, ROUGE)
# ══════════════════════════════════════════════
def _get_ngrams(text, n):
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

def rouge_n(reference, hypothesis, n=1):
    """ROUGE-N: n-gram overlap."""
    ref_ng = Counter(_get_ngrams(reference, n))
    hyp_ng = Counter(_get_ngrams(hypothesis, n))
    overlap = sum((ref_ng & hyp_ng).values())
    ref_count = sum(ref_ng.values())
    hyp_count = sum(hyp_ng.values())
    p = overlap / hyp_count if hyp_count else 0
    r = overlap / ref_count if ref_count else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return round(f1, 4)

def rouge_l(reference, hypothesis):
    """ROUGE-L: longest common subsequence."""
    ref_w = reference.lower().split()
    hyp_w = hypothesis.lower().split()
    m, n = len(ref_w), len(hyp_w)
    if m == 0 or n == 0: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref_w[i-1]==hyp_w[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n; r = lcs/m
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return round(f1, 4)

def bleu_score(reference, hypothesis, max_n=4):
    """BLEU score."""
    ref_w = reference.lower().split()
    hyp_w = hypothesis.lower().split()
    if not hyp_w: return 0.0
    bp = min(1.0, math.exp(1 - len(ref_w)/len(hyp_w))) if hyp_w else 0
    log_avg = 0; valid = 0
    for n in range(1, max_n+1):
        ref_ng = Counter(_get_ngrams(reference, n))
        hyp_ng = Counter(_get_ngrams(hypothesis, n))
        clipped = sum((hyp_ng & ref_ng).values())
        total = sum(hyp_ng.values())
        if total > 0 and clipped > 0:
            log_avg += math.log(clipped/total) / max_n
            valid += 1
    return round(bp * math.exp(log_avg), 4) if valid == max_n else 0.0

def eval_generation(index, vectors, chunks, metadata):
    """
    Compute BLEU & ROUGE using real data.
    Method: For each email, use its first chunk as query, retrieve top-5,
    compare retrieved text against the actual email text (ground truth).
    """
    print("\n── Generation Metrics (BLEU / ROUGE) ──")

    if not chunks or not metadata:
        print("  Skipped (no chunks)")
        return None

    email_groups = {}
    for i, m in enumerate(metadata):
        email_groups.setdefault(m.get("id", str(i)), []).append(i)

    bleu_scores = []
    rouge1_scores = []
    rougel_scores = []

    for eid, idxs in email_groups.items():
        if len(idxs) < 1:
            continue

        # Ground truth = all chunks of this email concatenated
        reference = " ".join(chunks[i] for i in idxs)

        # Query = first chunk's vector
        q_vec = vectors[idxs[0]:idxs[0]+1].copy()
        _, ret_indices = index.search(q_vec, 6)
        # Retrieved text (exclude self)
        ret_text_parts = []
        for ri in ret_indices[0]:
            ri = int(ri)
            if ri not in idxs and 0 <= ri < len(chunks):
                ret_text_parts.append(chunks[ri])
        if not ret_text_parts:
            continue
        hypothesis = " ".join(ret_text_parts[:3])

        bleu_scores.append(bleu_score(reference, hypothesis))
        rouge1_scores.append(rouge_n(reference, hypothesis, 1))
        rougel_scores.append(rouge_l(reference, hypothesis))

    gen = {
        "BLEU":    round(float(np.mean(bleu_scores)), 4),
        "ROUGE-1": round(float(np.mean(rouge1_scores)), 4),
        "ROUGE-L": round(float(np.mean(rougel_scores)), 4),
        "n_samples": len(bleu_scores),
    }

    print(f"  Samples: {gen['n_samples']}")
    print(f"  BLEU:    {gen['BLEU']:.4f}")
    print(f"  ROUGE-1: {gen['ROUGE-1']:.4f}")
    print(f"  ROUGE-L: {gen['ROUGE-L']:.4f}")

    return gen


# ══════════════════════════════════════════════
#  6. GRAPHS
# ══════════════════════════════════════════════
def make_graphs(retrieval, emb, lat_times, gen):
    print("\n── Generating Graphs ──")

    # 1: Precision / Recall / NDCG at K
    if retrieval:
        avg, KV = retrieval
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(KV)); w = 0.25
        ax.bar(x - w, [avg[f"P@{k}"] for k in KV], w, label='Precision@K', color=COLORS[0])
        ax.bar(x,     [avg[f"R@{k}"] for k in KV], w, label='Recall@K',    color=COLORS[1])
        ax.bar(x + w, [avg[f"NDCG@{k}"] for k in KV], w, label='NDCG@K',   color=COLORS[3])
        ax.set_xticks(x); ax.set_xticklabels([f'K={k}' for k in KV])
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
        ax.set_title('Retrieval Performance at K'); ax.legend()
        for bars in ax.containers:
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.3f}', ha='center', fontsize=8)
        fig.savefig(OUT_DIR / "1_retrieval_at_k.png", bbox_inches='tight'); plt.close()
        print("  Saved: 1_retrieval_at_k.png")

        # 2: MRR
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['MRR'], [avg['MRR']], color=COLORS[0], width=0.4)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
        ax.set_title('Mean Reciprocal Rank')
        ax.text(0, avg['MRR']+0.02, f"{avg['MRR']:.4f}", ha='center', fontsize=12)
        fig.savefig(OUT_DIR / "2_mrr.png", bbox_inches='tight'); plt.close()
        print("  Saved: 2_mrr.png")

    # 3: Similarity distribution
    if emb:
        sims = emb["_sims"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sims, bins=25, color=COLORS[0], alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(sims), color=COLORS[3], ls='--', lw=2, label=f'Mean={np.mean(sims):.4f}')
        ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Frequency')
        ax.set_title('Embedding Pairwise Cosine Similarity'); ax.legend()
        fig.savefig(OUT_DIR / "3_similarity_dist.png", bbox_inches='tight'); plt.close()
        print("  Saved: 3_similarity_dist.png")

        # 4: PCA variance
        fig, ax = plt.subplots(figsize=(7, 4))
        evr = emb["top5_variance"]
        ax.bar(range(1, len(evr)+1), evr, color=COLORS[4], alpha=0.8)
        ax.plot(range(1, len(evr)+1), np.cumsum(evr), 'o-', color=COLORS[3], lw=2, label='Cumulative')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance Explained')
        ax.set_title('Top-5 PCA Variance'); ax.legend()
        fig.savefig(OUT_DIR / "4_pca_variance.png", bbox_inches='tight'); plt.close()
        print("  Saved: 4_pca_variance.png")

    # 5: Latency
    if lat_times:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([t*1000 for t in lat_times], bins=10, color=COLORS[1], alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(lat_times)*1000, color=COLORS[3], ls='--', lw=2,
                   label=f'Mean={np.mean(lat_times)*1000:.3f}ms')
        ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count')
        ax.set_title('FAISS Search Latency Distribution'); ax.legend()
        fig.savefig(OUT_DIR / "5_latency.png", bbox_inches='tight'); plt.close()
        print("  Saved: 5_latency.png")

    # 6: Generation quality (BLEU / ROUGE)
    if gen:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = ['BLEU', 'ROUGE-1', 'ROUGE-L']
        vals = [gen[n] for n in names]
        bars = ax.bar(names, vals, color=[COLORS[0], COLORS[1], COLORS[4]], width=0.45, alpha=0.85)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
        ax.set_title('Generation Quality Metrics')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.4f}', ha='center', fontsize=10)
        fig.savefig(OUT_DIR / "6_generation_quality.png", bbox_inches='tight'); plt.close()
        print("  Saved: 6_generation_quality.png")

    # 7: Radar overview
    if retrieval:
        avg = retrieval[0]
        cats = ['MRR', 'P@5', 'R@5', 'NDCG@5']
        vals = [avg.get(c, 0) for c in cats]
        if gen:
            cats += ['BLEU', 'ROUGE-1']
            vals += [gen['BLEU'], gen['ROUGE-1']]
        angles = [n/len(cats)*2*np.pi for n in range(len(cats))] + [0]
        vals_p = vals + [vals[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, vals_p, 'o-', lw=2, color=COLORS[0])
        ax.fill(angles, vals_p, alpha=0.15, color=COLORS[0])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats)
        ax.set_ylim(0, 1); ax.set_title('RAG Performance Overview', pad=20)
        fig.savefig(OUT_DIR / "7_radar.png", bbox_inches='tight'); plt.close()
        print("  Saved: 7_radar.png")


# ══════════════════════════════════════════════
#  6. REPORT (one TXT + one JSON)
# ══════════════════════════════════════════════
def save_report(retrieval, emb, latency, gen, n_vectors, dim):
    lines = []
    lines.append("=" * 55)
    lines.append("  Gmail RAG System — Evaluation Report")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  Data Source: Real trained FAISS index")
    lines.append("=" * 55)

    lines.append("\nSYSTEM")
    lines.append(f"  Embedding Model : all-MiniLM-L6-v2 ({dim}-dim)")
    lines.append(f"  Cross-Encoder   : ms-marco-MiniLM-L-6-v2")
    lines.append(f"  Vector Index    : FAISS IndexFlatIP ({n_vectors} vectors)")
    lines.append(f"  LLM             : Groq - LLaMA 3.3 70B")
    lines.append(f"  Index Size      : {INDEX_PATH.stat().st_size/1024:.1f} KB")

    if retrieval:
        avg = retrieval[0]
        lines.append("\nRETRIEVAL METRICS")
        for k in ["MRR", "P@1", "P@3", "P@5", "P@10", "R@1", "R@3", "R@5", "R@10",
                   "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]:
            if k in avg:
                lines.append(f"  {k:>8s} : {avg[k]:.4f}")

    if emb:
        s = emb["similarity"]
        lines.append("\nEMBEDDING METRICS")
        lines.append(f"  Cosine Sim Mean  : {s['mean']:.4f}")
        lines.append(f"  Cosine Sim Std   : {s['std']:.4f}")
        lines.append(f"  Cosine Sim Range : [{s['min']:.4f}, {s['max']:.4f}]")
        lines.append(f"  95% Var Dims     : {emb['dims_for_95pct_variance']}")

    if gen:
        lines.append("\nGENERATION METRICS")
        lines.append(f"  BLEU             : {gen['BLEU']:.4f}")
        lines.append(f"  ROUGE-1          : {gen['ROUGE-1']:.4f}")
        lines.append(f"  ROUGE-L          : {gen['ROUGE-L']:.4f}")

    if latency:
        lines.append("\nLATENCY METRICS")
        lines.append(f"  Search Mean      : {latency['mean_ms']:.3f} ms")
        lines.append(f"  Search P95       : {latency['p95_ms']:.3f} ms")
        lines.append(f"  Search Range     : [{latency['min_ms']:.3f}, {latency['max_ms']:.3f}] ms")

    report = "\n".join(lines)

    with open(OUT_DIR / "evaluation_report.txt", "w") as f:
        f.write(report)

    json_data = {}
    if retrieval: json_data["retrieval"] = retrieval[0]
    if gen: json_data["generation"] = gen
    if emb:
        json_data["embeddings"] = {k: v for k, v in emb.items() if k != "_sims"}
    if latency: json_data["latency"] = latency
    json_data["system"] = {"vectors": n_vectors, "dim": dim}

    with open(OUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n{report}")
    print(f"\nFiles saved in: {OUT_DIR}")


# ══════════════════════════════════════════════
if __name__ == "__main__":
    print()
    index, vectors, chunks, metadata, n_vecs, dim = load_real_data()
    ret = eval_retrieval(index, vectors, chunks, metadata)
    emb = eval_embeddings(vectors, n_vecs, dim)
    lat, lat_times = eval_latency(index, vectors, n_vecs)
    gen = eval_generation(index, vectors, chunks, metadata)
    make_graphs(ret, emb, lat_times, gen)
    save_report(ret, emb, lat, gen, n_vecs, dim)
    print("\n✅ Done!")
