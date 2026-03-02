# ============================================================
# Standard Libraries
# ============================================================
import os
import time
import random
import pickle
from collections import Counter

# ============================================================
# Third-party Libraries
# ============================================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices

# ============================================================
# Project-specific Imports
# ============================================================

from common.common import (
    prompt_generate,
    avg_pairwise_distance,
    convert_prompt_with_discretized_pI
)
from model.gpt_model import PeptidePromptGPT, PeptideTokenizer


# ============================================================
# FASTA Utilities
# ============================================================
def load_fasta(path):
    """
    Load peptide sequences from a FASTA file.

    Parameters
    ----------
    path : str
        Path to FASTA file.

    Returns
    -------
    list[str]
        List of peptide sequences.
    """
    return [str(rec.seq) for rec in SeqIO.parse(path, "fasta")]


# ============================================================
# Sequence Similarity Evaluation
# ============================================================
def best_blosum_score(
    seq,
    ref_seqs,
    matrix,
    gap_open=-10,
    gap_extend=-0.5,
):
    """
    Compute the best normalized BLOSUM alignment score of a sequence
    against a reference sequence set.

    Parameters
    ----------
    seq : str
        Query peptide sequence.
    ref_seqs : list[str]
        Reference peptide sequences.
    matrix : dict
        Substitution matrix (e.g., BLOSUM62).
    gap_open : float
        Gap opening penalty.
    gap_extend : float
        Gap extension penalty.

    Returns
    -------
    float
        Best normalized alignment score.
    """
    best_score = float("-inf")

    for ref in ref_seqs:
        alignments = pairwise2.align.globalds(
            seq, ref, matrix, gap_open, gap_extend
        )
        if not alignments:
            continue

        best_align = max(alignments, key=lambda x: x.score)
        aligned_seq, _, score, _, _ = best_align

        align_length = len(aligned_seq)
        norm_score = score / align_length

        best_score = max(best_score, norm_score)

    return best_score


# ============================================================
# Task-level Evaluation
# ============================================================
def evaluate_task_group(task_list, group_name="Group"):
    """
    Evaluate generation quality for a group of tasks.

    Parameters
    ----------
    task_list : list[tuple]
        List of (fasta_path, target_length).
    group_name : str
        Name of the task group.
    """
    print(f"\n==================== {group_name} TASK SUMMARY ====================\n")

    # --------------------------------------------------------
    # Length matching statistics
    # --------------------------------------------------------
    print(">>> Length matching statistics")

    total_count = 0
    valid_count = 0

    for path, target_len in task_list:
        seqs = load_fasta(path)

        this_total = len(seqs)
        this_valid = sum(len(s) == target_len for s in seqs)

        total_count += this_total
        valid_count += this_valid

        print(
            f"{os.path.basename(path)} | Target length: {target_len} | "
            f"Total: {this_total} | Correct: {this_valid} | "
            f"Accuracy: {this_valid / max(this_total, 1):.3f}"
        )

    overall_ratio = valid_count / max(total_count, 1)
    print(f"\n{group_name} Overall length accuracy = {overall_ratio:.4f}\n")

    # --------------------------------------------------------
    # Duplication statistics
    # --------------------------------------------------------
    print(">>> Duplication statistics")

    all_seqs = []

    for path, _ in task_list:
        seqs = load_fasta(path)

        total = len(seqs)
        counter = Counter(seqs)
        unique = len(counter)
        dup = total - unique
        dup_ratio = dup / total if total > 0 else 0

        gen_set = set(seqs)
        ref_set = set(orig_seqs)

        exact_dup = len(gen_set & ref_set)
        exact_dup_rate = exact_dup / max(len(gen_set), 1)

        print(f"{os.path.basename(path)}:")
        print(f"  Exact duplication rate: {exact_dup_rate:.3f}")
        print(f"  Duplicated vs reference: {exact_dup}/{len(gen_set)}")
        print(f"  Total sequences: {total}")
        print(f"  Unique sequences: {unique}")
        print(f"  Duplicate count: {dup}")
        print(f"  Duplication ratio: {dup_ratio:.4f}\n")

        all_seqs.extend(seqs)

    all_total = len(all_seqs)
    all_unique = len(set(all_seqs))
    all_dup = all_total - all_unique
    all_dup_ratio = all_dup / max(all_total, 1)

    print(f"=== {group_name} Overall Duplication Summary ===")
    print(f"  Total sequences: {all_total}")
    print(f"  Unique sequences: {all_unique}")
    print(f"  Duplicate sequences: {all_dup}")
    print(f"  Overall duplication ratio: {all_dup_ratio:.4f}")

    print("\n==============================================================\n")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # Device configuration
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_LEN = 40
    num_epochs = 20

    # --------------------------------------------------------
    # Token definitions
    # --------------------------------------------------------
    aas = list("ACDEFGHIKLMNPQRSTVWY")

    activity_dir = "./data/peptipedia2_15/"
    activities = [
        f.replace(".fasta", "").strip()
        for f in os.listdir(activity_dir)
        if f.endswith(".fasta")
    ]

    pi_tokens = [
        f"pI:{round(x, 1)}-{round(x + 0.5, 1)}"
        for x in np.arange(3.0, 10.0, 0.5)
    ]
    length_tokens = [f"Length:{i}" for i in range(1, 16)]

    token_dict = {
        "AA": aas,
        "Activity": activities,
        "pI": pi_tokens,
        "Length": length_tokens,
    }

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------
    TOKENIZER_PATH = "./result_outputs/trained_models/tokenizer_ga.pkl"
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    embed_dict = {}

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = PeptidePromptGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=1024,
        tokenizer=tokenizer,
        activity_embed_dict=embed_dict,
        num_layers=12,
        nhead=16,
        dim_feedforward=512,
        max_len=MAX_LEN,
        distilled_dim=480,
        contract_proj_dim=256,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "./result_outputs/trained_models/"
            "model_finetune_selfassembly_final_l12_al_2-1.pt"
        )
    )

    # --------------------------------------------------------
    # Reference data
    # --------------------------------------------------------
    orig_fasta = "./data/peptipedia2_15/Self assembly.fasta"
    orig_seqs = load_fasta(orig_fasta)

    blosum62 = substitution_matrices.load("BLOSUM62")

    # --------------------------------------------------------
    # Generation benchmarks
    # --------------------------------------------------------
    elapsed = []

    print("Self-assembly generation benchmark started")

    generation_tasks = [
        ("Self assembly+pI:5.5-6.0+Length:4+GRAVY:2.5-3.0", 500, "Sa1", 10),
        ("Self assembly+pI:5.5-6.0+Length:4+GRAVY:0.5-1.0", 375, "Sa2", 10),
        ("Self assembly+pI:3.5-4.0+Length:6+GRAVY:1.0-1.5", 400, "Sa3", 10),
        ("Self assembly+pI:3.5-4.0+Length:6+GRAVY:1.5-2.0", 225, "Sa4", 10),
        ("Self assembly+pI:8.0-8.5+Length:15+GRAVY:1.0-1.5", 125, "Sa5", 20),
        ("Self assembly+pI:8.0-8.5+Length:10+GRAVY:0.0-0.5", 100, "Sa6", 20),
        ("Self assembly+pI:6.0-6.5+Length:5+GRAVY:0.0-0.5", 75, "Sa7", 10),
        ("Self assembly+pI:6.0-6.5+Length:7+GRAVY:-3.0--2.5", 75, "Sa8", 15),
        ("Self assembly+pI:7.0-7.5+Length:12+GRAVY:-0.5-0.0", 100, "Sa9", 20),
        ("Self assembly+pI:7.0-7.5+Length:6+GRAVY:1.0-1.5", 30, "Sa10", 10),
    ]

    for prompt, num, tag, max_len in generation_tasks:
        start = time.time()
        prompt_generate(
            prompt,
            num_samples=num,
            save_path=f"./result_outputs/generated_results/scheduler_e10_{tag}.fasta",
            model=model,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        elapsed.append(time.time() - start)

    print(
        f"Average generation time: {np.mean(elapsed):.4f} s, "
        f"Std: {np.std(elapsed):.4f} s"
    )

    # --------------------------------------------------------
    # Merge FASTA outputs
    # --------------------------------------------------------
    filenames = [
        f"./result_outputs/generated_results/scheduler_e10_Sa{i+1}.fasta"
        for i in range(10)
    ]

    merged_records = []
    for fname in filenames:
        merged_records.extend(list(SeqIO.parse(fname, "fasta")))

    SeqIO.write(
        merged_records,
        "./result_outputs/generated_results/scheduler_e10_Sa_merge.fasta",
        "fasta",
    )
