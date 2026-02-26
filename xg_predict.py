import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import Levenshtein

from common.xg_features import (
    extract_all_features_ids_batch_block,
    replace_j_with_il,
)


# ============================================================
# Prediction Utilities
# ============================================================
def predict_fasta_file(fasta_path, model_path, scaler_path, fids):
    """
    Predict labels and probabilities for protein sequences in a FASTA file.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the FASTA file containing protein sequences.
    model_path : str or Path
        Path to the trained classification model.
    scaler_path : str or Path
        Path to the fitted feature scaler.
    fids : list of str
        Feature IDs used during training (must match training configuration).

    Returns
    -------
    pd.DataFrame
        DataFrame containing sequence IDs, sequences, predicted labels,
        and prediction probabilities.
    """
    # ------------------------------------------------------------
    # 1. Load model and scaler
    # ------------------------------------------------------------
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # ------------------------------------------------------------
    # 2. Read FASTA file and preprocess sequences
    # ------------------------------------------------------------
    sequences = []
    ids = []
    seq2id = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        seq = replace_j_with_il(seq)  # Keep consistent with training

        if seq not in seq2id:
            seq2id[seq] = seq_id

    sequences = list(seq2id.keys())
    ids = [seq2id[seq] for seq in sequences]

    # ------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------
    start_time = time.time()
    X = extract_all_features_ids_batch_block(sequences, fids)

    # ------------------------------------------------------------
    # 4. Feature scaling
    # ------------------------------------------------------------
    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------
    # 5. Prediction
    # ------------------------------------------------------------
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class

    end_time = time.time()
    print("Feature extraction & inference time:", end_time - start_time)

    # ------------------------------------------------------------
    # 6. Collect results
    # ------------------------------------------------------------
    result_df = pd.DataFrame({
        "sequence_id": ids,
        "sequence": sequences,
        "predicted_label": y_pred,
        "prediction_probability": y_proba,
    })

    return result_df


# ============================================================
# MMR-Based Sequence Selection
# ============================================================
def select_mmr(results, top_k=50, lambda_=0.8):
    """
    Select sequences using Maximal Marginal Relevance (MMR),
    balancing prediction score and sequence diversity.

    Parameters
    ----------
    results : pd.DataFrame
        Prediction results containing 'sequence' and 'prediction_probability'.
    top_k : int, optional
        Number of sequences to select, by default 50.
    lambda_ : float, optional
        Trade-off parameter between relevance and diversity, by default 0.8.

    Returns
    -------
    list of str
        Selected sequences.
    """
    selected = []
    candidates = results.copy()

    while len(selected) < top_k and not candidates.empty:
        mmr_scores = []

        for _, row in candidates.iterrows():
            seq = row["sequence"]
            score = row["prediction_probability"]

            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(
                    Levenshtein.ratio(seq, s) for s in selected
                )

            mmr_score = lambda_ * score - (1.0 - lambda_) * diversity_penalty
            mmr_scores.append(mmr_score)

        best_idx = int(np.argmax(mmr_scores))
        best_seq = candidates.iloc[best_idx]["sequence"]

        print("Selected score:", candidates.iloc[best_idx]["prediction_probability"])

        selected.append(best_seq)
        candidates = candidates[candidates["sequence"] != best_seq]

    return selected


# ============================================================
# Physicochemical Property Calculations
# ============================================================

# Kyte–Doolittle hydropathy scale
hydropathy_scale = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

hydrophobic_residues = ["A", "V", "I", "L", "M", "F"]


def compute_gravy(seq):
    """
    Compute GRAVY (Grand Average of Hydropathy) score for a sequence.
    """
    values = [hydropathy_scale.get(aa, 0.0) for aa in seq]
    return sum(values) / len(values) if values else 0.0


def hydrophobic_ratio(seq):
    """
    Compute the fraction of hydrophobic residues in a sequence.
    """
    count = sum(aa in hydrophobic_residues for aa in seq)
    return count / len(seq) if seq else 0.0


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    fasta_file_path = Path(
        "./result_outputs/generated_results/scheduler_e10_Sa9.fasta"
    )
    model_path = Path(
        "./result_outputs/trained_models/brf_model_all15_2.pkl"
    )
    scaler_path = Path(
        "./result_outputs/trained_models/brf_scaler_all15_2.pkl"
    )

    # Feature IDs used during training
    fids = [
        "BAEK050101", "TANS770102", "YANJ020101", "VINM940102",
        "JOND920101", "PONP800103", "CIDH920104",
        "ROBB760113", "QIAN880128", "QIAN880121",
    ]

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    start_time = time.perf_counter()
    results = predict_fasta_file(
        fasta_file_path,
        model_path,
        scaler_path,
        fids,
    )
    inference_time = time.perf_counter() - start_time
    print(f"Inference time: {inference_time:.4f} s")

    # ------------------------------------------------------------
    # Post-processing and analysis
    # ------------------------------------------------------------
    results = results.drop_duplicates(subset="sequence", keep="first")

    selected = select_mmr(results, top_k=20, lambda_=0.9)
    print("MMR-selected sequences:", selected)

    hit_rate = len(results[results["prediction_probability"] > 0.5]) / len(results)
    print("Hit rate:", hit_rate)

    results_top = results.sort_values(
        by="prediction_probability", ascending=False
    ).head(50)
    print(results_top)
    print(results_top["sequence"].tolist())

    # Compute physicochemical properties
    results["gravy"] = results["sequence"].apply(compute_gravy)
    results["hydrophobic_ratio"] = results["sequence"].apply(hydrophobic_ratio)

    # ------------------------------------------------------------
    # Filtering conditions
    # ------------------------------------------------------------
    filtered_results = results[
        (results["gravy"] < 0.2) &
        (results["gravy"] > -0.7) &
        (results["hydrophobic_ratio"] <= 0.5) &
        (results["sequence"].str.len() == 10)
    ]

    print("Filter pass ratio:", len(filtered_results) / len(results))
    print("Mean prediction probability:", results["prediction_probability"].mean())

    top_filtered = filtered_results.sort_values(
        by="prediction_probability", ascending=False
    ).head(20)

    print("Top 20 filtered results:")
    print(top_filtered[[
        "sequence",
        "prediction_probability",
        "gravy",
        "hydrophobic_ratio",
    ]])

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    output_path = Path(
        "./result_outputs/generated_analysis/prediction_results.csv"
    )
    results.to_csv(output_path, index=False)
    print(f"Prediction results saved to {output_path}")
