import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    AutoTokenizer,
)

from common.common import (
    peptide_to_feature_ids,
    peptide_to_feature_ids_sum,
    predict_ag_scores,
)
from model.ap_model import (
    EnhancedPBertLSTM_UsingConf,
    predict,
    ESMPositionalRegressor,
)
from dataset.dataset import AGRegressionDataset


# ============================================================
#      Global Config
# ============================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

AAINDEX_PATH = "./data/aaindex1.xlsx"
BERT_PATH = "./pLM/prot_bert"
MAX_LEN = 15


# ============================================================
#  Basic Feature Utils
# ============================================================
def calculate_aac(sequence: str):
    """
    Calculate amino acid composition (AAC).
    """
    aac = [0.0] * 20
    total_len = len(sequence)

    for aa in AMINO_ACIDS:
        aac[AMINO_ACIDS.index(aa)] = sequence.count(aa) / total_len

    return aac


def replace_j_with_il(text):
    """
    Randomly replace 'J' with 'I' or 'L' in a sequence.
    """
    if not isinstance(text, str):
        return text

    return "".join(
        np.random.choice(["I", "L"]) if char == "J" else char
        for char in text
    )


# ============================================================
#   PAAC Feature Utils
# ============================================================
def calculate_paac(sequence, lamda=3, w=0.05):
    """
    Calculate pseudo amino acid composition (PAAC).
    """
    valid_aas = list(AMINO_ACIDS)

    invalid_chars = [c for c in sequence if c not in valid_aas]
    if invalid_chars:
        raise ValueError(
            f"Invalid amino acids detected: {invalid_chars}"
        )

    aac = calculate_aac(sequence)

    autocorr = []
    for lag in range(1, lamda + 1):
        corr = 0.0
        denom = len(sequence) - lag

        if denom <= 0:
            autocorr.append(0.0)
            continue

        for i in range(denom):
            idx1 = valid_aas.index(sequence[i])
            idx2 = valid_aas.index(sequence[i + lag])
            corr += (idx1 - idx2) ** 2

        autocorr.append(corr / denom)

    paac = []
    total_aac = sum(aac)

    if total_aac == 0:
        raise ValueError("AAC sum is zero.")

    paac.extend([(1 - w) * v / total_aac for v in aac])

    total_autocorr = sum(autocorr)
    for v in autocorr:
        paac.append(w * v / total_autocorr if total_autocorr != 0 else 0.0)

    return paac


# ============================================================
#   BPF / OPE Features
# ============================================================
def get_bpf(sequence, max_len=10):
    """
    Binary profile feature (BPF).
    """
    bpf = np.zeros((max_len, 20))

    for i in range(min(len(sequence), max_len)):
        aa = sequence[i]
        if aa in AA_TO_IDX:
            bpf[i, AA_TO_IDX[aa]] = 1

    return bpf


def get_ope(sequence, max_len=10):
    """
    Ordinal position encoding (OPE).
    """
    L = min(len(sequence), max_len)
    ope = np.linspace(0, 1, L).reshape(-1, 1)

    padded = np.zeros((max_len, 1))
    padded[:L] = ope

    return padded


def get_bpf_ope(sequence, max_len=10):
    """
    Combine flattened BPF and OPE features.
    """
    bpf_flat = get_bpf(sequence, max_len).flatten()
    ope_flat = get_ope(sequence, max_len).flatten()

    return np.hstack((bpf_flat, ope_flat))


# ============================================================
#  AG Score Prediction
# ============================================================
def get_ag_score(sequences):
    """
    Predict AP scores using a distilled ProtBERT-LSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BertConfig.from_pretrained(
        BERT_PATH,
        output_hidden_states=True,
    )

    model = EnhancedPBertLSTM_UsingConf(
        lstm_layers=3,
        lstm_hidden_size=256,
        return_features="attn",
        bert_config=config,
        bert_path=BERT_PATH,
    )

    model.load_state_dict(
        torch.load(
            "./result_outputs/trained_models/ap_distilled_v1.pt",
            map_location=device,
        )
    )

    tokenizer = BertTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case=False,
    )

    return predict(model, tokenizer, sequences)


def get_ag_scores(seqs):
    """
    Predict AG scores using ESM-based regression model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "./result_outputs/trained_models/best_ag_regressor.pt"
    model_name = "./pLM/esm2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ESMPositionalRegressor(
        model_name=model_name,
        freeze_esm=True,
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)

    dataset = AGRegressionDataset(seqs, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    df = predict_ag_scores(model, dataloader, device)
    df_sum = (
        df.groupby("sequence")["predicted_ag_score"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_ag_score": "total_ag_score"})
    )

    return df_sum["total_ag_score"].tolist()


# ============================================================
#   CKSAAGP Features
# ============================================================
G1 = {"G", "A", "V", "L", "M", "I"}
G2 = {"F", "Y", "W"}
G3 = {"K", "R", "H"}
G4 = {"D", "E"}
G5 = {"S", "T", "C", "P", "N", "Q"}


def calculate_cksaaagp(peptide_sequence, k):
    """
    Calculate CKSAAGP features.
    """
    L = len(peptide_sequence)
    if L - k - 1 <= 0:
        return [0.0] * 25

    group_map = [G1, G2, G3, G4, G5]
    counts = {(i, j): 0 for i in range(5) for j in range(5)}

    for i in range(L - k - 1):
        aa1 = peptide_sequence[i]
        aa2 = peptide_sequence[i + k + 1]

        g1 = next((idx for idx, g in enumerate(group_map) if aa1 in g), None)
        g2 = next((idx for idx, g in enumerate(group_map) if aa2 in g), None)

        if g1 is not None and g2 is not None:
            counts[(g1, g2)] += 1

    return [v / (L - k - 1) for v in counts.values()]


# ============================================================
#   Feature Extraction
# ============================================================
def extract_all_features_ids_batch_block(sequences, selected_feature_ids):
    """
    Batch extraction of all feature blocks.
    """
    aac_block = np.array([calculate_aac(s) for s in sequences])
    cksaa1 = np.array([calculate_cksaaagp(s, 1) for s in sequences])
    cksaa2 = np.array([calculate_cksaaagp(s, 2) for s in sequences])
    cksaa3 = np.array([calculate_cksaaagp(s, 3) for s in sequences])
    paac_block = np.array([calculate_paac(s) for s in sequences])
    bpf_ope = np.array([get_bpf_ope(s, MAX_LEN) for s in sequences])
    aaindex = np.array([
        peptide_to_feature_ids(s, selected_feature_ids)
        for s in sequences
    ])
    aaindex_sum = np.array([
        peptide_to_feature_ids_sum(s, selected_feature_ids)
        for s in sequences
    ])
    ag_scores = np.array(get_ag_scores(sequences)).reshape(-1, 1)

    return np.hstack([
        aac_block,
        cksaa1,
        cksaa2,
        cksaa3,
        paac_block,
        bpf_ope,
        aaindex,
        ag_scores,
        aaindex_sum,
    ])


# ============================================================
#   Feature Selection
# ============================================================
def select_important_aaindex_features(
    sequences,
    labels,
    top_n=32,
):
    """
    Select important AAIndex features using XGBoost.
    """
    aaindex_df = pd.read_excel(AAINDEX_PATH)

    aaindex_dict = {}
    for _, row in aaindex_df.iterrows():
        index_id = row["ID_name"]
        aaindex_dict[index_id] = {
            aa: row.get(aa, 0) for aa in AMINO_ACIDS
        }

    def sequence_to_features(seq):
        return [
            sum(aaindex_dict[idx].get(aa, 0) for aa in seq)
            for idx in aaindex_dict
        ]

    X = np.array([sequence_to_features(s) for s in sequences])
    y = np.array(labels)

    X_scaled = StandardScaler().fit_transform(X)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_scaled, y)

    importance = model.feature_importances_
    features = list(aaindex_dict.keys())

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    return df.head(top_n)["Feature"].tolist()


# ============================================================
#          Main
# ============================================================
if __name__ == "__main__":
    # ===== 1. Load test dataset =====
    test_df = pd.read_csv("../data/test_dataset.csv")

    # Apply sequence preprocessing
    test_df["sequence"] = test_df["sequence"].apply(replace_j_with_il)
    test_sequences = test_df["sequence"].tolist()

    # ===== 2. Load pre-selected feature IDs =====
    # These feature IDs must be saved during the training stage
    feature_ids = joblib.load("./result_outputs/trained_models/feature_ids.pkl")

    # ===== 3. Extract features for test data =====
    X_test = extract_all_features_ids_batch_block(
        test_sequences,
        feature_ids,
    )

    # ===== 4. Load fitted scaler and transform test features =====
    scaler = joblib.load("./result_outputs/trained_models/brf_scaler_all15_2.pkl")
    X_test = scaler.transform(X_test)

    # ===== 5. Load trained Balanced Random Forest model =====
    brf = joblib.load("./result_outputs/trained_models/brf_model_all15_2.pkl")

    # ===== 6. Perform prediction =====
    y_pred = brf.predict(X_test)
    y_proba = brf.predict_proba(X_test)[:, 1]

    # ===== 7. Evaluate model performance =====
    print("Accuracy:", accuracy_score(test_df["labels"], y_pred))
    print("F1 Score:", f1_score(test_df["labels"], y_pred))
    print("Classification Report:\n", classification_report(test_df["labels"], y_pred))
    print("AUC:", roc_auc_score(test_df["labels"], y_proba))