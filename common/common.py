import math
import re
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import Levenshtein
from transformers import (
    BioGptTokenizer,
    BioGptForCausalLM,
    BertTokenizer,
    BertModel,
)

from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolfiles import MolFromFASTA
from Bio import SeqIO


# ============================================================
# Learning Rate Scheduler
# ============================================================
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
):
    """
    Create a cosine learning rate schedule with warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule.
    num_warmup_steps : int
        Number of warmup steps.
    num_training_steps : int
        Total number of training steps.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# Prompt-Based Generation
# ============================================================
def prompt_generate(
    prompt_string,
    tokenizer,
    model,
    num_samples,
    max_len=10,
    save_path="generated_peptides.fasta",
):
    """
    Generate peptide sequences from a prompt and save them as a FASTA file.
    """
    prompt_tokens = tokenizer.encode_prompt(prompt_string)
    print("Prompt tokens:", prompt_tokens)

    generated_seqs = model.generate(
        prompt_tokens,
        num_samples,
        max_len,
        top_k=5,
    )

    cleaned_seqs = [
        "".join([aa for aa in seq if aa != "[PAD]"])
        for seq in generated_seqs
    ]

    print("Generated peptide sequences:", cleaned_seqs)

    with open(save_path, "w") as f:
        for i, seq in enumerate(cleaned_seqs):
            f.write(f">peptide_{i + 1}\n{seq}\n")

    print(f"Saved to {save_path}")
    return cleaned_seqs


# ============================================================
# Prompt Discretization Utilities
# ============================================================
def convert_prompt_with_discretized_pI(
    prompt,
    bin_size=0.5,
    precision=1,
):
    """
    Discretize pI values in the prompt into interval tokens.
    """

    def discretize_pI(value, bin_size):
        rounded = round(value, precision)
        lower = (rounded // bin_size) * bin_size
        upper = lower + bin_size
        return f"pI:{lower:.1f}-{upper:.1f}"

    def replacer(match):
        key, val = match.group(1), float(match.group(2))
        if key == "pI":
            return discretize_pI(val, bin_size)
        elif key == "Length":
            return f"{key}:{int(val)}"
        else:
            return f"{key}:{val}"

    return re.sub(r"(\w+):([\d.]+)", replacer, prompt)


def convert_prompt_with_discretized_gravy(
    prompt,
    step=0.5,
    min_val=-4.5,
    max_val=4.5,
):
    """
    Discretize GRAVY values in the prompt into interval tokens.
    """
    match = re.search(r"GRAVY:([-+]?\d*\.\d+|\d+)", prompt)
    if not match:
        return prompt

    gravy = float(match.group(1))

    if gravy < min_val:
        token = f"GRAVY:<{min_val}"
    elif gravy > max_val:
        token = f"GRAVY:>{max_val}"
    else:
        bins = np.arange(min_val, max_val, step)
        for b in bins:
            if b <= gravy < b + step:
                token = f"GRAVY:{b:.1f}-{b + step:.1f}"
                break
            if gravy == max_val:
                token = f"GRAVY:{max_val - step:.1f}-{max_val:.1f}"

    return re.sub(r"GRAVY:([-+]?\d*\.\d+|\d+)", token, prompt)


# ============================================================
# Prompt / Token Encoding
# ============================================================
def encode_activity_prompts(
    activity_list,
    model_path="./biogpt",
):
    """
    Encode activity names into embeddings using BioGPT.
    """
    tokenizer = BioGptTokenizer.from_pretrained(model_path)
    model = BioGptForCausalLM.from_pretrained(
        model_path,
        output_hidden_states=True,
    )
    model.eval()

    prompts = [str(activity) for activity in activity_list]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.hidden_states[-1]

    prompt_embeddings = embeddings.mean(dim=1)
    return dict(zip(activity_list, prompt_embeddings))


AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def encode_all_token(
    all_tokens,
    biogpt_path="./biogpt",
    protbert_path="./prot_bert",
):
    """
    Encode amino acids with ProtBERT and other tokens with BioGPT.
    """
    tokenizer_bio = BioGptTokenizer.from_pretrained(biogpt_path)
    model_bio = BioGptForCausalLM.from_pretrained(
        biogpt_path,
        output_hidden_states=True,
    )
    model_bio.eval()

    tokenizer_prot = BertTokenizer.from_pretrained(
        protbert_path,
        do_lower_case=False,
    )
    model_prot = BertModel.from_pretrained(
        protbert_path,
        output_hidden_states=True,
    )
    model_prot.eval()

    token_embeds = {}

    for tok in all_tokens:
        if len(tok) == 1 and tok in AA_SET:
            token_id = tokenizer_prot.convert_tokens_to_ids(tok)
            emb = model_prot.embeddings.word_embeddings.weight[token_id]
        else:
            with torch.no_grad():
                encoded = tokenizer_bio(
                    tok,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                outputs = model_bio(**encoded)
                emb = outputs.hidden_states[-1].mean(dim=1).squeeze(0)

        token_embeds[tok] = emb

    return token_embeds


# ============================================================
# AAIndex Feature Engineering
# ============================================================
effective_feature_ids = [
    "SNEP660101", "NAGK730103", "GRAR740101", "ROSM880104",
    "CIDH920102", "PLIV810101", "PARJ860101", "QIAN880121",
    "GEIM800105", "FAUJ880109", "QIAN880104", "FASG890101",
    "ROBB760104", "NAKH900111", "PONP800105", "PALJ810104",
    "SWER830101", "KHAG800101", "EISD860101", "ZHOH040101",
]

aaindex_path = "./data/aaindex1.xlsx"
aaindex_df = pd.read_excel(aaindex_path)


def peptide_to_feature_ids(peptide, selected_feature_ids):
    """
    Convert a peptide into an AAIndex-based feature vector (position-wise).
    """
    df = aaindex_df.copy()
    df["feature_id"] = df["ID_name"].str.split().str[1]
    amino_acids = df.columns[2:22]

    selected_df = (
        df[df["feature_id"].isin(selected_feature_ids)]
        .set_index("feature_id")
        .loc[selected_feature_ids]
        .reset_index()
    )

    selected_df[amino_acids] = selected_df[amino_acids].fillna(-1)

    feature_dicts = [
        {aa: row[aa] for aa in amino_acids}
        for _, row in selected_df.iterrows()
    ]

    max_len = 15
    padded_peptide = peptide.ljust(max_len, "X")[:max_len]

    feature_vector = []
    for aa in padded_peptide:
        if aa in amino_acids:
            for feature in feature_dicts:
                feature_vector.append(feature.get(aa, 0.0))
        else:
            feature_vector.extend([0.0] * len(feature_dicts))

    return feature_vector


def peptide_to_feature_ids_sum(peptide, selected_feature_ids):
    """
    Convert a peptide into an AAIndex feature vector by summation.
    """
    df = aaindex_df.copy()
    df["feature_id"] = df["ID_name"].str.split().str[1]
    amino_acids = df.columns[2:22]

    selected_df = (
        df[df["feature_id"].isin(selected_feature_ids)]
        .set_index("feature_id")
        .loc[selected_feature_ids]
        .reset_index()
    )

    selected_df[amino_acids] = selected_df[amino_acids].fillna(-1)

    feature_dicts = [
        {aa: row[aa] for aa in amino_acids}
        for _, row in selected_df.iterrows()
    ]

    feature_sum = [0.0] * len(feature_dicts)
    for aa in peptide.upper():
        if aa in amino_acids:
            for i, feature in enumerate(feature_dicts):
                feature_sum[i] += feature.get(aa, 0.0)

    return feature_sum


# ============================================================
# Loss Functions
# ============================================================
def clip_loss(
    prompt_embeds,
    peptide_embeds,
    temperature=0.07,
):
    """
    CLIP-style contrastive loss for prompt–peptide alignment.
    """
    prompt_embeds = F.normalize(prompt_embeds, dim=1)
    peptide_embeds = F.normalize(peptide_embeds, dim=1)

    logits = torch.matmul(prompt_embeds, peptide_embeds.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)

    return (loss_i2t + loss_t2i) / 2


class DistillationLoss(nn.Module):
    """
    Dual-teacher distillation loss (BioGPT + ProtBERT).
    """

    def __init__(
        self,
        student_dim,
        biogpt_dim,
        protbert_dim,
        alpha=0.5,
        beta=1.0,
        loss_type="cosine",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type

        self.prompt_proj = nn.Linear(student_dim, biogpt_dim)
        self.seq_proj = nn.Linear(student_dim, protbert_dim)

    def forward(
        self,
        student_prompt,
        student_seq,
        teacher_prompt,
        teacher_seq,
        contrastive_loss=None,
    ):
        sp = student_prompt
        ss = student_seq

        if self.loss_type == "mse":
            loss_prompt = F.mse_loss(sp, teacher_prompt)
            loss_seq = F.mse_loss(ss, teacher_seq)
        elif self.loss_type == "cosine":
            loss_prompt = 1 - F.cosine_similarity(sp, teacher_prompt, dim=-1).mean()
            loss_seq = 1 - F.cosine_similarity(ss, teacher_seq, dim=-1).mean()
        else:
            raise ValueError("loss_type must be 'mse' or 'cosine'")

        distill_loss = loss_prompt + loss_seq

        if contrastive_loss is not None:
            total_loss = self.alpha * distill_loss + self.beta * contrastive_loss
        else:
            total_loss = self.alpha * distill_loss

        return total_loss, distill_loss, contrastive_loss


# ============================================================
# Evaluation & Utilities
# ============================================================
def avg_pairwise_distance(seqs, sample_size=500):
    """
    Compute average Levenshtein distance (randomly sampled).
    """
    if len(seqs) > sample_size:
        seqs = np.random.choice(seqs, size=sample_size, replace=False)

    dists = [
        Levenshtein.distance(s1, s2)
        for s1, s2 in itertools.combinations(seqs, 2)
    ]
    return np.mean(dists)


def calculate_total_steps(num_epoch, dataset_size, batch_size):
    """
    Calculate total training steps.
    """
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    return num_epoch * steps_per_epoch


# ============================================================
# Inference
# ============================================================
def predict_ag_scores(model, dataloader, device):
    """
    Predict amino-acid-level activity scores.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask = batch["mask"].to(device)
            sequences = batch["sequence"]

            preds = model(input_ids, attention_mask).cpu().numpy()
            masks = mask.cpu().numpy()

            for i, seq in enumerate(sequences):
                valid_len = int(masks[i].sum())
                valid_preds = preds[i][:valid_len]
                valid_seq = seq[:valid_len]

                for pos, aa in enumerate(valid_seq):
                    results.append({
                        "sequence": "".join(seq),
                        "position": pos + 1,
                        "amino_acid": aa,
                        "predicted_ag_score": float(valid_preds[pos]),
                    })

    return pd.DataFrame(results)
