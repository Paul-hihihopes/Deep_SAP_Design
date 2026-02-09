import os
import math
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import (
    BioGptTokenizer,
    BioGptModel,
    AutoModel,
    AutoTokenizer,
)

from model.gpt_model import PeptideTokenizer, PeptidePromptGPT
from dataset.dataset import (
    ContrastDataset,
    PeptideDataset,
    PeptideDatasetDropout,
    contrast_collate_fn,
)

from common.common import (
    encode_activity_prompts,
    encode_all_token,
    convert_prompt_with_discretized_pI,
    convert_prompt_with_discretized_gravy,
    calculate_total_steps,
    compute_sample_weights_from_prompts,
    get_cosine_schedule_with_warmup,
    clip_loss,
    DistillationLoss,
)

# ============================================================
# Learning Rate Scheduler
# ============================================================
class WarmupCosineWithRestarts:
    """
    Warmup + cosine annealing scheduler with optional restarts.
    """

    def __init__(self, optimizer, base_lr, warmup_steps, T_max, eta_min=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min

        self.step_num = 0
        self.restart_step = 0

        # Initialize learning rate to zero
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def step(self):
        self.step_num += 1
        t = self.step_num - self.restart_step

        if t <= self.warmup_steps:
            lr = self.base_lr * t / self.warmup_steps
        else:
            progress = (t - self.warmup_steps) / max(1, self.T_max - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        return lr

    def warm_restart(self, new_T_max=None, new_warmup_steps=None):
        """
        Restart the scheduler with optional new cycle length.
        """
        self.restart_step = self.step_num
        if new_T_max is not None:
            self.T_max = new_T_max
        if new_warmup_steps is not None:
            self.warmup_steps = new_warmup_steps


# ============================================================
# Training Utilities
# ============================================================
def train_distill(
    student,
    biogpt_model,
    protbert_model,
    dataloader,
    optimizer,
    device,
    distill_loss_fn,
    epochs=5,
    lr_scheduler=None,
):
    """
    Knowledge distillation training loop.
    """
    student.to(device)
    biogpt_model.to(device).eval()
    protbert_model.to(device).eval()

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Distillation Epoch {epoch + 1}")
        for batch in pbar:
            prompt_ids = batch["prompt_ids"].to(device)
            sequence_ids = batch["sequence_ids"].to(device)

            bgpt_ids = {k: v.squeeze(1).to(device) for k, v in batch["biogpt_ids"].items()}
            bert_ids = {k: v.squeeze(1).to(device) for k, v in batch["bert_ids"].items()}

            student_prompt = student.encode(prompt_ids, mode="norm")
            student_seq = student.encode(sequence_ids, mode="distilled")

            with torch.no_grad():
                teacher_prompt = biogpt_model(**bgpt_ids).last_hidden_state.mean(dim=1)
                teacher_seq = protbert_model(**bert_ids).last_hidden_state.mean(dim=1)

            loss, _, _ = distill_loss_fn(
                student_prompt,
                student_seq,
                teacher_prompt,
                teacher_seq,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1}: avg_loss={total_loss / len(dataloader):.4f}")


def contrastive_train_one_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    device,
    temperature=0.07,
):
    """
    Contrastive alignment training for one epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Contrastive Training")
    for batch in pbar:
        prompt_ids = batch["prompt_ids"].to(device)
        sequence_ids = batch["sequence_ids"].to(device)

        prompt_emb = model.encode(prompt_ids, mode="contrast")
        seq_emb = model.encode(sequence_ids, mode="contrast")

        loss = clip_loss(prompt_emb, seq_emb, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Autoregressive language modeling training for one epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_len"]

        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        labels = labels[:, 1:]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Validation loop.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_len"]

            labels = input_ids.clone()
            for i, plen in enumerate(prompt_lens):
                labels[i, :plen] = -100

            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            labels = labels[:, 1:]

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Prompt Utilities
# ============================================================
def clean_prompt(prompt):
    """
    Remove 'Self assembly' from activity tokens while keeping attributes.
    """
    tokens = prompt.split("+")
    attrs = tokens[-3:]
    acts = [a for a in tokens[:-3] if "Self assembly" not in a]

    if len(acts) == 0:
        return None

    return "+".join(acts + attrs)


# ============================================================
# Main Training Pipeline
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 40

    lr = 1e-4
    contrast_lr = 1e-4
    distilled_lr = 1e-4

    num_epochs = 30
    contrast_epochs = 20
    distilled_epochs = 5

    batch_size = 512
    aas = list("ACDEFGHIKLMNPQRSTVWY")

    # ---------------- Dataset ----------------
    df = pd.read_csv("./data/dataset_full_prompt_with_gravy_4.csv")
    df["clean_prompt"] = df["full_prompt"].apply(clean_prompt)
    df = df[df["clean_prompt"].notnull()].reset_index(drop=True)

    df["clean_prompt"] = df["clean_prompt"].apply(convert_prompt_with_discretized_gravy)
    df["clean_prompt"] = df["clean_prompt"].apply(convert_prompt_with_discretized_pI)

    sequences = df["peptide"].tolist()
    prompts = df["clean_prompt"].tolist()

    # ---------------- Tokenizer ----------------
    activity_dir = "./data/peptipedia2_15/"
    activities = [
        f.replace(".fasta", "").strip()
        for f in os.listdir(activity_dir)
        if f.endswith(".fasta")
    ]

    pi_tokens = [f"pI:{round(x,1)}-{round(x+0.5,1)}" for x in np.arange(3.0, 10.0, 0.5)]
    length_tokens = [f"Length:{i}" for i in range(1, 16)]
    gravy_tokens = [f"GRAVY:{round(x,1)}-{round(x+0.5,1)}" for x in np.arange(-4.5, 4.5, 0.5)]

    token_dict = {
        "AA": aas,
        "Activity": activities,
        "pI": pi_tokens,
        "Length": length_tokens,
        "GRAVY": gravy_tokens,
    }

    tokenizer = PeptideTokenizer(token_dict)

    with open("./result_outputs/trained_models/tokenizer_ga4.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # ---------------- Models ----------------
    bert_tokenizer = AutoTokenizer.from_pretrained("./pLM/esm2_150m")
    bert_model = AutoModel.from_pretrained("./pLM/esm2_150m")

    biogpt_tokenizer = BioGptTokenizer.from_pretrained("./pLM/biogpt")
    biogpt_model = BioGptModel.from_pretrained("./pLM/biogpt")

    model = PeptidePromptGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=1024,
        tokenizer=tokenizer,
        activity_embed_dict=None,
        num_layers=12,
        nhead=16,
        dim_feedforward=512,
        max_len=MAX_LEN,
        distilled_dim=480,
        contract_proj_dim=256,
    ).to(device)

    print("\n🎉 Training complete!")
